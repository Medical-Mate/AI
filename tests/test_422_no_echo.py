"""422 본문이 **요청 값을 되돌려주면 안 된다.**

pydantic 기본 핸들러는 `detail[].input`에 문제가 된 값을 그대로 넣는다. 잘못된 요청 하나가
`patient_profile`(복용약·기저질환·알러지)과 `state`(환자 발화 원문)를 응답에 실어 보낸다.

백엔드는 `detail`을 밖으로 안 내보내지만, 웹이 우리를 직접 부르는 구성(CORS)이 되면 그 본문이
**브라우저 콘솔과 에러 리포트에 남는다.** 409 `detail`에 메모를 안 넣기로 한 것과 같은 자리다.
"""

import json

import pytest
from fastapi.testclient import TestClient

from medimate.api.app import create_app

DRUG = "와파린"
UTTERANCE = "왼쪽 무릎이 계단 내려갈 때 시큰해요"


@pytest.fixture
def client():
    return TestClient(create_app())


def _bad_turn(client):
    st = client.post("/v1/previsit/sessions").json()["state"]
    st["card"]["axes"]["onset"] = {
        "status": "이건 없는 값",
        "value": "3일 전",
        "evidence": [UTTERANCE],
    }
    return client.post(
        "/v1/previsit/turns",
        json={
            "state": st,
            "utterance": UTTERANCE,
            "question_candidates": True,
            "patient_profile": {"medications": [DRUG], "conditions": [], "allergies": []},
        },
    )


def test_health_info_and_utterances_do_not_come_back(client):
    r = _bad_turn(client)
    assert r.status_code == 422
    body = json.dumps(r.json(), ensure_ascii=False)
    for leaked in (DRUG, UTTERANCE, "3일 전"):
        assert leaked not in body, leaked


def test_the_field_path_is_still_there(client):
    """백엔드는 `loc`으로 어느 필드인지 본다. 진단에 필요한 것은 값이 아니라 위치다"""
    r = _bad_turn(client)
    locs = [e["loc"] for e in r.json()["detail"]]
    assert ["body", "state", "card", "axes", "onset", "status"] in locs
    assert all(set(e) == {"loc", "msg", "type"} for e in r.json()["detail"])


def test_unknown_field_is_reported_without_its_value(client):
    st = client.post("/v1/previsit/sessions").json()["state"]
    r = client.post(
        "/v1/previsit/turns",
        json={"state": st, "utterance": "x", "patient_profile": {"주민번호": "900101-1234567"}},
    )
    assert r.status_code == 422
    body = json.dumps(r.json(), ensure_ascii=False)
    assert "900101" not in body
    assert "주민번호" in body  # 필드 **이름**은 남는다 — 어디가 틀렸는지 알아야 고친다


def test_memo_path_too(client):
    """진료 후 메모도 같은 핸들러를 지난다"""
    r = client.post("/v1/postvisit/memo", json={"memo": "", "visit_date": "어제"})
    assert r.status_code == 422
    assert all(set(e) == {"loc", "msg", "type"} for e in r.json()["detail"])
