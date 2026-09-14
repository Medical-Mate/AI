"""문장 번호는 **라벨의 주소**다. 분리 규칙이 바뀌면 그 주소가 다른 문장을 가리킨다.

1p(메모 작성)와 1q-2(라벨 수정) 사이에 배포가 끼면, 앱은 예전 규칙으로 나눈 번호에 라벨을
매겨 보내는데 서버는 새 규칙으로 나눈 문장에 그 번호를 붙인다. **200이 나가고 카드도 멀쩡해
보이는데 내용이 어긋난다.** 조용히 틀린 카드가 저장된다.
"""

import pytest
from fastapi.testclient import TestClient

from medimate.api.app import create_app
from medimate.dialog.memo import SPLIT_RULE_DIGEST, SPLIT_VERSION, split_rule_digest

MEMO = "위염 초기라고 하셨어요. 2주 뒤에 다시 오라고 하셨어요."


def test_the_name_moves_with_the_rule():
    """**이 테스트가 `split_version`의 전부다.**

    값을 손으로 관리하면 규칙은 고치고 이름은 안 고치는 날이 온다. 그날 409는 안 나고
    라벨만 어긋난다 — 막으려고 만든 사고가 정확히 그 모양으로 일어난다.

    깨졌다면: 분리 규칙을 바꾼 것이다. `SPLIT_VERSION`을 올리고 아래 실제 값을 박아라.
    """
    assert split_rule_digest() == SPLIT_RULE_DIGEST, (
        f"분리 규칙이 바뀌었다. SPLIT_VERSION('{SPLIT_VERSION}')을 올리고 "
        f"SPLIT_RULE_DIGEST를 '{split_rule_digest()}'로 바꿔라"
    )


@pytest.fixture
def client():
    return TestClient(create_app())


def test_response_carries_the_version(client):
    r = client.post("/v1/postvisit/memo", json={"memo": MEMO, "classify": False})
    assert r.status_code == 200
    assert r.json()["split_version"] == SPLIT_VERSION


def test_matching_version_round_trips(client):
    """앞 응답의 값을 그대로 되보내는 정상 경로. 이게 막히면 앱이 못 쓴다"""
    first = client.post("/v1/postvisit/memo", json={"memo": MEMO, "classify": False}).json()
    r = client.post(
        "/v1/postvisit/memo",
        json={
            "memo": MEMO,
            "labels": {"0": "findings", "1": "follow_up"},
            "split_version": first["split_version"],
        },
    )
    assert r.status_code == 200
    assert r.json()["card"]["axes"]["findings"]["status"] == "filled"


def test_stale_version_is_409(client):
    r = client.post(
        "/v1/postvisit/memo",
        json={"memo": MEMO, "labels": {"0": "findings"}, "split_version": "split-v0"},
    )
    assert r.status_code == 409
    # 백엔드가 로그만 보고 판단할 수 있어야 한다 — 두 값이 다 보인다
    detail = r.json()["detail"]
    assert "split-v0" in detail and SPLIT_VERSION in detail


def test_omitting_the_field_still_works(client):
    """선택 필드다. 백엔드가 아직 안 보내는 동안에도 돌아야 한다"""
    r = client.post("/v1/postvisit/memo", json={"memo": MEMO, "labels": {"0": "findings"}})
    assert r.status_code == 200
