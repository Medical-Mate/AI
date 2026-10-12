"""질문 후보 — 앱 온디바이스 경로(2026-10-12). 폰 1.7B가 만든 후보를 서버가 검증·보충한다.

추론은 폰, 검증 계층(Kiwi 검증기·템플릿 폴백)은 서버. 외부 LLM 호출 0.
"""

import pytest
from fastapi.testclient import TestClient

from medimate.api.app import create_app
from medimate.llm import AxisUpdate, TurnExtraction
from medimate.schema import Axis, FieldStatus
from tests.fakes import ScriptedExtractor

pytest.importorskip("kiwipiepy")

VALUES = {
    Axis.SITE: "왼쪽 무릎",
    Axis.ONSET: "2주 전부터",
    Axis.CHARACTER: "시큰거림",
    Axis.RADIATION: "안 퍼짐",
    Axis.ASSOCIATED: "아침에 뻣뻣함",
    Axis.TIME_COURSE: "비슷함",
    Axis.EXACERBATING_RELIEVING: "계단 내려갈 때 더 아픔",
    Axis.SEVERITY: "5점",
}


class NoLLM(ScriptedExtractor):
    """추출은 대본대로. 후보 생성 LLM이 불리면 터진다 — 폰 경로에서는 불리면 안 된다."""

    def __init__(self):
        super().__init__(
            [
                TurnExtraction(
                    chief_complaint="무릎",
                    updates=[
                        AxisUpdate(axis=a, status=FieldStatus.FILLED, value=v, evidence=v)
                        for a, v in VALUES.items()
                    ],
                )
            ]
        )
        self.called = 0

    def complete_json(self, system, user, schema):
        self.called += 1
        raise AssertionError("폰 경로에서 서버 LLM이 불렸다")


def finish(phone, meta=None, flag=True):
    ex = NoLLM()
    c = TestClient(create_app(lambda: ex))
    # 부위를 짚지 않고 시작한다 — 짚은 부위와 말한 부위("왼쪽 무릎")가 다르면 엔진이 되물어
    # 세션이 안 끝난다. 여기서 보는 건 종료 턴의 후보 처리다
    st = c.post("/v1/previsit/sessions").json()["state"]
    # 근거는 발화 원문에 있어야 가드를 지난다 — 축 값 전부를 한 발화에 싣는다
    utt = " ".join(VALUES.values()) + " 다 말했어요"
    body = {"state": st, "utterance": utt, "question_candidates": flag}
    if phone is not None:
        body["phone_question_candidates"] = phone
    if meta:
        body["candidates_meta"] = meta
    r = c.post("/v1/previsit/turns", json=body)
    return r, ex


def test_phone_candidates_are_verified_and_topped_up_without_any_llm_call():
    phone = [
        {"text": "계단 내려갈 때 더 아픈 건 어떤 상태인가요?", "source": "exacerbating"},
        {"text": "혈압약을 계속 먹어도 되나요?", "source": "medications"},  # 카드에 없는 말 → 버림
    ]
    r, ex = finish(phone, meta={"model_id": "qwen3-1.7b-q4_0", "prompt_version": "q-small-v4"})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["ended"] and ex.called == 0
    cands = b["card"]["question_candidates"]
    assert cands and [c["rank"] for c in cands] == list(range(1, len(cands) + 1))
    texts = [c["text"] for c in cands]
    assert "계단 내려갈 때 더 아픈 건 어떤 상태인가요?" in texts
    assert all("혈압약" not in t for t in texts)
    origins = {c["origin"] for c in cands}
    assert "model" in origins and "template" in origins  # 통과 1 + 템플릿 보충
    qa = b["audit"]["question_candidates"]
    assert qa["source"] == "phone" and qa["received"] == 2 and qa["kept"] == 1
    assert qa["dropped"] and "혈압약" in qa["dropped"][0]["text"]
    assert qa["model_id"] == "qwen3-1.7b-q4_0" and qa["prompt_version"] == "q-small-v4"


def test_an_empty_phone_list_means_templates_only():
    r, ex = finish([])
    b = r.json()
    cands = b["card"]["question_candidates"]
    assert cands and all(c["origin"] == "template" for c in cands) and ex.called == 0
    assert b["audit"]["question_candidates"]["received"] == 0


def test_phone_candidates_require_the_flag_and_meta_requires_candidates():
    r, _ = finish([{"text": "x", "source": "site"}], flag=False)
    assert r.status_code == 422
    ex = NoLLM()
    c = TestClient(create_app(lambda: ex))
    st = c.post("/v1/previsit/sessions").json()["state"]
    r = c.post(
        "/v1/previsit/turns",
        json={
            "state": st,
            "utterance": "x",
            "question_candidates": True,
            "candidates_meta": {"model_id": "m", "prompt_version": "p"},
        },
    )
    assert r.status_code == 422


def test_server_path_is_unchanged_and_audited():
    """폰 후보를 안 보내면 전처럼 서버 LLM 경로다(여기서는 터지므로 후보 null, 카드는 산다)."""
    r, ex = finish(None)
    b = r.json()
    assert b["ended"] and ex.called == 1
    assert b["card"]["question_candidates"] is None
    assert b["audit"]["question_candidates"] == {"source": "server", "kept": 0}
