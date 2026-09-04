"""무상태 API — 상태를 왕복시키며 문진이 이어지는지. LLM 호출 없음(ScriptedExtractor)."""

from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from medimate.api.app import create_app
from medimate.dialog.questions import CLOSING, OPENING, QUESTIONS
from medimate.llm import AxisUpdate, TurnExtraction
from medimate.schema import Axis, FieldStatus
from tests.fakes import ScriptedExtractor


def filled(axis, value, evidence):
    return AxisUpdate(axis=axis, status=FieldStatus.FILLED, value=value, evidence=evidence)


@dataclass
class FakeUsage:
    input_tokens: int = 0
    output_tokens: int = 0


class SharedScriptExtractor(ScriptedExtractor):
    """요청마다 새 인스턴스지만 대본은 공유한다. extract가 불릴 때만 하나 소비.

    usage는 실제 어댑터처럼 호출 시점에 늘어난다(요청 단위로 0에서 시작).
    """

    def __init__(self, queue: list[TurnExtraction], per_call=(1000, 100)):
        super().__init__([])
        self._queue = queue
        self._per_call = per_call
        self.usage = FakeUsage()

    def extract(self, utterance, asked_axis, history=()):
        self._script = [self._queue.pop(0)] if self._queue else []
        self.usage.input_tokens += self._per_call[0]
        self.usage.output_tokens += self._per_call[1]
        return super().extract(utterance, asked_axis, history)


def make_client(script: list[TurnExtraction]) -> tuple[TestClient, list[ScriptedExtractor]]:
    """요청마다 새 Extractor를 만든다(서버와 같은 방식). 대본은 호출 순서대로 하나씩 소비."""
    made: list[ScriptedExtractor] = []
    queue = list(script)

    def factory():
        ex = SharedScriptExtractor(queue)
        made.append(ex)
        return ex

    return TestClient(create_app(factory)), made


def test_start_returns_opening_and_empty_state():
    client, made = make_client([])
    r = client.post("/v1/previsit/sessions")
    assert r.status_code == 200
    body = r.json()
    assert body["reply"] == OPENING
    assert body["state"]["turn"] == 0 and body["state"]["asked_axis"] is None
    assert body["state"]["card"]["provenance"]["model_id"] == "fake"
    assert all(ex.calls == [] for ex in made)  # 시작은 LLM을 부르지 않는다


def test_state_round_trip_continues_the_interview():
    client, made = make_client(
        [
            TurnExtraction(
                chief_complaint="오른쪽 무릎이 아프다",
                updates=[filled(Axis.SITE, "오른쪽 무릎", "오른쪽 무릎이")],
            ),
            TurnExtraction(updates=[filled(Axis.ONSET, "어제부터", "어제부터")]),
        ]
    )
    state = client.post("/v1/previsit/sessions").json()["state"]

    r1 = client.post(
        "/v1/previsit/turns", json={"state": state, "utterance": "오른쪽 무릎이 아파요"}
    )
    assert r1.status_code == 200
    b1 = r1.json()
    assert b1["reply"] == QUESTIONS[Axis.ONSET]
    assert b1["state"]["asked_axis"] == "onset" and b1["state"]["turn"] == 1
    assert b1["card"]["axes"]["site"]["value"] == "오른쪽 무릎"
    assert b1["audit"]["turn"] == 1 and b1["audit"]["asked_axis"] is None

    # 두 번째 요청은 다른 프로세스라고 가정 — 받은 state만 넘긴다
    r2 = client.post("/v1/previsit/turns", json={"state": b1["state"], "utterance": "어제부터요"})
    b2 = r2.json()
    assert b2["reply"] == QUESTIONS[Axis.CHARACTER]
    assert b2["card"]["axes"]["onset"]["value"] == "어제부터"
    assert b2["card"]["axes"]["site"]["value"] == "오른쪽 무릎"  # 이전 턴 값이 유지된다
    # 두 번째 Extractor는 직전 (질문, 답) 이력을 받았다
    assert made[-1].histories[0] == [(OPENING, "오른쪽 무릎이 아파요")]
    assert made[-1].calls[0][1] == Axis.ONSET  # 1턴에 SITE가 채워져 ONSET을 물었다


def test_stop_ends_and_card_is_returned_as_is():
    client, _ = make_client(
        [
            TurnExtraction(
                chief_complaint="어깨", updates=[filled(Axis.SITE, "왼쪽 어깨", "어깨")]
            ),
            TurnExtraction(wants_to_stop=True),
        ]
    )
    state = client.post("/v1/previsit/sessions").json()["state"]
    state = client.post("/v1/previsit/turns", json={"state": state, "utterance": "어깨"}).json()[
        "state"
    ]
    b = client.post("/v1/previsit/turns", json={"state": state, "utterance": "이만 할래요"}).json()
    assert b["ended"] and b["end_reason"] == "stop" and b["reply"] == CLOSING
    assert b["card"]["axes"]["site"]["value"] == "왼쪽 어깨"
    assert b["card"]["completeness"] < 1


def test_empty_utterance_has_no_audit_and_no_llm_call():
    client, made = make_client([TurnExtraction()])
    state = client.post("/v1/previsit/sessions").json()["state"]
    b = client.post("/v1/previsit/turns", json={"state": state, "utterance": "   "}).json()
    assert b["audit"] is None and b["state"]["turn"] == 0
    assert all(ex.calls == [] for ex in made)


def test_session_tokens_accumulate_across_requests():
    client, _ = make_client([TurnExtraction(), TurnExtraction()])
    state = client.post("/v1/previsit/sessions").json()["state"]
    assert state["session_tokens"] == 0
    b1 = client.post("/v1/previsit/turns", json={"state": state, "utterance": "무릎"}).json()
    b2 = client.post("/v1/previsit/turns", json={"state": b1["state"], "utterance": "어제"}).json()
    assert b1["state"]["session_tokens"] == 1100
    assert b2["state"]["session_tokens"] == 2200
    assert b1["audit"]["usage"]["input_tokens"] == 1000


def test_tampered_state_with_diagnosis_field_is_rejected():
    client, _ = make_client([])
    state = client.post("/v1/previsit/sessions").json()["state"]
    r = client.post(
        "/v1/previsit/turns", json={"state": state, "utterance": "x", "diagnosis": "ACL"}
    )
    assert r.status_code == 422


def test_extractor_parse_failure_is_502_and_state_untouched():
    class Broken:
        model_id = "fake"
        prompt_version = "test"

        def extract(self, *a, **k):
            raise ValueError("Expecting value: line 1")

    client = TestClient(create_app(lambda: Broken()))
    state = client.post("/v1/previsit/sessions").json()["state"]
    r = client.post("/v1/previsit/turns", json={"state": state, "utterance": "무릎"})
    assert r.status_code == 502


@pytest.mark.parametrize("path", ["/health"])
def test_health(path):
    client, _ = make_client([])
    assert client.get(path).json() == {"status": "ok"}


def test_preselected_site_fills_axis_and_anchors_opening():
    client, made = make_client([TurnExtraction(chief_complaint="허리가 뻐근하다")])
    r = client.post("/v1/previsit/sessions", json={"site_label": "허리"})
    b = r.json()
    assert "허리" in b["reply"] and b["reply"] != OPENING
    site = b["state"]["card"]["axes"]["site"]
    assert site["status"] == "filled" and site["value"] == "허리"
    assert site["evidence"][0].startswith("[부위 선택]")  # 발화가 아님이 드러난다
    # 첫 턴 뒤 SITE를 다시 묻지 않고 ONSET으로 간다
    t = client.post(
        "/v1/previsit/turns", json={"state": b["state"], "utterance": "뻐근해요"}
    ).json()
    assert t["reply"] == QUESTIONS[Axis.ONSET]
    assert t["state"]["history"][0][0] == b["reply"]  # 이력의 첫 질문도 앵커된 오프닝


def test_site_node_selection_fills_site_and_department_guidance():
    client, _ = make_client([TurnExtraction(chief_complaint="허리가 뻐근하다")])
    r = client.post("/v1/previsit/sessions", json={"site_node_id": "SUR:042", "side": "left"})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["reply"].startswith("왼쪽 허리 옆")
    card = b["state"]["card"]
    assert card["axes"]["site"]["value"] == "왼쪽 허리 옆"
    assert card["site_selection"]["anchor_id"] == "ANC:012"
    assert card["provenance"]["ontology_snapshot"]  # 스냅샷이 박힌다
    t = client.post(
        "/v1/previsit/turns", json={"state": b["state"], "utterance": "뻐근해요"}
    ).json()
    dg = t["card"]["department_guidance"]
    assert dg["departments"] == ["내과", "비뇨의학과", "정형외과"]  # 구역 값 그대로, 순서 유지
    assert "자문 확인 전" in dg["source"]


def test_zone_without_departments_falls_back_to_anchor():
    client, _ = make_client([TurnExtraction()])
    b = client.post(
        "/v1/previsit/sessions", json={"site_node_id": "SUR:091", "side": "right"}
    ).json()
    assert b["state"]["card"]["axes"]["site"]["value"] == "오른쪽 무릎"
    t = client.post("/v1/previsit/turns", json={"state": b["state"], "utterance": "아파요"}).json()
    assert t["card"]["department_guidance"]["departments"] == ["정형외과"]  # 앵커 「다리」의 값


def test_side_on_non_lateral_zone_is_rejected():
    client, _ = make_client([])
    r = client.post("/v1/previsit/sessions", json={"site_node_id": "SUR:041", "side": "left"})
    assert r.status_code == 422  # 허리 가운데에는 좌우가 없다
    r = client.post("/v1/previsit/sessions", json={"site_node_id": "UBERON:0001465"})
    assert r.status_code == 422  # 구조 노드는 짚는 대상이 아니다
