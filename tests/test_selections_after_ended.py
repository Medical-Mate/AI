"""문답이 끝난 **뒤에** 통증 슬라이더 값이 온다. 와이어프레임 순서가 그렇다.

2 문답 → 3 통증 슬라이더(1d) → 4 물어볼 것.

`severity`는 `ASK_ORDER`에 없어 **문답으로는 채울 길이 없는 축**이다. 끝난 뒤 선택을 막으면
통증 강도가 카드에 영원히 안 들어가고, 후보 프롬프트도 `심각도: (안 답함)`을 본다.

녹화 fixture가 이걸 못 잡았다 — 녹화기는 어느 턴이 마지막인지 미리 알고 같은 턴에 실었다.
**실제 클라이언트는 모른다.** 그래서 이 파일이 따로 있다.
"""

import json

from fastapi.testclient import TestClient

from medimate.api.app import create_app
from medimate.llm import AxisUpdate, TurnExtraction
from medimate.schema import Axis, FieldStatus
from tests.fakes import ScriptedExtractor

SLIDER = {"axis": "severity", "value": "3 (꽤 아파요)"}
CANDIDATES = json.dumps(
    {"items": [{"text": "이 정도 통증이면 어떤 상태인가요?", "source": "severity"}]},
    ensure_ascii=False,
)


class Fake(ScriptedExtractor):
    def __init__(self):
        super().__init__(
            [
                TurnExtraction(
                    chief_complaint="무릎",
                    updates=[
                        AxisUpdate(
                            axis=a, status=FieldStatus.FILLED, value="x", evidence="x 다 말했어요"
                        )
                        for a in Axis
                        if a is not Axis.SEVERITY
                    ],
                )
            ]
        )
        self.prompts: list[str] = []

    def complete_json(self, system, user, schema):
        self.prompts.append(user)
        return CANDIDATES, 10, 5


def _finished():
    """문답을 끝까지 진행해 `ended`인 상태를 만든다."""
    ex = Fake()
    c = TestClient(create_app(lambda: ex))
    st = c.post("/v1/previsit/sessions").json()["state"]
    b = c.post("/v1/previsit/turns", json={"state": st, "utterance": "x 다 말했어요"}).json()
    assert b["ended"] is True and b["end_reason"] == "complete"
    assert b["card"]["axes"]["severity"]["status"] == "not_asked"
    return c, ex, b["state"]


def test_slider_after_the_session_ends_reaches_the_card():
    c, _, st = _finished()
    b = c.post("/v1/previsit/turns", json={"state": st, "selections": [SLIDER]}).json()
    sev = b["card"]["axes"]["severity"]
    assert sev["status"] == "filled"
    assert sev["value"] == "3 (꽤 아파요)"
    assert sev["source"] == "selection"  # 발화가 아니다
    assert sev["evidence"] == ["[선택] 3 (꽤 아파요)"]


def test_the_session_stays_ended():
    """선택을 받는다고 세션이 다시 열리는 것은 아니다"""
    c, _, st = _finished()
    b = c.post("/v1/previsit/turns", json={"state": st, "selections": [SLIDER]}).json()
    assert b["ended"] is True and b["end_reason"] == "complete"


def test_an_utterance_after_the_end_still_changes_nothing():
    """발화는 여전히 무시한다 — 끝난 세션에서 LLM을 부르지 않는다"""
    c, ex, st = _finished()
    before = len(ex.calls)
    b = c.post(
        "/v1/previsit/turns", json={"state": st, "utterance": "아 그리고 어제부터 붓기도 해요"}
    ).json()
    assert len(ex.calls) == before  # 추출 호출 0
    assert b["card"]["axes"]["associated"]["value"] == "x"  # 카드 그대로


def test_candidates_see_the_slider_value_from_the_same_request():
    """같은 요청의 후보는 **갱신된 카드**로 만들어야 한다. 순서가 틀리면 (안 답함)을 본다"""
    c, ex, st = _finished()
    b = c.post(
        "/v1/previsit/turns",
        json={"state": st, "selections": [SLIDER], "question_candidates": True},
    ).json()
    assert b["card"]["question_candidates"], "후보가 나와야 한다"
    assert len(ex.prompts) == 1
    assert "심각도: 3 (꽤 아파요)" in ex.prompts[0]
    assert "심각도: (안 답함)" not in ex.prompts[0]
