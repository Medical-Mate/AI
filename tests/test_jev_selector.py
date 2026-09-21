"""Jev 선택기 — SDK 응답 모양을 가짜 클라이언트로 흉내 내 파싱·가드를 본다. 실호출 없음."""

from types import SimpleNamespace

import pytest

pytest.importorskip("kiwipiepy")
pytest.importorskip("typesafe_sdk")

from medimate.span.candidates import CandidateGenerator  # noqa: E402
from medimate.span.select import NONE, JevSelector, resolve  # noqa: E402


class FakeClient:
    def __init__(self, choice, confidence=0.9):
        self.choice, self.confidence = choice, confidence
        self.calls = []

    def system_one(self, state, questions):
        self.calls.append((state, questions))
        ans = SimpleNamespace(
            choice=self.choice,
            confidence=self.confidence,
            probabilities={self.choice: self.confidence, "NONE": 1 - self.confidence},
        )
        return SimpleNamespace(
            answers={"value": ans}, usage=SimpleNamespace(input_tokens=420, output_tokens=0)
        )


@pytest.fixture(scope="module")
def cset():
    return CandidateGenerator().generate("위산약 2주치 받고,", "medication_instructions")


def test_choice_is_resolved_to_candidate_text(cset):
    target = next(c for c in cset.candidates if c.text == "위산약 2주치")
    fake = FakeClient(target.id, 0.83)
    sel = JevSelector(client=fake)
    s = sel.select(cset, "medication_instructions")
    assert resolve(cset, s) == "위산약 2주치"
    assert s.confidence == 0.83 and s.selector == "jev"
    assert sel.usage.input_tokens == 420 and sel.last["probabilities"][target.id] == 0.83
    state, questions = fake.calls[0]
    assert state["조각"] == "위산약 2주치 받고," and "NONE" in questions["value"].criteria


def test_out_of_set_id_becomes_none(cset):
    s = JevSelector(client=FakeClient("C99")).select(cset, "medication_instructions")
    assert s.candidate_id == NONE and "후보 밖" in s.note


def test_budget_guard():
    from medimate.llm.providers import BudgetExceeded

    sel = JevSelector(client=FakeClient("NONE"), budget_usd=0.0)
    with pytest.raises(BudgetExceeded):
        sel.select(CandidateGenerator().generate("감기래요.", "findings"), "findings")
