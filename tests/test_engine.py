from medimate.dialog import Session
from medimate.dialog.questions import CLOSING, QUESTIONS
from medimate.llm import AxisUpdate, TurnExtraction
from medimate.schema import Axis, FieldStatus
from tests.fakes import ScriptedExtractor


def filled(axis, value, evidence):
    return AxisUpdate(axis=axis, status=FieldStatus.FILLED, value=value, evidence=evidence)


def test_first_turn_fills_site_and_asks_next_unfilled():
    ex = ScriptedExtractor(
        [
            TurnExtraction(
                chief_complaint="오른쪽 무릎이 계단 내려갈 때 아프다",
                updates=[
                    filled(Axis.SITE, "오른쪽 무릎", "오른쪽 무릎이"),
                    filled(Axis.EXACERBATING_RELIEVING, "계단 내려갈 때 악화", "계단 내려갈 때"),
                ],
            )
        ]
    )
    s = Session(ex)
    reply = s.step("오른쪽 무릎이 계단 내려갈 때 아파요")
    assert s.card.is_minimally_complete()
    assert reply == QUESTIONS[Axis.ONSET]  # SITE는 채워졌으니 다음 순서인 ONSET
    assert s.asked_axis == Axis.ONSET


def test_no_answer_marks_asked_axis_skipped_and_moves_on():
    ex = ScriptedExtractor([TurnExtraction(), TurnExtraction()])
    s = Session(ex)
    s.step("음...")  # asked_axis None → SITE 질문
    assert s.asked_axis == Axis.SITE
    s.step("...")  # SITE에 답 없음 → SKIPPED, 다음 ONSET
    assert s.card.axes[Axis.SITE].status == FieldStatus.SKIPPED
    assert s.asked_axis == Axis.ONSET


def test_patient_can_stop_anytime_and_card_is_kept_as_is():
    ex = ScriptedExtractor(
        [
            TurnExtraction(
                chief_complaint="어깨가 아프다",
                updates=[filled(Axis.SITE, "왼쪽 어깨", "왼쪽 어깨")],
            ),
            TurnExtraction(wants_to_stop=True),
        ]
    )
    s = Session(ex)
    s.step("왼쪽 어깨가 아파요")
    reply = s.step("이만 할래요")
    assert reply == CLOSING
    assert s.ended
    assert s.card.axes[Axis.SITE].value == "왼쪽 어깨"
    assert s.card.completeness() < 1  # 미완이어도 카드는 살아 있다


def test_update_without_evidence_is_dropped():
    bad = AxisUpdate(axis=Axis.SEVERITY, status=FieldStatus.FILLED, value="7", evidence="  ")
    s = Session(ScriptedExtractor([TurnExtraction(updates=[bad])]))
    s.step("아파요")
    assert s.card.axes[Axis.SEVERITY].status == FieldStatus.NOT_ASKED


def test_all_axes_answered_ends_session():
    updates = [filled(a, "x", "x") for a in Axis]
    ex = ScriptedExtractor([TurnExtraction(chief_complaint="c", updates=updates)])
    s = Session(ex)
    assert s.step("전부 말함") == CLOSING
    assert s.card.completeness() == 1.0


def test_provenance_and_log_are_stamped():
    s = Session(ScriptedExtractor([]))
    assert s.card.provenance.model_id == "fake"
    assert s.card.provenance.prompt_version == "test"
    s.step("네")
    assert s.logs[0].turn == 1


def test_ambiguous_triggers_one_clarification_then_closes():
    from medimate.dialog.questions import CLARIFY

    amb = AxisUpdate(axis=Axis.SITE, status=FieldStatus.AMBIGUOUS, value="거기", evidence="거기가")
    ex = ScriptedExtractor(
        [
            TurnExtraction(chief_complaint="거기가 또 아프다", updates=[amb]),
            TurnExtraction(updates=[filled(Axis.SITE, "왼쪽 무릎", "왼쪽 무릎")]),
        ]
    )
    s = Session(ex)
    reply = s.step("거기가 또 아파요")
    assert reply == CLARIFY[Axis.SITE]
    s.step("아 왼쪽 무릎이요")
    assert s.card.axes[Axis.SITE].status == FieldStatus.FILLED
    assert s.card.axes[Axis.SITE].value == "왼쪽 무릎"


def test_ambiguous_twice_is_closed_as_skipped():
    amb = AxisUpdate(axis=Axis.SITE, status=FieldStatus.AMBIGUOUS, value="거기", evidence="거기")
    ex = ScriptedExtractor([TurnExtraction(updates=[amb]), TurnExtraction(updates=[amb])])
    s = Session(ex)
    s.step("거기요")
    s.step("거기라니까요")
    assert s.card.axes[Axis.SITE].status == FieldStatus.SKIPPED
    assert s.asked_axis == Axis.ONSET


def test_extra_fields_are_rejected():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        TurnExtraction.model_validate({"updates": [], "diagnosis": "ACL 파열"})


def test_empty_input_does_not_call_llm_and_repeats_question():
    from medimate.dialog.questions import EMPTY_INPUT, OPENING

    ex = ScriptedExtractor([])
    s = Session(ex)
    reply = s.step("   ")
    assert reply.startswith(EMPTY_INPUT) and OPENING in reply
    assert ex.calls == [] and s.logs == []


def test_long_utterance_is_truncated_with_notice():
    from medimate.dialog.questions import TRUNCATED_NOTICE

    ex = ScriptedExtractor([TurnExtraction()])
    s = Session(ex)
    reply = s.step("아" * 500)
    assert len(ex.calls[0][0]) == 300
    assert reply.startswith(TRUNCATED_NOTICE)


def test_turn_cap_ends_session_with_reason():
    from medimate.dialog.engine import Limits

    ex = ScriptedExtractor([TurnExtraction() for _ in range(100)])
    s = Session(ex, limits=Limits(max_turns=3))
    for _ in range(3):
        s.step("음")
    assert not s.ended
    reply = s.step("음")
    assert s.ended and s.end_reason == "max_turns" and reply == CLOSING
    assert len(ex.calls) == 3  # 4번째는 호출하지 않았다


def test_normal_flow_never_hits_default_turn_cap():
    # 8축 모두 무응답 → 첫 발화 + 8질문 = 9턴에서 자연 종료. 20에 닿지 않는다
    ex = ScriptedExtractor([TurnExtraction() for _ in range(30)])
    s = Session(ex)
    n = 0
    while not s.ended and n < 50:
        s.step("...")
        n += 1
    assert s.end_reason == "complete" and n <= 17


def test_session_token_budget_ends_session():
    from dataclasses import dataclass

    from medimate.dialog.engine import Limits

    @dataclass
    class U:
        input_tokens: int = 50_000
        output_tokens: int = 0

    ex = ScriptedExtractor([TurnExtraction()])
    ex.usage = U()
    s = Session(ex, limits=Limits(max_session_tokens=40_000))
    s.step("무릎")
    assert s.ended and s.end_reason == "budget" and ex.calls == []


def test_history_passes_last_two_turns_with_questions():
    from medimate.dialog.engine import Limits
    from medimate.dialog.questions import OPENING, QUESTIONS

    ex = ScriptedExtractor([TurnExtraction() for _ in range(5)])
    s = Session(ex, limits=Limits(history_turns=2))
    s.step("무릎이 아파요")  # asked None → SITE 질문
    s.step("오른쪽")  # SITE → ONSET
    s.step("어제")  # ONSET → CHARACTER
    assert ex.histories[0] == []
    assert ex.histories[1] == [(OPENING, "무릎이 아파요")]
    assert ex.histories[2] == [(OPENING, "무릎이 아파요"), (QUESTIONS[Axis.SITE], "오른쪽")]


def test_preselected_site_is_not_asked_again_but_accepts_refinement():
    amb = AxisUpdate(
        axis=Axis.SITE, status=FieldStatus.AMBIGUOUS, value="아래쪽", evidence="아래쪽"
    )
    ex = ScriptedExtractor(
        [
            TurnExtraction(chief_complaint="아래쪽이 아파요", updates=[amb]),
            TurnExtraction(
                updates=[
                    filled(Axis.ONSET, "일주일 전", "일주일 전"),
                    filled(Axis.SITE, "아래쪽 중앙부", "아래쪽 중앙부요"),
                ]
            ),
        ]
    )
    s = Session(ex)
    s.preselect_site("허리")
    assert s.opening().startswith("허리")
    # 모델이 SITE를 애매하다 해도 되묻지 않고 다음 축(ONSET)으로
    assert s.step("아래쪽이 아파요") == QUESTIONS[Axis.ONSET]
    assert s.card.axes[Axis.SITE].status == FieldStatus.FILLED
    s.step("일주일 전부터요. 아래쪽 중앙부요")
    site = s.card.axes[Axis.SITE]
    assert site.value == "허리 아래쪽 중앙부"  # 좁힌 표현은 받되 선택 부위는 지워지지 않는다
    assert site.evidence == ["[부위 선택] 허리", "아래쪽 중앙부요"]
