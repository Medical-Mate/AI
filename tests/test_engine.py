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
