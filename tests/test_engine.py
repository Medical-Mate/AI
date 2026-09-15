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


def test_all_axes_answered_ends_without_a_message_turn():
    """진료 전은 "전하고 싶은 말" 턴이 없다(2026-09-11).

    와이어프레임 4단계("의사에게 물어볼 것")가 그 역할을 대신한다 — AI 후보 + 환자 직접 입력.
    축이 닫히면 바로 종료한다.
    """
    updates = [filled(a, "x", "x") for a in Axis]
    ex = ScriptedExtractor([TurnExtraction(chief_complaint="c", updates=updates)])
    s = Session(ex)
    assert s.step("x 전부 말함") == CLOSING
    assert s.ended and s.end_reason == "complete"
    assert s.card.patient_message is None  # 채울 경로가 없다


def test_severity_is_not_asked_in_the_interview():
    """3단계 슬라이더가 값을 주므로 문답에서 묻지 않는다(백엔드 합의 #7).

    슬라이더는 **NRS가 아니라** 1~5 서열척도 + 라벨이다(`"3 (꽤 아파요)"`). 2026-09-11 정정.

    축은 카드에 그대로 있고 채우는 경로만 바뀌었다 — 앱이 `selections`로 보낸다.
    """
    from medimate.dialog.questions import ASK_ORDER, QUESTIONS

    assert Axis.SEVERITY not in ASK_ORDER
    assert Axis.SEVERITY in QUESTIONS  # 템플릿은 남겨 둔다(되돌리기 쉽게)
    assert len(ASK_ORDER) == 7

    ex = ScriptedExtractor([TurnExtraction(updates=[filled(Axis.SITE, "무릎", "무릎")])])
    s = Session(ex)
    s.step("무릎이 아파요")
    # 다음 질문이 심각도가 아니다
    assert s.asked_axis is not None and s.asked_axis != Axis.SEVERITY


def test_postvisit_still_has_a_message_turn():
    """진료 전만 없애는 것이다. 진료 후 명세는 그대로 유지된다"""
    from medimate.dialog.spec import POSTVISIT_SPEC, PREVISIT_SPEC

    assert PREVISIT_SPEC.message_question is None
    assert POSTVISIT_SPEC.message_question is not None


def test_stop_skips_message_question():
    updates = [filled(a, "x", "x") for a in Axis]
    ex = ScriptedExtractor([TurnExtraction(updates=updates, wants_to_stop=True)])
    s = Session(ex)
    assert s.step("x 다 말했고 이만 할래요") == CLOSING
    assert s.card.patient_message is None


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
    # 8축 모두 무응답 → 첫 발화 + 8질문 + 전할 말 = 10턴에서 자연 종료. 20에 닿지 않는다
    ex = ScriptedExtractor([TurnExtraction() for _ in range(30)])
    s = Session(ex)
    n = 0
    while not s.ended and n < 50:
        s.step("...")
        n += 1
    assert s.end_reason == "complete" and n <= 18


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


# ── 인체도에서 짚은 곳과 환자가 말한 곳이 다를 때 (2026-09-15) ──────────────────
#
# QA: `이마`를 짚고 `눈`이라고 말했더니 카드에 `눈`만 남았다. 짚은 것이 사라지면 안 되고,
# 우리가 하나를 고르면 의사가 읽는 사실이 바뀐다. **환자에게 되묻는다.**


def _relation(pairs: dict[tuple[str, str], str]):
    """온톨로지 대신 쓰는 판정표. 없는 쌍은 `unknown`(모르면 되묻지 않는다)."""
    return lambda a, b: pairs.get((a, b), "unknown")


def _site_turn(label: str, spoken: str, relation=None) -> tuple[Session, str]:
    """부위를 짚고 한 턴 말한다. **evidence는 발화 원문과 같아야 한다** — 다르면 근거
    검증 가드가 업데이트를 버린다(그것도 정상 동작이다). 실제와 같게 문장으로 둔다.
    """
    utterance = f"{spoken}이 불편해요"
    s = Session(
        ScriptedExtractor(
            [
                TurnExtraction(
                    updates=[
                        AxisUpdate(
                            axis=Axis.SITE,
                            status=FieldStatus.FILLED,
                            value=spoken,
                            evidence=utterance,
                        )
                    ]
                )
            ]
        ),
        site_relation=relation,
    )
    s.preselect_site(label)
    s.opening()
    return s, s.step(utterance)


def test_a_different_site_is_asked_back_not_merged():
    s, reply = _site_turn("이마", "눈", _relation({("이마", "눈"): "other"}))
    entry = s.card.axes[Axis.SITE]
    assert entry.status == FieldStatus.AMBIGUOUS
    assert entry.value == "눈"  # 환자 말이 값. 우리가 고르지 않는다
    assert "[부위 선택] 이마" in entry.evidence  # 짚은 것은 남는다
    assert reply == "이마를 짚어 주셨는데 눈이 불편하다고 하셨어요. 어느 쪽을 적을까요?"


def test_the_narrower_site_wins_and_nothing_is_said_twice():
    """상하위로 이어져 있으면 같은 곳이다. 좁은 쪽 하나만 남긴다.

    예전에는 붙이기만 해서 `"어깨 팔"`·`"왼쪽 무릎 무릎 안쪽"`이 나왔다.
    """
    s, _ = _site_turn("배", "아랫배", _relation({("배", "아랫배"): "narrower"}))
    assert s.card.axes[Axis.SITE].value == "아랫배"
    assert s.card.axes[Axis.SITE].status == FieldStatus.FILLED

    s2, _ = _site_turn("어깨", "팔", _relation({("어깨", "팔"): "broader"}))
    assert s2.card.axes[Axis.SITE].value == "어깨"  # 짚은 쪽이 더 좁다


def test_an_unknown_site_is_never_asked_back():
    """온톨로지가 모르는 세부 표현은 되묻지 않는다. 헛되묻기가 놓치는 것보다 나쁘다."""
    s, _ = _site_turn("무릎", "무릎 안쪽")  # 판정 주입 없음 = 항상 unknown
    entry = s.card.axes[Axis.SITE]
    assert entry.status == FieldStatus.FILLED
    assert entry.value == "무릎 안쪽"  # 낱말이 겹치면 두 번 쓰지 않는다


def test_the_clarify_question_survives_the_state_round_trip():
    """무상태라 되묻기 문구는 카드에서 다시 만들어져야 한다 — 상태에 문장을 넣지 않는다."""
    rel = _relation({("이마", "눈"): "other"})
    s, _ = _site_turn("이마", "눈", rel)
    back = Session.from_state(s.extractor, s.to_state(), site_relation=rel)
    assert back._current_question() == (
        "이마를 짚어 주셨는데 눈이 불편하다고 하셨어요. 어느 쪽을 적을까요?"
    )


def test_the_clarify_text_names_the_site_not_the_whole_sentence():
    """evidence는 문장(`"오른쪽 눈이 아파요"`)이고 부위 이름은 `value`에 있다.

    문장으로 온톨로지를 찾으면 아무것도 안 걸려 구체 문구가 **일반 문구로 조용히 샌다.**
    QA에서 그렇게 새고 있었고, 테스트가 evidence에 낱말만 넣고 있어서 못 잡았다.
    """
    s = Session(
        ScriptedExtractor(
            [
                TurnExtraction(
                    updates=[
                        AxisUpdate(
                            axis=Axis.SITE,
                            status=FieldStatus.FILLED,
                            value="오른쪽 눈",
                            evidence="오른쪽 눈이 아파요",
                        )
                    ]
                )
            ]
        ),
        site_relation=_relation({("이마", "오른쪽 눈"): "other"}),
    )
    s.preselect_site("이마")
    s.opening()
    reply = s.step("오른쪽 눈이 아파요")
    assert reply == "이마를 짚어 주셨는데 오른쪽 눈이 불편하다고 하셨어요. 어느 쪽을 적을까요?"
