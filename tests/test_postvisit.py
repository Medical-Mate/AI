"""진료 후 문진(챗봇②) — 같은 엔진이 다른 명세로 돈다. LLM 호출 없음."""

from medimate.dialog import POSTVISIT_SPEC, Session
from medimate.dialog.postvisit_questions import CLOSING, MESSAGE_QUESTION, OPENING, QUESTIONS
from medimate.dialog.state import SessionState
from medimate.dialog.widening import compare_sites, widen_card
from medimate.llm import AxisUpdate, TurnExtraction
from medimate.ontology import load_ontology
from medimate.schema import Axis, FieldStatus
from medimate.schema.export import to_backend_payload
from medimate.schema.postvisit import PostAxis, PostVisitCard
from tests.fakes import ScriptedExtractor


def filled(axis, value, evidence):
    return AxisUpdate(axis=axis, status=FieldStatus.FILLED, value=value, evidence=evidence)


def test_postvisit_flow_asks_six_axes_then_message_then_closes():
    ex = ScriptedExtractor(
        [
            TurnExtraction(
                updates=[
                    filled(
                        PostAxis.HEARD_DIAGNOSIS,
                        "반월판이 조금 찢어졌다고",
                        "반월판이 조금 찢어졌대요",
                    ),
                    filled(
                        PostAxis.MEDICATION, "소염제 하루 두 번 일주일", "소염제 하루 두 번 일주일"
                    ),
                ]
            ),
            TurnExtraction(updates=[filled(PostAxis.TESTS_PROCEDURES, "MRI 예약", "MRI 찍기로")]),
            TurnExtraction(updates=[filled(PostAxis.FOLLOW_UP, "2주 뒤", "2주 뒤에 오라고")]),
            TurnExtraction(
                updates=[filled(PostAxis.INSTRUCTIONS, "계단 피하기", "계단은 피하래요")]
            ),
            TurnExtraction(
                updates=[
                    AxisUpdate(
                        axis=PostAxis.OPEN_QUESTIONS,
                        status=FieldStatus.UNKNOWN,
                        evidence="딱히 없어요",
                    )
                ]
            ),
            TurnExtraction(),
        ]
    )
    s = Session(ex, spec=POSTVISIT_SPEC)
    assert isinstance(s.card, PostVisitCard)
    assert s.opening() == OPENING
    assert (
        s.step("반월판이 조금 찢어졌대요. 소염제 하루 두 번 일주일")
        == QUESTIONS[PostAxis.TESTS_PROCEDURES]
    )
    s.step("MRI 찍기로 했어요")
    s.step("2주 뒤에 오라고")
    s.step("계단은 피하래요")
    assert s.step("딱히 없어요") == MESSAGE_QUESTION
    assert s.step("수술은 되도록 안 하고 싶다고 적어 주세요") == CLOSING
    assert s.ended and s.end_reason == "complete"
    assert s.card.axes[PostAxis.HEARD_DIAGNOSIS].value == "반월판이 조금 찢어졌다고"
    assert s.card.patient_message == "수술은 되도록 안 하고 싶다고 적어 주세요"
    assert s.card.completeness() == 1.0
    # 진료 전 축이 섞여 오면 버린다 — 스키마 경계
    assert Axis.SITE not in s.card.axes


def test_previsit_axis_update_is_dropped_on_postvisit_card():
    ex = ScriptedExtractor([TurnExtraction(updates=[filled(Axis.SEVERITY, "7", "7점")])])
    s = Session(ex, spec=POSTVISIT_SPEC)
    s.step("7점이요")
    assert all(e.status != FieldStatus.FILLED for e in s.card.axes.values())


def test_postvisit_state_round_trip_restores_card_type_and_axes():
    ex = ScriptedExtractor(
        [TurnExtraction(updates=[filled(PostAxis.MEDICATION, "약", "약 먹으래요")])]
    )
    s = Session(ex, spec=POSTVISIT_SPEC)
    s.step("약 먹으래요")
    state = SessionState.model_validate_json(s.to_state().model_dump_json())  # 백엔드 왕복
    assert state.spec == "postvisit"
    s2 = Session.from_state(ScriptedExtractor([]), state)
    assert isinstance(s2.card, PostVisitCard)
    assert s2.card.axes[PostAxis.MEDICATION].value == "약"
    assert s2.asked_axis == PostAxis.HEARD_DIAGNOSIS  # 첫 발화에서 안 채워진 첫 축을 물었다


def test_widening_annotates_heard_terms_with_body_part_and_compares_site():
    onto = load_ontology()
    card = PostVisitCard()
    e = card.axes[PostAxis.HEARD_DIAGNOSIS]
    e.status = FieldStatus.FILLED
    e.value = "앞십자인대가 좀 늘어났다고"
    e.evidence = ["앞십자인대가 좀 늘어났다고 하셨어요"]
    notes = widen_card(card, onto)
    assert len(notes) == 1
    n = notes[0]
    assert n.term == "앞십자인대"
    assert n.anchor_label == "다리"  # 구조 → 앵커로 넓힌다. 좁히지 않는다
    assert card.provenance is None or card.provenance.ontology_snapshot == onto.snapshot_id
    # 진료 전에 다리(무릎)를 짚었으면 같은 부위, 팔이면 다른 부위 — 판정이 아니라 표시
    assert compare_sites(onto, n.anchor_id, notes) == "same"
    assert compare_sites(onto, "ANC:013", notes) == "different"
    assert compare_sites(onto, None, notes) is None


def test_widening_does_not_invent_when_nothing_matches():
    onto = load_ontology()
    card = PostVisitCard()
    e = card.axes[PostAxis.HEARD_DIAGNOSIS]
    e.status = FieldStatus.FILLED
    e.value = "감기라고 하셨어요"
    e.evidence = ["감기라고 하셨어요"]
    assert widen_card(card, onto) == []


def test_export_marks_card_type_and_includes_widening():
    card = PostVisitCard()
    p = to_backend_payload(card)
    assert p["card_type"] == "postvisit"
    assert set(p["axes"]) == {a.value for a in PostAxis}
    assert p["widening"] == [] and p["site_comparison"] is None and p["document_codes"] == []
    assert "department_guidance" not in p  # 진료 후 카드에는 진료과 안내가 없다
