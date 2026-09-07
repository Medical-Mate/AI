"""진료 후(챗봇②) — 메모 → 4묶음 분류가 주 경로. 보조 문답은 같은 엔진. LLM 호출 없음."""

from datetime import date

from medimate.dialog import POSTVISIT_SPEC, Session
from medimate.dialog.memo import (
    MemoLabels,
    classify_memo,
    followup_date,
    split_sentences,
)
from medimate.dialog.postvisit_questions import MESSAGE_QUESTION, OPENING, QUESTIONS
from medimate.dialog.widening import compare_sites, widen_card
from medimate.llm import AxisUpdate, TurnExtraction
from medimate.ontology import load_ontology
from medimate.schema import Axis, FieldStatus
from medimate.schema.export import to_backend_payload
from medimate.schema.postvisit import PostAxis, PostVisitCard
from tests.fakes import ScriptedExtractor


class FakeClassifier:
    model_id = "fake"
    prompt_version = "test"

    def __init__(self, labels):
        self._labels = labels

    def classify(self, sentences):
        return MemoLabels.model_validate({"labels": list(self._labels)})


def test_split_sentences_on_period_newline_and_dot_separator():
    memo = "위염 초기라고 하셨어요. 혈액검사 했어요\n약은 2주분 · 커피 줄이래요"
    assert split_sentences(memo) == [
        "위염 초기라고 하셨어요.",
        "혈액검사 했어요",
        "약은 2주분",
        "커피 줄이래요",
    ]


def test_classify_memo_buckets_sentences_verbatim_and_keeps_unsorted():
    memo = (
        "위염 초기라고 하셨어요. 혈액검사 했고 결과는 다음에. 약은 2주분. 2주 뒤에 오라고. "
        "병원이 붐볐다."
    )
    res = classify_memo(
        memo,
        FakeClassifier(["findings", "tests", "medication_instructions", "follow_up", "none"]),
        visit_date=date(2026, 9, 12),
        clinic="서울OO병원 내과",
    )
    c = res.card
    assert c.memo == memo and c.clinic == "서울OO병원 내과"
    assert c.axes[PostAxis.FINDINGS].value == "위염 초기라고 하셨어요."
    assert c.axes[PostAxis.FINDINGS].evidence == ["위염 초기라고 하셨어요."]  # 원문 그대로
    assert c.axes[PostAxis.TESTS].status == FieldStatus.FILLED
    assert c.unsorted == ["병원이 붐볐다."]  # 버리지 않는다
    assert c.follow_up_date is not None
    assert c.follow_up_date.date == "2026-09-26" and c.follow_up_date.approximate
    assert c.is_minimally_complete()


def test_classify_memo_guards_bad_indices_and_marks_missing_buckets_unknown():
    class Bad:
        model_id = "fake"
        prompt_version = "test"

        def classify(self, sentences):
            # 문장은 2개인데 라벨 3개(초과) — 위치 기반이라 셋째는 버린다
            return MemoLabels.model_validate({"labels": ["findings", "none", "tests"]})

    res = classify_memo("감기래요. 해열제 먹으래요.", Bad())
    assert res.card.axes[PostAxis.FINDINGS].value == "감기래요."
    assert res.card.axes[PostAxis.TESTS].status == FieldStatus.UNKNOWN  # 메모에 없었다
    assert [d["reason"] for d in res.dropped] == ["extra_label"]
    assert res.card.unsorted == ["해열제 먹으래요."]  # none 라벨 문장은 unsorted
    short = classify_memo("감기래요. 해열제 먹으래요.", FakeClassifier(["findings"]))
    assert short.dropped[0]["reason"] == "missing_label"
    assert short.card.unsorted == ["해열제 먹으래요."]


def test_followup_date_relative_and_absolute():
    v = date(2026, 9, 12)
    assert followup_date("2주 뒤에 오라고", v).date == "2026-09-26"
    assert followup_date("한 달 뒤 보자고", v).date == "2026-10-12"
    assert followup_date("다음 주에 다시", v).date == "2026-09-19"
    fu = followup_date("9월 30일에 오라고", v)
    assert fu.date == "2026-09-30" and not fu.approximate
    assert followup_date("1월 5일", v).date == "2027-01-05"  # 지난 날짜는 다음 해
    assert followup_date("안 좋아지면 바로 오래요", v) is None


def test_export_marks_card_type_and_includes_memo_fields():
    res = classify_memo(
        "위염 초기라고 하셨어요. 2주 뒤에 오라고.",
        FakeClassifier(["findings", "follow_up"]),
        visit_date=date(2026, 9, 12),
    )
    p = to_backend_payload(res.card)
    assert p["card_type"] == "postvisit"
    assert set(p["axes"]) == {a.value for a in PostAxis}
    assert p["widening"] == [] and p["site_comparison"] is None
    assert p["follow_up_date"]["date"] == "2026-09-26"
    assert p["memo"].startswith("위염") and p["unsorted"] == []
    assert "department_guidance" not in p


def test_widening_annotates_findings_terms_and_compares_site():
    onto = load_ontology()
    res = classify_memo(
        "앞십자인대가 좀 늘어났다고 하셨어요.",
        FakeClassifier(["findings"]),
    )
    notes = widen_card(res.card, onto)
    assert len(notes) == 1 and notes[0].term == "앞십자인대" and notes[0].anchor_label == "다리"
    assert compare_sites(onto, notes[0].anchor_id, notes) == "same"
    assert compare_sites(onto, "ANC:013", notes) == "different"


def test_widening_does_not_invent_when_nothing_matches():
    onto = load_ontology()
    res = classify_memo("감기라고 하셨어요.", FakeClassifier(["findings"]))
    assert widen_card(res.card, onto) == []


def test_fallback_dialog_asks_four_axes_then_message():
    """메모가 비었을 때의 보조 문답. 같은 엔진, 진료 후 명세."""

    def filled(axis, value, evidence):
        return AxisUpdate(axis=axis, status=FieldStatus.FILLED, value=value, evidence=evidence)

    ex = ScriptedExtractor(
        [
            TurnExtraction(updates=[filled(PostAxis.FINDINGS, "위염 초기", "위염 초기래요")]),
            TurnExtraction(updates=[filled(PostAxis.TESTS, "피검사", "피검사 했어요")]),
            TurnExtraction(updates=[filled(PostAxis.MEDICATION_INSTRUCTIONS, "2주분", "2주분 약")]),
            TurnExtraction(updates=[filled(PostAxis.FOLLOW_UP, "2주 뒤", "2주 뒤에 오라고")]),
            TurnExtraction(),
        ]
    )
    s = Session(ex, spec=POSTVISIT_SPEC)
    assert isinstance(s.card, PostVisitCard) and s.opening() == OPENING
    assert s.step("위염 초기래요") == QUESTIONS[PostAxis.TESTS]
    s.step("피검사 했어요")
    s.step("2주분 약")
    assert s.step("2주 뒤에 오라고") == MESSAGE_QUESTION
    s.step("없어요")
    assert s.ended and s.card.completeness() == 1.0
    assert Axis.SITE not in s.card.axes  # 진료 전 축은 섞이지 않는다
