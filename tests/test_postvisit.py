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
    # value는 어미를 정리한 줄, evidence는 **문장 원문 그대로**(2026-09-14 ㉡)
    assert c.axes[PostAxis.FINDINGS].value == "위염 초기"
    assert c.axes[PostAxis.FINDINGS].evidence == ["위염 초기라고 하셨어요."]
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
    # 규칙에 없는 어미는 그대로 둔다. 끝 구두점만 뗀다
    assert res.card.axes[PostAxis.FINDINGS].value == "감기래요"
    assert res.card.axes[PostAxis.FINDINGS].evidence == ["감기래요."]
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


def test_the_duration_next_to_the_return_word_wins_not_the_first_one():
    """약 기간이 앞에 있어도 재방문 기간을 쓴다.

    제보: "약 3일치 받았고, 4일 뒤에 방문하세요"인데 캘린더가 3일 뒤로 잡혔다.
    `_bare_duration`이 문장의 **첫** 기간을 집고 있었다 — 그게 약 기간이다.
    """
    v = date(2026, 9, 15)
    assert followup_date("약 3일치 받았고, 4일 뒤에 방문하세요", v).date == "2026-09-19"
    assert followup_date("3일치 약처방 4일후 재방문", v).date == "2026-09-19"
    # `뒤`·`후`가 빠지면 _REL이 안 잡아 첫 기간을 집던 자리
    assert followup_date("3일치 약 먹고 4일에 다시 오세요", v).date == "2026-09-19"


def test_durations_written_as_words():
    """환자는 `7일`보다 `일주일`이라고 쓴다.

    못 읽으면 앞 절 폴백이 돌아 **약 기간**을 재방문으로 쓴다. 조용히 틀린 날짜가 된다.
    """
    v = date(2026, 9, 15)
    assert followup_date("3일 약 먹고 일주일 뒤에 다시 오세요", v, prev_text="3일 약 먹고")
    assert followup_date("일주일 뒤에 다시 오세요", v).date == "2026-09-22"
    assert followup_date("열흘 뒤에 오세요", v).date == "2026-09-25"
    assert followup_date("보름 뒤에 다시 오세요", v).date == "2026-09-30"
    assert followup_date("이주일 후에 재방문", v).date == "2026-09-29"
    # 앞 절 폴백도 단어 기간을 읽는다
    fu = followup_date("다시 오세요", v, prev_text="일주일 약 드시고")
    assert fu.date == "2026-09-22" and "앞 절" in fu.basis


def test_past_visit_is_not_a_return_word():
    """`방문하세요`는 재방문이고 `방문했고`는 지난 방문이다. 후자의 숫자를 끌어오면 안 된다."""
    v = date(2026, 9, 15)
    assert followup_date("3일치 약 받으러 방문했고", v) is None
    assert followup_date("4일 뒤에 내원하세요", v).date == "2026-09-19"


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


def test_two_weeks_written_without_il_is_read_not_stolen_from_the_medication():
    """ "일주일치 약처방받고, 이주뒤 재방문" → 재방문은 2주 뒤다.

    `이주뒤`를 못 읽어 앞 절 폴백이 돌았고 **약 기간 "일주일치"**를 재방문으로 썼다(09-22).
    어제 `일주일`을 고친 것과 같은 모양이 `이주`에서 다시 났다.
    """
    v = date(2026, 9, 15)
    fu = followup_date("이주뒤 재방문", v, prev_text="일주일치 약처방받고")
    assert fu.date == "2026-09-29" and "앞 절" not in fu.basis
    assert followup_date("일주 뒤 오세요", v).date == "2026-09-22"
    assert followup_date("삼주 후 재방문", v).date == "2026-10-06"


# ── 재방문 표현을 LLM이 읽고 날짜는 코드가 센다 (2026-09-15) ──────────────────────


class _Reader:
    def __init__(self, out):
        self.out = out
        self.calls = []

    def read(self, sentence, prev_text=None):
        self.calls.append((sentence, prev_text))
        if isinstance(self.out, Exception):
            raise self.out
        return self.out


def test_the_reader_is_only_asked_when_the_rules_fail_or_guess():
    """규칙이 확실히 읽으면 LLM을 안 부른다. 못 읽거나 앞 절에서 끌어왔을 때만 부른다."""
    from medimate.dialog.memo import classify_memo

    v = date(2026, 9, 15)
    r = _Reader({"text": "이주뒤", "days": 14, "month": None, "day": None})
    classify_memo("2주 뒤에 오라고", FakeClassifier(["follow_up"]), visit_date=v, followup_reader=r)
    assert r.calls == []  # 규칙이 읽었다

    # `담주`는 규칙이 모른다 → 앞 절 폴백이 약 기간을 집으려는 자리 → 여기서 LLM이 읽는다
    r = _Reader({"text": "담주", "days": 7, "month": None, "day": None})
    res = classify_memo(
        "일주일치 약처방받고. 담주에 보자고.",
        FakeClassifier(["medication_instructions", "follow_up"]),
        visit_date=v,
        followup_reader=r,
    )
    assert r.calls and r.calls[0][0] == "담주에 보자고."
    assert res.card.follow_up_date.date == "2026-09-22"
    assert res.card.follow_up_date.basis.startswith("LLM 읽음 '담주' = 7d")


def test_what_the_reader_says_must_be_in_the_sentence_verbatim():
    """LLM이 문장에 없는 표현을 내면 버린다.

    앞 문장에만 있는 것도 버린다 — 그게 약 기간이 재방문으로 넘어오는 길이다.
    """
    from medimate.dialog.memo import followup_from_reader

    v = date(2026, 9, 15)
    assert (
        followup_from_reader(
            _Reader({"text": "2주 뒤", "days": 14, "month": None, "day": None}), "담주에 보자고", v
        )
        is None
    )
    assert (
        followup_from_reader(
            _Reader({"text": "일주일치", "days": 7, "month": None, "day": None}),
            "다시 오세요",
            v,
            prev_text="일주일치 약",
        )
        is None
    )
    assert (
        followup_from_reader(
            _Reader({"text": "담주", "days": 0, "month": None, "day": None}), "담주에 보자고", v
        )
        is None
    )
    assert (
        followup_from_reader(
            _Reader({"text": "담주", "days": 9999, "month": None, "day": None}), "담주에 보자고", v
        )
        is None
    )
    assert followup_from_reader(_Reader(RuntimeError("boom")), "담주에 보자고", v) is None


def test_the_reader_can_give_an_absolute_date():
    from medimate.dialog.memo import followup_from_reader

    v = date(2026, 9, 15)
    fu = followup_from_reader(
        _Reader({"text": "10월 3일", "days": None, "month": 10, "day": 3}), "10월 3일에 오라고", v
    )
    assert fu.date == "2026-10-03" and not fu.approximate
    fu = followup_from_reader(
        _Reader({"text": "1월 5일", "days": None, "month": 1, "day": 5}), "1월 5일에 오라고", v
    )
    assert fu.date == "2027-01-05"  # 지난 날짜는 다음 해


def test_the_week_after_next_is_two_weeks_not_one():
    """`다다음주` 안에서 `다음 주`가 걸려 7일이 되고 있었다 — followup-v1 실측표에서 드러났다.

    규칙이 **자신 있게 틀리는** 경우라 LLM 폴백이 안 불린다. 규칙을 고쳐야 한다.
    """
    v = date(2026, 9, 15)
    assert followup_date("다다음주에 재방문", v).date == "2026-09-29"
    assert followup_date("다음 주에 다시", v).date == "2026-09-22"
