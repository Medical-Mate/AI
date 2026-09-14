"""앱 화면에서 나온 실제 메모로 고정한다 (2026-09-14).

화면 메모(쉼표 뒤 띄어쓰기 없음)에서 **소견 줄이 통째로 사라지고 재방문 날짜가 null**이었다.
둘 다 분리 규칙 문제였다 — 실제 사용자는 쉼표 뒤를 안 띄운다.
"""

from datetime import date

import pytest

from medimate.dialog.memo import (
    MemoLabels,
    classify_memo,
    followup_date,
    split_sentences,
    tidy_value,
)

SCREEN = (
    "위염 초기라고 하셨고,2주 약 먹고 다시 오라고 했어요. "
    "커피랑 매운 거 줄이라고. 피검사는 다음에 결과 보자고 하셨음."
)


class Fixed:
    model_id = "fake"
    prompt_version = "test"

    def __init__(self, labels):
        self.labels = labels

    def classify(self, sentences):
        return MemoLabels.from_keyed(
            {str(i): self.labels[i] for i in range(len(sentences))}, len(sentences)
        )


# --- 버그 1: 공백 없이 붙여 쓴 경계 -----------------------------------------


def test_the_screen_memo_splits_into_five():
    assert split_sentences(SCREEN) == [
        "위염 초기라고 하셨고",
        "2주 약 먹고",
        "다시 오라고 했어요.",
        "커피랑 매운 거 줄이라고.",
        "피검사는 다음에 결과 보자고 하셨음.",
    ]


@pytest.mark.parametrize(
    "memo,expected",
    [
        # 쉼표 뒤 공백 없음 — 실제 입력
        ("위염 초기라고 하셨고,2주 약 먹고", ["위염 초기라고 하셨고", "2주 약 먹고"]),
        # 마침표 뒤 공백 없음
        ("하셨음.피검사는 다음에", ["하셨음.", "피검사는 다음에"]),
        # 자르면 안 되는 것 — 숫자·영문 사이의 점은 문장 끝이 아니다
        ("나프록센 2.5mg 드세요", ["나프록센 2.5mg 드세요"]),
        ("example.com 참고", ["example.com 참고"]),
        # 낱말 안의 `고`
        ("사고 났어요", ["사고 났어요"]),
        ("그리고 약도", ["그리고 약도"]),
    ],
)
def test_boundaries(memo, expected):
    assert split_sentences(memo) == expected


# --- 버그 2: "N주 약 먹고 다시 오라고" ---------------------------------------

V = date(2026, 9, 12)


@pytest.mark.parametrize(
    "text,prev,expect",
    [
        ("2주 약 먹고 다시 오라고 했어요", None, "2026-09-26"),
        ("2주 약 드시고 오세요", None, "2026-09-26"),
        # 절이 갈린 경우 — 앞 절에서 기간을 끌어온다
        ("다시 오라고 했어요", "2주 약 먹고", "2026-09-26"),
        # 앞 절이 약·기간 말이 아니면 끌어오지 않는다
        ("다시 오라고 했어요", "피검사는 다음에", None),
        ("다시 오라고 했어요", "위염 초기라고 하셨고", None),
        # 재방문 말이 없으면 기간만으로 잡지 않는다 — 그건 약 기간이다
        ("2주 약 먹고", None, None),
        ("3일치 처방받았어요", None, None),
    ],
)
def test_followup(text, prev, expect):
    fu = followup_date(text, V, prev_text=prev)
    assert (fu.date if fu else None) == expect


def test_borrowed_duration_says_so_in_basis():
    """여기가 유일하게 추론이 들어가는 자리다. 카드를 읽는 사람이 어디서 온 날짜인지 알아야 한다"""
    fu = followup_date("다시 오라고 했어요", V, prev_text="2주 약 먹고")
    assert "앞 절" in fu.basis and "2주 약 먹고" in fu.basis


# --- ㉡ 어미 정리 -----------------------------------------------------------


@pytest.mark.parametrize(
    "raw,tidy",
    [
        # 인용(명사) — 꼬리만 뗀다
        ("위염 초기라고 하셨고", "위염 초기"),
        ("열은 바이러스성이라고 함", "열은 바이러스성"),
        # 명령(동사) — 명사형
        ("커피랑 매운 거 줄이라고.", "커피랑 매운 거 줄이기"),
        ("나프록센 500mg 먹으라고 하셨고", "나프록센 500mg 먹기"),
        ("하루 두 번 바르래요.", "하루 두 번 바르기"),
        ("탈수 안 되게 물 자주 먹이라고", "탈수 안 되게 물 자주 먹이기"),
        ("무거운 거 들지 말라고.", "무거운 거 들지 않기"),
        # 평서 인용은 어미를 되살린다 — 떼기만 하면 어간이 맨몸으로 남는다
        ("뼈는 괜찮다고 하셨어요.", "뼈는 괜찮다"),
        ("손목 금이 갔다고 함", "손목 금이 갔다"),
        ("결과는 다음에 알려준대요.", "결과는 다음에 알려준다"),
        # 규칙에 없는 모양은 **그대로 둔다**. 끝 구두점만 뗀다
        ("항히스타민제 하루 한 알", "항히스타민제 하루 한 알"),
        ("반깁스 4주", "반깁스 4주"),
        ("수영 금지.", "수영 금지"),
        ("오늘은 스케일링만 했음", "오늘은 스케일링만 했음"),
        ("해열제는 38도 넘을 때만", "해열제는 38도 넘을 때만"),
    ],
)
def test_tidy_value(raw, tidy):
    assert tidy_value(raw) == tidy


def test_tidy_never_invents_or_substitutes():
    """동의어 치환·단어 추가는 하지 않는다. 여기가 ㉡과 ㉢의 경계다"""
    for raw in ("피검사는 다음에 결과 보자고 하셨음.", "물리치료 받으라고 하셨고"):
        out = tidy_value(raw)
        assert "혈액검사" not in out and "시행" not in out and "처방" not in out
        # 숫자·단위·약 이름은 손대지 않는다
    assert "500mg" in tidy_value("나프록센 500mg 먹으라고 하셨고")


def test_tidy_never_empties_a_value():
    """다듬다가 값을 없애는 것이 제일 나쁘다"""
    for raw in ("라고 하셨어요", "함", "다고"):
        assert tidy_value(raw)


# --- 카드 전체 --------------------------------------------------------------


def test_the_screen_memo_produces_a_readable_card():
    res = classify_memo(
        SCREEN,
        Fixed(
            ["findings", "medication_instructions", "follow_up", "medication_instructions", "tests"]
        ),
        visit_date=V,
    )
    ax = res.card.axes
    from medimate.schema.postvisit import PostAxis

    assert ax[PostAxis.FINDINGS].value == "위염 초기"  # 예전에는 줄 자체가 없었다
    assert ax[PostAxis.MEDICATION_INSTRUCTIONS].value == "2주 약 먹고 · 커피랑 매운 거 줄이기"
    assert ax[PostAxis.FOLLOW_UP].value == "2주 (9월 26일 전후)"
    assert res.card.follow_up_date.date == "2026-09-26"  # 예전에는 null이었다
    # evidence는 원문 그대로 — 다듬은 것은 value뿐이다
    assert ax[PostAxis.FINDINGS].evidence == ["위염 초기라고 하셨고"]
