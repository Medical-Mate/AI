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
    assert ax[PostAxis.MEDICATION_INSTRUCTIONS].value == "2주 약 먹음 · 커피랑 매운 거 줄이기"
    # **`value`에 날짜를 넣지 않는다.** 날짜의 주인은 `follow_up_date` 하나다
    assert ax[PostAxis.FOLLOW_UP].value == "다시 오기"
    assert "9월" not in ax[PostAxis.FOLLOW_UP].value
    assert res.card.follow_up_date.date == "2026-09-26"  # 예전에는 null이었다
    # evidence는 원문 그대로 — 다듬은 것은 value뿐이다
    assert ax[PostAxis.FINDINGS].evidence == ["위염 초기라고 하셨고"]


# --- 날짜의 주인은 하나다 (2026-09-14) --------------------------------------


def test_the_date_appears_in_one_place_only():
    """화면에 날짜가 두 번 찍혔다 — `"재방문 4일후 (2026년 9월 21일) (2026년 9월 21일)"`.

    `value`에 `"{말} ({M월 D일} 전후)"`를 조합해 넣었는데 `follow_up_date`가 이미 같은 날짜를
    들고 있었다. **조합할 수 있는 자리가 둘이 되면 양쪽이 다 그린다.** 층을 되돌린다 —
    `value`는 다른 축과 같게 메모에서 온 말만 담는다.
    """
    res = classify_memo(
        "3일치 약처방 4일후 재방문", Fixed(["medication_instructions", "follow_up"]), visit_date=V
    )
    from medimate.schema.postvisit import PostAxis

    value = res.card.axes[PostAxis.FOLLOW_UP].value
    assert value and "월" not in value and "(" not in value
    assert res.card.follow_up_date.date == "2026-09-16"  # 날짜는 여기만(V + 4일)


# --- 띄어쓰기만 있는 전보문 -------------------------------------------------


@pytest.mark.parametrize(
    "memo,expected",
    [
        # 실제 입력. 연결어미도 구두점도 없다 — 한 문장으로 가면 약 처방이 재방문에 묻힌다
        ("3일치 약처방 4일후 재방문", ["3일치 약처방", "4일후 재방문"]),
        ("약 3일분 2주 뒤 다시 오세요", ["약 3일분", "2주 뒤 다시 오세요"]),
    ],
)
def test_telegraphic_memo_splits(memo, expected):
    assert split_sentences(memo) == expected


@pytest.mark.parametrize(
    "memo",
    [
        # 술어 한가운데서 자르면 문장이 부서진다 — 조사·연결어미로 끝나면 자르지 않는다
        "짜게 먹지 말라고 하셨어요.",
        "갑상선 수치가 약간 높다고.",
        "위내시경은 다다음주 월요일 아침으로 잡았다.",
        "다음 주 화요일에 MRI 찍기로 했어요",
        # 한 낱말·짧은 조각만 남는 자리
        "10월 2일 시야검사 예약함.",
        "엑스레이는 이상 없음.",
        # 주제가 한쪽뿐이면 안 자른다
        "혈압약 처방 2주 후 재검",
    ],
)
def test_telegraphic_memo_does_not_over_split(memo):
    assert split_sentences(memo) == [memo]


# ── 절 끝에 남은 연결어미 `-고` (2026-09-15 QA) ─────────────────────────────────


def test_a_trailing_connective_becomes_a_nominal_not_a_raw_slice():
    """문장을 절로 나누면 "일주일치 약처방받고"처럼 `-고`가 꼬리에 남는다.

    그대로 두면 카드가 메모를 잘라 놓은 것과 같다 — 그래서 "슬라이싱만 된다"고 보였다.
    """
    from medimate.dialog.memo import tidy_value

    assert tidy_value("일주일치 약처방받고") == "일주일치 약처방받음"
    assert tidy_value("3일치 약 먹고") == "3일치 약 먹음"
    assert tidy_value("물 많이 마시고") == "물 많이 마심"  # 받침 없는 어간은 ㅁ을 합성한다
    assert tidy_value("연고 바르고") == "연고 바름"


def test_go_that_is_not_a_connective_is_left_alone():
    """`사고`·`경고`·`참고`의 `고`는 어미가 아니다. 어간 목록에 없으면 손대지 않는다."""
    from medimate.dialog.memo import tidy_value

    assert tidy_value("교통사고") == "교통사고"
    assert tidy_value("결과 보고") == "결과 보고"  # report인지 "보고(see and)"인지 알 수 없다
    assert tidy_value("커피랑 매운 거 줄이라고.") == "커피랑 매운 거 줄이기"  # 인용 규칙이 먼저


# ── 칸 이름과 겹치는 꼬리 낱말 (2026-09-15) ────────────────────────────────────────


def test_a_word_the_label_already_says_is_dropped_from_the_tail():
    """ "일주일치 약처방 이주일 후 재방문" → 약 칸 `일주일치`, 재방문 칸 `이주일 후`.

    약 칸에 `약처방`, 재방문 칸에 `재방문`이 또 있으면 잘라 놓기만 한 것으로 읽힌다. 라벨이
    이미 말하는 낱말이라 빼도 사실이 안 바뀐다.
    """
    from medimate.dialog.memo import tidy_value
    from medimate.schema.postvisit import PostAxis

    M, F = PostAxis.MEDICATION_INSTRUCTIONS, PostAxis.FOLLOW_UP
    assert tidy_value("일주일치 약처방", M) == "일주일치"
    assert tidy_value("일주일치 약처방받고", M) == "일주일치"  # 연결어미 정리 뒤에도 뗀다
    assert tidy_value("이주일 후 재방문", F) == "이주일 후"
    assert tidy_value("2주 뒤에 재방문", F) == "2주 뒤"  # 남은 조사도 뗀다
    assert tidy_value("3일치 약처방", M) == "3일치"


def test_the_trailer_is_kept_when_nothing_would_remain_or_axis_is_unknown():
    from medimate.dialog.memo import tidy_value
    from medimate.schema.postvisit import PostAxis

    assert tidy_value("재방문", PostAxis.FOLLOW_UP) == "재방문"  # 떼면 빈다
    assert tidy_value("약처방", PostAxis.MEDICATION_INSTRUCTIONS) == "약처방"
    assert tidy_value("이주일 후 재방문") == "이주일 후 재방문"  # 칸을 모르면 안 뗀다
    assert (
        tidy_value("위산 줄이는 약", PostAxis.MEDICATION_INSTRUCTIONS) == "위산 줄이는 약"
    )  # `약`만은 안 뗀다


def test_the_reported_memo_end_to_end():
    from datetime import date

    from medimate.dialog.memo import classify_memo
    from medimate.schema.postvisit import PostAxis

    res = classify_memo(
        "일주일치 약처방 이주일 후 재방문",
        Fixed(["medication_instructions", "follow_up"]),
        visit_date=date(2026, 9, 15),
    )
    ax = res.card.axes
    assert ax[PostAxis.MEDICATION_INSTRUCTIONS].value == "일주일치"
    assert ax[PostAxis.FOLLOW_UP].value == "이주일 후"
    assert res.card.follow_up_date.date == "2026-09-29"
    # 원문은 그대로 남는다
    assert ax[PostAxis.FOLLOW_UP].evidence == ["이주일 후 재방문"]


# ── 운영에서 원문 그대로 남은 것들 (2026-09-15 저녁, 백엔드 #107) ────────────────────


def test_noun_plus_igo_at_the_tail_is_dropped():
    """조각 경로가 혼합 문장을 주제 경계에서 나누면 앞 조각에 `이고`가 남는다.

    "약은 2주분이고 커피 줄이래요" → "약은 2주분이고" / "커피 줄이래요".
    """
    from medimate.dialog.memo import tidy_value
    from medimate.schema.postvisit import PostAxis

    assert tidy_value("약은 2주분이고", PostAxis.MEDICATION_INSTRUCTIONS) == "약은 2주분"
    assert tidy_value("약 먹이고") == "약 먹임"  # 동사 `먹이`는 동사 규칙이 먼저 잡는다
    assert tidy_value("2주분이고") == "2주분"


def test_more_stems_from_production_memos():
    from medimate.dialog.memo import tidy_value

    assert tidy_value("베개 낮은 걸로 바꾸래요") == "베개 낮은 걸로 바꾸기"
    assert tidy_value("무거운 거 들지 말라고") == "무거운 거 들지 않기"
    assert tidy_value("자세 자주 바꾸고") == "자세 자주 바꿈"
