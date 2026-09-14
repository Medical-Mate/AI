"""한 문장에 두 주제가 오면 라벨이 하나뿐이라 한쪽이 사라진다(#78).

백엔드가 신고한 문장과, 우리 eval 벡터에 이미 있던 같은 모양(PM20)을 고정한다.
**쪼개면 안 되는 문장을 같이 고정하는 것이 이 파일의 절반이다** — 기계적으로 자르면
"아무 말도 안 하시고" 같은 조각이 생기고, 라벨이 없어 `unsorted`로 떨어진다.
문장 수만 늘고 카드는 나빠진다.
"""

import json
import pathlib
import re

import pytest

from medimate.dialog.memo import split_sentences

VECTORS = pathlib.Path("evals/ondevice/vectors-memo-qwen3-1.7b-q4_0-memo-small-v4.jsonl")

# --- 나눠야 하는 것: 양쪽이 서로 다른 묶음으로 갈 말이다 ---
SPLIT = [
    # 백엔드 신고 문장(#78). 약 + 검사
    (
        "나프록센 500mg 먹으라고 하셨고, 다음 주 화요일에 MRI 찍기로 했어요",
        ["나프록센 500mg 먹으라고 하셨고", "다음 주 화요일에 MRI 찍기로 했어요"],
    ),
    # 우리 벡터에 이미 있던 유실. findings 하나로 묶여 연고 처방이 사라졌다
    ("피부염이라고 하셨고 연고 처방받았어요.", ["피부염이라고 하셨고", "연고 처방받았어요."]),
    ("소염제 처방받았고 2주 뒤에 오래요.", ["소염제 처방받았고", "2주 뒤에 오래요."]),
    ("엑스레이 찍었고 뼈는 괜찮다고 하셨어요.", ["엑스레이 찍었고", "뼈는 괜찮다고 하셨어요."]),
]

# --- 나누면 안 되는 것 ---
KEEP = [
    # 양쪽 다 검사
    "혈액검사 했고 결과는 다음에 알려준대요.",
    "한 달 후에 피검사 다시 하고 결과 보자고.",
    # 양쪽 다 약·생활 지시 — 4축 접기라 한 묶음으로 간다
    "약은 2주분이고 커피랑 매운 거 줄이래요.",
    "물리치료 받으라고 하셨고 파스도 처방.",
    # 앞이 주제가 아니다. 자르면 라벨 없는 조각이 생긴다
    "아무 말도 안 하시고 약만 주셨어요.",
    # 인용 `-라고/-으라고`에서 자르면 술어가 잘린다
    "2주 뒤에 다시 오라고 하셨어요.",
    # `수술`이 `술`(금주)에 걸려 쪼개졌던 자리 — 회귀 실행이 잡아 줬다
    "결과 보고 수술 여부 정한다고.",
]


@pytest.mark.parametrize("memo,expected", SPLIT)
def test_two_topics_are_separated(memo, expected):
    assert split_sentences(memo) == expected


@pytest.mark.parametrize("memo", KEEP)
def test_one_topic_stays_whole(memo):
    assert split_sentences(memo) == [memo]


def _bare(text: str) -> str:
    """공백·쉼표를 뺀 글자만. 자를 때 그 자리의 구분자는 사라진다"""
    return re.sub(r"[\s,]", "", text)


@pytest.mark.parametrize("memo", [m for m, _ in SPLIT] + KEEP)
def test_nothing_is_lost_or_invented(memo):
    """**원문 보존이 이 분리기의 유일한 불변식이다.**

    카드 값이 문장 원문이라, 분리가 글자를 먹으면 환자가 적은 말이 카드에서 사라지고
    글자를 더하면 적지 않은 말이 카드에 남는다. 어느 쪽도 우리가 보증한 것을 깬다.
    """
    assert _bare("".join(split_sentences(memo))) == _bare(memo)


def test_the_vector_file_matches_the_current_rule():
    """폰 벡터 33개가 **지금 규칙으로 나눈 결과와 같아야 한다.**

    벡터의 `sentences`는 폰에 그대로 먹이는 입력이다. 분리 규칙을 고치고 벡터를 안 고치면
    **폰은 서버가 만들지 않는 문장으로 측정된다** — 그 점수는 제품의 점수가 아니다.
    규칙을 넓혀 멀쩡하던 메모가 조각나는 것도 여기서 먼저 깨진다.
    """
    rows = [json.loads(x) for x in VECTORS.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(rows) >= 33
    mismatched = {}
    for r in rows:
        memo = chr(10).join(r["sentences"])  # 마침표 없는 벡터가 있어 줄바꿈으로 잇는다
        now = split_sentences(memo)
        if now != r["sentences"]:
            mismatched[r["id"]] = (len(r["sentences"]), len(now))
    assert mismatched == {}, mismatched


def test_every_vector_is_internally_consistent():
    """문장을 늘렸으면 라벨도·`user` 문구도 같이 늘어야 한다. 손으로 고치다 어긋나는 자리다.

    **`expected_output`은 정답이 아니라 PC 모델이 실제로 낸 출력이다.** `expected_labels`가
    사람 정답이고, 둘이 다른 8건이 곧 기록된 93%(107/115)의 오답이다. 그래서 여기서
    둘의 일치를 요구하면 안 된다 — 요구하는 순간 "모델이 다 맞았다"를 강제하게 된다.
    없을 수도 있다(PC 재실행 전인 벡터). 있으면 모양만 본다.
    """
    rows = [json.loads(x) for x in VECTORS.read_text(encoding="utf-8").splitlines() if x.strip()]
    for r in rows:
        n = len(r["sentences"])
        assert len(r["expected_labels"]) == n, r["id"]
        assert f"문장 {n}개" in r["user"], r["id"]
        for i, sent in enumerate(r["sentences"]):
            assert f"{i}: {sent}" in r["user"], (r["id"], i)
        if r["expected_output"] is not None:
            assert set(json.loads(r["expected_output"])) == {str(i) for i in range(n)}, r["id"]
