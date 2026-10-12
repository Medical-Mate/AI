"""질문 후보 템플릿 v2 — 녹이기 어미, 자기 검사 폴백, 길이·꼬리 반복. 호출 0."""

import json
import re
from pathlib import Path

import pytest

pytest.importorskip("kiwipiepy")

from medimate.dialog.question_templates import MAX_LEN  # noqa: E402
from medimate.dialog.question_templates_v2 import (  # noqa: E402
    _connective,
    _inflect,
    natural_candidates,
)
from medimate.text.tokenize import base_kiwi  # noqa: E402

CARDS = {
    json.loads(x)["id"]: json.loads(x)
    for x in (Path(__file__).resolve().parents[1] / "evals" / "previsit_cards.jsonl")
    .read_text(encoding="utf-8")
    .splitlines()
    if x
}


@pytest.mark.parametrize(
    "value,expected",
    [
        ("가만히 있어도 아파요", "가만히 있어도 아픈데"),
        ("점점 심해져요", "점점 심해지는데"),
        ("처음보다 조금 심해진 것 같아요", "처음보다 조금 심해진 것 같은데"),
        ("감기 끝나가는 중이에요", "감기 끝나가는 중인데"),
        ("목감기를 앓고 나서였어요", "목감기를 앓고 나서였는데"),  # 이었 → 였
        (
            "오래 걸으면 다리가 무거워져서 쉬어야 해요",
            "오래 걸으면 다리가 무거워져서 쉬어야 하는데",
        ),
        ("가끔 붓기도 해요", "가끔 붓기도 하는데"),
        ("묵직하게 뻐근함", "묵직하게 뻐근한데"),
        ("묵직하게 아프고 다리로 저릿", "묵직하게 아프고 다리로 저릿한데"),
        ("커피 마시면 더 쓰려요", None),  # Kiwi가 쓰려고/하로 잘못 푼다 — 녹이지 않는다
        ("열은 안 나는데 목소리가 좀 갈라져요", None),  # "는데"가 두 번 된다
        ("계단 내려갈 때", None),  # 서술어 없음 — 칸별 붙임(_glue)이 맡는다
    ],
)
def test_connective(value, expected):
    assert _connective(value, base_kiwi()) == expected


@pytest.mark.parametrize("card_id", sorted(CARDS))
def test_v2_fits_and_tails_do_not_repeat(card_id):
    c = CARDS[card_id]
    items = natural_candidates(c["axes"], c.get("profile"))
    tails = [it["text"].rsplit("데 ", 1)[-1] for it in items if "데 " in it["text"]]
    assert all(len(it["text"]) <= MAX_LEN and it["text"].endswith("?") for it in items)
    assert len(tails) == len(set(tails)), items


@pytest.mark.parametrize(
    "value,kind,expected",
    [
        ("비슷해요", "면", "비슷하면"),
        ("어제보다 조금 나아요", "면", "어제보다 조금 나으면"),  # ㅅ 불규칙
        ("계속 그대로예요", "면", "계속 그대로면"),  # 줄어든 서술격
        ("부기는 빠졌는데 걸리는 건 여전해요", "면", "부기는 빠졌는데 걸리는 건 여전하면"),
        ("밤새 더 심해졌어요", "면", None),  # 있었던 일을 가정으로 바꾸지 않는다
        ("어제가 제일 심했고 오늘은 조금 덜해요", "면", None),
        ("한 번 생기면 열흘쯤 있다 아물어요", "면", None),  # 조건이 겹친다
        ("가끔 콕콕 찔러요", "관형", "가끔 콕콕 찌르는"),
        ("명치가 쓰려요", "관형", "명치가 쓰린"),
        ("묵직하게 눌리는 느낌이에요", "관형", None),  # "느낌인 건"은 어색하다
        ("번지는 것 같아요", "관형", None),  # "번지는 것 같은 건"
    ],
)
def test_inflect(value, kind, expected):
    assert _inflect(value, base_kiwi(), kind) == expected


def _texts(card_id: str) -> dict[str, str]:
    c = CARDS[card_id]
    return {it["source"]: it["text"] for it in natural_candidates(c["axes"], c.get("profile"))}


def test_team_review_2026_10_12():
    """팀원 판독(2026-10-12)에서 짚은 자리."""
    # 경과는 조건형 — "비슷한데 계속 이러면 다시 와야 하나요?"가 12장이었다
    assert _texts("PC03")["time_course"].startswith("계속 비슷하면 ")
    # 느낌 값에 이름씨가 없으면 부위를 붙인다 — PC77에서 "가슴"이 빠졌다
    assert _texts("PC77")["character"].startswith("가슴 가운데가 가끔 콕콕 찌르")
    # 악화 명사구는 "~에 더 심한데"
    assert _texts("PC30")["exacerbating"].startswith("아침 첫 걸음에 더 심한데 ")
    assert _texts("PC87")["exacerbating"].startswith("울퉁불퉁한 길에서 더 심한데 ")
    # 한 카드 안에서 잇는 말이 갈린다 — PC48은 모두 "~한데 + 꼬리"였다
    pc48 = _texts("PC48")
    assert " 건 " in pc48["character"] and "것도" in pc48["associated"]


@pytest.mark.parametrize("card_id", sorted(CARDS))
def test_no_awkward_joins(card_id):
    for text in _texts(card_id).values():
        assert not re.search(r"[가-힣]도 [^?]*[^이]것도 ", text), text  # "재채기도 자꾸 하는 것도"
        assert "느낌인 건" not in text and "중인 것도" not in text, text
