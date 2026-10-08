"""질문 후보 템플릿 v2 — 녹이기 어미, 자기 검사 폴백, 길이·꼬리 반복. 호출 0."""

import json
from pathlib import Path

import pytest

pytest.importorskip("kiwipiepy")

from medimate.dialog.question_templates import MAX_LEN  # noqa: E402
from medimate.dialog.question_templates_v2 import _connective, natural_candidates  # noqa: E402
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
