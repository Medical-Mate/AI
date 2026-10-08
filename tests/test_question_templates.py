"""질문 후보 템플릿 — 지어내기 0·조사·부정 답·환자 질문·길이·중복(2026-10-08)."""

import json
import re
from pathlib import Path

import pytest

from medimate.dialog.question_templates import MAX_LEN, template_candidates

CARDS = [
    json.loads(x)
    for x in (Path(__file__).resolve().parents[1] / "evals" / "previsit_cards.jsonl")
    .read_text(encoding="utf-8")
    .splitlines()
    if x
]
QUOTED = re.compile(r"“([^”]*)”")


@pytest.mark.parametrize("card", CARDS, ids=[c["id"] for c in CARDS])
def test_every_quote_is_patient_text_and_fits(card):
    """인용은 카드 값(또는 그 앞부분)이다 — 새 말을 만들지 않는다. 35자 안, 물음표로 끝난다."""
    source = json.dumps(card, ensure_ascii=False)
    for it in template_candidates(card["axes"], card.get("profile")):
        assert len(it["text"]) <= MAX_LEN and it["text"].endswith("?")
        for q in QUOTED.findall(it["text"]):
            assert q in source, (card["id"], it["text"])


def test_particles_follow_the_last_syllable():
    items = template_candidates(
        {
            "character": "쓰려요",
            "exacerbating": "계단 오를 때",
            "time_course": "오른쪽 엉덩이까지 뻐근",
        },
        {"medications": ["진통제(가끔)"]},
    )
    texts = [it["text"] for it in items]
    assert "“진통제(가끔)”는 계속 써도 되나요?" in texts  # 괄호 앞 낱말에 붙인다
    assert "“계단 오를 때”는 어떻게 해야 하나요?" in texts
    assert "“쓰려요”는 어떤 상태인가요?" in texts
    assert "“오른쪽 엉덩이까지 뻐근”이라면 언제 다시 와야 하나요?" in texts


def test_negative_answers_are_not_wrapped():
    items = template_candidates(
        {
            "associated": "다리 저림은 없어요",
            "radiation": "허리는 안 아파요",
            "character": "뻐근해요",
        }
    )
    assert [it["source"] for it in items] == ["character"]


def test_patient_question_is_kept_as_is():
    items = template_candidates({"associated": "이렇게 자꾸 꺾이는 게 정상인가요"})
    assert items == [{"text": "이렇게 자꾸 꺾이는 게 정상인가요?", "source": "associated"}]
    # "나요"로 끝나도 서술이면 질문이 아니다
    assert (
        template_candidates({"associated": "열이 나요"})[0]["text"]
        == "“열이 나요”도 관련이 있나요?"
    )


def test_long_value_uses_first_clause_and_short_values_are_skipped():
    items = template_candidates(
        {"exacerbating": "밥 먹고 30분쯤 지나면, 밤에 더 심하고 누우면 조금 나아지는 것 같아요"}
    )
    assert items[0]["text"] == "“밥 먹고 30분쯤 지나면”은 어떻게 해야 하나요?"
    assert template_candidates({"exacerbating": "밤에"}) == []


def test_same_words_are_quoted_once_and_sparse_cards_are_not_padded():
    items = template_candidates({"exacerbating": "긁으면 번져요", "time_course": "긁으면 번져요"})
    assert [it["source"] for it in items] == ["exacerbating"]  # 경과의 같은 말은 건너뜀
    # 재료가 적으면 적게 낸다(억지로 3개를 채우지 않는다). 시작 시점은 3개가 안 될 때만 채운다
    items = template_candidates({"character": "뻐근해요", "onset": "며칠 됐어요"})
    assert [it["source"] for it in items] == ["character", "onset"]
    assert template_candidates({}) == []
