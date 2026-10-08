"""질문 후보 검증기 — 걸려야 하는 질문, 통과해야 하는 질문, 세트 단위 규칙. 호출 0."""

import json
from pathlib import Path

import pytest

pytest.importorskip("kiwipiepy")

from medimate.dialog.question_verify import finalize, verify_items  # noqa: E402
from medimate.evals.run_assist import TEST_DRUG_TERMS  # noqa: E402
from medimate.evals.score import load_lexicon  # noqa: E402

CARDS = {
    json.loads(x)["id"]: json.loads(x)
    for x in (Path(__file__).resolve().parents[1] / "evals" / "previsit_cards.jsonl")
    .read_text(encoding="utf-8")
    .splitlines()
    if x
}
LEX = load_lexicon() + TEST_DRUG_TERMS


def _reasons(card_id: str, text: str) -> list[str]:
    _, dropped = verify_items(CARDS[card_id], [{"text": text, "source": "x"}], LEX)
    return dropped[0]["reasons"] if dropped else []


@pytest.mark.parametrize(
    "card_id,text,reason",
    [
        # LoRA v1 출력에서 사람이 찾은 실패(2026-10-08)
        ("PC47", "술을 마시면 더 아픈 건가요?", "unsourced"),  # 카드에 술이 없다
        ("PC83", "깃털이랑 진드기랑 꽃가루가 관련이 있을까요?", "unsourced"),
        ("PC80", "아이를 안아야 해서 손을 안 쓸 수가 있는데 관련이 있을까요?", "negation_dropped"),
        ("PC99", "낮에 많이 걸은 날 밤에 더 나은 건 왜일까요?", "unsourced"),  # 나요 → 나은(낫다)
        ("PC81", "디스크라고 했는데 진단은 어떤 상태인가요?", "diagnosis_ask"),  # 엄마가 들은 병명
        ("PC89", "다리가 저릿한 건 협착증과 관련이 있나요?", "other_person_dx"),  # 직접 찾아본 병명
        ("PC01", "진통제 먹어도 완전히 좋아지지 않나요?", "negation_added"),
        (
            "PC01",
            "계단 내려갈 때만 시큰한데 평지는 괜찮은 게 이상한 건가요?",
            "unsourced",
        ),  # v8 예시 누출
        ("PC35", "힘줄이 상했을 가능성도 있나요?", "guess"),
        ("PC01", "MRI를 찍어봐야 하나요?", "unsourced"),
    ],
)
def test_rejects(card_id, text, reason):
    assert any(r.startswith(reason) for r in _reasons(card_id, text)), _reasons(card_id, text)


@pytest.mark.parametrize(
    "card_id,text",
    [
        ("PC28", "운동을 계속하면 안 되나요?"),  # 틀의 부정("~하면 안 되나요")
        ("PC08", "멍이 들었는데 걱정할 건 없나요?"),  # "없나요"는 묻는 방식이다
        ("PC13", "엄지·검지·중지가 저리는 건 어떤 상태인가요?"),  # 저려요 → 저리(어간)
        ("PC02", "혈압약 먹고 있는데 계속 먹어도 되나요?"),
        ("PC27", "옷 입을 때나 머리 감을 때 더 심한데 그런 동작을 피해야 하나요?"),
    ],
)
def test_passes(card_id, text):
    assert _reasons(card_id, text) == []


def test_set_level_duplicate_and_single_why():
    items = [
        {"text": "왼쪽 무릎 앞쪽이 시큰한 건 어떤 상태인가요?", "source": "character"},
        {"text": "왼쪽 무릎 앞쪽이 시큰한 건 어떤 상태인가요?", "source": "character"},
        {"text": "계단 내려갈 때 더 시큰한 건 왜일까요?", "source": "exacerbating"},
        {"text": "쪼그려 앉으면 아픈 건 왜 그런가요?", "source": "associated"},
    ]
    kept, dropped = verify_items(CARDS["PC91"], items, LEX)
    assert [d["reasons"][0] for d in dropped] == ["duplicate", "why_repeat"]
    assert len(kept) == 2


def test_finalize_fills_with_templates_to_three():
    out, _ = finalize(CARDS["PC47"], [{"text": "술을 마시면 더 아픈 건가요?", "source": "x"}], LEX)
    assert len(out) >= 3 and all(o["origin"] == "template" for o in out)
