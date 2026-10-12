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
        # 같은 말의 다른 꼴만 받는다 — 카드에 없는 변화("가빠요"→"가빠지는")는 여전히 버린다
        ("PC90", "숨이 살짝 가빠지는 건 가슴 답답함과 관련이 있나요?", "unsourced"),
        (
            "PC22",
            "손가락 둘째·셋째 마디가 점점 여러 군데로 늘어나는 건 이상한 건가요?",
            "unsourced",
        ),
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
        # 2026-10-12 팀원 판독 — 카드 "찔러요"·"넓어졌어요"와 같은 말
        ("PC29", "숨을 크게 쉴 때 콕 찔리는 건 어떤 상태인가요?"),
        ("PC77", "가슴 가운데가 콕콕 찔리는 건 어떤 상태인가요?"),
        ("PC39", "따끔한 자리가 조금 넓어졌는데 걱정할 게 있나요?"),
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


def test_finalize_skips_template_the_model_already_said():
    """PC48 LoRA v1 — 모델이 느낌 값을 쓰고 악화라 달았다. 같은 말의 느낌 템플릿을 또
    넣지 않는다."""
    from medimate.dialog.question_templates_v2 import natural_candidates

    model = {"text": "밤에 누우면 더 아픈 건 어떤 상태인가요?", "source": "exacerbating"}
    out, _ = finalize(CARDS["PC48"], [model], LEX, templates=natural_candidates)
    assert out[0]["origin"] == "model"
    assert not any("누우면" in o["text"] for o in out[1:]), out
    assert len(out) == 3


def test_duplicate_ignores_site_words_when_asking_the_same():
    """PC77(2차 판독) — 부위를 붙였을 뿐 같은 것을 묻는 질문은 반복. 묻는 것이 다르면 둔다."""
    same = [
        {"text": "가끔 콕콕 찔리는 건 어떤 상태인가요?", "source": "character"},
        {"text": "가슴 가운데가 콕콕 찔리는 건 어떤 상태인가요?", "source": "character"},
    ]
    kept, dropped = verify_items(CARDS["PC77"], same, LEX)
    assert len(kept) == 1 and dropped[0]["reasons"] == ["duplicate"]
    other = [same[0], {"text": "가슴 가운데가 콕콕 찔리면 다시 와야 하나요?", "source": "x"}]
    kept, _ = verify_items(CARDS["PC77"], other, LEX)
    assert len(kept) == 2
