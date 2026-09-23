"""템플릿 후보(canon)와 정규화 연동. kiwipiepy 없으면 건너뛴다."""

import pytest

pytest.importorskip("kiwipiepy")

from medimate.span.candidates import CandidateGenerator  # noqa: E402
from medimate.span.select import RuleSelector, resolve  # noqa: E402


@pytest.fixture(scope="module")
def gen():
    return CandidateGenerator()


def pick(gen, seg, axis):
    cset = gen.generate(seg, axis)
    return resolve(cset, RuleSelector().select(cset, axis)), [c.text for c in cset.candidates]


@pytest.mark.parametrize(
    "seg, axis, gold",
    [
        ("혈압이 좀 높다고 하셨어요.", "findings", "혈압이 좀 높음"),
        ("귀에 물이 찼다고.", "findings", "귀에 물이 찼음"),
        ("초음파 봤는데 파열은 아니래요.", "findings", "파열은 아님"),
        ("오른쪽 발목 염좌라고.", "findings", "오른쪽 발목 염좌"),
        ("중이염은 아니고 귀지 때문이라고.", "findings", "중이염"),
        ("안 좋아지면 바로 오래요.", "follow_up", "안 좋아지면 재방문 필요"),
        ("심해지면 오라고", "follow_up", "악화 시 재방문 필요"),
        ("2주 뒤에 다시 오라고 하셨어요.", "follow_up", "2주 뒤"),
        ("다시 오라고 했어요.", "follow_up", "재방문 필요"),
        ("공복혈당 검사 3개월 뒤 다시.", "tests", "3개월 후 공복혈당 검사"),
        ("다음 주 화요일에 MRI 찍기로 했어요", "tests", "다음 주 화요일 MRI"),
        ("회전근개 쪽 문제일 수 있다고 하셨어요.", "findings", "회전근개"),
        ("주사 한 대 맞았어요.", "medication_instructions", None),
        ("충치 치료 받으라고 하셨어요", "medication_instructions", "충치 치료"),
        ("오늘은 스케일링만 했음.", "medication_instructions", None),
        ("MRI 찍어보자고 하셨어요.", "tests", "MRI 예정"),
        ("10월 2일 시야검사 예약함.", "tests", "10월 2일 시야검사 예약"),
        ("혈압약은 아침에 물로만 먹으라고.", "medication_instructions", "아침에 혈압약"),
        ("해열제는 38도 넘을 때만", "medication_instructions", "38도 넘으면 해열제"),
        ("항생제 5일.", "medication_instructions", "항생제 5일"),
        ("연고 처방받았어요.", "medication_instructions", "연고"),
        ("물리치료 6회 처방.", "medication_instructions", "물리치료 6회"),
        ("혈액검사 했고 결과는 다음에 알려준대요.", "tests", "혈액검사"),
        ("그때 피검사 결과 알려준다고.", "tests", "추후 피검사 결과 안내"),
        ("엑스레이 2주 뒤 다시 찍기로", "tests", "2주 후 엑스레이"),
        ("3주 뒤 재진.", "follow_up", "3주 뒤"),
        ("약은 아직 안 먹어도 된대요.", "medication_instructions", None),
        ("뼈는 괜찮다고 하셨어요.", "findings", None),
    ],
)
def test_rule_matches_reviewed_gold(gen, seg, axis, gold):
    got, cands = pick(gen, seg, axis)
    if gold is None:
        assert got is None, cands
    else:
        assert got is not None and got.replace(" ", "") == gold.replace(" ", ""), cands


def test_typo_and_marker_are_normalized_before_candidates(gen):
    got, cands = pick(gen, "감기레요. ㅋㅋ", "findings")
    assert got == "감기", cands
    cset = gen.generate("감기레요. ㅋㅋ", "findings")
    assert cset.raw == "감기레요. ㅋㅋ" and cset.segment == "감기래요."


def test_canon_is_marked_derived(gen):
    cset = gen.generate("안 좋아지면 바로 오래요.", "follow_up")
    canon = [c for c in cset.candidates if "canon" in c.kinds]
    assert canon and all(c.derived for c in canon)


@pytest.mark.parametrize(
    "value, want",
    [
        # 용어가 머리인 서술문 → 용어만
        ("어금니에 충치가 두 개 있음", "충치"),
        ("식도염은 이전과 비슷함", "식도염"),
        ("열은 바이러스성임", "바이러스성"),
        ("안구건조증이 좀 있음", "안구건조증"),
        ("요로결석 4mm", "요로결석"),
        ("역류성 식도염 소견은 이전과 비슷함", "역류성 식도염"),
        # 앞 수식(크기·좌우)은 남기고, 단계 낱말은 남긴다
        ("작은 근종이 보임", "작은 근종"),
        ("허리 디스크 초기", "허리 디스크 초기"),
        ("오른쪽 발목 염좌", "오른쪽 발목 염좌"),
        # 건드리지 않는 것: 수치 소견, 용어가 머리가 아닌 것, 용어 없는 서술문
        ("혈당이 경계에 가까움", "혈당이 경계에 가까움"),
        ("갑상선 수치가 조금 흔들림", "갑상선 수치가 조금 흔들림"),
        ("감기 뒤 피로가 남은 거 같음", "감기 뒤 피로가 남은 거 같음"),
        ("허리 근육이 많이 뭉침", "허리 근육이 많이 뭉침"),
        # 부정이 있으면 자르지 않는다 — 잘라내면 부정이 사라진다(2026-09-23)
        ("충치는 깊지 않음", "충치는 깊지 않음"),
        ("파열은 아님", "파열은 아님"),
    ],
)
def test_term_only(value, want):
    from medimate.span.canon import term_only
    from medimate.text.lexicon import load_lexicon

    assert term_only(value, load_lexicon()) == want


def test_term_only_is_idempotent_on_eval_gold():
    """확정 시트의 소견 gold는 후처리의 고정점이어야 한다 — gold를 바꾸는 규칙은 규칙이 틀린 것."""
    import json
    from pathlib import Path

    from medimate.span.canon import geo_nominal, term_only
    from medimate.text.lexicon import load_lexicon

    lex = load_lexicon()
    for path in (
        "evals/span_cases.jsonl",
        "evals/span_raw_cases.jsonl",
        "evals/span_unseen_cases.jsonl",
        "evals/span_unseen_r3_cases.jsonl",
        "evals/span_unseen_r4_cases.jsonl",
    ):
        for ln in Path(path).read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            for s in json.loads(ln)["segments"]:
                g = s.get("gold")
                if s["label"] == "findings" and g and g != "NONE":
                    assert term_only(g, lex) == g, (path, g)
                    assert geo_nominal(g) == g, (path, g)


@pytest.mark.parametrize(
    "value,want",
    [
        ("인대가 놀란 거래", "인대가 놀람"),
        ("기침은 기관지가 예민해진 거래", "기침은 기관지가 예민해짐"),
        ("어깨 근육이 뭉친 거", "어깨 근육이 뭉침"),
        # `것 같음`·`상태`처럼 뒤에 다른 말이 오면 그대로
        ("무릎 인대가 늘어난 것 같음", "무릎 인대가 늘어난 것 같음"),
        ("허리 근육이 많이 뭉친 상태", "허리 근육이 많이 뭉친 상태"),
        ("발목은 심하게 접질린 건 아님", "발목은 심하게 접질린 건 아님"),
    ],
)
def test_geo_nominal(value, want):
    from medimate.span.canon import geo_nominal

    assert geo_nominal(value) == want
