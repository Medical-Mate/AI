"""값 검증기(span/verify.py) — 통과해야 하는 값 모양과 걸려야 하는 값. 호출 0."""

import pytest

pytest.importorskip("kiwipiepy")

from medimate.span.verify import verify_value  # noqa: E402
from medimate.text.lexicon import load_lexicon  # noqa: E402

LEX = load_lexicon()


@pytest.mark.parametrize(
    "value,segment,axis",
    [
        # 재배열·조사 떼기·-음 명사형
        (
            "갑상선 수치가 조금 흔들림",
            "오늘 혈액검사에서 갑상선 수치가 조금 흔들렸지만 급한 문제는 아니라고 하셨어요.",
            "findings",
        ),
        ("아침 공복에 약", "약은 아침 공복에 먹고", "medication_instructions"),
        ("6주 후 검사", "6주 뒤 다시 검사하기로 했어요.", "tests"),
        ("엑스레이", "엑스레이를 찍었고 결과는 다음 진료 때 설명해 주신다고 했음.", "tests"),
        (
            "열이 나거나 붓기가 커지면 재방문 필요",
            "열이 나거나 붓기가 커지면 바로 오라고 했어요.",
            "follow_up",
        ),
        ("이번 주까지 소독", "이번 주까지만 소독하고", "medication_instructions"),
        ("추후 피검사 결과 안내", "피검사 결과는 다음에 알려 준다고 했어요.", "tests"),
        ("악화 시 재방문 필요", "심해지면 다시 오라고 했어요.", "follow_up"),
        ("석 달 후 피검사", "석달뒤피검사다시하기", "tests"),
        ("물이 조금 찼음", "관절초음파에서물이조금찼다함", "findings"),
        ("검사 결과 이상 없음", "보건소검사결과이상없다함", "tests"),
        ("사진은 괜찮게 나왔음", "흉부외과에서 찍은 사진은 괜찬게 나왓데요.", "tests"),
        (
            "속쓰림 안 나으면 월요일 재방문 필요",
            "속쓰림 안 낫으면 월욜에 다시 오래요.",
            "follow_up",
        ),
        ("다음 주", "담주에 또 오재.", "follow_up"),
        ("위염 초기", "위염 초기라고 하셨어요", "findings"),
        # 부정 — 부정 절에서 가져왔으면 부정까지, 부정 없는 절에서 가져왔으면 그대로(2026-09-23)
        ("코 안은 많이 부었음", "코 안은 많이 부었지만 축농증은 아니라고 하셨음.", "findings"),
        ("귀지 때문", "중이염은 아니고 귀지 때문이라고.", "findings"),
        ("충치는 깊지 않음", "충치 는 깊지 않 다고 했고", "findings"),
        ("작은 근종", "작은 근종이 보이지만 당장 치료할 정도는 아니라고 하셨어요.", "findings"),
        ("숨차면 재방문 필요", "숨차면 참지 말고 오라구.", "follow_up"),
        ("해열제", "해열제 먹으라는데 몇번인지 ㅁㄹ 기억안남", "medication_instructions"),
        ("붓기 안 빠지면 재방문 필요", "붓기안빠지면오라고함", "follow_up"),
    ],
)
def test_passes(value, segment, axis):
    v = verify_value(value, segment, axis, LEX)
    assert v.ok, v.reasons


@pytest.mark.parametrize(
    "value,segment,axis,reason",
    [
        ("인대 손상", "무릎 인대가 늘어난 것 같다고 했어요", "findings", "unsourced"),
        ("항생제 7일", "항생제 5일 처방받았어요", "medication_instructions", "digits"),
        ("안 퍼짐", "ㅇㅇ 퍼졌다고 했음", "findings", "negation_added"),
        ("위염", "속이 쓰리다고 했어요", "findings", "unsourced"),
        ("피견", "피곤하다고 했어요", "findings", "unsourced"),
        ("혈압이 높음", "혈압은 정상이래요", "findings", "unsourced"),
        # 사라진 부정 — 뜻이 뒤집힌다
        (
            "많이 부었지만 축농증",
            "코 안은 많이 부었지만 축농증은 아니라고 하셨음.",
            "findings",
            "negation_dropped",
        ),
        ("중이염", "중이염은 아니고 귀지 때문이라고.", "findings", "negation_dropped"),
        ("충치 깊음", "충치 는 깊지 않 다고 했고", "findings", "negation_dropped"),
        ("붓기 빠지면 재방문 필요", "붓기 안 빠지면 다시 오래요", "follow_up", "negation_dropped"),
        ("재방문 필요", "다시 오라는 말은 없었음", "follow_up", "negation_dropped"),
    ],
)
def test_rejects(value, segment, axis, reason):
    v = verify_value(value, segment, axis, LEX)
    assert not v.ok
    assert any(r.startswith(reason) for r in v.reasons), v.reasons


def test_none_passes():
    assert verify_value(None, "괜찮대요", "findings").ok
    assert verify_value("", "괜찮대요", "findings").ok


# 확정 gold 중 검증기가 못 통과시키는 것 — 늘면 검증기가 과하게 잡는 것이다.
# 중이염 2: 뜻이 뒤집힌 gold(부정 정의 2026-09-23), 고칠 대상.
# 치·빨갛: Kiwi 분절·오타 어간(기존 한계)
_KNOWN_GOLD_REJECTS = {
    ("PN03", "중이염"),
    ("RW12", "중이염"),
    ("SB01", "위산약 2주치"),
    ("SB02", "위산약 2주치"),
    ("SB01-E1", "위산약 2주치"),
    ("SB01-E2", "위산약 2주치"),
    ("SB01-E6", "위산약 2주치"),
    ("R413", "귀 안쪽이 빨갰음"),
}


def test_eval_gold_passes_verifier():
    import json
    from pathlib import Path

    axes = {"findings", "tests", "medication_instructions", "follow_up"}
    rejected = set()
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
            c = json.loads(ln)
            for s in c["segments"]:
                g = s.get("gold")
                if s["label"] in axes and g and g != "NONE":
                    if not verify_value(g, s["text"], s["label"], LEX).ok:
                        rejected.add((c["id"], g))
    assert rejected == _KNOWN_GOLD_REJECTS


def _offline_generator():
    """호출 없이 check()만 쓰는 생성기 — 재채점과 같은 방식."""
    from medimate.span.candidates import CandidateGenerator
    from medimate.span.generate import ValueGenerator
    from medimate.span.select import RuleSelector

    gen = ValueGenerator.__new__(ValueGenerator)
    gen.lexicon, gen.gen, gen.rule, gen.model_id = (
        LEX,
        CandidateGenerator(),
        RuleSelector(),
        "offline",
    )
    return gen


def test_fallback_keeps_negation():
    """모델 값을 부정 때문에 버리면 rule 폴백도 부정을 지켜야 한다.

    r4 R402: 폴백이 `축농증`을 냈다.
    """
    gen = _offline_generator()
    seg = "코 안은 많이 부었지만 축농증은 아니라고 하셨음."
    g = gen.check("많이 부었지만 축농증", seg, "findings")
    assert g.source == "fallback"
    assert g.value is None or "축농증" not in g.value


def test_changed_negation_form_is_not_added():
    """`아니` → `않`은 부정을 바꾼 게 아니다. 개수로 센다."""
    v = verify_value("발목은 심하게 접질리지 않음", "발목은심하게접질린건아니래", "findings", LEX)
    assert "negation_added" not in v.reasons
