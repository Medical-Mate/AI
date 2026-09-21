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
