"""후보 생성 + rule 선택. kiwipiepy 없으면 건너뛴다."""

import pytest

pytest.importorskip("kiwipiepy")

from medimate.span.candidates import CandidateGenerator  # noqa: E402
from medimate.span.select import NONE, RuleSelector, resolve  # noqa: E402


@pytest.fixture(scope="module")
def gen():
    return CandidateGenerator()


def texts(cset):
    return [c.text for c in cset.candidates]


def test_boundary_case_candidates_contain_gold(gen):
    for seg, axis, gold in [
        ("위염이래요.", "findings", "위염"),
        ("위산약 2주치 받고,", "medication_instructions", "위산약 2주치"),
        ("3주 후에 재방문 하래요", "follow_up", "3주 후"),
    ]:
        cset = gen.generate(seg, axis)
        assert cset.contains(gold), (seg, texts(cset))


def test_no_space_input_still_has_gold(gen):
    cset = gen.generate("위산약2주치받고", "medication_instructions")
    assert cset.contains("위산약 2주치"), texts(cset)
    cset = gen.generate("3주후에재방문하래요", "follow_up")
    assert cset.contains("3주 후"), texts(cset)


def test_every_substring_candidate_is_in_segment(gen):
    seg = "약은 2주분이고 커피랑 매운 거 줄이래요."
    cset = gen.generate(seg, "medication_instructions")
    for c in cset.candidates:
        if not c.derived:
            assert seg[c.start : c.end] == c.text
    whole = [c for c in cset.candidates if "whole" in c.kinds]
    assert len(whole) == 1 and whole[0].text == "약은 2주분이고 커피랑 매운 거 줄이래요"


def test_ids_follow_source_order(gen):
    cset = gen.generate("위염이래요. 위산약 2주치 받고", None)
    ids = [c.id for c in cset.candidates]
    assert ids == sorted(ids)
    starts = [c.start for c in cset.candidates if not c.derived and "whole" not in c.kinds]
    assert starts == sorted(starts)


def test_lexicon_type_is_carried(gen):
    cset = gen.generate("프로톤펌프억제제를 처방받았어요", "medication_instructions")
    c = cset.contains("프로톤펌프억제제")
    assert c and "lexicon:medication" in c.kinds


def test_rule_selector_boundary_case(gen):
    rule = RuleSelector()
    got = {}
    for seg, axis in [
        ("위염이래요.", "findings"),
        ("위산약 2주치 받고,", "medication_instructions"),
        ("3주 후에 재방문 하래요", "follow_up"),
    ]:
        cset = gen.generate(seg, axis)
        got[axis] = resolve(cset, rule.select(cset, axis))
    assert got == {
        "findings": "위염",
        "medication_instructions": "위산약 2주치",
        "follow_up": "3주 후",
    }


def test_rule_selector_abstains(gen):
    cset = gen.generate("아무 말도 안 하시고 약만 주셨어요.", "follow_up")
    assert RuleSelector().select(cset, "follow_up").candidate_id == NONE
