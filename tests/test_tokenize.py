"""Kiwi + 어휘집 사용자 사전. kiwipiepy 없으면 건너뛴다(선택 의존성)."""

import pytest

pytest.importorskip("kiwipiepy")

from medimate.text.lexicon import load_lexicon  # noqa: E402
from medimate.text.tokenize import base_kiwi, build_kiwi, needs_user_dict, tokens  # noqa: E402


@pytest.fixture(scope="module")
def kiwi():
    k, _ = build_kiwi(load_lexicon())
    return k


def test_only_split_terms_are_added():
    lex = load_lexicon()
    _, added = build_kiwi(lex)
    base = base_kiwi()
    assert added
    for t in added:
        assert needs_user_dict(base, t), t
    # 기본이 이미 아는 것은 넣지 않는다
    assert all(t.surface != "위염" for t in added)


def test_user_terms_survive_as_one_token(kiwi):
    for s, want in [
        ("위산약 2주치 받고", "위산약"),
        ("프로톤펌프억제제를 처방받았어요", "프로톤펌프억제제"),
        ("역류성식도염이라고 하셨어요", "역류성식도염"),
        ("인공눈물 하루 4번.", "인공눈물"),
    ]:
        # form은 사전형(`역류성 식도염`)일 수 있다. 후보가 쓰는 것은 원문 좌표다
        spans = [s[t.start : t.end] for t in tokens(kiwi, s)]
        assert want in spans, (s, spans)


def test_user_tokens_are_flagged_with_type(kiwi):
    toks = [t for t in tokens(kiwi, "위산약 2주치") if t.form == "위산약"]
    assert toks and toks[0].user and toks[0].tag == "NNG"


def test_token_offsets_point_into_source(kiwi):
    s = "위염이래요. 위산약 2주치 받고"
    for t in tokens(kiwi, s):
        assert s[t.start : t.end] == t.form or t.tag.startswith(("E", "V", "J", "X"))
