"""어휘집 로더 — 검증 규칙과 공백 무시 최장 일치."""

from pathlib import Path

import pytest

from medimate.text.lexicon import TYPES, LexiconError, load_lexicon


def _write(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text("surface,type,source,note\n" + body, encoding="utf-8")
    return p


def test_repo_lexicon_loads_and_has_all_types():
    lex = load_lexicon()
    assert len(lex) > 50
    assert {t.type for t in lex.terms} == set(TYPES)
    for t in lex.terms:
        assert t.source, t


def test_match_ignores_spaces_and_prefers_longest(tmp_path):
    _write(tmp_path, "a.csv", "위염,finding,t,\n역류성 식도염,finding,t,\n식도염,finding,t,\n")
    lex = load_lexicon(tmp_path)
    got = lex.match("역류성식도염이라고 하셨어요. 위염도요")
    assert [(m.text, m.term.type) for m in got] == [
        ("역류성식도염", "finding"),
        ("위염", "finding"),
    ]
    assert got[0].start == 0 and got[0].end == 6
    spaced = lex.match("역류성 식도염 같다고")
    assert spaced[0].text == "역류성 식도염" and spaced[0].term.surface == "역류성 식도염"


def test_match_returns_source_coordinates(tmp_path):
    _write(tmp_path, "a.csv", "위산약,medication,t,\n")
    lex = load_lexicon(tmp_path)
    s = "위염이래요. 위산약 2주치 받고"
    (m,) = lex.match(s)
    assert s[m.start : m.end] == "위산약"


@pytest.mark.parametrize(
    "body, msg",
    [
        ("위염,disease,t,\n", "type="),
        ("위염,finding,,\n", "source 필수"),
        ("약,medication,t,\n", "한 글자"),
        (",finding,t,\n", "비어 있음"),
        ("위염,finding,t,\n위 염,finding,t,\n", "중복"),
    ],
)
def test_validation_errors(tmp_path, body, msg):
    _write(tmp_path, "a.csv", body)
    with pytest.raises(LexiconError, match=msg):
        load_lexicon(tmp_path)


def test_missing_columns(tmp_path):
    (tmp_path / "a.csv").write_text("word,kind\n위염,finding\n", encoding="utf-8")
    with pytest.raises(LexiconError, match="컬럼"):
        load_lexicon(tmp_path)
