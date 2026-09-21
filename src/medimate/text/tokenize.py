"""Kiwi 형태소 분석 — 어휘집 사용자 사전을 얹은 분석기. 선택 의존성(`uv sync --group nlp`).

사용자 사전에는 **기본 분석이 쪼개는 용어만** 넣는다(`needs_user_dict`). `위염`처럼 이미 한 토큰인
것은 넣지 않는다 — 기본 사전이 아는 것을 덮어쓰면 다른 문장에서 무엇이 바뀌는지 알 수 없다.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from medimate.text.lexicon import Lexicon, Term, load_lexicon

# 명사류 태그. 후보 명사구는 이 태그의 연속으로 만든다
NOMINAL_TAGS = frozenset({"NNG", "NNP", "NNB", "NR", "SN", "SL", "XSN", "XPN", "MM"})
USER_TAG = "NNG"
USER_SCORE = 5.0  # 기본 조합(위산/약)보다 우선하게. 0이면 기본이 이기는 경우가 있다


@dataclass(frozen=True)
class Tok:
    form: str
    tag: str
    start: int  # 원문 좌표
    end: int
    user: bool = False  # 사용자 사전에서 왔는가


def _import_kiwi():
    try:
        from kiwipiepy import Kiwi
    except ImportError as e:  # pragma: no cover
        raise ImportError("kiwipiepy가 없다. `uv sync --group nlp`") from e
    return Kiwi


@lru_cache(maxsize=1)
def base_kiwi():
    """사용자 사전 없는 기본 분석기. 비교 기준."""
    return _import_kiwi()()


def tokens(kiwi, text: str) -> list[Tok]:
    return [
        Tok(t.form, t.tag, t.start, t.end, user=t.user_value is not None)
        for t in kiwi.tokenize(text)
    ]


def needs_user_dict(kiwi, term: Term) -> bool:
    """기본 분석이 이 표면형을 한 토큰으로 못 보면 True. 공백 있는 표면형은 공백을 뺀 형으로."""
    toks = tokens(kiwi, term.compact)
    return not (len(toks) == 1 and (toks[0].tag.startswith("NN") or toks[0].tag == "SL"))


def build_kiwi(lexicon: Lexicon | None = None, *, only_split: bool = True):
    """어휘집을 사용자 사전으로 얹은 분석기. `only_split=True`면 기본이 쪼개는 용어만 넣는다.

    돌려주는 것: (kiwi, 넣은 용어 목록).
    """
    lexicon = lexicon or load_lexicon()
    base = base_kiwi()
    kiwi = _import_kiwi()()
    added: list[Term] = []
    for term in lexicon.terms:
        if only_split and not needs_user_dict(base, term):
            continue
        # 표면형을 **그대로**(공백 포함) 넣는다. Kiwi 0.17+는 공백 있는 단어를 받고, 입력이
        # `역류성 식도염`이든 `역류성식도염`이든 같은 형태소로 잇는다(2026-09-21 확인). 돌아오는
        # `form`은 사전형이라 원문과 다를 수 있다 — 후보는 form이 아니라 start/end로 원문을 자른다
        ok = kiwi.add_user_word(term.surface, USER_TAG, USER_SCORE, user_value={"type": term.type})
        if ok:
            added.append(term)
    return kiwi, added


def space(kiwi, text: str) -> str:
    """띄어쓰기 복원. 원문을 대체하지 않는다 — 보조 입력."""
    return kiwi.space(text)
