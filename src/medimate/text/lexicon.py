"""의료 어휘집 로더 — `data/lexicon/*.csv`. 표면형·타입·출처. 온톨로지가 아니다.

역할 둘
1. Kiwi 사용자 사전의 재료 — 분석기가 `위산약`을 `위산/약`으로 쪼개지 않게
2. 후보 span에 타입(finding/medication/test/procedure)을 붙이기 — `Lexicon.match()`

매칭은 공백을 무시한다. `역류성 식도염`이 `역류성식도염`에도 잡힌다. 좌표는 원문 기준으로 돌려준다.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DIR = Path(__file__).resolve().parents[3] / "data" / "lexicon"
TYPES = ("finding", "medication", "test", "procedure")

# 후보 타입 → 카드 축. 어휘집이 카드에 쓰이는 유일한 경로다
TYPE_TO_AXIS = {
    "finding": "findings",
    "medication": "medication_instructions",
    "test": "tests",
    "procedure": "tests",
}


class LexiconError(ValueError):
    pass


@dataclass(frozen=True)
class Term:
    surface: str
    type: str
    source: str
    note: str = ""

    @property
    def compact(self) -> str:
        return self.surface.replace(" ", "")


@dataclass(frozen=True)
class TermMatch:
    term: Term
    start: int  # 원문 좌표
    end: int
    text: str  # 원문에서 잘라낸 그대로 (공백 포함)


class Lexicon:
    def __init__(self, terms: list[Term]):
        self.terms = terms
        self._rx = _build_regex(terms)
        self._by_compact = {t.compact: t for t in terms}

    def __len__(self) -> int:
        return len(self.terms)

    def by_type(self, type_: str) -> list[Term]:
        return [t for t in self.terms if t.type == type_]

    def match(self, text: str) -> list[TermMatch]:
        """원문에서 어휘집 표면형을 찾는다. 겹치면 **긴 것**이 이긴다. 원문 등장 순."""
        if self._rx is None:
            return []
        out: list[TermMatch] = []
        for m in self._rx.finditer(text):
            term = self._by_compact[re.sub(r"\s+", "", m.group(0))]
            out.append(TermMatch(term, m.start(), m.end(), m.group(0)))
        return out


def _spaced(compact: str) -> str:
    """`위산약` → `위\\s*산\\s*약`. 글자 사이 공백을 허용한다."""
    return r"\s*".join(re.escape(ch) for ch in compact)


def _build_regex(terms: list[Term]) -> re.Pattern[str] | None:
    if not terms:
        return None
    # 긴 것이 먼저 — 정규식 교대는 앞에서 맞는 것을 취하므로 순서가 곧 우선순위다
    compacts = sorted({t.compact for t in terms}, key=len, reverse=True)
    alt = "|".join(_spaced(c) for c in compacts)
    # 단어 경계를 두지 않는다. `위염`이 `급성위염` 안에서 잡히는 것은 의도한 것이고,
    # 뒤에 어미·조사가 붙는 한국어라 오른쪽 경계도 두지 않는다
    return re.compile(alt)


def load_lexicon(dir_: Path | str = DEFAULT_DIR) -> Lexicon:
    dir_ = Path(dir_)
    terms: list[Term] = []
    seen: dict[str, str] = {}
    for path in sorted(dir_.glob("*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            need = {"surface", "type", "source"}
            if not need <= set(reader.fieldnames or ()):
                raise LexiconError(f"{path.name}: 컬럼은 surface,type,source(,note)")
            for i, row in enumerate(reader, start=2):
                surface = (row.get("surface") or "").strip()
                type_ = (row.get("type") or "").strip()
                source = (row.get("source") or "").strip()
                if not surface:
                    raise LexiconError(f"{path.name}:{i} surface 비어 있음")
                if type_ not in TYPES:
                    raise LexiconError(f"{path.name}:{i} type={type_!r} — {TYPES} 중 하나")
                if not source:
                    raise LexiconError(f"{path.name}:{i} {surface!r} source 필수")
                if len(surface.replace(" ", "")) < 2:
                    raise LexiconError(f"{path.name}:{i} {surface!r} 한 글자는 넣지 않는다")
                compact = surface.replace(" ", "")
                if compact in seen:
                    raise LexiconError(
                        f"{path.name}:{i} {surface!r} 중복 (먼저 나온 곳 {seen[compact]})"
                    )
                seen[compact] = f"{path.name}:{i}"
                terms.append(Term(surface, type_, source, (row.get("note") or "").strip()))
    return Lexicon(terms)
