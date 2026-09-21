"""선택기 — 후보 중 하나(또는 NONE)를 고른다. 문자열을 만들지 않는다.

- `rule`  호출 0. 축별 우선순위 하나. 기준선이자 폴백
- `nova` / `jev` / `cascade`  실호출. 게이트(E0 CR ≥ 95%)를 넘은 뒤 붙인다
  (docs/candidate-selection.md §8)

선택기는 후보 ID를 돌려준다. 값으로 바꾸는 것은 `resolve()` 한 곳이고, 그 값은 후보의 text다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from medimate.span.candidates import Candidate, CandidateSet

NONE = "NONE"


@dataclass(frozen=True)
class Selection:
    candidate_id: str  # 후보 ID 또는 NONE
    confidence: float | None = None  # 선택기가 주면
    selector: str = "rule"
    note: str = ""


class Selector(Protocol):
    name: str

    def select(self, cset: CandidateSet, axis: str, context: dict | None = None) -> Selection: ...


def resolve(cset: CandidateSet, sel: Selection) -> str | None:
    """선택 → 값. NONE이면 None(호출자가 폴백을 정한다)."""
    if sel.candidate_id == NONE:
        return None
    for c in cset.candidates:
        if c.id == sel.candidate_id:
            return c.text
    return None


class RuleSelector:
    """축별 우선순위. 어느 규칙에도 안 걸리면 NONE — 억지로 고르지 않는다."""

    name = "rule"

    def select(self, cset: CandidateSet, axis: str, context: dict | None = None) -> Selection:
        subs = [c for c in cset.candidates if not c.derived and "whole" not in c.kinds]
        pick: Candidate | None = None
        if axis == "findings":
            pick = _longest(subs, "lexicon:finding") or _longest(subs, "chunk")
        elif axis == "medication_instructions":
            # 약 이름에서 chunk 끝까지(`위산약 2주치`) > 약 이름 > 가장 긴 chunk
            pick = (
                _longest(subs, "tail:medication")
                or _longest(subs, "lexicon:medication")
                or _longest(subs, "phrase")
                or _longest(subs, "chunk")
            )
        elif axis == "tests":
            pick = (
                _longest(subs, "lexicon:test")
                or _longest(subs, "lexicon:procedure")
                or _longest(subs, "chunk")
            )
        elif axis == "follow_up":
            # 기간 + 후/뒤 > 날짜 > 기간
            durs = [c for c in subs if "duration" in c.kinds]
            pick = next(
                (c for c in durs if c.text.rstrip().endswith(("후", "뒤"))), None
            ) or _longest(durs, "duration")
        return Selection(pick.id if pick else NONE, selector=self.name)


def _longest(cands: list[Candidate], kind: str) -> Candidate | None:
    hits = [c for c in cands if kind in c.kinds]
    return max(hits, key=lambda c: len(c.compact)) if hits else None
