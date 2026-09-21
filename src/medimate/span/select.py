"""선택기 — 후보 중 하나(또는 NONE)를 고른다. 문자열을 만들지 않는다.

- `rule`  호출 0. 축별 우선순위 하나. 기준선이자 폴백
- `nova` / `jev` / `cascade`  실호출. 게이트(E0 CR ≥ 95%)를 넘은 뒤 붙인다
  (docs/candidate-selection.md §8)

선택기는 후보 ID를 돌려준다. 값으로 바꾸는 것은 `resolve()` 한 곳이고, 그 값은 후보의 text다.
"""

from __future__ import annotations

import re
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
    """축별 우선순위. 어느 규칙에도 안 걸리면 NONE — 억지로 고르지 않는다.

    규칙은 gold 검토(2026-09-21)에서 읽은 카드 값 모양을 따른다(canon.py 머리말). 기준선이자 폴백.
    """

    name = "rule"

    def select(self, cset: CandidateSet, axis: str, context: dict | None = None) -> Selection:
        from medimate.span.canon import reassurance

        seg = cset.segment
        subs = [c for c in cset.candidates if not c.derived]
        canon = [c for c in cset.candidates if "canon" in c.kinds]
        pick: Candidate | None = None

        if axis == "findings":
            if reassurance(seg):
                return Selection(NONE, selector=self.name, note="안심·부정 소견")
            # 수치·변화 서술(높음·늘어났음·찼음·아님)이면 서술문 그대로. 병명·용어가 있으면 용어만
            stmt = [c for c in canon if _MEASURE.search(c.text)]
            pick = (
                (stmt[0] if stmt else None)
                or _earliest_containing(subs, "lexicon:finding", ("chunk", "tail:finding"))
                or _longest(subs, "lexicon:finding")
                or _longest(subs, "chunk")
            )
            # 칸 이름과 겹치는 꼬리(`소견`)는 뗀다 — 그 형이 후보에 있으면
            if pick and pick.text.endswith("소견"):
                shorter = cset.contains(pick.text[: -len("소견")])
                pick = shorter or pick

        elif axis == "medication_instructions":
            if _med_negated(seg):
                return Selection(NONE, selector=self.name, note="약 없음·보류")
            has_med = any("lexicon:medication" in c.kinds for c in subs)
            has_num = any("duration" in c.kinds for c in subs)
            if not has_med and not has_num and "약" not in seg:
                return Selection(NONE, selector=self.name, note="약·용법 없음")
            # 약 이름과 용법이 한 chunk에 이어져 있으면 그것(항생제 5일). 조건·시점이 앞에 있으면
            # 템플릿(아침에 혈압약). 그 외 약 이름
            cond_canon = [c for c in canon if not _starts_with_med(c.text, subs)]
            # 약 낱말(`약`)과 용법이 한 chunk에 있으면 그것(`2주분 약`, `2주 약`)
            yak = [
                c
                for c in subs
                if "chunk" in c.kinds
                and re.search(r"(?<![가-힣])약(?![가-힣])", c.text)
                and _contains_kind(c, subs, ("duration",))
            ]
            pick = (
                _longest_containing(
                    subs, "lexicon:medication", ("chunk", "tail:medication"), prefer_not_whole=True
                )
                or (min(yak, key=lambda c: len(c.compact)) if yak else None)
                or (cond_canon[0] if cond_canon else None)
                or (canon[0] if canon else None)
                or _longest(subs, "lexicon:medication")
                or _longest(subs, "chunk")
            )

        elif axis == "tests":
            if _result_statement(seg):
                pick = next((c for c in subs if "whole" in c.kinds), None)
            # 시점으로 시작하는 chunk가 검사를 품고 있으면 그것(`10월 2일 시야검사 예약`)
            lead = [
                c
                for c in subs
                if "chunk" in c.kinds
                and any(k in ("lexicon:test", "lexicon:procedure") for k in c.kinds) is False
                and _starts_with_time(c, subs)
                and _contains_kind(c, subs, ("lexicon:test", "lexicon:procedure"))
            ]
            pick = (
                pick
                or (max(lead, key=lambda c: len(c.compact)) if lead else None)
                or (canon[0] if canon else None)
                or _longest(subs, "lexicon:test")
                or _longest(subs, "lexicon:procedure")
                or _longest(subs, "chunk")
            )

        elif axis == "follow_up":
            durs = [c for c in subs if "duration" in c.kinds]
            cond = any(_has_condition(c.text) for c in canon)
            if canon and (cond or not durs):
                pick = canon[0]
            else:
                pick = next(
                    (c for c in durs if c.text.rstrip().endswith(("후", "뒤"))), None
                ) or _longest(durs, "duration")
        return Selection(pick.id if pick else NONE, selector=self.name)


_MEASURE = re.compile(
    r"(?:높음|낮음|늘어났음|찼음|부었음|뭉침|아님|커졌음|작아졌음|올랐음|떨어졌음)$"
)


def _has_condition(s: str) -> bool:
    # `~면 재방문`·`악화 시 재방문`. `다시`의 시는 조건이 아니다
    return bool(re.search(r"[가-힣]면\s|\s시\s", s)) or s.strip() == "재방문 필요"


def _starts_with_med(text: str, subs: list[Candidate]) -> bool:
    meds = [c.text for c in subs if "lexicon:medication" in c.kinds] + ["약"]
    return any(text.startswith(m) for m in meds)


def _starts_with_time(c: Candidate, subs: list[Candidate]) -> bool:
    return any(d.start == c.start and d.end < c.end for d in subs if "duration" in d.kinds)


def _contains_kind(c: Candidate, subs: list[Candidate], kinds: tuple[str, ...]) -> bool:
    return any(
        i.start >= c.start and i.end <= c.end and any(k in i.kinds for k in kinds)
        for i in subs
        if i is not c
    )


def _med_negated(seg: str) -> bool:
    return bool(
        re.search(r"약(?:은|만|을)?\s*(?:안|아직|아니|없)|약\s*먹을\s*정도는\s*아니|약만\s*주", seg)
    )


def _result_statement(seg: str) -> bool:
    return bool(re.search(r"이상\s*없|정상", seg))


def _longest_containing(
    cands: list[Candidate],
    inner_kind: str,
    outer_kinds: tuple[str, ...],
    *,
    prefer_not_whole: bool = False,
) -> Candidate | None:
    """inner_kind 후보를 **품는** outer_kinds 후보 중 가장 긴 것 (발목 염좌 → 오른쪽 발목 염좌).

    `prefer_not_whole`: 조각 전체가 아닌 것이 있으면 그것을 먼저(`수면제 대신 멜라토닌 2주` 말고
    `멜라토닌 2주`). 조각 전체밖에 없으면 그것.
    """
    inners = [c for c in cands if inner_kind in c.kinds]
    outers = [
        c
        for c in cands
        if any(k in c.kinds for k in outer_kinds)
        and any(i.start >= c.start and i.end <= c.end for i in inners)
    ]
    if prefer_not_whole:
        inner_only = [c for c in outers if "whole" not in c.kinds and c not in inners]
        if inner_only:
            return max(inner_only, key=lambda c: len(c.compact))
        rest = [c for c in outers if c not in inners]
        if rest:
            return max(rest, key=lambda c: len(c.compact))
        return None
    return max(outers, key=lambda c: len(c.compact)) if outers else None


def _earliest_containing(
    cands: list[Candidate], inner_kind: str, outer_kinds: tuple[str, ...]
) -> Candidate | None:
    """가장 **먼저 나오는** inner를 품는 후보 중 긴 것 — `중이염은 아니고 귀지 때문`은 중이염."""
    inners = sorted((c for c in cands if inner_kind in c.kinds), key=lambda c: c.start)
    if not inners:
        return None
    first = inners[0]
    durs = [c for c in cands if "duration" in c.kinds]
    outers = [
        c
        for c in cands
        if any(k in c.kinds for k in outer_kinds)
        and first.start >= c.start
        and first.end <= c.end
        # 수치(`4mm`)까지 늘리지 않는다 — `요로결석 4mm`는 요로결석
        and not any(d.start >= c.start and d.end <= c.end for d in durs)
    ]
    return max(outers, key=lambda c: len(c.compact)) if outers else first


def _longest(cands: list[Candidate], kind: str) -> Candidate | None:
    hits = [c for c in cands if kind in c.kinds]
    return max(hits, key=lambda c: len(c.compact)) if hits else None
