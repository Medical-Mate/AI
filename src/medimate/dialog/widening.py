"""넓히기 병기 — 환자가 옮긴 의사의 말에 부위를 붙인다 (docs/ai-design.md §3, 이슈 #9).

"반월판 손상이래요" → 반월판(안쪽반달연골, 연골) · 부위: 무릎.
설명이 아니라 **위치 표시**다. 온톨로지 구조 층의 한국어 이름과 사전 매칭만 하고,
LLM은 관여하지 않는다. 매칭이 안 되면 아무것도 붙이지 않는다 — 억지로 붙이지 않는다.

부위 대조: 진료 전 카드의 부위(앵커)와 여기서 닿은 앵커가 같은지 **표시**만 한다.
"같은 부위" / "다른 부위가 언급됨". 어느 쪽이 맞다는 판정은 없다.
"""

from __future__ import annotations

import re

from medimate.ontology import Node, Ontology
from medimate.schema.card import FieldStatus
from medimate.schema.postvisit import PostAxis, PostVisitCard, WideningNote

_MIN_ALIAS_LEN = 2  # 한 글자 별칭("뼈")은 오탐이라 쓰지 않는다


def _aliases(node: Node) -> list[str]:
    """노드 한국어 이름을 매칭용 별칭들로. "안쪽반달연골(내측반월판)" → 둘 다, 괄호 안 항목 분리."""
    name = node.name_ko or ""
    if not name:
        return []
    parts = re.split(r"[()（）·,/]", name)
    out = []
    for p in parts:
        p = p.strip()
        if len(p) >= _MIN_ALIAS_LEN and p not in out:
            out.append(p)
    return out


def _structure_aliases(onto: Ontology) -> list[tuple[str, Node]]:
    """(별칭, 노드). 긴 별칭이 먼저 매칭되게 정렬 — "앞십자인대"가 "십자인대"보다 먼저."""
    pairs = [(a, n) for n in onto.nodes.values() if n.kind == "structure" for a in _aliases(n)]
    pairs.sort(key=lambda p: -len(p[0]))
    return pairs


def find_terms(onto: Ontology, text: str) -> list[tuple[str, Node]]:
    """텍스트에 나온 구조 용어. 같은 구간은 한 번만(긴 별칭 우선). 노드 중복 없음."""
    hits: list[tuple[str, Node]] = []
    taken: list[tuple[int, int]] = []
    seen: set[str] = set()
    for alias, node in _structure_aliases(onto):
        for m in re.finditer(re.escape(alias), text):
            span = (m.start(), m.end())
            if any(s < span[1] and span[0] < e for s, e in taken):
                continue  # 이미 더 긴 별칭이 차지한 구간
            taken.append(span)  # 모든 출현을 차지해 둔다 — 짧은 별칭이 나머지 출현을 가져가지 않게
            if node.id not in seen:
                seen.add(node.id)
                hits.append((alias, node))
    return hits


def widen_card(card: PostVisitCard, onto: Ontology) -> list[WideningNote]:
    """heard_diagnosis의 값·근거에서 구조 용어를 찾아 부위를 병기한다.

    카드를 갱신하고 결과를 돌려준다.
    """
    entry = card.axes[PostAxis.HEARD_DIAGNOSIS]
    if entry.status != FieldStatus.FILLED:
        card.widening = []
        return []
    text = " ".join([entry.value or "", *entry.evidence])
    notes: list[WideningNote] = []
    for term, node in find_terms(onto, text):
        w = onto.widen(node.id)
        label = node.display_name
        if node.structure_type and node.structure_type not in label:
            label = (
                f"{label}({node.structure_type})"  # 구조 유형 병기 — 환자는 이걸 구분 못 한다(§3)
            )
        notes.append(
            WideningNote(
                term=term,
                node_id=node.id,
                node_label=label,
                anchor_id=w.anchor.id if w.anchor else None,
                anchor_label=w.anchor.display_name if w.anchor else None,
                ontology_snapshot=onto.snapshot_id,
            )
        )
    card.widening = notes
    if card.provenance is not None:
        card.provenance.ontology_snapshot = onto.snapshot_id
    return notes


def compare_sites(
    onto: Ontology, previsit_anchor_id: str | None, notes: list[WideningNote]
) -> str | None:
    """진료 전 부위(앵커)와 들은 용어의 부위를 대조. "same" / "different" / None(비교 불가).

    판정이 아니다. 다른 부위가 언급됐다는 사실만 표시한다.
    """
    if not previsit_anchor_id or not notes:
        return None
    anchors = {n.anchor_id for n in notes if n.anchor_id}
    if not anchors:
        return None
    return "same" if anchors == {previsit_anchor_id} else "different"
