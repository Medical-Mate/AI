"""부위 온톨로지 로더와 그래프 연산.

데이터는 `data/ontology/nodes.csv`, `edges.csv` (UBERON 추출 + 수동 보강, 40노드/39엣지).

방향 (docs/ai-design.md §3 「좁히지 않고 넓힌다」):
- 이 모듈은 차트 용어 → 환자가 아는 부위(앵커)로 **올리기만** 한다.
- 환자 발화 → 세부 구조로 내려가는 연산(좁히기)은 감별이므로 만들지 않는다.
  그래서 자손 조회 API를 두지 않는다.

관계 처리:
- `part_of`와 `is_a` 둘 다 "상위"로 탄다. ACL은 is_a로 십자인대 → 무릎관절 인대까지 가고
  거기서 part_of로 무릎관절 → 무릎에 닿는다. 한 관계만 타면 앵커에 못 닿는다.
- 앵커(`is_anchor`)는 디자이너 인체도 부위 단위와 같아야 한다. 확정 전이라 무릎·어깨만 임시.

등급(`tier`) — 환자에게 보이는 3등급. 그래프 깊이가 아니다:
- 1 상부: 다부위 정리용 묶음(상지·하지·몸통·머리·목). LCA가 여기서 멈춘다. 부모 없음
- 2 앵커: 환자가 말하는 단위(무릎·어깨). 넓히기가 여기서 멈춘다. 상부 정확히 하나에 닿는다
- 3 세부: 앵커 정확히 하나에 닿는다. 두 종류(`kind`)가 있다
  - structure: 차트에 적히는 해부 구조. 차트 용어 → 앵커로 넓힐 때 쓴다. 깊이는 자유(힘줄⊂근육 유지)
  - surface: 환자가 인체도에서 누르는 표면 구역(어깨 앞, 목 안, 아랫배). 진료 전 카드 SITE 축이 쓴다

두 층은 앵커에서 만난다. 그리고 `located_in`(structure → surface, 출처 필수)로만 이어진다.
- "이 구역 아래에 무엇이 있나"는 해부학 사실이라 한다. `structures_under()`는 전체를 이름순으로 준다
- "이 구역이 아프면 무엇이 원인인가"는 감별이라 하지 않는다. 고르거나 순위 매기는 API가 없다
- 출처가 빈 located_in은 조회에서 뺀다. 자문 회신 전 임시 채움을 구조로 막는다

진료과 안내 (docs/decisions/2026-09-04-department-guidance.md):
- `departments`는 부위 노드(구역 우선, 없으면 앵커)에 적힌 값이다. 로더는 **그대로 노출만** 한다.
  고르지 않고, 정렬하지 않고, 증상 축을 보지 않는다. 그 셋 중 하나라도 하면 감별이다.
- "구역에 값이 없으면 앵커 값"은 소비자(카드 export)가 적용한다. 여기에는 그 규칙조차 두지 않는다.
- 우리 콘텐츠다. 인용이 아니다. `departments_source`가 그 사실을 값마다 달고 다닌다.

검증:
- partonomy는 DAG여야 한다. 손보강 노드가 들어오면 사이클이 생길 수 있어 로드 시 검출한다.
- 엣지가 없는 노드를 가리키면 실패. 조용히 넘어가지 않는다.
"""

from __future__ import annotations

import csv
import hashlib
import re
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path

from medimate.ontology.korean_keys import (
    english_keys_to_hangul,
    looks_like_korean_typed_in_english,
)

RELATIONS = frozenset({"part_of", "is_a"})  # 상위로 타는 관계
ASSOC_RELATIONS = frozenset({"located_in"})  # 층 사이 연결. 조상 계산에 쓰지 않는다
KINDS = frozenset({"region", "anchor", "structure", "surface"})
LATERALITIES = frozenset({"none", "left_right"})
VIEWS = frozenset({"front", "back", "none"})  # 인체도 앞면/뒷면/화면에 없음(사이드 탭·구조·상부)
DEFAULT_DIR = Path(__file__).resolve().parents[3] / "data" / "ontology"


class OntologyError(ValueError):
    """데이터 파일이 규칙을 어긴다. 메시지에 문제 목록이 전부 들어간다."""


class UnknownNodeError(KeyError):
    """온톨로지에 없는 노드 ID."""


class AmbiguousAnchorError(LookupError):
    """같은 거리에 앵커가 둘 이상. 데이터가 결정을 못 내리는 상태라 호출자가 골라야 한다."""


@dataclass(frozen=True, slots=True)
class Node:
    id: str
    name_en: str
    name_ko: str  # 비어 있을 수 있다 (해부학용어 대조 전)
    region: str  # 임시 그룹 라벨(무릎·어깨). 앵커 판정에는 쓰지 않는다
    structure_type: str  # 인대·힘줄·뼈·연골… 환자는 이걸 구분 못 하므로 설명에 병기한다
    sctid: str
    fma: str
    definition_en: str
    source: str
    is_anchor: bool
    tier: int  # 1 상부 / 2 앵커 / 3 세부
    kind: str  # region / anchor / structure / surface
    laterality: str  # none / left_right — 좌·우를 물을 수 있는 노드인가
    view: str = "none"  # front / back / none — 인체도 어느 면에 그려지는가. 앵커·구역만 의미 있다
    departments: tuple[str, ...] = ()  # 진료과 안내. CSV 순서 그대로, 의미 있는 순서 아님
    departments_source: str = ""  # "팀 결정 …, 자문 확인 전". 인용 아님을 값마다 명시
    aliases: tuple[
        str, ...
    ] = ()  # 환자 표현·한자어·구어 유의어(aliases.csv). 폼 검색용. 증상·병명은 넣지 않는다

    @property
    def display_name(self) -> str:
        """환자에게 보이는 이름. 한국어가 없으면 영어로 낸다. 빈 문자열을 만들지 않는다."""
        return self.name_ko or self.name_en


@dataclass(frozen=True, slots=True)
class Widening:
    """넓히기 결과 — 대체가 아니라 병기 (§3). 원문 용어 + 앵커 부위 + 구조 유형을 같이 낸다."""

    term: Node
    anchor: Node | None  # 올라가서 닿은 앵커. 없으면 None (앵커 밖의 노드)
    region: Node | None  # 앵커 위의 상부(등급 1). 앵커가 없으면 None
    path: tuple[str, ...]  # term → … → anchor. 감사·디버깅용


@dataclass
class Ontology:
    nodes: dict[str, Node]
    parents: dict[str, list[tuple[str, str]]]  # child → [(relation, parent)]
    snapshot_id: str  # CSV 바이트 해시. Provenance.ontology_snapshot에 박는 값
    located_in: list[tuple[str, str, str]] = field(
        default_factory=list
    )  # (structure, surface, source)
    _children: dict[str, list[str]] = field(
        default_factory=dict, repr=False
    )  # part_of 역방향(구역용)
    anchors: frozenset[str] = field(init=False)
    regions: frozenset[str] = field(init=False)  # 등급 1

    def __post_init__(self) -> None:
        self.anchors = frozenset(n.id for n in self.nodes.values() if n.is_anchor)
        self.regions = frozenset(n.id for n in self.nodes.values() if n.tier == 1)
        self._children = {}
        for child, rels in self.parents.items():
            for rel, parent in rels:
                if rel == "part_of":
                    self._children.setdefault(parent, []).append(child)

    # ── 기본 조회 ──────────────────────────────────────────────

    def __contains__(self, node_id: str) -> bool:
        return node_id in self.nodes

    def __len__(self) -> int:
        return len(self.nodes)

    def get(self, node_id: str) -> Node:
        try:
            return self.nodes[node_id]
        except KeyError:
            raise UnknownNodeError(node_id) from None

    def parent_ids(self, node_id: str, relations: Iterable[str] = RELATIONS) -> list[str]:
        self.get(node_id)
        allowed = set(relations)
        return [p for rel, p in self.parents.get(node_id, ()) if rel in allowed]

    def ancestors(self, node_id: str, relations: Iterable[str] = RELATIONS) -> list[str]:
        """자기 자신을 제외한 조상. 가까운 것부터(BFS 순서), 중복 없음."""
        self.get(node_id)
        allowed = set(relations)
        seen: set[str] = {node_id}
        out: list[str] = []
        q: deque[str] = deque([node_id])
        while q:
            cur = q.popleft()
            for rel, p in self.parents.get(cur, ()):
                if rel in allowed and p not in seen:
                    seen.add(p)
                    out.append(p)
                    q.append(p)
        return out

    def is_ancestor(self, ancestor_id: str, node_id: str) -> bool:
        return ancestor_id in self.ancestors(node_id)

    # ── 넓히기: 앵커 올리기 ────────────────────────────────────

    def nearest_anchors(self, node_id: str) -> list[str]:
        """올라가다 처음 만나는 앵커들. 자기 자신이 앵커면 자기 자신 하나.

        결과가 둘 이상이면 같은 거리에 앵커가 여럿 있다는 뜻이다(데이터 이슈).
        빈 리스트면 이 노드 위에는 앵커가 없다.
        """
        node = self.get(node_id)
        if node.is_anchor:
            return [node_id]
        # 거리별 BFS. 첫 앵커가 나온 깊이에서 멈춘다
        frontier = [node_id]
        seen = {node_id}
        while frontier:
            nxt: list[str] = []
            for cur in frontier:
                for _, p in self.parents.get(cur, ()):
                    if p not in seen:
                        seen.add(p)
                        nxt.append(p)
            found = sorted(p for p in nxt if p in self.anchors)
            if found:
                return found
            frontier = nxt
        return []

    def anchor_of(self, node_id: str) -> str | None:
        """가장 가까운 앵커 하나. 없으면 None, 여럿이면 AmbiguousAnchorError."""
        found = self.nearest_anchors(node_id)
        if not found:
            return None
        if len(found) > 1:
            raise AmbiguousAnchorError(f"{node_id}: 같은 거리에 앵커가 여럿 {found}")
        return found[0]

    def region_of(self, node_id: str) -> str | None:
        """등급 1 상부. 자기 자신이 상부면 자기 자신. 여럿이면 데이터 오류라 validate가 잡는다."""
        node = self.get(node_id)
        if node.tier == 1:
            return node_id
        found = sorted(a for a in self.ancestors(node_id) if a in self.regions)
        if len(found) > 1:
            raise AmbiguousAnchorError(f"{node_id}: 상부가 여럿 {found}")
        return found[0] if found else None

    def widen(self, node_id: str) -> Widening:
        """차트 용어 → (용어, 앵커, 상부, 경로). 설명문은 만들지 않는다. 형태만 준다."""
        term = self.get(node_id)
        anchor_id = self.anchor_of(node_id)
        if anchor_id is None:
            return Widening(term=term, anchor=None, region=None, path=(node_id,))
        region_id = self.region_of(anchor_id)
        return Widening(
            term=term,
            anchor=self.nodes[anchor_id],
            region=self.nodes[region_id] if region_id else None,
            path=self._path(node_id, anchor_id),
        )

    def _path(self, src: str, dst: str) -> tuple[str, ...]:
        """src에서 위로 올라가 dst에 닿는 최단 경로. 없으면 (src,)."""
        prev: dict[str, str | None] = {src: None}
        q: deque[str] = deque([src])
        while q:
            cur = q.popleft()
            if cur == dst:
                out: list[str] = []
                node: str | None = cur
                while node is not None:
                    out.append(node)
                    node = prev[node]
                return tuple(reversed(out))
            for _, p in self.parents.get(cur, ()):
                if p not in prev:
                    prev[p] = cur
                    q.append(p)
        return (src,)

    # ── 표면 층: 인체도 입력 ──────────────────────────────────

    def search(self, query: str, limit: int = 8) -> list[tuple[Node, str, int]]:
        """폼 입력으로 부위 찾기 — 유의어 표 매칭. LLM·임베딩 없음.

        점수: 이름·별칭과 정확히 같으면 3, 질의가 이름·별칭으로 시작하거나 그 반대면 2,
        한쪽이 다른 쪽을 포함하면 1. 앵커·구역만 대상(구조 노드는 넓히기가 따로 맡는다).
        반환: (노드, 걸린 표현, 점수). 점수 내림차순.
        같으면 걸린 표현이 긴 것(구체적) 먼저, 그다음 앵커.

        결과가 0건이고 입력이 한/영 오타로 보이면 한글로 복원해 한 번 더 찾는다("qo" → "배").
        0건일 때만 도므로 지금 결과가 나오는 질의는 결과가 바뀌지 않는다.
        조합 중간 입력("옆굴", "아랫ㅂ")은 다루지 않는다 — 앱이 처리할 몫.

        앵커가 걸리면 그 아래 구역을 CSV 순서로 뒤에 붙인다(점수 0). "다리"만 아는 사람이
        무릎·종아리를 보게 하려는 것이다. 고르지 않고 전부 준다 — 추리면 감별이 된다.
        """
        q = _norm_text(query)
        if len(q) < 1:
            return []
        hits: list[tuple[Node, str, int]] = []
        for n in self.nodes.values():
            if n.kind not in ("anchor", "surface"):
                continue
            best: tuple[str, int] | None = None
            for cand in (n.name_ko, *n.aliases):
                c = _norm_text(cand)
                if not c:
                    continue
                if c == q:
                    score = 3
                elif c.startswith(q) or q.startswith(c):
                    score = 2
                elif q in c or c in q:
                    score = 1
                else:
                    continue
                if best is None or score > best[1]:
                    best = (cand, score)
            if best:
                hits.append((n, best[0], best[1]))
        # 점수 같으면 더 구체적인(긴) 표현이 먼저: "왼쪽 아랫배가 아파요" → 아랫배 > 배
        hits.sort(key=lambda h: (-h[2], -len(h[1]), 0 if h[0].kind == "anchor" else 1, h[0].id))
        # 복원은 정규화 전 원문으로 한다 — 소문자로 바꾸면 shift 자리(ㄲ ㅒ 등)가 죽는다
        if not hits and looks_like_korean_typed_in_english(query):
            restored = english_keys_to_hangul(query)
            if restored != query:  # 복원 결과에는 한글이 있어 오타 판정이 다시 참이 되지 않는다
                return self.search(restored, limit=limit)
        # 앵커가 걸렸으면 그 아래 구역을 딸려 보낸다. 점수 0 = 직접 매칭이 아니라 앵커에 딸려 온 것
        seen = {n.id for n, _, _ in hits}
        for node, matched, _ in list(hits):
            if node.kind != "anchor":
                continue
            for zone in self.zones(node.id):
                if zone.id not in seen:
                    seen.add(zone.id)
                    hits.append((zone, matched, 0))
        return hits[:limit]

    def anchors_in_order(self) -> list[Node]:
        """인체도 첫 화면의 앵커 목록. CSV 순서 그대로(디자이너 배치 순서를 데이터가 가진다)."""
        return [n for n in self.nodes.values() if n.kind == "anchor"]

    def zones(self, anchor_id: str) -> list[Node]:
        """앵커 아래 표면 구역. CSV 순서. 구조 노드는 섞이지 않는다.

        환자가 앵커를 누른 뒤 보는 2단계 선택지다. 비어 있으면(전신) 구역 선택을 건너뛴다.
        """
        node = self.get(anchor_id)
        if node.kind != "anchor":
            raise ValueError(f"{anchor_id}: 앵커가 아니다 (kind={node.kind})")
        return [
            self.nodes[c]
            for c in self._children.get(anchor_id, ())
            if self.nodes[c].kind == "surface"
        ]

    def structures_under(self, surface_id: str) -> list[Node]:
        """이 표면 구역 아래에 놓인 구조. 해부학 사실의 인용이고 감별이 아니다.

        전체를 이름순으로 준다. 고르지 않고 순위도 없다. 출처가 비어 있는 엣지는 뺀다.
        """
        node = self.get(surface_id)
        if node.kind != "surface":
            raise ValueError(f"{surface_id}: 표면 구역이 아니다 (kind={node.kind})")
        ids = {s for s, z, src in self.located_in if z == surface_id and src.strip()}
        return sorted((self.nodes[i] for i in ids), key=lambda n: n.display_name)

    # ── LCA: 다부위 정리 ──────────────────────────────────────

    def lca(self, node_ids: Iterable[str]) -> set[str]:
        """최소공통조상 집합. 자기 자신도 조상으로 친다(A가 B의 조상이면 LCA는 A).

        DAG라 하나가 아닐 수 있어 집합으로 준다. 공통 조상이 없으면 빈 집합.
        「무릎도 어깨도」처럼 상위 부위 노드가 데이터에 없으면 빈 집합이 정답이다.
        """
        ids = list(dict.fromkeys(node_ids))
        if not ids:
            return set()
        common: set[str] | None = None
        for nid in ids:
            up = set(self.ancestors(nid)) | {nid}
            common = up if common is None else common & up
        assert common is not None
        # 공통 조상 중 다른 공통 조상의 조상인 것은 뺀다 → 최소만 남는다
        not_minimal = {a for c in common for a in self.ancestors(c) if a in common}
        return common - not_minimal

    # ── 검증 ──────────────────────────────────────────────────

    def validate(self) -> list[str]:
        """규칙 위반 목록. 비어 있으면 정상. 로드 시 자동 호출되지만 손으로도 돌릴 수 있다."""
        problems: list[str] = []
        for child, rels in self.parents.items():
            if child not in self.nodes:
                problems.append(f"edges: 없는 노드를 child로 참조 {child}")
            for rel, parent in rels:
                if rel not in RELATIONS:
                    problems.append(f"edges: 모르는 관계 {child} -{rel}-> {parent}")
                if parent not in self.nodes:
                    problems.append(f"edges: 없는 노드를 parent로 참조 {child} -> {parent}")
                if parent == child:
                    problems.append(f"edges: 자기 참조 {child}")
        problems.extend(f"cycle: {' -> '.join(c)}" for c in self.find_cycles())
        if not self.anchors:
            problems.append("nodes: 앵커가 하나도 없다 (is_anchor)")
        for n in self.nodes.values():
            if not n.structure_type:
                problems.append(f"nodes: structure_type 비어 있음 {n.id}")
            if n.tier not in (1, 2, 3):
                problems.append(f"nodes: tier는 1/2/3 {n.id}={n.tier}")
            if (n.tier == 2) != n.is_anchor:
                problems.append(f"nodes: tier 2 ⇔ is_anchor 불일치 {n.id}")
            if n.kind not in KINDS:
                problems.append(f"nodes: kind는 {sorted(KINDS)} {n.id}={n.kind}")
            expected_kind = {1: "region", 2: "anchor"}.get(n.tier)
            if expected_kind and n.kind != expected_kind:
                problems.append(f"nodes: tier {n.tier}는 kind={expected_kind} {n.id}={n.kind}")
            if n.tier == 3 and n.kind not in ("structure", "surface"):
                problems.append(f"nodes: tier 3은 structure/surface {n.id}={n.kind}")
            if n.laterality not in LATERALITIES:
                problems.append(f"nodes: laterality는 {sorted(LATERALITIES)} {n.id}={n.laterality}")
            if n.view not in VIEWS:
                problems.append(f"nodes: view는 {sorted(VIEWS)} {n.id}={n.view}")
            if n.departments and not n.departments_source:
                problems.append(f"nodes: departments에는 departments_source가 필수 {n.id}")
            if n.departments and n.kind not in ("anchor", "surface"):
                problems.append(f"nodes: departments는 앵커·구역에만 {n.id} (kind={n.kind})")
        for s_id, z_id, _ in self.located_in:
            if s_id not in self.nodes or z_id not in self.nodes:
                problems.append(f"located_in: 없는 노드 {s_id} -> {z_id}")
                continue
            if self.nodes[s_id].kind != "structure" or self.nodes[z_id].kind != "surface":
                problems.append(f"located_in: structure → surface 방향만 {s_id} -> {z_id}")
        if self.find_cycles():
            return problems  # 아래 도달성 검사는 DAG 전제
        for n in self.nodes.values():
            if n.tier == 1 and self.parents.get(n.id):
                problems.append(f"tier: 상부는 부모가 없어야 한다 {n.id}")
            if n.tier == 2:
                regs = [a for a in self.ancestors(n.id) if a in self.regions]
                if len(regs) != 1:
                    problems.append(f"tier: 앵커는 상부 정확히 하나에 닿아야 한다 {n.id} -> {regs}")
            if n.tier == 3:
                found = self.nearest_anchors(n.id)
                if len(found) != 1:
                    problems.append(
                        f"tier: 세부는 앵커 정확히 하나에 닿아야 한다 {n.id} -> {found}"
                    )
                if n.kind == "surface" and self.parent_ids(n.id) != found:
                    problems.append(f"tier: 표면 구역은 앵커의 직접 자식이어야 한다 {n.id}")
        return problems

    def find_cycles(self) -> list[tuple[str, ...]]:
        """색칠 DFS. 사이클마다 닫히는 경로 하나를 준다. DAG면 빈 리스트."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = dict.fromkeys(self.parents, WHITE)
        stack: list[str] = []
        cycles: list[tuple[str, ...]] = []

        def visit(u: str) -> None:
            color[u] = GRAY
            stack.append(u)
            for _, v in self.parents.get(u, ()):
                c = color.get(v, WHITE)
                if c == GRAY:
                    cycles.append(tuple(stack[stack.index(v) :]) + (v,))
                elif c == WHITE:
                    visit(v)
            stack.pop()
            color[u] = BLACK

        for u in list(self.parents):
            if color[u] == WHITE:
                visit(u)
        return cycles


# ── 로딩 ─────────────────────────────────────────────────────


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "y", "yes"}


def _split_departments(value: str) -> tuple[str, ...]:
    """';' 구분. 공백 제거, 빈 항목 제외. 순서는 CSV 그대로(정렬하지 않는다)."""
    return tuple(d.strip() for d in value.split(";") if d.strip())


def _default_kind(row: dict[str, str]) -> str:
    """kind 컬럼이 없는 옛 파일용. tier로 유추하고 세부는 구조로 본다."""
    tier = (row.get("tier") or "3").strip()
    return {"1": "region", "2": "anchor"}.get(tier, "structure")


def _norm_text(s: str) -> str:
    """검색 비교용: 공백·중간점·괄호 제거, 소문자."""
    return re.sub(r"[\s·()（）,.]", "", s or "").lower()


def load_ontology(directory: Path | str = DEFAULT_DIR, *, strict: bool = True) -> Ontology:
    """nodes.csv + edges.csv → Ontology. strict면 검증 실패 시 OntologyError."""
    directory = Path(directory)
    nodes_path = directory / "nodes.csv"
    edges_path = directory / "edges.csv"
    nodes_bytes = nodes_path.read_bytes()
    edges_bytes = edges_path.read_bytes()
    aliases_path = directory / "aliases.csv"
    aliases_bytes = aliases_path.read_bytes() if aliases_path.exists() else b""
    snapshot = hashlib.sha256(
        nodes_bytes + b"\n" + edges_bytes + b"\n" + aliases_bytes
    ).hexdigest()[:12]

    problems: list[str] = []
    nodes: dict[str, Node] = {}
    # utf-8-sig: 추출 스크립트가 BOM을 붙인다
    for row in csv.DictReader(nodes_bytes.decode("utf-8-sig").splitlines()):
        nid = row["id"].strip()
        if not nid:
            continue
        if nid in nodes:
            problems.append(f"nodes: 중복 id {nid}")
            continue
        nodes[nid] = Node(
            id=nid,
            name_en=row.get("name_en", "").strip(),
            name_ko=row.get("name_ko", "").strip(),
            region=row.get("region", "").strip(),
            structure_type=row.get("structure_type", "").strip(),
            sctid=row.get("sctid", "").strip(),
            fma=row.get("fma", "").strip(),
            definition_en=row.get("definition_en", "").strip(),
            source=row.get("source", "").strip(),
            is_anchor=_truthy(row.get("is_anchor") or ""),
            tier=int(row.get("tier") or 3),  # 컬럼이 없으면 세부로 본다
            kind=(row.get("kind") or "").strip() or _default_kind(row),
            laterality=(row.get("laterality") or "").strip() or "left_right",
            view=(row.get("view") or "").strip() or "none",
            departments=_split_departments(row.get("departments") or ""),
            departments_source=(row.get("departments_source") or "").strip(),
        )

    # 유의어(aliases.csv): node_id,alias,note. 없는 노드를 가리키면 데이터 오류
    if aliases_bytes:
        by_node: dict[str, list[str]] = {}
        for row in csv.DictReader(aliases_bytes.decode("utf-8-sig").splitlines()):
            nid = (row.get("node_id") or "").strip()
            alias = (row.get("alias") or "").strip()
            if not nid or not alias:
                continue
            if nid not in nodes:
                problems.append(f"aliases: 없는 노드 {nid} ({alias})")
                continue
            if alias in by_node.setdefault(nid, []):
                problems.append(f"aliases: 중복 {nid} {alias}")
                continue
            by_node[nid].append(alias)
        for nid, al in by_node.items():
            nodes[nid] = replace(nodes[nid], aliases=tuple(al))

    parents: dict[str, list[tuple[str, str]]] = {nid: [] for nid in nodes}
    located_in: list[tuple[str, str, str]] = []
    seen_edges: set[tuple[str, str, str]] = set()
    for row in csv.DictReader(edges_bytes.decode("utf-8-sig").splitlines()):
        child, rel, parent = (row[k].strip() for k in ("child", "relation", "parent"))
        key = (child, rel, parent)
        if key in seen_edges:
            problems.append(f"edges: 중복 {child} -{rel}-> {parent}")
            continue
        seen_edges.add(key)
        if rel in ASSOC_RELATIONS:
            located_in.append((child, parent, (row.get("source") or "").strip()))
        else:
            parents.setdefault(child, []).append((rel, parent))

    onto = Ontology(nodes=nodes, parents=parents, snapshot_id=snapshot, located_in=located_in)
    problems.extend(onto.validate())
    if strict and problems:
        raise OntologyError(f"{directory}: 문제 {len(problems)}건\n" + "\n".join(problems))
    return onto
