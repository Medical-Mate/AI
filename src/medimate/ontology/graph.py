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

검증:
- partonomy는 DAG여야 한다. 손보강 노드가 들어오면 사이클이 생길 수 있어 로드 시 검출한다.
- 엣지가 없는 노드를 가리키면 실패. 조용히 넘어가지 않는다.
"""

from __future__ import annotations

import csv
import hashlib
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

RELATIONS = frozenset({"part_of", "is_a"})
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

    @property
    def display_name(self) -> str:
        """환자에게 보이는 이름. 한국어가 없으면 영어로 낸다. 빈 문자열을 만들지 않는다."""
        return self.name_ko or self.name_en


@dataclass(frozen=True, slots=True)
class Widening:
    """넓히기 결과 — 대체가 아니라 병기 (§3). 원문 용어 + 앵커 부위 + 구조 유형을 같이 낸다."""

    term: Node
    anchor: Node | None  # 올라가서 닿은 앵커. 없으면 None (앵커 밖의 노드)
    path: tuple[str, ...]  # term → … → anchor. 감사·디버깅용


@dataclass
class Ontology:
    nodes: dict[str, Node]
    parents: dict[str, list[tuple[str, str]]]  # child → [(relation, parent)]
    snapshot_id: str  # CSV 바이트 해시. Provenance.ontology_snapshot에 박는 값
    anchors: frozenset[str] = field(init=False)

    def __post_init__(self) -> None:
        self.anchors = frozenset(n.id for n in self.nodes.values() if n.is_anchor)

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

    def widen(self, node_id: str) -> Widening:
        """차트 용어 → (용어, 앵커, 경로). 설명문은 만들지 않는다. 형태만 준다."""
        term = self.get(node_id)
        anchor_id = self.anchor_of(node_id)
        if anchor_id is None:
            return Widening(term=term, anchor=None, path=(node_id,))
        return Widening(
            term=term, anchor=self.nodes[anchor_id], path=self._path(node_id, anchor_id)
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


def load_ontology(directory: Path | str = DEFAULT_DIR, *, strict: bool = True) -> Ontology:
    """nodes.csv + edges.csv → Ontology. strict면 검증 실패 시 OntologyError."""
    directory = Path(directory)
    nodes_path = directory / "nodes.csv"
    edges_path = directory / "edges.csv"
    nodes_bytes = nodes_path.read_bytes()
    edges_bytes = edges_path.read_bytes()
    snapshot = hashlib.sha256(nodes_bytes + b"\n" + edges_bytes).hexdigest()[:12]

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
        )

    parents: dict[str, list[tuple[str, str]]] = {nid: [] for nid in nodes}
    seen_edges: set[tuple[str, str, str]] = set()
    for row in csv.DictReader(edges_bytes.decode("utf-8-sig").splitlines()):
        child, rel, parent = (row[k].strip() for k in ("child", "relation", "parent"))
        key = (child, rel, parent)
        if key in seen_edges:
            problems.append(f"edges: 중복 {child} -{rel}-> {parent}")
            continue
        seen_edges.add(key)
        parents.setdefault(child, []).append((rel, parent))

    onto = Ontology(nodes=nodes, parents=parents, snapshot_id=snapshot)
    problems.extend(onto.validate())
    if strict and problems:
        raise OntologyError(f"{directory}: 문제 {len(problems)}건\n" + "\n".join(problems))
    return onto
