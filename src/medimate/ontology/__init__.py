"""부위 온톨로지 — 로더와 그래프 연산 (docs/ai-design.md §3).

그래프 DB를 쓰지 않는다. 인접 리스트 + 코드 수십 줄로 필요한 연산이 전부 된다.
- 조상 조회 · 앵커 올리기(넓히기) · LCA(다부위 정리) · 사이클 검출(DAG 검증)
"""

from medimate.ontology.graph import (
    DEFAULT_DIR,
    AmbiguousAnchorError,
    Node,
    Ontology,
    OntologyError,
    UnknownNodeError,
    Widening,
    load_ontology,
)

__all__ = [
    "DEFAULT_DIR",
    "AmbiguousAnchorError",
    "Node",
    "Ontology",
    "OntologyError",
    "UnknownNodeError",
    "Widening",
    "load_ontology",
]
