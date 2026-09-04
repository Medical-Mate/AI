"""온톨로지 로더·그래프 연산. LLM 호출 없음, 실제 CSV + 합성 그래프."""

from pathlib import Path

import pytest

from medimate.ontology import (
    AmbiguousAnchorError,
    Ontology,
    OntologyError,
    UnknownNodeError,
    load_ontology,
)

KNEE = "UBERON:0001465"
SHOULDER = "UBERON:0001467"
KNEE_JOINT = "UBERON:0001485"
ACL = "UBERON:0003671"
PCL = "UBERON:0003680"
MCL = "MAN:001"
ACROMION = "MAN:011"
SCAPULA = "MAN:008"
ROTATOR_CUFF = "UBERON:0003683"
SUPRASPINATUS = "UBERON:0002383"


@pytest.fixture(scope="module")
def onto() -> Ontology:
    return load_ontology()


# ── 실제 데이터 ─────────────────────────────────────────────


def test_real_data_loads_clean(onto: Ontology):
    assert len(onto) == 40
    assert onto.validate() == []
    assert onto.anchors == {KNEE, SHOULDER}
    assert len(onto.snapshot_id) == 12


def test_acl_climbs_to_knee_via_is_a_then_part_of(onto: Ontology):
    # README: ACL → 십자인대 → 무릎관절 인대 →(part_of)→ 무릎관절 →(part_of)→ 무릎
    assert onto.anchor_of(ACL) == KNEE
    w = onto.widen(ACL)
    assert w.anchor is not None and w.anchor.id == KNEE
    assert w.path[0] == ACL and w.path[-1] == KNEE
    assert KNEE_JOINT in w.path
    assert w.term.structure_type == "인대"


def test_part_of_alone_does_not_reach_anchor_from_acl(onto: Ontology):
    # is_a를 빼면 ACL은 십자인대 계열을 못 타고 끊긴다. 둘 다 타야 하는 이유
    assert KNEE not in onto.ancestors(ACL, relations=["part_of"])
    assert KNEE in onto.ancestors(ACL)


def test_manual_nodes_climb_too(onto: Ontology):
    assert onto.anchor_of(MCL) == KNEE
    assert onto.anchor_of(ACROMION) == SHOULDER  # MAN:011 → MAN:008 → 어깨
    assert onto.widen(ACROMION).path == (ACROMION, SCAPULA, SHOULDER)


def test_every_non_anchor_node_reaches_exactly_one_anchor(onto: Ontology):
    for nid in onto.nodes:
        assert onto.anchor_of(nid) in onto.anchors, nid


def test_anchor_of_anchor_is_itself(onto: Ontology):
    assert onto.anchor_of(KNEE) == KNEE
    assert onto.widen(KNEE).path == (KNEE,)


def test_lca_same_region(onto: Ontology):
    assert onto.lca([ACL, PCL]) == {"UBERON:0006659"}  # 십자인대
    assert onto.lca([SUPRASPINATUS, "UBERON:0001477"]) == {ROTATOR_CUFF}
    assert onto.lca([ACL, MCL]) == {"UBERON:0001485"}  # 무릎관절


def test_lca_includes_self_and_handles_no_common(onto: Ontology):
    assert onto.lca([ACL, KNEE]) == {KNEE}
    assert onto.lca([ACL]) == {ACL}
    assert onto.lca([]) == set()
    # 무릎·어깨 위에 상위 부위 노드가 없다 → 공통 조상 없음이 정답
    assert onto.lca([ACL, ACROMION]) == set()


def test_unknown_node_raises(onto: Ontology):
    with pytest.raises(UnknownNodeError):
        onto.ancestors("UBERON:9999999")
    with pytest.raises(UnknownNodeError):
        onto.widen("nope")


def test_display_name_falls_back_to_english(onto: Ontology):
    n = onto.get("UBERON:0014899")  # anterolateral ligament, name_ko 비어 있음
    assert n.name_ko == ""
    assert n.display_name == n.name_en


# ── 합성 그래프 ─────────────────────────────────────────────

NODE_COLS = "id,name_en,name_ko,region,structure_type,sctid,fma,definition_en,source,is_anchor\n"


def _write(tmp: Path, nodes: list[str], edges: list[str]) -> Path:
    (tmp / "nodes.csv").write_text(
        NODE_COLS + "".join(f"{n},,,,x,,,,,{'1' if n.startswith('A') else ''}\n" for n in nodes),
        encoding="utf-8",
    )
    (tmp / "edges.csv").write_text(
        "child,relation,parent\n" + "".join(f"{e}\n" for e in edges), encoding="utf-8"
    )
    return tmp


def test_cycle_is_detected(tmp_path: Path):
    d = _write(tmp_path, ["A1", "n1", "n2"], ["n1,part_of,n2", "n2,part_of,n1", "n1,part_of,A1"])
    with pytest.raises(OntologyError, match="cycle"):
        load_ontology(d)
    onto = load_ontology(d, strict=False)
    assert onto.find_cycles() == [("n1", "n2", "n1")]


def test_dangling_edge_and_unknown_relation_are_reported(tmp_path: Path):
    d = _write(tmp_path, ["A1", "n1"], ["n1,part_of,ghost", "n1,near,A1"])
    with pytest.raises(OntologyError) as ei:
        load_ontology(d)
    msg = str(ei.value)
    assert "ghost" in msg and "near" in msg


def test_ambiguous_anchor_at_equal_depth(tmp_path: Path):
    d = _write(tmp_path, ["A1", "A2", "n1"], ["n1,part_of,A1", "n1,part_of,A2"])
    onto = load_ontology(d)
    assert onto.nearest_anchors("n1") == ["A1", "A2"]
    with pytest.raises(AmbiguousAnchorError):
        onto.anchor_of("n1")


def test_nearest_anchor_stops_at_first_depth(tmp_path: Path):
    # n1 → A1 → A2 : 더 위에 앵커가 있어도 가장 가까운 A1에서 멈춘다
    d = _write(tmp_path, ["A1", "A2", "n1"], ["n1,part_of,A1", "A1,part_of,A2"])
    onto = load_ontology(d)
    assert onto.anchor_of("n1") == "A1"
    assert onto.anchor_of("n1") != "A2"


def test_node_without_anchor_above(tmp_path: Path):
    d = _write(tmp_path, ["A1", "n1", "n2"], ["n2,part_of,n1"])
    onto = load_ontology(d)
    assert onto.anchor_of("n2") is None
    assert onto.widen("n2").anchor is None


def test_lca_multiple_minimal_in_dag(tmp_path: Path):
    # x,y 둘 다 p,q의 자식이고 p,q는 서로 조상이 아님 → LCA = {p, q}
    d = _write(
        tmp_path,
        ["A1", "p", "q", "x", "y"],
        [
            "x,part_of,p",
            "x,part_of,q",
            "y,part_of,p",
            "y,part_of,q",
            "p,part_of,A1",
            "q,part_of,A1",
        ],
    )
    onto = load_ontology(d)
    assert onto.lca(["x", "y"]) == {"p", "q"}
