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
INFRASPINATUS = "UBERON:0001477"
INFRASPINATUS_TENDON = "UBERON:0012118"
UPPER_LIMB = "REG:001"
LOWER_LIMB = "REG:002"


@pytest.fixture(scope="module")
def onto() -> Ontology:
    return load_ontology()


# ── 실제 데이터 ─────────────────────────────────────────────


def test_real_data_loads_clean(onto: Ontology):
    assert len(onto) == 44  # 세부 38 + 앵커 2 + 상부 4
    assert onto.validate() == []
    assert onto.anchors == {KNEE, SHOULDER}
    assert onto.regions == {"REG:001", "REG:002", "REG:003", "REG:004"}
    assert len(onto.snapshot_id) == 12


def test_acl_climbs_to_knee_via_is_a_then_part_of(onto: Ontology):
    # README: ACL → 십자인대 → 무릎관절 인대 →(part_of)→ 무릎관절 →(part_of)→ 무릎
    assert onto.anchor_of(ACL) == KNEE
    w = onto.widen(ACL)
    assert w.anchor is not None and w.anchor.id == KNEE
    assert w.region is not None and w.region.id == LOWER_LIMB
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


def test_every_detail_and_anchor_reaches_exactly_one_anchor(onto: Ontology):
    for nid, n in onto.nodes.items():
        if n.tier == 1:
            assert onto.anchor_of(nid) is None, nid  # 상부 위에는 앵커가 없다
        else:
            assert onto.anchor_of(nid) in onto.anchors, nid


def test_tiers_are_presentation_levels_not_graph_depth(onto: Ontology):
    # 가시아래근힘줄 → 가시아래근 → 돌림근띠 → 어깨: 세부 안쪽 깊이 3. 등급은 그대로 3.
    # 힘줄⊂근육⊂돌림근띠 관계는 유지되고, 앵커까지는 더 짧은 is_a(어깨 결합조직) 경로가 잡힌다
    assert onto.get(INFRASPINATUS_TENDON).tier == 3
    up = onto.ancestors(INFRASPINATUS_TENDON)
    assert INFRASPINATUS in up and ROTATOR_CUFF in up
    w = onto.widen(INFRASPINATUS_TENDON)
    assert w.anchor is not None and w.anchor.id == SHOULDER
    assert w.path == (INFRASPINATUS_TENDON, "UBERON:0003579", SHOULDER)
    assert onto.region_of(INFRASPINATUS_TENDON) == UPPER_LIMB
    assert onto.region_of(SHOULDER) == UPPER_LIMB
    assert onto.region_of(UPPER_LIMB) == UPPER_LIMB


def test_anchor_of_anchor_is_itself(onto: Ontology):
    assert onto.anchor_of(KNEE) == KNEE
    assert onto.widen(KNEE).path == (KNEE,)


def test_lca_same_region(onto: Ontology):
    assert onto.lca([ACL, PCL]) == {"UBERON:0006659"}  # 십자인대
    assert onto.lca([SUPRASPINATUS, INFRASPINATUS]) == {ROTATOR_CUFF}
    assert onto.lca([ACL, MCL]) == {KNEE_JOINT}


def test_lca_includes_self_and_handles_no_common(onto: Ontology):
    assert onto.lca([ACL, KNEE]) == {KNEE}
    assert onto.lca([ACL]) == {ACL}
    assert onto.lca([]) == set()
    # 무릎(하지)·어깨(상지)는 상부가 다르고 상부 위는 없다 → 공통 조상 없음
    assert onto.lca([ACL, ACROMION]) == set()
    assert onto.lca([KNEE, LOWER_LIMB]) == {LOWER_LIMB}


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
# 이름 규칙: R* 상부(등급 1) / A* 앵커(등급 2) / 그 외 세부(등급 3)

NODE_COLS = (
    "id,name_en,name_ko,region,structure_type,sctid,fma,definition_en,source,is_anchor,tier\n"
)


def _tier(n: str) -> str:
    return "1" if n.startswith("R") else "2" if n.startswith("A") else "3"


def _write(tmp: Path, nodes: list[str], edges: list[str], *, auto_region: bool = True) -> Path:
    """auto_region이면 R1을 만들고 앵커를 전부 R1에 붙인다 (등급 규칙 충족용)."""
    nodes = list(nodes)
    edges = list(edges)
    if auto_region and not any(n.startswith("R") for n in nodes):
        nodes.append("R1")
        edges += [f"{n},part_of,R1" for n in nodes if n.startswith("A")]
    (tmp / "nodes.csv").write_text(
        NODE_COLS
        + "".join(f"{n},,,,x,,,,,{'1' if n.startswith('A') else ''},{_tier(n)}\n" for n in nodes),
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
    with pytest.raises(OntologyError, match="세부는 앵커 정확히 하나"):
        load_ontology(d)  # 등급 규칙 위반으로 로드 자체가 막힌다
    onto = load_ontology(d, strict=False)
    assert onto.nearest_anchors("n1") == ["A1", "A2"]
    with pytest.raises(AmbiguousAnchorError):
        onto.anchor_of("n1")


def test_nearest_anchor_stops_at_first_depth(tmp_path: Path):
    # n1 → A1 → A2 : 더 위에 앵커가 있어도 가장 가까운 A1에서 멈춘다.
    # 앵커 중첩은 등급 규칙 밖이라 느슨 로드
    d = _write(tmp_path, ["A1", "A2", "n1"], ["n1,part_of,A1", "A1,part_of,A2"])
    onto = load_ontology(d, strict=False)
    assert onto.anchor_of("n1") == "A1"


def test_node_without_anchor_above(tmp_path: Path):
    d = _write(tmp_path, ["A1", "n1", "n2"], ["n2,part_of,n1"])
    with pytest.raises(OntologyError, match="세부는 앵커 정확히 하나"):
        load_ontology(d)
    onto = load_ontology(d, strict=False)
    assert onto.anchor_of("n2") is None
    w = onto.widen("n2")
    assert w.anchor is None and w.region is None


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


def test_anchor_spanning_two_regions_is_rejected(tmp_path: Path):
    # 어깨가 상지·몸통 둘 다에 걸치면 거부. 상부는 부위마다 하나만
    d = _write(
        tmp_path,
        ["R1", "R2", "A1", "n1"],
        ["n1,part_of,A1", "A1,part_of,R1", "A1,part_of,R2"],
        auto_region=False,
    )
    with pytest.raises(OntologyError, match="앵커는 상부 정확히 하나"):
        load_ontology(d)


def test_region_must_be_root_and_tier_matches_anchor(tmp_path: Path):
    (tmp_path / "nodes.csv").write_text(
        NODE_COLS + "R1,,,,x,,,,,,1\nR2,,,,x,,,,,,1\nA1,,,,x,,,,,1,3\n", encoding="utf-8"
    )
    (tmp_path / "edges.csv").write_text(
        "child,relation,parent\nR1,part_of,R2\nA1,part_of,R1\n", encoding="utf-8"
    )
    with pytest.raises(OntologyError) as ei:
        load_ontology(tmp_path)
    msg = str(ei.value)
    assert "상부는 부모가 없어야" in msg and "is_anchor 불일치" in msg


def test_tier_column_missing_defaults_to_detail(tmp_path: Path):
    cols = "id,name_en,name_ko,region,structure_type,sctid,fma,definition_en,source,is_anchor\n"
    (tmp_path / "nodes.csv").write_text(cols + "n1,,,,x,,,,,\n", encoding="utf-8")
    (tmp_path / "edges.csv").write_text("child,relation,parent\n", encoding="utf-8")
    onto = load_ontology(tmp_path, strict=False)
    assert onto.get("n1").tier == 3
