"""온톨로지 로더·그래프 연산. LLM 호출 없음, 실제 CSV + 합성 그래프."""

import inspect
from pathlib import Path

import pytest

from medimate.ontology import (
    KINDS,
    AmbiguousAnchorError,
    Ontology,
    OntologyError,
    UnknownNodeError,
    load_ontology,
)
from medimate.ontology.korean_keys import (
    _KEY_TO_JAMO,
    english_keys_to_hangul,
    looks_like_korean_typed_in_english,
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
WHOLE_BODY = "ANC:010"
ARM = "ANC:013"
LEG = "ANC:014"
BACK_HIP = "ANC:012"


@pytest.fixture(scope="module")
def onto() -> Ontology:
    return load_ontology()


# ── 실제 데이터 ─────────────────────────────────────────────


def test_real_data_loads_clean(onto: Ontology):
    kinds = {k: sum(1 for n in onto.nodes.values() if n.kind == k) for k in KINDS}
    # 2026-09-04 통합: 무릎·어깨는 팔·다리 아래 구조 노드가 됐다 (38 + 2)
    assert kinds == {"region": 5, "anchor": 9, "structure": 40, "surface": 25}
    assert len(onto) == 79
    assert onto.validate() == []
    assert {ARM, LEG, BACK_HIP, WHOLE_BODY} <= onto.anchors
    assert KNEE not in onto.anchors and SHOULDER not in onto.anchors
    assert onto.regions == {"REG:001", "REG:002", "REG:003", "REG:004", "REG:005"}
    assert len(onto.snapshot_id) == 12


def test_acl_climbs_to_knee_via_is_a_then_part_of(onto: Ontology):
    # ACL → 십자인대 → 무릎관절 인대 →(part_of)→ 무릎관절 →(part_of)→ 무릎 →(part_of)→ 다리(앵커)
    assert onto.anchor_of(ACL) == LEG
    w = onto.widen(ACL)
    assert w.anchor is not None and w.anchor.id == LEG
    assert w.region is not None and w.region.id == LOWER_LIMB
    assert w.path[0] == ACL and w.path[-1] == LEG
    assert KNEE_JOINT in w.path and KNEE in w.path
    assert w.term.structure_type == "인대"


def test_part_of_alone_does_not_reach_anchor_from_acl(onto: Ontology):
    # is_a를 빼면 ACL은 십자인대 계열을 못 타고 끊긴다. 둘 다 타야 하는 이유
    assert LEG not in onto.ancestors(ACL, relations=["part_of"])
    assert LEG in onto.ancestors(ACL)


def test_manual_nodes_climb_too(onto: Ontology):
    assert onto.anchor_of(MCL) == LEG
    assert onto.anchor_of(ACROMION) == ARM  # MAN:011 → MAN:008 → 어깨 → 팔
    assert onto.widen(ACROMION).path == (ACROMION, SCAPULA, SHOULDER, ARM)


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
    assert w.anchor is not None and w.anchor.id == ARM
    assert w.path == (INFRASPINATUS_TENDON, "UBERON:0003579", SHOULDER, ARM)
    assert onto.region_of(INFRASPINATUS_TENDON) == UPPER_LIMB
    assert onto.region_of(SHOULDER) == UPPER_LIMB
    assert onto.region_of(UPPER_LIMB) == UPPER_LIMB


def test_anchor_of_anchor_is_itself(onto: Ontology):
    assert onto.anchor_of(ARM) == ARM
    assert onto.widen(ARM).path == (ARM,)
    assert onto.get(KNEE).kind == "structure"  # 통합 후 무릎은 앵커가 아니다


def test_lca_same_region(onto: Ontology):
    assert onto.lca([ACL, PCL]) == {"UBERON:0006659"}  # 십자인대
    assert onto.lca([SUPRASPINATUS, INFRASPINATUS]) == {ROTATOR_CUFF}
    assert onto.lca([ACL, MCL]) == {KNEE_JOINT}


def test_lca_across_anchors_stops_at_region(onto: Ontology):
    # 「무릎도 발목도」 → 다리(같은 앵커). 「팔도 다리도」 → 상부가 달라 공통 없음
    assert onto.lca(["SUR:091", "SUR:101"]) == {LEG}  # 무릎 + 발목 구역
    assert onto.lca([ACL, ACROMION]) == set()
    assert onto.lca([ARM, LEG]) == set()
    assert onto.lca([LEG, LOWER_LIMB]) == {LOWER_LIMB}
    assert onto.lca([ACL, KNEE]) == {KNEE}
    assert onto.lca([ACL]) == {ACL}
    assert onto.lca([]) == set()


def test_unknown_node_raises(onto: Ontology):
    with pytest.raises(UnknownNodeError):
        onto.ancestors("UBERON:9999999")
    with pytest.raises(UnknownNodeError):
        onto.widen("nope")


def test_display_name_falls_back_to_english(onto: Ontology):
    n = onto.get("UBERON:0014899")  # anterolateral ligament, name_ko 비어 있음
    assert n.name_ko == ""
    assert n.display_name == n.name_en


# ── 표면 층 (인체도 입력) ────────────────────────────────────


def test_surface_zones_per_anchor(onto: Ontology):
    names = [z.display_name for z in onto.zones(ARM)]
    assert names == ["어깨", "위팔", "팔꿈치", "아래팔", "손목", "손"]  # 팔 이미지 한 장을 세로로
    assert [z.display_name for z in onto.zones(LEG)] == ["허벅지", "무릎", "종아리", "발목", "발"]
    assert all(z.kind == "surface" for z in onto.zones(LEG))  # 구조 노드(무릎 등)는 섞이지 않는다
    assert onto.zones(WHOLE_BODY) == []  # 전신: 구역 없음, 선택 건너뜀
    assert onto.get("SUR:005").display_name == "입"  # 치아는 치과 영역, 제외
    assert onto.get("SUR:001").display_name == "머리 전체·이마"  # 두피는 피부 탭
    assert "치과" not in {d for n in onto.nodes.values() for d in n.departments}
    # 허리·엉덩이는 뒷면 한 점. 구역 축은 앵커마다 다르다
    assert [z.display_name for z in onto.zones(BACK_HIP)] == ["허리 가운데", "허리 옆", "엉덩이"]


def test_zones_reject_non_anchor(onto: Ontology):
    with pytest.raises(ValueError):
        onto.zones(ACL)
    with pytest.raises(ValueError):
        onto.zones(SHOULDER)  # 통합 후 어깨는 구조 노드다


def test_anchor_order_and_laterality(onto: Ontology):
    order = [a.id for a in onto.anchors_in_order()]
    assert len(order) == 9
    assert order[-3:] == [BACK_HIP, WHOLE_BODY, "ANC:011"]  # 앞면 6 → 뒷면 1 → 사이드 탭 2
    assert onto.get("ANC:001").laterality == "none"  # 머리
    assert onto.get("SUR:002").laterality == "left_right"  # 눈은 좌우가 있다
    assert onto.get(ARM).laterality == "left_right"  # 팔·다리는 누른 점으로 좌우가 정해진다
    # 인체도 면: 앞/뒤/없음
    assert onto.get(ARM).view == "front" and onto.get(BACK_HIP).view == "back"
    assert onto.get(WHOLE_BODY).view == "none" and onto.get("SUR:081").view == "back"
    assert onto.get(KNEE).view == "none"  # 구조 노드는 화면에 없다
    front = [a for a in onto.anchors_in_order() if a.view == "front"]
    assert [a.display_name for a in front] == ["머리", "목", "가슴", "배", "팔", "다리"]


def test_surface_zone_widens_to_anchor_and_region(onto: Ontology):
    w = onto.widen("SUR:011")  # 목 안(목구멍)
    assert w.anchor is not None and w.anchor.display_name == "목"
    assert w.region is not None and w.region.display_name == "머리·목"


def test_structures_under_is_empty_until_sourced(onto: Ontology):
    # located_in 엣지가 아직 없다(자문 회신 전). 비어 있는 게 정답이고 채워지면 이 테스트를 바꾼다
    assert onto.located_in == []
    assert onto.structures_under("SUR:051") == []
    with pytest.raises(ValueError):
        onto.structures_under(ACL)  # 구조 노드로는 묻지 못한다. 방향이 반대다


# ── 진료과 안내 (docs/decisions/2026-09-04-department-guidance.md) ──


def test_departments_follow_decision_table(onto: Ontology):
    src = "팀 결정 2026-09-04, 의료인 자문 확인 전"
    # 앵커
    assert onto.get("ANC:001").departments == ("신경과", "가정의학과")  # 머리
    assert onto.get("ANC:002").departments == ()  # 목: 구역 필수
    assert onto.get("ANC:003").departments == ("내과", "심장내과", "호흡기내과")  # 가슴
    assert onto.get(ARM).departments == ("정형외과",)
    assert onto.get(LEG).departments == ("정형외과",)
    assert onto.get(BACK_HIP).departments == ("정형외과", "신경외과")
    assert onto.get(KNEE).departments == ()  # 구조 노드에는 진료과가 없다
    assert onto.get(WHOLE_BODY).departments == ("내과", "가정의학과")
    # 구역
    assert onto.get("SUR:002").departments == ("안과",)
    assert onto.get("SUR:005").departments == ("이비인후과",)  # 치과는 연결하지 않는다
    assert onto.get("SUR:011").departments == ("이비인후과", "내과")
    assert onto.get("SUR:032").departments == ("내과", "소화기내과", "산부인과", "비뇨의학과")
    assert onto.get("SUR:042").departments == ("내과", "비뇨의학과", "정형외과")
    assert (
        onto.get("SUR:051").departments == ()
    )  # 어깨 구역: 값 없음 → 소비자가 앵커 값(정형외과)을 쓴다
    assert onto.get("SUR:081").departments == ()  # 엉덩이: 앵커 값(정형외과·신경외과)
    # 값이 있으면 출처가 붙어 있고, 그 출처는 "인용 아님"을 말한다
    for n in onto.nodes.values():
        if n.departments:
            assert n.departments_source == src, n.id
            assert n.kind in ("anchor", "surface"), n.id


def test_skin_anchor_is_side_tab_under_whole_body(onto: Ontology):
    skin = onto.get("ANC:011")
    assert skin.kind == "anchor" and skin.display_name == "피부"
    assert onto.zones("ANC:011") == []  # 구역 없음. 위치는 마네킹 앵커로 따로 받는다
    assert onto.region_of("ANC:011") == "REG:005"  # 전신 묶음. 상부 하나에만 닿는다
    assert skin.departments == ("피부과", "내과")
    assert [a.id for a in onto.anchors_in_order()][-2:] == [
        "ANC:010",
        "ANC:011",
    ]  # 사이드 탭은 마지막


def test_departments_are_exposed_only_never_selected(onto: Ontology):
    """로더는 진료과를 고르지도, 정렬하지도, 증상 축을 보지도 않는다.

    셋 중 하나라도 하면 감별이다.
    """
    import medimate.ontology.graph as graph

    source = inspect.getsource(graph)
    # 증상 축(schema.Axis)을 참조하지 않는다 — import도, 축 이름도 없다
    assert "medimate.schema" not in source and "medimate.dialog" not in source
    for axis in (
        "chief_complaint",
        "onset",
        "character",
        "radiation",
        "associated",
        "time_course",
        "exacerbating",
        "severity",
        "symptom",
    ):
        assert axis not in source, axis
    # departments를 다루는 메서드가 없다. Node 필드로만 존재한다
    dept_methods = [
        name
        for name, _ in inspect.getmembers(Ontology, inspect.isfunction)
        if "depart" in name.lower() or "recommend" in name.lower()
    ]
    assert dept_methods == []
    assert "recommend" not in source.lower()
    # 값은 CSV 순서 그대로. 정렬돼 있지 않다 (아랫배: 내과가 먼저, 비뇨의학과가 마지막)
    assert onto.get("SUR:032").departments[0] == "내과"
    assert onto.get("SUR:032").departments != tuple(sorted(onto.get("SUR:032").departments))


def test_departments_require_source_and_only_on_anchor_or_surface(tmp_path: Path):
    cols = NODE_COLS.rstrip("\n") + ",departments,departments_source\n"
    (tmp_path / "nodes.csv").write_text(
        cols
        + "R1,,,,x,,,,,,1,region,,\n"
        + "A1,,,,x,,,,,1,2,anchor,내과,\n"  # 출처 없음
        + "s1,,,,x,,,,,,3,structure,정형외과,팀 결정\n",  # 구조 노드에 진료과
        encoding="utf-8",
    )
    (tmp_path / "edges.csv").write_text(
        "child,relation,parent,source\nA1,part_of,R1,\ns1,part_of,A1,\n", encoding="utf-8"
    )
    with pytest.raises(OntologyError) as ei:
        load_ontology(tmp_path)
    msg = str(ei.value)
    assert "departments_source가 필수" in msg and "앵커·구역에만" in msg


def test_departments_split_and_strip(tmp_path: Path):
    cols = NODE_COLS.rstrip("\n") + ",departments,departments_source\n"
    (tmp_path / "nodes.csv").write_text(
        cols + "R1,,,,x,,,,,,1,region,,\nA1,,,,x,,,,,1,2,anchor, 내과 ;;가정의학과 ,팀 결정\n",
        encoding="utf-8",
    )
    (tmp_path / "edges.csv").write_text(
        "child,relation,parent,source\nA1,part_of,R1,\n", encoding="utf-8"
    )
    onto = load_ontology(tmp_path)
    assert onto.get("A1").departments == ("내과", "가정의학과")
    assert onto.get("R1").departments == ()


# ── 합성 그래프 ─────────────────────────────────────────────
# 이름 규칙: R* 상부 / A* 앵커 / Z* 표면 구역 / 그 외 구조

NODE_COLS = (
    "id,name_en,name_ko,region,structure_type,sctid,fma,definition_en,source,is_anchor,tier,kind\n"
)


def _tier(n: str) -> str:
    return "1" if n.startswith("R") else "2" if n.startswith("A") else "3"


def _kind(n: str) -> str:
    return {"R": "region", "A": "anchor", "Z": "surface"}.get(n[0], "structure")


def _write(tmp: Path, nodes: list[str], edges: list[str], *, auto_region: bool = True) -> Path:
    """auto_region이면 R1을 만들고 앵커를 전부 R1에 붙인다 (등급 규칙 충족용).

    엣지는 "child,relation,parent" 또는 "child,relation,parent,source".
    """
    nodes = list(nodes)
    edges = list(edges)
    if auto_region and not any(n.startswith("R") for n in nodes):
        nodes.append("R1")
        edges += [f"{n},part_of,R1" for n in nodes if n.startswith("A")]
    (tmp / "nodes.csv").write_text(
        NODE_COLS
        + "".join(
            f"{n},,,,x,,,,,{'1' if n.startswith('A') else ''},{_tier(n)},{_kind(n)}\n"
            for n in nodes
        ),
        encoding="utf-8",
    )
    (tmp / "edges.csv").write_text(
        "child,relation,parent,source\n"
        + "".join(f"{e}{'' if e.count(',') == 3 else ','}\n" for e in edges),
        encoding="utf-8",
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
        NODE_COLS + "R1,,,,x,,,,,,1,region\nR2,,,,x,,,,,,1,region\nA1,,,,x,,,,,1,3,structure\n",
        encoding="utf-8",
    )
    (tmp_path / "edges.csv").write_text(
        "child,relation,parent\nR1,part_of,R2\nA1,part_of,R1\n", encoding="utf-8"
    )
    with pytest.raises(OntologyError) as ei:
        load_ontology(tmp_path)
    msg = str(ei.value)
    assert "상부는 부모가 없어야" in msg and "is_anchor 불일치" in msg


def test_old_file_without_tier_and_kind_columns(tmp_path: Path):
    cols = "id,name_en,name_ko,region,structure_type,sctid,fma,definition_en,source,is_anchor\n"
    (tmp_path / "nodes.csv").write_text(cols + "n1,,,,x,,,,,\n", encoding="utf-8")
    (tmp_path / "edges.csv").write_text("child,relation,parent\n", encoding="utf-8")
    onto = load_ontology(tmp_path, strict=False)
    n = onto.get("n1")
    assert n.tier == 3 and n.kind == "structure" and n.laterality == "left_right"
    assert n.view == "none"


def test_located_in_requires_source_and_is_not_an_ancestor(tmp_path: Path):
    d = _write(
        tmp_path,
        ["A1", "Z1", "s1", "s2"],
        [
            "Z1,part_of,A1",
            "s1,part_of,A1",
            "s2,part_of,A1",
            "s1,located_in,Z1,해부학 교과서 p.1",
            "s2,located_in,Z1,",  # 출처 없음 → 조회에서 빠진다
        ],
    )
    onto = load_ontology(d)
    assert [n.id for n in onto.structures_under("Z1")] == ["s1"]
    assert onto.ancestors("s1") == ["A1", "R1"]  # located_in은 조상 계산에 안 들어간다


def test_located_in_wrong_direction_is_rejected(tmp_path: Path):
    d = _write(
        tmp_path, ["A1", "Z1", "s1"], ["Z1,part_of,A1", "s1,part_of,A1", "Z1,located_in,s1,x"]
    )
    with pytest.raises(OntologyError, match="structure → surface"):
        load_ontology(d)


def test_surface_must_hang_directly_under_anchor(tmp_path: Path):
    d = _write(tmp_path, ["A1", "s1", "Z1"], ["s1,part_of,A1", "Z1,part_of,s1"])
    with pytest.raises(OntologyError, match="직접 자식"):
        load_ontology(d)


# ---------------------------------------------------------------------------
# 유의어 검색 (aliases.csv) — 폼 입력 "복부"를 "배"로
def test_aliases_loaded_and_search_maps_common_words():
    onto = load_ontology()
    assert "복부" in onto.get("ANC:004").aliases
    top = lambda q: onto.search(q, limit=1)[0][0].id  # noqa: E731
    assert top("복부") == "ANC:004"  # 배
    assert top("옆구리") == "SUR:042"  # 허리 옆
    assert top("뒷목") == "SUR:012"  # 목 뒤·옆
    assert top("하복부") == "SUR:032"  # 아랫배
    assert top("명치") == "SUR:031"
    assert top("꼬리뼈") == "SUR:041"  # 디자이너 좌표표 '꼬리뼈' → 허리 가운데
    assert top("발가락") == "SUR:102"
    assert top("온몸") == "ANC:010"


def test_search_prefers_specific_zone_in_sentence_and_returns_empty_for_unknown():
    onto = load_ontology()
    hits = onto.search("왼쪽 아랫배가 아파요")
    assert hits and hits[0][0].id == "SUR:032"  # 아랫배가 배보다 먼저
    assert onto.search("무릅") == []  # 오타는 잡지 않는다(유의어 표 범위 밖)
    assert onto.search("") == []
    # 구조 노드(앞십자인대 등)는 검색 대상이 아니다 — 넓히기가 따로 맡는다
    assert all(n.kind in ("anchor", "surface") for n, _, _ in onto.search("인대"))


# ── 한/영 오타 복원 ────────────────────────────────────────────────


_CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_JONG = "ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"
_SPLIT = {
    "ㅘ": "ㅗㅏ",
    "ㅙ": "ㅗㅐ",
    "ㅚ": "ㅗㅣ",
    "ㅝ": "ㅜㅓ",
    "ㅞ": "ㅜㅔ",
    "ㅟ": "ㅜㅣ",
    "ㅢ": "ㅡㅣ",
    "ㄳ": "ㄱㅅ",
    "ㄵ": "ㄴㅈ",
    "ㄶ": "ㄴㅎ",
    "ㄺ": "ㄹㄱ",
    "ㄻ": "ㄹㅁ",
    "ㄼ": "ㄹㅂ",
    "ㄽ": "ㄹㅅ",
    "ㄾ": "ㄹㅌ",
    "ㄿ": "ㄹㅍ",
    "ㅀ": "ㄹㅎ",
    "ㅄ": "ㅂㅅ",
}


def _hangul_to_english_keys(text: str) -> str:
    """테스트 전용 역변환 — 한글을 영문 자판 키로 되돌린다. 복원기의 역함수."""
    jamo_to_key = {v: k for k, v in _KEY_TO_JAMO.items()}
    out = []
    for ch in text:
        if not ("가" <= ch <= "힣"):
            out.append(ch)
            continue
        code = ord(ch) - 0xAC00
        parts = [_CHO[code // 588], _JUNG[(code % 588) // 28]]
        if code % 28:
            parts.append(_JONG[code % 28 - 1])
        for jamo in parts:
            out.extend(jamo_to_key[x] for x in _SPLIT.get(jamo, jamo))
    return "".join(out)


@pytest.mark.parametrize(
    ("keys", "hangul"),
    [
        ("qo", "배"),
        ("qhrqn", "복부"),
        ("audcl", "명치"),
        ("duvrnfl", "옆구리"),
        ("anfmv", "무릎"),
        ("djRo", "어깨"),
        ("dkfotqo", "아랫배"),
        ("gjfl duv", "허리 옆"),  # 공백은 그대로
        ("djqtdj", "없어"),  # 복종성 ㅄ
        ("rkqtwl", "값지"),  # 복종성 뒤 자음 → 다음 초성
        ("ansdj", "문어"),  # 종성 ㄴ 뒤 ㅇ+모음 → 종성 이월
        ("gksrmf", "한글"),
        # shift 자리. 정규화(소문자)를 거치면 죽으므로 복원은 원문으로 해야 한다
        ("djRo", "어깨"),
        ("RhflQu", "꼬리뼈"),  # ㅃ은 받침이 될 수 없다 → 앞 글자를 확정하고 새로 시작
        ("dlQkf", "이빨"),
        ("qo dnlWhr", "배 위쪽"),  # ㅉ도 마찬가지
        ("wkdEkswl", "장딴지"),
    ],
)
def test_english_keys_to_hangul_composes_syllables(keys, hangul):
    assert english_keys_to_hangul(keys) == hangul


def test_looks_like_korean_typed_in_english_needs_letters_on_the_layout():
    assert looks_like_korean_typed_in_english("qo")
    assert not looks_like_korean_typed_in_english("")
    assert not looks_like_korean_typed_in_english("배")
    assert not looks_like_korean_typed_in_english("MRI")  # 자판 밖 대문자
    assert not looks_like_korean_typed_in_english("ct")  # 모음 자리 키가 없다


def test_search_restores_korean_typed_on_english_layout():
    onto = load_ontology()
    top = lambda q: onto.search(q, limit=1)[0][0].id  # noqa: E731
    assert top("qo") == "ANC:004"  # 배
    assert top("qhrqn") == "ANC:004"  # 복부 → 배
    assert top("audcl") == "SUR:031"  # 명치 → 윗배
    assert top("duvrnfl") == "SUR:042"  # 옆구리 → 허리 옆
    assert top("anfmv") == "SUR:091"  # 무릎
    assert top("djRo") == "SUR:051"  # 어깨 — shift 자리가 정규화로 죽지 않는다
    assert top("vkfRnacl") == "SUR:061"  # 팔꿈치
    assert top("RhflQu") == "SUR:041"  # 꼬리뼈 → 허리 가운데
    assert onto.search("zzzz") == []  # 복원해도 안 걸리면 그대로 0건


def test_every_anchor_and_zone_is_reachable_by_typing_its_name_on_the_english_layout():
    """앵커 9 · 구역 25의 이름과 별칭 전부가 자판 오타로도 상위 3위 안에 들어야 한다."""
    onto = load_ontology()
    misses = []
    for node in onto.nodes.values():
        if node.kind not in ("anchor", "surface"):
            continue
        for term in (node.name_ko, *node.aliases):
            keys = _hangul_to_english_keys(term)
            if any("가" <= c <= "힣" for c in keys):  # 역변환이 안 된 글자가 남으면 건너뛴다
                continue
            hits = onto.search(keys)
            if not any(h[0].id == node.id for h in hits[:3]):
                misses.append((term, keys, node.id))
    assert misses == []


def test_search_restoration_runs_only_when_there_is_no_hit():
    """복원은 0건일 때만 돈다 — 지금 결과가 나오는 질의는 그대로다(안드로이드 회귀 방지)."""
    onto = load_ontology()
    ids = lambda q: [n.id for n, _, _ in onto.search(q)]  # noqa: E731
    assert ids("배") == ["ANC:004", "SUR:032", "SUR:031"]
    assert ids("복부") == ["ANC:004", "SUR:031", "SUR:032"]
    assert ids("무릎") == ["SUR:091"]
    assert ids("허리 옆") == ["SUR:042", "SUR:041"]
    assert onto.search("무릅") == []  # 한글 오타는 여전히 안 잡는다


def test_search_on_an_anchor_name_lists_its_zones():
    """앵커 이름만 아는 사람도 구역을 본다 — "다리" → 무릎·종아리·발목·발."""
    onto = load_ontology()
    for anchor in [n for n in onto.nodes.values() if n.kind == "anchor"]:
        found = {n.id for n, _, _ in onto.search(anchor.name_ko, limit=20)}
        missing = [z.display_name for z in onto.zones(anchor.id) if z.id not in found]
        assert missing == [], f"{anchor.name_ko}: {missing}"

    ids = lambda q: [n.id for n, _, _ in onto.search(q, limit=20)]  # noqa: E731
    assert ids("머리") == ["ANC:001", "SUR:001", "SUR:002", "SUR:003", "SUR:004", "SUR:005"]
    # 딸려 온 구역은 점수 0 — 직접 매칭과 구분된다. 앵커·직접 매칭이 항상 앞
    scores = {n.display_name: s for n, _, s in onto.search("다리", limit=20)}
    assert scores["다리"] == 3 and scores["무릎"] == 0


def test_zone_expansion_does_not_reorder_or_drop_direct_matches():
    onto = load_ontology()
    ids = lambda q: [n.id for n, _, _ in onto.search(q)]  # noqa: E731
    assert ids("배")[:3] == ["ANC:004", "SUR:032", "SUR:031"]  # 구역이 이미 직접 걸린 경우 그대로
    assert ids("무릎") == ["SUR:091"]  # 앵커가 없으면 붙는 것도 없다
    assert ids("왼쪽 아랫배가 아파요")[0] == "SUR:032"
