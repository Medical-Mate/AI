"""인체도 선택 → 문진 세션의 부위.

앱 흐름 1l → 1c: 환자가 인체도에서 앵커(또는 그 아래 구역)를 짚고, 좌우가 있는 부위면 좌/우를 고르고
문진에 들어온다. 여기서는 그 선택을 **환자에게 보이는 라벨 하나**와 **진료과 안내**로 바꾼다.

- 라벨은 SITE 축의 값이 된다. evidence는 발화가 아니라 UI 선택임을 표시한다(engine.SITE_PRESELECTED)
- 진료과 안내는 노드에 적힌 값을 그대로 옮긴다. 구역에 있으면 구역, 없으면 앵커. 고르지 않는다
  (docs/decisions/2026-09-04-department-guidance.md).
  증상 축은 여기 들어오지 않는다 — 시그니처에 없다
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from medimate.ontology import Ontology, UnknownNodeError

SIDES = {"left": "왼쪽", "right": "오른쪽", "both": "양쪽"}


def normalize_side(side: str | None) -> str | None:
    """좌우 값을 소문자로 접는다. 대소문자를 가리지 않는다.

    2026-09-11: 백엔드가 `LEFT`/`RIGHT`/`BOTH` 대문자로 구현했는데 우리는 소문자만 받아
    422를 냈다. 그쪽 테스트가 `side: null`이라 안 걸렸고, **좌우가 있는 부위(34곳 중 21곳)를
    처음 보내는 순간 터지는** 상태였다. 계약 문서에 소문자라는 명시도 약했다.

    받는 쪽을 넓히는 게 맞다 — 좁히면 상대가 고쳐야 하고, 넓히면 기존 호출이 그대로 돈다.
    `Left` 같은 혼합도 받는다. 값을 만드는 곳은 여전히 여기 하나라 카드에는 소문자로만 남는다.
    """
    if side is None:
        return None
    return side.strip().lower()


class SiteSelection(BaseModel):
    """카드에 남는 선택 기록. 라벨·노드·진료과 안내. 나중에 온톨로지가 바뀌어도 그때 값이 남는다."""

    model_config = ConfigDict(extra="forbid")

    node_id: str  # 짚은 노드(구역 또는 앵커)
    anchor_id: str  # 그 노드가 속한 앵커(앵커를 짚었으면 자기 자신)
    side: str | None = None  # left / right / both / None
    label: str  # 환자에게 보이는 부위 이름. "왼쪽 무릎", "허리 가운데"
    departments: list[str]  # 진료과 안내. 노드에 적힌 순서 그대로, 의미 있는 순서 아님
    departments_source: str  # "팀 결정 …, 자문 확인 전". 인용이 아님을 값마다 명시
    ontology_snapshot: str


def resolve_site(onto: Ontology, node_id: str, side: str | None = None) -> SiteSelection:
    """선택된 노드(+좌우)를 라벨과 진료과 안내로. 잘못된 선택은 ValueError.

    - 앵커나 구역만 받는다. 구조(차트 용어)는 환자가 짚는 대상이 아니다
    - 좌우는 laterality가 left_right인 노드에만 붙일 수 있다
    """
    try:
        node = onto.get(node_id)
    except UnknownNodeError as e:
        raise ValueError(f"알 수 없는 부위: {node_id}") from e
    if node.kind not in ("anchor", "surface"):
        raise ValueError(f"부위 선택은 앵커·구역만 가능: {node_id} (kind={node.kind})")

    anchor_id = node.id if node.kind == "anchor" else onto.anchor_of(node.id)
    if anchor_id is None:
        raise ValueError(f"앵커에 닿지 않는 구역: {node_id}")
    anchor = onto.get(anchor_id)

    side = normalize_side(side)
    if side is not None:
        if side not in SIDES:
            raise ValueError(f"side는 {sorted(SIDES)} 중 하나: {side}")
        if node.laterality != "left_right":
            raise ValueError(f"{node.display_name}에는 좌우가 없다")

    label = node.display_name
    if side:
        label = f"{SIDES[side]} {label}"

    # 진료과: 구역 값이 있으면 구역, 없으면 앵커. 둘 다 없으면 빈 목록(안내 없음)
    src = node if node.departments else anchor
    return SiteSelection(
        node_id=node.id,
        anchor_id=anchor.id,
        side=side,
        label=label,
        departments=list(src.departments),
        departments_source=src.departments_source,
        ontology_snapshot=onto.snapshot_id,
    )


def find_site_by_name(onto: Ontology, name: str) -> str | None:
    """표시 이름으로 앵커·구역 ID를 찾는다(터미널 클라이언트용). 정확히 하나만 맞을 때 돌려준다."""
    name = name.strip()
    hits = [
        n.id
        for n in onto.nodes.values()
        if n.kind in ("anchor", "surface") and n.display_name == name
    ]
    return hits[0] if len(hits) == 1 else None
