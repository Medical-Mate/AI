"""백엔드가 보낸 `state`가 **엔진이 만들 수 있는 모양인가**를 본다.

AI 서버는 무상태라 상태를 백엔드가 들고 다닌다(`docs/ai-design.md` §7). 그 말은 상태가
왕복 중에 변형될 수 있다는 뜻이기도 하다 — 필드를 재조립하거나, DB 컬럼으로 펴서 저장했다가
다시 합치거나, 두 요청의 상태가 섞이거나. 그렇게 들어온 상태는 **스키마는 통과한다.**
타입이 맞기 때문이다. 대신 엔진이 결코 만들지 않는 조합이 된다.

여기서 보는 것은 **모양이 아니라 내용의 가능성**이다. 그래서 422가 아니라 400이다.
- 422 = 스키마 위반. pydantic이 낸다
- 400 = 스키마는 맞는데 우리가 만들 수 없는 상태

**엄격하게 보지 않는다.** 막는 것은 "엔진이 절대 만들지 않는 조합"뿐이고, 조금이라도 정상
경로가 있는 조합은 통과시킨다. 문답을 진행 중인 환자의 요청을 우리 추측으로 끊는 쪽이
잘못된 카드 한 장보다 나쁘다.

── 이 가드가 오늘 사고를 막았을까: 아니다 ─────────────────────────────────
2026-09-11의 `title`·`department_guidance`가 `null`로 보인 일은 백엔드가 **`state.card`를
읽어서**였다. 상태 자체는 멀쩡했고 응답의 다른 자리를 본 것이라, 이 가드는 조용히 통과시킨다.
그건 계약 문서와 `/health`가 맡는다. 이 가드가 잡는 것은 **다음번의 다른 사고**다 —
상태를 재조립해서 보내는 경우.
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Any

from medimate.dialog.questions import SITE_PRESELECTED
from medimate.dialog.state import SessionState
from medimate.schema.card import FieldStatus

SITE_AXIS = "site"


def check_state(state: SessionState) -> list[str]:
    """어긋난 것들을 메시지 리스트로. 빈 리스트면 통과."""
    problems: list[str] = []
    card = state.card

    # 규칙 A — 안 물은 축에 값이 있다.
    #
    # 엔진은 값을 넣을 때 반드시 status를 함께 옮긴다(`filled`·`unknown`·`skipped`·`ambiguous`).
    # `not_asked`인 채로 값만 있는 축은 **엔진을 거치지 않고 값이 들어왔다**는 뜻이다.
    # 이걸 통과시키면 그 값이 카드에 실려 의사에게 가는데, 근거 없는 값을 만들지 않는다는
    # 원칙(CLAUDE.md)이 상태 왕복에서 새는 자리가 된다.
    #
    # `ambiguous`에 값이 있는 것은 **정상이다** — 되물으면서 잠정 값을 들고 있다
    # (`llm/base.py` 참고). `unknown`·`skipped`에 값이 남아 있는 것도 막지 않는다.
    for axis, entry in card.axes.items():
        if entry.status == FieldStatus.NOT_ASKED and entry.value is not None:
            problems.append(f"axes.{axis}: status가 not_asked인데 value가 있음")

    # 규칙 B — 온톨로지 노드로 시작한 세션인데 선택 기록이 없다.
    #
    # `site_selection`이 비면 **제목과 진료과 안내가 조용히 사라진다** — `export.py`의
    # `_title`·`_department_guidance`가 둘 다 여기서 나온다. 200이 나가고 카드도 멀쩡해 보여서
    # 아무도 모른다. 그래서 400으로 세운다.
    #
    # **`[부위 선택]` 표시만으로는 판정하지 않는다.** 처음에 그렇게 짰다가 온디바이스 테스트
    # 셋이 잡아 줬다 — 세션 시작에는 길이 둘인데(`site_node_id` · `site_label`) 표시는 같고
    # `site_selection`은 앞의 길에서만 생긴다. 표시로 막으면 라벨로 시작한 정상 세션이
    # 전부 400이 된다.
    #
    # 그래서 앞의 길을 **거쳐야만 생기는 자국**을 본다: `provenance.ontology_snapshot`.
    # `site_selection`과 같은 자리에서 같이 들어가므로, 스냅샷만 있고 기록이 없으면
    # 왕복 중에 떨어진 것이다.
    if state.spec == "previsit":
        site = card.axes.get(SITE_AXIS)
        prov = card.provenance
        from_ontology = prov is not None and prov.ontology_snapshot is not None
        if site is not None and _has_selection_tag(site.evidence) and from_ontology:
            if getattr(card, "site_selection", None) is None:
                problems.append("site_selection 없음 — site 축은 인체도 선택으로 채워져 있음")

    return problems


def _has_selection_tag(evidence: Any) -> bool:
    """evidence는 리스트다. 문자열로 온 것도 받아 준다(백엔드가 펴서 저장했을 수 있다)."""
    if isinstance(evidence, str):
        return SITE_PRESELECTED in evidence
    if not isinstance(evidence, list):
        return False
    return any(SITE_PRESELECTED in str(e) for e in evidence)
