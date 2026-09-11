"""질문 후보 순위 — 생성은 제한하지 않고, 가중치로 정렬해 위에서 잘라 낸다.

왜 프롬프트가 아니라 여기서 하는가.

2026-09-11에 두 번 확인한 것이 있다. **프롬프트에 "꼭 넣어라"를 걸면 그 자리가 템플릿이 된다.**
- "환자가 덧붙인 말에서 하나 만든다"(의무) → 전할 말이 재료 있는 카드 52장 전부에서 나왔다(100%)
- 축별 끝맺음을 문자열로 제시 → 카드 100장 중 58장·49장·44장이 같은 꼬리로 굳었다

그래서 "약품 질문을 꼭 넣어라"로 밀면 `OO 계속 먹어도 되나요?`가 34장에 똑같이 박힌다.
대신 **생성은 느슨하게 두고 고르는 쪽을 결정론으로** 한다. 가중치를 바꿔도 호출이 0이므로
저장된 결과로 재정렬만 하면 된다(`scripts/tune_candidate_weights.py`).

한계는 분명하다 — **생성에 없는 것은 랭커가 만들어 내지 못한다.** 복용약 질문이 재료 있는
34장 중 1장에서만 나오는 상태라면, 가중치를 아무리 올려도 고를 게 없다. 프롬프트에서
"허용"을 "재료가 있으면 후보에 넣는다"로 올리는 일이 함께 필요하다(예시 문구는 주지 않는다).
"""

from __future__ import annotations

WEIGHTS_VERSION = "rank-v1"

# 원칙: **카드를 보면 의사가 바로 아는 것은 질문 가치가 낮고, 카드에 있어도 의사가 먼저
# 안 물어볼 수 있는 것은 높다.** 부위·시작·심각도는 카드 상단에 크게 있으니 질문으로 만들
# 이유가 적고, 복용약·알러지·기저질환은 환자가 꺼내야 처방 안전에 반영된다.
# 블라인드 판정 메모가 약 관련을 세 번 지목했다(PC07·PC09·PC10).
WEIGHTS: dict[str, float] = {
    # 환자가 꺼내야 하는 것 — 처방 안전에 직결
    "medications": 3.0,
    "allergies": 3.0,
    "conditions": 2.5,
    # 환자가 이미 궁금해한 것(전할 말 턴이 없어지면 이 source는 사라진다)
    "patient_message": 2.5,
    # 증상 축 중 카드 고유성이 높은 것
    "exacerbating": 2.0,
    "associated": 2.0,
    "time_course": 1.8,
    # 카드 "양상"에 이미 보이는 것
    "character": 1.2,
    "radiation": 1.0,
    # 카드 상단에 크게 보이는 것 — 의사가 읽으면 된다
    "onset": 0.6,
    "site": 0.4,
    "severity": 0.4,
    # 어느 진료에나 통하는 것. 앱 고정 칩이 따로 있다
    "general": 0.3,
}
DEFAULT_WEIGHT = 1.0  # 모르는 source는 중간

# 같은 source가 상위에 몰리는 것을 막는다. 와이어프레임의 3개도 서로 다른 종류다
# (① 검사 ② 약 ③ 일반). 하드 금지가 아니라 감점이라 재료가 한 축뿐이면 그 축으로 채운다.
REPEAT_PENALTY = 0.7  # 같은 source가 두 번째로 뽑힐 때 곱하는 값. 세 번째는 제곱


def score_items(
    items: list[dict],
    weights: dict[str, float] | None = None,
    repeat_penalty: float = REPEAT_PENALTY,
) -> list[dict]:
    """항목마다 `rank_score`와 `rank`를 붙여 정렬한 새 리스트를 돌려준다.

    원본을 건드리지 않는다. 같은 점수면 모델이 낸 순서를 유지한다(안정 정렬).
    `repeat_penalty`는 탐욕적으로 하나씩 고르면서 적용한다 — 이미 뽑힌 source가
    다시 나오면 점수를 깎는다.
    """
    w = weights if weights is not None else WEIGHTS
    pool = [dict(it) for it in items]
    for i, it in enumerate(pool):
        it["_i"] = i
        it["_base"] = w.get(it.get("source", ""), DEFAULT_WEIGHT)

    picked: list[dict] = []
    seen: dict[str, int] = {}
    while pool:
        best_i, best_s = None, None
        for idx, it in enumerate(pool):
            s = it["_base"] * (repeat_penalty ** seen.get(it.get("source", ""), 0))
            # 같은 점수면 원래 순서가 앞선 것을 고른다
            if best_s is None or s > best_s or (s == best_s and it["_i"] < pool[best_i]["_i"]):
                best_i, best_s = idx, s
        it = pool.pop(best_i)
        seen[it.get("source", "")] = seen.get(it.get("source", ""), 0) + 1
        it["rank_score"] = round(best_s, 4)
        it["rank"] = len(picked) + 1
        picked.append(it)

    for it in picked:
        it.pop("_i", None)
        it.pop("_base", None)
    return picked


def top_candidates(
    items: list[dict],
    top: int = 3,
    weights: dict[str, float] | None = None,
    repeat_penalty: float = REPEAT_PENALTY,
) -> list[dict]:
    """가중치로 정렬해 위에서 `top`개까지. 생성 개수는 제한하지 않는다.

    재료가 적어 후보가 `top`보다 적게 나오면 있는 만큼만 낸다 — 억지로 채우지 않는다
    (카드 100장 중 7장이 2개였다).
    """
    return score_items(items, weights, repeat_penalty)[:top]
