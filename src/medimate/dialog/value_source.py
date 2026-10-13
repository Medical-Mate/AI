"""진료 전 값 출처 검증(가드 규칙 10) — 모델이 쓴 값이 **그 발화에서 나왔는가**.

근거 필터(규칙 2)는 evidence가 발화에 있는지만 본다. 값은 따로 쓰이므로 근거와 무관한 값이
통과했다. 폰 1.7B 이어 돌리기(카드 100장, 2026-10-13)에서 나온 것:

- 깨진 토큰: "아침에 좀 뻣뻣해요" → `립구름`, "묵직하고 … 뜨끔해요" → `ruzuku`
- 부정 뒤집기: "열이 나지는 않아요" → `열이 나요`
- 예시 베끼기: "타이레놀 먹으면 좀 나아요" → `타이레놀 먹으면 더 심하고 누우면 괜찮아요`
- 없던 숫자: "서서히" → `두 달 전부터`

판정은 진료 후 값 검증기(`span/verify.verify_value`)를 그대로 쓴다 — 낱말 출처·숫자·부정·용어.
걸리면 값을 **버리지 않고 환자가 한 말로 바꾼다.** 같은 뜻을 다른 낱말로 쓴 값(`악화`, `증가`)도
여기서 걸리지만 손해는 값이 길어지는 것뿐이다. 그래서 허용 표는 방향이 없는 낱말만 둔다 —
`악화`·`호전`을 허용하면 "점점 나빠져요" → `호전`도 지난다.

진료 후와 다르게 하는 것 둘:
- 길이(`too_long`)는 보지 않는다. 진료 전 값은 환자 표현이라 길 수 있다
- 고쳐 말하기("3일이 아니라 3주예요")의 앞부분은 대조에서 뺀다. 남기면 `3주`가 부정 절에서
  왔다고 보아 걸린다
"""

from __future__ import annotations

import re

# "X이/가 아니라" — 고쳐 말하기의 버린 쪽. 대조할 발화에서만 뺀다(값·근거는 그대로)
_CORRECTED = re.compile(r"[^\s,.]+\s*(?:이|가)?\s*아니(?:라|고)\s*,?\s*")
# 진료 전 검증에서 보지 않는 이유
_IGNORED = {"too_long"}


def value_problems(value: str, utterance: str, axis: str, kiwi=None) -> list[str]:
    """값이 발화에서 나오지 않았으면 이유 목록, 나왔으면 빈 목록.

    Kiwi(nlp 그룹)가 없으면 검증을 못 하므로 `["no_verifier"]` — 검증 안 된 값은 쓰지 않는다.
    """
    try:
        from medimate.span.verify import verify_value
    except ImportError:
        return ["no_verifier"]
    segment = _CORRECTED.sub("", utterance).strip() or utterance
    try:
        verdict = verify_value(value, segment, axis, kiwi=kiwi)
    except ImportError:
        return ["no_verifier"]
    return [r for r in verdict.reasons if r.split(":")[0] not in _IGNORED]


def patient_words(utterance: str) -> str:
    """값 대신 쓰는 환자 말. 어미는 고치지 않는다 — 다듬다가 뜻을 바꾸는 것이 제일 나쁘다."""
    return utterance.strip().rstrip(".!~ ")
