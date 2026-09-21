# ruff: noqa: E501
"""값 span 선택 프롬프트 — 모델은 **후보 ID 하나**만 고른다(`span-select-v1`).

문자열을 만들지 않는다. 후보는 코드가 만들었고(span/candidates.py), 값의 모양 규칙은 gold 검토(2026-09-21)에서
정해진 것을 그대로 적었다(span/canon.py 머리말). 선택기가 규칙을 알아야 `혈액검사`와 `추후 혈액검사 결과 안내`
사이에서 고를 수 있다.

바꾸면 PROMPT_VERSION을 올리고 비교 대상 모델을 전부 다시 돌린다.
"""

from __future__ import annotations

from medimate.span.candidates import CandidateSet

PROMPT_VERSION = "span-select-v1"

AXIS_KO = {
    "findings": "소견",
    "tests": "검사",
    "medication_instructions": "약",
    "follow_up": "재방문",
}

_SYSTEM = """너는 환자가 진료실에서 들은 말을 적은 메모 한 조각에서, 기록 카드의 한 칸에 들어갈 **값**을 고르는 선택기다.
값은 주어진 후보 중 하나다. 새 문장을 쓰지 않는다. 후보의 ID만 답한다. 맞는 후보가 없으면 NONE.

카드 값의 모양(팀이 정한 것. 이대로 고른다)
- 소견 칸: 병명·용어가 있으면 용어만(앞의 좌우·부위 수식은 붙이고, 뒤는 "초기" 같은 단계 낱말까지만). 수치·변화를 말하는 서술문이면 `-음`으로 끝나는 서술문(혈압이 좀 높음, 귀에 물이 찼음). "괜찮다", "정상", "특별한 건 없다", "필요 없다" 같은 안심·부정 소견은 NONE.
- 약 칸: 약 이름과 용법·기간을 한 덩어리로(항생제 5일, 인공눈물 하루 4번). 조건·시점이 있으면 앞에(아침에 혈압약, 38도 넘으면 해열제). "처방"이라는 말은 뺀다(연고 처방 → 연고). 약을 안 줬다, 아직 안 먹는다, 주사만 맞았다, 처치만 했다(스케일링만 했음)는 NONE. 처치를 받으라고 한 것은 처치명(충치 치료).
- 검사 칸: 검사를 **했으면** 검사명만(혈액검사). 결과 얘기만 있으면 [시점] [검사] 결과 안내(추후 피검사 결과 안내). 예약·예정이면 [시점] [검사] 예약/예정. 시점을 앞에 둘 때 "뒤"는 "후"로. "이상 없음" 같은 결과 서술은 조각 그대로.
- 재방문 칸: 시점만 있으면 시점 그대로(2주 뒤, 다음 달 15일). 조건이 있으면 [조건] 재방문 필요(안 좋아지면 재방문 필요, 악화 시 재방문 필요). 시점도 조건도 없이 다시 오라고만 했으면 "재방문 필요". 결과 보고 이야기하자는 말만 있으면 NONE.
- 공통: 짧은 쪽을 고른다. 어미·조사·"~라고 하셨어요" 같은 보고 표현이 붙은 후보는 고르지 않는다. 후보 중 어느 것도 위 모양이 아니면 NONE.

출력은 JSON 하나: {"choice": "C02"} 또는 {"choice": "NONE"}. 다른 말은 쓰지 않는다."""


def system_prompt() -> str:
    return _SYSTEM


def user_message(cset: CandidateSet, axis: str) -> str:
    lines = [f"칸: {AXIS_KO.get(axis, axis)}", f"조각: {cset.segment}"]
    if cset.raw and cset.raw != cset.segment:
        lines.append(f"(원문: {cset.raw})")
    lines.append("후보:")
    for c in cset.candidates:
        lines.append(f"  {c.id}: {c.text}")
    lines.append("  NONE: 맞는 후보 없음")
    lines.append("JSON:")
    return "\n".join(lines)


def schema(cset: CandidateSet) -> dict:
    return {
        "type": "object",
        "properties": {"choice": {"type": "string", "enum": cset.ids() + ["NONE"]}},
        "required": ["choice"],
        "additionalProperties": False,
    }


# Jev(Choice)용 축별 지시. 위 _SYSTEM의 같은 규칙을 축 하나만 잘라 짧게. 바꾸면 PROMPT_VERSION을 올린다
JEV_INSTRUCTIONS = {
    "findings": (
        "기록 카드의 '소견' 칸 값을 후보 중에서 고른다. 병명·용어가 있으면 용어만(앞의 좌우·부위 수식은 붙이고, "
        "뒤는 '초기' 같은 단계 낱말까지만). 수치·변화 서술이면 '-음'으로 끝나는 서술문(혈압이 좀 높음). "
        "괜찮다·정상·특별한 건 없다·필요 없다 같은 안심·부정 소견은 NONE. 주어·조사·'~라고 하셨어요'가 붙은 긴 후보는 고르지 않는다."
    ),
    "medication_instructions": (
        "기록 카드의 '약' 칸 값을 후보 중에서 고른다. 약 이름과 용법·기간을 한 덩어리로(항생제 5일). 조건·시점이 있으면 앞에(아침에 혈압약). "
        "'처방'이라는 말은 뺀다(연고). 약을 안 줬다·아직 안 먹는다·주사만 맞았다·처치만 했다는 NONE. 처치를 받으라고 한 것은 처치명. "
        "'지금은', '~은/는' 같은 주어·조사가 붙은 후보는 고르지 않는다."
    ),
    "tests": (
        "기록 카드의 '검사' 칸 값을 후보 중에서 고른다. 검사를 이미 했으면 검사명만(혈액검사). 결과 얘기만 있으면 '[시점] [검사] 결과 안내'. "
        "예약·예정이면 '[시점] [검사] 예약/예정'. 시점만 있으면 시점(다음 주 화요일). '이상 없음' 같은 결과 서술은 조각 그대로. "
        "조사·어미가 붙은 긴 후보는 고르지 않는다."
    ),
    "follow_up": (
        "기록 카드의 '재방문' 칸 값을 후보 중에서 고른다. 시점만 있으면 시점 그대로(2주 뒤). 조건이 있으면 '[조건] 재방문 필요'. "
        "시점도 조건도 없이 다시 오라고만 했으면 '재방문 필요'. 안 와도 된다·결과 보고 이야기하자는 NONE. 어미가 붙은 긴 후보는 고르지 않는다."
    ),
}


def jev_state(cset: CandidateSet, axis: str) -> dict:
    st = {"칸": AXIS_KO.get(axis, axis), "조각": cset.segment}
    if cset.raw and cset.raw != cset.segment:
        st["원문"] = cset.raw
    return st


def jev_criteria(cset: CandidateSet) -> dict[str, str]:
    crit = {c.id: c.text for c in cset.candidates}
    crit["NONE"] = "맞는 후보 없음"
    return crit
