# ruff: noqa: E501
"""진료 후 메모 — 조각내기 + 라벨을 한 번에 (`memo-v5`, 서버 전용).

v4까지는 **문장을 우리가 규칙으로 나누고** LLM은 라벨만 붙였다. 문장 번호가 라벨의 주소라서
분리는 결정론이어야 했다. 그런데 사람 입력은 `"일주일치약처방이주일후재방문"`처럼 띄어쓰기도
구두점도 없이 온다 — 규칙은 늘 하나씩 늦었다(2026-09-15 하루에 구멍 여섯).

v5는 LLM이 **원문에서 글자 그대로 잘라낸 조각**과 라벨을 함께 낸다. 코드가 조각을 순서대로
이어 원문과 같은지 검증한다(공백만 무시). 안 맞으면 버리고 규칙 분리 + v4 분류로 폴백한다.
주소 문제는 구조로 푼다 — 라벨을 되보낼 때 번호 대신 **조각 자체**를 같이 보낸다.

공용 `memo-small-v4`(폰 짝)는 건드리지 않는다.
"""

from __future__ import annotations

PROMPT_VERSION = "memo-v6"

LABELS = (
    "findings",
    "tests",
    "medication_instructions",
    "lifestyle_instructions",
    "follow_up",
    "none",
)

SCHEMA: dict = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "label": {"type": "string", "enum": list(LABELS)},
                },
                "required": ["text", "label"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["segments"],
    "additionalProperties": False,
}

_SYSTEM = """너는 환자가 진료실에서 들은 말을 적은 메모를 **진료 내용 단위 조각**으로 나누고 조각마다 라벨 하나를 붙이는 분류기다. 새 말을 만들지 않는다. 고치지 않는다. 요약하지 않는다.

조각 규칙
- 조각의 text는 메모에서 **글자 그대로 잘라낸 것**이다. 한 글자도 바꾸지 않는다. 띄어쓰기·구두점도 원문 그대로.
- 조각을 **순서대로 전부 이어 붙이면 원문 메모**가 된다. 빠뜨리지 않고, 덧붙이지 않고, 겹치지 않는다.
- 한 조각에는 주제 하나만. 주제가 바뀌는 자리에서 자른다 — 마침표·쉼표가 없어도, 띄어쓰기가 없어도 자른다.
- 같은 주제가 이어지면 한 조각으로 둔다. 짧다고 붙이고 길다고 나누지 않는다.

라벨
- findings: 의사가 말한 상태·소견·병명·안심. "위염 초기래요", "뼈는 괜찮대요", "혈압이 좀 높다고"
- tests: 검사·영상·결과 안내·검사 예약. "피검사 했어요", "결과는 다음에", "MRI 예약은 다음 주"
- medication_instructions: 약·처방·복용법, 병원에서 한 주사·처치. "2주분 처방", "일주일치약처방", "하루 두 번", "주사 맞았어요"
- lifestyle_instructions: 집에서 지키라는 생활 지시·금지 — 식이·음료·활동·운동·자세·자가관리. "커피 줄이라고", "계단은 피하라고", "무리하지 말라고", "얼음찜질하라고"
- follow_up: 다시 오는 시점·조건. "2주 뒤에 오라고", "이주일후재방문", "안 좋아지면 바로 오래요"
- none: 진료 내용이 아닌 것만. 인사, 감상, 병원이 붐볐다, 제목만 있는 줄

판단 순서
1. 다시 오라는 말(오라고/보자고/재진/재방문)이 있으면 follow_up.
2. 약·처방·복용법·주사·처치가 있으면 medication_instructions. 약과 생활 지시가 한 조각에 같이 있으면 medication_instructions.
2-1. 약 얘기 없이 "~하라고/~하지 말라고/금지/피하라고/줄이라고"만 있으면 lifestyle_instructions.
3. 검사·영상·결과 얘기면 tests. 검사 날짜·예약도 tests.
4. 그 외 의사가 몸 상태에 대해 한 말은 findings.
5. 진료와 무관한 말만 none.

예시 1
메모: 위염 초기라고 하셨어요. 혈액검사 했고 결과는 다음에 알려준대요. 약은 2주분이고 커피랑 매운 거 줄이래요. 2주 뒤에 다시 오라고 하셨어요. 병원이 너무 붐볐다
{"segments":[{"text":"위염 초기라고 하셨어요.","label":"findings"},{"text":"혈액검사 했고 결과는 다음에 알려준대요.","label":"tests"},{"text":"약은 2주분이고 커피랑 매운 거 줄이래요.","label":"medication_instructions"},{"text":"2주 뒤에 다시 오라고 하셨어요.","label":"follow_up"},{"text":"병원이 너무 붐볐다","label":"none"}]}

예시 2 (띄어쓰기·구두점 없음)
메모: 일주일치약처방이주일후재방문
{"segments":[{"text":"일주일치약처방","label":"medication_instructions"},{"text":"이주일후재방문","label":"follow_up"}]}

예시 3
메모: 발목 인대가 조금 늘어났대요 2주 정도 붕대 감고 다니래요 무리하지 말라고 안 좋아지면 바로 오래요
{"segments":[{"text":"발목 인대가 조금 늘어났대요","label":"findings"},{"text":"2주 정도 붕대 감고 다니래요","label":"medication_instructions"},{"text":"무리하지 말라고","label":"lifestyle_instructions"},{"text":"안 좋아지면 바로 오래요","label":"follow_up"}]}"""


def system_prompt() -> str:
    return _SYSTEM


def user_message(memo: str) -> str:
    return f"메모:\n{memo}\nJSON:"
