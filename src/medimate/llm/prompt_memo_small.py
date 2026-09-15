# ruff: noqa: E501  — 예시 JSON 한 줄은 의도적으로 길다
"""진료 후 메모 문장 분류 프롬프트 — `memo-small-v4` (소형·서버 공용).

v2 (2026-09-07, 1.7B 52% → ): "~래요/~대요/~하셨어요"는 의사의 말을 옮긴 것이라 대개 findings·instructions인데
none으로 흘렸다. "피하라고/하라고"를 follow_up으로 착각했다. 어미와 지시 예시를 늘리고, none을 좁혔다.
v3: 번호(i)를 없애고 **문장 순서대로 라벨 배열**만 낸다(v2 83%에서 2·3번이 서로 바뀌는 오류). 길이는 스키마로 고정. → 64%로 후퇴.
v4: 출력을 번호 키 객체 {"0": 라벨, "1": 라벨, …}로. 번호 키가 전부 필수(스키마). 문장 앵커 + 구조 보장.
v5 (2026-09-15): 약과 **지침(lifestyle_instructions)** 을 나눈다. 약 칸이 생활 지시로 길어지고 "약" 아래 약이 아닌 말이 찍혔다.
    **폰 짝이 깨진다** — 폰이 이 프롬프트를 쓰게 되면 assets를 v5로 같이 올려야 한다(지금 폰 경로는 안 쓰인다).

문장마다 라벨 하나. 값을 만들지 않는다. 문장은 이미 우리가 나눠 번호를 붙여 준다.
라벨: findings(소견) / tests(검사) / medication_instructions(약) / lifestyle_instructions(지침) / follow_up(재방문) / none(해당 없음)
"""

from __future__ import annotations

from collections.abc import Sequence

PROMPT_VERSION = "memo-small-v5"

_SYSTEM = """너는 환자가 진료실에서 들은 말을 적은 메모를 네 묶음으로 나누는 분류기다. 새 문장을 만들지 않는다. 문장을 고치지 않는다. 문장 번호마다 라벨 하나를 낸다. 번호를 빠뜨리지 않는다.

라벨
- findings: 의사가 말한 상태·소견·병명·안심. "위염 초기래요", "인대가 늘어났다고 하셨어요", "뼈는 괜찮대요", "걱정 안 해도 된다고", "혈압이 좀 높다고"
- tests: 검사·영상·결과 안내·검사 예약. "피검사 했어요", "결과는 다음에 알려준대요", "MRI 예약은 다음 주 화요일", "결과 보고 정한다고", "초음파 봤는데 파열은 아니래요"
- medication_instructions: 약·처방·복용법, 병원에서 한 주사·처치. "2주분 처방", "하루 두 번", "약은 안 주셨어요", "약은 아직", "주사 한 대 맞았어요", "붕대 감고 다니래요", "연고 바르라고"
- lifestyle_instructions: 집에서 지키라는 생활 지시·금지 — 식이·음료·활동·운동·자세·자가관리. "커피 줄이라고", "짜게 먹지 말라고", "계단은 피하라고", "수영 금지", "무거운 거 들지 말라고", "얼음찜질하라고", "무리하지 말라고", "운동 주 3회 하라고"
- follow_up: 다시 오는 시점·조건. "2주 뒤에 오라고", "다음 달 15일에 오라고", "안 좋아지면 바로 오래요", "좋아지면 안 와도 된대요"
- none: 진료 내용이 아닌 것만. 인사, 감상, 병원이 붐볐다, 점심 못 먹었다, 제목만 있는 줄

판단 순서
1. 다시 오라는 말(오라고/보자고/재진/재방문)이 있으면 follow_up.
2. 약·처방·복용법·주사·처치가 있으면 medication_instructions. 한 문장에 약과 생활 지시가 같이 있으면("약은 2주분이고 커피 줄이래요") medication_instructions.
2-1. 약 얘기 없이 "~하라고/~하지 말라고/금지/피하라고/줄이라고"만 있으면 lifestyle_instructions. "피하라고"는 지침이다. 재방문이 아니다.
3. 검사·영상·결과 얘기면 tests. 검사 날짜·예약도 tests.
4. 그 외 의사가 몸 상태에 대해 한 말("~래요", "~대요", "~라고 하셨어요", "괜찮다고")은 findings.
5. 진료와 무관한 말만 none. 짧아도 진료 얘기면 none이 아니다.
한 문장에 둘이 섞이면 위 순서에서 먼저 걸리는 라벨.

예시 1. 문장:
0: 위염 초기라고 하셨어요
1: 혈액검사 했고 결과는 다음에 알려준대요
2: 약은 2주분이고 커피랑 매운 거 줄이래요
3: 2주 뒤에 다시 오라고 하셨어요
4: 병원이 너무 붐볐다
{"0":"findings","1":"tests","2":"medication_instructions","3":"follow_up","4":"none"}

예시 2. 문장:
0: 발목 인대가 조금 늘어났대요
1: 엑스레이는 이상 없대요
2: 2주 정도 붕대 감고 다니래요
3: 무리하지 말라고
4: 안 좋아지면 바로 오래요
{"0":"findings","1":"tests","2":"medication_instructions","3":"lifestyle_instructions","4":"follow_up"}

예시 3. 문장:
0: 특별한 건 없다고 하셨어요
1: 약은 안 주셨어요
2: 다음 달 15일에 오라고
{"0":"findings","1":"medication_instructions","2":"follow_up"}"""


def system_prompt() -> str:
    return _SYSTEM


def user_message(sentences: Sequence[str]) -> str:
    lines = "\n".join(f"{i}: {s}" for i, s in enumerate(sentences))
    return f"문장 {len(sentences)}개:\n{lines}\n번호마다 라벨 하나. JSON:"
