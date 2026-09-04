"""진료 후 문답 — 축별 질문 템플릿 (챗봇②, 이슈 #9).

환자가 "들은 것"만 묻는다. 그래서 문장에 진단·해석·권고 어휘가 없다.
"선생님이 ~라고 하셨나요"처럼 의사의 말을 되묻는 형태를 유지한다.
질문 문구는 의료인 자문 확인서(docs/advisor-checklist.md)의 검토 대상이다.
"""

from medimate.schema.postvisit import PostAxis

OPENING = "진료 잘 받으셨어요? 선생님이 어떤 얘기를 하셨는지 기억나는 대로 편하게 말씀해 주세요."

QUESTIONS: dict[PostAxis, str] = {
    PostAxis.HEARD_DIAGNOSIS: "선생님이 뭐라고 하셨어요? 들은 그대로 말씀해 주시면 돼요.",
    PostAxis.MEDICATION: "약은 어떻게 먹으라고 하셨어요? 몇 번, 며칠 같은 것 기억나는 대로요.",
    PostAxis.TESTS_PROCEDURES: "검사나 치료를 받았거나, 다음에 하기로 한 게 있나요?",
    PostAxis.FOLLOW_UP: "다음에 언제 다시 오라고 하셨어요?",
    PostAxis.INSTRUCTIONS: "하지 말라고 하거나 꼭 하라고 한 게 있었나요?",
    PostAxis.OPEN_QUESTIONS: (
        "못 물어본 거나 아직 헷갈리는 게 있으세요? 다음에 물어보실 수 있게 적어 둘게요."
    ),
}

# AMBIGUOUS일 때 — 발화가 그 축을 건드렸지만 확정이 안 될 때 한 번만 되묻는다.
CLARIFY: dict[PostAxis, str] = {
    PostAxis.HEARD_DIAGNOSIS: (
        "선생님이 쓰신 표현을 기억나는 대로 조금만 더 말씀해 주실 수 있을까요?"
    ),
    PostAxis.MEDICATION: "약 이름이나 먹는 횟수를 조금만 더 말씀해 주실 수 있을까요?",
    PostAxis.TESTS_PROCEDURES: "어떤 검사나 치료였는지 조금만 더 말씀해 주실 수 있을까요?",
    PostAxis.FOLLOW_UP: "다음 방문 시점을 대략이라도 말씀해 주실 수 있을까요?",
    PostAxis.INSTRUCTIONS: "어떤 걸 조심하라고 하셨는지 조금만 더 말씀해 주실 수 있을까요?",
    PostAxis.OPEN_QUESTIONS: "어떤 점이 헷갈리는지 조금만 더 말씀해 주실 수 있을까요?",
}

# 들은 말 → 약 → 검사 → 다음 방문 → 주의사항 → 물어볼 것. 기억이 선명한 것부터.
ASK_ORDER: list[PostAxis] = [
    PostAxis.HEARD_DIAGNOSIS,
    PostAxis.MEDICATION,
    PostAxis.TESTS_PROCEDURES,
    PostAxis.FOLLOW_UP,
    PostAxis.INSTRUCTIONS,
    PostAxis.OPEN_QUESTIONS,
]

MESSAGE_QUESTION = "마지막으로, 따로 적어 두고 싶은 말이 있으세요? 없으면 없다고 해 주세요."

CLOSING = "말씀해 주신 내용을 정리해 두었어요. 나중에 다시 보실 수 있어요."
