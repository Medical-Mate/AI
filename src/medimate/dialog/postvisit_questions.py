"""진료 후 — 질문 템플릿 (챗봇②).

와이어프레임 1p·1q(2026-09-07)는 문답이 아니라 자유 메모 → 4묶음 분류다.
그래서 주 경로는 `dialog/memo.py`이고, 이 템플릿은 메모가 비었거나
한 묶음이 비었을 때 되묻는 **보조 질문**으로 남긴다.
문장은 환자가 "들은 것"만 묻는다. 진단·해석·권고 어휘가 없다.
"""

from medimate.schema.postvisit import PostAxis

OPENING = "진료실에서 들은 말을 기억나는 대로 편하게 적어 주세요. 말로 이어서 적어도 돼요."

QUESTIONS: dict[PostAxis, str] = {
    PostAxis.FINDINGS: "선생님이 어떤 상태라고 하셨어요? 들은 그대로 말씀해 주시면 돼요.",
    PostAxis.TESTS: "검사를 받았거나 결과를 언제 알려준다고 했나요?",
    PostAxis.MEDICATION_INSTRUCTIONS: (
        "약은 어떻게 먹으라고 했고, 하지 말라거나 하라고 한 게 있었나요?"
    ),
    PostAxis.FOLLOW_UP: "다음에 언제 다시 오라고 하셨어요?",
}

CLARIFY: dict[PostAxis, str] = {
    PostAxis.FINDINGS: "선생님이 쓰신 표현을 기억나는 대로 조금만 더 말씀해 주실 수 있을까요?",
    PostAxis.TESTS: "어떤 검사였는지 조금만 더 말씀해 주실 수 있을까요?",
    PostAxis.MEDICATION_INSTRUCTIONS: (
        "약 이름이나 횟수, 지시 내용을 조금만 더 말씀해 주실 수 있을까요?"
    ),
    PostAxis.FOLLOW_UP: "다음 방문 시점을 대략이라도 말씀해 주실 수 있을까요?",
}

ASK_ORDER: list[PostAxis] = [
    PostAxis.FINDINGS,
    PostAxis.TESTS,
    PostAxis.MEDICATION_INSTRUCTIONS,
    PostAxis.FOLLOW_UP,
]

MESSAGE_QUESTION = "따로 적어 두고 싶은 말이 있으세요? 없으면 없다고 해 주세요."

CLOSING = "말씀해 주신 내용을 네 묶음으로 정리해 두었어요. 나중에 다시 보실 수 있어요."
