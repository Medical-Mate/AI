"""축별 되묻기 문장 — 템플릿.

환자가 앞에서 말한 것에만 앵커한다. 그래서 문장에 진단·병명 어휘가 없다.
질문 순서는 SOCRATES를 따르되 부위와 주 호소가 먼저다.
"""

from medimate.schema.card import Axis

OPENING = "어디가 어떻게 불편해서 오셨는지 편하게 말씀해 주세요."
# 인체도에서 부위를 먼저 짚고 들어온 경우. 부위를 다시 묻지 않고 그 부위에 앵커한다
OPENING_WITH_SITE = "{site} 쪽이 어떻게 불편하신지 편하게 말씀해 주세요."
SITE_PRESELECTED = "[부위 선택]"  # UI 선택으로 채운 SITE의 evidence 표시. 발화가 아님을 드러낸다

QUESTIONS: dict[Axis, str] = {
    Axis.SITE: "불편한 곳이 정확히 어디쯤인가요?",
    Axis.ONSET: "언제부터 그러셨어요? 갑자기 시작됐나요, 서서히 시작됐나요?",
    Axis.CHARACTER: (
        "어떤 느낌인가요? 예를 들면 욱신거리는지, 찌르는 것 같은지, 조이는 것 같은지요."
    ),
    Axis.RADIATION: "그 불편함이 다른 곳으로 퍼지기도 하나요?",
    Axis.ASSOCIATED: "같이 나타나는 다른 증상이 있나요?",
    Axis.TIME_COURSE: "처음보다 지금은 어떤가요? 심해졌는지, 비슷한지, 나아졌는지요.",
    Axis.EXACERBATING_RELIEVING: "어떤 때 더 심해지고, 어떤 때 좀 괜찮아지나요?",
    Axis.SEVERITY: "가장 심할 때를 10점이라고 하면 지금은 몇 점 정도인가요?",
}

# AMBIGUOUS일 때 — 발화가 그 축을 건드렸지만 확정이 안 될 때 한 번만 되묻는다.
CLARIFY: dict[Axis, str] = {
    Axis.SITE: "어느 쪽, 어디쯤인지 조금만 더 말씀해 주실 수 있을까요?",
    Axis.ONSET: "시작된 시점을 대략이라도 말씀해 주실 수 있을까요?",
    Axis.CHARACTER: "느낌을 조금만 더 설명해 주실 수 있을까요?",
    Axis.RADIATION: "퍼지는 곳이 어디인지 조금만 더 말씀해 주실 수 있을까요?",
    Axis.ASSOCIATED: "함께 나타나는 증상이 무엇인지 조금만 더 말씀해 주실 수 있을까요?",
    Axis.TIME_COURSE: "처음과 비교해서 어떤지 조금만 더 말씀해 주실 수 있을까요?",
    Axis.EXACERBATING_RELIEVING: "어떤 때 그런지 조금만 더 말씀해 주실 수 있을까요?",
    Axis.SEVERITY: "정도를 대략이라도 숫자나 말로 표현해 주실 수 있을까요?",
}

# 묻는 순서. **심각도는 여기 없다**(2026-09-11, 백엔드 합의 #7).
# 와이어프레임 3단계가 5단계 슬라이더로 받으므로 문답에서 또 물으면 중복이고,
# 슬라이더 값이 더 정확하다.
# **NRS가 아니다**(2026-09-11 정정). 1~5 서열척도 + 단계별 라벨이고
# `"3 (꽤 아파요)"` 같은 자유 문자열로 온다. 우리가 오래 NRS로 잘못 알고 있었고
# 백엔드가 와이어프레임 1d를 직접 읽어 잡아 줬다.
# 앱이 `selections: [{axis: "severity", value: ...}]`로 보내면
# LLM 없이 카드에 들어가고 근거는 `[선택] …`으로 남는다.
# 축 자체는 카드에 그대로 있다 — 채우는 경로만 바뀌었다. 질문 템플릿도 QUESTIONS에 남겨 둔다
# (문답으로 되돌리려면 이 목록에 한 줄 넣으면 된다).
ASK_ORDER: list[Axis] = [
    Axis.SITE,
    Axis.ONSET,
    Axis.CHARACTER,
    Axis.TIME_COURSE,
    Axis.EXACERBATING_RELIEVING,
    Axis.RADIATION,
    Axis.ASSOCIATED,
]

# 8축이 모두 닫힌 뒤 한 번. 답은 원문 그대로 카드의 patient_message에 남고,
# 증상이 섞여 있으면 축에도 반영
MESSAGE_QUESTION = (
    "마지막으로, 의사 선생님께 꼭 전하고 싶은 말이 더 있으세요? 없으면 없다고 해 주세요."
)
# 이렇게만 답하면 전하고 싶은 말이 없는 것으로 본다(원문을 카드에 남기지 않는다).
# 그 외는 전부 원문 보존
NO_MESSAGE = {
    "없어요",
    "없습니다",
    "없음",
    "없다",
    "아니요",
    "아니오",
    "아뇨",
    "괜찮아요",
    "됐어요",
    "no",
}

CLOSING = "말씀해 주신 내용을 정리해서 진료 때 보실 수 있게 준비했어요."

EMPTY_INPUT = "입력된 내용이 없어요."
TRUNCATED_NOTICE = "내용이 길어서 앞부분만 반영했어요."
