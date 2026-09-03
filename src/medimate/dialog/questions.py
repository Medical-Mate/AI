"""축별 되묻기 문장 — 템플릿.

환자가 앞에서 말한 것에만 앵커한다. 그래서 문장에 진단·병명 어휘가 없다.
질문 순서는 SOCRATES를 따르되 부위와 주 호소가 먼저다.
"""

from medimate.schema.card import Axis

OPENING = "어디가 어떻게 불편해서 오셨는지 편하게 말씀해 주세요."

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

ASK_ORDER: list[Axis] = [
    Axis.SITE,
    Axis.ONSET,
    Axis.CHARACTER,
    Axis.SEVERITY,
    Axis.TIME_COURSE,
    Axis.EXACERBATING_RELIEVING,
    Axis.RADIATION,
    Axis.ASSOCIATED,
]

CLOSING = "말씀해 주신 내용을 정리해서 진료 때 보실 수 있게 준비했어요."
