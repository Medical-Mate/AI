"""문진 명세 — 엔진이 어떤 카드·어떤 축·어떤 질문으로 도는지를 한 객체로.

엔진(engine.Session)은 진료 전·후 공통이다. 다른 것은 축 목록, 질문 템플릿, 카드 타입,
부위 사전 채움 유무뿐이고 그것만 여기 담는다. 새 문진을 추가하려면 명세 하나를 더 만든다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from medimate.dialog import postvisit_questions as post_q
from medimate.dialog import questions as pre_q
from medimate.schema.card import Axis, InterviewCard, PreVisitCard
from medimate.schema.postvisit import PostAxis, PostVisitCard


@dataclass(frozen=True)
class InterviewSpec:
    name: str  # "previsit" | "postvisit". 카드 export의 card_type
    axis_type: type[StrEnum]
    card_type: type[InterviewCard]
    questions: dict  # axis → 질문
    clarify: dict  # axis → 확인 질문(AMBIGUOUS 1회)
    ask_order: list  # 묻는 순서
    opening: str
    message_question: str  # 축이 모두 닫힌 뒤 한 번
    closing: str
    # 부위를 미리 짚고 들어올 수 있는 축(진료 전만). None이면 사전 채움 없음
    site_axis: StrEnum | None = None
    opening_with_site: str | None = None  # "{site}" 자리 표시자
    # 공통 문구
    empty_input: str = pre_q.EMPTY_INPUT
    truncated_notice: str = pre_q.TRUNCATED_NOTICE
    no_message: frozenset[str] = field(default_factory=lambda: frozenset(pre_q.NO_MESSAGE))
    site_preselected_tag: str = pre_q.SITE_PRESELECTED

    def axis(self, value: str) -> StrEnum:
        return self.axis_type(value)


PREVISIT_SPEC = InterviewSpec(
    name="previsit",
    axis_type=Axis,
    card_type=PreVisitCard,
    questions=pre_q.QUESTIONS,
    clarify=pre_q.CLARIFY,
    ask_order=pre_q.ASK_ORDER,
    opening=pre_q.OPENING,
    message_question=pre_q.MESSAGE_QUESTION,
    closing=pre_q.CLOSING,
    site_axis=Axis.SITE,
    opening_with_site=pre_q.OPENING_WITH_SITE,
)

POSTVISIT_SPEC = InterviewSpec(
    name="postvisit",
    axis_type=PostAxis,
    card_type=PostVisitCard,
    questions=post_q.QUESTIONS,
    clarify=post_q.CLARIFY,
    ask_order=post_q.ASK_ORDER,
    opening=post_q.OPENING,
    message_question=post_q.MESSAGE_QUESTION,
    closing=post_q.CLOSING,
)

SPECS: dict[str, InterviewSpec] = {s.name: s for s in (PREVISIT_SPEC, POSTVISIT_SPEC)}
