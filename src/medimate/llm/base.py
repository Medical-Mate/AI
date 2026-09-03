"""LLM 경계.

LLM이 하는 일은 하나다: 환자 발화에서 어느 축에 무엇이 말해졌는지 뽑는다.
질문 문장은 LLM이 아니라 템플릿이 만든다(초기). 경계 위반 여지를 줄이기 위해서다.

모델은 미정이다. 구현체는 이 Protocol만 맞추면 교체 가능하다.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from medimate.schema.card import Axis, FieldStatus


class AxisUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    axis: Axis
    status: FieldStatus  # FILLED / UNKNOWN / SKIPPED / AMBIGUOUS. NOT_ASKED는 쓰지 않는다
    value: str | None = None  # AMBIGUOUS일 때는 발화에 있는 후보들만
    evidence: str  # 환자 발화 원문 중 이 값의 근거가 된 구간. 비우면 안 된다


class TurnExtraction(BaseModel):
    """한 턴의 환자 발화에서 뽑은 것.

    extra="forbid": 모델이 diagnosis 같은 필드를 덧붙이면 스키마 위반으로 잡는다.
    """

    model_config = ConfigDict(extra="forbid")

    chief_complaint: str | None = None  # 첫 턴에 주로 채워진다
    updates: list[AxisUpdate] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)  # 축에 안 들어가는 환자 말
    wants_to_stop: bool = False  # "이만 할래요" 등


# (직전 질문, 환자 답) 쌍. 엔진이 최근 N턴만 넘긴다 — 토큰 절약, 요약 생성 없음
Turn = tuple[str, str]


class Extractor(Protocol):
    model_id: str
    prompt_version: str

    def extract(
        self, utterance: str, asked_axis: Axis | None, history: Sequence[Turn] = ()
    ) -> TurnExtraction: ...
