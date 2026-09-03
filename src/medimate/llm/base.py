"""LLM 경계.

LLM이 하는 일은 하나다: 환자 발화에서 어느 축에 무엇이 말해졌는지 뽑는다.
질문 문장은 LLM이 아니라 템플릿이 만든다(초기). 경계 위반 여지를 줄이기 위해서다.

모델은 미정이다. 구현체는 이 Protocol만 맞추면 교체 가능하다.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

from medimate.schema.card import Axis, FieldStatus


class AxisUpdate(BaseModel):
    axis: Axis
    status: FieldStatus  # FILLED / UNKNOWN / SKIPPED 중 하나. NOT_ASKED는 쓰지 않는다
    value: str | None = None
    evidence: str  # 환자 발화 원문 중 이 값의 근거가 된 구간. 비우면 안 된다


class TurnExtraction(BaseModel):
    """한 턴의 환자 발화에서 뽑은 것."""

    chief_complaint: str | None = None  # 첫 턴에 주로 채워진다
    updates: list[AxisUpdate] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)  # 축에 안 들어가는 환자 말
    wants_to_stop: bool = False  # "이만 할래요" 등


class Extractor(Protocol):
    model_id: str
    prompt_version: str

    def extract(self, utterance: str, asked_axis: Axis | None) -> TurnExtraction: ...
