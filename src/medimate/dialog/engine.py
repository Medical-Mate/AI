"""진료 전 카드 문진 — 상태 기계.

매 턴: 환자 발화 → Extractor로 축 갱신 → 카드 반영 → 다음 질문 선택.
종료를 강제하지 않는다. 언제 끝내도 그 시점 카드가 결과다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from medimate.dialog.questions import ASK_ORDER, CLOSING, OPENING, QUESTIONS
from medimate.llm.base import Extractor, TurnExtraction
from medimate.schema.card import Axis, FieldStatus, PreVisitCard, Provenance


@dataclass
class TurnLog:
    """판정 로그 한 줄. 경계 준수의 감사 기록 (docs/ai-design.md §7)."""

    turn: int
    asked_axis: Axis | None
    utterance: str
    extraction: TurnExtraction


@dataclass
class Session:
    extractor: Extractor
    card: PreVisitCard = field(default_factory=PreVisitCard)
    logs: list[TurnLog] = field(default_factory=list)
    asked_axis: Axis | None = None
    ended: bool = False

    def __post_init__(self) -> None:
        self.card.provenance = Provenance(
            prompt_version=self.extractor.prompt_version,
            model_id=self.extractor.model_id,
        )

    def opening(self) -> str:
        return OPENING

    def step(self, utterance: str) -> str:
        """환자 발화 하나를 처리하고 다음 발화(질문 또는 마무리)를 돌려준다."""
        if self.ended:
            return CLOSING
        ext = self.extractor.extract(utterance, self.asked_axis)
        self.logs.append(TurnLog(len(self.logs) + 1, self.asked_axis, utterance, ext))
        self._apply(ext)

        # 물었는데 아무 갱신이 없으면 그 축은 SKIPPED로 닫는다. 같은 질문을 반복하지 않는다.
        if self.asked_axis is not None:
            entry = self.card.axes[self.asked_axis]
            if entry.status == FieldStatus.NOT_ASKED:
                entry.status = FieldStatus.SKIPPED

        if ext.wants_to_stop:
            return self.end()
        nxt = self._next_axis()
        if nxt is None:
            return self.end()
        self.asked_axis = nxt
        return QUESTIONS[nxt]

    def end(self) -> str:
        self.ended = True
        self.asked_axis = None
        return CLOSING

    # ------------------------------------------------------------------
    def _apply(self, ext: TurnExtraction) -> None:
        if ext.chief_complaint and not self.card.chief_complaint:
            self.card.chief_complaint = ext.chief_complaint
        self.card.patient_notes.extend(ext.notes)
        for u in ext.updates:
            if not u.evidence.strip():
                continue  # 근거 없는 값은 카드에 넣지 않는다
            entry = self.card.axes[u.axis]
            entry.status = u.status
            if u.status == FieldStatus.FILLED:
                entry.value = u.value
            entry.evidence.append(u.evidence)

    def _next_axis(self) -> Axis | None:
        for a in ASK_ORDER:
            if self.card.axes[a].status == FieldStatus.NOT_ASKED:
                return a
        return None
