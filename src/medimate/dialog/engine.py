"""진료 전 카드 문진 — 상태 기계.

매 턴: 환자 발화 → Extractor로 축 갱신 → 카드 반영 → 다음 질문 선택.
종료를 강제하지 않는다. 언제 끝내도 그 시점 카드가 결과다.

상태는 `SessionState`(직렬화 가능) 하나에 모두 들어 있다. 무상태 API는
요청마다 `Session.from_state`로 복원하고 `to_state`로 돌려준다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from medimate.dialog.questions import (
    ASK_ORDER,
    CLARIFY,
    CLOSING,
    EMPTY_INPUT,
    OPENING,
    QUESTIONS,
    TRUNCATED_NOTICE,
)
from medimate.dialog.state import HistoryTurn, SessionState
from medimate.llm.base import Extractor, TurnExtraction
from medimate.schema.card import Axis, FieldStatus, PreVisitCard, Provenance


@dataclass
class TurnLog:
    """판정 로그 한 줄. 경계 준수의 감사 기록 (docs/ai-design.md §7)."""

    turn: int
    asked_axis: Axis | None
    utterance: str
    extraction: TurnExtraction


@dataclass(frozen=True)
class Limits:
    """세션 안전 상한. 정상 문진은 닿지 않는다 — 걸리면 버그·공격이다.

    - max_utterance_chars: 팀 합의 임시값 300 (앱 카운터·백엔드 계약과 같은 값). 이중 방어
    - max_turns: 정상 최대 17(첫 발화 1 + 8축 + 확인 8). 20은 안전 정지선
    - max_session_tokens: 이 세션의 LLM 입력+출력 누적 상한. Extractor가 usage를 주면 적용
    """

    max_utterance_chars: int = 300
    max_turns: int = 20
    max_session_tokens: int = 40_000  # 턴당 ~1.1K × 20턴 + 여유
    history_turns: int = (
        2  # Extractor에 넘기는 최근 대화 수. 수정 발화("아까 3일이라고 했는데") 대응
    )


@dataclass
class Session:
    extractor: Extractor
    card: PreVisitCard = field(default_factory=PreVisitCard)
    logs: list[TurnLog] = field(default_factory=list)
    asked_axis: Axis | None = None
    clarified: set[Axis] = field(default_factory=set)  # 확인 질문은 축당 한 번
    ended: bool = False
    limits: Limits = field(default_factory=Limits)
    end_reason: str | None = None  # None | "stop" | "complete" | "max_turns" | "budget"
    history: list[HistoryTurn] = field(default_factory=list)  # 최근 N턴의 (질문, 답)
    turn: int = 0  # 처리한 발화 수. logs 길이와 같지만 복원 시 logs는 비어 있다
    # 이전 요청들에서 쓴 토큰(상태로 넘어온 값). 현재 extractor.usage는 여기 더해서 본다
    carried_tokens: int = 0

    def __post_init__(self) -> None:
        if self.card.provenance is None:
            self.card.provenance = Provenance(
                prompt_version=self.extractor.prompt_version,
                model_id=self.extractor.model_id,
            )

    # --- 직렬화 --------------------------------------------------------
    @classmethod
    def from_state(
        cls, extractor: Extractor, state: SessionState, limits: Limits | None = None
    ) -> Session:
        """백엔드가 들고 있던 상태로 세션을 복원한다. 판정 로그는 복원하지 않는다."""
        return cls(
            extractor=extractor,
            card=state.card,
            asked_axis=state.asked_axis,
            clarified=set(state.clarified),
            ended=state.ended,
            end_reason=state.end_reason,
            history=list(state.history),
            turn=state.turn,
            carried_tokens=state.session_tokens,
            limits=limits or Limits(),
        )

    def to_state(self) -> SessionState:
        return SessionState(
            card=self.card,
            asked_axis=self.asked_axis,
            clarified=sorted(self.clarified),
            history=list(self.history),
            turn=self.turn,
            session_tokens=self._session_tokens(),
            ended=self.ended,
            end_reason=self.end_reason,
        )

    # --- 대화 ----------------------------------------------------------
    def opening(self) -> str:
        return OPENING

    def step(self, utterance: str) -> str:
        """환자 발화 하나를 처리하고 다음 발화(질문 또는 마무리)를 돌려준다."""
        if self.ended:
            return CLOSING

        # 빈 입력은 LLM을 부르지 않는다. 턴으로도 세지 않는다
        utterance = utterance.strip()
        if not utterance:
            return EMPTY_INPUT + " " + self._current_question()

        # 길이 상한 — 앱·백엔드가 먼저 막지만 이중 방어. 잘린 사실을 사용자에게 알린다
        notice = ""
        if len(utterance) > self.limits.max_utterance_chars:
            utterance = utterance[: self.limits.max_utterance_chars]
            notice = TRUNCATED_NOTICE + " "

        # 턴 상한 — 정상 흐름은 닿지 않는다
        if self.turn >= self.limits.max_turns:
            return self.end("max_turns")

        # 세션 토큰 예산 — 어댑터가 usage를 노출하면 적용
        if self._session_tokens() >= self.limits.max_session_tokens:
            return self.end("budget")

        asked_question = self._current_question()
        ext = self.extractor.extract(utterance, self.asked_axis, self.history)
        self.turn += 1
        self.logs.append(TurnLog(self.turn, self.asked_axis, utterance, ext))
        self._remember(asked_question, utterance)
        self._apply(ext)

        # 물었는데 닫히지 않은 축은 SKIPPED로 닫는다. 같은 질문을 반복하지 않는다.
        # (AMBIGUOUS로 되물은 뒤에도 또 AMBIGUOUS면 그것도 닫는다)
        if self.asked_axis is not None:
            entry = self.card.axes[self.asked_axis]
            if entry.status == FieldStatus.NOT_ASKED or (
                entry.status == FieldStatus.AMBIGUOUS and self.asked_axis in self.clarified
            ):
                entry.status = FieldStatus.SKIPPED

        if ext.wants_to_stop:
            return notice + self.end("stop")
        nxt = self._next_axis()
        if nxt is None:
            return notice + self.end("complete")
        self.asked_axis = nxt
        if self.card.axes[nxt].status == FieldStatus.AMBIGUOUS:
            self.clarified.add(nxt)
            return notice + CLARIFY[nxt]
        return notice + QUESTIONS[nxt]

    def end(self, reason: str = "stop") -> str:
        self.ended = True
        self.end_reason = reason
        self.asked_axis = None
        return CLOSING

    # ------------------------------------------------------------------
    def _remember(self, question: str, utterance: str) -> None:
        """이번 턴의 (질문, 답)을 이력에 넣고 최근 N턴만 남긴다. 요약 생성 없음."""
        n = self.limits.history_turns
        if n <= 0:
            self.history = []
            return
        self.history.append((question, utterance))
        del self.history[:-n]

    def _current_question(self) -> str:
        if self.asked_axis is None:
            return OPENING
        if self.asked_axis in self.clarified:
            return CLARIFY[self.asked_axis]
        return QUESTIONS[self.asked_axis]

    def _session_tokens(self) -> int:
        usage = getattr(self.extractor, "usage", None)
        if usage is None:
            return self.carried_tokens
        now = int(getattr(usage, "input_tokens", 0)) + int(getattr(usage, "output_tokens", 0))
        return self.carried_tokens + now

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
        # 확인이 필요한 축이 먼저, 그 다음 아직 안 물은 축
        for a in ASK_ORDER:
            if self.card.axes[a].status == FieldStatus.AMBIGUOUS and a not in self.clarified:
                return a
        for a in ASK_ORDER:
            if self.card.axes[a].status == FieldStatus.NOT_ASKED:
                return a
        return None
