"""문진 — 상태 기계. 진료 전·후 공통.

매 턴: 환자 발화 → Extractor로 축 갱신 → 카드 반영 → 다음 질문 선택.
종료를 강제하지 않는다. 언제 끝내도 그 시점 카드가 결과다.

무엇을 묻는지(축·질문·카드 타입)는 `InterviewSpec`이 정한다. 기본은 진료 전(PREVISIT_SPEC).
상태는 `SessionState`(직렬화 가능) 하나에 모두 들어 있다. 무상태 API는
요청마다 `Session.from_state`로 복원하고 `to_state`로 돌려준다.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from medimate.dialog.guard import GuardConfig, guard_extraction
from medimate.dialog.spec import PREVISIT_SPEC, SPECS, InterviewSpec
from medimate.dialog.state import HistoryTurn, SessionState
from medimate.llm.base import Extractor, TurnExtraction
from medimate.schema.card import FieldStatus, InterviewCard, Provenance
from medimate.text.chatnorm import normalize as _normalize_chat


def chat_normalize_enabled() -> bool:
    """env `MEDIMATE_CHAT_NORMALIZE` (기본 꺼짐). 켜면 가드가 채팅 표기 복원문도 근거로 인정한다."""
    import os

    return os.getenv("MEDIMATE_CHAT_NORMALIZE", "").strip().lower() in ("1", "true", "yes", "on")


def _chat_normalized(utterance: str) -> str | None:
    n = _normalize_chat(utterance)
    return n.text if n.changed else None


@dataclass
class TurnLog:
    """판정 로그 한 줄. 경계 준수의 감사 기록 (docs/ai-design.md §7)."""

    turn: int
    asked_axis: StrEnum | None
    utterance: str
    extraction: TurnExtraction  # 가드를 통과해 카드에 반영된 것
    raw_extraction: TurnExtraction | None = None  # 모델이 낸 원본(가드가 버린 것 포함). 감사용
    dropped: list[dict] = field(default_factory=list)  # 가드가 버린 갱신과 이유


@dataclass(frozen=True)
class Limits:
    """세션 안전 상한. 정상 문진은 닿지 않는다 — 걸리면 버그·공격이다.

    - max_utterance_chars: 팀 합의 임시값 300 (앱 카운터·백엔드 계약과 같은 값). 이중 방어
    - max_turns: 진료 전 정상 최대 18(첫 발화 1 + 8축 + 확인 8 + 전할 말 1). 20은 안전 정지선
    - max_session_tokens: 이 세션의 LLM 입력+출력 누적 상한. Extractor가 usage를 주면 적용
    """

    max_utterance_chars: int = 300
    max_turns: int = 20
    max_session_tokens: int = 40_000  # 턴당 ~1.1K × 20턴 + 여유
    history_turns: int = (
        2  # Extractor에 넘기는 최근 대화 수. 수정 발화("아까 3일이라고 했는데") 대응
    )


def _has_final(word: str) -> bool:
    """마지막 글자에 받침이 있나. 한글이 아니면 없는 것으로 본다(조사를 덜 틀리는 쪽)."""
    ch = word.strip()[-1:] if word.strip() else ""
    if not ch or not ("가" <= ch <= "힣"):
        return False
    return (ord(ch) - 0xAC00) % 28 != 0


def _eul(word: str) -> str:
    return "을" if _has_final(word) else "를"


def _i_ga(word: str) -> str:
    return "이" if _has_final(word) else "가"


@dataclass
class Session:
    extractor: Extractor
    spec: InterviewSpec = PREVISIT_SPEC
    card: InterviewCard | None = None  # None이면 spec.card_type()으로 만든다
    logs: list[TurnLog] = field(default_factory=list)
    asked_axis: StrEnum | None = None
    clarified: set[StrEnum] = field(default_factory=set)  # 확인 질문은 축당 한 번
    ended: bool = False
    limits: Limits = field(default_factory=Limits)
    end_reason: str | None = None  # None | "stop" | "complete" | "max_turns" | "budget"
    history: list[HistoryTurn] = field(default_factory=list)  # 최근 N턴의 (질문, 답)
    turn: int = 0  # 처리한 발화 수. logs 길이와 같지만 복원 시 logs는 비어 있다
    message_asked: bool = False  # 마지막 "전하고 싶은 말" 질문을 냈는가
    # 이전 요청들에서 쓴 토큰(상태로 넘어온 값). 현재 extractor.usage는 여기 더해서 본다
    carried_tokens: int = 0
    # 런타임 가드. 서버 기본값은 근거·숫자만, 온디바이스 프로필은 GuardConfig.ondevice()
    guard: GuardConfig = field(default_factory=GuardConfig)
    # True면 첫 자유 발화 없이 첫 축 질문부터 시작한다
    # (온디바이스: 소형 모델은 다축 첫 발화를 못 뽑는다)
    skip_open_ended: bool = False
    profile: str = "server"  # server | ondevice. 상태로 왕복된다
    # 짚은 부위와 말한 부위의 관계를 본다. API가 온톨로지로 만들어 넣는다.
    # 반환: "other"(다른 곳) | "narrower"(말한 쪽이 더 좁다) | "broader"(짚은 쪽이 더 좁다)
    #       | "same" | "unknown"(모름). None이면 대조를 안 한다 — 엔진 단위 테스트는
    # 주입 없이 예전 동작 그대로 돈다
    site_relation: Callable[[str, str], str] | None = None

    def __post_init__(self) -> None:
        if self.card is None:
            self.card = self.spec.card_type()
        if self.card.provenance is None:
            self.card.provenance = Provenance(
                prompt_version=self.extractor.prompt_version,
                model_id=self.extractor.model_id,
            )

    # --- 직렬화 --------------------------------------------------------
    @classmethod
    def from_state(
        cls,
        extractor: Extractor,
        state: SessionState,
        limits: Limits | None = None,
        spec: InterviewSpec | None = None,
        site_relation: Callable[[str, str], str] | None = None,
    ) -> Session:
        """백엔드가 들고 있던 상태로 세션을 복원한다. 판정 로그는 복원하지 않는다."""
        spec = spec or SPECS[state.spec]
        # 상태의 카드는 공통 뼈대로 들어오므로(extra 보존) 명세의 카드 타입으로 다시 읽는다
        card = spec.card_type.model_validate(state.card.model_dump())
        ax = spec.axis
        ondevice = state.profile == "ondevice"
        return cls(
            extractor=extractor,
            spec=spec,
            card=card,
            profile=state.profile,
            guard=GuardConfig.ondevice() if ondevice else GuardConfig(),
            skip_open_ended=ondevice,
            asked_axis=ax(state.asked_axis) if state.asked_axis else None,
            clarified={ax(a) for a in state.clarified},
            ended=state.ended,
            end_reason=state.end_reason,
            history=list(state.history),
            turn=state.turn,
            message_asked=state.message_asked,
            carried_tokens=state.session_tokens,
            limits=limits or Limits(),
            site_relation=site_relation,
        )

    def to_state(self) -> SessionState:
        # 카드는 공통 뼈대 타입으로 낸다(extra 보존). 하위 카드 인스턴스를 그대로 넣으면
        # 응답 직렬화가 선언 타입(InterviewCard) 기준으로 잘라 site_selection 등이 사라진다
        return SessionState(
            spec=self.spec.name,
            profile=self.profile,
            card=InterviewCard.model_validate(self.card.model_dump()),
            asked_axis=self.asked_axis.value if self.asked_axis else None,
            clarified=sorted(a.value for a in self.clarified),
            history=list(self.history),
            turn=self.turn,
            message_asked=self.message_asked,
            session_tokens=self._session_tokens(),
            ended=self.ended,
            end_reason=self.end_reason,
        )

    # --- 대화 ----------------------------------------------------------
    SELECTED_TAG = "[선택]"  # 폼·칩으로 고른 값의 evidence 표시. 발화가 아님을 드러낸다

    def prefill(self, axis: StrEnum, value: str, tag: str | None = None) -> None:
        """폼·칩(선택지)으로 받은 값을 축에 넣는다. LLM을 부르지 않는다.

        evidence는 발화가 아니라 UI 선택임을 태그로 표시한다. 이미 값이 있으면 덮어쓴다
        (앱의 "이전 답 고치기"가 이 경로를 쓴다).
        """
        value = value.strip()
        if not value or axis not in self.card.axes:
            return
        entry = self.card.axes[axis]
        entry.status = FieldStatus.FILLED
        entry.value = value
        entry.evidence.append(f"{tag or self.SELECTED_TAG} {value}")

    def preselect_site(self, label: str) -> None:
        """인체도에서 짚은 부위로 SITE를 채운다. 부위 축이 없는 명세(진료 후)에서는 무시."""
        if self.spec.site_axis is None:
            return
        self.prefill(self.spec.site_axis, label, self.spec.site_preselected_tag)

    def _ensure_started(self) -> None:
        """skip_open_ended면 첫 자유 발화 없이 첫 축을 정한다.

        부위 사전 채움 뒤에 호출되도록 지연한다.
        """
        if self.skip_open_ended and self.asked_axis is None and self.turn == 0 and not self.ended:
            self.asked_axis = self._next_axis()

    def opening(self) -> str:
        self._ensure_started()
        if self.skip_open_ended and self.asked_axis is not None:
            return self.spec.questions[self.asked_axis]
        if self.spec.site_axis is not None and self.spec.opening_with_site:
            site = self.card.axes[self.spec.site_axis]
            if site.status == FieldStatus.FILLED and site.value:
                return self.spec.opening_with_site.format(site=site.value)
        return self.spec.opening

    def step(
        self,
        utterance: str,
        extraction: TurnExtraction | None = None,
        selections: Sequence[tuple[StrEnum, str]] = (),
    ) -> str:
        """한 턴을 처리하고 다음 발화(질문 또는 마무리)를 돌려준다.

        - utterance만: 서버 Extractor로 추출 (기본)
        - extraction 주어짐: 폰이 이미 추출한 JSON. Extractor를 부르지 않고 가드만.
          근거 검증을 위해 utterance(원문)도 함께 와야 한다
        - selections: 칩·폼으로 고른 (축, 값). LLM 없이 카드에 바로. utterance 없이 selections만
          와도
          한 턴으로 처리해 다음 질문을 정한다
        """
        # **끝난 뒤에도 선택은 받는다.** 와이어프레임 순서가 2 문답 → 3 통증 슬라이더라,
        # 문답이 `complete`로 닫힌 **뒤에** 슬라이더 값이 온다. 여기서 막으면 통증 강도가
        # 카드에 영원히 안 들어간다 — `severity`는 묻지 않는 축이라 문답으로 채울 길도 없다.
        #
        # 선택 반영은 결정론이고 LLM을 부르지 않으므로 끝난 세션에서도 안전하다.
        # 발화는 여전히 무시한다(LLM 호출 0). 상태도 바꾸지 않는다 — `end_reason`은 그대로다.
        if self.ended:
            for axis, value in selections:
                self.prefill(axis, value)
            return self.spec.closing
        self._ensure_started()

        for axis, value in selections:
            self.prefill(axis, value)

        # 빈 입력은 LLM을 부르지 않는다. 턴으로도 세지 않는다 (선택지만 온 턴은 예외 — 아래)
        utterance = utterance.strip()
        if not utterance and not selections:
            return self.spec.empty_input + " " + self._current_question()

        # 길이 상한 — 앱·백엔드가 먼저 막지만 이중 방어. 잘린 사실을 사용자에게 알린다
        notice = ""
        if len(utterance) > self.limits.max_utterance_chars:
            utterance = utterance[: self.limits.max_utterance_chars]
            notice = self.spec.truncated_notice + " "

        # 턴 상한 — 정상 흐름은 닿지 않는다
        if self.turn >= self.limits.max_turns:
            return self.end("max_turns")

        # 세션 토큰 예산 — 어댑터가 usage를 노출하면 적용
        if self._session_tokens() >= self.limits.max_session_tokens:
            return self.end("budget")

        asked_question = self._current_question()
        if extraction is not None:
            raw = extraction  # 폰이 뽑은 것. 서버는 부르지 않는다
        elif utterance:
            raw = self.extractor.extract(utterance, self.asked_axis, self.history)
        else:
            raw = TurnExtraction()  # 선택지만 온 턴
        # 채팅 표기 복원(#117). 기본 꺼짐 — v5 프롬프트 eval 전까지 운영 동작을 바꾸지 않는다.
        # 켜면 `ㄴㄴ`·`ㄱㅊ`처럼 자음만 있는 답이 글자 없음 필터를 통과할 수 있다(복원문 기준)
        normalized = _chat_normalized(utterance) if chat_normalize_enabled() else None
        g = guard_extraction(raw, utterance, self.asked_axis, self.guard, normalized=normalized)
        ext = g.extraction
        self.turn += 1
        self.logs.append(TurnLog(self.turn, self.asked_axis, utterance, ext, raw, g.dropped))
        if utterance:
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

        # 마지막 질문의 답: 원문을 카드에 남기고 끝낸다 (축 반영은 위 _apply에서 이미 됐다)
        if self.message_asked:
            if utterance.rstrip(".!~ ") not in self.spec.no_message:
                self.card.patient_message = utterance
            return notice + self.end("complete")

        nxt = self._next_axis()
        if nxt is None:
            # 축이 모두 닫혔다. 명세에 전할 말 질문이 있으면 한 번 묻고, 없으면 바로 끝낸다
            self.asked_axis = None
            if self.spec.message_question is None:
                return notice + self.end("complete")
            self.message_asked = True
            return notice + self.spec.message_question
        self.asked_axis = nxt
        if self.card.axes[nxt].status == FieldStatus.AMBIGUOUS:
            self.clarified.add(nxt)
            return notice + self._clarify_text(nxt)
        return notice + self.spec.questions[nxt]

    def end(self, reason: str = "stop") -> str:
        self.ended = True
        self.end_reason = reason
        self.asked_axis = None
        return self.spec.closing

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
            if self.message_asked and self.spec.message_question is not None:
                return self.spec.message_question
            return self.opening()
        if self.asked_axis in self.clarified:
            return self._clarify_text(self.asked_axis)
        return self.spec.questions[self.asked_axis]

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
            if u.axis not in self.card.axes:
                continue  # 다른 문진의 축(진료 전 축이 진료 후 카드에)은 버린다. 스키마 경계
            axis = self.spec.axis(u.axis)
            entry = self.card.axes[axis]
            is_site = axis == self.spec.site_axis
            if is_site and self._preselected_site() and u.status != FieldStatus.FILLED:
                # 부위는 인체도에서 이미 골랐다. 모델이 "아래쪽"만 보고 애매하다 해도
                # 다시 묻지 않는다. 환자가 더 좁혀 말하면(FILLED) 받아서 라벨 뒤에 붙인다
                continue
            entry.status = u.status
            if u.status == FieldStatus.FILLED:
                entry.value = self._merge_site_label(axis, u.value)
                # 짚은 곳과 다른 부위를 말했다. 어느 쪽인지는 **환자가 정한다** — 우리가
                # 고르면 의사가 읽는 사실이 바뀐다. 되묻고, 안 답하면 그대로 닫힌다
                label = self._preselected_site() if is_site else None
                if label and u.value and self._site_conflict(label, u.value):
                    entry.status = FieldStatus.AMBIGUOUS
            entry.evidence.append(u.evidence)

    def _preselected_site(self) -> str | None:
        """인체도에서 짚은 부위 라벨. evidence의 [부위 선택] 표시로 구분한다."""
        if self.spec.site_axis is None:
            return None
        tag = self.spec.site_preselected_tag
        for ev in self.card.axes[self.spec.site_axis].evidence:
            if ev.startswith(tag):
                return ev[len(tag) :].strip() or None
        return None

    def _merge_site_label(self, axis: StrEnum, value: str | None) -> str | None:
        """SITE에 부위가 미리 선택돼 있으면 환자의 세부 표현("아래쪽 중앙")이 라벨을 지우지 않게
        앞에 붙인다. 모델은 선택된 부위를 모르므로 엔진이 지킨다.

        **다른 부위를 말한 경우는 붙이지 않는다**(2026-09-15). `이마`를 짚고 `눈`이라고 하면
        `"이마 눈"`이 되어 어느 쪽인지 알 수 없는 줄이 카드에 남았다. 그때는 값을 환자 말로
        두고 `AMBIGUOUS`로 되묻는다 — 판정은 `_site_conflict`가 한다.
        """
        if axis != self.spec.site_axis or not value:
            return value
        label = self._preselected_site()
        if not label or label in value:
            return value
        rel = self._site_relation(label, value)
        if rel in ("other", "narrower", "same"):
            # 다른 곳이면 되묻고(값은 환자 말), 더 좁혀 말했으면 좁은 쪽이 맞다
            return value
        if rel == "broader":
            # 짚은 쪽이 더 좁다("어깨"를 짚고 "팔"이라고 함). 넓은 말을 붙이면 흐려진다
            return label
        # 온톨로지가 모르는 세부 표현("무릎 안쪽"). 겹치는 낱말은 두 번 쓰지 않는다
        if any(w and w in value for w in label.split()):
            return value
        return f"{label} {value}"

    def _site_conflict(self, label: str, value: str) -> bool:
        """짚은 부위와 말한 부위가 **다른 곳**인가.

        온톨로지를 모르면(주입 없음·둘 중 하나라도 못 찾음) **충돌로 보지 않는다.**
        헛되묻기는 환자에게 "왜 못 알아듣지"로 읽혀서, 확실할 때만 묻는다.
        """
        return self._site_relation(label, value) == "other"

    def _site_relation(self, label: str, value: str) -> str:
        if self.site_relation is None:
            return "unknown"
        return self.site_relation(label, value)

    def _clarify_text(self, axis: StrEnum) -> str:
        """되묻기 문구. 부위가 어긋난 경우만 무엇과 무엇이 어긋났는지 짚어 준다."""
        generic = self.spec.clarify[axis]
        if axis != self.spec.site_axis:
            return generic
        label = self._preselected_site()
        spoken = self._spoken_site()
        if not label or not spoken or not self._site_conflict(label, spoken):
            return generic
        return (
            f"{label}{_eul(label)} 짚어 주셨는데 {spoken}{_i_ga(spoken)} 불편하다고 하셨어요. "
            "어느 쪽을 적을까요?"
        )

    def _spoken_site(self) -> str | None:
        """환자가 말한 부위. **`value`를 쓴다**(모델이 정리한 부위 이름).

        evidence는 발화 원문이라 `"눈이 불편해요"`처럼 문장이다(2026-09-15). 문장으로는
        온톨로지를 못 찾아서 구체 되묻기 문구가 일반 문구로 조용히 새어나갔다 — 테스트가
        evidence에 낱말(`"눈"`)만 넣고 있어서 못 잡았다. 실제로는 문장이 들어온다.
        """
        if self.spec.site_axis is None:
            return None
        entry = self.card.axes[self.spec.site_axis]
        if entry.value:
            return entry.value
        tag = self.spec.site_preselected_tag
        spoken = [e for e in entry.evidence if not e.startswith(tag)]
        return spoken[-1] if spoken else None

    def _next_axis(self) -> StrEnum | None:
        # 확인이 필요한 축이 먼저, 그 다음 아직 안 물은 축
        for a in self.spec.ask_order:
            if self.card.axes[a].status == FieldStatus.AMBIGUOUS and a not in self.clarified:
                return a
        for a in self.spec.ask_order:
            if self.card.axes[a].status == FieldStatus.NOT_ASKED:
                return a
        return None
