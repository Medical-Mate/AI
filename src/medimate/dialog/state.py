"""문진 세션의 직렬화 가능한 상태.

AI 서버는 무상태다(docs/ai-design.md §7 — 세션·저장은 백엔드). 한 턴을 처리하려면
백엔드가 이 상태를 그대로 보내고, 갱신된 상태를 그대로 받아 보관한다.
AI 서버는 상태를 기억하지 않으므로 서버 여러 대·재시작에 영향받지 않는다.

여기 담긴 것은 다음 턴을 결정하는 데 필요한 것만이다. 판정 로그(TurnLog)는
턴 응답으로 매번 돌려주고 여기에는 쌓지 않는다 — 백엔드가 감사 기록으로 저장한다.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from medimate.schema.card import InterviewCard

# (직전 질문, 환자 답). Extractor에 넘기는 최근 대화
HistoryTurn = tuple[str, str]


class SessionState(BaseModel):
    spec: str = "previsit"  # 어느 문진인가. dialog/spec.py SPECS 키 (previsit | postvisit)
    # server: 서버가 추출(Terra), 첫 자유 발화 있음, 가드는 근거·숫자만
    # ondevice: 폰이 추출해 JSON을 보냄, 첫 축 질문부터, 가드에 "물은 축만" 추가
    profile: str = "server"
    card: InterviewCard = Field(default_factory=InterviewCard)
    asked_axis: str | None = None  # 직전에 물은 축(값 문자열). None이면 첫 발화 대기
    clarified: list[str] = Field(default_factory=list)  # 확인 질문을 이미 한 축(축당 한 번)
    history: list[HistoryTurn] = Field(default_factory=list)  # 최근 N턴만 유지
    turn: int = 0  # 처리한 발화 수(빈 입력은 세지 않는다)
    message_asked: bool = False  # 8축 뒤 "전하고 싶은 말" 질문을 냈는가. 답이 오면 종료
    session_tokens: int = 0  # 이 세션의 LLM 입력+출력 누적. 세션 예산 상한 판정용
    ended: bool = False
    end_reason: str | None = None  # None | "stop" | "complete" | "max_turns" | "budget"
