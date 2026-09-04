"""진료 전 카드 문진 — 무상태 HTTP API.

경계 (docs/ai-design.md §7): AI 서버는 판정·조립만 한다. 인증·세션 식별·저장은 백엔드.
그래서 이 서버는 세션을 기억하지 않는다. 백엔드가 `state`를 들고 다니며 매 턴 보내고,
갱신된 `state`를 받아 저장한다. 서버 여러 대·재시작에 영향받지 않는다.

엔드포인트 두 개:
- POST /v1/previsit/sessions  세션 시작. 첫 질문과 초기 상태. LLM 호출 0.
                              {site_label}이 있으면 부위 사전 채움
- POST /v1/previsit/turns     발화 한 개 처리. 다음 질문·갱신 상태·카드·판정 로그. LLM 호출 ≤1

실행:  uv run uvicorn medimate.api.app:app --reload
설정:  MEDIMATE_PROVIDER(openai|anthropic|google), MEDIMATE_MODEL — 기본 openai / gpt-5.6-terra
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from medimate.dialog.engine import Limits, Session
from medimate.dialog.site import resolve_site
from medimate.dialog.state import SessionState
from medimate.llm.base import Extractor, TurnExtraction
from medimate.llm.providers import BudgetExceeded, LLMExtractor
from medimate.ontology import load_ontology
from medimate.schema.card import Axis, SiteSelectionRecord
from medimate.schema.export import to_backend_payload

DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = "gpt-5.6-terra"  # docs/decisions/2026-09-03-extractor-model.md

ExtractorFactory = Callable[[], Extractor]


# --- 요청·응답 -----------------------------------------------------------
class StartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # 인체도 선택 (앱 1l). 노드 ID(앵커 또는 구역) + 좌우. SITE 축을 채우고 첫 질문을 앵커하며
    # 진료과 안내가 카드에 붙는다
    site_node_id: str | None = Field(default=None, max_length=40)
    side: str | None = Field(default=None, pattern="^(left|right|both)$")
    # 온톨로지 없이 라벨만 넘길 때(테스트·임시). site_node_id가 있으면 무시된다
    site_label: str | None = Field(default=None, max_length=40)


class StartResponse(BaseModel):
    reply: str  # 환자에게 보여줄 첫 질문
    state: SessionState


class TurnRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: SessionState
    utterance: str = Field(max_length=4000)  # 300자 상한은 엔진이 잘라서 알린다. 이건 남용 방어


class TurnUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class TurnAudit(BaseModel):
    """판정 로그 한 줄 (docs/ai-design.md §7). 백엔드가 감사 기록으로 저장한다."""

    turn: int
    asked_axis: Axis | None
    utterance: str  # 실제로 LLM에 들어간 발화(잘렸으면 잘린 것)
    extraction: TurnExtraction
    usage: TurnUsage


class TurnResponse(BaseModel):
    reply: str  # 다음 질문 또는 마무리 문장
    ended: bool
    end_reason: str | None
    state: SessionState  # 다음 턴에 그대로 보낼 것
    card: dict[str, Any]  # 백엔드 형식 카드(schema/export.py). 매 턴 그 시점 카드
    audit: TurnAudit | None  # LLM을 부르지 않은 턴(빈 입력·상한 종료)은 None


# --- 앱 -----------------------------------------------------------------
def _default_factory() -> ExtractorFactory:
    """요청마다 새 LLMExtractor(usage는 요청 단위), SDK 클라이언트는 공유."""
    try:
        from dotenv import load_dotenv

        load_dotenv()  # 저장소 루트 .env. 없으면 조용히 넘어간다
    except ImportError:
        pass
    provider = os.getenv("MEDIMATE_PROVIDER", DEFAULT_PROVIDER)
    model = os.getenv("MEDIMATE_MODEL", DEFAULT_MODEL)
    shared: dict[str, object] = {}

    def make() -> Extractor:
        ex = LLMExtractor(provider, model, _client=shared.get("client"))
        return _ClientCaching(ex, shared)

    return make


class _ClientCaching:
    """첫 호출 뒤 생성된 SDK 클라이언트를 다음 요청이 재사용하도록 붙잡아 둔다."""

    def __init__(self, inner: LLMExtractor, shared: dict[str, object]):
        self._inner = inner
        self._shared = shared

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def extract(self, utterance, asked_axis, history=()) -> TurnExtraction:
        try:
            return self._inner.extract(utterance, asked_axis, history)
        finally:
            if self._inner._client is not None:
                self._shared["client"] = self._inner._client


def get_extractor(request: Request) -> Extractor:
    factory = request.app.state.extractor_factory
    if factory is None:
        factory = request.app.state.extractor_factory = _default_factory()
    return factory()


# 모듈 수준에 둔다 — `from __future__ import annotations` 아래에서 FastAPI가 문자열
# 애너테이션을 모듈 전역으로 해석하므로, 함수 안의 지역 별칭은 본문 파라미터로 오해된다
Ex = Annotated[Extractor, Depends(get_extractor)]


def create_app(extractor_factory: ExtractorFactory | None = None) -> FastAPI:
    app = FastAPI(
        title="진료 메이트 AI — 진료 전 카드",
        version="0.1.0",
        description="무상태 문진 API. 진단·감별을 하지 않는다. 출력에 병명 자리가 없다.",
    )
    app.state.extractor_factory = extractor_factory
    app.state.limits = Limits()
    app.state.ontology = None  # 첫 요청에 로드. data/ontology CSV, 로드 시 검증

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/previsit/sessions", response_model=StartResponse)
    def start_session(extractor: Ex, body: StartRequest | None = None) -> StartResponse:
        s = Session(extractor, limits=app.state.limits)
        if body and body.site_node_id:
            if app.state.ontology is None:
                app.state.ontology = load_ontology()
            try:
                sel = resolve_site(app.state.ontology, body.site_node_id, body.side)
            except ValueError as e:
                raise HTTPException(status_code=422, detail=str(e)) from e
            s.preselect_site(sel.label)
            s.card.site_selection = SiteSelectionRecord(**sel.model_dump())
            s.card.provenance.ontology_snapshot = sel.ontology_snapshot
        elif body and body.site_label:
            s.preselect_site(body.site_label)
        return StartResponse(reply=s.opening(), state=s.to_state())

    @app.post("/v1/previsit/turns", response_model=TurnResponse)
    def process_turn(body: TurnRequest, extractor: Ex) -> TurnResponse:
        s = Session.from_state(extractor, body.state, limits=app.state.limits)
        try:
            reply = s.step(body.utterance)
        except BudgetExceeded as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        except ValueError as e:
            # 모델 출력 파싱 실패. 상태는 바뀌지 않았다 — 백엔드는 같은 상태로 재시도 가능
            raise HTTPException(status_code=502, detail=f"extractor: {e}") from e

        audit = None
        if s.logs:
            log = s.logs[-1]
            usage = getattr(extractor, "usage", None)
            tu = TurnUsage()
            if usage is not None:
                tu = TurnUsage(
                    input_tokens=int(getattr(usage, "input_tokens", 0)),
                    output_tokens=int(getattr(usage, "output_tokens", 0)),
                    cost_usd=round(float(usage.cost_usd(extractor.model_id)), 6)
                    if hasattr(usage, "cost_usd")
                    else 0.0,
                )
            audit = TurnAudit(
                turn=log.turn,
                asked_axis=log.asked_axis,
                utterance=log.utterance,
                extraction=log.extraction,
                usage=tu,
            )

        return TurnResponse(
            reply=reply,
            ended=s.ended,
            end_reason=s.end_reason,
            state=s.to_state(),
            card=to_backend_payload(s.card),
            audit=audit,
        )

    return app


app = create_app()
