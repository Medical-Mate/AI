"""진료 전 카드 문진 — 무상태 HTTP API.

경계 (docs/ai-design.md §7): AI 서버는 판정·조립만 한다. 인증·세션 식별·저장은 백엔드.
그래서 이 서버는 세션을 기억하지 않는다. 백엔드가 `state`를 들고 다니며 매 턴 보내고,
갱신된 `state`를 받아 저장한다. 서버 여러 대·재시작에 영향받지 않는다.

엔드포인트 두 개:
- POST /v1/previsit/sessions  세션 시작. 첫 질문과 초기 상태. LLM 호출 0.
                              {site_label}이 있으면 부위 사전 채움
- POST /v1/previsit/turns     발화 한 개 처리. 다음 질문·갱신 상태·카드·판정 로그. LLM 호출 ≤1

실행:  uv run uvicorn medimate.api.app:app --reload
설정:  MEDIMATE_PROVIDER(openai|anthropic|google|bedrock|local), MEDIMATE_MODEL
       기본 openai / gpt-5.6-terra. 배포는 bedrock / apac.amazon.nova-pro-v1:0
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from datetime import date
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

from medimate.dialog.engine import Limits, Session
from medimate.dialog.guard import GuardConfig
from medimate.dialog.memo import classify_memo
from medimate.dialog.site import resolve_site
from medimate.dialog.spec import PREVISIT_SPEC
from medimate.dialog.state import SessionState
from medimate.dialog.widening import compare_sites, widen_card
from medimate.llm.base import Extractor, TurnExtraction
from medimate.llm.memo_classifier import FixedLabels, LLMMemoClassifier
from medimate.llm.providers import BudgetExceeded, LLMExtractor
from medimate.ontology import load_ontology
from medimate.schema.card import Axis, Provenance, SiteSelectionRecord
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
    # server(기본): 서버가 추출, 첫 자유 발화 있음. ondevice: 폰이 추출, 첫 축 질문부터, 가드 강화
    profile: str = Field(default="server", pattern="^(server|ondevice)$")
    request_id: str | None = Field(default=None, max_length=128)  # 응답·audit에 그대로 되돌린다


class StartResponse(BaseModel):
    reply: str  # 환자에게 보여줄 첫 질문
    state: SessionState
    request_id: str | None = None


class Selection(BaseModel):
    """칩·폼으로 고른 값. LLM 없이 카드에 바로 들어간다. evidence는 "[선택] 값"."""

    model_config = ConfigDict(extra="forbid")

    axis: Axis
    value: str = Field(min_length=1, max_length=200)


class ExtractionMeta(BaseModel):
    """폰이 추출했을 때 무엇으로 뽑았나. provenance에 기록된다."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(max_length=80)
    prompt_version: str = Field(max_length=40)


class TurnRequest(BaseModel):
    """한 턴. 세 가지 조합 중 하나:
    - utterance만 → 서버가 추출
    - utterance + extraction → 폰이 뽑은 JSON을 쓰고 서버는 가드만. utterance는 근거 검증용
    - selections(±utterance) → 칩·폼 값. utterance가 없으면 LLM 호출 없음
    """

    model_config = ConfigDict(extra="forbid")

    state: SessionState
    utterance: str = Field(default="", max_length=4000)  # 300자 상한은 엔진이 잘라서 알린다
    extraction: TurnExtraction | None = None
    extraction_meta: ExtractionMeta | None = None
    selections: list[Selection] = Field(default_factory=list)
    request_id: str | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def _check(self) -> TurnRequest:
        if self.extraction is not None and not self.utterance.strip():
            raise ValueError("extraction에는 근거 검증용 utterance가 함께 와야 한다")
        return self


class TurnUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class TurnAudit(BaseModel):
    """판정 로그 한 줄 (docs/ai-design.md §7). 백엔드가 감사 기록으로 저장한다."""

    turn: int
    asked_axis: Axis | None
    utterance: str  # 실제로 LLM에 들어간 발화(잘렸으면 잘린 것)
    extraction: TurnExtraction  # 가드를 통과해 카드에 반영된 것
    raw_extraction: TurnExtraction | None = None  # 모델(서버 또는 폰)이 낸 원본
    dropped: list[dict[str, Any]] = Field(default_factory=list)  # 가드가 버린 갱신과 이유
    source: str = "server"  # server | device | none(선택지만)
    usage: TurnUsage


class MemoRequest(BaseModel):
    """진료 후 메모 하나 → 4묶음 카드 (와이어프레임 1p → 1q).

    - labels 없음: 서버가 문장 분류 LLM을 부른다
    - labels 있음: 폰이 이미 분류한 결과({"0": 라벨, …}). 서버는 LLM 없이 카드만 조립한다
    - classify=false (labels 없음): **문장만 나눠 돌려준다.** LLM 없음. 폰이 분류할 때 1단계
    """

    model_config = ConfigDict(extra="forbid")

    memo: str = Field(min_length=1, max_length=2000)
    visit_date: date | None = None  # 재방문 날짜 계산 기준. 앱이 준다
    clinic: str | None = Field(default=None, max_length=80)  # "서울OO병원 내과". 앱이 준다
    labels: dict[str, str] | None = None  # 폰 분류 결과 또는 1q-2에서 고친 라벨
    classify: bool = (
        True  # false면 서버 LLM을 부르지 않고 sentences만(폰 분류 1단계). labels가 있으면 무시
    )
    labels_meta: ExtractionMeta | None = None  # labels가 폰 모델에서 왔으면 무엇으로
    previsit_anchor_id: str | None = Field(default=None, max_length=40)  # 진료 전 부위(대조용)
    request_id: str | None = Field(default=None, max_length=128)


class MemoResponse(BaseModel):
    card: dict[str, Any]  # 백엔드 형식 카드(card_type=postvisit)
    sentences: list[str]  # 우리가 나눈 문장. 1q-2 수정 화면이 이 번호로 라벨을 바꾼다
    labels: dict[str, str]  # 번호 → 라벨(none 포함)
    dropped: list[dict[str, Any]]  # 가드가 무시한 라벨
    source: str  # server | client | none(문장 분리만)
    usage: TurnUsage
    request_id: str | None = None


class TurnResponse(BaseModel):
    reply: str  # 다음 질문 또는 마무리 문장
    ended: bool
    end_reason: str | None
    state: SessionState  # 다음 턴에 그대로 보낼 것
    card: dict[str, Any]  # 백엔드 형식 카드(schema/export.py). 매 턴 그 시점 카드
    audit: TurnAudit | None  # LLM을 부르지 않은 턴(빈 입력·상한 종료)은 None
    request_id: str | None = None


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


def create_app(
    extractor_factory: ExtractorFactory | None = None, memo_factory: Callable | None = None
) -> FastAPI:
    app = FastAPI(
        title="진료 메이트 AI — 진료 전 카드",
        version="0.1.0",
        description="무상태 문진 API. 진단·감별을 하지 않는다. 출력에 병명 자리가 없다.",
    )
    app.state.extractor_factory = extractor_factory
    from medimate.api import auth

    auth.install(app)  # MEDIMATE_HMAC_SECRET 없으면 검증 생략
    app.state.limits = Limits()
    app.state.ontology = None  # 첫 요청에 로드. data/ontology CSV, 로드 시 검증
    app.state.memo_factory = (
        memo_factory  # 진료 후 메모 분류기. 테스트는 create_app(memo_factory=...)
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/previsit/sessions", response_model=StartResponse)
    def start_session(extractor: Ex, body: StartRequest | None = None) -> StartResponse:
        ondevice = bool(body and body.profile == "ondevice")
        s = Session(
            extractor,
            limits=app.state.limits,
            profile="ondevice" if ondevice else "server",
            guard=GuardConfig.ondevice() if ondevice else GuardConfig(),
            skip_open_ended=ondevice,
        )
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
        return StartResponse(
            reply=s.opening(), state=s.to_state(), request_id=body.request_id if body else None
        )

    @app.post("/v1/previsit/turns", response_model=TurnResponse)
    def process_turn(body: TurnRequest, extractor: Ex) -> TurnResponse:
        s = Session.from_state(extractor, body.state, limits=app.state.limits)
        if body.extraction_meta is not None and s.card.provenance is not None:
            # 폰이 뽑았으면 어느 모델·프롬프트였는지 카드에 남긴다
            s.card.provenance.model_id = body.extraction_meta.model_id
            s.card.provenance.prompt_version = body.extraction_meta.prompt_version
        selections = [(sel.axis, sel.value) for sel in body.selections]
        try:
            reply = s.step(body.utterance, extraction=body.extraction, selections=selections)
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
            if body.extraction is not None:
                source = "device"
            elif log.utterance:
                source = "server"
            else:
                source = "none"
            audit = TurnAudit(
                turn=log.turn,
                asked_axis=log.asked_axis,
                utterance=log.utterance,
                extraction=log.extraction,
                raw_extraction=log.raw_extraction,
                dropped=log.dropped,
                source=source,
                usage=tu if source == "server" else TurnUsage(),
            )

        return TurnResponse(
            reply=reply,
            ended=s.ended,
            end_reason=s.end_reason,
            state=s.to_state(),
            card=to_backend_payload(s.card),
            audit=audit,
            request_id=body.request_id,
        )

    def _memo_classifier(request: Request):
        factory = request.app.state.memo_factory
        if factory is None:
            provider = os.getenv("MEDIMATE_PROVIDER", DEFAULT_PROVIDER)
            model = os.getenv("MEDIMATE_MODEL", DEFAULT_MODEL)
            shared: dict[str, object] = {}

            def make():
                c = LLMMemoClassifier(provider, model, client=shared.get("client"))
                return c

            factory = request.app.state.memo_factory = make
        return factory()

    @app.post("/v1/postvisit/memo", response_model=MemoResponse)
    def postvisit_memo(body: MemoRequest, request: Request) -> MemoResponse:
        """메모 → 4묶음 카드. labels가 오면 LLM 없이 조립(폰 분류·1q-2 수정 모두 이 경로)."""
        if body.labels is not None:
            clf = FixedLabels(body.labels)
            source = "client"
        elif not body.classify:
            # 폰 분류 1단계: 문장 번호만 필요. 전부 unsorted인 카드가 나오지만 앱은 sentences만 쓴다
            clf = FixedLabels({})
            source = "none"
        else:
            clf = _memo_classifier(request)
            source = "server"
        try:
            res = classify_memo(body.memo, clf, visit_date=body.visit_date, clinic=body.clinic)
        except BudgetExceeded as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=502, detail=f"classifier: {e}") from e

        card = res.card
        card.provenance = Provenance(
            prompt_version=(
                body.labels_meta.prompt_version if body.labels_meta else clf.prompt_version
            ),
            model_id=body.labels_meta.model_id if body.labels_meta else clf.model_id,
        )
        # 소견 용어에 부위 병기 + 진료 전 부위와 대조(판정 아님)
        if app.state.ontology is None:
            app.state.ontology = load_ontology()
        notes = widen_card(card, app.state.ontology)
        card.site_comparison = compare_sites(app.state.ontology, body.previsit_anchor_id, notes)

        usage = getattr(clf, "usage", None)
        tu = TurnUsage()
        if usage is not None and source == "server":
            tu = TurnUsage(
                input_tokens=int(usage.input_tokens),
                output_tokens=int(usage.output_tokens),
                cost_usd=round(float(usage.cost_usd(clf.model_id)), 6),
            )
        return MemoResponse(
            card=to_backend_payload(card),
            sentences=res.sentences,
            labels={str(i): res.labels.get(i, "none") for i in range(len(res.sentences))},
            dropped=res.dropped,
            source=source,
            usage=tu,
            request_id=body.request_id,
        )

    @app.get("/v1/ontology/body-map")
    def body_map() -> dict[str, Any]:
        """부위 마스터 — 인체도가 짚을 수 있는 앵커·구역과 진료과 안내. 앱·백엔드 공용, 읽기 전용.

        노드에 적힌 값을 그대로 낸다. 선택·정렬 로직 없음. 진료과 안내는 우리 콘텐츠(인용 아님).
        """
        if app.state.ontology is None:
            app.state.ontology = load_ontology()
        onto = app.state.ontology

        def image_key(n) -> str:
            # 앱 이미지 자산 파일명 키. 영문 이름 슬러그라 ID가 바뀌어도 유지된다
            # 예: head, lower-back-and-hip
            return re.sub(r"[^a-z0-9]+", "-", n.name_en.lower()).strip("-")

        def node(n) -> dict[str, Any]:
            return {
                "id": n.id,
                "label": n.display_name,
                "image_key": image_key(n),  # 앱이 id→파일을 하드코딩하지 않게
                "aliases": list(
                    n.aliases
                ),  # 폼 검색용 유의어(복부→배, 옆구리→허리 옆). 앱이 로컬에서 매칭
                "laterality": n.laterality,  # none | left_right
                "view": n.view,  # front | back | none(사이드 탭)
                "departments": list(n.departments),
                "departments_source": n.departments_source,
            }

        anchors = []
        for a in onto.anchors_in_order():
            d = node(a)
            d["region"] = onto.region_of(a.id)
            d["zones"] = [node(z) for z in onto.zones(a.id)]
            anchors.append(d)
        return {
            "ontology_snapshot": onto.snapshot_id,
            "axes": [ax.value for ax in PREVISIT_SPEC.axis_type],
            "anchors": anchors,
            "image_key_note": (
                "image_key는 앵커·구역 이미지 자산의 파일명 키(예: head, lower-back-and-hip). "
                "이미지는 앱 자산이고 서버는 키만 낸다. "
                "부위가 늘면 새 키 + 앱에 그 이미지가 필요하다"
            ),
            "note": "진료과 안내는 팀 콘텐츠(의료인 자문 확인 전). 증상과 무관하게 부위에만 붙는다",
        }

    @app.get("/v1/ontology/search")
    def ontology_search(q: str, limit: int = 8) -> dict[str, Any]:
        """폼 입력으로 부위 찾기 — 유의어 표 매칭(복부→배, 옆구리→허리 옆). LLM 없음.

        앱은 body-map의 aliases로 같은 매칭을 로컬에서 해도 된다. 이 엔드포인트는 서버 쪽 소비자와
        규칙을 한 곳에 두기 위한 것이다. 증상·병명은 매칭하지 않는다(부위 이름·유의어만).

        q는 발화 상한과 같은 값으로 자른다(팀 합의 300자). 정상 입력은 닿지 않는다 — 부위 이름은
        길어야 열 글자다. 엔진과 같은 방식(거절이 아니라 자르기)으로 맞춘다.
        """
        if app.state.ontology is None:
            app.state.ontology = load_ontology()
        onto = app.state.ontology
        q = q[: app.state.limits.max_utterance_chars]
        out = []
        for n, matched, score in onto.search(q, limit=max(1, min(limit, 20))):
            anchor_id = n.id if n.kind == "anchor" else onto.anchor_of(n.id)
            out.append(
                {
                    "id": n.id,
                    "label": n.display_name,
                    "kind": n.kind,
                    "anchor_id": anchor_id,
                    "matched": matched,
                    "score": score,
                }
            )
        return {"query": q, "results": out, "ontology_snapshot": onto.snapshot_id}

    return app


app = create_app()
