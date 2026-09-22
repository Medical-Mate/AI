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
       기본값 = 운영값 = bedrock / apac.amazon.nova-pro-v1:0 (2026-09-15부터)
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable
from datetime import date
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from medimate.api import auth
from medimate.api.budget import BudgetGuarded, DailyBudget
from medimate.api.state_guard import check_state
from medimate.dialog.engine import Limits, Session
from medimate.dialog.guard import GuardConfig
from medimate.dialog.memo import (
    SPLIT_VERSION,
    assemble_from_client,
    classify_memo,
    lifestyle_axis_enabled,
    segment_memo,
    split_sentences,
)
from medimate.dialog.site import normalize_side, resolve_site, strip_side
from medimate.dialog.spec import PREVISIT_SPEC
from medimate.dialog.state import SessionState
from medimate.dialog.widening import compare_sites, widen_card
from medimate.llm import assist_prompts as ap
from medimate.llm.assist_rank import top_candidates
from medimate.llm.base import Extractor, TurnExtraction
from medimate.llm.memo_classifier import (
    FixedLabels,
    LLMFollowUpReader,
    LLMMemoClassifier,
    LLMMemoSegmenter,
)
from medimate.llm.providers import (
    PRICES,
    PROMPTS,
    BudgetExceeded,
    LLMExtractor,
    prompt_family_for,
)
from medimate.obs import tracing
from medimate.ontology import load_ontology
from medimate.schema.card import (
    Axis,
    FieldStatus,
    PreVisitCard,
    Provenance,
    SiteSelectionRecord,
)
from medimate.schema.export import to_backend_payload

# 앱 4단계 화면이 3개를 보여준다. 생성은 제한하지 않고 가중치로 정렬해 위에서 자른다
QUESTION_CANDIDATES_TOP = 3

# 기본값이 곧 운영값이다(2026-09-15). 전에는 openai / gpt-5.6-terra였는데, env 한 줄이
# 빠지면 **조용히 다른 회사 모델로 떨어지고 다른 카드로 돈이 나갔다** — 로컬 점검 스크립트가
# 실제로 그렇게 OpenAI를 불렀다. 운영·평가·기본값이 전부 Nova Pro다(팀 확정 2026-09-11).
# Terra는 `docs/decisions/2026-09-03-extractor-model.md`의 비교 기준으로만 남는다.
DEFAULT_PROVIDER = "bedrock"
DEFAULT_MODEL = "apac.amazon.nova-pro-v1:0"

ExtractorFactory = Callable[[], Extractor]


# --- 요청·응답 -----------------------------------------------------------
class StartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # 인체도 선택 (앱 1l). 노드 ID(앵커 또는 구역) + 좌우. SITE 축을 채우고 첫 질문을 앵커하며
    # 진료과 안내가 카드에 붙는다
    site_node_id: str | None = Field(default=None, max_length=40)
    # 대소문자를 가리지 않는다(2026-09-11) — 백엔드가 `LEFT`로 보내 422가 났다.
    # 검증 **전에** 소문자로 접으므로 `LEFT`·`Left`·`left` 모두 통과하고 카드에는 소문자로 남는다
    side: str | None = Field(default=None, pattern="^(left|right|both)$")

    @field_validator("side", mode="before")
    @classmethod
    def _fold_side(cls, v):
        return normalize_side(v) if isinstance(v, str) else v

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


class PatientProfile(BaseModel):
    """환자가 앱 온보딩에 적은 복용약·기저질환·알러지. **질문 후보 재료로만 쓰고 버린다.**

    카드에 넣지 않는다 — 축이 아니고, 이 문진에서 환자가 말한 것도 아니다. 정규화·해석도
    하지 않는다(약 이름을 표준명으로 바꾸는 순간 우리가 만든 값이 된다). 프롬프트에 그대로
    실리고 응답과 함께 사라진다.

    eval에서 이 셋이 질문 후보의 복용약·기저질환·알러지 카테고리를 3% → 65%로 올렸다.
    와이어프레임의 3개 중 하나가 약 질문인데, 그 자리가 이 필드 없이는 채워지지 않는다.

    `null` 상태를 따로 두지 않는다 — **필드가 없으면 온보딩을 안 거친 것, `[]`면 적었는데
    없는 것**이다. 지금은 둘 다 프롬프트에 "없음"으로 가고 그 이상은 만들지 않는다.
    """

    model_config = ConfigDict(extra="forbid")

    medications: list[str] = Field(default_factory=list, max_length=50)
    conditions: list[str] = Field(default_factory=list, max_length=50)
    allergies: list[str] = Field(default_factory=list, max_length=50)

    @field_validator("medications", "conditions", "allergies", mode="after")
    @classmethod
    def _drop_blanks(cls, v: list[str]) -> list[str]:
        """빈 칸·공백만 남은 원소를 버린다.

        앱이 빈 입력칸을 그대로 보내면 프롬프트에 빈 줄이 들어간다.
        """
        return [x.strip() for x in v if x and x.strip()]

    def as_prompt_payload(self) -> dict[str, list[str]]:
        return {
            "medications": self.medications,
            "conditions": self.conditions,
            "allergies": self.allergies,
        }


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

    # 종료 턴에 "의사에게 물어볼 것" 후보를 만들지. **기본 false — 안 보내면 안 돈다.**
    # 옵트인인 이유가 둘이다. ① 온디바이스 프로필은 "발화가 외부로 안 나간다"가 전제인데
    # 후보 생성은 카드 값을 외부 LLM으로 보낸다 ② 백엔드 AWS 크레딧이 한정이고 Bedrock이
    # 같은 크레딧에서 나간다. 무조건 도는 구조면 끌 방법이 없다
    question_candidates: bool = False

    # 앱 온보딩의 복용약·기저질환·알러지. **`question_candidates`가 켜진 턴에만 보낼 수 있다.**
    # 필드를 생략하면 지금까지와 같다(빈 배열). 카드에 저장되지 않는다
    patient_profile: PatientProfile | None = None

    @model_validator(mode="after")
    def _profile_only_with_candidates(self) -> TurnRequest:
        """후보를 안 만드는 턴에 건강정보가 오면 거부한다.

        프롬프트에 들어가는 것이 문제가 아니라 **요청 본문에 실려 오는 것**이 문제다.
        편의상 매 턴 붙여 보내기 시작하면 20턴짜리 문진에서 건강정보가 20번 오가고,
        서버 로그·에러 리포트에 남는 표면이 그만큼 늘어난다. 아무도 모르게.
        규칙으로 부탁하지 않고 구조로 막는다(CLAUDE.md).
        """
        if self.patient_profile is not None and not self.question_candidates:
            raise ValueError(
                "patient_profile은 question_candidates: true인 턴에만 보낼 수 있습니다"
            )
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
    # 앞 응답의 `sentences`를 **그대로** 되보낸다(memo-v5). 있으면 서버는 다시 나누지 않고
    # 이 조각에 라벨을 붙인다 — 번호가 다른 문장을 가리킬 길이 없어진다. 원문을 못 덮으면 409
    sentences: list[str] | None = Field(default=None, max_length=200)
    classify: bool = (
        True  # false면 서버 LLM을 부르지 않고 sentences만(폰 분류 1단계). labels가 있으면 무시
    )
    labels_meta: ExtractionMeta | None = None  # labels가 폰 모델에서 왔으면 무엇으로
    previsit_anchor_id: str | None = Field(default=None, max_length=40)  # 진료 전 부위(대조용)
    request_id: str | None = Field(default=None, max_length=128)

    # 앞 응답에서 받은 분리 규칙 이름. **`labels`를 보낼 때 같이 보낸다.**
    # 다르면 409 — 그 사이 규칙이 바뀌어 번호가 다른 문장을 가리키게 됐다는 뜻이다
    split_version: str | None = Field(default=None, max_length=40)


class MemoAudit(BaseModel):
    """진료 후 메모 판정 로그 한 덩이 — previsit `TurnAudit`과 같은 자리(#113, 2026-09-21).

    백엔드가 가공 없이 JSON으로 적재한다. 심사 기간 실사용 메모를 회귀 eval 재료로 쓰려는 것이라
    **모델이 낸 원본 출력**(`raw_output`)과 **접히기 전 경로**를 남긴다. 식별자는 없다 — 메모 원문
    안에 환자가 적은 것 외에는 아무것도 들어가지 않는다.
    """

    memo: str  # 실제로 처리한 메모 원문
    sentences: list[str]  # 조각(주소). 규칙 분리든 LLM 조각이든 되보낸 것이든 최종 사용분
    labels: dict[str, str]  # 번호 → 라벨. 지침 축이 접혀 있으면 접힌 뒤 값(응답 labels와 같다)
    raw_output: str | None = None  # 모델이 낸 원본 텍스트(분류기 또는 세그멘터). client 경로는 None
    dropped: list[dict[str, Any]] = Field(default_factory=list)
    source: str  # server | client | none
    prompt_version: str  # memo-v6(조각) | memo-small-v5(규칙+분류) | client-labels | …
    model_id: str
    lifestyle_axis: bool  # 지침 축이 따로 나갔는가(false면 약 칸에 접힘)
    split_version: str = SPLIT_VERSION
    usage: TurnUsage


class MemoResponse(BaseModel):
    card: dict[str, Any]  # 백엔드 형식 카드(card_type=postvisit)
    sentences: list[str]  # 우리가 나눈 문장. 1q-2 수정 화면이 이 번호로 라벨을 바꾼다
    labels: dict[str, str]  # 번호 → 라벨(none 포함)
    dropped: list[dict[str, Any]]  # 가드가 무시한 라벨
    source: str  # server | client | none(문장 분리만)
    # 이 `sentences` 번호를 만든 분리 규칙. 라벨을 되보낼 때 그대로 실어 보낸다
    split_version: str = SPLIT_VERSION
    usage: TurnUsage
    # 판정 로그. previsit의 `audit`과 같은 자리 — 백엔드가 그대로 적재한다(#113)
    audit: MemoAudit | None = None
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
def _install_validation_handler(app: FastAPI) -> None:
    """422 본문에서 **요청 값을 벗긴다.** `loc`·`msg`·`type`만 남긴다.

    pydantic 기본 핸들러는 `detail[].input`에 문제가 된 값을 그대로 넣는다. 그래서 잘못된
    요청 하나가 `patient_profile`(복용약·기저질환·알러지)이나 `state`(환자 발화 원문)를
    **응답에 실어 되돌려준다.**

    백엔드는 `detail`을 밖으로 안 내보내지만, 웹이 우리를 직접 부르는 구성(CORS)이 되면
    그 본문이 **브라우저 콘솔과 에러 리포트에 남는다.** 409 `detail`에 메모를 안 넣기로 한 것과
    같은 자리다 — 진단에 필요한 것은 "어느 필드가 왜 틀렸나"이지 그 값이 아니다.

    `loc`은 남긴다. 백엔드가 그걸로 어느 필드인지 본다.
    """

    @app.exception_handler(RequestValidationError)
    def _handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        safe = [
            {"loc": list(e.get("loc", [])), "msg": e.get("msg", ""), "type": e.get("type", "")}
            for e in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": safe})


CORS_ENV = "MEDIMATE_CORS_ORIGINS"


def _install_cors(app: FastAPI) -> list[str]:
    """`MEDIMATE_CORS_ORIGINS`가 있을 때만 미들웨어를 붙인다. 없으면 아무것도 안 한다.

    웹 데모가 브라우저에서 우리를 **직접** 부를 때만 필요하다. 백엔드가 프록시하면 요청이
    서버에서 오므로 CORS는 개입하지 않는다. 아직 안 정해졌으니 env 하나로 켤 수 있게만 둔다.

    **`*`는 받지 않는다.** 브라우저에는 HMAC 시크릿을 둘 수 없어서 CORS를 여는 순간 그 출처가
    사실상 유일한 문지기가 된다. `*`면 문지기가 없는 것과 같다.
    """
    raw = (os.getenv(CORS_ENV) or "").strip()
    if not raw:
        return []
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    if "*" in origins:
        raise ValueError(
            f"{CORS_ENV}에 '*'는 쓸 수 없습니다 — 브라우저에 HMAC 시크릿을 둘 수 없어"
            " 출처 목록이 유일한 문지기입니다. 도메인을 적어 주세요"
        )
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,  # 쿠키를 안 쓴다. 켜면 `*` 금지와 같은 이유로 위험만 는다
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-Signature", "X-Timestamp", "X-Request-Id"],
    )
    return origins


def extractor_env() -> tuple[str, str]:
    """env가 정한 (공급자, 모델). **읽는 자리는 여기 하나다.**

    `strip()`을 건다. 프롬프트 계열은 모델 id로 고르는데 정확히 일치해야 해서, `.env`에
    따라붙은 공백 하나로 `extract-v4-nova`가 `extract-v3`로 조용히 떨어진다
    (88케이스 84 → 76, 인젝션 3/3 → 0/3). 감지보다 없애는 쪽이 낫다.

    `/health`와 실제 호출이 **같은 값**을 봐야 하므로 둘 다 이 함수를 쓴다.
    """
    provider = os.getenv("MEDIMATE_PROVIDER", DEFAULT_PROVIDER).strip() or DEFAULT_PROVIDER
    model = os.getenv("MEDIMATE_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    return provider, model


def _default_factory() -> ExtractorFactory:
    """요청마다 새 LLMExtractor(usage는 요청 단위), SDK 클라이언트는 공유."""
    try:
        from dotenv import load_dotenv

        load_dotenv()  # 저장소 루트 .env. 없으면 조용히 넘어간다
    except ImportError:
        pass
    provider, model = extractor_env()
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
    # 일일 상한은 **실제로 LLM을 부르는 자리**에 붙인다(테스트 대역도 같은 길로 지난다)
    return BudgetGuarded(factory(), request.app.state.daily_budget)


# 모듈 수준에 둔다 — `from __future__ import annotations` 아래에서 FastAPI가 문자열
# 애너테이션을 모듈 전역으로 해석하므로, 함수 안의 지역 별칭은 본문 파라미터로 오해된다
Ex = Annotated[Extractor, Depends(get_extractor)]


def create_app(
    extractor_factory: ExtractorFactory | None = None,
    memo_factory: Callable | None = None,
    followup_factory: Callable | None = None,
    segment_factory: Callable | None = None,
) -> FastAPI:
    app = FastAPI(
        title="진료 메이트 AI — 진료 전 카드",
        version="0.1.0",
        description="무상태 문진 API. 진단·감별을 하지 않는다. 출력에 병명 자리가 없다.",
    )
    app.state.extractor_factory = extractor_factory

    _install_validation_handler(app)
    auth.install(app)  # MEDIMATE_HMAC_SECRET 없으면 검증 생략
    app.state.daily_budget = DailyBudget.from_env()  # 변수 없으면 꺼진 상태
    app.state.cors_origins = _install_cors(app)
    app.state.limits = Limits()
    app.state.ontology = None  # 첫 요청에 로드. data/ontology CSV, 로드 시 검증
    app.state.memo_factory = (
        memo_factory  # 진료 후 메모 분류기. 테스트는 create_app(memo_factory=...)
    )
    # 재방문 표현 리더. 주입 없으면 운영은 _memo_classifier가 분류기와 짝으로 채운다
    app.state.followup_factory = followup_factory
    app.state.segment_factory = segment_factory

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        """상태 + **서버가 실제로 검증하는 서명 규격.**

        `signing`은 `auth.sign()`이 쓰는 템플릿에서 나온다(손으로 관리하는 버전 번호가 아니다).
        점검 도구가 자기 것과 문자열 비교해 **로컬 코드와 배포본이 갈린 상태**를 잡는다.
        2026-09-11에 그게 안 잡혀서 벡터 10 pass인데 운영만 401이 났다.

        `hmac_enforced`는 무검증으로 떠 있는 걸 아무도 모르는 상태를 없앤다. 노출이 아니다 —
        꺼져 있으면 서명 없이 아무 요청이나 통과하니 이미 알 수 있는 사실이고,
        켜져 있으면 "켜져 있다"는 정보에 값이 없다.
        """
        cfg = getattr(request.app.state, "hmac", None)
        return {
            "status": "ok",
            "hmac_enforced": bool(cfg and cfg.secret),
            # 서버가 `MEDIMATE_REQUIRE_HMAC`을 **실제로 읽었는지**를 낸다. 2026-09-11에
            # 백엔드가 그 env를 넣고 켠 줄 알았는데 우리 코드에 변수가 없어 아무 일도 안
            # 일어났고, `hmac_enforced`는 시크릿 유무만 보므로 그걸 잡아주지 못했다.
            # 값이 `true`면 플래그가 읽혔다는 뜻이다
            # (켜졌는데 시크릿이 없으면 기동 자체가 안 된다).
            "hmac_required": bool(cfg and cfg.require),
            "signing": auth.signing_spec(),
            # 우리 가격표 기준 **추정치**다(청구서와 다를 수 있다). 밖에서 보이게 두는 이유는
            # `hmac_required`와 같다 — "켠 줄 알았는데 안 켜진" 상태를 없앤다. 꺼져 있으면 null
            "llm_budget": request.app.state.daily_budget.status(),
            "cors_origins": len(request.app.state.cors_origins) or None,
            # 어떤 모델·프롬프트로 뜰 것인가. **턴을 태우지 않고** 확인하라고 낸다.
            #
            # 프롬프트 계열은 모델 id로 고르는데 **정확히 일치**해야 한다. 뒤에 공백 하나,
            # `apac.` 누락, 대문자 — 무엇이든 어긋나면 조용히 `extract-v3`로 떨어진다.
            # 그때 88케이스 84 → 76, 인젝션 방어 3/3 → 0/3이 되는데 200도 나오고 카드도
            # 멀쩡하다. 배포 직후 `curl /health` 한 번으로 그 상태를 없앤다.
            #
            # 여기 나오는 것은 **설정값**(이 서버가 쓸 것)이다. 그 턴이 실제로 무엇으로
            # 돌았는지는 `card.provenance`가 낸다. 둘 다 필요하다.
            "extractor": _extractor_config(),
        }

    def _extractor_config() -> dict[str, object]:
        """env가 정한 공급자·모델과, 그 모델이 쓸 프롬프트 버전.

        LLM 클라이언트를 만들지 않는다 — 자격증명 없이도 `/health`는 200이어야 한다.
        `pricing_known`이 false면 가격표에 없는 모델이라 **지출 상한이 세지 못한다**.
        """
        provider, model = extractor_env()
        return {
            "provider": provider,
            "model_id": model,
            "prompt_version": PROMPTS[prompt_family_for(model)].PROMPT_VERSION,
            "pricing_known": model in PRICES,
            # 지침 축을 따로 내는가. 꺼져 있으면 약 칸에 접힌다(앱이 행을 그리기 전까지)
            "lifestyle_axis": lifestyle_axis_enabled(),
        }

    def _parse_site(text: str) -> tuple[str | None, str | None]:
        """자유 텍스트 부위 → (노드 id, 좌우). 못 찾으면 (None, 좌우).

        **점수 3(이름·별칭과 정확히 같음)만** 인정한다. 점수 2(앞뒤로 걸림)까지 받으면
        `"배"`가 `"아랫배"`에도 걸려 엉뚱한 쌍이 만들어진다. 확실할 때만 되묻는다.

        좌우는 떼고 찾는다. 온톨로지 노드 이름에는 좌우가 없는데(`눈`·`무릎`) 환자는 붙여서
        말해서, 안 떼면 `"오른쪽 눈"`이 점수 1로 떨어져 **대조가 통째로 조용히 꺼졌다**
        (2026-09-15 QA — 이마를 짚고 "오른쪽 눈"이라 해도 아무 말 없이 지나갔다).
        """
        if app.state.ontology is None:
            app.state.ontology = load_ontology()
        side, bare = strip_side(text)
        for query in (text, bare):
            if not query:
                continue
            for node, _term, score in app.state.ontology.search(query, limit=4):
                if score >= 3:
                    return node.id, side
                break
        return None, side

    def _site_relation(label: str, spoken: str) -> str:
        """인체도에서 짚은 곳과 환자가 말한 곳의 관계.

        `other`만 되묻는다. 상하위로 이어져 있으면 같은 곳이고, **더 좁은 쪽을 남긴다** —
        `어깨`를 짚고 `팔`이라고 했다고 `"어깨 팔"`로 적으면 의사가 읽는 자리가 흐려진다.
        한쪽이라도 온톨로지에 없으면 `unknown`이다. 모르면 되묻지 않는다 — 헛되묻기가
        "왜 못 알아듣지"로 읽히는 쪽이 놓치는 쪽보다 나쁘다.

        **같은 부위라도 좌우가 다르면 다른 곳이다.** 왼쪽 무릎을 짚고 오른쪽 무릎이라고
        말한 것은 되물어야 한다 — 의사가 보는 곳이 바뀐다.
        """
        a, a_side = _parse_site(label)
        b, b_side = _parse_site(spoken)
        sides_differ = bool(a_side and b_side and a_side != b_side)
        if not a or not b:
            # 부위를 못 찾아도 **좌우가 맞부딪히면** 그것만으로 되묻는다
            return "other" if sides_differ and a == b else "unknown"
        if sides_differ:
            return "other"
        if a == b:
            return "same"
        onto = app.state.ontology
        assert onto is not None
        if onto.is_ancestor(a, b):
            return "narrower"  # 말한 쪽이 짚은 것의 하위
        if onto.is_ancestor(b, a):
            return "broader"  # 말한 쪽이 짚은 것의 상위
        return "other"

    @app.post("/v1/previsit/sessions", response_model=StartResponse)
    def start_session(extractor: Ex, body: StartRequest | None = None) -> StartResponse:
        ondevice = bool(body and body.profile == "ondevice")
        s = Session(
            extractor,
            limits=app.state.limits,
            site_relation=_site_relation,
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
            # 이름만 온 경우도 **노드로 풀어 본다**(2026-09-15). 앱이 노드 id를 안 보내고
            # 화면에 보이는 이름만 보내고 있었다(백엔드 #7 확인). 이름만으로 두면 되묻기는
            # 돌지만 `title`·`department_guidance`가 비어 나간다 — 그 둘은 노드에서 온다.
            # 못 풀면 예전처럼 이름만 쓴다. **422로 끊지 않는다** — 환자가 짚은 것은 맞고,
            # 우리가 그 이름을 모르는 것뿐이다.
            node_id, side = _parse_site(body.site_label)
            sel = None
            if node_id:
                if app.state.ontology is None:
                    app.state.ontology = load_ontology()
                try:
                    sel = resolve_site(app.state.ontology, node_id, body.side or side)
                except ValueError:
                    sel = None
            if sel is not None:
                # **이름은 받은 그대로 쓴다.** 노드는 진료과·제목 재료를 얻으려고 푼 것이지
                # 이름을 고치려고 푼 것이 아니다 — `"허리"`로 보냈는데 `"허리 가운데"`로
                # 되돌려주면 앱 화면에 없던 말이 환자에게 보인다.
                record = sel.model_dump()
                record["label"] = body.site_label.strip()
                s.preselect_site(record["label"])
                s.card.site_selection = SiteSelectionRecord(**record)
                s.card.provenance.ontology_snapshot = sel.ontology_snapshot
            else:
                s.preselect_site(body.site_label)
        return StartResponse(
            reply=s.opening(), state=s.to_state(), request_id=body.request_id if body else None
        )

    @app.post("/v1/previsit/turns", response_model=TurnResponse)
    def process_turn(body: TurnRequest, extractor: Ex) -> TurnResponse:
        # 엔진이 만들 수 없는 상태는 여기서 끊는다. 스키마는 맞고 내용이 불가능한 것이라 400
        problems = check_state(body.state)
        if problems:
            raise HTTPException(status_code=400, detail="; ".join(problems))
        s = Session.from_state(
            extractor, body.state, limits=app.state.limits, site_relation=_site_relation
        )
        if body.extraction_meta is not None and s.card.provenance is not None:
            # 폰이 뽑았으면 어느 모델·프롬프트였는지 카드에 남긴다
            s.card.provenance.model_id = body.extraction_meta.model_id
            s.card.provenance.prompt_version = body.extraction_meta.prompt_version
        selections = [(sel.axis, sel.value) for sel in body.selections]
        try:
            # Langfuse trace 하나 = 턴 하나. 본문은 MEDIMATE_TRACE_CONTENT=on 전까지 가림(#111)
            with tracing.context(
                trace_name="previsit-turn",
                session_id=body.request_id or None,
                tags=["api", "previsit"],
                metadata={"turn": s.turn, "asked_axis": s.asked_axis},
            ):
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

        cands = None
        if body.question_candidates and s.ended and isinstance(s.card, PreVisitCard):
            cands = _make_question_candidates(s.card, extractor, body.patient_profile)

        return TurnResponse(
            reply=reply,
            ended=s.ended,
            end_reason=s.end_reason,
            state=s.to_state(),
            card=to_backend_payload(s.card, question_candidates=cands),
            audit=audit,
            request_id=body.request_id,
        )

    def _make_question_candidates(
        card: PreVisitCard, extractor, profile: PatientProfile | None = None
    ) -> list[dict[str, Any]] | None:
        """카드로 "의사에게 물어볼 것" 후보를 만든다. **실패해도 카드를 잃지 않는다.**

        어떤 이유로든 안 되면 `None`을 돌려준다 — 문답을 다 마친 환자의 카드가 후보 생성
        실패로 날아가면 안 된다. 우리 원칙이기도 하다: 카드 완성을 강제하지 않고, 환자가
        언제 끝내도 그 시점 카드가 결과다.

        **프롬프트에 식별자가 들어가지 않는다.** `ap.questions_user()`가 읽는 것은 축 값과
        복용약·기저질환·알러지, 환자가 덧붙인 말뿐이다. 이름·나이·성별·병원·진료일·ID·
        `request_id`는 카드에 있어도 프롬프트에 안 들어간다(`docs/api-previsit.md` 지키는 선).

        복용약·기저질환·알러지는 **요청의 `patient_profile`에서 온다.** 카드에는 그 자리가 없다 —
        축이 아니고 이 문진에서 환자가 말한 것도 아니어서, 재료로 쓰고 버린다. 안 보내면
        빈 값이고, 그러면 eval에서 그 세 카테고리를 3%→65%로 올린 규칙(questions-v8)이
        제품에서 안 걸린다(2026-09-11 백엔드 합의, #7).
        """
        axes = {
            str(getattr(a, "value", a)): e.value
            for a, e in card.axes.items()
            if e.value and e.status == FieldStatus.FILLED
        }
        payload = {
            "axes": axes,
            # 요청에서 온 것. 안 보냈으면 빈 값 — 카드에는 저장하지 않는다
            "profile": profile.as_prompt_payload()
            if profile
            else {"medications": [], "conditions": [], "allergies": []},
            "patient_message": None,  # 전할 말 턴은 없앴다(2026-09-11)
        }
        try:
            version = ap.QUESTIONS_VERSION
            text, _, _ = extractor.complete_json(
                ap.questions_system(version), ap.questions_user(payload), ap.QUESTIONS_SCHEMA
            )
            items = ap.parse_items(text)
        except Exception:  # noqa: BLE001 — 후보가 없어도 카드는 나가야 한다
            logging.getLogger("medimate.api").warning("질문 후보 생성 실패 — 카드는 그대로 낸다")
            return None
        top = top_candidates(items, top=QUESTION_CANDIDATES_TOP)
        return [{"text": it["text"], "source": it["source"], "rank": it["rank"]} for it in top]

    def _memo_classifier(request: Request):
        factory = request.app.state.memo_factory
        if factory is None:
            provider, model = extractor_env()
            shared: dict[str, object] = {}

            def make():
                c = LLMMemoClassifier(provider, model, client=shared.get("client"))
                return c

            # 리더는 **여기서 같이** 만든다(2026-09-15). 전에는 `_followup_reader`가
            # "memo_factory가 차 있으면 테스트 주입"이라고 추론했는데, 운영이 바로 이 줄에서
            # 스스로 채운 캐시와 구별이 안 돼 **운영에서 리더가 한 번도 안 만들어졌다.**
            # 백엔드가 운영 12건을 태워 `LLM 읽음`이 한 번도 안 나온 것으로 잡았다.
            # 추론하지 않는다 — 진짜 분류기를 만드는 자리가 진짜 리더도 만든다.
            def make_reader():
                return LLMFollowUpReader(provider, model, client=shared.get("client"))

            def make_segmenter():
                return LLMMemoSegmenter(provider, model, client=shared.get("client"))

            factory = request.app.state.memo_factory = make
            if request.app.state.followup_factory is None:
                request.app.state.followup_factory = make_reader
            if request.app.state.segment_factory is None:
                request.app.state.segment_factory = make_segmenter
        return BudgetGuarded(factory(), request.app.state.daily_budget)

    def _followup_reader(request: Request):
        """재방문 표현 읽기(followup-v1). 팩토리가 있으면 쓰고, 없으면 규칙 파서만.

        팩토리는 `create_app(followup_factory=…)`로 주입되거나, 운영에서는 `_memo_classifier`가
        진짜 분류기를 만들 때 **짝으로** 채운다. 여기서 "주입인지 운영인지"를 추론하지 않는다 —
        그 추론이 운영에서 리더를 영영 안 만들게 했다.
        """
        factory = request.app.state.followup_factory
        if factory is None:
            return None
        return BudgetGuarded(factory(), request.app.state.daily_budget)

    def _memo_segmenter(request: Request):
        """조각내기(memo-v5). 팩토리가 있으면 쓰고, 없으면 규칙 분리 + v4 분류로 간다."""
        factory = request.app.state.segment_factory
        if factory is None:
            return None
        return BudgetGuarded(factory(), request.app.state.daily_budget)

    @app.post("/v1/postvisit/memo", response_model=MemoResponse)
    def postvisit_memo(body: MemoRequest, request: Request) -> MemoResponse:
        """메모 → 4묶음 카드. labels가 오면 LLM 없이 조립(폰 분류·1q-2 수정 모두 이 경로)."""
        with tracing.context(
            trace_name="postvisit-memo",
            session_id=body.request_id or None,
            tags=["api", "postvisit"],
        ):
            return _postvisit_memo(body, request)

    def _postvisit_memo(body: MemoRequest, request: Request) -> MemoResponse:
        # 라벨의 번호는 우리가 나눈 문장의 주소다. 그 사이 분리 규칙이 바뀌었으면 같은 메모가
        # 다르게 나뉘어 **예전 번호가 다른 문장을 가리킨다.** 200에 카드도 멀쩡해 보이므로
        # 여기서 끊는다. 다시 분류하면 되는 일이라 4xx이고, 상태 충돌이라 409다
        res = None
        used: list = []  # 이 요청이 실제로 LLM을 태운 객체들(usage 합산용)
        prov_obj = None  # provenance를 낼 객체
        if body.labels is not None and body.sentences is not None:
            # memo-v5 되보내기: 조각 자체가 왔다. **다시 나누지 않는다.** 번호가 어긋날 길이 없다
            source = "client"
            res = assemble_from_client(
                body.memo,
                body.sentences,
                body.labels,
                visit_date=body.visit_date,
                clinic=body.clinic,
            )
            if res is None:
                raise HTTPException(
                    status_code=409,
                    detail="되보낸 sentences가 memo를 덮지 않습니다. 다시 분류해 주세요",
                )
            prov_obj = FixedLabels(body.labels)
        else:
            # 번호로만 되보낸 경로(v4 호환). 그 사이 분리 규칙이 바뀌었으면 같은 메모가 다르게
            # 나뉘어 예전 번호가 다른 문장을 가리킨다. 200에 카드도 멀쩡해 보이므로 여기서 끊는다
            if body.split_version is not None and body.split_version != SPLIT_VERSION:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "분리 규칙이 바뀌었습니다. 다시 분류해 주세요 "
                        f"(보낸 값 {body.split_version}, 서버 {SPLIT_VERSION})"
                    ),
                )
            if body.labels is not None:
                clf = FixedLabels(body.labels)
                source = "client"
            elif not body.classify:
                clf = FixedLabels({})
                source = "none"
            else:
                clf = _memo_classifier(request)
                source = "server"
            prov_obj = clf
            reader = _followup_reader(request) if source == "server" else None
            try:
                # **규칙이 한 덩어리밖에 못 만들 때만** LLM에게 조각을 맡긴다(2026-09-15 실측).
                # 마침표·띄어쓰기가 있는 정상 입력은 규칙 분리 + v4 분류가 더 정확했다
                # (라벨 98.5% vs v5 95.1%, v5는 "2주 약 먹고 다시 오라고"를 한 조각으로 합쳐
                # 약 칸을 비우기도 했다). v5가 이기는 자리는 `"일주일치약처방이주일후재방문"`처럼
                # 규칙이 자를 곳을 못 찾는 입력이다. 둘의 장점만 남긴다.
                if source == "server" and len(split_sentences(body.memo)) == 1:
                    seg = _memo_segmenter(request)
                    if seg is not None:
                        used.append(seg)
                        res = segment_memo(
                            body.memo,
                            seg,
                            visit_date=body.visit_date,
                            clinic=body.clinic,
                            followup_reader=reader,
                        )
                        if res is not None:
                            prov_obj = seg
                if res is None:
                    # 조각이 원문을 못 덮었거나 세그멘터가 없다 — 규칙 분리 + v4 분류
                    used.append(clf)
                    res = classify_memo(
                        body.memo,
                        clf,
                        visit_date=body.visit_date,
                        clinic=body.clinic,
                        followup_reader=reader,
                    )
                if reader is not None:
                    used.append(reader)
            except BudgetExceeded as e:
                raise HTTPException(status_code=503, detail=str(e)) from e
            except ValueError as e:
                raise HTTPException(status_code=502, detail=f"classifier: {e}") from e

        card = res.card
        card.provenance = Provenance(
            prompt_version=(
                body.labels_meta.prompt_version if body.labels_meta else prov_obj.prompt_version
            ),
            model_id=body.labels_meta.model_id if body.labels_meta else prov_obj.model_id,
        )
        # 소견 용어에 부위 병기 + 진료 전 부위와 대조(판정 아님)
        if app.state.ontology is None:
            app.state.ontology = load_ontology()
        notes = widen_card(card, app.state.ontology)
        card.site_comparison = compare_sites(app.state.ontology, body.previsit_anchor_id, notes)

        # 응답 `usage`는 **이 요청이 부른 LLM 전부**를 담는다 — 분류기 + 재방문 리더(2차 호출).
        # 2026-09-15에 분류기 것만 담아 리더가 붙은 건에서 38%를 빠뜨렸다. 예산 카운터는 맞게
        # 세고 있었는데 응답만 달라서, 비용으로 "켜졌나"를 판단하면 틀리게 읽혔다(백엔드 #7).
        tu = TurnUsage()
        if source == "server":
            in_t = out_t = 0
            cost = 0.0
            for obj in used:
                u = getattr(obj, "usage", None) if obj is not None else None
                if u is None:
                    continue
                in_t += int(u.input_tokens)
                out_t += int(u.output_tokens)
                cost += float(u.cost_usd(obj.model_id))
            tu = TurnUsage(input_tokens=in_t, output_tokens=out_t, cost_usd=round(cost, 6))
        labels_out = {str(i): res.labels.get(i, "none") for i in range(len(res.sentences))}
        raw_text = getattr(prov_obj, "last_text", None) or None
        audit = MemoAudit(
            memo=body.memo,
            sentences=res.sentences,
            labels=labels_out,
            raw_output=raw_text,
            dropped=res.dropped,
            source=source,
            prompt_version=card.provenance.prompt_version if card.provenance else "",
            model_id=card.provenance.model_id if card.provenance else "",
            lifestyle_axis=lifestyle_axis_enabled(),
            usage=tu,
        )
        return MemoResponse(
            card=to_backend_payload(card),
            sentences=res.sentences,
            labels=labels_out,
            dropped=res.dropped,
            source=source,
            split_version=SPLIT_VERSION,
            usage=tu,
            audit=audit,
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
