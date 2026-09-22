"""공급자별 Extractor 구현. 셋 다 같은 프롬프트, 같은 방식(텍스트 JSON → 파싱).

structured output 강제는 쓰지 않는다. 한쪽만 쓰면 모델 비교가 아니라 도구 비교가 된다.
SDK는 선택 의존성이다: `uv sync --group providers`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field

from medimate.llm import prompt as prompt_v3
from medimate.llm import prompt_small, prompt_v4_nova
from medimate.llm.base import Turn, TurnExtraction, parse_json_text
from medimate.schema.card import Axis

# 프롬프트 계열. small은 온디바이스 소형 모델용, v4-nova는 Nova Pro 전용(인젝션·값 창작 방어)
PROMPTS = {"v3": prompt_v3, "small": prompt_small, "v4-nova": prompt_v4_nova}

# 모델 id → 프롬프트 계열. **여기 없는 모델은 v3**다.
# Nova에만 v4를 붙이는 이유: v3 88케이스에서 Terra 85 · Sonnet 83 · Nova 76이었고 Nova의
# 실패가 전부 "값 창작" 계열이었다. v3를 고치면 그 비교 표가 무효가 되고 재실행은 사비다.
_MODEL_PROMPT = {"apac.amazon.nova-pro-v1:0": "v4-nova"}


_AUTO = "auto"


def prompt_family_for(model_id: str) -> str:
    """이 모델이 쓸 프롬프트 계열. 운영이 모델만 바꿔도 프롬프트가 따라간다."""
    return _MODEL_PROMPT.get(model_id, "v3")


# $/1M tokens (input, output). 지출 가드용. 공급자 가격 페이지에서 확인 후 갱신.
PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-opus-5": (5.00, 25.00),
    "gpt-5.6-luna": (0.20, 1.20),
    "gpt-5.6-terra": (2.00, 12.00),
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.6-flash": (1.50, 7.50),
    "gemini-3.1-pro": (2.00, 12.00),
    # Bedrock 추론 프로파일 ID. 값은 서울(APN2) 요금표를 API로 확인한 것(2026-09-11).
    # raw 모델 ID는 온디맨드가 안 되므로 프로파일 ID를 그대로 키로 쓴다.
    "apac.amazon.nova-lite-v1:0": (0.06, 0.24),
    "apac.amazon.nova-pro-v1:0": (0.80, 3.20),
    "global.anthropic.claude-haiku-4-5-20251001-v1:0": (1.10, 5.50),
    "global.anthropic.claude-sonnet-4-6": (3.00, 15.00),
    # TypeSafe Jev(System One). 입력 $0.042/M, 출력 무료 (docs.typesafe.ai/models, 2026-09-21)
    "jev-1.13.0": (0.042, 0.0),
}
# 로컬(온디바이스 후보) 모델은 가격표에 없다 → cost 0.
# model_id는 "local/" + llama-server가 알려주는 이름
LOCAL_PREFIX = "local/"


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def cost_usd(self, model_id: str) -> float:
        # local/… 은 가격표에 없어 0. 그 밖의 미등록 ID는 LLMExtractor가 생성 때 막는다
        i, o = PRICES.get(model_id, (0.0, 0.0))
        return (self.input_tokens * i + self.output_tokens * o) / 1_000_000


def _parse_json(text: str) -> TurnExtraction:
    return TurnExtraction.model_validate(parse_json_text(text))


@dataclass
class RawResult:
    """채점기가 소비하는 원본. 파싱 실패도 기록해야 D1이 채점된다."""

    text: str
    parsed: TurnExtraction | None
    error: str | None
    input_tokens: int
    output_tokens: int


def require_price(model_id: str) -> tuple[float, float]:
    """가격표에서 단가를 꺼낸다. 없으면 호출하지 않는다.

    `PRICES.get(model_id, (0, 0))`은 미등록 모델의 비용을 0으로 계산한다. 그러면
    `--budget` 상한이 영원히 안 걸리고 예상 비용도 $0.000으로 보인다. 사비로 운영하므로
    조용히 새는 쪽보다 멈추는 쪽을 택한다. 모델 ID 오타도 여기서 걸린다.

    러너(실호출 경로)만 부른다. 서버는 요청마다 LLMExtractor를 만들고 상한이 쓰이지 않으므로
    생성자에 두면 배포된 서버가 가격표 누락으로 전 요청을 실패시킨다.
    로컬(온디바이스 후보)은 값이 0이 맞다.
    """
    if model_id.startswith(LOCAL_PREFIX):
        return (0.0, 0.0)
    if model_id not in PRICES:
        raise BudgetExceeded(
            f"{model_id}: PRICES에 없다. 지출 상한이 무력화되므로 호출하지 않는다. "
            f"providers.PRICES에 (입력, 출력) $/1M 을 추가할 것"
        )
    return PRICES[model_id]


@dataclass
class LLMExtractor:
    """Extractor 프로토콜 구현 + 원본 텍스트 접근. eval은 extract_raw를 쓴다."""

    # anthropic | openai | google | local(llama-server 등 OpenAI 호환. GGUF 온디바이스 후보)
    provider: str
    model_id: str
    budget_usd: float = 0.50  # 모델당 상한. 넘으면 호출 전에 예외
    # PROMPTS 키. 기본 `auto`는 모델 id로 고른다(`prompt_family_for`). eval 러너는 명시할 수 있다
    prompt_family: str = "auto"
    prompt_version: str = ""  # 비우면 계열의 PROMPT_VERSION
    # local 공급자용 요청별 JSON 스키마. None이면 MEDIMATE_LOCAL_JSON_SCHEMA 파일을 쓴다
    response_schema: dict | None = None
    usage: Usage = field(default_factory=Usage)
    _client: object = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.prompt_family == _AUTO:
            self.prompt_family = prompt_family_for(self.model_id)
        if not self.prompt_version:
            self.prompt_version = PROMPTS[self.prompt_family].PROMPT_VERSION

    def extract(
        self, utterance: str, asked_axis: Axis | None, history: Sequence[Turn] = ()
    ) -> TurnExtraction:
        r = self.extract_raw(utterance, asked_axis, history)
        if r.parsed is None:
            raise ValueError(r.error)
        return r.parsed

    def extract_raw(
        self, utterance: str, asked_axis: Axis | None, history: Sequence[Turn] = ()
    ) -> RawResult:
        if self.usage.cost_usd(self.model_id) >= self.budget_usd:
            raise BudgetExceeded(f"{self.model_id}: ${self.budget_usd} 상한 도달")
        pm = PROMPTS[self.prompt_family]
        text, i, o = self._call(pm.system_prompt(), pm.user_message(utterance, asked_axis, history))
        self.usage.calls += 1
        self.usage.input_tokens += i
        self.usage.output_tokens += o
        try:
            return RawResult(text, _parse_json(text), None, i, o)
        except Exception as e:  # noqa: BLE001 — 파싱 실패 자체가 채점 대상
            return RawResult(text, None, f"{type(e).__name__}: {e}", i, o)

    def complete_json(self, system: str, user: str, schema: dict) -> tuple[str, int, int]:
        """추출 프롬프트가 아닌 일반 JSON 호출(질문 후보·할 일 등). 상한·사용량 집계는 같다."""
        if self.usage.cost_usd(self.model_id) >= self.budget_usd:
            raise BudgetExceeded(f"{self.model_id}: ${self.budget_usd} 상한 도달")
        prev = self.response_schema
        self.response_schema = schema
        try:
            text, i, o = self._call(system, user)
        finally:
            self.response_schema = prev
        self.usage.calls += 1
        self.usage.input_tokens += i
        self.usage.output_tokens += o
        return text, i, o

    # ------------------------------------------------------------------
    def _call(self, system: str, user: str) -> tuple[str, int, int]:
        """공급자 호출 한 번 = Langfuse generation 하나(키 없으면 무동작). 모든 프롬프트가 지남."""
        from medimate.obs import tracing

        with tracing.generation(
            self.prompt_version or self.prompt_family,
            model=self.model_id,
            input={"system": system, "user": user},
            metadata={"provider": self.provider, "prompt_family": self.prompt_family},
            version=self.prompt_version or None,
        ) as g:
            try:
                text, i, o = self._dispatch(system, user)
            except Exception as e:
                g.done(level="ERROR", status_message=f"{type(e).__name__}: {str(e)[:200]}")
                raise
            pi, po = PRICES.get(self.model_id, (0.0, 0.0))
            g.done(
                output=text,
                input_tokens=i,
                output_tokens=o,
                cost_usd=(i * pi + o * po) / 1_000_000,
            )
            return text, i, o

    def _dispatch(self, system: str, user: str) -> tuple[str, int, int]:
        if self.provider == "anthropic":
            return self._anthropic(system, user)
        if self.provider == "openai":
            return self._openai(system, user)
        if self.provider == "google":
            return self._google(system, user)
        if self.provider == "local":
            return self._local(system, user)
        if self.provider == "bedrock":
            return self._bedrock(system, user)
        raise ValueError(self.provider)

    def _bedrock(self, system: str, user: str) -> tuple[str, int, int]:
        """Amazon Bedrock — converse API. Claude·Nova를 같은 형식으로 받는다.

        인증은 IAM이다(태스크 역할 또는 ~/.aws). API 키가 없다.
        model_id는 추론 프로파일 ID를 쓴다: `apac.amazon.nova-pro-v1:0`,
        `global.anthropic.claude-sonnet-4-6`. raw 모델 ID는 온디맨드가 안 된다.
        리전은 MEDIMATE_BEDROCK_REGION (기본 ap-northeast-2).
        """
        import os

        if self._client is None:
            # 클라이언트를 만들 때만 임포트한다. providers 그룹 없이도 어댑터 테스트가 돈다
            import boto3

            self._client = boto3.client(
                "bedrock-runtime",
                region_name=os.getenv("MEDIMATE_BEDROCK_REGION", "ap-northeast-2"),
            )
        kwargs: dict = {}
        if self.response_schema is not None:
            # converse는 response_format이 없다. 스키마는 시스템 프롬프트로 지시한다
            system = (
                system
                + "\n\n반드시 이 JSON 스키마에 맞는 JSON만 출력한다:\n"
                + json.dumps(self.response_schema, ensure_ascii=False)
            )
        r = self._client.converse(
            modelId=self.model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"maxTokens": 8192, "temperature": 0},
            **kwargs,
        )
        text = "".join(b["text"] for b in r["output"]["message"]["content"] if "text" in b)
        u = r["usage"]
        return text, u["inputTokens"], u["outputTokens"]

    def _anthropic(self, system: str, user: str) -> tuple[str, int, int]:
        import anthropic

        if self._client is None:
            self._client = anthropic.Anthropic()
        # Sonnet 5는 사고(thinking)가 기본이고 max_tokens에 포함된다.
        # 1024면 긴 발화에서 본문이 빈다.
        r = self._client.messages.create(
            model=self.model_id,
            max_tokens=8192,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in r.content if b.type == "text")
        return text, r.usage.input_tokens, r.usage.output_tokens

    def _openai(self, system: str, user: str) -> tuple[str, int, int]:
        from openai import OpenAI

        if self._client is None:
            self._client = OpenAI()
        kwargs: dict = {}
        if self.response_schema is not None:
            # 메모 분류처럼 출력 형태가 정해진 호출은 서버 모델에도 같은 스키마를 강제한다
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": self.response_schema, "strict": True},
            }
        r = self._client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_completion_tokens=1024,
            **kwargs,
        )
        u = r.usage
        return r.choices[0].message.content or "", u.prompt_tokens, u.completion_tokens

    def _local(self, system: str, user: str) -> tuple[str, int, int]:
        """llama-server(`llama-server -m x.gguf --port 8080`) 등 OpenAI 호환 로컬 서버.

        온디바이스 후보 스크리닝용. 기기(llama.cpp)와 같은 엔진·같은 GGUF라 온도 0·시드 고정이면
        PC 결과가 기기 결과다. 사고(thinking) 모델은 서버 쪽 옵션으로 끈다.
        MEDIMATE_LOCAL_BASE_URL (기본 http://127.0.0.1:8080/v1). model_id는 "local/<이름>".
        """
        import os

        from openai import OpenAI

        if self._client is None:
            self._client = OpenAI(
                base_url=os.getenv("MEDIMATE_LOCAL_BASE_URL", "http://127.0.0.1:8080/v1"),
                api_key="local",
            )
        # 스키마 강제는 요청별 response_format으로 넘긴다. 서버 전역 --json-schema-file은
        # 채팅 템플릿 토큰(<|im_start|>)까지 문법으로 검사해 400이 난다
        kwargs: dict = {}
        schema = self.response_schema
        if schema is None:
            schema_path = os.getenv(
                "MEDIMATE_LOCAL_JSON_SCHEMA", "evals/ondevice/turn_extraction.schema.json"
            )
            if schema_path and os.path.exists(schema_path):
                import json as _json

                with open(schema_path, encoding="utf-8") as f:
                    schema = _json.load(f)
        if schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": schema},
            }
        r = self._client.chat.completions.create(
            model=self.model_id.removeprefix(LOCAL_PREFIX),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=1024,
            temperature=0,
            seed=42,
            **kwargs,
        )
        u = r.usage
        return (
            r.choices[0].message.content or "",
            (u.prompt_tokens if u else 0) or 0,
            (u.completion_tokens if u else 0) or 0,
        )

    def _google(self, system: str, user: str) -> tuple[str, int, int]:
        from google import genai
        from google.genai import types

        if self._client is None:
            self._client = genai.Client()
        r = self._client.models.generate_content(
            model=self.model_id,
            contents=user,
            # Gemini는 사고(thinking) 토큰이 max_output_tokens에 포함된다. 1024면 JSON이 잘린다.
            config=types.GenerateContentConfig(system_instruction=system, max_output_tokens=8192),
        )
        m = r.usage_metadata
        out = (m.candidates_token_count or 0) + (getattr(m, "thoughts_token_count", 0) or 0)
        return r.text or "", m.prompt_token_count or 0, out
