"""공급자별 Extractor 구현. 셋 다 같은 프롬프트, 같은 방식(텍스트 JSON → 파싱).

structured output 강제는 쓰지 않는다. 한쪽만 쓰면 모델 비교가 아니라 도구 비교가 된다.
SDK는 선택 의존성이다: `uv sync --group providers`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from medimate.llm.base import Turn, TurnExtraction
from medimate.llm.prompt import PROMPT_VERSION, system_prompt, user_message
from medimate.schema.card import Axis

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
}


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def cost_usd(self, model_id: str) -> float:
        i, o = PRICES.get(model_id, (0.0, 0.0))
        return (self.input_tokens * i + self.output_tokens * o) / 1_000_000


def _parse_json(text: str) -> TurnExtraction:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    return TurnExtraction.model_validate(json.loads(t))


@dataclass
class RawResult:
    """채점기가 소비하는 원본. 파싱 실패도 기록해야 D1이 채점된다."""

    text: str
    parsed: TurnExtraction | None
    error: str | None
    input_tokens: int
    output_tokens: int


@dataclass
class LLMExtractor:
    """Extractor 프로토콜 구현 + 원본 텍스트 접근. eval은 extract_raw를 쓴다."""

    provider: str  # anthropic | openai | google
    model_id: str
    budget_usd: float = 0.50  # 모델당 상한. 넘으면 호출 전에 예외
    prompt_version: str = PROMPT_VERSION
    usage: Usage = field(default_factory=Usage)
    _client: object = field(default=None, repr=False)

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
        text, i, o = self._call(system_prompt(), user_message(utterance, asked_axis, history))
        self.usage.calls += 1
        self.usage.input_tokens += i
        self.usage.output_tokens += o
        try:
            return RawResult(text, _parse_json(text), None, i, o)
        except Exception as e:  # noqa: BLE001 — 파싱 실패 자체가 채점 대상
            return RawResult(text, None, f"{type(e).__name__}: {e}", i, o)

    # ------------------------------------------------------------------
    def _call(self, system: str, user: str) -> tuple[str, int, int]:
        if self.provider == "anthropic":
            return self._anthropic(system, user)
        if self.provider == "openai":
            return self._openai(system, user)
        if self.provider == "google":
            return self._google(system, user)
        raise ValueError(self.provider)

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
        r = self._client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_completion_tokens=1024,
        )
        u = r.usage
        return r.choices[0].message.content or "", u.prompt_tokens, u.completion_tokens

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
