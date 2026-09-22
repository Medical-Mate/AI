"""Langfuse 트레이싱 — LLM 호출 하나가 generation 하나. 키가 없으면 전부 no-op.

붙는 자리는 둘. `llm/providers.py: LLMExtractor._call`(진료 전 추출·memo·후보 선택, 모든 공급자)
과 `span/select.py: JevSelector.select`. 러너는 `context()`로 trace 속성(session_id·tags·metadata)을
얹고 채점 결과를 `score()`로 붙인다. 대시보드에서 프롬프트 버전·카테고리·선택기별 통과율·비용·지연.

환경변수
- LANGFUSE_PUBLIC_KEY · LANGFUSE_SECRET_KEY · LANGFUSE_BASE_URL — 없으면 꺼진다(SDK가 경고 한 줄)
- MEDIMATE_TRACING=off — 키가 있어도 끈다
- MEDIMATE_TRACE_CONTENT=on — 입력·출력 본문을 보낸다. 기본은 가림(길이만 남긴다).
  호스트가 us.cloud.langfuse.com이라 환자 발화가 국외로 나간다(#111 국외 이전 안내, #113 동의).
  eval 러너는 우리가 만든 케이스라 켠다. 운영 API는 동의 논의 전까지 끈 채로 토큰·지연·버전만 남긴다
- MEDIMATE_ENV — Langfuse environment 태그(eval / prod). 기본 "local"

지키는 선: request_id·session_id는 트레이스에 실리지만 프롬프트에는 들어가지 않는다(기존 규칙).
user_id는 쓰지 않는다 — 개인 식별자를 밖으로 보내지 않는다. 필요하면 세션 단위까지만.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Iterator
from typing import Any

logger = logging.getLogger("medimate.obs")

_client: Any = None
_tried = False


def enabled() -> bool:
    if os.getenv("MEDIMATE_TRACING", "").strip().lower() in ("off", "0", "false", "no"):
        return False
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def content_allowed() -> bool:
    return os.getenv("MEDIMATE_TRACE_CONTENT", "").strip().lower() in ("1", "true", "yes", "on")


def _mask(*, data: Any, **_: Any) -> Any:
    """본문 가림. 문자열은 길이만, dict·list는 재귀. `MEDIMATE_TRACE_CONTENT=on`이면 그대로."""
    if content_allowed():
        return data
    if isinstance(data, str):
        return f"<masked {len(data)} chars>"
    if isinstance(data, dict):
        return {k: _mask(data=v) for k, v in data.items()}
    if isinstance(data, list):
        return [_mask(data=v) for v in data]
    return data


def client():
    """Langfuse 클라이언트(싱글턴). 꺼져 있거나 SDK가 없으면 None."""
    global _client, _tried
    if _tried:
        return _client
    _tried = True
    if not enabled():
        return None
    try:
        from langfuse import Langfuse

        _client = Langfuse(
            mask=_mask,
            environment=os.getenv("MEDIMATE_ENV", "local"),
            release=os.getenv("MEDIMATE_RELEASE") or None,
        )
    except Exception as e:  # noqa: BLE001 — 관측이 본 기능을 죽이면 안 된다
        logger.warning("Langfuse 초기화 실패, 트레이싱 끔: %s", e)
        _client = None
    return _client


def reset_for_tests() -> None:
    global _client, _tried
    _client, _tried = None, False


class _Gen:
    """generation 핸들. `done()`으로 출력·토큰·비용을 채운다. 클라이언트가 없으면 무동작."""

    def __init__(self, span):
        self._span = span

    def done(
        self,
        *,
        output: Any = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cost_usd: float | None = None,
        metadata: dict | None = None,
        level: str | None = None,
        status_message: str | None = None,
    ) -> None:
        if self._span is None:
            return
        usage = {}
        if input_tokens is not None:
            usage["input"] = input_tokens
        if output_tokens is not None:
            usage["output"] = output_tokens
        kw: dict[str, Any] = {"output": output}
        if usage:
            kw["usage_details"] = usage
        if cost_usd is not None:
            kw["cost_details"] = {"total": cost_usd}
        if metadata:
            kw["metadata"] = metadata
        if level:
            kw["level"] = level
        if status_message:
            kw["status_message"] = status_message
        try:
            self._span.update(**kw)
        except Exception as e:  # noqa: BLE001
            logger.debug("trace update 실패: %s", e)


@contextlib.contextmanager
def generation(
    name: str,
    *,
    model: str,
    input: Any = None,
    metadata: dict | None = None,
    version: str | None = None,
) -> Iterator[_Gen]:
    """LLM 호출 하나. `with generation(...) as g: ...; g.done(output=...)`."""
    lf = client()
    if lf is None:
        yield _Gen(None)
        return
    try:
        cm = lf.start_as_current_observation(
            as_type="generation",
            name=name,
            model=model,
            input=input,
            metadata=metadata,
            version=version,
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("trace 시작 실패: %s", e)
        yield _Gen(None)
        return
    with cm as span:
        yield _Gen(span)


@contextlib.contextmanager
def context(
    *,
    trace_name: str | None = None,
    session_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict | None = None,
    version: str | None = None,
) -> Iterator[None]:
    """이 안에서 만들어지는 generation에 trace 속성을 얹는다. 러너·API 요청 단위로 감싼다."""
    lf = client()
    if lf is None:
        yield
        return
    from langfuse import propagate_attributes

    md = {k: str(v) for k, v in (metadata or {}).items() if v is not None}
    # 루트 span을 하나 연다 — 그래야 trace가 이 블록 전체를 덮고, generation이 끝난 뒤 붙이는
    # score_current_trace()가 "활성 span 없음"으로 버려지지 않는다(2026-09-22 첫 스모크에서 확인)
    with (
        lf.start_as_current_observation(as_type="span", name=trace_name or "request"),
        propagate_attributes(
            trace_name=trace_name,
            session_id=session_id,
            tags=tags,
            metadata=md or None,
            version=version,
        ),
    ):
        yield


def score(name: str, value: float | str, *, comment: str | None = None) -> None:
    """현재 trace에 점수. 채점기 결과(D 통과·EM·confidence)를 붙인다."""
    lf = client()
    if lf is None:
        return
    try:
        lf.score_current_trace(name=name, value=value, comment=comment)
    except Exception as e:  # noqa: BLE001
        logger.debug("score 실패: %s", e)


def flush() -> None:
    lf = client()
    if lf is not None:
        with contextlib.suppress(Exception):
            lf.flush()
