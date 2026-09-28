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
- 본문을 보낼 때는 식별자를 가린다(`obs/pii.py`, 2026-09-28). generation 입력은 시작이 아니라
  `done()` 때 싣는다 — Langfuse `mask`는 호출 스레드에서 **동기로** 돌아서, 거기서 Guardrails를
  부르면 LLM 호출마다 지연이 붙는다. 탐지는 Nova 호출과 겹치게 먼저 걸어 두고 끝날 때 받는다

지키는 선: request_id·session_id는 트레이스에 실리지만 프롬프트에는 들어가지 않는다(기존 규칙).
user_id는 쓰지 않는다 — 개인 식별자를 밖으로 보내지 않는다. 필요하면 세션 단위까지만.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Iterator
from typing import Any

from medimate.obs import pii

logger = logging.getLogger("medimate.obs")

_client: Any = None
_tried = False


def enabled() -> bool:
    if os.getenv("MEDIMATE_TRACING", "").strip().lower() in ("off", "0", "false", "no"):
        return False
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def content_allowed() -> bool:
    return os.getenv("MEDIMATE_TRACE_CONTENT", "").strip().lower() in ("1", "true", "yes", "on")


def _length_only(data: Any) -> Any:
    """문자열은 길이만, dict·list는 재귀."""
    if isinstance(data, str):
        return f"<masked {len(data)} chars>"
    if isinstance(data, dict):
        return {k: _length_only(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_length_only(v) for v in data]
    return data


def _mask(*, data: Any, **_: Any) -> Any:
    """SDK 훅. 본문 수집이 꺼져 있으면 길이만. 켜져 있으면 정규식 가림(호출 0) — Guardrails
    조각 치환은 `_Gen.done()`이 먼저 해 두고, 여기는 빠진 자리를 막는 안전망이다."""
    if not content_allowed():
        return _length_only(data)
    return pii.scrub(data, []) if pii.enabled() else data


# 탐지 결과를 generation 끝에서 얼마나 기다리나. Nova 호출(중앙값 0.7초)보다 탐지(p50 146ms)가
# 먼저 끝나므로 보통은 기다리지 않는다. 넘기면 그 generation은 길이만 남긴다
PII_WAIT_S = 1.5


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


def status() -> dict[str, Any]:
    """`/health`용 — 트레이싱이 **실제로** 무엇을 하고 있는가(#127, 백엔드 요청).

    - `enabled`: 키가 있는가가 아니라 **클라이언트가 만들어졌는가**. 키가 있어도 SDK가 없거나
      초기화에 실패하면 false다(#125 전 이미지가 그랬다 — 키를 넣어도 무동작)
    - `content`: **본문이 실제로 나가는가**. 스위치(`MEDIMATE_TRACE_CONTENT`)가 켜져 있어도
      트레이싱이 꺼져 있으면 false. 국외 이전 안내(#111)와 맞는지 밖에서 보는 유일한 신호다
    - `env`·`host`: 어느 Langfuse 환경·주소로 가는가. 비밀이 아니다
    - `pii`: 본문을 보낼 때 식별자 가림 — "guardrails"(이름·주소·나이까지) · "regex"(전화·주민번호
      등만) · null(본문을 안 보내거나 가림을 끔)
    """
    on = client() is not None
    sending = on and content_allowed()
    return {
        "enabled": on,
        "content": sending,
        "pii": pii.mode() if sending else None,
        "env": os.getenv("MEDIMATE_ENV", "local") if on else None,
        "host": (os.getenv("LANGFUSE_BASE_URL") or None) if on else None,
    }


def reset_for_tests() -> None:
    global _client, _tried
    _client, _tried = None, False


class _Gen:
    """generation 핸들. `done()`으로 출력·토큰·비용을 채운다. 클라이언트가 없으면 무동작."""

    def __init__(self, span, pending_input: Any = None, detection=None, deferred=False):
        self._span = span
        self._pending = pending_input
        self._detection = detection  # Future[list[(유형, 조각)]] 또는 None(정규식만)
        self._deferred = deferred

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
        if self._deferred:
            try:
                matches = self._detection.result(timeout=PII_WAIT_S) if self._detection else []
                kw["input"] = pii.scrub(self._pending, matches)
                kw["output"] = pii.scrub(output, matches)
            except Exception as e:  # noqa: BLE001 — 가림 실패는 원문이 아니라 전부 가림으로
                logger.warning("PII 탐지 실패, 이 호출은 길이만 남깁니다: %s", type(e).__name__)
                kw["input"] = _length_only(self._pending)
                kw["output"] = _length_only(output)
                metadata = {**(metadata or {}), "pii": "failed"}
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
    # 본문을 보내면 입력은 done() 때 가려서 싣는다. 탐지는 지금 걸어 LLM 호출과 겹친다
    deferred = content_allowed() and pii.enabled()
    detection = None
    if deferred and pii.guardrail_id():
        try:
            detection = pii.detect_async(pii.user_texts(input))
        except Exception as e:  # noqa: BLE001
            logger.warning("PII 탐지 시작 실패: %s", type(e).__name__)
    try:
        cm = lf.start_as_current_observation(
            as_type="generation",
            name=name,
            model=model,
            input=None if deferred else input,
            metadata=metadata,
            version=version,
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("trace 시작 실패: %s", e)
        yield _Gen(None)
        return
    with cm as span:
        yield _Gen(span, pending_input=input, detection=detection, deferred=deferred)


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
