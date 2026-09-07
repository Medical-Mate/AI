"""HMAC 요청 서명 검증 — 백엔드 ↔ AI 서버.

백엔드와 AI 서버가 같은 비밀키를 갖는다. 백엔드는 요청마다
    message = f"{timestamp}.{request_id}.{body}"
를 HMAC-SHA256(key)으로 서명해 헤더에 넣고, 여기서 같은 계산으로 검증한다.
- 키는 네트워크로 다니지 않는다
- 본문이 바뀌면 서명이 틀어진다(위조 차단)
- timestamp가 허용 폭을 벗어나면 거부(오래된 재전송 차단). request_id 중복 차단은 백엔드 캐시 몫
헤더 이름·허용 폭은 백엔드 규격이 정해지면 환경변수로 맞춘다. 키가 없으면(로컬 개발) 검증을
건너뛴다.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import time
from dataclasses import dataclass

from fastapi import Request
from fastapi.responses import JSONResponse


@dataclass(frozen=True)
class HmacConfig:
    secret: str | None
    header_signature: str = "X-Signature"
    header_timestamp: str = "X-Timestamp"
    header_request_id: str = "X-Request-Id"
    max_skew_s: int = 300  # ±5분
    exempt_paths: tuple[str, ...] = ("/health", "/docs", "/openapi.json", "/redoc")

    @classmethod
    def from_env(cls) -> HmacConfig:
        return cls(
            secret=os.getenv("MEDIMATE_HMAC_SECRET") or None,
            header_signature=os.getenv("MEDIMATE_HMAC_HEADER_SIGNATURE", "X-Signature"),
            header_timestamp=os.getenv("MEDIMATE_HMAC_HEADER_TIMESTAMP", "X-Timestamp"),
            header_request_id=os.getenv("MEDIMATE_HMAC_HEADER_REQUEST_ID", "X-Request-Id"),
            max_skew_s=int(os.getenv("MEDIMATE_HMAC_MAX_SKEW_S", "300")),
        )


def sign(secret: str, timestamp: str, request_id: str, body: bytes) -> str:
    """백엔드가 쓰는 것과 같은 계산. 테스트와 문서의 기준."""
    msg = f"{timestamp}.{request_id}.".encode() + body
    return hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()


def verify(cfg: HmacConfig, headers, body: bytes, now: float | None = None) -> str | None:
    """문제가 있으면 이유 문자열, 통과면 None."""
    sig = headers.get(cfg.header_signature)
    ts = headers.get(cfg.header_timestamp)
    rid = headers.get(cfg.header_request_id, "")
    if not sig or not ts:
        return "missing signature or timestamp"
    try:
        t = int(ts)
    except ValueError:
        return "bad timestamp"
    if abs((now or time.time()) - t) > cfg.max_skew_s:
        return "timestamp out of window"
    expected = sign(cfg.secret or "", str(t), rid, body)
    if not hmac.compare_digest(expected, sig.lower()):
        return "signature mismatch"
    return None


def install(app, cfg: HmacConfig | None = None) -> None:
    """앱에 미들웨어를 붙인다. secret이 없으면 아무것도 하지 않는다(로컬 개발)."""
    cfg = cfg or HmacConfig.from_env()
    app.state.hmac = cfg
    if not cfg.secret:
        return

    @app.middleware("http")
    async def _hmac_middleware(request: Request, call_next):
        if request.url.path in cfg.exempt_paths or request.method == "GET":
            return await call_next(request)
        body = await request.body()
        why = verify(cfg, request.headers, body)
        if why:
            return JSONResponse(status_code=401, content={"detail": f"hmac: {why}"})
        return await call_next(request)
