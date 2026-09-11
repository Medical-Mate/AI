"""HMAC 요청 서명 검증 — 백엔드 ↔ AI 서버.

백엔드와 AI 서버가 같은 비밀키를 갖는다. 백엔드는 요청마다

    message   = f"{METHOD}.{path}.{timestamp}.{request_id}." + body_bytes
    signature = hex(HMAC_SHA256(secret, message))

를 헤더에 넣고, 여기서 같은 계산으로 검증한다.

- 키는 네트워크로 다니지 않는다
- 본문이 바뀌면 서명이 틀어진다(위조 차단)
- **메서드·경로가 들어간다**(2026-09-11, 백엔드 합의 #7). 넣기 전에는 한 엔드포인트에 유효한
  서명을 본문이 같은 다른 엔드포인트로 재전송할 수 있었다. 같은 사설망이라 악용 경로는
  좁았지만 그건 "지금 구성이 그렇다"는 것이고 구성은 바뀐다
- `path`는 **쿼리스트링을 제외**하고 앞 `/`를 포함한다(`/v1/previsit/turns`). 쿼리를 넣으면
  인코딩 차이로 깨진다. 나중에 POST에 쿼리가 붙으면 규격을 다시 정한다
- `path`는 **프록시를 지난 뒤의 경로**다. 게이트웨이가 경로를 다시 쓰면 양쪽이 다른 문자열을
  보게 되고, 그게 서명 불일치의 제일 흔한 원인이다
- timestamp가 허용 폭을 벗어나면 거부(오래된 재전송 차단). request_id 중복 차단은 백엔드 캐시 몫
- `GET`은 **면제하지 않는다**(2026-09-11 변경). 이전에는 메서드 단위로 GET 전체를 면제해서
  경로 화이트리스트가 무의미했고, 그 사이 `GET /v1/ontology/search`가 생겼다. 그런 조회
  엔드포인트가 조용히 무인증이 되는 것을 막는다. 면제는 `exempt_paths`뿐이다

헤더 이름·허용 폭은 환경변수로 바꿀 수 있다. 키가 없으면(로컬 개발) 검증을 건너뛴다.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
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


# 서명 메시지 조립 템플릿. **`sign()`이 이 문자열 하나만 쓴다.**
#
# 손으로 관리하는 버전 번호를 두지 않는 이유: 계산식은 바꾸고 번호는 안 바꾸는 일이 생기고,
# 그러면 `/health`가 "v2"라는데 실제로는 v3를 검증하는 상태가 된다. 점검 도구는 또 통과를 낸다.
# 2026-09-11에 정확히 그 사고가 났다 — `main`의 계산식이 옛 형식인데 새 형식 벡터로 대조해
# 10 pass가 나왔고 운영에서만 401이 났다. **검증 자료와 구현이 같이 움직이면 서로를 확인해
# 주지 못한다.** 그래서 형식을 바꾸면 `/health` 값이 자동으로 따라가게 묶어 둔다.
SIGNING_TEMPLATE = "{method}.{path}.{timestamp}.{request_id}."
SIGNING_ALGORITHM = "hmac-sha256-hex"


def signing_spec() -> dict[str, str]:
    """서버가 **실제로 검증하는** 서명 규격. `/health`가 그대로 내보낸다.

    점검 도구(`scripts/check_ai_server.py`)가 자기 템플릿과 문자열 비교해서, 로컬 코드와
    배포본이 갈린 상태를 잡는다. 서명 형식은 계약 문서·벡터 파일에 이미 공개돼 있고
    비밀은 키뿐이라 내보내도 새로 알려주는 것이 없다.
    """
    return {"template": SIGNING_TEMPLATE, "algorithm": SIGNING_ALGORITHM}


def sign(secret: str, method: str, path: str, timestamp: str, request_id: str, body: bytes) -> str:
    """백엔드가 쓰는 것과 같은 계산. 테스트와 문서의 기준.

    `method`는 대문자로 맞춘다. `path`는 쿼리 없이, 앞 `/`를 포함해서 넘긴다.
    `request_id`가 없으면 빈 문자열로 계산한다(백엔드는 항상 보내기로 했다 — #7).
    """
    head = SIGNING_TEMPLATE.format(
        method=method.upper(), path=path, timestamp=timestamp, request_id=request_id
    )
    return hmac.new(secret.encode(), head.encode() + body, hashlib.sha256).hexdigest()


def verify(
    cfg: HmacConfig,
    method: str,
    path: str,
    headers,
    body: bytes,
    now: float | None = None,
) -> str | None:
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
    expected = sign(cfg.secret or "", method, path, str(t), rid, body)
    if not hmac.compare_digest(expected, sig.lower()):
        return "signature mismatch"
    return None


def install(app, cfg: HmacConfig | None = None) -> None:
    """앱에 미들웨어를 붙인다. secret이 없으면 아무것도 하지 않는다(로컬 개발)."""
    cfg = cfg or HmacConfig.from_env()
    app.state.hmac = cfg
    if not cfg.secret:
        # **꺼질 때 반드시 남긴다**(2026-09-11). 이전에는 조용히 return해서 운영 컨테이너가
        # 무검증으로 떠 있는데 아무도 몰랐다. `/health`의 `hmac_enforced`도 같은 사실을 낸다 —
        # 로그는 사람이 봐야 알고 `/health`는 점검 도구가 자동으로 본다.
        logging.getLogger("medimate.auth").warning(
            "HMAC 검증이 꺼진 상태로 기동합니다 — MEDIMATE_HMAC_SECRET이 비어 있어 "
            "모든 요청이 서명 없이 통과합니다. 운영이면 백엔드와 같은 값을 양쪽에 채우세요."
        )
        return

    @app.middleware("http")
    async def _hmac_middleware(request: Request, call_next):
        # 면제는 경로 화이트리스트뿐이다. 메서드 단위 면제는 2026-09-11에 없앴다
        if request.url.path in cfg.exempt_paths:
            return await call_next(request)
        body = await request.body()
        why = verify(cfg, request.method, request.url.path, request.headers, body)
        if why:
            return JSONResponse(status_code=401, content={"detail": f"hmac: {why}"})
        return await call_next(request)
