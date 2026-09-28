"""트레이스로 나가는 본문의 식별자 가림 — Langfuse 내보내기 전용(2026-09-28).

본문 수집(`MEDIMATE_TRACE_CONTENT=on`)을 켜면 발화가 Langfuse(미국)로 나간다. 그 전에 식별자를
`{NAME}`처럼 바꾼다. **카드·모델 입력에는 쓰지 않는다** — Guardrails는 한국어 낱말을 이름으로
보고 증상어를 깬다(`욱신` → `{NAME}신`, evals/RESULTS.md "PII 가림").
트레이스에서 조금 잃는 것은 감수한다.

두 층
- 정규식(호출 0, 항상): 전화·주민번호·이메일 + 관리형 유형에 없는 병원명·진료일·등록번호
- Bedrock Guardrails 민감정보 필터(`MEDIMATE_TRACE_PII_GUARDRAIL`이 있을 때): 이름·주소·나이.
  **탐지된 원문 조각(match)만 받아** 입력·출력·근거에서 같은 문자열을 모두 바꾼다 — 발화 한 번
  탐지로 앞 턴 기록과 출력 근거까지 덮는다. 교차 리전 없이 만든 가드레일이면 처리가 서울 안에서
  끝난다

환경변수
- MEDIMATE_TRACE_PII_GUARDRAIL — 가드레일 id. 없으면 정규식만(`/health.tracing.pii: "regex"`)
- MEDIMATE_TRACE_PII_GUARDRAIL_VERSION — 기본 "DRAFT"
- MEDIMATE_TRACE_PII=off — 가림을 끈다(eval 러너처럼 우리가 만든 케이스만 보낼 때)
"""

from __future__ import annotations

import logging
import os
import re
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

logger = logging.getLogger("medimate.obs")

# 순서가 뜻을 가진다 — 주민번호를 전화보다 먼저(13자리가 전화로 잘려 먹히지 않게)
_REGEX: list[tuple[str, re.Pattern[str], str]] = [
    ("RRN", re.compile(r"\d{6}-?[1-4]\d{6}"), "{RRN}"),
    ("PHONE", re.compile(r"(?:\+82[\s-]?)?0\d{1,2}[\s-]?\d{3,4}[\s-]?\d{4}"), "{PHONE}"),
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), "{EMAIL}"),
    # "동네 병원"처럼 띄어 쓴 일반명사는 남고, 붙여 쓴 고유명("서울아산병원")은 가린다.
    # "대학병원"도 가려지지만 트레이스에서 잃는 것은 감수한다
    (
        "HOSPITAL",
        re.compile(r"[가-힣A-Za-z0-9]{2,}(?:한의원|병원|의원|치과|클리닉|의료원|보건소)"),
        "{HOSPITAL}",
    ),
    (
        "DATE",
        re.compile(r"\d{4}[.\-/]\s?\d{1,2}[.\-/]\s?\d{1,2}|\d{1,2}월\s?\d{1,2}일"),
        "{DATE}",
    ),
    (
        "ID",
        re.compile(r"((?:환자|등록|차트|진료|카드)\s?번호\s?[:은는이가]?\s?)\d{4,}"),
        r"\1{ID}",
    ),
]

_pool: ThreadPoolExecutor | None = None
_rt: Any = None


def enabled() -> bool:
    return os.getenv("MEDIMATE_TRACE_PII", "").strip().lower() not in ("off", "0", "false", "no")


def guardrail_id() -> str | None:
    return os.getenv("MEDIMATE_TRACE_PII_GUARDRAIL") or None


def mode() -> str | None:
    """`/health`용 — "guardrails" · "regex" · None(가림 끔)."""
    if not enabled():
        return None
    return "guardrails" if guardrail_id() else "regex"


def regex_scrub(text: str) -> str:
    for _, rx, repl in _REGEX:
        text = rx.sub(repl, text)
    return text


def _runtime():
    global _rt
    if _rt is None:
        import boto3
        from botocore.config import Config

        _rt = boto3.client(
            "bedrock-runtime",
            region_name=os.getenv("MEDIMATE_BEDROCK_REGION", "ap-northeast-2"),
            config=Config(connect_timeout=2, read_timeout=3, retries={"max_attempts": 1}),
        )
    return _rt


def detect(text: str) -> list[tuple[str, str]]:
    """Guardrails가 찾은 (유형, 원문 조각). 가드레일이 없으면 빈 목록. 실패는 예외로 올린다."""
    gid = guardrail_id()
    if not gid or not text.strip():
        return []
    r = _runtime().apply_guardrail(
        guardrailIdentifier=gid,
        guardrailVersion=os.getenv("MEDIMATE_TRACE_PII_GUARDRAIL_VERSION", "DRAFT"),
        source="INPUT",
        content=[{"text": {"text": text}}],
    )
    out: list[tuple[str, str]] = []
    for a in r.get("assessments") or []:
        pol = a.get("sensitiveInformationPolicy") or {}
        for e in pol.get("piiEntities") or []:
            if e.get("match"):
                out.append((e.get("type", "PII"), e["match"]))
        for e in pol.get("regexes") or []:
            if e.get("match"):
                out.append((e.get("name", "PII"), e["match"]))
    return out


def detect_async(text: str) -> Future:
    """Nova 호출과 겹치게 먼저 건다. 결과는 generation이 끝날 때(`done()`) 받는다."""
    global _pool
    if _pool is None:
        _pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="pii")
    return _pool.submit(detect, text)


def user_texts(data: Any) -> str:
    """탐지에 넘길 글. 시스템 프롬프트는 고정 문자열이라 뺀다(넘기면 호출당 5단위가 넘는다)."""
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        return "\n".join(user_texts(v) for k, v in data.items() if k != "system")
    if isinstance(data, list):
        return "\n".join(user_texts(v) for v in data)
    return ""


def scrub(data: Any, matches: list[tuple[str, str]]) -> Any:
    """정규식 + 탐지 조각 치환. dict·list는 재귀, 시스템 프롬프트는 그대로."""
    if isinstance(data, str):
        s = data
        # 긴 조각부터 — "김철수"를 "김철"보다 먼저 바꿔야 끝 글자가 남지 않는다
        # 한 글자 조각은 버린다 — 조각은 트레이스 **전체**에서 바뀌므로 `욱`(욱신을 이름으로 본 것)
        # 하나가 모든 `욱`을 지운다. 한국어 이름은 두 글자 이상이다
        for typ, m in sorted(matches, key=lambda x: -len(x[1])):
            if len(m.strip()) >= 2:
                s = s.replace(m, "{" + typ + "}")
        return regex_scrub(s)
    if isinstance(data, dict):
        return {k: (v if k == "system" else scrub(v, matches)) for k, v in data.items()}
    if isinstance(data, list):
        return [scrub(v, matches) for v in data]
    return data


def reset_for_tests() -> None:
    global _pool, _rt
    _pool, _rt = None, None
