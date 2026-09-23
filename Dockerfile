# 진료 메이트 AI 서버 — 무상태 FastAPI. 백엔드와 같은 내부망에 컨테이너 하나.
# 모델 가중치 없음(추출은 OpenAI 폴백 또는 폰). 1 vCPU · 1GB면 충분.
#
#   docker build -t medimate-ai .
#   docker run --rm -p 8000:8000 --env-file .env medimate-ai
#
# 환경변수: OPENAI_API_KEY(폴백 추출), MEDIMATE_PROVIDER/MEDIMATE_MODEL, MEDIMATE_HMAC_SECRET(비우면 검증 생략)

FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
# 의존성 먼저 (캐시)
# obs = Langfuse. 키(LANGFUSE_*)가 없으면 무동작이고, 있어도 본문은 가림(MEDIMATE_TRACE_CONTENT=on 전까지)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --group providers --group api --group obs --no-install-project

# 코드 + 데이터(온톨로지 CSV는 부위 마스터 API가 읽는다)
COPY src ./src
COPY data/ontology ./data/ontology
RUN uv sync --frozen --no-dev --group providers --group api --group obs

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD ["uv", "run", "python", "-c", \
  "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"]
CMD ["uv", "run", "uvicorn", "medimate.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
