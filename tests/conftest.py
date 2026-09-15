# ── 테스트는 진짜 LLM을 부르지 않는다 (2026-09-15) ──────────────────────────────────
#
# 오늘 두 번 새 나갔다. 임시 스크립트의 주입이 조용히 무시돼 OpenAI가 돌았고(~$0.2),
# 운영 모양 테스트가 세그멘터를 가짜로 안 바꿔 Bedrock이 한 번 돌았다($0.0003). 둘 다
# "가짜를 끼운 줄 알았는데 진짜가 돈" 모양이다. 어댑터가 실제 클라이언트를 만들려는 순간
# 여기서 끊는다 — 실호출 평가는 tests/가 아니라 scripts/·evals/에서 한다.
import pytest as _pytest


@_pytest.fixture(autouse=True)
def no_real_llm_calls(monkeypatch):
    def _blocked(*a, **k):
        raise RuntimeError("테스트에서 실제 LLM 클라이언트를 만들었다 — 어댑터를 가짜로 바꿔라")

    for mod, attr in (("boto3", "client"), ("openai", "OpenAI"), ("anthropic", "Anthropic")):
        try:
            m = __import__(mod)
        except ImportError:
            continue
        monkeypatch.setattr(m, attr, _blocked, raising=False)
    yield
