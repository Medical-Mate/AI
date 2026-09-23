"""Langfuse 트레이싱 — 키 없으면 무동작, 있으면 generation·score가 클라이언트로 간다(가짜)."""

import contextlib

import pytest

from medimate.obs import tracing


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    tracing.reset_for_tests()
    for k in (
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_BASE_URL",
        "MEDIMATE_TRACING",
    ):
        monkeypatch.delenv(k, raising=False)
    yield
    tracing.reset_for_tests()


def test_noop_without_keys():
    assert tracing.client() is None
    with tracing.context(trace_name="t", tags=["x"]):
        with tracing.generation("g", model="m", input={"a": 1}) as g:
            g.done(output="o", input_tokens=1, output_tokens=2, cost_usd=0.0)
    tracing.score("em", 1.0)
    tracing.flush()  # 아무 일도 없어야 한다


def test_off_switch_beats_keys(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setenv("MEDIMATE_TRACING", "off")
    assert tracing.enabled() is False and tracing.client() is None


class FakeSpan:
    def __init__(self):
        self.updates = []

    def update(self, **kw):
        self.updates.append(kw)


class FakeClient:
    def __init__(self):
        self.span = FakeSpan()
        self.scores = []
        self.flushed = 0

    @contextlib.contextmanager
    def start_as_current_observation(self, **kw):
        self.started = kw
        yield self.span

    def score_current_trace(self, **kw):
        self.scores.append(kw)

    def flush(self):
        self.flushed += 1


def test_generation_records_usage_cost_and_masks_content(monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(tracing, "_client", fake)
    monkeypatch.setattr(tracing, "_tried", True)
    with tracing.generation("extract-v5-nova", model="nova", input={"user": "ㄴㄴ"}) as g:
        g.done(output="{}", input_tokens=10, output_tokens=2, cost_usd=0.001)
    assert fake.started["as_type"] == "generation" and fake.started["model"] == "nova"
    up = fake.span.updates[-1]
    assert up["usage_details"] == {"input": 10, "output": 2} and up["cost_details"] == {
        "total": 0.001
    }
    tracing.score("em", 1.0, comment="gold=x")
    assert fake.scores[-1]["name"] == "em" and fake.scores[-1]["value"] == 1.0
    tracing.flush()
    assert fake.flushed == 1


def test_mask_hides_content_unless_allowed(monkeypatch):
    monkeypatch.delenv("MEDIMATE_TRACE_CONTENT", raising=False)
    assert tracing._mask(data={"user": "무릎이 아파요", "n": 3}) == {
        "user": "<masked 7 chars>",
        "n": 3,
    }
    monkeypatch.setenv("MEDIMATE_TRACE_CONTENT", "on")
    assert tracing._mask(data={"user": "무릎이 아파요"}) == {"user": "무릎이 아파요"}


def test_llm_extractor_call_goes_through_tracing(monkeypatch):
    from medimate.llm.providers import LLMExtractor

    fake = FakeClient()
    monkeypatch.setattr(tracing, "_client", fake)
    monkeypatch.setattr(tracing, "_tried", True)
    ex = LLMExtractor("bedrock", "apac.amazon.nova-pro-v1:0")
    monkeypatch.setattr(ex, "_dispatch", lambda s, u: ('{"ok":1}', 100, 5))
    text, i, o = ex._call("sys", "usr")
    assert text == '{"ok":1}' and (i, o) == (100, 5)
    assert fake.started["name"] == ex.prompt_version and fake.started["model"] == ex.model_id
    up = fake.span.updates[-1]
    assert up["usage_details"] == {"input": 100, "output": 5}
    assert abs(up["cost_details"]["total"] - (100 * 0.80 + 5 * 3.20) / 1e6) < 1e-12


def test_status_is_all_off_without_keys(monkeypatch):
    monkeypatch.setenv("MEDIMATE_TRACE_CONTENT", "on")  # 스위치만 켜도
    assert tracing.status() == {"enabled": False, "content": False, "env": None, "host": None}


def test_status_reports_content_only_when_actually_sent(monkeypatch):
    """`/health.tracing.content`는 본문이 실제로 나가는지다 — 국외 이전 안내와 맞춰 볼 값(#127)."""
    monkeypatch.setattr(tracing, "_client", FakeClient())
    monkeypatch.setattr(tracing, "_tried", True)
    monkeypatch.setenv("MEDIMATE_ENV", "prod")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")
    monkeypatch.delenv("MEDIMATE_TRACE_CONTENT", raising=False)
    assert tracing.status() == {
        "enabled": True,
        "content": False,
        "env": "prod",
        "host": "https://us.cloud.langfuse.com",
    }
    monkeypatch.setenv("MEDIMATE_TRACE_CONTENT", "on")
    assert tracing.status()["content"] is True


def test_health_carries_tracing_status():
    from fastapi.testclient import TestClient

    from medimate.api.app import create_app

    body = TestClient(create_app()).get("/health").json()
    assert body["tracing"] == {"enabled": False, "content": False, "env": None, "host": None}
