"""트레이스 본문 식별자 가림 — 정규식·조각 치환·지연 실음·실패 시 전부 가림(2026-09-28)."""

import contextlib

import pytest

from medimate.obs import pii, tracing


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    tracing.reset_for_tests()
    pii.reset_for_tests()
    for k in (
        "MEDIMATE_TRACE_CONTENT",
        "MEDIMATE_TRACE_PII",
        "MEDIMATE_TRACE_PII_GUARDRAIL",
    ):
        monkeypatch.delenv(k, raising=False)
    yield
    tracing.reset_for_tests()
    pii.reset_for_tests()


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("연락은 010-1234-5678로", "연락은 {PHONE}로"),
        ("번호 01098765432 이고요", "번호 {PHONE} 이고요"),
        ("병원 번호가 02-3010-1234였어요", "병원 번호가 {PHONE}였어요"),
        ("주민번호 900101-1234567", "주민번호 {RRN}"),
        ("9001011234567 이에요", "{RRN} 이에요"),
        ("결과는 a.b@example.com 으로", "결과는 {EMAIL} 으로"),
        ("서울아산병원 정형외과에서", "{HOSPITAL} 정형외과에서"),
        ("10월 12일 오후 3시에 다시 오래요", "{DATE} 오후 3시에 다시 오래요"),
        ("환자번호 12345678번이에요", "환자번호 {ID}번이에요"),
    ],
)
def test_regex_masks_identifiers(text, want):
    assert pii.regex_scrub(text) == want


@pytest.mark.parametrize(
    "text",
    [
        "3일 전부터 오른쪽 무릎이 욱신거려요",
        "타이레놀 500mg 하루 3번 먹으래요",
        "아픈 건 7점 정도예요",
        "10월부터 그랬어요",
        "동네 병원에서 물리치료 받았어요",
        "아침 7시쯤 제일 뻣뻣해요",
    ],
)
def test_regex_keeps_symptom_text(text):
    assert pii.regex_scrub(text) == text


def test_scrub_replaces_matches_everywhere_but_system():
    data = {
        "system": "예시: 김철수",
        "user": "- 질문: 어디가?\n  환자: 저는 김철수고요\n현재 발화: 김철수 선생님이",
    }
    got = pii.scrub(data, [("NAME", "김철"), ("NAME", "김철수")])
    assert got["system"] == "예시: 김철수"  # 고정 프롬프트는 그대로
    assert "김철" not in got["user"] and got["user"].count("{NAME}") == 2
    assert pii.scrub(["저는 김철수", {"evidence": "김철수"}], [("NAME", "김철수")]) == [
        "저는 {NAME}",
        {"evidence": "{NAME}"},
    ]


def test_one_letter_matches_are_ignored():
    # Guardrails가 "욱신"의 "욱"을 이름으로 본 실측(E03) — 전체 치환이라 한 글자는 버린다
    assert pii.scrub("욱신욱신 아파요", [("NAME", "욱")]) == "욱신욱신 아파요"


def test_user_texts_skip_system_prompt():
    assert pii.user_texts({"system": "긴 프롬프트", "user": "무릎"}) == "무릎"


def test_mode(monkeypatch):
    assert pii.mode() == "regex"
    monkeypatch.setenv("MEDIMATE_TRACE_PII_GUARDRAIL", "gid")
    assert pii.mode() == "guardrails"
    monkeypatch.setenv("MEDIMATE_TRACE_PII", "off")
    assert pii.mode() is None


class FakeSpan:
    def __init__(self):
        self.updates = []

    def update(self, **kw):
        self.updates.append(kw)


class FakeClient:
    def __init__(self):
        self.span = FakeSpan()

    @contextlib.contextmanager
    def start_as_current_observation(self, **kw):
        self.started = kw
        yield self.span


def _fake(monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(tracing, "_client", fake)
    monkeypatch.setattr(tracing, "_tried", True)
    return fake


def test_content_on_defers_input_and_masks_with_detected_matches(monkeypatch):
    fake = _fake(monkeypatch)
    monkeypatch.setenv("MEDIMATE_TRACE_CONTENT", "on")
    monkeypatch.setenv("MEDIMATE_TRACE_PII_GUARDRAIL", "gid")
    seen = []
    monkeypatch.setattr(pii, "detect", lambda t: seen.append(t) or [("NAME", "박민수")])
    inp = {"system": "sys", "user": "박민수 선생님이 010-1111-2222로 연락하래요"}
    with tracing.generation("extract-v5-nova", model="nova", input=inp) as g:
        g.done(output='{"evidence": "박민수 선생님이"}', input_tokens=1, output_tokens=1)
    assert fake.started["input"] is None  # 시작 때는 싣지 않는다
    assert seen == [inp["user"]]  # 탐지에는 시스템 프롬프트를 넘기지 않는다
    up = fake.span.updates[-1]
    assert up["input"] == {"system": "sys", "user": "{NAME} 선생님이 {PHONE}로 연락하래요"}
    assert up["output"] == '{"evidence": "{NAME} 선생님이"}'


def test_detection_failure_masks_everything(monkeypatch):
    fake = _fake(monkeypatch)
    monkeypatch.setenv("MEDIMATE_TRACE_CONTENT", "on")
    monkeypatch.setenv("MEDIMATE_TRACE_PII_GUARDRAIL", "gid")

    def boom(_):
        raise TimeoutError

    monkeypatch.setattr(pii, "detect", boom)
    with tracing.generation("g", model="m", input={"user": "박민수"}) as g:
        g.done(output="박민수")
    up = fake.span.updates[-1]
    assert up["input"] == {"user": "<masked 3 chars>"} and up["output"] == "<masked 3 chars>"
    assert up["metadata"]["pii"] == "failed"


def test_regex_only_without_guardrail(monkeypatch):
    fake = _fake(monkeypatch)
    monkeypatch.setenv("MEDIMATE_TRACE_CONTENT", "on")
    with tracing.generation("g", model="m", input={"user": "010-1234-5678 박민수"}) as g:
        g.done(output="ok")
    assert fake.span.updates[-1]["input"] == {"user": "{PHONE} 박민수"}  # 이름은 못 가린다


def test_content_off_is_unchanged(monkeypatch):
    fake = _fake(monkeypatch)
    with tracing.generation("g", model="m", input={"user": "박민수"}) as g:
        g.done(output="o")
    assert fake.started["input"] == {"user": "박민수"}  # SDK의 _mask가 길이만 남긴다
    assert "input" not in fake.span.updates[-1]
    assert tracing._mask(data={"user": "박민수"}) == {"user": "<masked 3 chars>"}


def test_status_reports_pii_mode_only_when_content_is_sent(monkeypatch):
    _fake(monkeypatch)
    assert tracing.status()["pii"] is None
    monkeypatch.setenv("MEDIMATE_TRACE_CONTENT", "on")
    assert tracing.status()["pii"] == "regex"
    monkeypatch.setenv("MEDIMATE_TRACE_PII_GUARDRAIL", "gid")
    assert tracing.status()["pii"] == "guardrails"
