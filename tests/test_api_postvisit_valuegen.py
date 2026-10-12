"""진료 후 카드 값 — C안(생성+검증) 운영 연결 (2026-10-12).

env `MEDIMATE_SPAN_SELECT=gen`일 때 서버 분류 경로의 소견·검사·약·재방문 값은 생성기가 만든다.
폰 라벨 경로는 외부 LLM 없음이 전제라 생성기를 안 부른다. 꺼져 있으면 지금까지와 똑같다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

from medimate.api.app import create_app
from medimate.dialog.memo import MemoLabels
from medimate.span.verify import Verdict
from tests.fakes import ScriptedExtractor

MEMO = (
    "위염 초기라고 하셨어요. 위산 줄이는 약 2주분 먹으라고 하셨어요. 2주 뒤에 다시 오라고 하셨어요."
)
LABELS = ["findings", "medication_instructions", "follow_up"]


class FakeClf:
    model_id = "fake"
    prompt_version = "memo-small-v5"

    def __init__(self):
        self.usage = _Usage()

    def classify(self, sentences):
        labs = LABELS[: len(sentences)]
        return MemoLabels.from_keyed({str(i): x for i, x in enumerate(labs)}, len(sentences))


@dataclass
class _Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def cost_usd(self, model):
        return self.input_tokens * 1e-6


@dataclass
class _Gen:
    value: str | None
    source: str
    model_value: str | None = None
    verdict: Verdict = field(default_factory=lambda: Verdict(True))
    dropped: str = ""


class FakeGen:
    """조각별로 정해진 답을 낸다. 어떤 조각에 뭘 냈는지 기록한다."""

    model_id = "fake"
    prompt_version = "value-gen-v2"

    def __init__(self, answers):
        self.answers = answers
        self.calls = []
        self.usage = _Usage()

    def generate(self, segment, axis):
        self.calls.append((segment, axis))
        self.usage.input_tokens += 100
        return self.answers.get(segment, _Gen(None, "none"))


def _client(gen):
    return TestClient(
        create_app(
            lambda: ScriptedExtractor([]),
            memo_factory=FakeClf,
            value_gen_factory=(lambda: gen) if gen else None,
        )
    )


def test_off_by_default_values_are_tidied_lines_and_the_generator_is_never_called(monkeypatch):
    monkeypatch.delenv("MEDIMATE_SPAN_SELECT", raising=False)
    gen = FakeGen({})
    r = _client(gen).post("/v1/postvisit/memo", json={"memo": MEMO, "visit_date": "2026-09-15"})
    assert r.status_code == 200, r.text
    b = r.json()
    assert gen.calls == []
    assert b["audit"]["span_select"] == "rule" and b["audit"]["values"] is None
    assert b["card"]["axes"]["findings"]["value"] == "위염 초기"
    assert r.json()["card"]["axes"]["findings"]["evidence"] == ["위염 초기라고 하셨어요."]


def test_gen_mode_uses_generated_values_falls_back_per_sentence_and_audits(monkeypatch):
    monkeypatch.setenv("MEDIMATE_SPAN_SELECT", "gen")
    gen = FakeGen(
        {
            "위염 초기라고 하셨어요.": _Gen("위염 초기", "generated", "위염 초기"),
            # 검증 실패 → rule 폴백 값
            "위산 줄이는 약 2주분 먹으라고 하셨어요.": _Gen(
                "위산 줄이는 약 2주분", "fallback", "위산약 2주", dropped="'위산약 2주' ← source"
            ),
            # 모델이 값 없음 → 어미 정리 줄로
            "2주 뒤에 다시 오라고 하셨어요.": _Gen(None, "none"),
        }
    )
    r = _client(gen).post("/v1/postvisit/memo", json={"memo": MEMO, "visit_date": "2026-09-15"})
    assert r.status_code == 200, r.text
    b = r.json()
    axes = b["card"]["axes"]
    assert axes["findings"]["value"] == "위염 초기"
    assert axes["medication_instructions"]["value"] == "위산 줄이는 약 2주분"
    assert axes["follow_up"]["value"] == "2주 뒤에 다시 오기"  # none → tidy_value
    # evidence는 언제나 원문
    assert axes["medication_instructions"]["evidence"] == [
        "위산 줄이는 약 2주분 먹으라고 하셨어요."
    ]
    assert [a for _, a in gen.calls] == ["findings", "medication_instructions", "follow_up"]
    audit = b["audit"]
    assert audit["span_select"] == "gen"
    assert [v["source"] for v in audit["values"]] == ["generated", "fallback", "none"]
    assert audit["values"][1]["dropped"].startswith("'위산약 2주'")
    # 응답 usage에 생성기 몫이 든다
    assert b["usage"]["input_tokens"] >= 300
    # 재방문 날짜는 그대로 규칙
    assert b["card"]["follow_up_date"]["date"] == "2026-09-29"


def test_gen_mode_does_not_touch_the_phone_label_path(monkeypatch):
    """폰이 라벨을 보낸 경로는 외부 LLM 없음 — 생성기를 부르지 않는다."""
    monkeypatch.setenv("MEDIMATE_SPAN_SELECT", "gen")
    gen = FakeGen({"위염 초기라고 하셨어요.": _Gen("위염 초기", "generated", "위염 초기")})
    body = {
        "memo": MEMO,
        "visit_date": "2026-09-15",
        "sentences": [
            "위염 초기라고 하셨어요.",
            "위산 줄이는 약 2주분 먹으라고 하셨어요.",
            "2주 뒤에 다시 오라고 하셨어요.",
        ],
        "labels": {"0": "findings", "1": "medication_instructions", "2": "follow_up"},
    }
    r = _client(gen).post("/v1/postvisit/memo", json=body)
    assert r.status_code == 200, r.text
    assert gen.calls == []
    assert r.json()["audit"]["span_select"] == "rule"


def test_a_generator_error_falls_back_to_the_tidied_line_but_budget_errors_propagate(monkeypatch):
    monkeypatch.setenv("MEDIMATE_SPAN_SELECT", "gen")

    class Boom(FakeGen):
        def generate(self, segment, axis):
            raise RuntimeError("offline")

    r = _client(Boom({})).post(
        "/v1/postvisit/memo", json={"memo": MEMO, "visit_date": "2026-09-15"}
    )
    assert r.status_code == 200
    b = r.json()
    assert b["card"]["axes"]["findings"]["value"] == "위염 초기"
    assert all(v["source"] == "error" for v in b["audit"]["values"])

    from medimate.llm.providers import BudgetExceeded

    class Broke(FakeGen):
        def generate(self, segment, axis):
            raise BudgetExceeded("예산 초과")

    r = _client(Broke({})).post(
        "/v1/postvisit/memo", json={"memo": MEMO, "visit_date": "2026-09-15"}
    )
    assert r.status_code == 503


def test_health_reports_span_select(monkeypatch):
    monkeypatch.setenv("MEDIMATE_SPAN_SELECT", "gen")
    assert _client(None).get("/health").json()["span_select"] == "gen"
    monkeypatch.setenv("MEDIMATE_SPAN_SELECT", "nonsense")
    assert _client(None).get("/health").json()["span_select"] == "rule"


@pytest.mark.parametrize("flag", ["gen", "GEN", " gen "])
def test_flag_parsing_is_forgiving(monkeypatch, flag):
    from medimate.dialog.memo import span_select_mode

    monkeypatch.setenv("MEDIMATE_SPAN_SELECT", flag)
    assert span_select_mode() == "gen"
