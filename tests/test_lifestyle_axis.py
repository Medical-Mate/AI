"""지침(lifestyle_instructions) 축 (2026-09-15).

모델은 항상 나누고, 카드는 플래그에 따라 접는다.
"""

from datetime import date

from fastapi.testclient import TestClient

from medimate.api.app import create_app
from medimate.dialog.memo import MemoLabels, classify_memo, fold_lifestyle
from medimate.schema.export import to_backend_payload
from medimate.schema.postvisit import PostAxis

MEMO = "위산 줄이는 약을 2주분 먹으라고 하셨어요. 커피랑 매운 음식은 줄이라고 하셨어요."


class Fixed:
    model_id = "fake"
    prompt_version = "test"

    def __init__(self, labels):
        self.labels = labels

    def classify(self, sentences):
        return MemoLabels.from_keyed(
            {str(i): self.labels[i] for i in range(len(sentences))}, len(sentences)
        )


def test_flag_off_folds_lifestyle_into_medication_and_hides_the_axis(monkeypatch):
    """앱이 "지침" 행을 그리기 전까지는 예전 화면 그대로 — 줄이 사라지면 안 된다."""
    monkeypatch.delenv("MEDIMATE_LIFESTYLE_AXIS", raising=False)
    res = classify_memo(
        MEMO,
        Fixed(["medication_instructions", "lifestyle_instructions"]),
        visit_date=date(2026, 9, 15),
    )
    med = res.card.axes[PostAxis.MEDICATION_INSTRUCTIONS]
    assert med.value == "위산 줄이는 약을 2주분 먹기 · 커피랑 매운 음식은 줄이기"
    assert res.labels == {0: "medication_instructions", 1: "medication_instructions"}
    payload = to_backend_payload(res.card)
    assert "lifestyle_instructions" not in payload["axes"]
    assert set(payload["axes"]) == {"findings", "tests", "medication_instructions", "follow_up"}
    assert payload["completeness"] == 1.0  # 메모 카드는 빈 축을 unknown으로 닫는다 — 접혀도 1.0


def test_flag_on_splits_the_two_axes(monkeypatch):
    monkeypatch.setenv("MEDIMATE_LIFESTYLE_AXIS", "1")
    res = classify_memo(
        MEMO,
        Fixed(["medication_instructions", "lifestyle_instructions"]),
        visit_date=date(2026, 9, 15),
    )
    ax = res.card.axes
    assert ax[PostAxis.MEDICATION_INSTRUCTIONS].value == "위산 줄이는 약을 2주분 먹기"
    assert ax[PostAxis.LIFESTYLE_INSTRUCTIONS].value == "커피랑 매운 음식은 줄이기"
    assert res.labels == {0: "medication_instructions", 1: "lifestyle_instructions"}
    payload = to_backend_payload(res.card)
    assert payload["axes"]["lifestyle_instructions"]["status"] == "filled"
    assert payload["completeness"] == 1.0


def test_fold_is_the_only_place_the_two_meet(monkeypatch):
    monkeypatch.delenv("MEDIMATE_LIFESTYLE_AXIS", raising=False)
    assert fold_lifestyle({0: "lifestyle_instructions", 1: "tests"}) == {
        0: "medication_instructions",
        1: "tests",
    }
    monkeypatch.setenv("MEDIMATE_LIFESTYLE_AXIS", "true")
    assert fold_lifestyle({0: "lifestyle_instructions"}) == {0: "lifestyle_instructions"}


def test_health_shows_whether_the_axis_is_on(monkeypatch):
    monkeypatch.setenv("MEDIMATE_LIFESTYLE_AXIS", "1")
    assert TestClient(create_app()).get("/health").json()["extractor"]["lifestyle_axis"] is True
    monkeypatch.delenv("MEDIMATE_LIFESTYLE_AXIS", raising=False)
    assert TestClient(create_app()).get("/health").json()["extractor"]["lifestyle_axis"] is False


def test_prompts_carry_the_new_label():
    from medimate.llm import prompt_memo_segments, prompt_memo_small

    assert prompt_memo_small.PROMPT_VERSION == "memo-small-v5"
    assert "lifestyle_instructions" in prompt_memo_small.system_prompt()
    assert prompt_memo_segments.PROMPT_VERSION == "memo-v6"
    assert "lifestyle_instructions" in prompt_memo_segments.LABELS
