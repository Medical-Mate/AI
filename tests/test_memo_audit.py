"""postvisit 응답의 `audit` — previsit `TurnAudit`과 같은 자리 (#113, 2026-09-21)."""

from fastapi.testclient import TestClient

from medimate.api.app import create_app
from medimate.dialog.memo import MemoLabels

MEMO = "위염 초기라고 하셨어요. 2주 뒤에 다시 오라고 하셨어요."


class Fixed:
    model_id = "fake-model"
    prompt_version = "memo-small-v5"
    last_text = '{"0":"findings","1":"follow_up"}'

    def classify(self, sentences):
        return MemoLabels.from_keyed({"0": "findings", "1": "follow_up"}, len(sentences))


def test_server_path_returns_an_audit_with_the_raw_model_output():
    client = TestClient(create_app(memo_factory=Fixed))
    r = client.post("/v1/postvisit/memo", json={"memo": MEMO, "visit_date": "2026-09-21"})
    assert r.status_code == 200
    a = r.json()["audit"]
    assert a["memo"] == MEMO
    assert a["sentences"] == ["위염 초기라고 하셨어요.", "2주 뒤에 다시 오라고 하셨어요."]
    assert a["labels"] == r.json()["labels"] == {"0": "findings", "1": "follow_up"}
    assert a["raw_output"] == Fixed.last_text  # 모델 원본 그대로
    assert a["source"] == "server" and a["prompt_version"] == "memo-small-v5"
    assert a["model_id"] == "fake-model" and a["split_version"] == "split-v4"
    assert a["lifestyle_axis"] in (True, False)
    assert set(a["usage"]) == {"input_tokens", "output_tokens", "cost_usd"}


def test_client_path_has_an_audit_without_raw_output():
    client = TestClient(create_app())
    r = client.post(
        "/v1/postvisit/memo",
        json={"memo": MEMO, "labels": {"0": "findings", "1": "follow_up"}},
    )
    a = r.json()["audit"]
    assert a["source"] == "client" and a["raw_output"] is None
    assert a["prompt_version"] == "client-labels"
    assert a["labels"] == {"0": "findings", "1": "follow_up"}


def test_audit_carries_nothing_but_what_the_patient_typed():
    """식별자 자리가 없다 — 키 목록이 계약이다."""
    client = TestClient(create_app(memo_factory=Fixed))
    a = client.post("/v1/postvisit/memo", json={"memo": MEMO, "clinic": "OO내과"}).json()["audit"]
    assert set(a) == {
        "memo",
        "sentences",
        "labels",
        "raw_output",
        "dropped",
        "source",
        "prompt_version",
        "model_id",
        "lifestyle_axis",
        "split_version",
        "usage",
    }
    assert "OO내과" not in str(a)  # clinic은 카드에만, audit에는 없다
