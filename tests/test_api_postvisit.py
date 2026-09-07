"""진료 후 메모 API — 서버 분류 / 클라이언트 라벨 / 1q-2 수정 / 대조. LLM 호출 없음."""

from fastapi.testclient import TestClient

from medimate.api.app import create_app
from medimate.dialog.memo import MemoLabels
from tests.fakes import ScriptedExtractor


class FakeMemoClassifier:
    model_id = "fake-memo"
    prompt_version = "test"

    def __init__(self, labels):
        self._labels = labels

    def classify(self, sentences):
        return MemoLabels(labels=[*self._labels][: len(sentences)])


def make(labels):
    return TestClient(
        create_app(lambda: ScriptedExtractor([]), memo_factory=lambda: FakeMemoClassifier(labels))
    )


MEMO = (
    "위염 초기라고 하셨어요. 혈액검사 했고 결과는 다음에. 약은 2주분. 2주 뒤에 다시 오라고. "
    "병원이 붐볐다."
)


def test_memo_server_classification_builds_card_and_followup_date():
    client = make(["findings", "tests", "medication_instructions", "follow_up", "none"])
    r = client.post(
        "/v1/postvisit/memo",
        json={
            "memo": MEMO,
            "visit_date": "2026-09-12",
            "clinic": "서울OO병원 내과",
            "request_id": "r-1",
        },
    )
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["source"] == "server" and b["request_id"] == "r-1"
    assert len(b["sentences"]) == 5 and b["labels"]["4"] == "none"
    card = b["card"]
    assert card["card_type"] == "postvisit"
    assert card["axes"]["findings"]["value"] == "위염 초기라고 하셨어요."
    assert card["axes"]["findings"]["evidence"] == ["위염 초기라고 하셨어요."]  # 원문 그대로
    assert card["unsorted"] == ["병원이 붐볐다."]
    assert card["follow_up_date"]["date"] == "2026-09-26"
    assert card["clinic"] == "서울OO병원 내과" and card["memo"] == MEMO
    assert card["provenance"]["model_id"] == "fake-memo"


def test_memo_client_labels_skip_llm_and_relabel_flow():
    client = make(["none"] * 5)  # 서버 분류기는 불리면 전부 none — 불리지 않아야 한다
    labels = {
        "0": "findings",
        "1": "tests",
        "2": "medication_instructions",
        "3": "follow_up",
        "4": "none",
    }
    b = client.post(
        "/v1/postvisit/memo",
        json={
            "memo": MEMO,
            "visit_date": "2026-09-12",
            "labels": labels,
            "labels_meta": {"model_id": "qwen3-1.7b-q4_0", "prompt_version": "memo-small-v4"},
        },
    ).json()
    assert b["source"] == "client" and b["usage"]["input_tokens"] == 0
    assert b["card"]["axes"]["tests"]["status"] == "filled"
    assert b["card"]["provenance"]["model_id"] == "qwen3-1.7b-q4_0"
    # 1q-2: 사용자가 4번 문장을 재방문으로 고침 → 같은 엔드포인트, LLM 없음
    labels["4"] = "follow_up"
    b2 = client.post(
        "/v1/postvisit/memo", json={"memo": MEMO, "visit_date": "2026-09-12", "labels": labels}
    ).json()
    assert "병원이 붐볐다." in b2["card"]["axes"]["follow_up"]["evidence"]
    assert b2["card"]["unsorted"] == []


def test_memo_widening_and_site_comparison():
    client = make(["findings"])
    b = client.post(
        "/v1/postvisit/memo",
        json={"memo": "앞십자인대가 좀 늘어났다고 하셨어요.", "previsit_anchor_id": "ANC:014"},
    ).json()
    assert b["card"]["widening"][0]["anchor_label"] == "다리"
    assert b["card"]["site_comparison"] == "same"
    b2 = client.post(
        "/v1/postvisit/memo",
        json={"memo": "앞십자인대가 좀 늘어났다고 하셨어요.", "previsit_anchor_id": "ANC:013"},
    ).json()
    assert b2["card"]["site_comparison"] == "different"


def test_memo_rejects_empty_and_unknown_fields():
    client = make([])
    assert client.post("/v1/postvisit/memo", json={"memo": ""}).status_code == 422
    assert (
        client.post("/v1/postvisit/memo", json={"memo": "x", "diagnosis": "y"}).status_code == 422
    )
