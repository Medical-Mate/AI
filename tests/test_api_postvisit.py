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
    # value는 어미를 정리한 줄, evidence는 **문장 원문 그대로**(2026-09-14 ㉡)
    assert card["axes"]["findings"]["value"] == "위염 초기"
    assert card["axes"]["findings"]["evidence"] == ["위염 초기라고 하셨어요."]
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


def test_memo_split_only_skips_llm_and_returns_sentences():
    client = make(["findings"] * 5)  # 서버 분류기가 불리면 findings가 찍힌다 — 불리지 않아야 한다
    b = client.post(
        "/v1/postvisit/memo",
        json={"memo": MEMO, "visit_date": "2026-09-12", "classify": False},
    ).json()
    assert b["source"] == "none" and b["usage"]["input_tokens"] == 0
    assert len(b["sentences"]) == 5 and all(v == "none" for v in b["labels"].values())
    assert b["card"]["unsorted"] == b["sentences"]  # 분류 안 했으니 전부 미분류로 보존
    assert b["card"]["follow_up_date"] is None
    # labels가 오면 classify=false는 무시되고 조립한다
    labels = {str(i): "none" for i in range(5)}
    labels["3"] = "follow_up"
    b2 = client.post(
        "/v1/postvisit/memo",
        json={"memo": MEMO, "visit_date": "2026-09-12", "classify": False, "labels": labels},
    ).json()
    assert b2["source"] == "client" and b2["card"]["follow_up_date"]["date"] == "2026-09-26"


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


# ── 운영 경로: 지연 초기화에서 리더가 같이 만들어지는가 (2026-09-15) ────────────────


def test_the_lazy_production_path_builds_the_reader_too(monkeypatch):
    """`create_app()`에 아무것도 주입하지 않은 **운영 모양**으로 두 번 부른다.

    전에는 `_followup_reader`가 "memo_factory가 차 있으면 테스트 주입"이라고 추론해서
    운영에서 리더가 한 번도 안 만들어졌다 — 백엔드가 운영 12건에서 `LLM 읽음`이 0번인 것으로
    잡았다. 첫 요청(캐시 비어 있음)과 두 번째 요청(캐시 차 있음) 둘 다 리더가 붙어야 한다.
    """
    import medimate.api.app as appmod

    calls = {"clf": 0, "read": 0}

    class _Usage:
        def __init__(self, i, o, c):
            self.input_tokens, self.output_tokens, self._c = i, o, c

        def cost_usd(self, model_id):
            return self._c

    class FakeClf:
        model_id = "fake"
        prompt_version = "memo-small-v4"

        def __init__(self, *a, **k):
            self.usage = _Usage(1200, 30, 0.0013)

        def classify(self, sentences):
            from medimate.dialog.memo import MemoLabels

            calls["clf"] += 1
            labs = ["medication_instructions", "follow_up"][: len(sentences)]
            return MemoLabels.from_keyed({str(i): x for i, x in enumerate(labs)}, len(sentences))

    class FakeReader:
        model_id = "fake"
        prompt_version = "followup-v1"

        def __init__(self, *a, **k):
            self.usage = _Usage(0, 0, 0.0)

        def read(self, sentence, prev_text=None):
            calls["read"] += 1
            self.usage = _Usage(800, 20, 0.0008)
            return {"text": "담주", "days": 7, "month": None, "day": None}

    monkeypatch.setattr(appmod, "LLMMemoClassifier", FakeClf)
    monkeypatch.setattr(appmod, "LLMFollowUpReader", FakeReader)
    client = TestClient(create_app())  # 주입 없음 = 운영 모양
    body = {"memo": "약 받았어요. 담주에 다시 오래요.", "visit_date": "2026-09-15"}
    for expected_reads in (1, 2):
        r = client.post("/v1/postvisit/memo", json=body)
        assert r.status_code == 200
        fu = r.json()["card"]["follow_up_date"]
        assert fu["date"] == "2026-09-22" and fu["basis"].startswith("LLM 읽음 '담주'")
        assert calls["read"] == expected_reads
        # 응답 usage는 분류기 + 리더 **둘 다** 담는다. 리더 몫이 빠지면 켜졌는지 비용으로 못 본다
        u = r.json()["usage"]
        assert u["cost_usd"] == 0.0021 and u["input_tokens"] == 2000


def test_a_dead_reader_leaves_one_line_in_the_log_without_the_sentence(caplog):
    """2차 호출이 죽어도 카드는 나간다. 다만 **죽었다는 사실은 남는다** — 문장은 안 남는다."""
    import logging
    from datetime import date

    from medimate.dialog.memo import followup_from_reader

    class Dead:
        def read(self, sentence, prev_text=None):
            raise TimeoutError("bedrock")

    with caplog.at_level(logging.WARNING, logger="medimate.memo"):
        assert followup_from_reader(Dead(), "담주에 보자고", date(2026, 9, 15)) is None
    lines = [r.getMessage() for r in caplog.records if r.name == "medimate.memo"]
    assert len(lines) == 1 and "TimeoutError" in lines[0]
    assert "담주" not in lines[0]  # 진료 내용은 로그에 안 적는다
