"""memo-v5 — LLM이 조각을 내고 코드가 원문을 덮는지 검증한다 (2026-09-15)."""

from datetime import date

from fastapi.testclient import TestClient

from medimate.api.app import create_app
from medimate.dialog.memo import assemble_from_client, segment_memo, slice_by_coverage
from medimate.schema.postvisit import PostAxis

MEMO = "일주일치약처방 이주일후재방문"


class Seg:
    model_id = "fake"
    prompt_version = "memo-v5"
    usage = None

    def __init__(self, segments):
        self.segments = segments
        self.calls = 0

    def segment(self, memo):
        self.calls += 1
        if isinstance(self.segments, Exception):
            raise self.segments
        return {"segments": self.segments}


def test_coverage_slices_are_verbatim_substrings_of_the_memo():
    got = slice_by_coverage(
        "일주일치 약처방 이주일 후 재방문", ["일주일치약처방", "이주일후재방문"]
    )
    assert got == ["일주일치 약처방", "이주일 후 재방문"]  # 원문 띄어쓰기로 돌아온다
    assert slice_by_coverage(MEMO, ["일주일치약처방", "이주일후재방문"]) == [
        "일주일치약처방",
        "이주일후재방문",
    ]


def test_coverage_rejects_paraphrase_omission_and_addition():
    assert slice_by_coverage(MEMO, ["일주일치 약 처방", "이주일 뒤 재방문"]) is None  # 고쳐 씀
    assert slice_by_coverage(MEMO, ["일주일치약처방"]) is None  # 빠뜨림
    assert (
        slice_by_coverage(MEMO, ["일주일치약처방", "이주일후재방문", "감사합니다"]) is None
    )  # 덧붙임
    assert slice_by_coverage(MEMO, []) is None


def test_segment_memo_builds_the_card_from_llm_pieces():
    seg = Seg(
        [
            {"text": "일주일치약처방", "label": "medication_instructions"},
            {"text": "이주일후재방문", "label": "follow_up"},
        ]
    )
    res = segment_memo(MEMO, seg, visit_date=date(2026, 9, 15))
    ax = res.card.axes
    assert res.sentences == ["일주일치약처방", "이주일후재방문"]
    assert ax[PostAxis.MEDICATION_INSTRUCTIONS].value == "일주일치"
    assert ax[PostAxis.FOLLOW_UP].value == "이주일후"
    assert ax[PostAxis.FOLLOW_UP].evidence == ["이주일후재방문"]
    assert res.card.follow_up_date.date == "2026-09-29"


def test_segment_memo_returns_none_when_pieces_do_not_cover_the_memo():
    bad = Seg([{"text": "일주일치 약을 처방받음", "label": "medication_instructions"}])
    assert segment_memo(MEMO, bad, visit_date=date(2026, 9, 15)) is None
    assert segment_memo(MEMO, Seg(RuntimeError("boom")), visit_date=date(2026, 9, 15)) is None
    assert segment_memo(MEMO, Seg([{"text": MEMO, "label": "진단"}])) is None  # 라벨 밖


def test_client_can_send_the_pieces_back_and_nothing_is_resplit():
    res = assemble_from_client(
        MEMO,
        ["일주일치약처방", "이주일후재방문"],
        {"0": "medication_instructions", "1": "follow_up"},
        visit_date=date(2026, 9, 15),
    )
    assert res.sentences == ["일주일치약처방", "이주일후재방문"]
    assert res.card.axes[PostAxis.FOLLOW_UP].value == "이주일후"
    assert assemble_from_client(MEMO, ["다른 메모"], {"0": "follow_up"}) is None


def test_api_server_path_uses_segments_and_falls_back_when_they_fail(monkeypatch):
    """운영 모양(주입 없음)으로 앱을 만들고 LLM 어댑터만 가짜로 바꾼다."""
    import medimate.api.app as appmod
    from medimate.dialog.memo import MemoLabels

    calls = {"clf": 0, "seg": 0}

    class _U:
        def __init__(self, c):
            self.input_tokens, self.output_tokens, self._c = 1000, 50, c

        def cost_usd(self, m):
            return self._c

    class FakeClf:
        model_id = "fake"
        prompt_version = "memo-small-v4"

        def __init__(self, *a, **k):
            self.usage = _U(0.0013)

        def classify(self, sentences):
            calls["clf"] += 1
            labs = ["medication_instructions", "follow_up"][: len(sentences)]
            return MemoLabels.from_keyed({str(i): x for i, x in enumerate(labs)}, len(sentences))

    class FakeReader:
        model_id = "fake"
        prompt_version = "followup-v1"

        def __init__(self, *a, **k):
            self.usage = _U(0.0)

        def read(self, sentence, prev_text=None):
            return {"text": "", "days": None, "month": None, "day": None}

    good = [
        {"text": "일주일치약처방", "label": "medication_instructions"},
        {"text": "이주일후재방문", "label": "follow_up"},
    ]
    mode = {"ok": True}

    class FakeSeg:
        model_id = "fake"
        prompt_version = "memo-v5"

        def __init__(self, *a, **k):
            self.usage = _U(0.0015)

        def segment(self, memo):
            calls["seg"] += 1
            return {"segments": good if mode["ok"] else [{"text": "엉뚱한 말", "label": "none"}]}

    monkeypatch.setattr(appmod, "LLMMemoClassifier", FakeClf)
    monkeypatch.setattr(appmod, "LLMFollowUpReader", FakeReader)
    monkeypatch.setattr(appmod, "LLMMemoSegmenter", FakeSeg)
    client = TestClient(create_app())
    body = {"memo": MEMO, "visit_date": "2026-09-15"}

    r = client.post("/v1/postvisit/memo", json=body).json()
    assert r["sentences"] == ["일주일치약처방", "이주일후재방문"]
    assert r["card"]["provenance"]["prompt_version"] == "memo-v5"
    assert r["card"]["axes"]["follow_up"]["value"] == "이주일후"
    assert calls == {"clf": 0, "seg": 1}  # 조각이 원문을 덮었으니 v4 분류는 안 부른다
    assert r["usage"]["cost_usd"] == 0.0015

    mode["ok"] = False
    r = client.post("/v1/postvisit/memo", json=body).json()
    assert r["card"]["provenance"]["prompt_version"] == "memo-small-v4"  # 폴백
    assert calls == {"clf": 1, "seg": 2}
    assert r["usage"]["cost_usd"] == 0.0028  # 실패한 세그멘터 호출도 비용은 나갔다


def test_api_labels_with_sentences_echo_does_not_resplit_and_ignores_split_version():
    client = TestClient(create_app())
    body = {
        "memo": MEMO,
        "visit_date": "2026-09-15",
        "sentences": ["일주일치약처방", "이주일후재방문"],
        "labels": {"0": "medication_instructions", "1": "follow_up"},
        "split_version": "split-v0",  # 조각을 되보냈으니 번호 규칙은 상관없다
    }
    r = client.post("/v1/postvisit/memo", json=body)
    assert r.status_code == 200
    assert r.json()["sentences"] == ["일주일치약처방", "이주일후재방문"]
    assert r.json()["card"]["axes"]["follow_up"]["value"] == "이주일후"

    body["sentences"] = ["일주일치약처방"]  # 원문을 못 덮는다
    assert client.post("/v1/postvisit/memo", json=body).status_code == 409


def test_well_formed_memos_keep_the_v4_path_and_never_call_the_segmenter(monkeypatch):
    """마침표가 있는 정상 입력은 규칙 분리 + v4 분류(라벨 98.5%). v5는 붙여 쓴 입력 전용이다."""
    import medimate.api.app as appmod
    from medimate.dialog.memo import MemoLabels

    calls = {"clf": 0, "seg": 0}

    class _U:
        input_tokens = output_tokens = 0

        def cost_usd(self, m):
            return 0.0

    class FakeClf:
        model_id = "fake"
        prompt_version = "memo-small-v4"
        usage = _U()

        def __init__(self, *a, **k):
            pass

        def classify(self, sentences):
            calls["clf"] += 1
            labs = ["findings", "follow_up"][: len(sentences)]
            return MemoLabels.from_keyed({str(i): x for i, x in enumerate(labs)}, len(sentences))

    class FakeSeg:
        model_id = "fake"
        prompt_version = "memo-v5"
        usage = _U()

        def __init__(self, *a, **k):
            pass

        def segment(self, memo):
            calls["seg"] += 1
            return {"segments": [{"text": memo, "label": "none"}]}

    class FakeReader:
        model_id = "fake"
        prompt_version = "followup-v1"
        usage = _U()

        def __init__(self, *a, **k):
            pass

        def read(self, sentence, prev_text=None):
            return {"text": "", "days": None, "month": None, "day": None}

    monkeypatch.setattr(appmod, "LLMMemoClassifier", FakeClf)
    monkeypatch.setattr(appmod, "LLMMemoSegmenter", FakeSeg)
    monkeypatch.setattr(appmod, "LLMFollowUpReader", FakeReader)
    client = TestClient(create_app())
    r = client.post(
        "/v1/postvisit/memo",
        json={
            "memo": "위염 초기라고 하셨어요. 2주 뒤에 다시 오라고 하셨어요.",
            "visit_date": "2026-09-15",
        },
    ).json()
    assert calls == {"clf": 1, "seg": 0}
    assert r["card"]["provenance"]["prompt_version"] == "memo-small-v4"
    assert r["sentences"] == ["위염 초기라고 하셨어요.", "2주 뒤에 다시 오라고 하셨어요."]
