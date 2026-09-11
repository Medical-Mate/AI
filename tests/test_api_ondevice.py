"""온디바이스 프로필·선택지·HMAC·부위 마스터 — API 계약의 선택 필드들. LLM 호출 없음."""

import json
import re
import time

from fastapi.testclient import TestClient

from medimate.api import auth
from medimate.api.app import create_app
from medimate.dialog.questions import QUESTIONS
from medimate.llm import AxisUpdate, TurnExtraction
from medimate.schema import Axis, FieldStatus
from tests.fakes import ScriptedExtractor
from tests.test_api import make_client


def filled(axis, value, evidence):
    return AxisUpdate(axis=axis, status=FieldStatus.FILLED, value=value, evidence=evidence)


def test_ondevice_profile_starts_with_first_axis_question_and_no_llm():
    client, made = make_client([])
    b = client.post(
        "/v1/previsit/sessions",
        json={"site_node_id": "SUR:041", "profile": "ondevice", "request_id": "r-1"},
    ).json()
    assert b["state"]["profile"] == "ondevice" and b["request_id"] == "r-1"
    assert b["reply"] == QUESTIONS[Axis.ONSET]  # 부위는 인체도로 채워졌으니 첫 축 질문부터
    assert all(ex.calls == [] for ex in made)


def test_device_extraction_is_used_without_calling_server_extractor():
    client, made = make_client([])
    state = client.post(
        "/v1/previsit/sessions", json={"profile": "ondevice", "site_label": "허리"}
    ).json()["state"]
    ext = TurnExtraction(updates=[filled(Axis.ONSET, "3일 전부터", "3일 전부터요")]).model_dump()
    t = client.post(
        "/v1/previsit/turns",
        json={
            "state": state,
            "utterance": "3일 전부터요",
            "extraction": ext,
            "extraction_meta": {
                "model_id": "qwen3-1.7b-q4_0",
                "prompt_version": "extract-small-v4",
            },
            "request_id": "r-2",
        },
    ).json()
    assert all(ex.calls == [] for ex in made)  # 서버 Extractor는 부르지 않았다
    assert t["card"]["axes"]["onset"]["value"] == "3일 전부터"
    assert t["audit"]["source"] == "device" and t["audit"]["usage"]["input_tokens"] == 0
    assert t["card"]["provenance"]["model_id"] == "qwen3-1.7b-q4_0"
    assert t["request_id"] == "r-2"


def test_device_extraction_is_guarded_and_dropped_items_are_audited():
    client, _ = make_client([])
    state = client.post(
        "/v1/previsit/sessions", json={"profile": "ondevice", "site_label": "허리"}
    ).json()["state"]
    # 물은 축(onset)이 아닌 severity를 끼우고, 근거를 변조한 갱신
    ext = TurnExtraction(
        updates=[
            filled(Axis.ONSET, "3일", "3일 전부터요"),
            filled(Axis.SEVERITY, "7", "7점"),  # 발화에 없음
            filled(Axis.CHARACTER, "치릿", "치릿해요"),  # 근거 변조
        ]
    ).model_dump()
    t = client.post(
        "/v1/previsit/turns", json={"state": state, "utterance": "3일 전부터요", "extraction": ext}
    ).json()
    assert t["card"]["axes"]["onset"]["status"] == "filled"
    assert t["card"]["axes"]["severity"]["status"] == "not_asked"
    assert t["card"]["axes"]["character"]["status"] == "not_asked"
    reasons = {d["axis"]: d["reason"] for d in t["audit"]["dropped"]}
    assert reasons["severity"] == "not_asked_axis"
    assert reasons["character"] == "not_asked_axis"  # 물은 축이 아니라 먼저 걸림
    assert len(t["audit"]["raw_extraction"]["updates"]) == 3  # 원본은 감사용으로 남는다


def test_extraction_requires_utterance_for_evidence_check():
    client, _ = make_client([])
    state = client.post("/v1/previsit/sessions").json()["state"]
    ext = TurnExtraction(updates=[filled(Axis.ONSET, "3일", "3일")]).model_dump()
    r = client.post("/v1/previsit/turns", json={"state": state, "extraction": ext})
    assert r.status_code == 422


def test_selections_fill_axes_without_llm_and_advance():
    client, made = make_client([])
    state = client.post(
        "/v1/previsit/sessions", json={"profile": "ondevice", "site_label": "허리"}
    ).json()["state"]
    # 첫 질문 ONSET에 칩으로 답
    t = client.post(
        "/v1/previsit/turns",
        json={"state": state, "selections": [{"axis": "onset", "value": "3일 이내"}]},
    ).json()
    assert all(ex.calls == [] for ex in made)
    assert t["card"]["axes"]["onset"]["value"] == "3일 이내"
    assert t["card"]["axes"]["onset"]["evidence"] == ["[선택] 3일 이내"]
    assert t["reply"] == QUESTIONS[Axis.CHARACTER]  # 다음 질문으로 넘어간다
    assert t["audit"]["source"] == "none"
    # 심각도 폼 + 직접 입력을 같은 턴에
    client2, made2 = make_client(
        [TurnExtraction(updates=[filled(Axis.CHARACTER, "욱신", "욱신거려요")])]
    )
    st = client2.post("/v1/previsit/sessions", json={"site_label": "허리"}).json()["state"]
    t2 = client2.post(
        "/v1/previsit/turns",
        json={
            "state": st,
            "utterance": "욱신거려요",
            "selections": [{"axis": "severity", "value": "5"}],
        },
    ).json()
    assert t2["card"]["axes"]["severity"]["value"] == "5"
    assert t2["card"]["axes"]["character"]["value"] == "욱신"


def test_ontology_search_endpoint_maps_alias_to_zone():
    client = TestClient(create_app())
    b = client.get("/v1/ontology/search", params={"q": "복부"}).json()
    assert b["results"][0]["id"] == "ANC:004" and b["results"][0]["matched"] == "복부"
    b = client.get("/v1/ontology/search", params={"q": "옆구리"}).json()
    r = b["results"][0]
    assert r["id"] == "SUR:042" and r["kind"] == "surface" and r["anchor_id"] == "ANC:012"
    assert client.get("/v1/ontology/search", params={"q": "무릅"}).json()["results"] == []
    # body-map에도 aliases가 실려 앱이 로컬 매칭할 수 있다
    bm = client.get("/v1/ontology/body-map").json()
    abdomen = next(a for a in bm["anchors"] if a["id"] == "ANC:004")
    assert "복부" in abdomen["aliases"]


def test_body_map_image_keys_are_stable_slugs_and_unique():
    client = TestClient(create_app())
    b = client.get("/v1/ontology/body-map").json()
    keys = [a["image_key"] for a in b["anchors"]] + [
        z["image_key"] for a in b["anchors"] for z in a["zones"]
    ]
    assert all(re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", k) for k in keys), keys
    assert len(keys) == len(set(keys)), "image_key 중복"
    head = next(a for a in b["anchors"] if a["id"] == "ANC:001")
    assert head["image_key"] == "head"
    assert b["image_key_note"]


def test_body_map_lists_anchors_zones_and_departments():
    client, _ = make_client([])
    b = client.get("/v1/ontology/body-map").json()
    assert b["ontology_snapshot"]
    ids = {a["id"] for a in b["anchors"]}
    assert "ANC:012" in ids  # 허리·엉덩이
    back = next(a for a in b["anchors"] if a["id"] == "ANC:012")
    assert back["view"] == "back" and back["departments"] == ["정형외과", "신경외과"]
    zone_ids = {z["id"] for z in back["zones"]}
    assert {"SUR:041", "SUR:042"} <= zone_ids
    side = next(z for z in back["zones"] if z["id"] == "SUR:042")
    assert side["laterality"] == "left_right" and "비뇨의학과" in side["departments"]


def test_hmac_rejects_unsigned_and_accepts_signed_requests(monkeypatch):
    cfg = auth.HmacConfig(secret="s3cret")
    app = create_app(lambda: ScriptedExtractor([]))
    auth.install(app, cfg)
    client = TestClient(app)
    assert client.get("/health").status_code == 200  # 예외 경로
    r = client.post("/v1/previsit/sessions", json={})
    assert r.status_code == 401
    body = json.dumps({}).encode()
    ts = str(int(time.time()))
    sig = auth.sign("s3cret", "POST", "/v1/previsit/sessions", ts, "rid-1", body)
    r = client.post(
        "/v1/previsit/sessions",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-Signature": sig,
            "X-Timestamp": ts,
            "X-Request-Id": "rid-1",
        },
    )
    assert r.status_code == 200
    # 오래된 타임스탬프는 거부
    old = str(int(time.time()) - 3600)
    r = client.post(
        "/v1/previsit/sessions",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-Signature": auth.sign("s3cret", "POST", "/v1/previsit/sessions", old, "rid-1", body),
            "X-Timestamp": old,
            "X-Request-Id": "rid-1",
        },
    )
    assert r.status_code == 401


def test_hmac_signature_covers_method_and_path(monkeypatch):
    """2026-09-11: 서명에 메서드·경로가 없어서, 본문이 같으면 다른 엔드포인트로 재전송할 수 있었다.

    같은 사설망이라 악용 경로는 좁았지만 구성은 바뀐다. 백엔드 합의로 넣었다(#7).
    """
    cfg = auth.HmacConfig(secret="s3cret")
    app = create_app(lambda: ScriptedExtractor([]))
    auth.install(app, cfg)
    client = TestClient(app)
    body = json.dumps({}).encode()
    ts = str(int(time.time()))

    def post(path, sig):
        return client.post(
            path,
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Signature": sig,
                "X-Timestamp": ts,
                "X-Request-Id": "rid-1",
            },
        )

    right = auth.sign("s3cret", "POST", "/v1/previsit/sessions", ts, "rid-1", body)
    assert post("/v1/previsit/sessions", right).status_code == 200
    # 다른 경로로 서명한 것은 거부된다
    wrong_path = auth.sign("s3cret", "POST", "/v1/previsit/turns", ts, "rid-1", body)
    assert post("/v1/previsit/sessions", wrong_path).status_code == 401
    # 메서드가 다르면 거부된다
    wrong_method = auth.sign("s3cret", "GET", "/v1/previsit/sessions", ts, "rid-1", body)
    assert post("/v1/previsit/sessions", wrong_method).status_code == 401


def test_hmac_no_longer_exempts_get(monkeypatch):
    """메서드 단위 GET 면제를 없앴다(2026-09-11).

    이전에는 `request.method == "GET"` 조건 때문에 경로 화이트리스트가 무의미했고,
    그 사이 `GET /v1/ontology/search`가 생겼다. 조회 엔드포인트가 조용히 무인증이 되는 것을 막는다.
    """
    cfg = auth.HmacConfig(secret="s3cret")
    app = create_app(lambda: ScriptedExtractor([]))
    auth.install(app, cfg)
    client = TestClient(app)

    # 서명 없는 GET은 이제 거부된다
    assert client.get("/v1/ontology/body-map").status_code == 401
    assert client.get("/v1/ontology/search", params={"q": "무릎"}).status_code == 401

    # 서명하면 통과한다. GET은 본문이 비어 있다
    ts = str(int(time.time()))
    sig = auth.sign("s3cret", "GET", "/v1/ontology/body-map", ts, "rid-g", b"")
    r = client.get(
        "/v1/ontology/body-map",
        headers={"X-Signature": sig, "X-Timestamp": ts, "X-Request-Id": "rid-g"},
    )
    assert r.status_code == 200

    # 경로 화이트리스트는 그대로 면제
    for path in ("/health", "/openapi.json"):
        assert client.get(path).status_code == 200


def test_hmac_request_id_header_may_be_absent(monkeypatch):
    """백엔드는 항상 보내기로 했지만, 빠뜨렸을 때 원인을 찾기 쉽게 빈 문자열로 계산한다"""
    cfg = auth.HmacConfig(secret="s3cret")
    app = create_app(lambda: ScriptedExtractor([]))
    auth.install(app, cfg)
    client = TestClient(app)
    body = json.dumps({}).encode()
    ts = str(int(time.time()))
    sig = auth.sign("s3cret", "POST", "/v1/previsit/sessions", ts, "", body)
    r = client.post(
        "/v1/previsit/sessions",
        content=body,
        headers={"Content-Type": "application/json", "X-Signature": sig, "X-Timestamp": ts},
    )
    assert r.status_code == 200
