"""무상태 API — 상태를 왕복시키며 문진이 이어지는지. LLM 호출 없음(ScriptedExtractor)."""

from dataclasses import dataclass

from fastapi.testclient import TestClient

from medimate.api.app import create_app
from medimate.dialog.questions import CLOSING, OPENING, QUESTIONS
from medimate.llm import AxisUpdate, TurnExtraction
from medimate.schema import Axis, FieldStatus
from tests.fakes import ScriptedExtractor


def filled(axis, value, evidence):
    return AxisUpdate(axis=axis, status=FieldStatus.FILLED, value=value, evidence=evidence)


@dataclass
class FakeUsage:
    input_tokens: int = 0
    output_tokens: int = 0


class SharedScriptExtractor(ScriptedExtractor):
    """요청마다 새 인스턴스지만 대본은 공유한다. extract가 불릴 때만 하나 소비.

    usage는 실제 어댑터처럼 호출 시점에 늘어난다(요청 단위로 0에서 시작).
    """

    def __init__(self, queue: list[TurnExtraction], per_call=(1000, 100)):
        super().__init__([])
        self._queue = queue
        self._per_call = per_call
        self.usage = FakeUsage()

    def extract(self, utterance, asked_axis, history=()):
        self._script = [self._queue.pop(0)] if self._queue else []
        self.usage.input_tokens += self._per_call[0]
        self.usage.output_tokens += self._per_call[1]
        return super().extract(utterance, asked_axis, history)


def make_client(script: list[TurnExtraction]) -> tuple[TestClient, list[ScriptedExtractor]]:
    """요청마다 새 Extractor를 만든다(서버와 같은 방식). 대본은 호출 순서대로 하나씩 소비."""
    made: list[ScriptedExtractor] = []
    queue = list(script)

    def factory():
        ex = SharedScriptExtractor(queue)
        made.append(ex)
        return ex

    return TestClient(create_app(factory)), made


def test_start_returns_opening_and_empty_state():
    client, made = make_client([])
    r = client.post("/v1/previsit/sessions")
    assert r.status_code == 200
    body = r.json()
    assert body["reply"] == OPENING
    assert body["state"]["turn"] == 0 and body["state"]["asked_axis"] is None
    assert body["state"]["card"]["provenance"]["model_id"] == "fake"
    assert all(ex.calls == [] for ex in made)  # 시작은 LLM을 부르지 않는다


def test_state_round_trip_continues_the_interview():
    client, made = make_client(
        [
            TurnExtraction(
                chief_complaint="오른쪽 무릎이 아프다",
                updates=[filled(Axis.SITE, "오른쪽 무릎", "오른쪽 무릎이")],
            ),
            TurnExtraction(updates=[filled(Axis.ONSET, "어제부터", "어제부터")]),
        ]
    )
    state = client.post("/v1/previsit/sessions").json()["state"]

    r1 = client.post(
        "/v1/previsit/turns", json={"state": state, "utterance": "오른쪽 무릎이 아파요"}
    )
    assert r1.status_code == 200
    b1 = r1.json()
    assert b1["reply"] == QUESTIONS[Axis.ONSET]
    assert b1["state"]["asked_axis"] == "onset" and b1["state"]["turn"] == 1
    assert b1["card"]["axes"]["site"]["value"] == "오른쪽 무릎"
    assert b1["audit"]["turn"] == 1 and b1["audit"]["asked_axis"] is None

    # 두 번째 요청은 다른 프로세스라고 가정 — 받은 state만 넘긴다
    r2 = client.post("/v1/previsit/turns", json={"state": b1["state"], "utterance": "어제부터요"})
    b2 = r2.json()
    assert b2["reply"] == QUESTIONS[Axis.CHARACTER]
    assert b2["card"]["axes"]["onset"]["value"] == "어제부터"
    assert b2["card"]["axes"]["site"]["value"] == "오른쪽 무릎"  # 이전 턴 값이 유지된다
    # 두 번째 Extractor는 직전 (질문, 답) 이력을 받았다
    assert made[-1].histories[0] == [(OPENING, "오른쪽 무릎이 아파요")]
    assert made[-1].calls[0][1] == Axis.ONSET  # 1턴에 SITE가 채워져 ONSET을 물었다


def test_stop_ends_and_card_is_returned_as_is():
    client, _ = make_client(
        [
            TurnExtraction(
                chief_complaint="어깨", updates=[filled(Axis.SITE, "왼쪽 어깨", "어깨")]
            ),
            TurnExtraction(wants_to_stop=True),
        ]
    )
    state = client.post("/v1/previsit/sessions").json()["state"]
    state = client.post("/v1/previsit/turns", json={"state": state, "utterance": "어깨"}).json()[
        "state"
    ]
    b = client.post("/v1/previsit/turns", json={"state": state, "utterance": "이만 할래요"}).json()
    assert b["ended"] and b["end_reason"] == "stop" and b["reply"] == CLOSING
    assert b["card"]["axes"]["site"]["value"] == "왼쪽 어깨"
    assert b["card"]["completeness"] < 1


def test_empty_utterance_has_no_audit_and_no_llm_call():
    client, made = make_client([TurnExtraction()])
    state = client.post("/v1/previsit/sessions").json()["state"]
    b = client.post("/v1/previsit/turns", json={"state": state, "utterance": "   "}).json()
    assert b["audit"] is None and b["state"]["turn"] == 0
    assert all(ex.calls == [] for ex in made)


def test_session_tokens_accumulate_across_requests():
    client, _ = make_client([TurnExtraction(), TurnExtraction()])
    state = client.post("/v1/previsit/sessions").json()["state"]
    assert state["session_tokens"] == 0
    b1 = client.post("/v1/previsit/turns", json={"state": state, "utterance": "무릎"}).json()
    b2 = client.post("/v1/previsit/turns", json={"state": b1["state"], "utterance": "어제"}).json()
    assert b1["state"]["session_tokens"] == 1100
    assert b2["state"]["session_tokens"] == 2200
    assert b1["audit"]["usage"]["input_tokens"] == 1000


def test_tampered_state_with_diagnosis_field_is_rejected():
    client, _ = make_client([])
    state = client.post("/v1/previsit/sessions").json()["state"]
    r = client.post(
        "/v1/previsit/turns", json={"state": state, "utterance": "x", "diagnosis": "ACL"}
    )
    assert r.status_code == 422


def test_extractor_parse_failure_is_502_and_state_untouched():
    class Broken:
        model_id = "fake"
        prompt_version = "test"

        def extract(self, *a, **k):
            raise ValueError("Expecting value: line 1")

    client = TestClient(create_app(lambda: Broken()))
    state = client.post("/v1/previsit/sessions").json()["state"]
    r = client.post("/v1/previsit/turns", json={"state": state, "utterance": "무릎"})
    assert r.status_code == 502


def test_health_reports_the_signing_spec_the_server_actually_verifies():
    """`/health`가 서버가 **실제로 검증하는** 계산식을 낸다. 손으로 관리하는 버전이 아니다.

    2026-09-11 사고: `main`의 계산식이 옛 형식인데 새 형식 벡터로 대조해 10 pass가 나왔고
    운영에서만 401이 났다. 검증 자료와 구현이 같이 움직이면 서로를 확인해 주지 못한다.
    점검 도구가 이 값을 자기 것과 문자열 비교해서 그 간격을 잡는다.
    """
    from medimate.api import auth

    client, _ = make_client([])
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["signing"] == {
        "template": auth.SIGNING_TEMPLATE,
        "algorithm": auth.SIGNING_ALGORITHM,
    }


def test_health_says_whether_hmac_is_enforced():
    """무검증으로 떠 있는 걸 아무도 모르는 상태를 없앤다.

    노출이 아니다 — 꺼져 있으면 서명 없이 아무 요청이나 통과하니 이미 알 수 있는 사실이고,
    켜져 있으면 "켜져 있다"는 정보에 값이 없다.
    """
    from medimate.api import auth

    client, _ = make_client([])
    assert client.get("/health").json()["hmac_enforced"] is False  # 테스트는 키 없음

    app = create_app(lambda: ScriptedExtractor([]))
    auth.install(app, auth.HmacConfig(secret="s3cret"))
    assert TestClient(app).get("/health").json()["hmac_enforced"] is True


def test_signing_template_is_what_sign_uses():
    """템플릿을 바꾸면 서명도 따라 바뀐다 — 둘이 갈릴 수 없게 묶여 있다"""
    from medimate.api import auth

    assert auth.SIGNING_TEMPLATE == "{method}.{path}.{timestamp}.{request_id}."
    head = auth.SIGNING_TEMPLATE.format(method="POST", path="/v1/x", timestamp="1", request_id="r")
    import hashlib
    import hmac as _hmac

    want = _hmac.new(b"k", head.encode() + b"{}", hashlib.sha256).hexdigest()
    assert auth.sign("k", "post", "/v1/x", "1", "r", b"{}") == want  # 메서드는 대문자로


def test_preselected_site_fills_axis_and_anchors_opening():
    client, made = make_client([TurnExtraction(chief_complaint="허리가 뻐근하다")])
    r = client.post("/v1/previsit/sessions", json={"site_label": "허리"})
    b = r.json()
    assert "허리" in b["reply"] and b["reply"] != OPENING
    site = b["state"]["card"]["axes"]["site"]
    assert site["status"] == "filled" and site["value"] == "허리"
    assert site["evidence"][0].startswith("[부위 선택]")  # 발화가 아님이 드러난다
    # 첫 턴 뒤 SITE를 다시 묻지 않고 ONSET으로 간다
    t = client.post(
        "/v1/previsit/turns", json={"state": b["state"], "utterance": "뻐근해요"}
    ).json()
    assert t["reply"] == QUESTIONS[Axis.ONSET]
    assert t["state"]["history"][0][0] == b["reply"]  # 이력의 첫 질문도 앵커된 오프닝


def test_site_node_selection_fills_site_and_department_guidance():
    client, _ = make_client([TurnExtraction(chief_complaint="허리가 뻐근하다")])
    r = client.post("/v1/previsit/sessions", json={"site_node_id": "SUR:042", "side": "left"})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["reply"].startswith("왼쪽 허리 옆")
    card = b["state"]["card"]
    assert card["axes"]["site"]["value"] == "왼쪽 허리 옆"
    assert card["site_selection"]["anchor_id"] == "ANC:012"
    assert card["provenance"]["ontology_snapshot"]  # 스냅샷이 박힌다
    t = client.post(
        "/v1/previsit/turns", json={"state": b["state"], "utterance": "뻐근해요"}
    ).json()
    dg = t["card"]["department_guidance"]
    assert dg["departments"] == ["내과", "비뇨의학과", "정형외과"]  # 구역 값 그대로, 순서 유지
    assert "자문 확인 전" in dg["source"]


def test_zone_without_departments_falls_back_to_anchor():
    client, _ = make_client([TurnExtraction()])
    b = client.post(
        "/v1/previsit/sessions", json={"site_node_id": "SUR:091", "side": "right"}
    ).json()
    assert b["state"]["card"]["axes"]["site"]["value"] == "오른쪽 무릎"
    t = client.post("/v1/previsit/turns", json={"state": b["state"], "utterance": "아파요"}).json()
    assert t["card"]["department_guidance"]["departments"] == ["정형외과"]  # 앵커 「다리」의 값


def test_side_on_non_lateral_zone_is_rejected():
    client, _ = make_client([])
    r = client.post("/v1/previsit/sessions", json={"site_node_id": "SUR:041", "side": "left"})
    assert r.status_code == 422  # 허리 가운데에는 좌우가 없다
    r = client.post("/v1/previsit/sessions", json={"site_node_id": "UBERON:0001465"})
    assert r.status_code == 422  # 구조 노드는 짚는 대상이 아니다


def test_axes_closed_ends_the_session_without_a_message_turn():
    """진료 전은 "전하고 싶은 말" 턴이 없다(2026-09-11). 4단계 화면이 대신한다.

    `patient_message`도 채울 경로가 없어져 export에서 뺐다(백엔드 요청 #7 — 항상 null이면
    앱이 헷갈린다).
    """
    updates = [AxisUpdate(axis=a, status=FieldStatus.FILLED, value="x", evidence="x") for a in Axis]
    client, _ = make_client([TurnExtraction(chief_complaint="c", updates=updates)])
    state = client.post("/v1/previsit/sessions").json()["state"]
    b = client.post("/v1/previsit/turns", json={"state": state, "utterance": "x 전부"}).json()
    assert b["ended"] and b["end_reason"] == "complete"
    assert "patient_message" not in b["card"]


def test_severity_arrives_by_selection_not_by_question():
    """3단계 슬라이더 값을 `selections`로 받는다. LLM을 부르지 않고 근거는 `[선택] …`"""
    client, _ = make_client([TurnExtraction(updates=[])])
    state = client.post("/v1/previsit/sessions").json()["state"]
    b = client.post(
        "/v1/previsit/turns",
        json={"state": state, "selections": [{"axis": "severity", "value": "4 (매우 심함)"}]},
    ).json()
    sev = b["card"]["axes"]["severity"]
    assert sev["status"] == "filled" and sev["value"] == "4 (매우 심함)"
    assert sev["evidence"] == ["[선택] 4 (매우 심함)"]  # evidence는 배열이다
    # LLM을 안 불렀지만 audit은 **있다**. source로 구분하고 usage가 0이다.
    # audit이 None인 것은 빈 입력·상한 종료처럼 턴 처리 자체가 없을 때뿐이다
    assert b["audit"]["source"] == "none"
    assert b["audit"]["usage"] == {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}


def test_ontology_search_endpoint_exposes_restoration_and_zone_expansion():
    client = TestClient(create_app())

    r = client.get("/v1/ontology/search", params={"q": "qo"})  # 한/영 오타
    assert r.status_code == 200
    assert r.json()["results"][0]["id"] == "ANC:004"

    r = client.get("/v1/ontology/search", params={"q": "다리"})
    labels = {x["label"]: x["score"] for x in r.json()["results"]}
    assert labels["다리"] == 3
    assert labels["무릎"] == 0  # 앵커에 딸려 온 구역. score > 0으로 거르면 사라진다

    r = client.get("/v1/ontology/search", params={"q": "전체"})
    assert all(x["score"] > 0 for x in r.json()["results"])  # 약한 매칭은 펼치지 않는다


def test_ontology_search_truncates_a_long_query_instead_of_failing():
    client = TestClient(create_app())
    r = client.get("/v1/ontology/search", params={"q": "배" + "가" * 5000})
    assert r.status_code == 200
    assert len(r.json()["query"]) == 300  # 발화 상한과 같은 값에서 자른다


def test_side_is_case_insensitive():
    """백엔드가 `LEFT`/`RIGHT`/`BOTH` 대문자로 구현했는데 우리는 소문자만 받아 422를 냈다.

    2026-09-11. 그쪽 테스트가 `side: null`이라 안 걸렸고, **좌우가 있는 부위(34곳 중 21곳)를
    처음 보내는 순간 터지는** 상태였다. 받는 쪽을 넓힌다 — 좁히면 상대가 고쳐야 하고,
    넓히면 기존 호출이 그대로 돈다.
    """
    client, _ = make_client([])
    for value, want in (
        ("left", "왼쪽 허리 옆"),
        ("LEFT", "왼쪽 허리 옆"),
        ("Left", "왼쪽 허리 옆"),
        ("RIGHT", "오른쪽 허리 옆"),
        ("BOTH", "양쪽 허리 옆"),
        ("  left  ", "왼쪽 허리 옆"),  # 공백도 접는다
    ):
        r = client.post("/v1/previsit/sessions", json={"site_node_id": "SUR:042", "side": value})
        assert r.status_code == 200, f"{value}: {r.text}"
        assert r.json()["state"]["card"]["axes"]["site"]["value"] == want


def test_side_is_stored_lowercase_whatever_came_in():
    """값을 만드는 곳은 여전히 하나라 카드에는 소문자로만 남는다"""
    client, _ = make_client([])
    r = client.post("/v1/previsit/sessions", json={"site_node_id": "SUR:042", "side": "LEFT"})
    assert r.json()["state"]["card"]["site_selection"]["side"] == "left"


def test_a_bogus_side_is_still_rejected():
    """넓히는 것이지 검증을 없애는 게 아니다"""
    client, _ = make_client([])
    for bad in ("sideways", "왼쪽", "l"):
        r = client.post("/v1/previsit/sessions", json={"site_node_id": "SUR:042", "side": bad})
        assert r.status_code == 422, bad


# ── 어떤 모델·프롬프트로 떠 있나 (2026-09-15) ──────────────────────────────────


def test_health_says_which_model_and_prompt_the_server_will_use(monkeypatch):
    """턴을 태우지 않고 확인하라고 낸다.

    프롬프트 계열은 모델 id로 고르는데 **정확히 일치**해야 한다. 어긋나면 조용히
    `extract-v3`로 떨어지고 88케이스 84 → 76, 인젝션 방어 3/3 → 0/3이 된다.
    200도 나오고 카드도 멀쩡해서 밖에서는 안 보였다.
    """
    monkeypatch.setenv("MEDIMATE_PROVIDER", "bedrock")
    monkeypatch.setenv("MEDIMATE_MODEL", "apac.amazon.nova-pro-v1:0")
    client, _ = make_client([])
    assert client.get("/health").json()["extractor"] == {
        "provider": "bedrock",
        "model_id": "apac.amazon.nova-pro-v1:0",
        "prompt_version": "extract-v5-nova",
        "pricing_known": True,
        "lifestyle_axis": False,  # 테스트 기본은 꺼짐
    }


def test_a_wrong_model_id_is_visible_instead_of_silent(monkeypatch):
    """`apac.`이 빠지면 Nova 전용 프롬프트가 아니다. 그 사실이 `/health`에 보여야 한다."""
    monkeypatch.setenv("MEDIMATE_MODEL", "amazon.nova-pro-v1:0")
    client, _ = make_client([])
    ex = client.get("/health").json()["extractor"]
    assert ex["prompt_version"] == "extract-v3"
    # 가격표에도 없다 = 지출 상한이 이 모델을 세지 못한다
    assert ex["pricing_known"] is False


def test_whitespace_in_the_model_env_is_removed_not_just_reported(monkeypatch):
    """공백 하나로 프롬프트가 바뀌는 것은 **감지**가 아니라 **제거**로 막는다.

    `/health`와 실제 호출이 같은 값을 보게 읽는 자리가 하나여야 한다.
    """
    from medimate.api.app import extractor_env

    monkeypatch.setenv("MEDIMATE_MODEL", "  apac.amazon.nova-pro-v1:0  ")
    client, _ = make_client([])
    assert client.get("/health").json()["extractor"]["prompt_version"] == "extract-v5-nova"
    assert extractor_env()[1] == "apac.amazon.nova-pro-v1:0"

    monkeypatch.setenv("MEDIMATE_MODEL", "   ")  # 빈 값은 기본값으로 — 기본값이 곧 운영값(Nova)
    assert extractor_env() == ("bedrock", "apac.amazon.nova-pro-v1:0")


# ── 앱이 노드 id를 안 보내고 이름만 보낼 때 (2026-09-15, 백엔드 #7) ───────────────


def test_a_site_name_alone_still_fills_title_and_department():
    """앱이 `siteText`(화면 이름)만 보내고 노드 id를 안 보내고 있었다.

    이름만 쓰면 되묻기는 돌지만 `title`·`department_guidance`가 빈다 — 둘 다 노드에서
    나온다. 이름으로 노드를 풀어 그 둘을 채운다.
    """
    client, _ = make_client([])
    b = client.post("/v1/previsit/sessions", json={"site_label": "팔"}).json()
    card = b["state"]["card"]
    assert card["site_selection"]["node_id"] == "ANC:013"
    assert card["site_selection"]["departments"] == ["정형외과"]


def test_the_name_the_caller_sent_is_the_name_we_show():
    """노드를 푸는 것은 재료를 얻으려는 것이지 이름을 고치려는 것이 아니다.

    `"허리"`로 보냈는데 `"허리 가운데"`로 되돌려주면 앱 화면에 없던 말이 환자에게 보인다.
    """
    client, _ = make_client([])
    b = client.post("/v1/previsit/sessions", json={"site_label": "허리"}).json()
    assert "허리 가운데" not in b["reply"]
    assert b["state"]["card"]["axes"]["site"]["value"] == "허리"
    assert b["state"]["card"]["site_selection"]["label"] == "허리"


def test_an_unknown_site_name_is_kept_not_rejected():
    """온톨로지에 없는 이름이어도 422로 끊지 않는다.

    환자가 짚은 것은 맞고 우리가 그 이름을 모르는 것뿐이다. 이름만 쓰고 넘어간다.
    """
    client, _ = make_client([])
    r = client.post("/v1/previsit/sessions", json={"site_label": "거시기"})
    assert r.status_code == 200
    card = r.json()["state"]["card"]
    assert card["axes"]["site"]["value"] == "거시기"
    assert card["site_selection"] is None


def test_the_default_is_nova_so_a_missing_env_cannot_fall_back_to_another_vendor(monkeypatch):
    """env가 통째로 빠져도 Terra·OpenAI로 떨어지지 않는다.

    2026-09-15: 로컬 점검 스크립트가 주입에 실패한 채 기본값으로 돌아 **OpenAI 카드로 돈이
    나갔다.** 기본값이 운영값과 다르면 그 차이만큼 조용히 새는 자리가 생긴다.
    """
    monkeypatch.delenv("MEDIMATE_PROVIDER", raising=False)
    monkeypatch.delenv("MEDIMATE_MODEL", raising=False)
    client, _ = make_client([])
    ex = client.get("/health").json()["extractor"]
    assert (ex["provider"], ex["model_id"]) == ("bedrock", "apac.amazon.nova-pro-v1:0")
    assert ex["prompt_version"] == "extract-v5-nova"
