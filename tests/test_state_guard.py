"""상태 왕복에서 **엔진이 만들 수 없는 모양**만 끊는다.

AI 서버는 무상태라 상태를 백엔드가 들고 다닌다. 재조립·분해 저장·요청 섞임으로 변형된
상태는 **스키마를 통과한다** — 타입이 맞기 때문이다. 그래서 400이 따로 필요하다.

이 파일에서 더 중요한 것은 **막지 않는 쪽**이다. 가드가 넓으면 문답 중인 환자의 요청이
우리 추측으로 끊긴다. 정상 조합을 하나씩 못박아 둔다.
"""

from fastapi.testclient import TestClient

from medimate.api.app import create_app
from medimate.api.state_guard import check_state
from medimate.dialog.questions import SITE_PRESELECTED
from medimate.dialog.state import SessionState
from medimate.schema.card import AxisEntry, FieldStatus, InterviewCard, Provenance


def _state(**card_kw) -> SessionState:
    return SessionState(card=InterviewCard(**card_kw))


# --- 막는 것 ---------------------------------------------------------------


def test_rejects_value_on_an_unasked_axis():
    """엔진은 값을 넣을 때 status를 함께 옮긴다. 이 조합은 엔진 밖에서 들어온 값이다"""
    st = _state(axes={"onset": AxisEntry(status=FieldStatus.NOT_ASKED, value="3일 전")})
    assert check_state(st) == ["axes.onset: status가 not_asked인데 value가 있음"]


def _site_card(**kw):
    return dict(
        axes={
            "site": AxisEntry(
                status=FieldStatus.FILLED,
                value="오른쪽 무릎",
                evidence=[f"{SITE_PRESELECTED} 오른쪽 무릎"],
            )
        },
        **kw,
    )


_ONTOLOGY_PROV = Provenance(prompt_version="p", model_id="m", ontology_snapshot="2026-09-04")


def test_rejects_a_dropped_selection_record():
    """`site_selection`이 비면 제목·진료과 안내가 조용히 사라진다. 200에 카드는 멀쩡해 보인다"""
    st = _state(**_site_card(provenance=_ONTOLOGY_PROV))
    assert check_state(st) == ["site_selection 없음 — site 축은 인체도 선택으로 채워져 있음"]


def test_allows_a_label_started_session_with_no_record():
    """세션 시작에는 길이 둘이고(`site_node_id`·`site_label`) 표시는 같다.

    `site_selection`은 앞의 길에서만 생긴다 — 표시만 보고 막으면 라벨로 시작한 정상 세션이
    전부 400이 된다. 처음에 그렇게 짰다가 온디바이스 테스트 셋이 잡았다.
    """
    st = _state(**_site_card(provenance=Provenance(prompt_version="p", model_id="m")))
    assert check_state(st) == []


# --- 막지 않는 것 (이쪽이 더 중요하다) ---------------------------------------


def test_allows_ambiguous_with_a_value():
    """되물으면서 잠정 값을 들고 있는 정상 상태다 (llm/base.py)"""
    st = _state(axes={"onset": AxisEntry(status=FieldStatus.AMBIGUOUS, value="좀 됐어요")})
    assert check_state(st) == []


def test_allows_unknown_and_skipped_with_a_leftover_value():
    """앞 턴 값이 남아 있을 수 있다. 카드에 실리지 않으므로 끊을 이유가 없다"""
    for status in (FieldStatus.UNKNOWN, FieldStatus.SKIPPED):
        st = _state(axes={"severity": AxisEntry(status=status, value="5점")})
        assert check_state(st) == [], status


def test_allows_an_unasked_axis_with_no_value():
    """8축의 기본값이다. 여기서 걸리면 모든 첫 턴이 400이 된다"""
    st = _state(axes={a: AxisEntry() for a in ("site", "onset", "character")})
    assert check_state(st) == []


def test_allows_spoken_site_without_a_selection_record():
    """환자가 말로 답한 site다. 인체도를 안 썼으니 선택 기록이 없는 게 맞다"""
    st = _state(
        axes={
            "site": AxisEntry(
                status=FieldStatus.FILLED, value="오른쪽 무릎", evidence=["오른쪽 무릎이 아파요"]
            )
        }
    )
    assert check_state(st) == []


def test_real_session_start_state_round_trips():
    """가장 중요한 테스트 — 우리가 실제로 낸 상태가 우리 가드에 걸리면 안 된다"""
    app = create_app()
    client = TestClient(app)
    r = client.post("/v1/previsit/sessions", json={"site_node_id": "SUR:042", "side": "left"})
    assert r.status_code == 200, r.text
    state = SessionState.model_validate(r.json()["state"])
    # 인체도로 시작했으니 규칙 B가 걸릴 수 있는 상태다. 정상이므로 통과해야 한다
    assert SITE_PRESELECTED in str(state.card.axes["site"].evidence)
    assert check_state(state) == []


# --- API 경계 --------------------------------------------------------------


def test_turn_returns_400_not_422():
    """422는 스키마 위반. 이건 스키마가 맞고 내용이 불가능한 것이라 400이다"""
    app = create_app()
    client = TestClient(app)
    r = client.post("/v1/previsit/sessions", json={})
    state = r.json()["state"]
    state["card"]["axes"]["onset"] = {"status": "not_asked", "value": "3일 전", "evidence": []}
    r = client.post("/v1/previsit/turns", json={"state": state, "utterance": "무릎이 아파요"})
    assert r.status_code == 400
    assert "not_asked" in r.json()["detail"]
