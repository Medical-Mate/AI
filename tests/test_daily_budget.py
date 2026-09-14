"""공개 데모는 4주를 떠 있다. 요청 단위 상한은 요청마다 새로 시작해서 누적을 못 막는다.

여기서 제일 중요한 테스트는 **막히지 않는 쪽**이다 — 예산이 죽어도 LLM을 안 쓰는 문진은
이어져야 한다. 예산 때문에 환자가 답하던 문답이 끊기면 우리가 막으려던 것보다 나쁘다.
"""

import json
import logging

import pytest
from fastapi.testclient import TestClient

from medimate.api.app import create_app
from medimate.api.budget import (
    ENV_CAP,
    ENV_STATE_FILE,
    BudgetGuarded,
    DailyBudget,
    DailyBudgetExceeded,
)
from medimate.llm import AxisUpdate, TurnExtraction
from medimate.schema import Axis, FieldStatus
from tests.fakes import ScriptedExtractor


class Fake(ScriptedExtractor):
    """호출마다 사용량이 쌓이는 대역. 가격표에 있는 모델 ID를 쓴다"""

    model_id = "gpt-5.6-terra"

    def __init__(self):
        super().__init__(
            [
                TurnExtraction(
                    chief_complaint="무릎",
                    updates=[
                        AxisUpdate(
                            axis=Axis.SITE,
                            status=FieldStatus.FILLED,
                            value="무릎",
                            evidence="무릎이 아파요",
                        )
                    ],
                )
            ]
            * 6
        )
        self.calls = 0

    def extract(self, *a, **kw):
        self.calls += 1
        return super().extract(*a, **kw)


def _client(cap: str | None, monkeypatch, state_file=None):
    if cap is None:
        monkeypatch.delenv(ENV_CAP, raising=False)
    else:
        monkeypatch.setenv(ENV_CAP, cap)
    if state_file is None:
        monkeypatch.delenv(ENV_STATE_FILE, raising=False)
    else:
        monkeypatch.setenv(ENV_STATE_FILE, str(state_file))
    ex = Fake()
    return TestClient(create_app(lambda: ex)), ex


def _start(c):
    return c.post("/v1/previsit/sessions").json()["state"]


# --- 켜고 끄기 -------------------------------------------------------------


def test_off_by_default(monkeypatch):
    """변수가 없으면 아무 일도 하지 않는다 — 로컬·테스트·eval이 이 파일 때문에 달라지면 안 된다"""
    c, _ = _client(None, monkeypatch)
    assert c.get("/health").json()["llm_budget"] is None


def test_health_shows_the_cap_and_what_was_spent(monkeypatch):
    """`hmac_required`와 같은 이유 — "켠 줄 알았는데 안 켜진" 상태를 없앤다"""
    c, _ = _client("5.0", monkeypatch)
    b = c.get("/health").json()["llm_budget"]
    assert b["cap_usd"] == 5.0 and b["spent_today_usd"] == 0.0
    assert b["resets_at"].endswith("T00:00:00+09:00")  # 기본은 KST 자정
    assert b["persisted"] is False  # 파일 경로를 주지 않으면 지금처럼 메모리


def test_state_file_survives_a_new_app_process(tmp_path, monkeypatch):
    """배포가 새 프로세스를 띄워도 같은 날짜의 누적액을 다시 읽는다."""
    state_file = tmp_path / "budget.json"
    c, _ = _client("5.0", monkeypatch, state_file)
    c.app.state.daily_budget.record(0.75)
    assert c.get("/health").json()["llm_budget"]["persisted"] is True

    restarted, _ = _client("5.0", monkeypatch, state_file)
    restored = restarted.get("/health").json()["llm_budget"]
    assert restored["spent_today_usd"] == pytest.approx(0.75)
    assert restored["persisted"] is True


def test_state_write_failure_does_not_stop_the_questionnaire(tmp_path, monkeypatch):
    """볼륨 권한이 깨져도 메모리 상한은 남고 문진 요청은 파일 I/O 때문에 죽지 않는다."""
    state_file = tmp_path / "a-directory-not-a-file"
    state_file.mkdir()
    c, _ = _client("5.0", monkeypatch, state_file)

    c.app.state.daily_budget.record(0.25)  # 예외가 밖으로 나오지 않는다
    status = c.get("/health").json()["llm_budget"]
    assert status["spent_today_usd"] == pytest.approx(0.25)
    assert status["persisted"] is False


def test_a_broken_volume_warns_once_not_every_call(tmp_path, monkeypatch, caplog):
    """볼륨이 안 잡힌 채 배포되면 쓰기는 **LLM 호출마다** 일어난다.

    매번 경고를 내면 심사 4주짜리 공개 데모의 로그가 그 한 줄로 덮여 정작 봐야 할 것이 묻힌다.
    그렇다고 지우면 조용한 실패가 된다 — **첫 실패는 남기고 되풀이하지 않는다.**
    지금 파일이 살아 있는지는 `/health`의 `persisted`가 들고 있다.
    """
    state_file = tmp_path / "a-directory-not-a-file"
    state_file.mkdir()
    c, _ = _client("5.0", monkeypatch, state_file)

    with caplog.at_level(logging.WARNING, logger="medimate.api.budget"):
        for _ in range(5):
            c.app.state.daily_budget.record(0.01)

    failed = [
        r
        for r in caplog.records
        if r.name == "medimate.api.budget" and "쓰지 못해" in r.getMessage()
    ]
    assert len(failed) == 1, f"호출 5회에 경고 {len(failed)}줄 — 한 줄이어야 한다"
    assert c.get("/health").json()["llm_budget"]["persisted"] is False


def test_recovery_is_worth_one_line(tmp_path, monkeypatch, caplog):
    """실패하다 다시 쓸 수 있게 되면 그때 한 줄. 운영자가 볼륨을 고친 것을 로그로 확인한다."""
    good = tmp_path / "state.json"
    c, _ = _client("5.0", monkeypatch, good)
    budget = c.app.state.daily_budget

    def budget_logs() -> list[str]:
        # 예산 로거만 본다 — HMAC 기동 경고 같은 다른 로거가 섞이면 이 테스트가 흔들린다
        return [r.getMessage() for r in caplog.records if r.name == "medimate.api.budget"]

    with caplog.at_level(logging.WARNING, logger="medimate.api.budget"):
        budget.record(0.01)  # 정상 — 조용해야 한다
        assert not budget_logs()

        budget.state_file = tmp_path / "gone" / "state.json"  # 볼륨이 빠졌다
        for _ in range(3):
            budget.record(0.01)

        budget.state_file = good  # 운영자가 고쳤다
        for _ in range(3):
            budget.record(0.01)

    messages = budget_logs()
    assert sum("쓰지 못해" in m for m in messages) == 1
    assert sum("다시 쓸 수 있습니다" in m for m in messages) == 1
    assert budget.status()["persisted"] is True


def test_the_day_boundary_is_seoul_midnight_by_default(monkeypatch):
    """**UTC 자정은 KST 오전 9시다.** 데모 날 오후에 상한이 차면 다음 날 아침까지 안 풀린다.

    상한은 새는 것을 막는 장치이지 하루를 날리는 장치가 아니다.
    """
    monkeypatch.delenv("MEDIMATE_BUDGET_RESET_TZ", raising=False)
    c, _ = _client("5.0", monkeypatch)
    assert c.get("/health").json()["llm_budget"]["resets_at"].endswith("T00:00:00+09:00")


def test_the_timezone_can_be_changed(monkeypatch):
    monkeypatch.setenv("MEDIMATE_BUDGET_RESET_TZ", "UTC")
    c, _ = _client("5.0", monkeypatch)
    assert c.get("/health").json()["llm_budget"]["resets_at"].endswith("T00:00:00+00:00")


def test_a_bad_timezone_is_not_silently_utc(monkeypatch):
    """오타로 조용히 UTC가 되면 리셋 시각이 9시간 어긋난다"""
    monkeypatch.setenv("MEDIMATE_BUDGET_RESET_TZ", "Asia/Seoul_오타")
    monkeypatch.setenv(ENV_CAP, "5.0")
    with pytest.raises(ValueError, match="MEDIMATE_BUDGET_RESET_TZ"):
        DailyBudget.from_env().status()


def test_a_bad_value_is_not_silently_off(monkeypatch):
    """오타로 조용히 꺼지면 켠 줄 알고 4주를 떠 있게 된다"""
    for bad in ("오점영", "0", "-1"):
        monkeypatch.setenv(ENV_CAP, bad)
        with pytest.raises(ValueError, match=ENV_CAP):
            DailyBudget.from_env()


# --- 막는 쪽 ---------------------------------------------------------------


def test_turn_is_503_once_the_cap_is_reached(monkeypatch):
    c, ex = _client("1.0", monkeypatch)
    st = _start(c)
    c.app.state.daily_budget.record(1.0)  # 이미 다 썼다고 두고
    r = c.post("/v1/previsit/turns", json={"state": st, "utterance": "무릎이 아파요"})
    assert r.status_code == 503
    assert "일일 LLM 예산 소진" in r.json()["detail"]
    assert "초기화" in r.json()["detail"]
    assert ex.calls == 0  # **호출이 나가지 않았다.** 막는 게 목적이지 세는 게 목적이 아니다


def test_spending_accumulates_across_requests(monkeypatch):
    """요청 단위 상한이 못 하는 일이 이것이다 — 인스턴스가 매번 새로 생겨도 누적된다"""
    budget = DailyBudget(cap_usd=1.0)
    for _ in range(3):
        budget.record(0.3)
    assert budget.status()["spent_today_usd"] == pytest.approx(0.9)
    budget.check()  # 아직 아래
    budget.record(0.2)
    with pytest.raises(DailyBudgetExceeded):
        budget.check()


def test_cost_is_recorded_even_when_the_call_raises():
    """파싱에 실패한 호출도 요금은 나간다. 안 세면 실패가 반복될수록 새기만 한다"""

    class Boom:
        model_id = "gpt-5.6-terra"

        def __init__(self):
            from medimate.llm.providers import Usage

            self.usage = Usage()

        def extract(self, *a, **kw):
            self.usage.input_tokens += 1_000_000  # $2.00
            raise ValueError("모델 출력 파싱 실패")

    budget = DailyBudget(cap_usd=10.0)
    with pytest.raises(ValueError):
        BudgetGuarded(Boom(), budget).extract("x", None)
    assert budget.status()["spent_today_usd"] == pytest.approx(2.0)


# --- 막지 않는 쪽 (이게 더 중요하다) ----------------------------------------


def test_a_turn_without_an_llm_call_still_works_after_the_cap(monkeypatch):
    """선택지만 온 턴은 LLM을 안 쓴다. 예산이 죽어도 문진은 이어져야 한다"""
    c, ex = _client("1.0", monkeypatch)
    st = _start(c)
    c.app.state.daily_budget.record(1.0)
    r = c.post(
        "/v1/previsit/turns",
        json={"state": st, "selections": [{"axis": "severity", "value": "3"}]},
    )
    assert r.status_code == 200
    assert r.json()["card"]["axes"]["severity"]["value"] == "3"
    assert ex.calls == 0


def test_device_extraction_still_works_after_the_cap(monkeypatch):
    """폰이 뽑아 온 턴도 서버 LLM을 안 쓴다"""
    c, ex = _client("1.0", monkeypatch)
    st = _start(c)
    c.app.state.daily_budget.record(1.0)
    ext = TurnExtraction(
        updates=[
            AxisUpdate(
                axis=Axis.ONSET, status=FieldStatus.FILLED, value="3일 전", evidence="3일 전부터요"
            )
        ]
    ).model_dump()
    r = c.post(
        "/v1/previsit/turns",
        json={"state": st, "utterance": "3일 전부터요", "extraction": ext},
    )
    assert r.status_code == 200
    assert r.json()["card"]["axes"]["onset"]["value"] == "3일 전"
    assert ex.calls == 0


def test_reads_still_work_after_the_cap(monkeypatch):
    """온톨로지 조회·상태 확인은 LLM과 무관하다"""
    c, _ = _client("1.0", monkeypatch)
    c.app.state.daily_budget.record(1.0)
    assert c.get("/health").status_code == 200
    assert c.get("/v1/ontology/body-map").status_code == 200
    assert c.get("/v1/ontology/search?q=무릎").status_code == 200


def test_a_new_day_starts_over():
    """UTC 자정 기준. 날이 바뀌면 0부터"""
    from datetime import date

    budget = DailyBudget(cap_usd=1.0)
    budget.record(1.0)
    with pytest.raises(DailyBudgetExceeded):
        budget.check()
    budget._day = date(2000, 1, 1)  # 어제 쓴 것으로 만든다
    budget.check()  # 통과해야 한다
    assert budget.status()["spent_today_usd"] == 0.0


def test_memo_json_is_unchanged_by_the_guard(monkeypatch):
    """감싸기가 응답 모양을 바꾸지 않는다"""
    c, _ = _client("5.0", monkeypatch)
    r = c.post("/v1/postvisit/memo", json={"memo": "위염 초기라고 하셨어요.", "classify": False})
    assert r.status_code == 200
    assert set(json.loads(r.text)) >= {"card", "sentences", "labels", "split_version"}
