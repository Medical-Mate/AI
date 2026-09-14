"""웹 데모 폴백 fixture가 **실제 응답과 갈리지 않게** 잡는다.

프론트가 이 JSON으로 화면을 만든다. 필드명이 한 글자라도 다르면 개발은 되는데 라이브에서
깨지고, 그때는 이미 심사 기간이다. `body-map.json` 드리프트 테스트와 같은 자리다.
"""

import json
import pathlib
import subprocess
import sys

FIXTURE = pathlib.Path("docs/examples/demo-session.json")


def _load() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_fixture_matches_the_live_api():
    """생성기를 다시 돌려 파일과 비교한다. LLM 호출 0"""
    r = subprocess.run(
        [sys.executable, "scripts/gen_demo_session.py", "--check"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert r.returncode == 0, r.stdout + r.stderr


def test_it_is_a_finished_session():
    """중간에 끊긴 세션을 폴백으로 두면 데모가 미완성으로 보인다"""
    d = _load()
    turns = [s for s in d["steps"] if s["step"].startswith("turn_")]
    assert turns, "문진 턴이 없다"
    last = turns[-1]["response"]
    assert last["ended"] is True and last["end_reason"] == "complete"
    assert last["audit"] is not None  # 끝난 뒤 보낸 빈 턴이 아니다


def test_every_screen_the_demo_needs_is_covered():
    d = _load()
    steps = [s["step"] for s in d["steps"]]
    assert steps[0] == "session_start" and steps[-1] == "postvisit_memo"
    card = d["steps"][-2]["response"]["card"]
    assert card["title"], "제목 줄"
    assert card["department_guidance"], "진료과 안내"
    assert len(card["question_candidates"]) == 3, "4단계 물어볼 것"
    memo = d["steps"][-1]["response"]
    assert memo["sentences"] and memo["card"]["follow_up_date"], "메모 분류·재방문 날짜"


def test_the_body_map_path_survives():
    """부위를 발화로 덮으면 `source`가 `selection`이 아니게 되고 인체도 경로가 사라진다"""
    card = _load()["steps"][-2]["response"]["card"]
    assert card["axes"]["site"]["source"] == "selection"
    assert card["question_candidates"][0]["rank"] == 1


def test_health_info_is_sent_on_the_last_turn_only():
    """매 턴 붙이면 건강정보가 턴 수만큼 오가는 녹화가 되고, 프론트가 그대로 구현한다"""
    turns = [s for s in _load()["steps"] if s["step"].startswith("turn_")]
    with_profile = [s["step"] for s in turns if "patient_profile" in s["request"]]
    assert with_profile == [turns[-1]["step"]], with_profile


def test_the_slider_value_is_recorded_as_a_selection():
    """통증 강도는 **묻지 않고 화면에서 고른다**(1d 슬라이더).

    `ASK_ORDER`에 `severity`가 없어서 문답으로는 절대 안 채워진다 — NRS를 말로 물으면 환자가
    숫자를 지어내고 그 값은 근거가 없다. 녹화에 이 자리가 없으면 프론트가 **슬라이더 값을
    어디에 실어야 하는지 배울 데가 없고**, 데모 카드의 심각도가 빈다.
    """
    d = _load()
    turns = [s for s in d["steps"] if s["step"].startswith("turn_")]
    with_sel = [s for s in turns if s["request"].get("selections")]
    assert len(with_sel) == 1 and with_sel[0]["step"] == turns[-1]["step"]

    sev = d["steps"][-2]["response"]["card"]["axes"]["severity"]
    assert sev["status"] == "filled" and sev["value"]
    assert sev["source"] == "selection"  # 발화가 아니다
    assert sev["evidence"] == ["[선택] 3 (꽤 아파요)"]


def test_the_card_is_complete():
    """8축이 다 차야 데모 카드가 제 모습으로 보인다"""
    axes = _load()["steps"][-2]["response"]["card"]["axes"]
    empty = [k for k, v in axes.items() if not v["value"]]
    assert empty == [], empty


def test_no_identifiers():
    blob = json.dumps(_load(), ensure_ascii=False)
    for banned in ("생년", "나이", "성별", "주민", "전화", "이메일", "@"):
        assert banned not in blob, banned
