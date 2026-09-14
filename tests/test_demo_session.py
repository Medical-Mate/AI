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


def test_no_identifiers():
    blob = json.dumps(_load(), ensure_ascii=False)
    for banned in ("생년", "나이", "성별", "주민", "전화", "이메일", "@"):
        assert banned not in blob, banned
