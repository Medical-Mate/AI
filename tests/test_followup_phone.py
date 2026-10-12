"""폰 모델이 읽은 재방문 표현으로 날짜를 세는 경로(2026-10-12, 온디바이스 이식).

실측(evals/RESULTS.md 2026-10-12): 1.7B는 표현은 글자 그대로 자르지만 고유어 수사를 못 센다.
그래서 순서가 **표현 대조 → 월·일 → 규칙 파서 → 폰 일수**다. LLM 호출 없음.
"""

from __future__ import annotations

from datetime import date

from fastapi.testclient import TestClient

from medimate.api.app import create_app
from medimate.dialog.memo import assemble_from_client, followup_date, followup_from_reading

V = date(2026, 9, 15)


def _r(text, days=None, month=None, day=None):
    return {"text": text, "days": days, "month": month, "day": day}


def test_rules_win_over_the_phone_count_when_they_know_the_word():
    """열흘·사나흘·두 달은 폰이 틀려도(14·12·없음) 규칙이 센다."""
    fu = followup_from_reading(_r("한 열흘 뒤에", days=14), "한 열흘 뒤에 오라고", V)
    assert fu is not None and fu.date == "2026-09-25"
    assert fu.basis.startswith("폰 읽음 '한 열흘 뒤에' → 규칙")
    fu = followup_from_reading(_r("사나흘 뒤에", days=12), "사나흘 뒤에 경과 보자고", V)
    assert fu is not None and fu.date in ("2026-09-18", "2026-09-19")
    fu = followup_from_reading(_r("두 달 후", days=None), "두 달 후 재검", V)
    assert fu is not None and fu.date == "2026-11-15"  # 달력으로 두 달, 60일이 아니다
    fu = followup_from_reading(_r("다다음주에", days=7), "다다음주에 재방문", V)
    assert fu is not None and fu.date == "2026-09-29"


def test_the_phone_count_is_used_only_when_the_rules_do_not_know_the_word():
    """`연말`은 파서에 없다 — 폰이 센 100을 쓴다. basis에 폰 일수라고 남는다.

    (`담주`·`이틀`도 1차 실측 때는 여기였다. 2026-10-12에 규칙 표에 들어가 이제 규칙이 먼저 센다)
    """
    fu = followup_from_reading(_r("연말", days=100), "연말에 보자고", V)
    assert fu is not None and fu.date == "2026-12-24" and "= 100d" in fu.basis and fu.approximate
    fu = followup_from_reading(_r("담주", days=99), "담주에 보자고", V)
    assert fu is not None and fu.date == "2026-09-22" and "→ 규칙" in fu.basis


def test_an_explicit_date_is_taken_as_is():
    fu = followup_from_reading(_r("10월 3일", month=10, day=3), "10월 3일에 오라고", V)
    assert fu is not None and fu.date == "2026-10-03" and not fu.approximate


def test_what_the_phone_says_must_be_in_the_sentence_verbatim():
    """앞 문장의 약 기간을 끌어온 것, 빈 표현, 범위 밖 일수는 전부 None — 지어내지 않는다."""
    assert followup_from_reading(_r("3일치 약 먹고", days=3), "다시 오세요", V) is None
    assert followup_from_reading(_r("", days=7), "안 좋아지면 바로 오래요", V) is None
    assert followup_from_reading(_r("연말", days=0), "연말에 보자고", V) is None
    assert followup_from_reading(_r("연말", days=9999), "연말에 보자고", V) is None
    assert followup_from_reading(_r("연말", days=True), "연말에 보자고", V) is None
    assert followup_from_reading("not a dict", "담주에 보자고", V) is None


def test_a_month_and_a_half_is_now_a_rule():
    """실측에서 규칙도 폰도 몰랐던 유일한 표현. 규칙 표에 넣었다 — 45일."""
    fu = followup_date("한 달 반 뒤에 오라고", V)
    assert fu is not None and fu.date == "2026-10-30"
    assert followup_date("한 달 뒤에 오라고", V).date == "2026-10-15"  # 기존 규칙은 그대로


def test_client_path_uses_the_phone_reading_without_any_llm():
    """labels + sentences + follow_up_readings → 서버 LLM 0으로 날짜까지 나온다."""
    memo = "위염 초기래요. 연말에 보자고 하셨어요."
    sents = ["위염 초기래요.", "연말에 보자고 하셨어요."]
    res = assemble_from_client(
        memo,
        sents,
        {"0": "findings", "1": "follow_up"},
        visit_date=V,
        phone_readings={"연말에보자고하셨어요.": _r("연말", days=100)},
    )
    assert res is not None
    assert res.card.follow_up_date is not None and res.card.follow_up_date.date == "2026-12-24"


def test_the_rule_result_stays_when_it_is_certain_even_if_the_phone_disagrees():
    """규칙이 확실히 읽은 문장은 폰 읽기를 보지 않는다(기존 리더 게이트와 같다)."""
    res = assemble_from_client(
        "2주 뒤에 오라고",
        ["2주 뒤에 오라고"],
        {"0": "follow_up"},
        visit_date=V,
        phone_readings={"2주뒤에오라고": _r("2주 뒤에", days=3)},
    )
    assert res.card.follow_up_date.date == "2026-09-29"


def _no_llm():
    raise AssertionError("LLM must not be built on the client path")


def test_api_accepts_follow_up_readings_on_the_client_path():
    """온디바이스 경로: 폰 라벨 + 폰 읽기. 호출 0, basis가 폰에서 왔다고 남는다."""
    c = TestClient(create_app(_no_llm))
    body = {
        "memo": "위염 초기래요. 연말에 보자고 하셨어요.",
        "visit_date": "2026-09-15",
        "sentences": ["위염 초기래요.", "연말에 보자고 하셨어요."],
        "labels": {"0": "findings", "1": "follow_up"},
        "follow_up_readings": [
            {
                "sentence": "연말에 보자고 하셨어요.",
                "text": "연말",
                "days": 100,
                "month": None,
                "day": None,
            }
        ],
    }
    r = c.post("/v1/postvisit/memo", json=body)
    assert r.status_code == 200, r.text
    fu = r.json()["card"]["follow_up_date"]
    assert fu["date"] == "2026-12-24" and fu["basis"].startswith("폰 읽음")
    assert r.json()["usage"]["cost_usd"] == 0


def test_api_rejects_a_malformed_reading():
    c = TestClient(create_app(_no_llm))
    body = {
        "memo": "담주에 보자고",
        "visit_date": "2026-09-15",
        "sentences": ["담주에 보자고"],
        "labels": {"0": "follow_up"},
        "follow_up_readings": [
            {"sentence": "담주에 보자고", "text": "담주", "days": 7, "extra": 1}
        ],
    }
    assert c.post("/v1/postvisit/memo", json=body).status_code == 422


def test_native_korean_day_words_are_rules_now():
    """예시에 없는 12표현 실측(2026-10-12)에서 폰이 못 세던 말. 규칙이 세면 폰 결과는 안 쓴다."""
    cases = {
        "모레 다시 오라고 하셨어요": "2026-09-17",
        "글피에 보자고": "2026-09-18",
        "석 달 후에 검사 다시 하자고": "2026-12-15",
        "엿새 뒤에 실밥 뽑으러 오라고": "2026-09-21",
        "닷새 후 재방문": "2026-09-20",
        "스무날 뒤쯤 보자고": "2026-10-05",
        "담주에 보자고": "2026-09-22",
        "이틀 뒤에 다시 오세요": "2026-09-17",
        "넉 달 뒤에 오라고": "2027-01-15",
    }
    for sent, want in cases.items():
        fu = followup_date(sent, V)
        assert fu is not None and fu.date == want, (sent, fu)
    # 복용법의 "하루"는 재방문이 아니다 — 표에 넣지 않은 이유
    fu = followup_date("하루 두 번 먹고 다시 오라고", V)
    assert fu is None or "하루" not in fu.text
