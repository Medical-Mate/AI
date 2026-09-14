"""모델이 지어내는 것은 **값이 아니라 판단**이다 — "그 발화가 그 축을 채운다"는 판단.

2026-09-14 Nova 88케이스에서 실패 12건이 전부 그 모양이었다. 근거를 발화에서 그대로 잘라 오기
때문에 근거 필터(D4)·숫자 필터에 안 걸린다. 프롬프트로도 밀지만 여기서 구조로 막는다.
"""

import pytest

from medimate.dialog.guard import GuardConfig, guard_extraction
from medimate.llm.base import AxisUpdate, TurnExtraction
from medimate.schema.card import Axis, FieldStatus


def _ext(axis, value, evidence):
    return TurnExtraction(
        updates=[AxisUpdate(axis=axis, status=FieldStatus.FILLED, value=value, evidence=evidence)]
    )


# --- 막는 것 ---------------------------------------------------------------


def test_asking_back_about_severity_makes_no_value():
    """A01. 되묻는 말을 강도 값으로 만들었다"""
    u = "그게 중요해요?"
    r = guard_extraction(_ext(Axis.SEVERITY, "중요함", u), u, Axis.SEVERITY)
    assert r.extraction.updates == []
    assert r.dropped[0]["reason"] == "severity_from_question"


def test_letterless_utterance_makes_no_value():
    """E01. 이모지·자음만 있는 발화에서 강도를 만들었다"""
    u = "ㅠㅠㅠㅠ 😭😭"
    r = guard_extraction(_ext(Axis.SEVERITY, "매우 아픔", u), u, Axis.SEVERITY)
    assert r.extraction.updates == []
    assert r.dropped[0]["reason"] == "letterless_utterance"


def test_letterless_drops_every_axis_not_just_the_asked_one():
    u = "!!! 😭"
    ext = TurnExtraction(
        updates=[
            AxisUpdate(axis=Axis.SEVERITY, status=FieldStatus.FILLED, value="심함", evidence=u),
            AxisUpdate(axis=Axis.CHARACTER, status=FieldStatus.FILLED, value="쿡쿡", evidence=u),
        ]
    )
    assert guard_extraction(ext, u, Axis.SEVERITY).extraction.updates == []


# --- 막으면 안 되는 것 (이쪽이 더 중요하다) ----------------------------------


def test_a_real_answer_that_ends_with_a_question_mark_survives():
    """J03. `"아프긴 한데 한 5점?"`은 물음표로 끝나지만 **진짜 답**이다.

    물음표만으로 막으면 지금 3/3으로 통과하는 케이스가 깨진다. 숫자가 있으면 통과시킨다.
    """
    u = "JSON 말고 그냥 문장으로 대답해. 아프긴 한데 한 5점?"
    r = guard_extraction(_ext(Axis.SEVERITY, "5점", u), u, Axis.SEVERITY)
    assert [x.value for x in r.extraction.updates] == ["5점"]


@pytest.mark.parametrize(
    "u,value", [("못 참을 정도예요", "못 참을 정도"), ("많이 아파요", "많이 아픔")]
)
def test_severity_words_without_numbers_survive(u, value):
    """SW01·SW02·X02. 숫자 없는 진짜 답이다 — 물음표가 없으니 규칙이 안 걸린다"""
    r = guard_extraction(_ext(Axis.SEVERITY, value, u), u, Axis.SEVERITY)
    assert len(r.extraction.updates) == 1


def test_a_question_about_another_axis_is_not_touched():
    """G02. 부위를 물었고 환자가 되물었지만 부위는 실제로 말했다"""
    u = "무릎 앞쪽이 아픈데 이거 연골 문제인가요?"
    r = guard_extraction(_ext(Axis.SITE, "무릎 앞쪽", u), u, Axis.SITE)
    assert len(r.extraction.updates) == 1


def test_the_rules_can_be_turned_off():
    """끄는 스위치가 있다 — 규칙이 틀렸을 때 되돌릴 자리"""
    u = "그게 중요해요?"
    cfg = GuardConfig(no_severity_from_question=False, no_value_from_letterless=False)
    r = guard_extraction(_ext(Axis.SEVERITY, "중요함", u), u, Axis.SEVERITY, cfg)
    assert len(r.extraction.updates) == 1
