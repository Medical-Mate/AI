"""채팅 표기 복원과 가드·프롬프트 v5·채점기 D4의 연동 (#117). 실호출 없음."""

import pytest

from medimate.dialog.guard import GuardConfig, guard_extraction
from medimate.llm import AxisUpdate, TurnExtraction, prompt_v4_nova, prompt_v5_nova
from medimate.schema import Axis, FieldStatus
from medimate.text.chatnorm import normalize


def up(axis, value, evidence, status=FieldStatus.FILLED):
    return AxisUpdate(axis=axis, status=status, value=value, evidence=evidence)


def test_consonant_only_answer_is_dropped_without_normalized():
    # 지금 운영 동작: 자음만 있는 발화는 글자 없음 필터가 갱신을 전부 버린다
    ext = TurnExtraction(updates=[up(Axis.RADIATION, "안 퍼짐", "ㄴㄴ")])
    g = guard_extraction(ext, "ㄴㄴ", Axis.RADIATION)
    assert g.extraction.updates == [] and g.dropped[0]["reason"] == "letterless_utterance"


def test_consonant_only_answer_passes_with_normalized():
    ext = TurnExtraction(updates=[up(Axis.RADIATION, "안 퍼짐", "ㄴㄴ")])
    n = normalize("ㄴㄴ")
    g = guard_extraction(ext, "ㄴㄴ", Axis.RADIATION, normalized=n.text)
    assert len(g.extraction.updates) == 1 and g.dropped == []


def test_evidence_from_normalized_text_is_accepted():
    # 모델이 표기 참고(`괜찮다`)에서 근거를 잘라도 받는다 — 표로만 바꾼 것이라 출처가 남는다
    ext = TurnExtraction(updates=[up(Axis.TIME_COURSE, "괜찮음", "괜찮다")])
    g = guard_extraction(ext, "ㄱㅊ", Axis.TIME_COURSE, normalized=normalize("ㄱㅊ").text)
    assert len(g.extraction.updates) == 1


def test_markers_only_stay_letterless():
    # `ㅠㅠ ㅋㅋ`는 복원해도 글자가 없다 — 값이 없다는 판정 그대로
    n = normalize("ㅠㅠ ㅋㅋ")
    ext = TurnExtraction(updates=[up(Axis.SEVERITY, "심함", "ㅠㅠ")])
    g = guard_extraction(ext, "ㅠㅠ ㅋㅋ", Axis.SEVERITY, normalized=n.text or None)
    assert g.extraction.updates == []


def test_invented_evidence_still_dropped_with_normalized():
    ext = TurnExtraction(updates=[up(Axis.ONSET, "3일", "3일 전부터")])
    g = guard_extraction(ext, "ㅁㄹ", Axis.ONSET, normalized=normalize("ㅁㄹ").text)
    assert g.extraction.updates == []


def test_v5_user_message_adds_reference_line_only_when_changed():
    m = prompt_v5_nova.user_message("ㄱㅊ", Axis.TIME_COURSE)
    assert "표기 참고: 괜찮다" in m and prompt_v4_nova.OPEN + "ㄱㅊ" + prompt_v4_nova.CLOSE in m
    plain = prompt_v5_nova.user_message("한 달 전부터요", Axis.ONSET)
    assert "표기 참고" not in plain
    assert plain == prompt_v4_nova.user_message("한 달 전부터요", Axis.ONSET)


def test_v5_system_prompt_keeps_v4_and_adds_rule17():
    s = prompt_v5_nova.system_prompt()
    assert "17." in s and "표기 참고" in s
    assert prompt_v4_nova.OPEN in s  # v4 인젠션 방어 그대로
    assert prompt_v5_nova.PROMPT_VERSION == "extract-v5-nova"


def test_score_d4_accepts_normalized_substring():
    from medimate.evals.score import score

    case = {
        "id": "x",
        "utterance": "ㄴㄴ",
        "asked_axis": "radiation",
        "first_turn": False,
        "expect": {"axes": {"radiation": "filled"}, "wants_to_stop": False},
    }
    parsed = TurnExtraction(updates=[up(Axis.RADIATION, "안 퍼짐", "아니")])
    sc = score(case, "{}", parsed, None, [])
    # 복원문 `아니`에서 글자 그대로 잘랐으니 D4·D4S 둘 다 통과. 복원은 표로만 하므로 출처가 남는다
    assert sc.checks["D4"] is True and sc.checks["D4S"] is True
    assert sc.passed
    parsed2 = TurnExtraction(updates=[up(Axis.RADIATION, "안 퍼짐", "안퍼짐")])
    assert score(case, "{}", parsed2, None, []).checks["D4"] is False  # 지어낸 근거는 여전히 실패


@pytest.mark.parametrize("flag,expected", [("off", None), ("on", "아니")])
def test_engine_flag_controls_normalized(monkeypatch, flag, expected):
    from medimate.dialog import engine as eng

    monkeypatch.setenv("MEDIMATE_CHAT_NORMALIZE", flag)
    assert eng.chat_normalize_enabled() == (flag != "off")
    got = eng._chat_normalized("ㄴㄴ") if eng.chat_normalize_enabled() else None
    assert got == expected


def test_guard_config_unchanged_defaults():
    cfg = GuardConfig()
    assert cfg.no_value_from_letterless and cfg.evidence_substring


@pytest.mark.parametrize("utt", ["ㅇㅇ", "네", "응 ㅋㅋ", "그렇다"])
def test_bare_affirmative_becomes_ambiguous_instead_of_a_value(utt):
    # "퍼지나요?"에 ㅇㅇ → 모델이 `안 퍼짐`을 내도 카드에는 안 들어간다. 되묻는다
    ext = TurnExtraction(updates=[up(Axis.RADIATION, "안 퍼짐", utt)])
    n = normalize(utt)
    g = guard_extraction(ext, utt, Axis.RADIATION, normalized=n.text if n.changed else None)
    assert len(g.extraction.updates) == 1
    u = g.extraction.updates[0]
    assert u.status == FieldStatus.AMBIGUOUS and u.value is None and u.evidence == utt
    assert g.dropped[0]["reason"] == "bare_affirmative_asks_back"


def test_affirmative_with_content_is_kept():
    # "네 허벅지로요"는 값이 있다 — 되묻지 않는다
    ext = TurnExtraction(updates=[up(Axis.RADIATION, "허벅지", "네 허벅지로요")])
    g = guard_extraction(ext, "네 허벅지로요", Axis.RADIATION)
    assert g.extraction.updates[0].status == FieldStatus.FILLED and g.dropped == []


def test_bare_negative_is_still_a_value():
    # ㄴㄴ(아니)는 "안 퍼짐"이라는 값이다 — NG01 결정. 되묻지 않는다
    ext = TurnExtraction(updates=[up(Axis.RADIATION, "안 퍼짐", "ㄴㄴ")])
    g = guard_extraction(ext, "ㄴㄴ", Axis.RADIATION, normalized=normalize("ㄴㄴ").text)
    assert g.extraction.updates[0].status == FieldStatus.FILLED
