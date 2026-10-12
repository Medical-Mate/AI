"""가드 규칙 7·8 (2026-10-12): 모름은 unknown, 예/아니오 축의 짧은 부정은 "없음".

폰 1.7B 실측에서 남은 채팅 실패 3건(CA03 `ㅁㄹ`→skipped, CA06 `ㅁㄹ 그냥 계속 아픔`→빈 칸,
CM03 `ㄴㄴ ㅠㅠ`→skipped)을 프롬프트가 아니라 가드로 닫는다. 모든 프로필에 걸린다.
"""

from __future__ import annotations

from medimate.dialog.guard import GuardConfig, guard_extraction
from medimate.llm.base import AxisUpdate, TurnExtraction
from medimate.schema.card import Axis, FieldStatus


def _ext(*updates):
    return TurnExtraction(updates=[AxisUpdate(**u) for u in updates], notes=[])


def test_a_bare_unknown_turns_skipped_into_unknown_on_the_asked_axis():
    """`ㅁㄹ`(→모르겠다)을 폰 모델이 skipped로 냈다. 모름은 건너뜀이 아니다."""
    ext = _ext({"axis": Axis.ONSET, "status": "skipped", "value": "", "evidence": "ㅁㄹ"})
    g = guard_extraction(ext, "ㅁㄹ", Axis.ONSET, GuardConfig.ondevice(), normalized="모르겠다")
    (u,) = g.extraction.updates
    assert u.status == FieldStatus.UNKNOWN and u.evidence == "ㅁㄹ"
    assert any(d["reason"] == "bare_unknown_is_unknown" for d in g.dropped)


def test_a_bare_unknown_fills_an_emptied_axis_but_keeps_a_real_answer():
    """`ㅁㄹ 그냥 계속 아픔`: 모델이 비우면 unknown, 모델이 filled로 내용을 뽑았으면 그대로."""
    axis = Axis.EXACERBATING_RELIEVING
    norm = "모르겠다 그냥 계속 아픔"
    g = guard_extraction(
        _ext(), "ㅁㄹ 그냥 계속 아픔", axis, GuardConfig.ondevice(), normalized=norm
    )
    (u,) = g.extraction.updates
    assert u.axis == axis and u.status == FieldStatus.UNKNOWN
    assert u.evidence == "ㅁㄹ 그냥 계속 아픔"

    ext = _ext(
        {"axis": axis, "status": "filled", "value": "계속 아픔", "evidence": "그냥 계속 아픔"}
    )
    g = guard_extraction(ext, "ㅁㄹ 그냥 계속 아픔", axis, GuardConfig.ondevice(), normalized=norm)
    (u,) = g.extraction.updates
    assert u.status == FieldStatus.FILLED and u.value == "계속 아픔"


def test_a_bare_negative_fills_a_yes_no_axis_with_the_axis_word():
    """`ㄴㄴ ㅠㅠ`(→아니)를 퍼짐 축에: filled "안 퍼짐", 근거는 원문 그대로."""
    ext = _ext({"axis": Axis.RADIATION, "status": "skipped", "value": "", "evidence": "ㄴㄴ ㅠㅠ"})
    g = guard_extraction(
        ext, "ㄴㄴ ㅠㅠ", Axis.RADIATION, GuardConfig.ondevice(), normalized="아니"
    )
    (u,) = g.extraction.updates
    assert u.status == FieldStatus.FILLED and u.value == "안 퍼짐" and u.evidence == "ㄴㄴ ㅠㅠ"

    # 모델이 비워도 채운다. 동반 축은 "없음"
    g = guard_extraction(_ext(), "ㄴㄴ", Axis.ASSOCIATED, GuardConfig(), normalized="아니")
    (u,) = g.extraction.updates
    assert u.status == FieldStatus.FILLED and u.value == "없음" and u.evidence == "ㄴㄴ"


def test_a_bare_negative_leaves_other_axes_and_real_answers_alone():
    # 예/아니오 축이 아니면 그대로(비어 있으면 비어 있는 대로 — 엔진이 되묻거나 닫는다)
    g = guard_extraction(_ext(), "ㄴㄴ", Axis.ONSET, GuardConfig(), normalized="아니")
    assert g.extraction.updates == []

    # 모델이 이미 filled로 냈으면(값 "ㄴㄴ" 그대로라도) 바꾸지 않는다
    ext = _ext({"axis": Axis.RADIATION, "status": "filled", "value": "ㄴㄴ", "evidence": "ㄴㄴ"})
    g = guard_extraction(ext, "ㄴㄴ", Axis.RADIATION, GuardConfig(), normalized="아니")
    assert g.extraction.updates[0].value == "ㄴㄴ"

    # 내용이 붙은 부정은 짧은 부정이 아니다
    ext = _ext(
        {
            "axis": Axis.RADIATION,
            "status": "filled",
            "value": "종아리",
            "evidence": "아니 종아리로 내려가요",
        }
    )
    g = guard_extraction(ext, "아니 종아리로 내려가요", Axis.RADIATION, GuardConfig())
    assert g.extraction.updates[0].value == "종아리" and not g.dropped


def test_rules_can_be_switched_off():
    cfg = GuardConfig(bare_unknown_is_unknown=False, bare_negative_fills_yes_no_axis=False)
    g = guard_extraction(_ext(), "ㅁㄹ", Axis.ONSET, cfg, normalized="모르겠다")
    assert g.extraction.updates == []
    g = guard_extraction(_ext(), "ㄴㄴ", Axis.RADIATION, cfg, normalized="아니")
    assert g.extraction.updates == []
