"""가드 규칙 10 — 값이 그 발화에서 나오지 않았으면 환자가 한 말로 바꾼다(폰 프로필).

사례는 폰 1.7B 이어 돌리기(카드 100장, 2026-10-13)에서 나온 실제 출력이다.
"""

import pytest

from medimate.dialog.guard import GuardConfig, guard_extraction
from medimate.llm.base import AxisUpdate, TurnExtraction
from medimate.schema.card import Axis, FieldStatus

pytest.importorskip("kiwipiepy")

ONDEVICE = GuardConfig.ondevice()


def _one(axis, value, evidence, utterance, cfg=ONDEVICE):
    ext = TurnExtraction(
        updates=[AxisUpdate(axis=axis, status=FieldStatus.FILLED, value=value, evidence=evidence)]
    )
    g = guard_extraction(ext, utterance, axis, cfg)
    return g.extraction.updates, g.dropped


@pytest.mark.parametrize(
    "axis,value,utterance",
    [
        (Axis.ASSOCIATED, "립구름", "아침에 좀 뻣뻣해요"),  # 깨진 토큰
        (Axis.ASSOCIATED, "열이 나요", "빨갛거나 열이 나지는 않아요"),  # 부정 뒤집기
        (Axis.ASSOCIATED, "다리 저림", "다리 저림은 없어요"),  # 부정 빠뜨림
        # 프롬프트 예시 베끼기 — 뜻이 반대다
        (
            Axis.EXACERBATING_RELIEVING,
            "타이레놀 먹으면 더 심하고 누우면 괜찮아요",
            "타이레놀 먹으면 좀 나아요",
        ),
    ],
)
def test_value_not_from_utterance_becomes_patient_words(axis, value, utterance):
    ups, dropped = _one(axis, value, utterance, utterance)
    assert [(u.status, u.value, u.evidence) for u in ups] == [
        (FieldStatus.FILLED, utterance, utterance)
    ]
    assert dropped[0]["reason"].startswith("value_not_from_utterance:")
    assert dropped[0]["value"] == value  # 무엇을 바꿨는지 audit에 남는다


def test_value_taken_from_beyond_the_evidence_still_counts():
    """대조는 근거가 아니라 발화 전체다. 근거를 짧게 잡아도 값이 발화에 있으면 통과"""
    ups, dropped = _one(Axis.ONSET, "3주쯤", "이삿짐 정리하고부터", "3주쯤, 이삿짐 정리하고부터")
    assert ups[0].value == "3주쯤" and not dropped


@pytest.mark.parametrize(
    "axis,value,utterance",
    [
        (Axis.ONSET, "2주 전부터", "2주 전부터요, 등산 다음 날부터"),
        (Axis.ONSET, "3주", "3일이 아니라 3주예요"),  # 고쳐 말하기의 앞부분은 대조에서 뺀다
        (Axis.CHARACTER, "시큰하고 찌릿해요", "시큰하고 가끔 찌릿해요"),
        (Axis.EXACERBATING_RELIEVING, "계단 내려갈 때", "계단 내려갈 때, 오래 앉았다 일어날 때"),
    ],
)
def test_sourced_values_pass_unchanged(axis, value, utterance):
    ups, dropped = _one(axis, value, utterance, utterance)
    assert ups[0].value == value
    assert not dropped


def test_server_profile_is_unchanged():
    """서버(Nova) 경로는 켜지 않는다 — 저장된 큰 모델 결과에 이런 값이 없었다"""
    ups, dropped = _one(
        Axis.ASSOCIATED, "립구름", "아침에 좀 뻣뻣해요", "아침에 좀 뻣뻣해요", GuardConfig()
    )
    assert ups[0].value == "립구름" and not dropped


def test_bare_negative_value_is_not_rechecked():
    """규칙 8이 만든 값(`안 퍼짐`)은 모델 값이 아니다 — 규칙 10이 되돌리지 않는다"""
    ext = TurnExtraction(
        updates=[
            AxisUpdate(axis=Axis.RADIATION, status=FieldStatus.SKIPPED, value="", evidence="없어요")
        ]
    )
    g = guard_extraction(ext, "없어요", Axis.RADIATION, ONDEVICE)
    assert [(u.status, u.value) for u in g.extraction.updates] == [(FieldStatus.FILLED, "안 퍼짐")]
