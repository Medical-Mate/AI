"""가드 규칙 11 — 근거·값이 원문과 띄어쓰기·한두 글자만 다르면 원문 글자로 되돌린다(폰 프로필).

규칙 7·8 확장(모름 앞 군말, 부정 두 낱말)도 여기서 본다. 사례는 Qwen3.5-2B 108문항 실측(2026-10-13).
"""

import pytest

from medimate.dialog.guard import GuardConfig, guard_extraction
from medimate.llm.base import AxisUpdate, TurnExtraction
from medimate.schema.card import Axis, FieldStatus
from medimate.text.chatnorm import normalize

pytest.importorskip("kiwipiepy")  # 폰 프로필은 규칙 10(Kiwi)과 같이 돈다

ONDEVICE = GuardConfig.ondevice()


def _run(axis, value, evidence, utterance, status=FieldStatus.FILLED, cfg=ONDEVICE, norm=False):
    ups = [AxisUpdate(axis=axis, status=status, value=value, evidence=evidence)] if evidence else []
    n = normalize(utterance).text if norm else None
    return guard_extraction(TurnExtraction(updates=ups), utterance, axis, cfg, normalized=n)


def test_one_letter_off_evidence_is_snapped_to_the_original():
    g = _run(Axis.CHARACTER, "욱신거림", "욕신거려요", "욱신거려요")
    [u] = g.extraction.updates
    assert (u.value, u.evidence) == ("욱신거림", "욱신거려요")
    assert any(d["reason"] == "evidence_snapped" for d in g.dropped)


def test_whitespace_only_evidence_and_value_get_the_original_spacing():
    g = _run(Axis.ONSET, "4 일 전부터", "3 일, 아니 4 일 전부터요", "3일, 아니 4일 전부터요")
    [u] = g.extraction.updates
    assert (u.value, u.evidence) == ("4일 전부터", "3일, 아니 4일 전부터요")


def test_far_off_evidence_is_still_dropped():
    """한두 글자가 아니라 지어낸 근거는 그대로 버린다"""
    g = _run(Axis.ONSET, "3일 전부터", "3일 전부터", "언제부터인지 기억이 잘 안 나요")
    assert all(u.value != "3일 전부터" for u in g.extraction.updates)
    assert g.dropped[0]["reason"] == "evidence_not_in_utterance"


def test_server_profile_keeps_dropping_changed_letters():
    """서버(Nova)는 규칙 10이 없으므로 글자 바꾼 근거를 되돌리지 않고 버린다(기존 규칙 2)"""
    g = _run(Axis.CHARACTER, "치릿", "치릿해요", "가끔 찌릿해요", cfg=GuardConfig())
    assert g.extraction.updates == []


@pytest.mark.parametrize("utterance", ["그건 잘 모르겠어요", "음 모르겠어요"])
def test_unknown_after_a_filler_word(utterance):
    g = _run(Axis.RADIATION, None, None, utterance)
    assert [(u.status, u.value) for u in g.extraction.updates] == [(FieldStatus.UNKNOWN, None)]


def test_unknown_phrase_filled_as_a_value_becomes_unknown():
    g = _run(Axis.ASSOCIATED, "기억안남", "기억안남", "기억안남", norm=True)
    assert [(u.status, u.value) for u in g.extraction.updates] == [(FieldStatus.UNKNOWN, None)]


def test_two_negative_words_close_a_yes_no_axis():
    g = _run(Axis.ASSOCIATED, None, None, "ㄴㄴ 없음", norm=True)
    assert [(u.status, u.value) for u in g.extraction.updates] == [(FieldStatus.FILLED, "없음")]
