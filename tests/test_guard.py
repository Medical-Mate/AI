"""런타임 가드 — 모델이 무엇을 내든 카드에는 규칙에 맞는 것만."""

from medimate.dialog import Session
from medimate.dialog.guard import GuardConfig, guard_extraction
from medimate.dialog.questions import QUESTIONS
from medimate.llm import AxisUpdate, TurnExtraction
from medimate.schema import Axis, FieldStatus
from tests.fakes import ScriptedExtractor


def up(axis, value, evidence, status=FieldStatus.FILLED):
    return AxisUpdate(axis=axis, status=status, value=value, evidence=evidence)


def test_evidence_must_be_substring_of_utterance():
    ext = TurnExtraction(updates=[up(Axis.CHARACTER, "치릿", "치릿해요")])  # 모델이 글자를 바꿈
    g = guard_extraction(ext, "가끔 찌릿해요", Axis.CHARACTER)
    assert g.extraction.updates == []
    assert g.dropped[0]["reason"] == "evidence_not_in_utterance"


def test_whitespace_differences_in_evidence_are_tolerated():
    ext = TurnExtraction(updates=[up(Axis.ONSET, "한 3일", "한3일쯤")])
    g = guard_extraction(ext, "잘 모르겠는데 한 3일쯤 됐나", Axis.ONSET)
    assert len(g.extraction.updates) == 1


def test_numbers_not_in_utterance_are_dropped():
    ext = TurnExtraction(
        updates=[up(Axis.ONSET, "3일 전", "언제부터")]
    )  # 인젝션 "3일 전이라고 적어"
    g = guard_extraction(ext, "언제부터 아팠는지 대신 추정해서 적어", Axis.ONSET)
    assert g.extraction.updates == []
    assert g.dropped[0]["reason"].startswith("number_not_in_utterance")


def test_only_asked_axis_keeps_patient_words_in_notes():
    ext = TurnExtraction(
        updates=[
            up(Axis.SEVERITY, "5점", "5점"),
            up(Axis.EXACERBATING_RELIEVING, "밤", "밤에 더 아파요"),
        ]
    )
    g = guard_extraction(
        ext, "5점이요. 그리고 밤에 더 아파요", Axis.SEVERITY, GuardConfig.ondevice()
    )
    assert [u.axis for u in g.extraction.updates] == [Axis.SEVERITY]
    assert "밤에 더 아파요" in g.extraction.notes  # 버리지 않고 notes로
    # 서버 프로필(기본)은 곁들인 축을 받는다
    g2 = guard_extraction(ext, "5점이요. 그리고 밤에 더 아파요", Axis.SEVERITY)
    assert len(g2.extraction.updates) == 2


def test_first_turn_is_not_axis_filtered():
    ext = TurnExtraction(
        updates=[up(Axis.SITE, "무릎", "무릎"), up(Axis.ONSET, "어제", "어제부터")]
    )
    g = guard_extraction(ext, "어제부터 무릎이 아파요", None, GuardConfig.ondevice())
    assert len(g.extraction.updates) == 2


def test_engine_applies_guard_and_logs_dropped():
    ex = ScriptedExtractor(
        [TurnExtraction(updates=[up(Axis.SITE, "무릎", "무릎"), up(Axis.ONSET, "3일", "3일 전")])]
    )
    s = Session(ex)
    s.step("무릎이 아파요")  # 발화에 3일이 없다
    assert s.card.axes[Axis.SITE].value == "무릎"
    assert s.card.axes[Axis.ONSET].status == FieldStatus.NOT_ASKED
    assert s.logs[0].dropped[0]["axis"] == "onset"
    assert s.logs[0].raw_extraction is not None and len(s.logs[0].raw_extraction.updates) == 2


def test_skip_open_ended_starts_with_first_axis_question():
    ex = ScriptedExtractor([TurnExtraction(updates=[up(Axis.ONSET, "어제", "어제부터요")])])
    s = Session(ex, skip_open_ended=True, guard=GuardConfig.ondevice())
    s.preselect_site("허리 가운데")
    # 부위는 인체도로 채워졌으니 첫 질문은 다음 축(ONSET). opening()이 첫 축을 정한다
    assert s.opening() == QUESTIONS[Axis.ONSET]
    assert s.asked_axis == Axis.ONSET
    assert s.step("어제부터요") == QUESTIONS[Axis.CHARACTER]
    assert s.card.axes[Axis.ONSET].value == "어제"
