from medimate.schema import Axis, AxisEntry, FieldStatus, PreVisitCard, to_backend_payload


def test_new_card_is_empty_and_incomplete():
    card = PreVisitCard()
    assert card.completeness() == 0
    assert not card.is_minimally_complete()
    assert card.unfilled_axes() == list(Axis)


def test_minimal_completion_needs_site_and_chief_complaint():
    card = PreVisitCard(chief_complaint="무릎이 아파요")
    assert not card.is_minimally_complete()
    card.axes[Axis.SITE] = AxisEntry(
        status=FieldStatus.FILLED, value="오른쪽 무릎", evidence=["무릎"]
    )
    assert card.is_minimally_complete()


def test_unknown_counts_as_answered():
    card = PreVisitCard()
    card.axes[Axis.ONSET] = AxisEntry(status=FieldStatus.UNKNOWN, evidence=["모르겠어요"])
    assert card.completeness() == 1 / len(Axis)


def test_export_keeps_status_per_axis():
    card = PreVisitCard()
    card.axes[Axis.SITE] = AxisEntry(status=FieldStatus.FILLED, value="무릎", evidence=["무릎이요"])
    p = to_backend_payload(card)
    assert p["axes"]["site"] == {
        "status": "filled",
        "value": "무릎",
        "evidence": ["무릎이요"],
        "source": "ai_extraction",  # 발화에서 뽑았다
    }
    assert p["axes"]["onset"]["status"] == "not_asked"
    assert p["axes"]["onset"]["source"] is None  # 값이 없으면 출처도 없다
    assert p["minimally_complete"] is False


def test_axis_source_tells_where_the_value_came_from():
    """백엔드 합의(#7). `ai_extraction` / `selection` / `patient_edit` 세 값.

    `patient_edit`은 **백엔드가 채운다** — 환자가 카드 화면에서 고친 값이라 우리는 만들지
    않는다. 우리 몫은 앞의 둘을 채우고 자리를 만들어 두는 것이다.
    """
    card = PreVisitCard()
    card.axes[Axis.SITE] = AxisEntry(
        status=FieldStatus.FILLED, value="왼쪽 허리 옆", evidence=["[부위 선택] 왼쪽 허리 옆"]
    )
    card.axes[Axis.SEVERITY] = AxisEntry(
        status=FieldStatus.FILLED, value="4 (매우 심함)", evidence=["[선택] 4 (매우 심함)"]
    )
    card.axes[Axis.ONSET] = AxisEntry(
        status=FieldStatus.FILLED, value="2주 전", evidence=["2주 전부터요"]
    )
    axes = to_backend_payload(card)["axes"]
    # 인체도·슬라이더로 고른 값은 LLM을 안 불렀다. evidence 표시가 그대로 판정 근거가 된다
    assert axes["site"]["source"] == "selection"
    assert axes["severity"]["source"] == "selection"
    assert axes["onset"]["source"] == "ai_extraction"


def test_axis_source_is_none_for_an_empty_axis():
    """빈 값에 출처를 붙이면 백엔드가 "AI가 뽑았는데 비어 있다"로 읽는다"""
    card = PreVisitCard()
    card.axes[Axis.SITE] = AxisEntry(status=FieldStatus.SKIPPED, value=None, evidence=[])
    axes = to_backend_payload(card)["axes"]
    assert axes["site"]["source"] is None
    assert axes["character"]["source"] is None
