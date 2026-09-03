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
    assert p["axes"]["site"] == {"status": "filled", "value": "무릎", "evidence": ["무릎이요"]}
    assert p["axes"]["onset"]["status"] == "not_asked"
    assert p["minimally_complete"] is False
