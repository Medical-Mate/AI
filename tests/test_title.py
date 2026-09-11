"""카드 제목 — `{부위 라벨} · {기간}`. LLM을 부르지 않는다.

2026-09-11 백엔드 합의(#7). 처음 제안은 `chief_complaint`에서 뽑는 것이었는데 양쪽이 철회했다 —
환자 첫 발화라 길고 구어체여서 20자에 맞추려면 잘라야 하고, **자르는 순간 환자 말이 아니게 된다.**
"""

from medimate.schema.card import Axis, AxisEntry, FieldStatus, PreVisitCard, SiteSelectionRecord
from medimate.schema.export import TITLE_MAX, _title, to_backend_payload


def card(site=None, onset=None, time_course=None, label=None):
    c = PreVisitCard()
    for axis, v in ((Axis.SITE, site), (Axis.ONSET, onset), (Axis.TIME_COURSE, time_course)):
        if v:
            c.axes[axis] = AxisEntry(status=FieldStatus.FILLED, value=v, evidence=[v])
    if label:
        c.site_selection = SiteSelectionRecord(
            node_id="SUR:001", anchor_id="ANC:001", label=label, ontology_snapshot="test"
        )
    return c


def test_site_and_duration():
    assert (
        _title(card(site="왼쪽 무릎 안쪽", onset="2주 전부터요, 등산 다음 날부터"))
        == "왼쪽 무릎 안쪽 · 2주"
    )


def test_native_korean_date_words():
    """숫자+단위가 아니라 한 낱말이라 정규식이 못 잡는다. 실제 카드에 있었다"""
    for onset, want in (
        ("이틀 전부터", "이틀"),
        ("닷새 전에 넘어지고부터", "닷새"),
        ("열흘쯤 전에", "열흘"),
    ):
        assert _title(card(site="귀", onset=onset)) == f"귀 · {want}"


def test_relative_words():
    assert _title(card(site="허리", onset="어제 이사하면서 짐 들다가")) == "허리 · 어제"


def test_duration_is_extracted_not_truncated():
    """원문을 자르지 않는다 — 숫자+단위만 뽑는다"""
    t = _title(card(site="명치", onset="한 3주쯤 됐어요"))
    assert t == "명치 · 3주"
    assert "됐어요" not in t


def test_falls_back_to_site_when_no_duration():
    assert _title(card(site="무릎", onset="언제부턴지 정확히 모르겠어요")) == "무릎"


def test_none_without_a_site():
    """백엔드가 제목 줄을 숨긴다. department_guidance와 같은 처리"""
    assert _title(card(onset="3일 전")) is None
    assert _title(PreVisitCard()) is None


def test_map_selection_label_wins_over_the_axis():
    """인체도에서 짚은 라벨에는 좌우가 이미 붙어 있다"""
    c = card(site="무릎", onset="3일 전", label="왼쪽 무릎")
    assert _title(c) == "왼쪽 무릎 · 3일"


def test_never_exceeds_the_limit_and_never_cuts_mid_label():
    """20자를 넘으면 **기간을 뺀다.** 자르면 부위 이름이 깨진다"""
    long_site = "오른쪽 발바닥, 뒤꿈치 앞쪽"
    t = _title(card(site=long_site, onset="한 달쯤"))
    assert t == long_site and len(t) <= TITLE_MAX


def test_no_symptom_word_is_invented():
    """와이어프레임 제목이 "복부 통증 · 3주"인데 "통증"은 부위 라벨에도 축에도 없다.

    붙이면 우리가 만들지 않은 사실을 넣는 것이고, 느낌 축이 "먹먹해요"인 카드에 "통증"이라고
    쓰면 틀린다. 부위 라벨만 쓴다.
    """
    t = _title(card(site="오른쪽 귀 안", onset="이틀 전", time_course="먹먹하고 웅웅 울려요"))
    assert t == "오른쪽 귀 안 · 이틀"
    for word in ("통증", "아픔", "증상"):
        assert word not in t


def test_title_is_in_the_previsit_payload_only():
    out = to_backend_payload(card(site="무릎", onset="3일 전"))
    assert out["title"] == "무릎 · 3일"
    assert "evidence" not in str(out["title"])  # 제목에는 근거를 붙이지 않는다


def test_time_course_is_not_a_duration_source():
    """경과 축은 쓰지 않는다(백엔드 합의 #7).

    카드 100장에서 경과가 기여하는 건 PC31 한 장인데 그게 정확히 틀린 예다 —
    "한 번 생기면 열흘쯤 있다 아물어요"의 `열흘`은 **삽화 하나의 지속 기간**이지
    발병 후 경과가 아니다. 제목에 붙이면 "열흘 전에 시작됐다"로 읽힌다.
    """
    c = card(
        site="입안",
        onset="이번 달에 벌써 세 번째예요",
        time_course="한 번 생기면 열흘쯤 있다 아물어요",
    )
    assert _title(c) == "입안"  # 열흘이 붙지 않는다


def test_severity_slider_is_not_nrs():
    """1~5 서열척도 + 단계별 라벨이다. 우리가 오래 NRS로 잘못 알고 있었다(2026-09-11 정정)"""
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "medimate" / "dialog" / "questions.py"
    assert "NRS가 아니다" in src.read_text(encoding="utf-8")
