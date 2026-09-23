"""값 검증기(span/verify.py) — 통과해야 하는 값 모양과 걸려야 하는 값. 호출 0."""

import pytest

pytest.importorskip("kiwipiepy")

from medimate.span.verify import verify_value  # noqa: E402
from medimate.text.lexicon import load_lexicon  # noqa: E402

LEX = load_lexicon()


@pytest.mark.parametrize(
    "value,segment,axis",
    [
        # 재배열·조사 떼기·-음 명사형
        (
            "갑상선 수치가 조금 흔들림",
            "오늘 혈액검사에서 갑상선 수치가 조금 흔들렸지만 급한 문제는 아니라고 하셨어요.",
            "findings",
        ),
        ("아침 공복에 약", "약은 아침 공복에 먹고", "medication_instructions"),
        ("6주 후 검사", "6주 뒤 다시 검사하기로 했어요.", "tests"),
        ("엑스레이", "엑스레이를 찍었고 결과는 다음 진료 때 설명해 주신다고 했음.", "tests"),
        (
            "열이 나거나 붓기가 커지면 재방문 필요",
            "열이 나거나 붓기가 커지면 바로 오라고 했어요.",
            "follow_up",
        ),
        ("이번 주까지 소독", "이번 주까지만 소독하고", "medication_instructions"),
        ("추후 피검사 결과 안내", "피검사 결과는 다음에 알려 준다고 했어요.", "tests"),
        ("악화 시 재방문 필요", "심해지면 다시 오라고 했어요.", "follow_up"),
        ("석 달 후 피검사", "석달뒤피검사다시하기", "tests"),
        ("물이 조금 찼음", "관절초음파에서물이조금찼다함", "findings"),
        ("검사 결과 이상 없음", "보건소검사결과이상없다함", "tests"),
        ("사진은 괜찮게 나왔음", "흉부외과에서 찍은 사진은 괜찬게 나왓데요.", "tests"),
        (
            "속쓰림 안 나으면 월요일 재방문 필요",
            "속쓰림 안 낫으면 월욜에 다시 오래요.",
            "follow_up",
        ),
        ("다음 주", "담주에 또 오재.", "follow_up"),
        ("위염 초기", "위염 초기라고 하셨어요", "findings"),
    ],
)
def test_passes(value, segment, axis):
    v = verify_value(value, segment, axis, LEX)
    assert v.ok, v.reasons


@pytest.mark.parametrize(
    "value,segment,axis,reason",
    [
        ("인대 손상", "무릎 인대가 늘어난 것 같다고 했어요", "findings", "unsourced"),
        ("항생제 7일", "항생제 5일 처방받았어요", "medication_instructions", "digits"),
        ("안 퍼짐", "ㅇㅇ 퍼졌다고 했음", "findings", "negation_added"),
        ("위염", "속이 쓰리다고 했어요", "findings", "unsourced"),
        ("피견", "피곤하다고 했어요", "findings", "unsourced"),
        ("혈압이 높음", "혈압은 정상이래요", "findings", "unsourced"),
    ],
)
def test_rejects(value, segment, axis, reason):
    v = verify_value(value, segment, axis, LEX)
    assert not v.ok
    assert any(r.startswith(reason) for r in v.reasons), v.reasons


def test_none_passes():
    assert verify_value(None, "괜찮대요", "findings").ok
    assert verify_value("", "괜찮대요", "findings").ok
