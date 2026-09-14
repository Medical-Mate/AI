"""카드 화면에 찍히는 축 이름. **한 줄에 들어가야 한다.**

우리는 부위 라벨(`body-map`의 `label`)은 주면서 축 라벨은 주지 않았다. 앱이 8개를 혼자 지었고
`"심해질 때, 나아질 때"`(9자)·`"같이있는 증상"`(6자)이 카드 레이아웃을 깨뜨렸다(2026-09-14).
"""

from medimate.dialog.questions import ASK_ORDER, CLARIFY, QUESTIONS
from medimate.schema.card import AXIS_LABEL_KO, AXIS_LABEL_MAX, Axis, axis_label_len


def test_every_axis_has_a_label():
    """축을 늘리고 라벨을 안 만들면 화면에 키가 그대로 찍힌다"""
    assert set(AXIS_LABEL_KO) == set(Axis)


def test_labels_fit_one_line():
    """길어져야 할 이유가 생기면 라벨이 아니라 화면 설계를 먼저 본다"""
    too_long = {a: v for a, v in AXIS_LABEL_KO.items() if axis_label_len(v) > AXIS_LABEL_MAX}
    assert too_long == {}, too_long


def test_labels_are_not_diagnoses_or_invented_facts():
    """라벨은 축 이름이지 내용이 아니다. 증상어·병명이 들어가면 없는 사실이 붙는다"""
    joined = " ".join(AXIS_LABEL_KO.values())
    for banned in ("통증", "염", "질환", "병", "진단", "의심"):
        assert banned not in joined, banned


def test_the_exacerbating_question_asks_only_one_thing():
    """카드 100장에서 **둘 다 답한 것이 3%**였다. 두 개를 묻고 하나를 받고 있었다.

    라벨은 `악화·완화`로 남긴다 — 환자가 자발적으로 말한 완화는 여전히 이 축에 담긴다.
    """
    q = QUESTIONS[Axis.EXACERBATING_RELIEVING]
    assert "괜찮아지" not in q and "나아지" not in q
    assert len(q) <= 20, q
    # 되묻기는 중립이라 완화가 나와도 받는다
    assert "그런지" in CLARIFY[Axis.EXACERBATING_RELIEVING]
    assert AXIS_LABEL_KO[Axis.EXACERBATING_RELIEVING] == "악화·완화"


def test_axis_key_did_not_change():
    """**키를 바꾸면 백엔드 컬럼·앱·진행 중이던 state가 전부 깨진다.** 질문만 좁혔다"""
    assert Axis.EXACERBATING_RELIEVING.value == "exacerbating_relieving"
    assert Axis.EXACERBATING_RELIEVING in ASK_ORDER
