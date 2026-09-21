"""채팅 정규화 — 표현만, 의미는 아니다. 원문 보존."""

import pytest

from medimate.text.chatnorm import normalize


@pytest.mark.parametrize(
    "raw, want",
    [
        ("ㄱㅊ", "괜찮다"),
        ("ㅇㅇ ㄱㅊ", "응 괜찮다"),
        ("ㄴㄴ", "아니"),
        ("ㅁㄹ 기억안남", "모르겠다 기억 안 남"),
        ("몰겠음", "모르겠음"),
        ("아픈듯 ㅠㅠ", "아픈 듯"),
        ("괜찬대요", "괜찮대요"),
        ("위염이래요 ㅋㅋ", "위염이래요"),
        ("괜찮아요~~", "괜찮아요"),
    ],
)
def test_normalize(raw, want):
    n = normalize(raw)
    assert n.text == want
    assert n.raw == raw  # 원문은 언제나 그대로


def test_unknown_left_alone():
    s = "위염이래요. 위산약 2주치 받고, 3주 후에 재방문 하래요"
    n = normalize(s)
    assert n.text == s and not n.changed


def test_ops_point_into_raw():
    raw = "열은 ㄴㄴ 기억안남"
    n = normalize(raw)
    assert [(raw[o.start : o.end], o.after) for o in n.ops] == [
        ("ㄴㄴ", "아니"),
        ("기억안남", "기억 안 남"),
    ]


def test_no_meaning_added():
    # `ㄱㅊ`을 `통증 없음`으로 바꾸지 않는다 — 의미는 뒤 층
    assert "통증" not in normalize("ㄱㅊ").text
    # 문어체 메모 어미 `-음/-함`은 그대로
    assert normalize("스케일링만 했음").text == "스케일링만 했음"
