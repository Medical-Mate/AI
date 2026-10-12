# ruff: noqa: E501
"""한국어 기간·상대 날짜 표현을 **조립해서** 세는 파서 (2026-10-12).

왜 표가 아니라 문법인가. 재방문 표현은 열거할 수 없는 집합이다 — `열사흘`(열+사흘), `스무이틀`(스무+이틀),
`그글피`(글피+1), `다다다음 주`(다×2 + 다음 주), `두 주 반`, `달포`. 낱말을 표에 넣으면 다음 환자가 그
다음 조합을 쓴다(메모 분리에서 2026-09-15 하루에 구멍 여섯). 그래서 **구성 요소와 결합 규칙**만 둔다:

    수사(고유어·한자어·숫자) × 단위(일·주·달·해) [+ 반]  |  고유어 날 이름(하루…아흐레, 열흘·보름·스무날 + 조합)
    |  상대 낱말(오늘·내일·모레·글피·그글피, 다[다…]음 주/달, 담주·담달, 달포·반달·반년·내년·후년)

날짜는 세지 않는다 — **며칠인지**만 낸다. 더하는 것은 호출자(`dialog/memo.followup_date`).
온디바이스 경로의 전제이기도 하다: 폰 1.7B는 표현을 자를 수는 있어도 고유어 수를 못 센다(실측 5/12).
세는 쪽은 언제나 코드다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ── 구성 요소 ──────────────────────────────────────────────────────────────

# 고유어 수사(관형형). 단위 앞에 선다: 한 주 · 두 달 · 열두 주 · 스무 날
_NATIVE_ONES = {
    "한": 1,
    "두": 2,
    "세": 3,
    "석": 3,
    "네": 4,
    "넉": 4,
    "다섯": 5,
    "여섯": 6,
    "일곱": 7,
    "여덟": 8,
    "아홉": 9,
}
_NATIVE_TENS = {"열": 10, "스무": 20, "스물": 20, "서른": 30, "마흔": 40, "쉰": 50}
# 한자어 수사. 단위 앞: 삼 일 · 오 주 · 십 개월. '일'(1)은 `일주일`·`일 년`에서만 받고 `일 일`은 안 본다
_SINO_ONES = {
    "일": 1,
    "이": 2,
    "삼": 3,
    "사": 4,
    "오": 5,
    "육": 6,
    "륙": 6,
    "칠": 7,
    "팔": 8,
    "구": 9,
}
_SINO_TENS = {"십": 10, "이십": 20, "삼십": 30, "사십": 40, "오십": 50}

# 고유어 날 이름. 그 자체가 일수다: 하루 · 이틀 · … · 열흘 · 보름 · 스무날
_DAY_NAMES = {
    "하루": 1,
    "이틀": 2,
    "사흘": 3,
    "나흘": 4,
    "닷새": 5,
    "엿새": 6,
    "이레": 7,
    "여드레": 8,
    "아흐레": 9,
    "열흘": 10,
    "보름": 15,
    "스무날": 20,
}
# 조합 날 이름: 열하루(11) … 열아흐레(19) · 스무하루(21) … 스무아흐레(29). `열흘`·`보름`은 위 표
_DAY_ONES = {k: v for k, v in _DAY_NAMES.items() if v < 10}

_UNIT_DAYS = {"일": 1, "날": 1, "주": 7, "주일": 7, "달": 30, "개월": 30, "해": 365, "년": 365}

# 상대 낱말. 그 자체로 며칠 뒤인지 정해진다
_RELATIVE = {
    "오늘": 0,
    "내일": 1,
    "내일모레": 2,
    "모레": 2,
    "글피": 3,
    "그글피": 4,
    "담주": 7,
    "다담주": 14,
    "담달": 30,
    "다담달": 60,
    "달포": 30,  # 한 달 남짓
    "반달": 15,
    "반년": 180,
    "내년": 365,
    "후년": 730,
    "내후년": 730,
}

_SP = r"\s*"
_NATIVE_NUM = rf"(?:(?:{'|'.join(_NATIVE_TENS)}){_SP}(?:{'|'.join(_NATIVE_ONES)})?|(?:{'|'.join(_NATIVE_ONES)}))"
_SINO_NUM = rf"(?:(?:{'|'.join(sorted(_SINO_TENS, key=len, reverse=True))}){_SP}(?:{'|'.join(_SINO_ONES)})?|(?:{'|'.join(_SINO_ONES)}))"
_UNIT = r"(?:주일|개월|주|일|날|달|해|년)"
_HALF = rf"(?:{_SP}반)?"
_DAY_NAME = rf"(?:(?:열|스무|스물){_SP}(?:{'|'.join(_DAY_ONES)})|{'|'.join(sorted(_DAY_NAMES, key=len, reverse=True))})"
_NEXT = rf"(?:다{_SP})+음{_SP}(?:주|달)"  # 다음 주 · 다다음 달 · 다다다음 주
_REL_WORD = "|".join(sorted(_RELATIVE, key=len, reverse=True))

# 한 패턴으로 모두 찾는다. 긴 것이 먼저 걸리도록 순서를 둔다(`내일모레` > `모레`, `그글피` > `글피`)
_PATTERN = re.compile(
    rf"(?P<num_digit>\d+){_SP}(?P<unit_d>{_UNIT}){_HALF}"
    rf"|(?P<num_native>{_NATIVE_NUM}){_SP}(?P<unit_n>{_UNIT}){_HALF}"
    rf"|(?P<num_sino>{_SINO_NUM}){_SP}(?P<unit_s>주일|개월|주|달|년|일)(?P<half_s>{_SP}반)?"
    rf"|(?P<next>{_NEXT})"
    rf"|(?P<rel>{_REL_WORD})"
    rf"|(?P<dayname>{_DAY_NAME})"
)
_HALF_TAIL = re.compile(rf"{_SP}반$")
_NEXT_DA = re.compile(r"다")

# `하루 두 번`·`하루 세 알`은 복용법이지 기간이 아니다. `하루`만 이 검사를 받는다
_DOSAGE_AFTER = re.compile(
    rf"^{_SP}(?:에{_SP})?(?:한|두|세|네|\d+){_SP}(?:번|차례|회|알|정|포|캡슐)"
)


@dataclass(frozen=True)
class Duration:
    text: str  # 원문에서 글자 그대로
    start: int
    end: int
    days: int
    # 시점 낱말(모레·담주·다다음 달·내년·오늘)은 그 자체로 "언제"다. 기간(열흘·두 주)은 "다시 온다"는 말이
    # 같이 있어야 재방문으로 읽는다 — 약 기간일 수 있어서. 호출자가 이 둘을 다르게 다룬다
    point: bool = False


def _native(num: str) -> int:
    s = re.sub(r"\s+", "", num)
    for t, tv in sorted(_NATIVE_TENS.items(), key=lambda kv: -len(kv[0])):
        if s.startswith(t):
            rest = s[len(t) :]
            return tv + (_NATIVE_ONES[rest] if rest else 0)
    return _NATIVE_ONES[s]


def _sino(num: str) -> int:
    s = re.sub(r"\s+", "", num)
    for t, tv in sorted(_SINO_TENS.items(), key=lambda kv: -len(kv[0])):
        if s.startswith(t):
            rest = s[len(t) :]
            return tv + (_SINO_ONES[rest] if rest else 0)
    return _SINO_ONES[s]


def _dayname(word: str) -> int:
    s = re.sub(r"\s+", "", word)
    if s in _DAY_NAMES:
        return _DAY_NAMES[s]
    for pre, pv in (("열", 10), ("스무", 20), ("스물", 20)):
        if s.startswith(pre) and s[len(pre) :] in _DAY_ONES:
            return pv + _DAY_ONES[s[len(pre) :]]
    raise ValueError(word)


def _with_half(days: int, unit: str, matched: str) -> int:
    if not _HALF_TAIL.search(matched):
        return days
    return days + _UNIT_DAYS[unit] // 2


def find_durations(text: str) -> list[Duration]:
    """문장 안의 기간·상대 날짜 표현을 전부, 나온 순서로. 각각 며칠인지와 원문 위치."""
    out: list[Duration] = []
    for m in _PATTERN.finditer(text):
        g = m.groupdict()
        raw = m.group(0)
        point = False
        try:
            if g["num_digit"] is not None:
                unit = g["unit_d"]
                days = _with_half(int(g["num_digit"]) * _UNIT_DAYS[unit], unit, raw)
            elif g["num_native"] is not None:
                unit = g["unit_n"]
                days = _with_half(_native(g["num_native"]) * _UNIT_DAYS[unit], unit, raw)
            elif g["num_sino"] is not None:
                unit = g["unit_s"]
                n = _sino(g["num_sino"])
                if unit == "일" and n == 1:
                    continue  # `일 일`은 안 본다 — `일주일`은 unit=주일로 잡힌다
                days = _with_half(n * _UNIT_DAYS[unit], unit, raw)
            elif g["next"] is not None:
                k = len(_NEXT_DA.findall(raw))  # 다음=1, 다다음=2, …
                days = k * (7 if raw.endswith("주") else 30)
                point = True
            elif g["rel"] is not None:
                days = _RELATIVE[re.sub(r"\s+", "", raw)]
                point = True
            else:
                days = _dayname(raw)
        except (KeyError, ValueError):
            continue
        if raw == "하루" and _DOSAGE_AFTER.match(text[m.end() :]):
            continue
        out.append(Duration(text=raw, start=m.start(), end=m.end(), days=days, point=point))
    return out
