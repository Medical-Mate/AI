# ruff: noqa: E501
"""한국어 기간·상대 날짜 표현을 **조립해서** 세는 파서 (2026-10-12).

왜 표가 아니라 문법인가. 재방문 표현은 열거할 수 없는 집합이다 — `열사흘`(열+사흘), `스무이틀`(스무+이틀),
`그글피`(글피+1), `다다다음 주`(다×2 + 다음 주), `두 주 반`, `달포`. 낱말을 표에 넣으면 다음 환자가 그
다음 조합을 쓴다(메모 분리에서 2026-09-15 하루에 구멍 여섯). 그래서 **구성 요소와 결합 규칙**만 둔다:

    수사(고유어·한자어·숫자) × 단위(일·주·달·해) [+ 반]  |  고유어 날 이름(하루…아흐레, 열흘·보름·스무날 + 조합)
    |  상대 낱말(오늘·내일·모레·글피·그글피, 다[다…]음 주/달, 담주·담달, 달포·반달·반년·내년·후년)

**달과 해는 일수로 곱하지 않는다.** 9월 15일의 "두 달 뒤"는 11월 15일이지 60일 뒤(11월 14일)가 아니다.
그래서 `Duration`은 `days`(일·주 몫)와 `months`(달·해 몫)를 따로 들고, 날짜에 더할 때 달력으로 더한다
(1월 31일 + 1달 = 2월 28일). 날짜 산수는 여기(코드)다. 폰 1.7B·서버 LLM은 세지 않는다 — 폰은 표현을 자를
수는 있어도 고유어 수를 못 센다(실측 13/39).
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, timedelta

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

# 단위 → (일, 달). 달·해는 달력으로 더하므로 일수가 아니라 달 수다
_UNITS: dict[str, tuple[int, int]] = {
    "일": (1, 0),
    "날": (1, 0),
    "주": (7, 0),
    "주일": (7, 0),
    "달": (0, 1),
    "개월": (0, 1),
    "해": (0, 12),
    "년": (0, 12),
}

# 상대 낱말 → (일, 달). 그 자체로 "언제"가 정해진다
_RELATIVE: dict[str, tuple[int, int]] = {
    "오늘": (0, 0),
    "내일": (1, 0),
    "내일모레": (2, 0),
    "모레": (2, 0),
    "글피": (3, 0),
    "그글피": (4, 0),
    "담주": (7, 0),
    "다담주": (14, 0),
    "담달": (0, 1),
    "다담달": (0, 2),
    "달포": (0, 1),  # 한 달 남짓
    "반달": (15, 0),
    "반년": (0, 6),
    "내년": (0, 12),
    "후년": (0, 24),
    "내후년": (0, 24),
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
    rf"|(?P<num_sino>{_SINO_NUM}){_SP}(?P<unit_s>주일|개월|주|달|년|일){_HALF}"
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
    days: int  # 일·주 몫 (정확한 일수)
    months: int = 0  # 달·해 몫 (달력으로 더한다)
    # 시점 낱말(모레·담주·다다음 달·내년·오늘)은 그 자체로 "언제"다. 기간(열흘·두 주)은 "다시 온다"는 말이
    # 같이 있어야 재방문으로 읽는다 — 약 기간일 수 있어서. 호출자가 이 둘을 다르게 다룬다
    point: bool = False

    @property
    def is_zero(self) -> bool:
        return self.days == 0 and self.months == 0

    @property
    def approx_days(self) -> int:
        """어림 일수(달=30일). 비교·정렬용이지 날짜 계산에 쓰지 않는다."""
        return self.months * 30 + self.days

    @property
    def label(self) -> str:
        """basis에 남기는 표기: `14d` · `2개월` · `1개월 15d`."""
        parts = []
        if self.months:
            parts.append(f"{self.months}개월")
        if self.days or not parts:
            parts.append(f"{self.days}d")
        return " ".join(parts)

    def add_to(self, start: date) -> date:
        return add_months(start, self.months) + timedelta(days=self.days)


def add_months(start: date, months: int) -> date:
    """달력으로 달을 더한다. 같은 날이 없으면 그 달의 마지막 날(1월 31일 + 1달 = 2월 28일)."""
    if not months:
        return start
    y, m0 = divmod(start.month - 1 + months, 12)
    y += start.year
    m = m0 + 1
    d = min(start.day, calendar.monthrange(y, m)[1])
    return date(y, m, d)


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


def _times_unit(n: int, unit: str, matched: str) -> tuple[int, int]:
    """수사 × 단위 (+ 반) → (일, 달). 반은 단위의 절반: 주 반=+3일, 달 반=+15일, 해 반=+6달."""
    ud, um = _UNITS[unit]
    days, months = n * ud, n * um
    if _HALF_TAIL.search(matched):
        if um == 12:
            months += 6
        elif um == 1:
            days += 15
        else:
            days += ud // 2
    return days, months


def find_durations(text: str) -> list[Duration]:
    """문장 안의 기간·상대 날짜 표현을 전부, 나온 순서로. 각각 (일, 달)과 원문 위치."""
    out: list[Duration] = []
    for m in _PATTERN.finditer(text):
        g = m.groupdict()
        raw = m.group(0)
        point = False
        try:
            if g["num_digit"] is not None:
                days, months = _times_unit(int(g["num_digit"]), g["unit_d"], raw)
            elif g["num_native"] is not None:
                days, months = _times_unit(_native(g["num_native"]), g["unit_n"], raw)
            elif g["num_sino"] is not None:
                n, unit = _sino(g["num_sino"]), g["unit_s"]
                if unit == "일" and n == 1:
                    continue  # `일 일`은 안 본다 — `일주일`은 unit=주일로 잡힌다
                days, months = _times_unit(n, unit, raw)
            elif g["next"] is not None:
                k = len(_NEXT_DA.findall(raw))  # 다음=1, 다다음=2, …
                days, months = (7 * k, 0) if raw.endswith("주") else (0, k)
                point = True
            elif g["rel"] is not None:
                days, months = _RELATIVE[re.sub(r"\s+", "", raw)]
                point = True
            else:
                days, months = _dayname(raw), 0
        except (KeyError, ValueError):
            continue
        if raw == "하루" and _DOSAGE_AFTER.match(text[m.end() :]):
            continue
        out.append(
            Duration(text=raw, start=m.start(), end=m.end(), days=days, months=months, point=point)
        )
    return out
