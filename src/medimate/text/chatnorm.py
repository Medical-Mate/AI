"""채팅 한국어 정규화 — 표현만 되살린다. 의미는 붙이지 않는다.

`ㄱㅊ → 괜찮다`까지만 한다. `통증 없음`으로 바꾸지 않는다 — 그 판단은 질문 문맥과 함께 선택기가 한다
(docs/candidate-selection.md §3.3). 표에 없는 것은 그대로 둔다. 원문은 언제나 같이 돌려준다.

출력 `Normalized(raw, text, ops)`. `ops`는 (원문 좌표, 원래 조각, 바뀐 조각) 목록.
어디를 건드렸는지 남긴다. 후보의 좌표는 원문 기준이므로 정규화문은 **보조 입력**이다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# 초성 축약·구어 표기 → 표준 표기. 한 표에 다 둔다. 의미(예/아니오)는 여기 없다 — `ㅇㅇ`은 `응`이다
ABBREV: dict[str, str] = {
    "ㅇㅇ": "응",
    "ㄴㄴ": "아니",
    "ㄱㅊ": "괜찮다",
    "ㅁㄹ": "모르겠다",
    "ㅇㅋ": "오케이",
    "ㄳ": "감사",
    "ㄱㅅ": "감사",
    "ㅈㅅ": "죄송",
    "ㅊㅊ": "추천",
    "ㅇㄷ": "어디",
    "ㅁㅊ": "미친",  # 감탄으로 쓰인다. 의미 판단은 뒤 층
}

# 초성이 어미에 붙은 형 — `ㄱㅊ대요`·`ㄱㅊ다고`. 어간형으로 되살린다(날것 메모 검토 2026-09-21)
STEM_ABBREV: dict[str, str] = {
    "ㄱㅊ": "괜찮",
    "ㅁㄹ": "모르",
    "ㄴㄴ": "아니",
}
_STEM_RX = re.compile(r"(ㄱㅊ|ㅁㄹ|ㄴㄴ)(?=[가-힣])")

# 표현 표지 — 지운다. 원문에는 남는다. `ㅠ` 하나도 표지다(`찍어보재 ㅠ`)
MARKERS = re.compile(
    r"[ㅋㅎㅠㅜㄷ]{2,}|[ㅋㅎ]+(?=\s|$)|[ㅠㅜ]+(?=\s|$)|[!?.~]{2,}|(?<=\s)[ㅠㅜ](?=\s|$)"
)

# 붙여 쓴 구어 → 띄어 쓴 표준형. 어미 `-남/-음/-함`은 그대로 둔다(문어체 메모의 정상 어미다)
CASUAL: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"기억안남"), "기억 안 남"),
    (re.compile(r"기억안나"), "기억 안 나"),
    (re.compile(r"몰겠(음|어|다|는데)"), r"모르겠\1"),
    (re.compile(r"몰라"), "몰라"),
    (re.compile(r"(?<![가-힣])아픈듯"), "아픈 듯"),
    (re.compile(r"괜찬"), "괜찮"),  # PM10·PM15 실데이터 오타. 의미 아니라 표기
    # 어미 오타 — 메신저 표기. E2 파생 그룹(derive_span_groups.TYPOS)과 짝
    (re.compile(r"(?<=[가-힣])레요"), "래요"),
    (re.compile(r"(?<=[가-힣])어여(?![가-힣])"), "어요"),
    (re.compile(r"햇(?=[어다고])"), "했"),
    (re.compile(r"(?<=[가-힣])이라구"), "이라고"),
    (re.compile(r"(?<=[가-힣])다구(?![가-힣])"), "다고"),
    (re.compile(r"됬"), "됐"),
    (re.compile(r"안됨"), "안 됨"),
    (re.compile(r"안함"), "안 함"),
    (re.compile(r"안먹"), "안 먹"),
    # ── 날것 메모 검토(2026-09-21)에서 나온 표현. 전부 표기 복원이고 뜻은 안 바꾼다 ──
    # 구어 축약: 시점
    (re.compile(r"(?<![가-힣])담주"), "다음 주"),
    (re.compile(r"(?<![가-힣])담달"), "다음 달"),
    (re.compile(r"(?<![가-힣])담에"), "다음에"),
    (re.compile(r"(?<![가-힣])(월|화|수|목|금|토|일)욜"), r"\1요일"),
    (re.compile(r"(?<![가-힣])전이랑"), "이전과"),
    # 구어 인용 어미 → 표준 인용형. `-다네`·`-다함`·`-다구`·`-랬음`·`-랜다`·`-댔음`·`-라구`
    (re.compile(r"(?<=[가-힣])다네(?![가-힣])"), "다고"),
    (re.compile(r"(?<=[가-힣])다함(?![가-힣])"), "다고 함"),
    (re.compile(r"(?<=[가-힣])다구(?![가-힣])"), "다고"),
    (re.compile(r"(?<=[가-힣])라구(?![가-힣])"), "라고"),
    (re.compile(r"(?<=[가-힣])랬음(?![가-힣])"), "라고 했음"),
    (re.compile(r"(?<=[가-힣])랬어(?![가-힣])"), "라고 했어"),
    (re.compile(r"(?<=[가-힣])랜다(?![가-힣])"), "라고 한다"),
    (re.compile(r"(?<=[가-힣])댔음(?![가-힣])"), "다고 했음"),
    (re.compile(r"(?<=[가-힣])댓음(?![가-힣])"), "다고 했음"),
    (re.compile(r"(?<=[가-힣])하대(?![가-힣])"), "하다고"),
    (re.compile(r"(?<=[가-힣])괜찮대(?![가-힣])"), "괜찮다고"),
    # 청유 구어 `-재`: 보재·재보재·찍어보재 → 보자고
    (re.compile(r"(?<=[가-힣])보재(?![가-힣])"), "보자고"),
    (re.compile(r"(?<=[가-힣])하재(?![가-힣])"), "하자고"),
    # 초성 `ㄱㄱ`(가자) — 표기만
    (re.compile(r"(?<![가-힣])ㄱㄱ(?![가-힣])"), "가기"),
]

_TOKEN = re.compile(r"\S+")


@dataclass(frozen=True)
class Op:
    start: int
    end: int
    before: str
    after: str


@dataclass(frozen=True)
class Normalized:
    raw: str
    text: str
    ops: list[Op] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.ops)


def normalize(raw: str) -> Normalized:
    ops: list[Op] = []
    out: list[str] = []
    pos = 0
    for m in _TOKEN.finditer(raw):
        out.append(raw[pos : m.start()])
        tok = m.group(0)
        new = tok
        if tok in ABBREV:
            new = ABBREV[tok]
        else:
            stripped = MARKERS.sub("", tok)
            if stripped != tok:
                new = stripped
            new = _STEM_RX.sub(lambda m: STEM_ABBREV[m.group(1)], new)
            for rx, repl in CASUAL:
                new = rx.sub(repl, new)
        if new != tok:
            ops.append(Op(m.start(), m.end(), tok, new))
        out.append(new)
        pos = m.end()
    out.append(raw[pos:])
    text = re.sub(r"[ \t]{2,}", " ", "".join(out)).strip()
    return Normalized(raw, text, ops)
