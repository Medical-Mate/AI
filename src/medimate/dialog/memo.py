"""진료 후 메모 → 4묶음 분류 (챗봇② B1, 와이어프레임 1p → 1q).

흐름
1. 메모를 문장으로 나눈다 (결정론. 마침표·물음표·줄바꿈·"· ")
2. LLM은 **문장마다 라벨 하나**만 고른다:
   findings / tests / medication_instructions / follow_up / none
   (다축 추출이 아니라 단일 선택이라 소형 모델도 안정적이고,
   값이 문장 원문이라 근거 검증이 자동 통과)
3. 카드의 각 묶음에 그 문장들을 **원문 그대로** 넣는다. none은 unsorted로 보존한다
4. 재방문 날짜는 진료일 기준 결정론 계산(followup_date). LLM 무관

가드: 라벨 인덱스가 범위 밖이거나 중복이면 무시. 문장은 우리가 나눈 것이라 모델이 바꿀 수 없다.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from medimate.schema.card import AxisEntry, FieldStatus
from medimate.schema.postvisit import FollowUpDate, PostAxis, PostVisitCard

LABELS = tuple(a.value for a in PostAxis) + ("none",)


Label = Literal["findings", "tests", "medication_instructions", "follow_up", "none"]


class MemoLabels(BaseModel):
    """LLM 출력 스키마. **문장 순서대로** 라벨 하나씩.

    번호를 모델에게 맡기지 않는다(v1·v2에서 번호가 서로 바뀌는 오류가 있었다).
    길이는 요청별 스키마(minItems=maxItems=N)로 강제한다.
    """

    model_config = ConfigDict(extra="forbid")

    labels: list[Label] = Field(default_factory=list)

    @classmethod
    def schema_for(cls, n: int) -> dict:
        s = cls.model_json_schema()
        s["properties"]["labels"]["minItems"] = n
        s["properties"]["labels"]["maxItems"] = n
        return s

    @staticmethod
    def keyed_schema_for(n: int) -> dict:
        """LLM에게 주는 출력 스키마: {"0": 라벨, "1": 라벨, …} — 번호 키가 전부 필수.

        v3의 위치 배열은 문장 앵커가 없어 흔들렸고(64%),
        v2의 [{i,label}]은 번호가 뒤바뀌었다(83%).
        번호 키 객체는 둘의 장점만 남긴다: 문장마다 앵커가 있고, 누락·중복·뒤바뀜이 문법에서
        불가능하다.
        """
        labels = list(Label.__args__)  # type: ignore[attr-defined]
        return {
            "type": "object",
            "properties": {str(i): {"type": "string", "enum": labels} for i in range(n)},
            "required": [str(i) for i in range(n)],
            "additionalProperties": False,
        }

    @classmethod
    def from_keyed(cls, obj: dict, n: int) -> MemoLabels:
        """번호 키 객체 → 위치 배열. 없는 번호는 none."""
        return cls(labels=[obj.get(str(i), "none") for i in range(n)])


class MemoClassifier(Protocol):
    model_id: str
    prompt_version: str

    def classify(self, sentences: Sequence[str]) -> MemoLabels: ...


_SPLIT = re.compile(r"(?<=[.!?。])\s+|\n+|\s+·\s+|\s*/\s*")


# ── 연결절 분리 (2026-09-14, #78) ──────────────────────────────────────────
#
# 한 문장에 두 주제가 오면 라벨이 하나뿐이라 한쪽이 사라진다. 백엔드가 신고한 문장:
#   "나프록센 500mg 먹으라고 하셨고, 다음 주 화요일에 MRI 찍기로 했어요"  → 약만 남고 MRI 유실
# 우리 eval 벡터에도 이미 있었다:
#   PM20 "피부염이라고 하셨고 연고 처방받았어요."  → findings 하나. 연고 처방이 사라진다
#
# **연결어미를 전부 자르면 안 된다.** 같은 벡터에 쪼개면 안 되는 문장이 같은 모양으로 있다:
#   PM01 "혈액검사 했고 결과는 다음에 알려준대요."      양쪽 다 검사
#   PM01 "약은 2주분이고 커피랑 매운 거 줄이래요."      양쪽 다 약·생활 지시(4축 접기)
#   PM13 "아무 말도 안 하시고 약만 주셨어요."          앞은 주제가 아니다
#   PN06 "한 달 후에 피검사 다시 하고 결과 보자고."     양쪽 다 검사
# 기계적으로 자르면 "아무 말도 안 하시고" 같은 조각이 생기고, 그건 라벨이 없는 문장이 되어
# `unsorted`로 떨어진다. 문장 수만 늘고 카드는 나빠진다.
#
# 그래서 **자르는 조건을 주제 신호로 건다**: 양쪽이 각각 주제를 갖고, 그 주제가 서로 다를 때만.
# 한쪽이라도 신호가 없으면 안 자른다(모르면 두는 쪽). 겹치면 안 자른다(같은 묶음으로 갈 말이다).
#
# 분류기가 아니라 **분리기의 판단이다.** 어느 묶음인지는 여전히 모델이 정한다 — 여기서 보는
# 것은 "두 조각이 서로 다른 묶음으로 갈 만한가"뿐이고, 틀려도 최악이 문장 하나를 안 나눈
# 지금 상태다.
_TOPIC = {
    "tests": (
        "검사",
        "촬영",
        "찍",
        "엑스레이",
        "엑스선",
        "mri",
        "ct",
        "초음파",
        "내시경",
        "피검사",
        "혈액",
        "소변",
        "결과",
        "판독",
        "조직",
    ),
    "medication": (
        "약",
        "처방",
        "연고",
        "주사",
        "복용",
        "먹",
        "바르",
        "파스",
        "드시",
        "물리치료",
        # `술`은 넣지 않는다 — `수술`이 걸린다. 회귀 실행에서 "결과 보고 / 수술 여부 정한다고"를
        # 쪼갰다(PM07). 술은 `금주`·`음주`로만 본다
        "줄이",
        "피하",
        "끊",
        "금주",
        "음주",
        "커피",
        "짜게",
        "매운",
    ),
    "follow_up": (
        "다시 오",
        "다시 와",
        "재방문",
        "경과 보",
        "예약",
        "다음 주",
        "다음 달",
        "뒤에 오",
        "후에 오",
    ),
    # findings 어휘는 **좁게 간다.** 처음에 "염"을 넣었더니 `소염제`가 findings로 잡혀
    # "소염제 처방받았고 2주 뒤에 오래요"가 안 갈렸다. "래요"도 `오래요`·`줄이래요`에 걸린다.
    # 신호가 넓으면 양쪽이 겹쳐서 **안 자르는 쪽으로 틀린다** — 조용해서 더 안 보인다.
    "findings": (
        "라고 하",
        "소견",
        "정상",
        "괜찮",
        "골절",
        "뭉친",
        "수치",
        "염증",
        "초기",
        "이상 없",
        "문제 없",
        "늘어났",
    ),
}

# 자를 수 있는 자리 — 연결어미 `-고`·`-며` 뒤. 쉼표가 붙어도 같다.
# `-라고/-다고/-자고/-냐고`는 인용이라 제외한다("오라고 하셨고"에서 앞쪽 `오라고`를 자르면
# "다시 오라고" + "하셨고 …"가 되어 술어가 잘린다). "그리고"도 같은 이유로 제외.
_CLAUSE = re.compile(r"(?<=[가-힣])(고|며),?\s+")
_QUOTATIVE = ("라고", "다고", "자고", "냐고", "리고", "으라고")


def _topics(text: str) -> set[str]:
    low = text.lower()
    return {name for name, words in _TOPIC.items() if any(w in low for w in words)}


def _split_clauses(sentence: str) -> list[str]:
    """한 문장을 연결어미에서 나눈다. **주제가 갈릴 때만.**

    조건을 다 만족해야 자른다:
      1. 연결어미 `-고`·`-며` 뒤다(인용 `-라고/-다고/-자고/-냐고`와 "그리고"는 제외)
      2. 양쪽 조각이 각각 주제 신호를 갖는다 — 하나라도 없으면 안 자른다
      3. 두 주제가 서로 겹치지 않는다 — 겹치면 같은 묶음으로 갈 말이다
      4. 양쪽이 4자 이상이다 — 조각만 남기지 않는다
    """
    for m in _CLAUSE.finditer(sentence):
        head, tail = sentence[: m.start()], sentence[m.end() :]
        head_full = head + m.group(1)  # 자를 때 연결어미는 앞쪽에 남긴다(원문 보존)
        # 인용은 자르지 않는다 — 연결어미까지 붙여서 봐야 `먹으라고`가 걸린다
        if any(head_full.endswith(q) for q in _QUOTATIVE):
            continue
        if len(head_full) < 4 or len(tail) < 4:
            continue
        ht, tt = _topics(head_full), _topics(tail)
        if not ht or not tt or ht & tt:
            continue
        # 뒤쪽도 다시 본다 — 한 문장에 세 주제가 오는 일이 있다
        return [head_full.strip()] + _split_clauses(tail.strip())
    return [sentence]


# 분리 규칙의 이름. **규칙을 바꾸면 이 값도 바꾼다.**
#
# 왜 필요한가. `sentences`의 번호는 화면 1q-2가 라벨을 가리키는 주소다. 규칙이 바뀌면 같은
# 메모가 다른 개수·다른 번호로 나뉘는데, 배포 순간 1p와 1q-2 사이에 떠 있던 세션은 **예전
# 번호로 매긴 `labels`를 새 문장에 붙인다.** 200이 나가고 카드도 멀쩡해 보이는데 내용이
# 어긋난다. 그래서 값을 실어 보내고 다르면 409로 끊는다(`api/app.py`).
#
# 사람이 읽는 이름으로 둔 이유: 이 값이 백엔드 로그와 409 본문에 그대로 찍힌다. 정규식
# 해시를 쓰면 자동으로 따라 바뀌지만 `a3f1c2`가 로그에 남아도 아무도 못 읽는다. 대신
# **이름은 그대로 두고 규칙만 고치는 일**을 테스트가 막는다 — 아래 지문이 그 장치다
# (`tests/test_memo_split_version.py`). 규칙을 고치면 테스트가 깨지고, 그때 둘 다 고치게 된다.
SPLIT_VERSION = "split-v2"

# 분리 규칙 전체의 지문. 규칙과 이름이 같이 움직이는지 테스트가 이걸로 확인한다.
# 규칙을 바꿨으면 `SPLIT_VERSION`을 올리고 이 값도 새로 박는다(테스트 실패 메시지가 새 값을 준다)
SPLIT_RULE_DIGEST = "3882206e6d2e"


def split_rule_digest() -> str:
    """현재 분리 규칙의 지문. 상수와 비교해 규칙만 바뀐 상태를 잡는다.

    **문장 경계를 바꿀 수 있는 것을 전부 넣는다.** 정규식만 해싱하면 주제 사전에 단어를
    하나 더한 날 번호가 안 움직인다 — 그때 나뉘는 문장이 달라지는데 409가 안 난다.
    """
    material = [
        _SPLIT.pattern,
        _CLAUSE.pattern,
        repr(_QUOTATIVE),
        repr(sorted((k, tuple(sorted(v))) for k, v in _TOPIC.items())),
    ]
    return hashlib.sha256("\x1f".join(material).encode("utf-8")).hexdigest()[:12]


def split_sentences(memo: str) -> list[str]:
    """결정론 문장 분리.

    마침표·물음표·느낌표 뒤 공백, 줄바꿈, ' · ', '/' 에서 나눈다. 빈 조각은 버린다.

    **바꿀 때는 `SPLIT_VERSION`도 같이 올린다.** 번호가 곧 라벨의 주소라서, 규칙이 바뀌면
    이전 번호로 매긴 라벨이 엉뚱한 문장을 가리킨다.
    """
    parts = [p.strip() for p in _SPLIT.split(memo.strip()) if p and p.strip()]
    # 한 문장 안에 두 주제가 붙어 있으면 한 번 더 나눈다(라벨이 문장당 하나라 한쪽이 사라진다)
    out: list[str] = []
    for part in parts:
        out.extend(c for c in _split_clauses(part) if c)
    # 너무 짧은 조각("네", "음")도 문장으로 둔다 — 버리는 건 모델이 아니라 규칙이 정한다
    return out


_REL = [
    (re.compile(r"(\d+)\s*주\s*(뒤|후|있다가|지나서)"), lambda m: timedelta(weeks=int(m.group(1)))),
    (re.compile(r"(\d+)\s*일\s*(뒤|후|있다가|지나서)"), lambda m: timedelta(days=int(m.group(1)))),
    (
        re.compile(r"(\d+)\s*(개월|달)\s*(뒤|후|있다가|지나서)"),
        lambda m: timedelta(days=30 * int(m.group(1))),
    ),
    (re.compile(r"다음\s*주"), lambda m: timedelta(weeks=1)),
    (
        re.compile(r"(한|두|세|네)\s*주\s*(뒤|후)"),
        lambda m: timedelta(weeks={"한": 1, "두": 2, "세": 3, "네": 4}[m.group(1)]),
    ),
    (
        re.compile(r"(한|두|세)\s*달\s*(뒤|후)"),
        lambda m: timedelta(days=30 * {"한": 1, "두": 2, "세": 3}[m.group(1)]),
    ),
]
_ABS = re.compile(r"(\d{1,2})\s*월\s*(\d{1,2})\s*일")
_NEXT_MONTH_DAY = re.compile(r"다음\s*달\s*(\d{1,2})\s*일")


def followup_date(text: str, visit_date: date) -> FollowUpDate | None:
    """재방문 문장에서 날짜를 결정론으로. 못 읽으면 None(앱이 달력에서 직접 고른다)."""
    m = _ABS.search(text)
    if m:
        mo, d = int(m.group(1)), int(m.group(2))
        yr = visit_date.year + (1 if (mo, d) < (visit_date.month, visit_date.day) else 0)
        try:
            dt = date(yr, mo, d)
        except ValueError:
            return None
        return FollowUpDate(
            text=m.group(0), date=dt.isoformat(), approximate=False, basis="절대 날짜"
        )
    m = _NEXT_MONTH_DAY.search(text)
    if m:
        mo = visit_date.month % 12 + 1
        yr = visit_date.year + (1 if mo == 1 else 0)
        try:
            dt = date(yr, mo, int(m.group(1)))
        except ValueError:
            return None
        return FollowUpDate(
            text=m.group(0), date=dt.isoformat(), approximate=False, basis="다음 달 + 일자"
        )
    for rx, fn in _REL:
        m = rx.search(text)
        if m:
            delta = fn(m)
            dt = visit_date + delta
            return FollowUpDate(
                text=m.group(0),
                date=dt.isoformat(),
                approximate=True,
                basis=f"visit_date {visit_date.isoformat()} + {delta.days}d",
            )
    return None


@dataclass
class MemoResult:
    card: PostVisitCard
    sentences: list[str]
    labels: dict[int, str]
    dropped: list[dict] = field(default_factory=list)  # 가드가 무시한 라벨


def classify_memo(
    memo: str,
    classifier: MemoClassifier,
    visit_date: date | None = None,
    clinic: str | None = None,
    card: PostVisitCard | None = None,
) -> MemoResult:
    """메모 하나를 4묶음 카드로. 문장 원문 보존, 라벨 없는 문장은 unsorted."""
    card = card or PostVisitCard()
    card.memo = memo
    card.clinic = clinic
    card.visit_date = visit_date.isoformat() if visit_date else card.visit_date
    sentences = split_sentences(memo)
    if not sentences:
        return MemoResult(card, [], {})

    raw = classifier.classify(sentences)
    labels: dict[int, str] = {}
    dropped: list[dict] = []
    for i, lab in enumerate(raw.labels):
        if i >= len(sentences):
            dropped.append({"i": i, "label": lab, "reason": "extra_label"})
            continue
        labels[i] = lab
    for i in range(len(raw.labels), len(sentences)):
        dropped.append({"i": i, "label": None, "reason": "missing_label"})  # none으로 처리된다

    buckets: dict[PostAxis, list[str]] = {a: [] for a in PostAxis}
    unsorted: list[str] = []
    for i, s in enumerate(sentences):
        lab = labels.get(i, "none")
        if lab == "none":
            unsorted.append(s)
        else:
            buckets[PostAxis(lab)].append(s)

    for axis, sents in buckets.items():
        entry: AxisEntry = card.axes[axis]
        if sents:
            entry.status = FieldStatus.FILLED
            entry.value = " · ".join(sents)  # 원문을 잇기만 한다. 고치지 않는다
            entry.evidence = list(sents)
        else:
            entry.status = FieldStatus.UNKNOWN  # 메모에 그 묶음 얘기가 없었다
            entry.value = None
            entry.evidence = []
    card.unsorted = unsorted

    if buckets[PostAxis.FOLLOW_UP] and visit_date:
        for s in buckets[PostAxis.FOLLOW_UP]:
            fu = followup_date(s, visit_date)
            if fu:
                card.follow_up_date = fu
                break
    return MemoResult(card, sentences, labels, dropped)
