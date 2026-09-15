"""진료 후 메모 → 4묶음 분류 (챗봇② B1, 와이어프레임 1p → 1q).

흐름
1. 메모를 문장으로 나눈다 (결정론. 마침표·물음표·줄바꿈·"· ")
2. LLM은 **문장마다 라벨 하나**만 고른다:
   findings / tests / medication_instructions / follow_up / none
   (다축 추출이 아니라 단일 선택이라 소형 모델도 안정적이고,
   값이 문장 원문이라 근거 검증이 자동 통과)
3. 카드의 각 묶음에 넣는다 — `value`는 어미를 정리한 줄, `evidence`는 **문장 원문 그대로**.
   none은 unsorted로 보존한다 (진료 전 카드가 이미 이 구조다)
4. 재방문 날짜는 진료일 기준 결정론 계산(followup_date). LLM 무관

가드: 라벨 인덱스가 범위 밖이거나 중복이면 무시. 문장은 우리가 나눈 것이라 모델이 바꿀 수 없다.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from medimate.schema.card import AxisEntry, FieldStatus
from medimate.schema.postvisit import FollowUpDate, PostAxis, PostVisitCard

logger = logging.getLogger("medimate.memo")

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


# 문장 경계. **공백이 없어도 자른다** — 실제 사용자는 `하셨음.피검사는`처럼 붙여 쓴다
# (2026-09-14, 앱 화면 메모에서 확인). 예전 규칙은 `[.!?] 뒤 공백`만 봐서 통째로 한 조각이 됐고,
# 그러면 라벨이 문장당 하나라 한쪽 주제가 통째로 사라진다.
#
# 마침표에만 조건이 붙는다. 숫자·영문 사이의 점은 문장 끝이 아니다.
#   `2.5mg`       앞이 숫자      → 자르지 않는다
#   `example.com` 뒤가 영문자    → 자르지 않는다
#   `하셨음.피검사`  앞뒤가 한글    → 자른다
# `!`·`?`·`。`는 그런 충돌이 없어 조건 없이 자른다.
_SPLIT = re.compile(
    r"(?<=[!?。])\s*"
    r"|(?<=\.)(?<![A-Za-z0-9]\.)(?![A-Za-z0-9])\s*"
    r"|\n+"
    r"|\s+·\s+"
    r"|\s*/\s*"
)


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

# 자를 수 있는 자리 — 연결어미 `-고`·`-며` 뒤.
# `-라고/-다고/-자고/-냐고`는 인용이라 제외한다("오라고 하셨고"에서 앞쪽 `오라고`를 자르면
# "다시 오라고" + "하셨고 …"가 되어 술어가 잘린다). "그리고"도 같은 이유로 제외.
#
# **쉼표가 있으면 공백이 없어도 자른다** — `하셨고,2주`가 실제 입력이다(앱 화면 메모).
# 쉼표가 없으면 공백을 요구한다. 요구하지 않으면 `그리고`·`사고`·`먹고` 같은 낱말 안에서 잘린다.
_CLAUSE = re.compile(r"(?<=[가-힣])(고|며)(?:,\s*|\s+)")
_QUOTATIVE = ("라고", "다고", "자고", "냐고", "리고", "으라고")


def _topics(text: str) -> set[str]:
    low = text.lower()
    return {name for name, words in _TOPIC.items() if any(w in low for w in words)}


# 띄어쓰기 경계. **동사 없이 명사만 나열하는 메모**가 실제로 온다(2026-09-14).
#   "3일치 약처방 4일후 재방문"   → 연결어미도 구두점도 없다. 한 문장으로 가면 약 처방이 묻힌다
#   "혈압약 처방 2주 후 재검"      → 같은 모양
# 연결어미 규칙은 `-고`·`-며`를 요구하므로 이런 전보문을 못 나눈다.
#
# **주제 검사만으로는 모자랐다.** 띄어쓰기는 한 문장에 수십 개라 자리가 훨씬 많고, 34벡터
# 회귀에서 6개가 깨졌다 — `"짜게 먹지" / "말라고 하셨어요"`, `"갑상선 수치가" / "약간 높다고"`
# 처럼 **술어 한가운데**서 잘렸다. 그래서 세 조건을 더 건다.
#
#   1. 앞 조각이 조사·연결어미로 끝나지 않는다 — `~는`·`~가`·`~을`·`~지`로 끝나면 뒤에 술어가
#      온다는 뜻이다. 거기서 자르면 문장이 부서진다
#   2. 양쪽이 **낱말 두 개 이상**이다 — `"예약함."` 같은 한 낱말 조각은 라벨을 받아도 뜻이 없다
#   3. 양쪽이 공백 빼고 **4자 이상**이다 — `"다음 주"`(3자)·`"예약함"`(3자)에서 잘리는 것을 막는다.
#
# 셋을 걸고 34벡터를 다시 돌려 회귀 0을 확인했다.
_SPACE = re.compile(r"\s+")
# 이 글자로 끝나면 뒤에 술어가 온다. 조사와 연결어미의 마지막 글자다
_DANGLING = tuple("은는이가을를에의로와과도만지고며서면")
_MIN_TOKENS = 2
_MIN_CHARS = 4  # 공백 제외. 5로 올리면 "약 3일분 | 2주 뒤 다시 오세요"가 한 칸 밀린다


def _substantial(part: str) -> bool:
    return len(part.split()) >= _MIN_TOKENS and len(part.replace(" ", "")) >= _MIN_CHARS


def _split_at_space(sentence: str) -> list[str]:
    """띄어쓰기에서 나눈다. 연결어미 규칙과 같은 주제 조건 + 위 셋."""
    for m in _SPACE.finditer(sentence):
        head, tail = sentence[: m.start()], sentence[m.end() :]
        if not _substantial(head) or not _substantial(tail):
            continue
        if head.rstrip("·,.。!?").endswith(_DANGLING):
            continue
        ht, tt = _topics(head), _topics(tail)
        if not ht or not tt or ht & tt:
            continue
        return [head.strip()] + _split_at_space(tail.strip())
    return [sentence]


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
    # 연결어미가 없으면 띄어쓰기 경계를 본다(명사만 나열한 메모)
    return _split_at_space(sentence)


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
SPLIT_VERSION = "split-v4"

# 분리 규칙 전체의 지문. 규칙과 이름이 같이 움직이는지 테스트가 이걸로 확인한다.
# 규칙을 바꿨으면 `SPLIT_VERSION`을 올리고 이 값도 새로 박는다(테스트 실패 메시지가 새 값을 준다)
SPLIT_RULE_DIGEST = "63baa8a1d49c"


def split_rule_digest() -> str:
    """현재 분리 규칙의 지문. 상수와 비교해 규칙만 바뀐 상태를 잡는다.

    **문장 경계를 바꿀 수 있는 것을 전부 넣는다.** 정규식만 해싱하면 주제 사전에 단어를
    하나 더한 날 번호가 안 움직인다 — 그때 나뉘는 문장이 달라지는데 409가 안 난다.

    2026-09-14에 실제로 그랬다. 띄어쓰기 규칙(`_SPACE`·`_DANGLING`·최소 크기)을 새로 넣었는데
    이 목록에 안 더해서 **지문이 그대로였다.** 이 함수가 막으라고 있는 사고가 이 함수에 났다.
    규칙 재료를 새로 만들면 여기에도 넣는다.
    """
    material = [
        _SPLIT.pattern,
        _CLAUSE.pattern,
        _SPACE.pattern,
        repr(_QUOTATIVE),
        repr(_DANGLING),
        repr((_MIN_TOKENS, _MIN_CHARS)),
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


# 숫자로 안 쓰는 기간. 공백을 뺀 형태를 키로 둔다(`한 주일` → `한주일`).
_WORD_DAYS = {
    "일주일": 7,
    "한주일": 7,
    "이주일": 14,
    "두주일": 14,
    "삼주일": 21,
    "세주일": 21,
    # '일'을 빼고 쓰는 형태(2026-09-15). "이주뒤 재방문"이 안 읽혀서 앞 절의 약 기간
    # "일주일치"를 재방문으로 썼다 — 어제 고친 것과 같은 모양이 다른 낱말에서 났다
    "일주": 7,
    "이주": 14,
    "삼주": 21,
    "열흘": 10,
    "보름": 15,
    "사흘": 3,
    "나흘": 4,
}
_WORD_ALT = "|".join(sorted((k[0] + r"\s*" + k[1:] for k in _WORD_DAYS), key=len, reverse=True))

_REL = [
    (re.compile(r"(\d+)\s*주\s*(뒤|후|있다가|지나서)"), lambda m: timedelta(weeks=int(m.group(1)))),
    (re.compile(r"(\d+)\s*일\s*(뒤|후|있다가|지나서)"), lambda m: timedelta(days=int(m.group(1)))),
    (
        re.compile(r"(\d+)\s*(개월|달)\s*(뒤|후|있다가|지나서)"),
        lambda m: timedelta(days=30 * int(m.group(1))),
    ),
    # `다다음주`는 2주다. `다음 주`가 그 안에서 걸려 7일이 되던 것을 실측(followup-v1)이 잡았다
    (re.compile(r"다다음\s*주"), lambda m: timedelta(weeks=2)),
    (re.compile(r"(?<!다)다음\s*주"), lambda m: timedelta(weeks=1)),
    (
        re.compile(r"(한|두|세|네)\s*주\s*(뒤|후)"),
        lambda m: timedelta(weeks={"한": 1, "두": 2, "세": 3, "네": 4}[m.group(1)]),
    ),
    (
        re.compile(r"(한|두|세)\s*달\s*(뒤|후)"),
        lambda m: timedelta(days=30 * {"한": 1, "두": 2, "세": 3}[m.group(1)]),
    ),
    # 숫자 없는 기간 (2026-09-15). `\d+`만 보다가 `일주일 뒤에 다시 오세요`를 못 읽고
    # **앞 절의 약 기간으로 새는** 사고가 있었다. 환자는 `7일`보다 `일주일`이라고 쓴다.
    (
        re.compile(rf"({_WORD_ALT})\s*(뒤|후|있다가|지나서)"),
        lambda m: timedelta(days=_WORD_DAYS[m.group(1).replace(" ", "")]),
    ),
]
_ABS = re.compile(r"(\d{1,2})\s*월\s*(\d{1,2})\s*일")
_NEXT_MONTH_DAY = re.compile(r"다음\s*달\s*(\d{1,2})\s*일")

# ── 재방문 기간을 뒤에 오는 말 없이 읽는 자리 (2026-09-14, 앱 화면 메모) ────────────
#
# `_REL`은 `2주 뒤`처럼 **기간 + 뒤/후/있다가/지나서**만 본다. 그런데 실제 메모는
# "2주 약 먹고 다시 오라고 했어요"였고 재방문 날짜가 `null`이었다. 환자는 "뒤"를 안 쓴다.
#
# **기간만 보고 잡으면 안 된다.** "2주 약 먹고"의 2주는 약 기간이지 재방문 시점이 아닐 수 있다.
# 그래서 **같은 절에 다시 온다는 말이 있을 때만** 기간을 재방문으로 읽는다.
# `방문했`은 안 걸리고 `방문하세요`·`방문해 주세요`만 걸린다 — 지난 방문의 숫자를 끌어오면 안 된다
_RETURN_WORDS = (
    "다시 오",
    "다시 와",
    "재방문",
    "경과 보",
    "보자",
    "오라고",
    "오세요",
    "뵙",
    "방문하",
    "방문해",
    "내원",
)
_BARE_DURATION = re.compile(rf"(\d+)\s*(주일|개월|주|일|달|년)|({_WORD_ALT})")
_UNIT_DAYS = {"주": 7, "주일": 7, "일": 1, "개월": 30, "달": 30, "년": 365}

# 앞 절에서 기간을 끌어올 수 있는 조건. 여기만 추론이 들어가므로 `basis`에 남긴다.
# 약·기간 말만 있는 절이어야 한다 — 소견이나 검사 절의 숫자를 끌어오면 엉뚱한 날짜가 된다.
_MED_ONLY = ("약", "복용", "먹", "드시", "처방", "바르", "주사")


def _has_return_word(text: str) -> bool:
    return any(w in text for w in _RETURN_WORDS)


def _return_word_pos(text: str) -> int | None:
    """가장 먼저 나오는 재방문 말의 위치. 기간을 고르는 기준점이다."""
    hits = [text.index(w) for w in _RETURN_WORDS if w in text]
    return min(hits) if hits else None


def _match_days(m: re.Match[str]) -> int:
    if m.group(3):  # 숫자 없는 기간(`일주일`·`열흘`)
        return _WORD_DAYS[m.group(3).replace(" ", "")]
    return _UNIT_DAYS[m.group(2)] * int(m.group(1))


def _bare_duration(text: str, near: int | None = None) -> tuple[str, timedelta] | None:
    """기간 하나를 고른다. `near`가 있으면 **그 위치에 가장 가까운** 것으로.

    첫 번째를 집으면 안 된다 (2026-09-15). `"3일치 약처방 4일후 재방문"`에서 첫 기간은
    **약 기간**이라 재방문이 3일 뒤로 잡힌다. 지금은 `_REL`이 `4일후`를 먼저 잡아 가려져
    있지만, `"3일치 약 먹고 4일에 다시 오세요"`처럼 `뒤`·`후`가 빠지면 바로 드러난다.
    재방문 말 옆에 있는 기간이 재방문 기간이다.
    """
    ms = list(_BARE_DURATION.finditer(text))
    if not ms:
        return None
    if near is not None:
        ms.sort(key=lambda m: min(abs(m.start() - near), abs(m.end() - near)))
    m = ms[0]
    return m.group(0), timedelta(days=_match_days(m))


def followup_date(text: str, visit_date: date, prev_text: str | None = None) -> FollowUpDate | None:
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

    # 뒤에 오는 말 없이 기간만 있는 경우 — **같은 절에 다시 온다는 말이 있을 때만**
    if _has_return_word(text):
        found = _bare_duration(text, near=_return_word_pos(text))
        if found:
            raw, delta = found
            return FollowUpDate(
                text=raw,
                date=(visit_date + delta).isoformat(),
                approximate=True,
                basis=f"visit_date {visit_date.isoformat()} + {delta.days}d",
            )

        # 이 절에는 기간이 없다. **앞 절이 약·기간만 말하고 있으면** 그 기간을 쓴다
        # ("2주 약 먹고" / "다시 오라고 했어요"로 갈린 경우). 여기가 유일한 추론이라
        # 근거를 `basis`에 남긴다 — 카드를 읽는 사람이 어디서 온 날짜인지 알아야 한다.
        if prev_text:
            found = _bare_duration(prev_text)
            if found and any(w in prev_text for w in _MED_ONLY):
                raw, delta = found
                return FollowUpDate(
                    text=raw,
                    date=(visit_date + delta).isoformat(),
                    approximate=True,
                    basis=(
                        f"앞 절 '{prev_text.strip()}'의 기간을 씀 · "
                        f"visit_date {visit_date.isoformat()} + {delta.days}d"
                    ),
                )
    return None


# ── 카드 값 다듬기 (2026-09-14, ㉡) ────────────────────────────────────────
#
# 진료 전 카드는 이미 `value`=정리한 한 줄 / `evidence`=발화 원문이다. 진료 후만 문장을 그대로
# 이어 붙여서 카드가 메모를 잘라 놓은 것처럼 보였다. **`value`만 다듬고 `evidence`는 원문 그대로
# 남긴다** — 원문이 사라지면 우리가 보증하는 "환자가 한 말"이 사라진다.
#
# **경계가 이 규칙표의 전부다.** 여기는 인용의 어미를 떼는 자리이지 다시 쓰는 자리가 아니다.
#   하지 않는다 — 동의어 치환(`피검사`→`혈액검사`), 없던 말 추가(`시행`·`처방`),
#                 요약, 순서 바꾸기, 숫자·단위·약 이름 손대기
#   모르는 어미는 **그대로 둔다.** 어색하게 남는 쪽이 없는 말을 넣는 쪽보다 낫다
# CLAUDE.md의 "설명문을 생성하지 않고 인용한다"가 여기서 지켜지는 방식이다.
#
# 결정론이라 폰·서버가 같은 값을 낸다. LLM을 부르지 않는다.

# **인용과 명령을 가른다.** 둘 다 `라고`로 끝나는데 처리가 반대다.
#   위염 초기라고 하셨고   인용(명사+이라고) → 꼬리를 뗀다      → 위염 초기
#   커피 줄이라고         명령(동사+(으)라고) → 명사형          → 커피 줄이기
# 형태만으로는 못 가른다(`초기라고`·`줄이라고`가 같은 모양이다). 그래서 **동사 어간을 목록으로**
# 둔다 — 닫힌 부류이고, 목록에 없으면 인용으로 보아 떼기만 한다(모르면 덜 건드리는 쪽).
_VERB_STEMS = (
    "먹",
    "드시",
    "드세",
    "바르",
    "줄이",
    "피하",
    "끊",
    "들",
    "오",
    "가",
    "쓰",
    "하",
    "쉬",
    "있",
    "자",
    "말",
    "붙이",
    "씻",
    "마시",
    "받",
    "보",
    # `먹`보다 길어야 먼저 걸린다(아래 정렬이 그 일을 한다). 없으면 "물 자주 먹이라고"가
    # 인용으로 잡혀 `먹`만 남는다 — 어간을 잘라먹는 쪽이라 제일 나쁘다
    "먹이",
)
# 긴 어간부터 본다 — `먹이`가 `먹`보다 먼저 걸려야 한다
_STEM_ALT = "|".join(sorted(_VERB_STEMS, key=len, reverse=True))

# 뒤에 붙는 보고 어미. 있어도 되고 없어도 된다("줄이라고." / "줄이라고 하셨어요." / "…라고 함")
_SAID = r"(?:\s*(?:하[셨했]\S*|했어요|하더라\S*|합니다|한답니다|하심|함))?[\s.。!?]*$"


def _to_noun(m: re.Match[str]) -> str:
    """동사 어간 + 기. 역참조 문자열 대신 함수로 둔다."""
    return m.group(1) + "기"


# 연결어미 `-고`로 끝나는 절의 어간. 문장을 절 단위로 나누면 "일주일치 약처방받고"처럼
# **연결어미가 꼬리에 남는다.** 그대로 두면 카드가 메모를 잘라 놓은 것과 같다(2026-09-15 QA).
# `보`는 뺐다 — "결과 보고"(report)와 "결과 보고"(see and)를 가를 수 없다. 모르면 둔다.
_CONNECTIVE_STEMS = tuple(
    st
    for st in dict.fromkeys(
        _VERB_STEMS + ("받", "맞", "찍", "넣", "빼", "붙이", "마시", "걷", "뛰", "복용하", "처방받")
    )
    if st != "보"  # `_VERB_STEMS`에 있어도 여기서는 뺀다 — "결과 보고"가 report일 수 있다
)
_CONNECTIVE_ALT = "|".join(sorted(_CONNECTIVE_STEMS, key=len, reverse=True))


def _nominal(stem: str) -> str:
    """어간 → 명사형(`-음`/`-ㅁ`). 받침 있으면 `음`, 없으면 ㅁ 받침을 합성한다.

    받 → 받음 · 먹 → 먹음 · 하 → 함 · 오 → 옴 · 쓰 → 씀 · 드시 → 드심 · 줄이 → 줄임
    """
    last = stem[-1]
    if not ("가" <= last <= "힣"):
        return stem + "음"
    code = ord(last) - 0xAC00
    if code % 28:  # 받침 있음
        return stem + "음"
    return stem[:-1] + chr(0xAC00 + code + 16)  # ㅁ 종성 index 16


def _connective_to_noun(m: re.Match[str]) -> str:
    return _nominal(m.group(1))


_RULES: list[tuple[re.Pattern[str], object]] = [
    # ~지 말라고 → ~지 않기 (금지). `말`이 어간 목록에 있어 아래 규칙보다 먼저 와야 한다
    (re.compile(r"지\s*말라고" + _SAID), "지 않기"),
    # 동사 + (으)라고 / (으)래요 → 명사형
    (re.compile(rf"({_STEM_ALT})으?라고" + _SAID), _to_noun),
    (re.compile(rf"({_STEM_ALT})으?래요" + _SAID), _to_noun),
    # 명사 + (이)라고 → 인용. 꼬리만 뗀다
    (re.compile(r"이?라고" + _SAID), ""),
    # 평서 인용은 어미를 되살린다. 떼기만 하면 어간이 맨몸으로 남는다(`괜찮다고` → `괜찮`)
    (re.compile(r"다고" + _SAID), "다"),
    (re.compile(r"대요" + _SAID), "다"),
    (re.compile(r"자고" + _SAID), "자"),
    # 절 끝에 남은 연결어미 `-고` → 명사형(`받고` → `받음`). 인용 규칙들 **뒤**에 둔다 —
    # `~라고`·`~다고`·`~자고`가 먼저 걸려야 하고, 여기는 그 무엇도 아닌 맨 `-고`만 받는다
    (re.compile(rf"({_CONNECTIVE_ALT})고[\s.。!?,]*$"), _connective_to_noun),
    # 인용 없이 명사형 어미(`했음`·`갔음`)만 붙은 것은 **건드리지 않는다.**
    # 떼면 "오늘은 스케일링만 했음" → "…했"으로 어간이 맨몸으로 남는다. 어색하게 남는 쪽이
    # 잘라먹는 쪽보다 낫다 — 모르는 어미는 그대로 둔다는 원칙이 이 자리다.
]
_TRAILING = re.compile(r"[\s·,.。!?]+$")


# 칸 이름이 이미 말하는 낱말이 값 꼬리에 또 있으면 뗀다(2026-09-15). "일주일치 약처방 이주일 후
# 재방문"이 약 칸에 `일주일치 약처방`, 재방문 칸에 `이주일 후 재방문`으로 나가 "잘라 놓기만 했다"로
# 읽혔다. 사실을 바꾸는 게 아니라 **라벨과 겹치는 단어**만 빼는 것이라 ㉡(어미 정리 + 틀) 안이다.
# 긴 것부터 본다. 떼고 남는 것이 없으면 안 뗀다 — `재방문` 한 낱말은 그대로 둔다.
_AXIS_TRAILERS: dict[PostAxis, tuple[str, ...]] = {
    PostAxis.FOLLOW_UP: ("재방문", "재내원", "재진", "방문"),
    PostAxis.MEDICATION_INSTRUCTIONS: (
        "약처방받음",
        "약 처방받음",
        "약처방",
        "약 처방",
        "처방받음",
        "처방",
    ),
}
_TRAILING_PARTICLE = re.compile(r"(에|에서|으로|로)$")


def _strip_axis_trailer(s: str, axis: PostAxis | None) -> str:
    if axis is None:
        return s
    for word in _AXIS_TRAILERS.get(axis, ()):
        if s.endswith(word):
            rest = s[: -len(word)].strip()
            if len(rest.replace(" ", "")) < 2:
                return s  # 떼면 남는 게 없다
            return _TRAILING_PARTICLE.sub("", rest).strip() or s
    return s


def tidy_value(text: str, axis: PostAxis | None = None) -> str:
    """카드 `value`용으로 어미를 정리한다. **원문(`evidence`)은 건드리지 않는다.**

    `axis`를 주면 그 칸 이름과 겹치는 꼬리 낱말(`재방문`·`약처방`)도 뗀다 — 라벨이 이미 말한다.
    규칙에 없는 모양은 그대로 돌려준다. 결과가 비면 원문을 쓴다 — 다듬다가 값을 없애는 것이
    제일 나쁘다.
    """
    s = text.strip()
    out = None
    for rx, repl in _RULES:
        new = rx.sub(repl, s)
        if new != s:
            out = _TRAILING.sub("", new).strip()
            break
    if out is None:
        out = _TRAILING.sub("", s).strip()
    out = _strip_axis_trailer(out, axis)
    return out or s


class FollowUpReader(Protocol):
    """재방문 표현을 읽는 것. `read`는 {"text","days","month","day"}를 돌려준다(followup-v1)."""

    def read(self, sentence: str, prev_text: str | None = None) -> dict: ...


_READ_MAX_DAYS = 400


def followup_from_reader(
    reader: FollowUpReader, sentence: str, visit_date: date, prev_text: str | None = None
) -> FollowUpDate | None:
    """LLM이 **읽은** 표현으로 날짜를 **계산**한다. 검증에 하나라도 걸리면 None — 지어내지 않는다.

    - `text`는 현재 문장에 글자 그대로 있어야 한다(근거 가드). 앞 문장에만 있으면 버린다 —
      그게 약 기간을 재방문으로 옮겨 적는 경로다
    - `days`는 1..400. 아니면 버린다. 월·일이 오면 진료일 기준 다음 도래일
    - 어떤 예외든 None. 2차 호출이 죽어도 카드는 나간다
    """
    try:
        out = reader.read(sentence, prev_text)
    except Exception as e:  # noqa: BLE001 — 2차 호출이 죽어도 카드는 나간다
        # 삼키되 **남긴다**(2026-09-15). 전에는 조용히 None이라 "안 켜진 것"과 "죽은 것"이
        # 밖에서 구별되지 않았다. 문장은 안 적는다 — 진료 내용이다. 예외 종류만 남긴다
        logger.warning("재방문 표현 읽기 실패(%s) — 규칙 결과로 계속합니다", type(e).__name__)
        return None
    if not isinstance(out, dict):
        return None
    text = str(out.get("text") or "").strip()
    if not text or text not in sentence:
        return None
    days, mo, d = out.get("days"), out.get("month"), out.get("day")
    if isinstance(days, int) and not isinstance(days, bool) and 1 <= days <= _READ_MAX_DAYS:
        dt = visit_date + timedelta(days=days)
        return FollowUpDate(
            text=text,
            date=dt.isoformat(),
            approximate=True,
            basis=f"LLM 읽음 '{text}' = {days}d · visit_date {visit_date.isoformat()} + {days}d",
        )
    if isinstance(mo, int) and isinstance(d, int) and not isinstance(mo, bool):
        yr = visit_date.year + (1 if (mo, d) < (visit_date.month, visit_date.day) else 0)
        try:
            dt = date(yr, mo, d)
        except ValueError:
            return None
        return FollowUpDate(
            text=text,
            date=dt.isoformat(),
            approximate=False,
            basis=f"LLM 읽음 '{text}' · 절대 날짜",
        )
    return None


@dataclass
class MemoResult:
    card: PostVisitCard
    sentences: list[str]
    labels: dict[int, str]
    dropped: list[dict] = field(default_factory=list)  # 가드가 무시한 라벨


def _assemble(
    card: PostVisitCard,
    sentences: list[str],
    labels: dict[int, str],
    visit_date: date | None,
    followup_reader: FollowUpReader | None,
) -> None:
    """(문장, 라벨) → 카드. 규칙 분리든 LLM 조각이든 되보낸 조각이든 여기 하나로 온다."""
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
            # `value`는 어미를 정리한 줄, `evidence`는 **문장 원문 그대로**.
            # 진료 전 카드가 이미 이 구조다. 원문이 사라지면 우리가 보증하는 것이 사라진다
            entry.value = " · ".join(tidy_value(s, axis) for s in sents)
            entry.evidence = list(sents)
        else:
            entry.status = FieldStatus.UNKNOWN  # 메모에 그 묶음 얘기가 없었다
            entry.value = None
            entry.evidence = []
    card.unsorted = unsorted

    if buckets[PostAxis.FOLLOW_UP] and visit_date:
        # **앞 절을 같이 넘긴다.** "2주 약 먹고 / 다시 오라고 했어요"처럼 기간과 재방문 말이
        # 다른 조각으로 갈리는 일이 있다(우리 분리기가 그렇게 나눈다 — 서로 다른 묶음이니 맞다).
        # 재방문 절만 보면 기간이 없어 날짜가 `null`이 된다.
        for i, s in enumerate(sentences):
            if labels.get(i) != PostAxis.FOLLOW_UP.value:
                continue
            prev = sentences[i - 1] if i else None
            fu = followup_date(s, visit_date, prev_text=prev)
            # 규칙이 먼저다 — 확인 가능하고 공짜다. 규칙이 **못 읽었거나 앞 절에서 끌어왔을 때만**
            # LLM에게 읽힌다(2026-09-15). "이주뒤"를 못 읽어 앞 절의 약 기간을 재방문으로 쓴
            # 것이 계기다. LLM이 읽은 것도 문장에 글자 그대로 있어야만 받는다
            if followup_reader is not None and (fu is None or fu.basis.startswith("앞 절")):
                read = followup_from_reader(followup_reader, s, visit_date, prev_text=prev)
                if read is not None:
                    fu = read
            if fu:
                card.follow_up_date = fu
                # **`value`에는 날짜를 넣지 않는다.**
                #
                # 2026-09-14에 여기서 `"{text} ({M월 D일} 전후)"`를 조합해 넣었다가 화면에
                # 날짜가 두 번 찍혔다("재방문 4일후 (2026년 9월 21일) (2026년 9월 21일)").
                # `follow_up_date`가 이미 같은 날짜를 들고 있었고, 조합할 수 있는 자리가 둘이
                # 되니 양쪽이 다 그렸다.
                #
                # 층을 되돌린다. `value`는 다른 축과 같게 **메모에서 온 말**만 담고,
                # 날짜의 주인은 `follow_up_date` 하나다. 화면 문장은 앱이 한 번만 만든다 —
                # 우리가 계산한 값을 환자 말 옆에 끼워 넣는 순간 그 줄의 주인이 둘이 된다.
                break


_WS = re.compile(r"\s+")


def slice_by_coverage(memo: str, pieces: Sequence[str]) -> list[str] | None:
    """조각들이 순서대로 원문을 **빠짐없이·겹침없이** 덮는지 보고, 덮으면 원문에서 잘라 돌려준다.

    공백만 무시한다. 글자가 하나라도 다르거나 빠지거나 남으면 None — LLM이 고쳐 쓴 것이다.
    돌려주는 조각은 **원문 substring**이라 띄어쓰기·구두점이 원문 그대로다(evidence 보증).
    """
    if not pieces:
        return None
    out: list[str] = []
    i = 0
    n = len(memo)
    for piece in pieces:
        target = _WS.sub("", piece)
        if not target:
            return None
        while i < n and memo[i].isspace():
            i += 1
        start = i
        k = 0
        while k < len(target):
            while i < n and memo[i].isspace():
                i += 1
            if i >= n or memo[i] != target[k]:
                return None
            i += 1
            k += 1
        out.append(memo[start:i].strip())
    while i < n and memo[i].isspace():
        i += 1
    if i != n:
        return None  # 남은 글자가 있다 — 빠뜨렸다
    return out


class MemoSegmenter(Protocol):
    """메모를 조각 + 라벨로. `segment`는 {"segments":[{"text","label"},…]}를 돌려준다(memo-v5)."""

    def segment(self, memo: str) -> dict: ...


def segment_memo(
    memo: str,
    segmenter: MemoSegmenter,
    visit_date: date | None = None,
    clinic: str | None = None,
    card: PostVisitCard | None = None,
    followup_reader: FollowUpReader | None = None,
) -> MemoResult | None:
    """LLM이 나눈 조각으로 카드를 만든다.

    **원문을 정확히 덮지 못하면 None** — 호출자가 규칙 분리로 폴백한다.
    """
    try:
        out = segmenter.segment(memo)
    except Exception as e:  # noqa: BLE001
        from medimate.llm.providers import BudgetExceeded  # 순환 import 회피용 지연 import

        if isinstance(e, BudgetExceeded):
            raise  # 예산 초과는 삼키지 않는다 — 503으로 올라가야 한다
        logger.warning("메모 조각내기 실패(%s) — 규칙 분리로 계속합니다", type(e).__name__)
        return None
    segs = out.get("segments") if isinstance(out, dict) else None
    if not isinstance(segs, list) or not segs:
        return None
    texts, labs = [], []
    for sgm in segs:
        if not isinstance(sgm, dict):
            return None
        t, lab = sgm.get("text"), sgm.get("label")
        if not isinstance(t, str) or lab not in LABELS:
            return None
        texts.append(t)
        labs.append(lab)
    sentences = slice_by_coverage(memo, texts)
    if sentences is None:
        logger.warning("메모 조각이 원문을 덮지 못했습니다 — 규칙 분리로 계속합니다")
        return None
    card = card or PostVisitCard()
    card.memo = memo
    card.clinic = clinic
    card.visit_date = visit_date.isoformat() if visit_date else card.visit_date
    labels = dict(enumerate(labs))
    _assemble(card, sentences, labels, visit_date, followup_reader)
    return MemoResult(card, sentences, labels, [])


def assemble_from_client(
    memo: str,
    sentences: Sequence[str],
    keyed_labels: dict[str, str],
    visit_date: date | None = None,
    clinic: str | None = None,
    card: PostVisitCard | None = None,
) -> MemoResult | None:
    """앱이 되보낸 조각 + 라벨로 카드를 만든다(1q-2 수정). **다시 나누지 않는다.**

    조각이 원문을 덮지 못하면 None — 되보낸 것이 이 메모의 조각이 아니다(409 감).
    """
    sliced = slice_by_coverage(memo, sentences)
    if sliced is None:
        return None
    card = card or PostVisitCard()
    card.memo = memo
    card.clinic = clinic
    card.visit_date = visit_date.isoformat() if visit_date else card.visit_date
    raw = MemoLabels.from_keyed(keyed_labels, len(sliced))
    labels = dict(enumerate(raw.labels))
    _assemble(card, sliced, labels, visit_date, None)
    return MemoResult(card, sliced, labels, [])


def classify_memo(
    memo: str,
    classifier: MemoClassifier,
    visit_date: date | None = None,
    clinic: str | None = None,
    card: PostVisitCard | None = None,
    followup_reader: FollowUpReader | None = None,
) -> MemoResult:
    """메모 하나를 4묶음 카드로. 문장 원문 보존, 라벨 없는 문장은 unsorted.

    `followup_reader`가 있으면 규칙이 못 읽은 재방문 표현을 LLM으로 **읽고** 날짜는 코드가 센다.
    """
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

    _assemble(card, sentences, labels, visit_date, followup_reader)
    return MemoResult(card, sentences, labels, dropped)
