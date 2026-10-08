"""질문 후보 — 템플릿(모델 없음). 앱(온디바이스 프로필)용 (2026-10-08).

Qwen3-1.7B는 질문 후보를 만들 때 환자가 하지 않은 말을 지어냈다(프롬프트 예시의 "이사"를 59장에,
예시를 빼면 "계란"·"계단"을 — evals/RESULTS.md "질문 후보 · 온디바이스").
앱은 외부 API 없이 가기로 해서
생성 대신 **환자 말을 따옴표로 그대로 인용하고 고정된 질문 꼬리를 붙인다.**

- 인용은 카드 값 원문(또는 그 첫 쉼표 절)이다. 새 말을 만들지 않는다 → 지어내기 0이 구조로 보장된다
- 값을 문장 안에 녹이지 않는다. 카드 값은 "시큰하고 가끔 찌릿해요"처럼 환자 문장이라,
  녹이면 어미가 깨진다
- 조사(은/는·이/가·면/이면)만 인용의 마지막 한글 받침으로 고른다
- 웹 데모(서버 프로필)는 지금처럼 Nova가 만든다. 이 모듈은 모델이 없는 경로 전용이다
"""

from __future__ import annotations

import re
from typing import Any

MAX_LEN = 35  # 백엔드 검증 40자보다 여유를 둔다(assist_prompts v7과 같은 선)
MAX_ITEMS = 4
OPEN, CLOSE = "“", "”"

# 값이 아니라 "없다/모른다"인 답 — 질문 재료가 아니다
_NEGATIVE = re.compile(
    r"^\s*(없|딱히|특별히 없|따로 없|모르|잘 모르|아니|안 퍼|그런 거 없|\(안 답함\))"
)
_HANGUL = re.compile(r"[가-힣]")
# 동반·퍼짐에서 "없다"로 끝나는 답("허리는 안 아파요", "다리 저림은 없어요") — "~도 관련이 있나요"로
# 감싸면 없는 것을 묻게 된다. 앞에 내용이 있어도 끝이 부정이면 뺀다
_ENDS_NEGATIVE = re.compile(
    r"(없어요|없음|없고|않아요|않음|안 아파요|안 나요|괜찮아요|괜찮음)[.!~\s]*$"
)
# 환자가 이미 질문으로 말한 값("이렇게 자꾸 꺾이는 게 정상인가요") — 감싸지 않고 그대로 살린다
# "나요"·"가요"만으로는 안 된다 — "열이 나요"·"소리가 나요"는 서술이다
_QUESTION = re.compile(
    r"(\?|까요|인가요|는가요|건가요|닌가요|하나요|되나요|있나요|없나요|맞나요|는지)[.!~\s]*$"
)
_CLAUSE = re.compile(r"[,.]")
MIN_VALUE = 3  # "밤에"처럼 짧은 값은 질문이 되지 않는다


def _has_batchim(text: str) -> bool:
    """마지막 한글 글자에 받침이 있는가. 한글이 없으면 받침 없음으로 본다.

    끝의 괄호는 건너뛴다 — "진통제(가끔)"의 조사는 "진통제"에 붙는다("진통제(가끔)는")."""
    text = re.sub(r"\s*\([^()]*\)\s*$", "", text) or text
    for ch in reversed(text):
        if _HANGUL.match(ch):
            return (ord(ch) - 0xAC00) % 28 != 0
    return False


def _quote(v: str) -> str:
    return f"{OPEN}{v}{CLOSE}"


def _topic(v: str) -> str:
    return _quote(v) + ("은" if _has_batchim(v) else "는")


def _subject(v: str) -> str:
    return _quote(v) + ("이" if _has_batchim(v) else "가")


def _cond(v: str) -> str:
    # 인용 뒤 가정은 "라면"이 표준형이다("“비슷해요”라면"). 받침 있는 말 뒤는 "이라면"
    return _quote(v) + ("이라면" if _has_batchim(v) else "라면")


# 칸마다 꼬리가 다르다 — 한 세트 안에서 같은 끝맺음이 반복되지 않는다.
# 칸마다 긴 꼬리·짧은 꼬리 순서로 둔다. 값이 길면 짧은 꼬리로 35자에 넣는다(값을 자르기 전에)
_AXIS_TEMPLATES: list[tuple[str, list[Any]]] = [
    (
        "exacerbating",
        [lambda v: f"{_topic(v)} 어떻게 해야 하나요?", lambda v: f"{_topic(v)} 어떻게 하나요?"],
    ),
    (
        "character",
        [lambda v: f"{_topic(v)} 어떤 상태인가요?", lambda v: f"{_topic(v)} 어떤 상태예요?"],
    ),
    (
        "associated",
        [lambda v: f"{_quote(v)}도 관련이 있나요?", lambda v: f"{_quote(v)}도 관련 있나요?"],
    ),
    (
        "time_course",
        [lambda v: f"{_cond(v)} 언제 다시 와야 하나요?", lambda v: f"{_cond(v)} 다시 와야 하나요?"],
    ),
    ("radiation", [lambda v: f"{_topic(v)} 같은 증상인가요?"]),
]
# 위 칸으로 3개가 안 될 때만 쓴다
_FILL_TEMPLATES: list[tuple[str, list[Any]]] = [
    # 시작 값은 "3주쯤"·"작년부터요"처럼 기간이기도 해서 "시작됐는데"에 녹이지 않고
    # "시작이 …인데"로 받는다
    ("onset", [lambda v: f"시작이 {_quote(v)}인데 문제가 되나요?"]),
]
MIN_ITEMS = 3
_PROFILE_TEMPLATES: list[tuple[str, list[Any]]] = [
    # "먹어도"가 아니라 "써도" — 바르는 약·안약도 복용약 칸에 온다("바르는 약(이름은 몰라요)")
    ("medications", [lambda v: f"{_topic(v)} 계속 써도 되나요?"]),
    ("allergies", [lambda v: f"{_quote(v)} 알러지가 있는데 처방에 반영되나요?"]),
    ("conditions", [lambda v: f"{_subject(v)} 있는데 이번 증상과 관련이 있나요?"]),
]


def _usable(v: Any) -> str | None:
    if not isinstance(v, str):
        return None
    v = v.strip()
    if not v or _NEGATIVE.match(v):
        return None
    return v


def _fit(makes: list[Any], value: str) -> str | None:
    """35자 안에 드는 질문. 값 전체·첫 절(쉼표·마침표 앞, 원문의 앞부분) 순으로
    긴 꼬리부터 넣어 본다.
    그래도 길면 버린다 — 값을 중간에서 자르지 않는다(자르면 환자 말이 바뀐다)."""
    first = _CLAUSE.split(value)[0].strip()
    for make in makes:
        for v in (value, first):
            if len(v) < MIN_VALUE:
                continue
            text = make(v)
            if len(text) <= MAX_LEN:
                return text
    return None


def _as_question(value: str) -> str | None:
    """환자가 이미 질문으로 말했으면 그 말 그대로(물음표만 붙인다)."""
    v = value.rstrip(".!~ ")
    if not _QUESTION.search(v):
        return None
    text = v if v.endswith("?") else v + "?"
    return text if len(text) <= MAX_LEN else None


def template_candidates(
    axes: dict[str, Any], profile: dict[str, list[str]] | None = None
) -> list[dict]:
    """카드 값 → 질문 후보 [{text, source}]. 재료가 없으면 빈 목록(억지로 만들지 않는다)."""
    items: list[dict] = []
    # 복용약·알러지·기저질환이 있으면 그중 하나 — 환자가 먼저 꺼내야 처방에 반영된다(v7 규칙)
    for source, makes in _PROFILE_TEMPLATES:
        for name in (profile or {}).get(source) or []:
            v = _usable(name)
            if v and (text := _fit(makes, v)):
                items.append({"text": text, "source": source})
                break
        if items:
            break
    for source, makes in _AXIS_TEMPLATES:
        if len(items) >= MAX_ITEMS:
            break
        v = _usable(axes.get(source))
        if not v:
            continue
        if source in ("associated", "radiation") and _ENDS_NEGATIVE.search(v):
            continue
        if any(v in it["text"] for it in items):  # 같은 말을 두 칸에 적었으면 한 번만 인용한다
            continue
        if text := (_as_question(v) or _fit(makes, v)):
            items.append({"text": text, "source": source})
    for source, makes in _FILL_TEMPLATES:
        if len(items) >= MIN_ITEMS:
            break
        v = _usable(axes.get(source))
        if v and (text := _fit(makes, v)):
            items.append({"text": text, "source": source})
    return items
