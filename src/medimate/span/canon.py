"""템플릿 후보(canon) — 원문 조각을 **자르고 순서만 바꾸고, 정해진 표로만 줄인** 값. 호출 0.

gold 검토(2026-09-21)에서 나온 카드 값의 모양:
- 소견   서술문이면 명사형 `-음`          혈압이 좀 높다고 → 혈압이 좀 높음
- 검사   [시점] [검사] (결과 안내|예약|예정)  공복혈당 검사 3개월 뒤 다시 → 3개월 뒤 공복혈당 검사
- 약     [조건·시점] [약]                    혈압약은 아침에 물로만 → 아침에 혈압약
- 재방문 [조건] (시점) 재방문 필요            안 좋아지면 바로 오래요 → 안 좋아지면 재방문 필요

지키는 선
- 슬롯에 들어가는 말은 전부 **조각에서 잘라낸 것**이다. 순서를 바꿀 뿐 새 말을 넣지 않는다
- 예외는 `ABBREV`(축약 표)와 축이 정한 꼬리말(`재방문 필요`, `결과 안내`, `예약`, `예정`)뿐이다.
  표에 없는 축약은 하지 않는다. 카드가 길어지는 것을 디자이너·웹이 싫어해서 두는 최소한이다
- 후보일 뿐이다. 고르는 것은 선택기이고, 못 고르면 원문형(부분 문자열)으로 간다
"""

from __future__ import annotations

import re

from medimate.dialog.memo import _nominal, tidy_value
from medimate.schema.postvisit import PostAxis
from medimate.span.candidates import Candidate, CandidateSet, compact
from medimate.text.lexicon import Lexicon

# 축약 표. 관찰된 것만. 왼쪽은 조각에 글자 그대로 있어야 한다
ABBREV: dict[str, str] = {
    "심해지면": "악화 시",
    "다음에": "추후",
    "그때": "추후",
    "넘을 때만": "넘으면",
}

_WS = re.compile(r"\s+")
_TRAIL = re.compile(r"[\s·,.。!?]+$")

# 재방문 축 꼬리 — 이 말을 떼고 `재방문 필요`를 붙인다
_RETURN_TAIL = re.compile(
    r"\s*(?:바로\s*)?(?:다시\s*)?"
    r"(?<![가-힣])(?:오래요|오라고(?:\s*(?:했어요|하셨어요|했다|하셨다))?|오세요|오라|"
    r"보자고(?:\s*(?:하셨다|하셨어요|했어요))?|보자|봐요|뵙|"
    r"재진|재방문\s*하래요|재방문|방문하래요|경과\s*보자고|경과\s*보러\s*오라고(?:\s*했다)?)"
    r"[\s.。!?]*$"
)
_COND = re.compile(r"(?:면|시)\s*$")  # 조건절 끝

# 검사 축
_RESULT = re.compile(r"결과.*(?:알려|보자|안내|나오|준다|줄)")
_PLAN = re.compile(r"(?:하자고|해보자고|찍어보자고|찍기로|하기로|해보자)")
_TIME_WORDS = re.compile(r"(?:다음에|그때|추후)")

# 약 축 — 조건·시점 절
_MED_TIME = re.compile(r"(?:아침|저녁|점심|식후|식전|자기\s*전|공복에?|밤)(?:에)?")
_MED_COND = re.compile(r"[가-힣0-9\s]+?(?:넘을 때만|넘으면|심하면|있을 때만|아플 때만|때만|면)")

_REASSURE = re.compile(r"괜찮|괜찬|정상|특별한\s*건\s*없|필요\s*없|걱정")


def _abbrev(s: str) -> str:
    for a, b in ABBREV.items():
        s = s.replace(a, b)
    return s


def _statement_nominal(seg: str, axis: PostAxis) -> str | None:
    """`혈압이 좀 높다고 하셨어요` → `혈압이 좀 높음`. tidy_value가 만든 `-다`형을 `-음`으로."""
    t = tidy_value(seg, axis)
    if not t or compact(t) == compact(seg):
        # 어미 정리가 안 됐다(`-음/-함` 문어체, `아니래요`). 끝의 인용·보고 어미만 뗀다
        t = re.sub(r"(?:이?라고|다고)(?:\s*(?:하셨\S*|했\S*|함))?[\s.。!?]*$", "", seg).strip()
        t = re.sub(r"(?:래요|대요)[\s.。!?]*$", "다", t)
        t = _TRAIL.sub("", t)
        if compact(t) == compact(seg):
            return None
    # 앞 절이 배경이면(`초음파 봤는데 파열은 아니다`) 뒤 절만 — 소견은 뒤 절이다
    t = re.split(r"(?:는데|은데|ㄴ데)\s+", t)[-1]
    m = re.search(r"([가-힣]+)다$", t)
    if m:
        stem = m.group(1)
        return t[: m.start()] + _nominal(stem)
    return t


def canon_candidates(cset: CandidateSet, axis: str, lexicon: Lexicon) -> list[Candidate]:
    seg = cset.segment
    ax = PostAxis(axis)
    subs = [c for c in cset.candidates if not c.derived]
    out: list[str] = []

    if ax == PostAxis.FINDINGS:
        n = _statement_nominal(seg, ax)
        if n:
            out.append(n)

    elif ax == PostAxis.FOLLOW_UP:
        m = _RETURN_TAIL.search(seg)
        if m:
            prefix = _abbrev(_TRAIL.sub("", seg[: m.start()]).strip())
            prefix = re.sub(r"(?:에|에는|엔)$", "", prefix).strip()
            out.append((prefix + " 재방문 필요").strip())

    elif ax == PostAxis.TESTS:
        times = [c.text for c in subs if "duration" in c.kinds]
        tw = _TIME_WORDS.search(seg)
        if tw:
            times.append(_abbrev(tw.group(0)))
        terms = [
            c.text for c in subs if any(k in ("lexicon:test", "lexicon:procedure") for k in c.kinds)
        ]
        if not terms:
            terms = [c.text for c in subs if "chunk" in c.kinds and "검사" in c.text]
        suffix = ""
        if _RESULT.search(seg):
            suffix = "결과 안내"
        elif "예약" in seg:
            suffix = "예약"
        elif _PLAN.search(seg) and not times:
            suffix = "예정"
        for term in terms[:2]:
            term_c = _TRAIL.sub("", term)
            for time in times[:2] or [""]:
                time_c = re.sub(r"(?:에|에는)$", "", time.strip())
                if time_c and compact(time_c) in compact(term_c):
                    continue  # 시점이 검사 chunk 안에 이미 있다
                base = f"{time_c} {term_c}".strip()
                if suffix == "결과 안내":
                    out.append(f"{base} 결과 안내".replace("결과 결과", "결과"))
                elif suffix:
                    out.append(f"{base} {suffix}")
                out.append(base)

    elif ax == PostAxis.MEDICATION_INSTRUCTIONS:
        meds = [c.text for c in subs if "lexicon:medication" in c.kinds]
        if not meds and re.search(r"(?<![가-힣])약(?=[은는이가을를도만\s.,]|$)", seg):
            meds = ["약"]
        conds: list[str] = []
        mt = _MED_TIME.search(seg)
        if mt:
            conds.append(mt.group(0))
        mc = _MED_COND.search(seg)
        if mc:
            cond = mc.group(0).strip()
            for med in meds:
                cond = cond.replace(med, "").strip(" 은는이가")
            if cond:
                conds.append(_abbrev(cond))
        durs = [c.text for c in subs if "duration" in c.kinds]
        for med in meds[:2]:
            for cond in conds:
                if compact(cond) not in compact(med):
                    out.append(f"{cond} {med}")
            for d in durs[:2]:
                if compact(d) not in compact(med) and compact(med) not in compact(d):
                    out.append(f"{med} {d}")

    # 중복·기존 후보와 같은 것은 뺀다. 조각 전체와 같은 것도
    have = {c.compact for c in cset.candidates} | {compact(seg)}
    result: list[Candidate] = []
    for text in dict.fromkeys(out):
        text = _WS.sub(" ", text).strip()
        if len(compact(text)) < 2 or compact(text) in have:
            continue
        have.add(compact(text))
        result.append(Candidate("", text, -1, -1, ("canon",), derived=True))
    return result


def reassurance(seg: str) -> bool:
    """안심·부정 소견 — gold 검토에서 값 없음(NONE)으로 정해졌다."""
    return bool(_REASSURE.search(seg))
