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
    # 붙여 쓴 `다시오라고`도 받는다. `재보자고`의 보자고는 앞 글자 검사로 막는다
    r"(?<![가-힣])\s*(?:참지\s*말고\s*)?(?:바로\s*)?(?:다시\s*)?"
    r"(?:오래요|오래|오라고(?:\s*(?:했어요|하셨어요|했다|하셨다|했음|하셨음|함))?|오세요|오라|"
    r"보자고(?:\s*(?:하셨다|하셨어요|했어요|했음|하셨음|함))?|보자|봐요|보기|뵙|"
    r"재진|재방문\s*하래요|재방문|방문하래요|경과\s*보자고|경과\s*보러\s*오라고(?:\s*했다)?)"
    r"[\s.。!?]*$"
)
_COND = re.compile(r"(?:면|시)\s*$")  # 조건절 끝

# 검사 축
_RESULT = re.compile(r"결과.*(?:알려|보자|안내|나오|준다|줄)")
# Kiwi가 띄운 조각(`찍어 보자고`)도 받는다
_PLAN = re.compile(r"(?:하자고|해\s*보자고|찍어\s*보자고|찍기로|하기로|해\s*보자)")
_TIME_WORDS = re.compile(r"(?:다음에|그때|추후)")

# 약 축 — 조건·시점 절
_MED_TIME = re.compile(r"(?:아침|저녁|점심|식후|식전|자기\s*전|공복에?|밤)(?:에)?")
_MED_COND = re.compile(r"[가-힣0-9\s]+?(?:넘을 때만|넘으면|심하면|있을 때만|아플 때만|때만|면)")

# 안심·부정 소견. 끝의 `아니`(ㄴㄴ 정규화형)는 잡고 `~은 아니고`는 안 잡는다(뒤에 소견이 온다)
_REASSURE = re.compile(r"괜찮|괜찬|정상|특별한\s*건\s*없|필요\s*없|걱정|(?<![가-힣])아니\s*$")


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
    if not t.endswith("다"):
        return t
    joined = _kiwi_nominal(t)
    if joined:
        return joined
    m = re.search(r"([가-힣]+)다$", t)
    if m:
        return t[: m.start()] + _nominal(m.group(1))
    return t


def _kiwi_nominal(clause: str) -> str | None:
    """`-다`로 끝나는 절 → `-음` 명사형을 Kiwi join으로. 불규칙(ㅂ·ㅅ)을 Kiwi가 안다(#122).

    `혈당이 경계에 가깝다` → 가깝/VA-I + 음/ETN → `혈당이 경계에 가까움`. 손으로 짠 _nominal은 폴백.
    """
    from medimate.text.tokenize import base_kiwi

    try:
        kiwi = base_kiwi()
        toks = list(kiwi.tokenize(clause))
        while toks and toks[-1].tag in ("EF", "EC", "SF", "SP", "SE"):
            toks.pop()
        # `뭉친 상태`·`뭉친 거`·`늘어난 것` → 관형형 + 형식명사는 그 동사의 명사형이다
        if (
            len(toks) >= 2
            and toks[-1].tag in ("NNG", "NNB")
            and toks[-1].form in ("상태", "거", "것", "중")
            and toks[-2].tag == "ETM"
        ):
            toks = toks[:-2]
        if not toks or not toks[-1].tag.startswith(
            ("VV", "VA", "VX", "VCP", "VCN", "XSA", "XSV", "EP")
        ):
            return None
        return kiwi.join([(t.form, t.tag) for t in toks] + [("음", "ETN")])
    except Exception:  # noqa: BLE001 — 실패하면 정규식 폴백
        return None


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
        lex_terms = [
            c for c in subs if any(k in ("lexicon:test", "lexicon:procedure") for k in c.kinds)
        ]
        terms = [c.text for c in lex_terms]
        # 용어를 품은 chunk(`뇌 MRI`)를 앞에 — 수식어 붙은 검사명이 값이다(#122). 시점 든 chunk 제외
        for lt in lex_terms:
            for c in subs:
                if (
                    "chunk" in c.kinds
                    and c is not lt
                    and c.start <= lt.start
                    and c.end >= lt.end
                    and "duration" not in c.kinds
                    and not _TIME_WORDS.match(c.text)
                    and "결과" not in c.text
                    and not any(
                        d.start >= c.start and d.end <= c.end for d in subs if "duration" in d.kinds
                    )
                ):
                    terms.insert(0, c.text)
        if not terms:
            terms = [c.text for c in subs if "chunk" in c.kinds and "검사" in c.text]
        # `공복혈당을 다시 검사하기로` — 검사한다는 동사가 있고 용어에 `검사`가 없으면 붙인다
        if re.search(r"검사\s*(?:하|받|다시)", seg):
            terms = [t if "검사" in t else f"{t} 검사" for t in terms]
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
                # 재배열(압축)할 때만 `뒤`를 `후`로. 재방문 축의 `2주 뒤`는 원문 그대로 둔다(결정 4)
                time_c = re.sub(r"뒤$", "후", time_c)
                if time_c and compact(time_c) in compact(term_c):
                    continue  # 시점이 검사 chunk 안에 이미 있다
                base = f"{time_c} {term_c}".strip()
                if suffix == "결과 안내":
                    out.append(f"{base} 결과 안내".replace("결과 결과", "결과"))
                elif suffix and not base.endswith(suffix):
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
                    for d in durs[:2]:  # `저녁에 약 한 알` — 시점 + 약 + 용량(#122)
                        if compact(d) not in compact(med) and compact(d) not in compact(cond):
                            out.append(f"{cond} {med} {d}")
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


# ── 소견 용어 후처리 ────────────────────────────────────────────────────────────
# 용어 뒤가 이렇게 시작하면 용어가 문장의 머리다(`충치가 두 개 있음`, `식도염은 이전과 비슷함`,
# `바이러스성임`, `요로결석 4mm`, `역류성 식도염 소견은 …`). `감기 뒤 피로`처럼 다른 명사가 이어지면
# 용어는 수식일 뿐이라 건드리지 않는다
_TERM_HEAD_TAIL = re.compile(r"^(?:\s*소견)?(?:[이가은는도만]|이?[라란]|임|이었|였|인|\s*\d)")
_STAGE = re.compile(r"^\s*(초기|중기|말기|전단계)")
# 값 속 부정. `코 안은`의 `안`(명사)은 뒤에 조사가 붙어 걸리지 않는다
_NEG_IN_VALUE = re.compile(r"않|없|아니|아님|(?:^|\s)(?:안|못)\s|지\s*마")


def _is_measure_term(term) -> bool:
    """수치 소견(혈압·혈당·간수치·갑상선 수치)은 `높음`·`흔들림` 같은 서술이 곧 값이다."""
    return "수치" in term.note or "수치" in term.surface


def _strip_topic_prefix(prefix: str) -> str:
    """용어 앞 수식에서 조사로 끝나는 말(`어금니에`, `열은`)까지 버린다.

    `작은`·`오른쪽` 같은 수식은 남긴다.
    """
    prefix = prefix.strip()
    if not prefix:
        return ""
    try:
        from medimate.text.tokenize import base_kiwi

        toks = list(base_kiwi().tokenize(prefix))
    except Exception:  # noqa: BLE001 — Kiwi 없으면 수식은 그대로
        return prefix
    last_j = max((i for i, t in enumerate(toks) if t.tag.startswith("J")), default=-1)
    if last_j < 0:
        return prefix
    cut = toks[last_j].start + toks[last_j].len
    return prefix[cut:].strip()


_GEO_TAIL = {"VCP", "EF", "SF"}


def geo_nominal(value: str) -> str:
    """소견 값 끝의 `-ㄴ 거(래)`를 `-ㅁ`으로: `인대가 놀란 거래` → `인대가 놀람`(#122, r4).

    gold 모양 "서술문은 -음"(`어깨 근육이 뭉친 거` → `어깨 근육이 뭉침`)을 코드로. 모델이 조각의
    `-ㄴ 거래`를 그대로 옮긴 값에만 탄다. `것 같음`처럼 `거/것` 뒤에 다른 말이 오면 건드리지 않는다.
    """
    try:
        from medimate.text.tokenize import base_kiwi

        k = base_kiwi()
        toks = list(k.tokenize(value))
    except Exception:  # noqa: BLE001 — Kiwi 없으면 그대로
        return value
    for i in range(1, len(toks) - 1):
        etm, nnb = toks[i], toks[i + 1]
        if not (etm.tag == "ETM" and nnb.tag == "NNB" and nnb.form in ("거", "것")):
            continue
        if not all(t.tag in _GEO_TAIL for t in toks[i + 2 :]):
            return value
        # 서술어가 든 어절만 다시 붙인다 — 앞말의 띄어쓰기는 원문 그대로
        cut = value.rfind(" ", 0, etm.start) + 1
        stem = [(t.form, t.tag) for t in toks[:i] if t.start >= cut]
        if not stem or not stem[-1][1].startswith(("VV", "VA", "VX", "XSA", "XSV")):
            return value
        return (value[:cut] + k.join(stem + [("ᆷ", "ETN")])).strip()
    return value


def term_only(value: str, lexicon: Lexicon) -> str:
    """소견 값이 어휘집 용어를 머리로 한 서술문이면 용어(+앞 수식·단계 낱말)만 남긴다(#122).

    gold 검토의 모양 "병명·용어가 있으면 용어만"을 코드로 강제한다.
    모델이 `어금니에 충치가 두 개 있음`이라 써도 카드에는 `충치`.
    낱말을 지우기만 하므로 검증기가 통과시킨 값은 그대로 통과한다.

    건드리지 않는 것: 용어 없는 서술문(`허리 근육이 많이 뭉침`), 수치 소견(`혈당이 경계에 가까움`),
    용어가 머리가 아닌 것(`감기 뒤 피로`), 이미 용어(+단계)인 값(`허리디스크 초기`),
    부정이 있는 값(`충치는 깊지 않음`, `파열은 아님`) — 잘라내면 부정이 사라진다
    (2026-09-23 부정 정의)
    """
    if _NEG_IN_VALUE.search(value):
        return value
    hits = [m for m in lexicon.match(value) if m.term.type == "finding"]
    if not hits or any(_is_measure_term(m.term) for m in hits):
        return value
    m = hits[0]
    tail = value[m.end :]
    st = _STAGE.match(tail)
    if st:
        end, tail = m.end + st.end(), tail[st.end() :]
    else:
        end = m.end
    if not tail.strip() or not _TERM_HEAD_TAIL.match(tail):
        return value
    head = _strip_topic_prefix(value[: m.start])
    out = f"{head} {value[m.start : end]}".strip()
    return out or value
