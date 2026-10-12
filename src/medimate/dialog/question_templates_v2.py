"""질문 후보 템플릿 v2 — 환자 말을 **문장 안에 녹인다**(2026-10-08). 서버 전용(Kiwi).

v1(`question_templates`)은 값을 따옴표로 통째로 인용하고 "는"을 붙였다("“가만히 있어도 아파요”는
어떻게 해야 하나요?"). 사람은 그렇게 말하지 않는다 — 사용자 판독에서 "너무 부자연스럽다". 그리고
칸마다
꼬리가 하나라 카드 82%에 "어떻게 해야 하나요"가 들어갔다.

v2
- 값의 **마지막 어미만** 연결형으로 바꾼다: 아파요→아픈데, 심해져요→심해지는데, 같아요→같은데,
  뻐근함→뻐근한데. 어간과 그 앞의 말은 원문 그대로다(Kiwi로 끝 서술어 묶음만 다시 붙인다)
- 서술어 없이 끝나는 값은 칸 뜻에 맞는 말만 붙인다: 악화 "계단 내려갈 때" → "…때 더 심한데",
  느낌 "칼로 찌르는 것처럼" → "…처럼 아픈데", 동반 "기침" → "기침도 있는데". 그 밖은 v1 인용으로
  돌아간다
- 꼬리는 칸마다 여럿, 값에서 정해지는 순번으로 고른다(같은 카드는 늘 같은 결과). 한 세트 안에서
  같은 꼬리는 두 번 쓰지 않는다
- 복용약·알러지·기저질환은 따옴표를 뺀다

2026-10-12 팀원 판독 반영
- 잇는 말을 칸마다 하나 더 둔다. 느낌 "~는 건", 동반 "~는 것도", 경과 "~면". 한 카드 안에서
  같은 잇는 말이 이어지지 않게 고른다(PC48 — 모두 "~한데 + 꼬리"라 틀처럼 들렸다)
- 경과는 조건형이 먼저다: "비슷해요" → "계속 비슷하면 다시 와야 하나요?"("비슷한데 계속 이러면"이
  12장). 과거 값은 가정으로 바꾸지 않고 "~는데"
- 느낌 값에 이름씨가 없으면 부위를 주어로 붙인다(PC77 "가끔 콕콕 찌르는데" — 가슴이 빠졌다)
- 악화 명사구는 "~에 더 심한데"로 받는다("아침 첫 걸음에"). 쉼표로 이어진 조각도 하나씩 맞춘다

앱 경로에서 템플릿은 서버가 만든다(추론만 폰, 검증·폴백은 서버). 그래서 Kiwi를 쓸 수 있다.
"""

from __future__ import annotations

import re
from typing import Any

from medimate.dialog.question_templates import _AXIS_TEMPLATES as _V1_AXIS
from medimate.dialog.question_templates import (
    _CLAUSE,
    _ENDS_NEGATIVE,
    MAX_ITEMS,
    MAX_LEN,
    MIN_ITEMS,
    MIN_VALUE,
    _as_question,
    _has_batchim,
    _usable,
)
from medimate.dialog.question_templates import _FILL_TEMPLATES as _V1_FILL
from medimate.text.tokenize import base_kiwi

# 끝 서술어 묶음에 들어가는 태그 — 어말어미(EF)에서 거꾸로 이 태그들을 걷는다
_PRED = ("VV", "VA", "VX", "VCP", "VCN", "XSA", "XSV")
# "있다·없다·계시다"는 형용사여도 "-는데"
_NEUN_STEMS = {"있", "없", "계시"}
_PAREN = re.compile(r"\s*\([^()]*\)\s*$")
_MID_DE = re.compile(r"(는데|은데|던데|ᆫ데)\s")
# 서술어 없이 끝나는 흉내말·느낌말("저릿", "뻐근", "부글부글") — "-한데"로 녹인다
_MIMETIC = re.compile(r"(릿|근|글|끔|큰|콕|쿡|끈|질|룩|직|뻑|따끔|화끈|욱신|쑤셔?)$")
_YEOT = re.compile(r"([가-힣])이(었|에요)")


def _contract(text: str) -> str:
    """Kiwi는 받침 없는 말 뒤 `이었`·`이에요`를 줄이지 않는다("나서이었는데", "그대로이에요")."""
    return _YEOT.sub(
        lambda m: (
            m.group(0)
            if _has_batchim(m.group(1))
            else m.group(1) + {"었": "였", "에요": "예요"}[m.group(2)]
        ),
        text,
    )


_TOPICAL = re.compile(r"(바르|안약|연고|파스|크림|스프레이|패치|붙이)")

# 끝 어미를 바꿀 꼴. 데 — 연결("아픈데"), 관형 — 뒤에 "건"·"것도"가 온다("아픈 건"),
# 면 — 조건("비슷하면"). (동사·과거 뒤, 형용사 뒤, 태그)
_ENDINGS = {
    "데": ("는데", "ᆫ데", "EC"),
    "관형": ("는", "ᆫ", "ETM"),
    "면": ("면", "면", "EC"),
}
_MIMETIC_TAIL = {"데": "한데", "관형": "한", "면": "하면"}


def _inflect(value: str, kiwi, kind: str = "데") -> str | None:
    """값의 끝 어미를 `kind` 꼴로 바꾼 절. 서술어로 끝나지 않으면 None.

    관형·면은 과거 값("심해졌어요")에는 만들지 않는다 — "심해졌으면"은 있었던 일을 가정으로 바꾼다.
    면은 값 안에 이미 조건("한 번 생기면")이 있으면 만들지 않는다."""
    # 값 안에 이미 "-는데"가 있으면 하나 더 붙여 "나는데 … 갈라지는데"가 된다 — 녹이지 않는다.
    # 조건형은 겹치지 않는다("부기는 빠졌는데 걸리는 건 여전하면")
    if kind != "면" and _MID_DE.search(value):
        return None
    if kind == "면" and re.search(r"면\s", value):
        return None
    if kind == "관형" and re.search(r"(것|거)\s*같", value):
        return None  # "번지는 것 같은 건"
    toks = [t for t in kiwi.tokenize(value) if not t.tag.startswith(("SF", "SP", "SE", "SS"))]
    if toks and toks[-1].tag in ("XR", "MAG") and _MIMETIC.search(toks[-1].form):
        # "다리로 저릿" → "저릿한데"
        return f"{value.rstrip(' .,~!')}{_MIMETIC_TAIL[kind]}"
    if not toks or toks[-1].tag != "EF":
        return None
    i = len(toks) - 1  # EF
    j = i - 1
    # EF 앞의 선어말어미를 건너고, 보조용언은 앞에 연결어미가 있을 때만(심해/지/었, 쉬어야/하)
    # 건넌다.
    # 앞이 연결어미가 아니면 그 보조용언이 본 서술어다("붓기도 해요"의 `하`)
    while j >= 0:
        if toks[j].tag == "EP":
            j -= 1
        elif toks[j].tag.startswith("VX") and j >= 1 and toks[j - 1].tag == "EC":
            j -= 2
        else:
            break
    if j < 0 or not toks[j].tag.startswith(_PRED):
        return None
    # 줄어든 서술격("그대로예요"의 `이`, 길이 0)은 앞말과 같이 다시 붙여야 줄어든다("그대로면")
    if toks[j].tag == "VCP" and toks[j].len == 0 and j >= 1:
        j -= 1
    chain = toks[j:i]

    def morphs(ts):  # 원문 띄어쓰기를 살린다("쉬어야 하는데"). 축약된 어미(`해요`의 어요)는 어간과
        # 같은 자리에서 시작하므로 앞 형태소가 끝난 뒤에 시작할 때만 띄어쓰기를 본다
        out = []
        for k, t in enumerate(ts):
            prev = ts[k - 1] if k else None
            sp = (
                bool(prev)
                and t.start >= prev.start + prev.len
                and value[t.start - 1 : t.start] == " "
            )
            out.append((t.form, t.tag, sp))
        return out

    # 자기 검사: 분석한 그대로 다시 붙여 원문이 나와야 한다. Kiwi가 축약을 잘못 풀면("쓰려요" →
    # 쓰/려고/하/어요) 바꿔 붙인 결과도 틀린다("쓰려고 하는데") — 그때는 녹이지 않는다
    head = value[: chain[0].start]
    again = _contract(head + kiwi.join(morphs(chain + [toks[i]])))  # 축약은 앞 글자를 봐야 한다
    if again.replace(" ", "") != value.rstrip(" .,~!?").replace(" ", ""):
        return None
    past = any(t.tag == "EP" for t in chain)
    if past and kind != "데":
        return None
    # 앞 절의 과거도 가정으로 바꾸지 않는다("어제가 제일 심했고 오늘은 조금 덜하면"). 배경으로 깐
    # "~었는데"는 둔다("부기는 빠졌는데 걸리는 건 여전하면")
    if kind == "면" and any(
        t.tag == "EP" and not (n < len(toks) and toks[n].form in ("는데", "은데"))
        for n, t in enumerate(toks, 1)
    ):
        return None
    last_pred = [t for t in chain if t.tag.startswith(_PRED)][-1]
    # "느낌이에요"·"중이에요"를 "느낌인 건"·"중인 것도"로 받으면 어색하다 — "~인데"로 둔다
    if kind == "관형" and last_pred.tag == "VCP":
        return None
    verbish = last_pred.tag.startswith(("VV", "VX", "XSV")) or last_pred.form in _NEUN_STEMS
    neun, n, tag = _ENDINGS[kind]
    ending = neun if (past or verbish) else n
    rebuilt = kiwi.join(morphs(chain) + [(ending, tag, False)])
    # Kiwi는 받침 없는 말 뒤 `이었`을 줄이지 않는다("나서이었는데", "그대로이었는데")
    return _contract(head + rebuilt).strip()


def _connective(value: str, kiwi) -> str | None:
    """값의 끝 어미를 연결어미(-는데/-ㄴ데)로 바꾼 절. 서술어로 끝나지 않으면 None."""
    return _inflect(value, kiwi, "데")


# 장소 이름씨 — 악화 명사구에 "에" 대신 "에서"를 붙인다("울퉁불퉁한 길에서")
_PLACE = re.compile(r"(길|곳|밖|바닥|회사|학교|집|가게|식당)$")
_ADVERBIAL = re.compile(r"(때|면|에|에서|뒤|후|날|하고|해도)$")


def _ends_noun(v: str, kiwi, tags: tuple[str, ...] = ("NNG", "NNP")) -> bool:
    toks = [t for t in kiwi.tokenize(v) if not t.tag.startswith(("SF", "SP", "SE", "SS"))]
    return bool(toks) and toks[-1].tag in tags


def _adverbial(v: str, kiwi) -> str | None:
    """악화 값 한 조각을 "언제·어디서"로 읽히게. "아침 첫 걸음" → "아침 첫 걸음에".
    "뭘 먹으면 바로 화장실"처럼 서술어가 빠진 조건은 이름씨로 끝나도 붙이지 않는다(None)."""
    v = v.rstrip(" .,~!")
    if _ADVERBIAL.search(v):
        return v
    if re.search(r"면\s", v) or not _ends_noun(v, kiwi):
        return None
    return f"{v}{'에서' if _PLACE.search(v) else '에'}"


def _glue(source: str, value: str, kiwi=None, kind: str = "데") -> str | None:
    """서술어 없이 끝나는 값에 칸 뜻에 맞는 말을 붙여 절로 만든다. 맞는 게 없으면 None.
    kind — 데: "…처럼 아픈데", 관형: "…처럼 아픈"(뒤에 "건"이 온다)."""
    v = value.rstrip(" .,~!")
    if source == "exacerbating" and kind == "데":
        if a := _adverbial(v, kiwi or base_kiwi()):
            return f"{a} 더 심한데"
    if source == "character" and re.search(r"(처럼|듯|듯이)$", v):
        return f"{v} {'아픈데' if kind == '데' else '아픈'}"
    if source == "character" and v.endswith("느낌"):
        return f"{v}이 {'드는데' if kind == '데' else '드는'}"
    if source == "radiation" and kind == "데" and re.search(r"(까지|으로|로|쪽)$", v):
        return f"{v} 퍼지는데"
    # 이름씨로 끝나는 짧은 값만("기침", "콧물"). 어미로 끝나는 값("붓기도 해요")은 여기 오면 안 된다
    if (
        source == "associated"
        and re.fullmatch(r"[가-힣·\s]+", v)
        and len(v) <= 12
        and not re.search(r"(요|다|함|음)$", v)
    ):
        do = "" if v.endswith("도") else "도"
        # 데: "기침도 있는데 …", 관형 자리(것도): "기침도 관련이 있나요?" — "것"을 붙이지 않는다
        return f"{v}{do} 있는데" if kind == "데" else f"{v}{do}"
    return None


# 칸별 문장 꼴 — (잇는 말, 꼬리들). 꼬리는 값에서 정해지는 순번으로 하나를 고르고, 35자를 넘거나
# 세트 안에서 이미 썼으면 다음 것
# - 경과는 조건형이 먼저다. 과거 값("심해졌어요")은 가정으로 바꾸지 않고 "~는데"로 받는다
# - "그럴 땐 피해야 하나요"는 뺐다 — 기침·밤·샤워처럼 피할 수 없는 상황에도 붙었다
_FORMS: dict[str, list[tuple[str, list[str]]]] = {
    "exacerbating": [
        ("데", ["어떻게 하면 되나요?", "조심할 게 있나요?", "그럴 때 어떻게 해야 하나요?"]),
    ],
    "character": [
        ("데", ["어떤 상태인가요?", "걱정할 정도인가요?", "괜찮은 건가요?"]),
        ("건", ["어떤 상태인가요?", "걱정할 정도인가요?", "신경 써야 하나요?"]),
    ],
    "associated": [
        ("데", ["이것도 관련이 있나요?", "같은 원인인가요?"]),
        ("것도", ["관련이 있나요?", "같은 원인인가요?"]),
    ],
    "time_course": [
        ("면", ["다시 와야 하나요?", "언제 다시 와야 하나요?", "그냥 지켜봐도 되나요?"]),
        (
            "데",
            [
                "언제까지 이러면 다시 와야 하나요?",
                "계속 이러면 다시 와야 하나요?",
                "괜찮은 건가요?",
            ],
        ),
    ],
    "radiation": [("데", ["같은 증상인가요?", "관련이 있나요?"])],
}
_ORDER = ["exacerbating", "character", "associated", "time_course", "radiation"]
# 흐름을 말하는 부사가 없는 한 낱말 경과("비슷해요", "그대로예요")는 "계속"을 앞에 둔다 —
# "비슷하면"만으로는 무엇과 비슷한지 모른다
_COURSE_ADV = re.compile(r"계속|점점|조금씩|자꾸|늘|요즘|여전")


def _pick(clause: str, tails: list[str], used: set[str]) -> str | None:
    start = sum(map(ord, clause)) % len(tails)
    for k in range(len(tails)):
        tail = tails[(start + k) % len(tails)]
        text = f"{clause} {tail}"
        if tail not in used and len(text) <= MAX_LEN:
            return tail
    return None


def _strip_paren(v: str) -> str:
    return _PAREN.sub("", v).strip() or v


def _no_noun(v: str, kiwi) -> bool:
    return not any(t.tag in ("NNG", "NNP") for t in kiwi.tokenize(v))


def _site_np(site: Any, kiwi) -> str | None:
    """부위 값의 앞 절이 이름씨로 끝나면 주어로 붙일 꼴("가슴 가운데가"). 아니면 None —
    "특정 부위 없음", "왼쪽 무릎만", "몸 전체가 무거워요"는 붙이지 않는다."""
    s = _usable(site)
    if not s:
        return None
    s = _strip_paren(_CLAUSE.split(s)[0].strip())
    # "명치 아래쪽"의 `쪽`은 의존명사다
    if len(s) < 2 or "없" in s or not _ends_noun(s, kiwi, ("NNG", "NNP", "NNB")):
        return None
    return f"{s}{'이' if _has_batchim(s) else '가'}"


def _clause(source: str, v: str, joint: str, kiwi) -> str | None:
    if joint == "데":
        return _connective(v, kiwi) or _glue(source, v, kiwi)
    if joint == "면":
        c = _inflect(v, kiwi, "면")
        if c and " " not in v.strip() and not _COURSE_ADV.search(v):
            c = f"계속 {c}"
        return c
    # 건·것도 — 관형형 뒤에 붙인다. 이름씨 동반("기침")은 _glue가 "기침도"로 바로 준다.
    # 값에 이미 `도`가 있으면 "재채기도 자꾸 하는 것도"가 된다 — 것도는 만들지 않는다
    if joint == "것도" and any(t.form == "도" and t.tag == "JX" for t in kiwi.tokenize(v)):
        return None
    if c := _inflect(v, kiwi, "관형"):
        return f"{c} {'건' if joint == '건' else '것도'}"
    g = _glue(source, v, kiwi, "관형")
    if g and joint == "건":
        return f"{g} 건"
    return g if g and joint == "것도" and g.endswith("도") else None


def _axis_question(
    source: str,
    value: str,
    used: set[str],
    kiwi,
    joints: set[str] | None = None,
    site: Any = None,
) -> tuple[str, str, str] | None:
    """(질문, 쓴 꼬리, 잇는 말). 값 전체 → 첫 절 순으로 녹여 본다. 안 되면 v1 인용 꼴."""
    joints = joints if joints is not None else set()
    first = _CLAUSE.split(value)[0].strip()
    whole = value
    if "." in value.rstrip("."):
        whole = first  # "화면 오래 보면. 주말엔 괜찮아요" — 문장이 둘이면 앞 절만
    elif source == "exacerbating" and "," in value:
        parts = [x.strip() for x in value.split(",") if x.strip()]
        # "허리 숙일 때, 기침할 때" → "허리 숙일 때나 기침할 때". 꼴이 다르면 조각마다
        # "언제·어디서"로 맞춰 쉼표로 잇는다("아침 첫 걸음에, 오래 서 있은 다음 날").
        # 하나라도 못 맞추면 앞 절만
        if all(x.endswith("때") for x in parts):
            whole = "나 ".join(parts)
        else:
            adv = [_adverbial(x, kiwi) for x in parts]
            whole = ", ".join(a for a in adv if a) if all(adv) else first
    forms = _FORMS[source]
    if len(forms) > 1 and source != "time_course":
        # 값에서 정해지는 순번으로 돌리고, 이 카드에서 아직 안 쓴 잇는 말을 앞에 둔다
        k = sum(map(ord, value)) % len(forms)
        forms = sorted(forms[k:] + forms[:k], key=lambda f: f[0] in joints)
    # 느낌 값에 이름씨가 하나도 없으면("가끔 콕콕 찔러요") 부위를 주어로 붙인다(PC77 — "가슴"이
    # 빠졌다). 35자를 넘으면 부위 없이
    np = _site_np(site, kiwi) if source == "character" else None
    for joint, tails in forms:
        for v in dict.fromkeys((whole, first)):
            if len(v) < MIN_VALUE:
                continue
            clause = _clause(source, v, joint, kiwi)
            if not clause:
                continue
            heads = [f"{np} {clause}", clause] if np and _no_noun(v, kiwi) else [clause]
            for h in heads:
                if tail := _pick(h, tails, used):
                    return f"{h} {tail}", tail, joint
    for src, makes in _V1_AXIS:  # v1 인용 꼴로 돌아간다
        if src != source:
            continue
        for make in makes:
            for v in (value, first):
                if len(v) >= MIN_VALUE and len(t := make(v)) <= MAX_LEN:
                    return t, t, "인용"
    return None


def _profile_question(source: str, name: str) -> str | None:
    n = _strip_paren(name)
    if source == "medications":
        verb = "써도" if _TOPICAL.search(name) else "먹어도"
        text = f"{n}{'은' if _has_batchim(n) else '는'} 계속 {verb} 되나요?"
    elif source == "allergies":
        text = f"{n} 알러지가 있는데 처방에 반영되나요?"
    else:
        text = f"{n}{'이' if _has_batchim(n) else '가'} 있는데 이번 증상과 관련이 있나요?"
    return text if len(text) <= MAX_LEN else None


def _onset_question(value: str) -> str | None:
    v = _CLAUSE.split(value)[0].strip().removesuffix("요").strip()
    if len(v) < 2 or not re.search(r"(부터|전|쯤|째|때|됐어|넘었어)$", v):
        return None
    v = re.sub(r"(됐어|넘었어)$", lambda m: {"됐어": "됐는데", "넘었어": "넘었는데"}[m.group(1)], v)
    if v.endswith("는데"):
        text = f"{v} 문제가 되나요?"
    elif re.search(
        r"(쯤|째)$", v
    ):  # "3주쯤" — 기간이다. "3주쯤 시작됐는데"가 아니라 "3주쯤 됐는데"
        text = f"{v} 됐는데 문제가 되나요?"
    else:
        text = f"{v} 시작됐는데 문제가 되나요?"
    return text if len(text) <= MAX_LEN else None


def natural_candidates(
    axes: dict[str, Any], profile: dict[str, list[str]] | None = None, kiwi=None
) -> list[dict]:
    """카드 값 → 질문 후보 [{text, source}]. v1과 같은 칸·같은 순서·같은 개수 규칙, 문장만
    다르다."""
    kiwi = kiwi or base_kiwi()
    items: list[dict] = []
    used: set[str] = set()
    joints: set[str] = set()
    seen: set[str] = set()
    for source in ("medications", "allergies", "conditions"):
        for name in (profile or {}).get(source) or []:
            v = _usable(name)
            if v and (text := _profile_question(source, v)):
                items.append({"text": text, "source": source})
                break
        if items:
            break
    for source in _ORDER:
        if len(items) >= MAX_ITEMS:
            break
        v = _usable(axes.get(source))
        if not v:
            continue
        if source in ("associated", "radiation") and _ENDS_NEGATIVE.search(v):
            continue
        if any(v in it["text"] for it in items) or v in seen:  # 같은 말을 두 칸에 적었으면 한 번만
            continue
        seen.add(v)
        if q := _as_question(v):
            items.append({"text": q, "source": source})
        elif r := _axis_question(source, v, used, kiwi, joints, axes.get("site")):
            items.append({"text": r[0], "source": source})
            used.add(r[1])
            joints.add(r[2])
    if len(items) < MIN_ITEMS:
        v = _usable(axes.get("onset"))
        if v:
            text = _onset_question(v)
            if not text:
                for _src, makes in _V1_FILL:
                    text = next((t for m in makes if len(t := m(v)) <= MAX_LEN), None)
            if text:
                items.append({"text": text, "source": "onset"})
    return items
