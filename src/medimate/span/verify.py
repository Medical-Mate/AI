"""값 검증기 — 모델이 쓴 값의 **모든 낱말에 출처가 있는가**를 기계로 증명한다(#122, B안).

후보 열거(코드)는 처음 보는 메모에서 재현율 50% 근처가 천장이었다(두 라운드). 값의 모양이
"원문 낱말로 다시 쓰기"라서다. 그래서 모델이 값을 **쓰고**, 여기서 다섯 층으로 검증한다.
하나라도 걸리면 값을 버리고 폴백(rule → tidy_value)한다.

1. 낱말 출처 — 값의 내용 형태소(명사·수사·외래어·동사/형용사 어간·부사)는 전부 조각(원문·정규화문·
   띄운 문)의 형태소에 있거나, 축별 허용 표에 있어야 한다. 기능 형태소(조사·어미·명사형 접미)는
   자유 → 재배열·`-음` 명사형·조사 떼기가 통과한다
2. 숫자 — 값의 숫자는 조각에 있는 것만
3. 부정 — 두 방향. (a) 값의 부정(안·못·없·않·아니·-지 말)은 조각에도 있어야 한다.
   `ㅇㅇ → 안 퍼짐`이 여기서 떨어진다. (b) 부정이 있는 절에서 값을 가져왔으면 그 부정도 값에
   있어야 한다(2026-09-23 부정 정의). `코 안은 많이 부었지만 축농증은 아니라고` →
   `많이 부었지만 축농증`이 여기서 떨어진다. 절은 연결어미(-지만·-고·-는데·-면…)로 자르고,
   값의 내용 형태소가 걸치는 절을 "가져온 절"로 본다. 어느 절과도 안 겹치면(허용 표 낱말뿐:
   `재방문 필요`) 조각 전체를 가져온 것으로 본다
4. 용어 — 어휘집 용어가 값에 있으면 조각에도 글자 그대로 있어야 한다(D6과 같은 자리).
   추론(`인대 손상`)을 막는다
5. 길이 — 값의 형태소 수 ≤ 조각, 글자 수 ≤ MAX_CHARS

편집거리(spell.py)와 다른 축이다: 그쪽은 "얼마나 바꿨나", 여기는 "무엇으로 바꿨나".
`피곤→피견`은 여기서는 낱말 출처에서 떨어진다.

못 잡는 것: 원문 낱말만으로 결합을 바꾸는 경우(`커피 줄이고 약 2주분` → `커피 2주분`).
조각 층이 주제 하나로 자르는 데 기댄다. 실측에서 이 유형을 센다(215조각에서 0건, 2026-09-22).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from medimate.text.chatnorm import ABBREV as CHAT_ABBREV
from medimate.text.chatnorm import normalize
from medimate.text.lexicon import Lexicon
from medimate.text.tokenize import base_kiwi, tokens

MAX_CHARS = 30

# 내용 형태소 — 출처가 있어야 하는 것. 나머지(조사·어미·접미·기호·감탄)는 자유
_CONTENT_TAGS = ("NNG", "NNP", "NNB", "NR", "SN", "SL", "SH", "XR", "VV", "VA", "MAG", "MAJ", "MM")
_NEG_FORMS = {"안", "못", "없", "않", "아니", "말"}  # 말: `-지 말-`
# 절을 끊는 연결어미. Kiwi는 인용과 붙은 어미를 한 덩어리로 낸다(`먹으라는데` → `으라는데`).
# 그래서 끝으로 맞춘다.
# `-면`도 끊는다 — `숨차면 참지 말고 오라구`에서 조건(`숨차면`)은 부정 절이 아니다.
# `-지`(깊지 않)·`-게`·`-라고/-다고`(인용)는 끊지 않는다 — 부정과 한 덩어리다
_CLAUSE_EC_EXACT = {"고", "서", "데", "며", "으며", "면", "으면"}
_CLAUSE_EC_TAIL = ("지만", "는데", "은데", "ᆫ데", "아서", "어서", "니까", "면서", "다가")
# 절이 겹치는지 볼 때 세지 않는 내용 형태소 — 어느 절에나 있어서 출처를 가르지 못한다
_GENERIC = {"하", "있", "되", "것", "거", "수", "좀", "말", "다시"}

# 축별 허용 낱말. (낱말, 조각에 있어야 하는 근거 정규식 | None)
_ALLOWED: dict[str, list[tuple[str, str | None]]] = {
    "follow_up": [
        ("재방문", None),
        ("필요", None),
        ("재", None),
        ("방문", None),
        ("문의", r"물어|문의|연락"),
    ],
    "tests": [
        ("결과", r"결과|알려|나오|전화|문자"),
        ("안내", r"결과|알려|나오|전화|문자|설명"),
        ("예약", r"예약|잡"),
        ("예정", r"하자고|보자고|기로|예정|할 예정|하기로"),
        ("검사", r"검사|검진|재검"),
    ],
    "medication_instructions": [],
    "findings": [],
}
_COMMON: list[tuple[str, str | None]] = [
    ("후", r"뒤|후|있다가|지나"),
    ("추후", r"다음에|그때|나중에|추후|다음 진료|담에"),
    ("이상", r"이상|괜찮|정상"),
    ("없음", r"없|괜찮|정상"),
    ("없", r"없|괜찮|정상"),
]
# 축약 표의 오른쪽 낱말은 왼쪽이 조각에 있으면 허용
_ABBREV_PAIRS: list[tuple[str, str]] = [
    ("심해지면", "악화"),
    ("심해지면", "시"),
    ("다음에", "추후"),
    ("그때", "추후"),
    ("넘을 때만", "넘으면"),
] + [(k, v) for k, v in CHAT_ABBREV.items()]

_DIGITS = re.compile(r"\d+")
_WS = re.compile(r"\s+")


@dataclass
class Verdict:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    unsourced: list[str] = field(default_factory=list)  # 출처 없는 형태소

    @property
    def reason(self) -> str:
        return ";".join(self.reasons)


def _forms(text: str, kiwi) -> list[tuple[str, str]]:
    return [(t.form, t.tag) for t in tokens(kiwi, text)]


def _content(forms: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return [(f, t) for f, t in forms if t.startswith(_CONTENT_TAGS)]


def _neg(forms: list[tuple[str, str]]) -> list[str]:
    out = []
    for i, (f, t) in enumerate(forms):
        if f in _NEG_FORMS and (t.startswith(("MAG", "VA", "VCN", "VX", "VV"))):
            # `말`은 `-지 말`일 때만 부정
            if f == "말" and not (i > 0 and forms[i - 1][0] == "지"):
                continue
            out.append(f)
    return out


def _clauses(forms: list[tuple[str, str]]) -> list[list[tuple[str, str]]]:
    out: list[list[tuple[str, str]]] = [[]]
    for f, t in forms:
        out[-1].append((f, t))
        if t == "EC" and (f in _CLAUSE_EC_EXACT or f.endswith(_CLAUSE_EC_TAIL)):
            out.append([])
    return [c for c in out if c]


def _negation_dropped(v_forms: list[tuple[str, str]], seg_forms: list[tuple[str, str]]) -> bool:
    """부정이 있는 절에서 가져온 값인데 값에 부정이 그만큼 없는가."""
    v_content = {f for f, _ in _content(v_forms)} - _GENERIC
    clauses = _clauses(seg_forms)
    drawn = [c for c in clauses if v_content & ({f for f, _ in _content(c)} - _GENERIC)]
    if not drawn:
        drawn = clauses  # 허용 표 낱말뿐인 값 — 조각 전체에서 온 것으로 본다
    need = max((len(_neg(c)) for c in drawn), default=0)
    return len(_neg(v_forms)) < need


def _prepare(segment: str) -> tuple[str, str]:
    """출처 절(`알레르기내과에서`)을 떼고 정규화한다.

    거기서 낱말을 가져오면(`꽃가루 알레르기`) 추론이 된다.
    """
    from medimate.span.candidates import _SOURCE_PREFIX

    segment = _SOURCE_PREFIX.sub("", segment, count=1) or segment
    return segment, normalize(segment).text


def drops_negation(value: str | None, segment: str, kiwi=None) -> bool:
    """부정 불변식(3b)만.

    폴백(rule) 값은 검증을 타지 않아서 `축농증`이 여기로 새어 나갔다(r4 R402).
    """
    if value is None or not str(value).strip():
        return False
    kiwi = kiwi or base_kiwi()
    _, norm = _prepare(segment)
    return _negation_dropped(_forms(str(value), kiwi), _forms(norm, kiwi))


def verify_value(
    value: str | None, segment: str, axis: str, lexicon: Lexicon | None = None, kiwi=None
) -> Verdict:
    if value is None or not str(value).strip():
        return Verdict(True)  # NONE은 값이 없다는 답 — 검증할 것이 없다
    value = str(value).strip()
    kiwi = kiwi or base_kiwi()
    reasons: list[str] = []

    # 조각의 세 얼굴: 원문, 정규화문, 띄운 문. 어느 것에든 있으면 출처가 있다
    segment, norm = _prepare(segment)
    norm_forms = _forms(norm, kiwi)
    faces = {segment, norm}
    try:
        faces.add(kiwi.join([(t.form, t.tag) for t in kiwi.tokenize(norm)]))
    except Exception:  # noqa: BLE001
        pass
    seg_forms: list[tuple[str, str]] = []
    for f in faces:
        seg_forms += _forms(f, kiwi)
    seg_content = {f for f, _ in _content(seg_forms)}
    seg_text = " ".join(faces)

    # 1. 낱말 출처
    allowed = {
        w
        for w, cond in _ALLOWED.get(axis, []) + _COMMON
        if cond is None or re.search(cond, seg_text)
    }
    for src, dst in _ABBREV_PAIRS:
        if src in seg_text:
            allowed.update(f for f, _ in _content(_forms(dst, kiwi)))
            allowed.add(dst)
    v_forms = _forms(value, kiwi)
    v_content = _content(v_forms)
    unsourced = [f for f, _ in v_content if f not in seg_content and f not in allowed]
    # 어간 변형(찼/차, 늘어났/늘어나): Kiwi가 어간을 통일하므로 대개 같다. 남은 것은 글자 포함으로.
    # 값과 조각의 분절이 다를 때(`간수치`↔`간수/치`)는 공백 뺀 조각에 글자열로 있으면 출처로 본다
    seg_compact = _WS.sub("", seg_text)
    unsourced = [
        f
        for f in unsourced
        if not (len(f) >= 2 and f in seg_compact)
        and not any(f in s or s in f for s in seg_content if len(s) >= 2)
    ]
    if unsourced:
        reasons.append(f"unsourced:{','.join(unsourced)}")

    # 2. 숫자
    if set(_DIGITS.findall(value)) - set(_DIGITS.findall(seg_text)):
        reasons.append("digits")

    # 3. 부정
    # 형태가 아니라 개수로 센다 — `접질린 건 아니래` → `접질리지 않음`은 부정을 바꾼 게 아니다
    v_neg, s_neg = _neg(v_forms), _neg(seg_forms)
    if v_neg and (not s_neg or len(v_neg) > len(_neg(norm_forms))):
        reasons.append("negation_added")
    # 절은 정규화문 하나로 자른다(세 얼굴을 합치면 같은 절이 겹쳐 센다)
    if _negation_dropped(v_forms, norm_forms):
        reasons.append("negation_dropped")

    # 4. 용어 — 한 낱말 용어만. 띄어 쓴 용어(`공복혈당 검사`)는 조각의 낱말이 재배열돼 생길 수 있고,
    # 그 낱말들은 1에서 이미 출처를 확인했다
    if lexicon is not None:
        for m in lexicon.match(value):
            if " " in m.term.surface:
                continue
            if m.term.compact not in _WS.sub("", seg_text):
                reasons.append(f"term_not_in_segment:{m.term.surface}")
                break

    # 5. 길이 — 형태소 수는 가장 길게 분석된 얼굴 기준(무공백 원문은 덜 쪼개진다). 여유 +3
    seg_len = max(len(_forms(f, kiwi)) for f in faces)
    if len(v_forms) > seg_len + 3 or len(value) > MAX_CHARS:
        reasons.append("too_long")

    return Verdict(not reasons, reasons, unsourced)
