"""후보 span 생성 — 형태소(Kiwi) + 정규식 + 어휘집. 호출 0.

한 조각에서 값이 될 수 있는 자리를 **전부** 만든다. 고르는 것은 `select.py`의 일이다.
중요한 지표는 Candidate Recall — 정답이 이 목록에 있는가. 없으면 어떤 선택기도 못 맞춘다.

후보 종류(kind)
- lexicon:<type>  어휘집 최장 일치 (위산약, 역류성 식도염)
- duration        기간·횟수·날짜 (2주치, 3주 후, 하루 두 번, 10월 2일)
- chunk           명사류 토큰의 연속 (위산약 2주치, 무릎 인대)
- phrase          chunk + 조사 + chunk (약은 2주분)
- tail:<type>     어휘집 매치에서 chunk 끝까지 (위산약 2주치 — chunk와 같으면 kind가 합쳐진다)
- derived         지금의 tidy_value 결과. **부분 문자열이 아닐 수 있다** — derived=True로 표시
- whole           조각 전체(끝 구두점만 뗌). 다른 후보와 같으면 그 후보에 kind로 붙는다
- canon           템플릿 후보(canon.py). 원문 조각을 재배열·정해진 표로 축약. derived=True

순서는 원문 등장 순(start, 긴 것 먼저). ID는 그 순서로 C01…. 선택기가 순서에 민감할 수 있어
(Jev 문서) 여기서 고정한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from medimate.text.chatnorm import normalize
from medimate.text.lexicon import Lexicon, load_lexicon
from medimate.text.tokenize import NOMINAL_TAGS, Tok, build_kiwi, tokens

MAX_SUBSTRING = 8  # 부분 문자열 후보 상한. derived·whole은 별도

_WS = re.compile(r"\s+")
_TRAIL = re.compile(r"[\s·,.。!?]+$")
_LEAD = re.compile(r"^[\s·,.。!?]+")

_NUM = r"(?:\d+|한|두|세|네|다섯|여섯|일곱|여덟|아홉|열|하루|이틀|사흘|나흘|열흘|보름)"
_UNIT = r"(?:주일|개월|주|일|달|년|번|회|알|정|mm|리터|도)"
_WORD = (
    r"(?:일주일|이주일|삼주일|일주|이주|삼주|한주일|두주일|세주일|열흘|보름|사흘|나흘|하루|이틀)"
)
# 기간·용량: `2주치` `2주분` `3주 후` `일주일` `한 달 뒤` `하루 두 번` `4mm` `500mg` `38도`
_DURATION = re.compile(
    rf"(?:{_WORD}|{_NUM}\s*{_UNIT})(?:\s*(?:치|분|일치|정도))?(?:\s*(?:후|뒤|이상|이내|마다))?"
    rf"|\d+\s*mg"
    rf"|(?:하루|매일|아침|저녁|식후|식전)(?:\s*{_NUM}\s*(?:번|회|알|정))+"
)
# 날짜: `10월 2일` `다음 달 15일` `다음 주 화요일` `다다음주 월요일`
_DATE = re.compile(
    r"\d{1,2}\s*월\s*\d{1,2}\s*일"
    r"|다음\s*달\s*\d{1,2}\s*일"
    r"|다?다음\s*달"
    r"|다?다음\s*주(?:\s*(?:월|화|수|목|금|토|일)요일)?"
    r"|(?:월|화|수|목|금|토|일)요일"
)
_PARTICLES = frozenset({"JX", "JKS", "JKO", "JKG", "JKB", "JC"})
# 출처 접두 절: `산부인과에서`, `건강검진 결과 상담에서`, `응급실에서`, `한의원에서맥이…`(무공백)
_SOURCE_PREFIX = re.compile(
    r"^\s*(?:[가-힣A-Za-z]+\s*){0,2}?[가-힣A-Za-z]*"
    r"(?:과|의원|병원|응급실|센터|약국|검진|상담|진료실|외래|클리닉)\s*(?:에서|에선)\s*"
)


@dataclass(frozen=True)
class Candidate:
    id: str
    text: str  # 원문에서 잘라낸 그대로
    start: int
    end: int
    kinds: tuple[str, ...]
    derived: bool = False  # True면 부분 문자열이 아닐 수 있다(tidy_value 결과)

    @property
    def compact(self) -> str:
        return _WS.sub("", self.text)


@dataclass
class CandidateSet:
    segment: str  # 후보의 좌표가 가리키는 텍스트. 정규화가 일어났으면 정규화문
    candidates: list[Candidate]
    overflow: list[str] = field(default_factory=list)  # 상한에 걸려 버린 것
    raw: str | None = None  # 정규화 전 원문. None이면 segment가 원문

    def ids(self) -> list[str]:
        return [c.id for c in self.candidates]

    def contains(self, text: str) -> Candidate | None:
        """공백·끝 구두점 무시로 같은 후보를 찾는다. Candidate Recall이 쓰는 비교."""
        want = compact(text)
        for c in self.candidates:
            if c.compact == want:
                return c
        return None


def compact(s: str) -> str:
    return _WS.sub("", _TRAIL.sub("", _LEAD.sub("", s)))


class CandidateGenerator:
    def __init__(self, lexicon: Lexicon | None = None, kiwi=None):
        self.lexicon = lexicon or load_lexicon()
        self.kiwi = kiwi if kiwi is not None else build_kiwi(self.lexicon)[0]

    def generate(self, segment: str, axis: str | None = None) -> CandidateSet:
        # 채팅 정규화(오타·표지·초성)는 표로만 바꾼다. 바뀌었으면 정규화문 위에서 후보를 만든다.
        # `감기레요`를 Kiwi가 `감기레/NNG`로 읽는 것을 막는 유일한 자리다. 원문은 CandidateSet.raw에
        norm = normalize(segment)
        seg = norm.text if norm.changed else segment
        # 조각 앞의 `[진료과·장소]에서`는 출처이지 값이 아니다(#122). 처음 보는 메모 34조각 중 9개가
        # 여기서 후보를 잃었다. 뗀 뒤의 텍스트 위에서 후보를 만들고, 원문은 raw에 남는다
        seg = _SOURCE_PREFIX.sub("", seg, count=1) or seg
        # 띄어쓰기 없는 조각(`새약은반알부터먹고`)은 Kiwi로 띄운다(#122). 비교는 공백을 무시하므로
        # 채점에는 영향이 없고, chunk 경계가 살아난다. 8자 미만은 그대로(짧은 초성·용어)
        if " " not in seg and len(seg) >= 8:
            respaced = self.kiwi.join([(t.form, t.tag) for t in self.kiwi.tokenize(seg)])
            if _WS.sub("", respaced) == _WS.sub("", seg):
                seg = respaced
        raw: list[tuple[int, int, str]] = []  # (start, end, kind)

        for m in self.lexicon.match(seg):
            raw.append((m.start, m.end, f"lexicon:{m.term.type}"))

        for rx in (_DURATION, _DATE):
            for m in rx.finditer(seg):
                if compact(m.group(0)):
                    raw.append((m.start(), m.end(), "duration"))

        toks = tokens(self.kiwi, seg)
        chunks = _chunks(toks)
        for c in chunks:
            raw.append((c[0], c[1], "chunk"))
        # 어휘집 매치 → 그 chunk 끝까지 (`위산약` → `위산약 2주치`)
        for s, e, kind in list(raw):
            if kind.startswith("lexicon:"):
                for cs, ce in chunks:
                    if cs <= s < ce and ce > e:
                        raw.append((s, ce, "tail:" + kind.split(":")[1]))
        # chunk + 조사 + chunk (`약은 2주분`)
        for (s1, e1), (s2, e2) in zip(chunks, chunks[1:], strict=False):
            between = [t for t in toks if e1 <= t.start and t.end <= s2]
            if between and all(t.tag in _PARTICLES for t in between):
                raw.append((s1, e2, "phrase"))

        merged = _merge(seg, raw)
        merged.sort(key=lambda x: (x[0], -(x[1] - x[0])))

        whole = _TRAIL.sub("", _LEAD.sub("", seg))
        whole_c = _WS.sub("", whole)
        # 조각 전체와 같은 후보(`항생제 5일.`의 chunk `항생제 5일`)는 빼지 않고 kind에 whole을
        # 더한다. 빼 버리면 선택기가 chunk를 찾아도 없어서 약 이름만 고른다(gold 검토 2026-09-21)
        subs = []
        whole_merged = False
        for s, e, k in merged:
            if _WS.sub("", seg[s:e]) == whole_c:
                subs.append((s, e, k + ("whole",)))
                whole_merged = True
            else:
                subs.append((s, e, k))

        overflow: list[str] = []
        if len(subs) > MAX_SUBSTRING:
            ranked = sorted(subs, key=lambda x: (_priority(x[2]), -(x[1] - x[0])))
            keep = set(ranked[:MAX_SUBSTRING])
            overflow = [seg[s:e] for s, e, k in subs if (s, e, k) not in keep]
            subs = [x for x in subs if x in keep]

        cands: list[Candidate] = []
        for s, e, kinds in subs:
            cands.append(Candidate("", seg[s:e], s, e, kinds))

        derived = _derived(seg, axis)
        if derived and _WS.sub("", derived) not in {c.compact for c in cands} | {whole_c}:
            cands.append(Candidate("", derived, -1, -1, ("derived",), derived=True))
        if not whole_merged:
            start = seg.index(whole) if whole in seg else 0
            cands.append(Candidate("", whole, start, start + len(whole), ("whole",)))

        cset = CandidateSet(seg, cands, overflow)
        if axis:
            from medimate.span.canon import canon_candidates  # 순환 import 회피

            cands = cands + canon_candidates(cset, axis, self.lexicon)

        out = [
            Candidate(f"C{i + 1:02d}", c.text, c.start, c.end, c.kinds, c.derived)
            for i, c in enumerate(cands)
        ]
        return CandidateSet(seg, out, overflow, raw=segment if seg != segment else None)


def _chunks(toks: list[Tok]) -> list[tuple[int, int]]:
    """명사류 토큰의 연속을 원문 좌표 구간으로. 사이 공백은 허용, 다른 태그가 오면 끊는다."""
    out: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = None
    for t in toks:
        if t.tag in NOMINAL_TAGS:
            cur = (t.start, t.end) if cur is None else (cur[0], t.end)
        else:
            if cur:
                out.append(cur)
            cur = None
    if cur:
        out.append(cur)
    # 한 글자 chunk(`물`, `열`)는 후보로 두지 않는다 — 어휘집이 한 글자를 안 받는 것과 같은 선.
    # `약`만 예외다. "약은 2주분", "2주분 약"의 약은 내용이다(gold 검토 2026-09-21)
    return [(s, e) for s, e in out if e - s >= 2 or toks_text(toks, s, e) == "약"]


def toks_text(toks: list[Tok], s: int, e: int) -> str:
    return "".join(t.form for t in toks if t.start >= s and t.end <= e)


def _merge(seg: str, raw: list[tuple[int, int, str]]) -> list[tuple[int, int, tuple[str, ...]]]:
    """같은 텍스트(공백 무시)를 가리키는 후보를 하나로 합치고 kind를 모은다. 빈 것은 버린다."""
    by_text: dict[str, tuple[int, int, list[str]]] = {}
    for s, e, k in raw:
        text = seg[s:e]
        # 앞뒤 공백·구두점을 좌표째로 뗀다
        while s < e and seg[s] in " \t·,.。!?":
            s += 1
        while e > s and seg[e - 1] in " \t·,.。!?":
            e -= 1
        key = _WS.sub("", seg[s:e])
        if len(key) < 2:
            continue
        if key in by_text:
            by_text[key][2].append(k)
        else:
            by_text[key] = (s, e, [k])
        del text
    return [(s, e, tuple(dict.fromkeys(ks))) for s, e, ks in by_text.values()]


_PRIORITY = {"lexicon": 0, "duration": 1, "tail": 2, "chunk": 3, "phrase": 4}


def _priority(kinds: tuple[str, ...]) -> int:
    return min(_PRIORITY.get(k.split(":")[0], 9) for k in kinds)


def _derived(seg: str, axis: str | None) -> str | None:
    from medimate.dialog.memo import tidy_value
    from medimate.schema.postvisit import PostAxis

    ax = None
    if axis:
        try:
            ax = PostAxis(axis)
        except ValueError:
            ax = None
    out = tidy_value(seg, ax)
    return out if out and compact(out) != compact(seg) else None
