"""한/영 키보드 오타 복원 — 영문 자판(두벌식 자리)으로 친 글자를 한글로 조합한다.

"qo" → "배", "audcl" → "명치", "duvrnfl" → "옆구리". 부위 폼 검색에서 한글 전환을 잊고 친 입력용.
LLM·사전 없이 결정론. 영문 실제 단어(MRI 등)와 구분하지 않는다.
그래서 `Ontology.search`는 원문 결과가 0건일 때만 복원을 쓴다.
"""

from __future__ import annotations

# 두벌식 자판: 영문 키 → 자모 (shift 자리는 대문자)
_KEY_TO_JAMO = {
    "q": "ㅂ", "w": "ㅈ", "e": "ㄷ", "r": "ㄱ", "t": "ㅅ",
    "y": "ㅛ", "u": "ㅕ", "i": "ㅑ", "o": "ㅐ", "p": "ㅔ",
    "a": "ㅁ", "s": "ㄴ", "d": "ㅇ", "f": "ㄹ", "g": "ㅎ",
    "h": "ㅗ", "j": "ㅓ", "k": "ㅏ", "l": "ㅣ",
    "z": "ㅋ", "x": "ㅌ", "c": "ㅊ", "v": "ㅍ", "b": "ㅠ", "n": "ㅜ", "m": "ㅡ",
    "Q": "ㅃ", "W": "ㅉ", "E": "ㄸ", "R": "ㄲ", "T": "ㅆ", "O": "ㅒ", "P": "ㅖ",
}  # fmt: skip

_CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_JONG = "ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"  # 인덱스 1부터. 0은 종성 없음

_VOWEL_COMPOUND = {
    ("ㅗ", "ㅏ"): "ㅘ", ("ㅗ", "ㅐ"): "ㅙ", ("ㅗ", "ㅣ"): "ㅚ",
    ("ㅜ", "ㅓ"): "ㅝ", ("ㅜ", "ㅔ"): "ㅞ", ("ㅜ", "ㅣ"): "ㅟ",
    ("ㅡ", "ㅣ"): "ㅢ",
}  # fmt: skip
_FINAL_COMPOUND = {
    ("ㄱ", "ㅅ"): "ㄳ", ("ㄴ", "ㅈ"): "ㄵ", ("ㄴ", "ㅎ"): "ㄶ",
    ("ㄹ", "ㄱ"): "ㄺ", ("ㄹ", "ㅁ"): "ㄻ", ("ㄹ", "ㅂ"): "ㄼ",
    ("ㄹ", "ㅅ"): "ㄽ", ("ㄹ", "ㅌ"): "ㄾ", ("ㄹ", "ㅍ"): "ㄿ", ("ㄹ", "ㅎ"): "ㅀ",
    ("ㅂ", "ㅅ"): "ㅄ",
}  # fmt: skip


def _is_vowel(j: str) -> bool:
    return j in _JUNG


def _compose(cho: str, jung: str, jong: str) -> str:
    return chr(
        0xAC00
        + (_CHO.index(cho) * 21 + _JUNG.index(jung)) * 28
        + (_JONG.index(jong) + 1 if jong else 0)
    )


def looks_like_korean_typed_in_english(s: str) -> bool:
    """영문자만 있고(공백 허용) 그중 모음 자리 키가 하나라도 있으면 한영 오타 후보로 본다."""
    letters = [c for c in s if c.isalpha()]
    if not letters or not all(c in _KEY_TO_JAMO for c in letters):
        return False
    return any(_is_vowel(_KEY_TO_JAMO[c]) for c in letters)


def english_keys_to_hangul(s: str) -> str:
    """영문 키 나열을 한글 음절로 조합한다. 자판 밖 문자(공백·숫자)는 그대로 둔다.

    조합 규칙은 두벌식 오토마타의 단순형: 초성 → 중성(복모음 결합) → 종성(복종성 결합).
    종성 뒤에 모음이 오면 종성을 다음 글자의 초성으로 넘긴다("ajdcl" → 명+치).
    """
    out: list[str] = []
    cho = jung = jong = ""

    def flush() -> None:
        nonlocal cho, jung, jong
        if cho and jung:
            out.append(_compose(cho, jung, jong))
        else:  # 조합이 안 된 낱자는 그대로
            out.extend(x for x in (cho, jung, jong) if x)
        cho = jung = jong = ""

    for ch in s:
        j = _KEY_TO_JAMO.get(ch)
        if j is None:
            flush()
            out.append(ch)
            continue
        if _is_vowel(j):
            if jong:  # 종성을 다음 초성으로 넘긴다 (복종성이면 뒤 자음만)
                carry = jong
                for (a, b), comp in _FINAL_COMPOUND.items():
                    if comp == jong:
                        jong, carry = a, b
                        break
                else:
                    jong = ""
                flush()
                cho, jung = carry, j
            elif jung:
                comp = _VOWEL_COMPOUND.get((jung, j))
                if comp:
                    jung = comp
                else:
                    flush()
                    jung = j
            else:
                jung = j
        else:  # 자음
            if cho and jung and not jong and j in _JONG:
                jong = j
            elif cho and jung and jong:
                comp = _FINAL_COMPOUND.get((jong, j))
                if comp:
                    jong = comp
                else:
                    flush()
                    cho = j
            elif cho and not jung:
                flush()
                cho = j
            else:
                cho = j
    flush()
    return "".join(out)
