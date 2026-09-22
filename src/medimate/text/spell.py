"""맞춤법 교정 — LLM이 생성하고 코드가 검증한다(#122).

값 선택과 달리 교정은 **원문과의 거리를 코드가 잴 수 있다.** 그래서 생성을 허용한다. 가드는 좁다:
- 어절 수가 같다(띄어쓰기만 다른 것은 허용 — 어절을 다시 맞춘다)
- 어절마다 자소 편집거리 ≤ MAX_JAMO_EDITS (`나왓지만→나왔지만` 1, `낫으면→나으면` 2)
- 숫자는 한 글자도 바뀌지 않는다
- 원문에 있던 어휘집 용어는 교정문에도 그대로 있다(`위산약`을 `위산 약`으로 쪼개면 버린다)
통과 못 하면 원문을 쓴다. evidence는 언제나 원문이다(교정문은 chatnorm처럼 보조 입력).

교정문은 `spell_ops`로 어절별 (전, 후)가 남는다. 모델은 프롬프트 `spell-v1`.
"""

# ruff: noqa: E501  — 프롬프트 본문은 줄을 나누면 모델이 보는 텍스트가 바뀐다
from __future__ import annotations

import re
from dataclasses import dataclass, field

MAX_JAMO_EDITS = 2
PROMPT_VERSION = "spell-v1"

_SYSTEM = """너는 한국어 표기 교정기다. 환자가 채팅으로 적은 진료 메모의 **표기만** 표준으로 고친다.

고치는 것: 오타(나왓지만→나왔지만, 괜찬대요→괜찮대요, 됬음→됐음), 불규칙 활용 오기(안 낫으면→안 나으면), 어미 오타(레요→래요, 데요→대요), 띄어쓰기.
고치지 않는 것: 낱말 선택, 어순, 문장 길이, 숫자·단위, 약 이름·병명·검사명, 구어 어미(~랬음·~다네·~재 같은 말투는 그대로), 초성(ㅇㅇ·ㄴㄴ·ㄱㅊ 그대로), ㅋㅋ·ㅠㅠ.
뜻을 바꾸지 않는다. 문장을 다듬지 않는다. 모르면 그대로 둔다.

출력은 JSON 하나: {"corrected": "..."}. 다른 말은 쓰지 않는다."""

SCHEMA = {
    "type": "object",
    "properties": {"corrected": {"type": "string"}},
    "required": ["corrected"],
    "additionalProperties": False,
}

_DIGITS = re.compile(r"\d+")
_WS = re.compile(r"\s+")


def system_prompt() -> str:
    return _SYSTEM


def user_message(text: str) -> str:
    return f"메모:\n{text}\nJSON:"


# ── 자소 편집거리 ──
_CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_JONG = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"


def jamo(s: str) -> str:
    """음절을 자소로 푼다. 받침 없는 자리는 뺀다."""
    out = []
    for ch in s:
        code = ord(ch) - 0xAC00
        if 0 <= code < 11172:
            out.append(_CHO[code // 588])
            out.append(_JUNG[(code % 588) // 28])
            if code % 28:
                out.append(_JONG[code % 28])
        else:
            out.append(ch)
    return "".join(out)


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@dataclass
class SpellResult:
    raw: str
    text: str  # 채택된 텍스트(가드 통과면 교정문, 아니면 원문)
    accepted: bool
    reason: str = ""  # 거절 이유
    ops: list[tuple[str, str]] = field(default_factory=list)  # 바뀐 어절 (전, 후)
    model_text: str = ""  # 모델이 낸 교정문 원문


def guard(raw: str, corrected: str, protected: list[str] = ()) -> SpellResult:
    """교정문이 원문에서 얼마나 멀어졌나 잰다. 멀면 원문을 돌려준다."""
    corrected = corrected.strip()
    if not corrected or corrected == raw:
        return SpellResult(
            raw, raw, False, "unchanged" if corrected else "empty", model_text=corrected
        )
    if _DIGITS.findall(raw) != _DIGITS.findall(corrected):
        return SpellResult(raw, raw, False, "digits_changed", model_text=corrected)
    for term in protected:
        if _WS.sub("", term) in _WS.sub("", raw) and _WS.sub("", term) not in _WS.sub(
            "", corrected
        ):
            return SpellResult(raw, raw, False, f"term_lost:{term}", model_text=corrected)
    # 어절 정렬: 공백을 무시한 전체 자소 거리로 먼저 상한을 두고, 어절은 공백 기준으로 비교
    if edit_distance(jamo(_WS.sub("", raw)), jamo(_WS.sub("", corrected))) > MAX_JAMO_EDITS * max(
        1, len(raw.split())
    ):
        return SpellResult(raw, raw, False, "too_far", model_text=corrected)
    a, b = raw.split(), corrected.split()
    ops: list[tuple[str, str]] = []
    if len(a) == len(b):
        for x, y in zip(a, b, strict=True):
            if x != y:
                if edit_distance(jamo(x), jamo(y)) > MAX_JAMO_EDITS:
                    return SpellResult(
                        raw, raw, False, f"token_too_far:{x}->{y}", model_text=corrected
                    )
                ops.append((x, y))
    else:
        # 띄어쓰기가 바뀐 경우: 공백 없는 문자열끼리 거리만 본다(위에서 이미 상한 검사)
        if _WS.sub("", raw) != _WS.sub("", corrected):
            ops.append((raw, corrected))
    return SpellResult(raw, corrected, True, "", ops, model_text=corrected)


class LLMSpeller:
    """Nova 등 공급자 어댑터로 교정문을 받아 가드에 통과시킨다. 실패·거절은 원문."""

    def __init__(
        self, provider: str, model_id: str, budget_usd: float = 0.5, client=None, protected=()
    ):
        from medimate.llm.providers import LLMExtractor

        self.ex = LLMExtractor(provider, model_id, budget_usd=budget_usd, _client=client)
        self.model_id = model_id
        self.protected = list(protected)
        self.last: dict = {}

    @property
    def usage(self):
        return self.ex.usage

    def correct(self, text: str) -> SpellResult:
        from medimate.llm.base import parse_json_text

        raw_text, i, o = self.ex.complete_json(system_prompt(), user_message(text), SCHEMA)
        self.last = {"text": raw_text, "input_tokens": i, "output_tokens": o}
        try:
            corrected = str(parse_json_text(raw_text).get("corrected", ""))
        except Exception as e:  # noqa: BLE001
            return SpellResult(text, text, False, f"parse:{type(e).__name__}", model_text=raw_text)
        return guard(text, corrected, self.protected)
