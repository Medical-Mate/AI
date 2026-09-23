"""값 생성기 — LLM이 값을 쓰고, 검증기가 확인하고, 실패하면 폴백(rule)한다(#122 B안).

프롬프트는 llm/prompt_value_gen.py(`value-gen-v3`), 검증기는 span/verify.py.

호출자에게 돌아가는 값은 세 가지 중 하나다:
- `generated`  모델이 쓴 값이 검증을 통과
- `fallback`   검증 실패(또는 파싱 실패) → RuleSelector가 고른 후보. 버린 값과 이유는 `dropped`에
- `none`       모델이 null(값 없음). 검증할 것이 없다 — NONE도 답이다
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from medimate.span.candidates import CandidateGenerator
from medimate.span.canon import geo_nominal, term_only
from medimate.span.select import NONE, RuleSelector, resolve
from medimate.span.verify import Verdict, drops_negation, verify_value
from medimate.text.lexicon import Lexicon, load_lexicon

# 값 생성 호출의 출력 한도. 정상 답은 최대 48토큰(425회 실측, 2026-09-23), 값은 30자
# (verify.MAX_CHARS). 한도에 닿는 것은 멈추지 않는 응답뿐이다 — Nova가 한글을 `\uXXXX`로
# 쓰다 같은 글자를 8192토큰까지 반복한다
MAX_OUTPUT_TOKENS = 256
_ESCAPE = re.compile(r"\\u[0-9a-fA-F]{4}")


def screen(text: str | None, output_tokens: int | None) -> str | None:
    """모델 응답을 쓰기 전에 거른다. 이스케이프·잘림이면 그 이유, 아니면 None.

    이스케이프된 값은 코드포인트부터 엉뚱한 글자다(`전 전 주`, `이전두`) — 풀어도 못 쓴다.
    v2 264회 중 26, v3 161회 중 32(RESULTS.md "value-gen-v3").
    """
    if output_tokens is not None and output_tokens >= MAX_OUTPUT_TOKENS:
        return "truncated"
    if text and _ESCAPE.search(text):
        return "escape"
    return None


# soft hyphen · zero-width space/non-joiner · BOM · 대괄호 · 따옴표
_JUNK = re.compile("[" + "".join(map(chr, (0xAD, 0x200B, 0x200C, 0xFEFF))) + r"\[\]\"'`]")


def polish(value: str) -> str:
    """모델 값의 표기를 결정론으로 다듬는다. 뜻을 바꾸는 일은 하지 않는다 — 그건 검증기·폴백의 몫.

    1. 보이지 않는 글자(soft hyphen 등)·대괄호·따옴표 제거
    2. 표기 정규화(chatnorm): 모델이 조각의 오타를 그대로 옮긴 것(`안 낫으면`)을 표준 표기로
    3. 무공백 값(6자↑)은 Kiwi로 띄운다 — 무공백 조각을 그대로 베낀 값
    """
    from medimate.text.chatnorm import normalize

    s = _JUNK.sub("", value).strip()
    s = normalize(s).text.strip()
    if len(s) >= 6 and " " not in s:
        try:
            from medimate.text.tokenize import base_kiwi

            k = base_kiwi()
            s = k.join([(t.form, t.tag) for t in k.tokenize(s)]).strip()
        except Exception:  # noqa: BLE001 — 띄우기 실패는 원값 유지
            pass
    return s


@dataclass
class Generated:
    value: str | None
    source: str  # generated | fallback | none
    model_value: str | None  # 모델이 쓴 원값(검증 전)
    verdict: Verdict
    dropped: str = ""  # 폴백했을 때 버린 값과 이유
    note: str = ""


class ValueGenerator:
    def __init__(
        self,
        provider: str,
        model_id: str,
        budget_usd: float = 0.5,
        client=None,
        lexicon: Lexicon | None = None,
    ):
        from medimate.llm.providers import LLMExtractor

        self.ex = LLMExtractor(provider, model_id, budget_usd=budget_usd, _client=client)
        self.name = "gen"
        self.model_id = model_id
        self.lexicon = lexicon or load_lexicon()
        self.gen = CandidateGenerator()
        self.rule = RuleSelector()
        self.last: dict = {}

    @property
    def usage(self):
        return self.ex.usage

    def _fallback(self, segment: str, axis: str) -> str | None:
        cset = self.gen.generate(segment, axis)
        fb = resolve(cset, self.rule.select(cset, axis))
        # 폴백도 부정은 지킨다 — 모델 값을 부정 때문에 버렸는데 rule이 `축농증`을 내면 같은 오류다.
        # 값 없음이 뒤집힌 값보다 낫다(원문 문장은 묶음에 그대로 있다)
        return None if drops_negation(fb, segment) else fb

    def generate(self, segment: str, axis: str) -> Generated:
        from medimate.llm import prompt_value_gen as P
        from medimate.llm.base import parse_json_text
        from medimate.text.chatnorm import normalize

        t0 = time.perf_counter()
        norm = normalize(segment).text
        text, i, o = self.ex.complete_json(
            P.system_prompt(),
            P.user_message(segment, axis, norm),
            P.SCHEMA,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        lat = time.perf_counter() - t0
        self.last = {
            "text": text,
            "input_tokens": i,
            "output_tokens": o,
            "latency_s": round(lat, 3),
            "prompt_version": P.PROMPT_VERSION,
        }
        bad = screen(text, o)
        if bad:
            return self.rejected_response(bad, segment, axis)
        try:
            got = parse_json_text(text).get("value")
        except Exception as e:  # noqa: BLE001 — 파싱 실패는 폴백 사유
            fb = self._fallback(segment, axis)
            return Generated(
                fb, "fallback", None, Verdict(False, ["parse"]), f"parse:{type(e).__name__}"
            )
        return self.check(got, segment, axis)

    def rejected_response(self, reason: str, segment: str, axis: str) -> Generated:
        """응답 자체를 못 쓸 때(이스케이프·잘림) — 값을 보지 않고 폴백."""
        fb = self._fallback(segment, axis)
        return Generated(fb, "fallback", None, Verdict(False, [reason]), reason)

    def check(self, got, segment: str, axis: str) -> Generated:
        """모델 값 하나를 다듬고·검증·폴백한다. 재채점(호출 0)도 여기를 탄다."""
        if got is None or (isinstance(got, str) and got.strip().upper() in ("", NONE, "NULL")):
            return Generated(None, "none", None, Verdict(True))
        got = polish(str(got))
        if not got:
            return Generated(None, "none", None, Verdict(True))
        if axis == "findings":
            got = geo_nominal(got)  # `-ㄴ 거래` → `-ㅁ`
            # 용어가 머리인 서술문 → 용어만(부정이 있으면 그대로)
            got = term_only(got, self.lexicon)
        v = verify_value(got, segment, axis, self.lexicon)
        if v.ok:
            return Generated(got, "generated", got, v)
        fb = self._fallback(segment, axis)
        return Generated(fb, "fallback", got, v, dropped=f"{got!r} ← {v.reason}")
