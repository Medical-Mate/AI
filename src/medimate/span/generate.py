"""값 생성기 — LLM이 값을 쓰고, 검증기가 확인하고, 실패하면 폴백(rule)한다(#122 B안).

프롬프트는 llm/prompt_value_gen.py(`value-gen-v2`), 검증기는 span/verify.py.

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
from medimate.span.select import NONE, RuleSelector, resolve
from medimate.span.verify import Verdict, verify_value
from medimate.text.lexicon import Lexicon, load_lexicon

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
        return resolve(cset, self.rule.select(cset, axis))

    def generate(self, segment: str, axis: str) -> Generated:
        from medimate.llm import prompt_value_gen as P
        from medimate.llm.base import parse_json_text
        from medimate.text.chatnorm import normalize

        t0 = time.perf_counter()
        norm = normalize(segment).text
        text, i, o = self.ex.complete_json(
            P.system_prompt(), P.user_message(segment, axis, norm), P.SCHEMA
        )
        lat = time.perf_counter() - t0
        self.last = {
            "text": text,
            "input_tokens": i,
            "output_tokens": o,
            "latency_s": round(lat, 3),
            "prompt_version": P.PROMPT_VERSION,
        }
        try:
            got = parse_json_text(text).get("value")
        except Exception as e:  # noqa: BLE001 — 파싱 실패는 폴백 사유
            fb = self._fallback(segment, axis)
            return Generated(
                fb, "fallback", None, Verdict(False, ["parse"]), f"parse:{type(e).__name__}"
            )
        return self.check(got, segment, axis)

    def check(self, got, segment: str, axis: str) -> Generated:
        """모델 값 하나를 다듬고·검증·폴백한다. 재채점(호출 0)도 여기를 탄다."""
        if got is None or (isinstance(got, str) and got.strip().upper() in ("", NONE, "NULL")):
            return Generated(None, "none", None, Verdict(True))
        got = polish(str(got))
        if not got:
            return Generated(None, "none", None, Verdict(True))
        v = verify_value(got, segment, axis, self.lexicon)
        if v.ok:
            return Generated(got, "generated", got, v)
        fb = self._fallback(segment, axis)
        return Generated(fb, "fallback", got, v, dropped=f"{got!r} ← {v.reason}")
