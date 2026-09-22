"""Nova Pro 추출 프롬프트 v5 — v4 + 채팅 표기 복원 한 줄 (#117).

환자는 의사에게는 문장으로 말하고 AI에게는 전보를 친다: `ㄱㅊ`, `ㄴㄴ`, `아픈듯`, `사흘전부터요`, `ㅠㅠ`.
v4까지는 이런 발화가 측정된 적이 없고, 런타임 가드의 "글자 없음" 필터가 `ㄱㅊ`·`ㄴㄴ`를 통째로 버렸다.

v4와 다른 것 둘.
1. user_message에 **표기 참고** 한 줄을 붙인다 — `text/chatnorm.normalize`가 표로만 되살린 표준 표기.
   원문이 그대로면 붙이지 않는다(정상 발화는 v4와 같은 입력을 받는다)
2. 규칙 17: 표기 참고는 뜻을 읽을 때만 쓴다. evidence는 언제나 <<<환자발화>>> 원문에서 자른다.
   `ㄱㅊ`가 무슨 값인지는 asked_axis가 정한다 — 통증 질문의 괜찮다와 부작용 질문의 괜찮다는 다르다

시스템 프롬프트의 나머지는 v4 그대로다(인젝션 방어·값 창작 방어·빈 updates 예시).
"""

# ruff: noqa: E501  — 프롬프트 본문은 줄을 나누면 모델이 보는 텍스트가 바뀐다
from __future__ import annotations

from collections.abc import Sequence

from medimate.llm.base import Turn
from medimate.llm.prompt import schema_text
from medimate.llm.prompt_v4_nova import CLOSE, OPEN
from medimate.llm.prompt_v4_nova import SYSTEM as V4_SYSTEM
from medimate.schema.card import Axis
from medimate.text.chatnorm import normalize

PROMPT_VERSION = "extract-v5-nova"

_RULE17 = f"""
17. **"표기 참고"가 주어지면** 그것은 환자 발화의 초성 축약(ㄱㅊ→괜찮다, ㄴㄴ→아니), 구어 표기(담주→다음 주), 오타를 표로 되살린 것이다. 뜻을 읽을 때 참고한다. **evidence는 언제나 {OPEN} 안의 원문에서 글자 그대로 자른다** — 표기 참고에서 자르지 않는다. 초성만 있는 답(ㄴㄴ, ㅇㅇ, ㄱㅊ, ㅁㄹ)도 물은 축에 대한 답이면 값이다: 퍼지냐는 질문에 ㄴㄴ면 radiation을 filled(안 퍼짐)로, 모르겠다는 ㅁㄹ면 unknown으로. 웃음·울음 표지(ㅋㅋ, ㅠㅠ)만 있는 답은 값이 아니다.
"""

SYSTEM = V4_SYSTEM.replace("\n출력은 아래 JSON 스키마만.", _RULE17 + "\n출력은 아래 JSON 스키마만.")


def user_message(utterance: str, asked_axis: Axis | None, history: Sequence[Turn] = ()) -> str:
    asked = asked_axis.value if asked_axis else "(없음 — 첫 발화)"
    parts = []
    if history:
        lines = "\n".join(f"- 질문: {q}\n  환자: {a}" for q, a in history)
        parts.append(f"이전 대화(참고만):\n{lines}")
    body = f"asked_axis: {asked}\n환자 발화:\n{OPEN}{utterance}{CLOSE}"
    n = normalize(utterance)
    if n.changed:
        body += f"\n표기 참고: {n.text}"
    parts.append(body)
    return "\n\n".join(parts)


def system_prompt() -> str:
    return SYSTEM + "\nJSON 스키마:\n" + schema_text()
