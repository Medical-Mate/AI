# ruff: noqa: E501
"""재방문 기간 표현 읽기 — `followup-v1` (서버 전용, 2차 호출).

날짜는 LLM이 아니라 계산한다는 원칙은 그대로다. **읽기만** LLM에 맡긴다.
규칙 파서(`memo.followup_date`)가 못 읽거나 앞 절 추론으로 샜을 때만 부른다 — 한국어 기간
표현은 열린 집합이라 규칙이 늘 하나씩 늦었다(`4일후` → `일주일` → `이주뒤`, 2026-09-15).

LLM은 **문장에 있는 글자 그대로의 표현**과 그것이 며칠인지만 낸다. 코드가 그 표현이 문장에
실제로 있는지 대조하고(근거 가드), 날짜는 코드가 더한다. 표현이 문장에 없으면 버린다.
"""

from __future__ import annotations

PROMPT_VERSION = "followup-v1"

SCHEMA: dict = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "days": {"type": ["integer", "null"]},
        "month": {"type": ["integer", "null"]},
        "day": {"type": ["integer", "null"]},
    },
    "required": ["text", "days", "month", "day"],
    "additionalProperties": False,
}

_SYSTEM = """환자가 진료실에서 들은 말을 적은 메모의 한 문장에서 **다시 오는 시점**을 읽는다. 새 말을 만들지 않는다.

출력
- text: 시점을 나타내는 부분을 문장에서 **글자 그대로** 복사한다(한 글자도 바꾸지 않는다). 없으면 "".
- days: 진료일로부터 며칠 뒤인지 정수. 주는 7일, 달은 30일로 센다. 정확한 날짜(몇 월 며칠)면 null.
- month, day: "10월 3일"처럼 날짜를 말했으면 그 숫자. 아니면 null.

규칙
- 약을 며칠치 받았다는 말(3일치, 일주일분)은 **재방문 시점이 아니다.** 다시 오라는 말과 같은 절에 있을 때만 시점으로 본다.
- "안 좋아지면 오라고", "필요하면"처럼 조건만 있고 기간이 없으면 text는 "" 이고 days도 null.
- 앞 문장이 함께 주어지면 참고만 한다. text는 반드시 **현재 문장**에서 복사한다.

예시
문장: 이주뒤 재방문 → {"text":"이주뒤","days":14,"month":null,"day":null}
문장: 보름쯤 있다가 다시 오래요 → {"text":"보름쯤 있다가","days":15,"month":null,"day":null}
문장: 담주 화요일에 보자고 → {"text":"담주 화요일","days":7,"month":null,"day":null}
문장: 10월 3일에 오라고 → {"text":"10월 3일","days":null,"month":10,"day":3}
문장: 안 좋아지면 바로 오래요 → {"text":"","days":null,"month":null,"day":null}
문장: 3일치 약 먹고 다시 오세요 → {"text":"3일치 약 먹고 다시","days":3,"month":null,"day":null}"""


def system_prompt() -> str:
    return _SYSTEM


def user_message(sentence: str, prev_text: str | None = None) -> str:
    head = f"앞 문장(참고): {prev_text}\n" if prev_text else ""
    return f"{head}문장: {sentence}\nJSON:"
