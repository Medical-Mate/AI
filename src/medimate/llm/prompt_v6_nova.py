"""Nova Pro 추출 프롬프트 v6 — v5 + 악화·완화 요인 한 줄 (2026-10-12).

v4부터 같은 자리에서 틀리던 3건(L01·L02·XA01)의 공통점은 **악화·완화 요인을 다른 칸에 넣는 것**이다:
"밤에 더 아파요"를 time_course로(XA01), "약 먹으면 괜찮아져요"를 notes로(L02), "파스 붙이면 낫고
움직이면 풀리고"를 associated로(L01). 축 목록의 한 단어 설명("악화·완화 요인")만으로는 Nova가
"언제 더 아픈가"를 경과·동반과 못 가른다.

v5와 다른 것은 **축 목록의 exacerbating_relieving 설명 한 곳**뿐이다. 번호 규칙 18로 넣은 두 판은
S04에서 부위를 notes로 보내는 회귀가 있었다(3회 중 2회·3회). 한 번에 하나만 바꾸고 전부 다시 잰다(v5.1 교훈: 한 줄이
무관한 케이스를 흔든다). 채택 기준: 회귀 88·채팅 20·인젝션 25에서 망친 것 0, L01·L02·XA01 중 둘 이상.
"""

# ruff: noqa: E501  — 프롬프트 본문은 줄을 나누면 모델이 보는 텍스트가 바뀐다
from __future__ import annotations

from medimate.llm.prompt import schema_text
from medimate.llm.prompt_v5_nova import SYSTEM as V5_SYSTEM
from medimate.llm.prompt_v5_nova import user_message as _v5_user_message

PROMPT_VERSION = "extract-v6-nova"

# 세 번째 판(v6c): 번호 규칙을 더하지 않고 **축 목록의 한 단어 설명만** 넓힌다. 번호 규칙 18은 두 판 모두
# S04("언제인진 모르겠고 오른쪽만 아파요")의 부위를 notes로 보냈다(3회 중 2회·3회). 규칙 하나가 늘면 모델이
# "notes"를 더 쓴다 — 정의를 고치는 쪽이 다른 칸을 안 건드린다
_OLD_AXIS = "exacerbating_relieving(악화·완화 요인)"
_NEW_AXIS = 'exacerbating_relieving(악화·완화 요인 — 어떤 때·무엇을 하면 더 아프거나 덜 아픈지: "밤에 더", "계단 오를 때", "약 먹으면 나아짐". 경과·동반이 아니다)'
assert _OLD_AXIS in V5_SYSTEM
SYSTEM = V5_SYSTEM.replace(_OLD_AXIS, _NEW_AXIS, 1)
assert SYSTEM != V5_SYSTEM, "축 설명이 바뀌지 않았다"

user_message = _v5_user_message


def system_prompt() -> str:
    return SYSTEM + "\nJSON 스키마:\n" + schema_text()
