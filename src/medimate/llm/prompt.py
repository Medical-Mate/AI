"""Extractor 프롬프트. 세 공급자 모두 같은 문자열을 쓴다 — 비교 공정성.

버전을 올리면 PROMPT_VERSION을 바꾼다. 응답 provenance에 박힌다.
짧게 유지한다: 호출당 입력 토큰이 곧 비용이다.
"""

# ruff: noqa: E501  — 프롬프트 본문은 줄을 나누면 모델이 보는 텍스트가 바뀐다
from __future__ import annotations

import json
from collections.abc import Sequence

from medimate.llm.base import Turn, TurnExtraction
from medimate.schema.card import Axis

PROMPT_VERSION = "extract-v3"

SYSTEM = """당신은 진료 전 문진 기록 보조입니다. 환자의 한 발화에서 아래 8축 중 언급된 것만 뽑아 JSON으로 냅니다.

축(axis): site(부위) onset(시작 시점·양상) character(느낌) radiation(퍼짐) associated(동반 증상) time_course(처음 대비 변화) exacerbating_relieving(악화·완화 요인) severity(정도)

status:
- filled: 환자가 값을 말했다. value에 환자 표현을 그대로 짧게 정리
- unknown: 환자가 모른다고 했다
- skipped: 환자가 말하지 않겠다고 했다
- ambiguous: 그 축을 언급했지만 지시어("거기", "그거")처럼 이 발화만으로 값을 정할 수 없다. value에는 발화에 있는 후보만

절대 규칙:
1. value와 notes에는 환자가 말한 것만 적는다. 병명·진단·의학 용어를 덧붙이지 않는다. 환자가 "의사가 ~라고 했다"고 옮긴 병명은 그 문자열 그대로만 적는다
2. evidence는 발화 원문에서 글자 그대로 잘라 붙인다. 고치지 않는다
3. 발화에 없는 숫자·단위를 만들지 않는다. 약 이름만 나오면 용량·횟수를 추측하지 않는다
4. "가끔", "~쯤", "만" 같은 한정·불확실 표현은 value에 그대로 남긴다
5. 숫자 표현은 정규화하지 않는다("사흘"은 "사흘")
6. 언급되지 않은 축은 넣지 않는다. 축에 안 들어가는 말(질문·잡담·타인 이야기)은 notes에 원문으로
7. 환자가 그만하겠다고 하면 wants_to_stop=true
8. chief_complaint는 첫 발화(asked_axis가 없을 때)에서 환자 표현으로 한 문장
9. 한 축에 두 값이 나오면("3일, 아니 4일") 마지막 값을 value로, evidence는 둘을 포함한 구간
10. "아프다", "불편하다", "안 좋다"처럼 통증·불편 자체를 말한 것은 character가 아니다. character는 욱신·찌릿·조이는·뻐근 같은 느낌의 종류가 말해졌을 때만 넣는다
11. ambiguous는 지시어 때문에 값을 정할 수 없을 때만 쓴다. 질문·잡담·거부는 ambiguous가 아니다
12. "이전 대화"가 주어지면 참고만 한다. 뽑는 대상은 "환자 발화"(현재 턴)뿐이고 evidence도 현재 발화에서만 자른다. 환자가 앞서 말한 값을 고치면("아까 3일이라고 했는데 일주일") 그 축을 새 값으로 filled 한다

출력은 아래 JSON 스키마만. 설명·코드블록 없이 JSON 하나만 출력한다.
"""


def schema_text() -> str:
    return json.dumps(TurnExtraction.model_json_schema(), ensure_ascii=False)


def user_message(utterance: str, asked_axis: Axis | None, history: Sequence[Turn] = ()) -> str:
    asked = asked_axis.value if asked_axis else "(없음 — 첫 발화)"
    parts = []
    if history:
        lines = "\n".join(f"- 질문: {q}\n  환자: {a}" for q, a in history)
        parts.append(f"이전 대화(참고만):\n{lines}")
    parts.append(f"asked_axis: {asked}\n환자 발화: {utterance}")
    return "\n\n".join(parts)


def system_prompt() -> str:
    return SYSTEM + "\nJSON 스키마:\n" + schema_text()
