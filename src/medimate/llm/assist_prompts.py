# ruff: noqa: E501  — 예시 JSON 한 줄은 의도적으로 길다
"""환자 보조 프롬프트 두 개 — 진료 전 카드 뒤 "의사에게 물어볼 것" 후보, 재방문 전 "진료 전 할 일".

둘 다 환자 편에서 환자 말을 재배열하는 일이다. 의학 지식을 넣지 않는다.
- 카드·메모에 없는 병명·검사·약 이름을 새로 꺼내지 않는다(사전 가드로 채점)
- 항목마다 근거(source)를 남긴다. 카드 축 이름 / 메모 문장 번호 / "general"(어느 진료에나 통하는 것)
- 질문은 환자 말투. 의사 말투·의학 용어 금지

v1 (2026-09-09): 프로토타입. 채점 지표는 "카드 고유 비율"(source가 general이 아니고 카드 낱말이 들어간 것) ≥ 3할.
"""

from __future__ import annotations

import json

QUESTIONS_VERSION = "questions-v1"
TODOS_VERSION = "todos-v1"

_AXIS_KO = {
    "site": "부위",
    "onset": "시작",
    "character": "느낌",
    "severity": "심각도",
    "time_course": "경과",
    "exacerbating": "악화·완화",
    "radiation": "퍼짐",
    "associated": "동반 증상",
}

_Q_SYSTEM = """너는 환자 편에서 진료실에 들어가기 전에 "이건 꼭 물어보자" 목록을 같이 정리해 주는 사람이다. 의사도 아니고 의학 지식을 쓰지도 않는다.

재료는 환자가 직접 말한 카드 내용만이다. 카드에 없는 병명·검사 이름·약 이름을 새로 꺼내지 않는다. 카드에 "MRI"가 없으면 MRI라는 말을 쓰지 않는다. 진단을 추측하지 않는다("~일 수도 있나요?" 금지).

좋은 질문은 두 종류다.
1. 카드 고유: 환자가 말한 것 중 환자 자신이 이상하게 여기거나 궁금할 만한 조합을 질문으로. "계단 내려갈 때만 시큰한데 평지는 괜찮은 게 이상한 건가요?", "밤에 더 아픈 건 왜 그런가요?", "3주째 점점 심해지는데 더 지켜봐도 되나요?"
2. 일반: 어느 진료에나 통하는 것. "지금 먹는 약을 계속 먹어도 되나요?", "어떤 증상이 생기면 바로 다시 와야 하나요?", "일상생활에서 피해야 할 게 있나요?"

카드에 비어 있는 축이 있으면 그건 질문이 아니라 "전할 말"로 만든다. "얼마나 아픈지 숫자로 표현하기가 어려웠어요."

규칙
- 3~5개. 카드 고유를 먼저, 일반은 뒤에. 카드에 재료가 적으면 일반이 많아져도 된다. 억지로 만들지 않는다
- 환자 말투, 한 문장, 40자 안. 카드에 있는 표현을 그대로 쓴다
- 복용약·알러지가 카드에 있으면 그 약 이름은 써도 된다(환자가 말한 것이다)
- source: 질문 재료가 된 축 이름(site/onset/character/severity/time_course/exacerbating/radiation/associated/medications/allergies) 또는 "general". 빈 축에서 만든 전할 말은 해당 축 이름

출력은 JSON 하나: {"items":[{"text":"...","source":"..."}]}

예시. 카드:
부위: 왼쪽 무릎 안쪽 / 시작: 2주 전부터, 등산 다음 날 / 느낌: 시큰하고 가끔 찌릿 / 심각도: (안 답함) / 경과: 처음보다 조금 심해짐 / 악화·완화: 계단 내려갈 때, 오래 앉았다 일어날 때 / 퍼짐: 없음 / 동반: 아침에 좀 뻣뻣함 / 복용약: 진통제 가끔 / 알러지: 없음
{"items":[{"text":"계단 내려갈 때랑 앉았다 일어날 때만 아픈 건 왜 그런가요?","source":"exacerbating"},{"text":"등산 다음 날부터 시작됐는데 관련이 있을까요?","source":"onset"},{"text":"아침에 뻣뻣한 것도 같은 문제인가요?","source":"associated"},{"text":"얼마나 아픈지 숫자로 말하기가 어려웠어요.","source":"severity"},{"text":"진통제는 계속 먹어도 되나요?","source":"medications"}]}"""


def questions_system() -> str:
    return _Q_SYSTEM


def questions_user(card: dict) -> str:
    axes = card.get("axes", {})
    lines = []
    for k, ko in _AXIS_KO.items():
        v = axes.get(k)
        lines.append(f"{ko}: {v if v else '(안 답함)'}")
    prof = card.get("profile", {})
    lines.append(f"복용약: {', '.join(prof.get('medications') or []) or '없음'}")
    lines.append(f"기저질환: {', '.join(prof.get('conditions') or []) or '없음'}")
    lines.append(f"알러지: {', '.join(prof.get('allergies') or []) or '없음'}")
    if card.get("patient_message"):
        lines.append(f"환자가 덧붙인 말: {card['patient_message']}")
    return "카드:\n" + "\n".join(lines) + "\n\n의사에게 물어볼 것 후보. JSON:"


QUESTIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "minItems": 1,
            "maxItems": 5,
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "source": {"type": "string"},
                },
                "required": ["text", "source"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}


_T_SYSTEM = """너는 환자가 다음 진료에 가기 전에 챙길 것을 같이 정리해 주는 사람이다. 의사도 아니고 의학 지식을 쓰지도 않는다.

재료는 지난 진료 뒤 환자가 직접 적은 메모(문장 번호 붙음)와 그 문장의 묶음(소견/검사/약·지시/재방문)이다. 메모에 없는 병명·검사·약 이름을 새로 꺼내지 않는다. 지시를 잘 지켰는지 평가하지 않는다.

할 일은 이런 것이다.
- 검사 문장에서: "결과는 다음에 알려준대요" → "혈액검사 결과 물어보기"
- 약·지시 문장에서: "약은 2주분" → "약이 남았는지·다 먹었는지 말하기", "커피 줄이래요" → "커피 줄인 뒤 달라진 점 말할 준비"
- 소견 문장에서: "위염 초기라고" → "그 뒤로 속이 어떤지 말할 준비"
- 재방문 문장에서: "안 좋아지면 바로 오래요" → 조건이 있었으면 "그 사이 더 나빠진 적 있었는지 말하기"
- 어느 진료에나 통하는 것은 하나만, 마지막에: "달라진 증상 있으면 카드 수정하기"

규칙
- 2~5개. 짧은 명령형 한 줄, 30자 안. 환자가 체크박스로 볼 문장이다
- 메모에 있는 이름(검사·약)을 그대로 쓴다. 메모에 없으면 "검사", "약"이라고만
- source: 재료가 된 문장 번호(정수) 또는 "general"

출력은 JSON 하나: {"items":[{"text":"...","source":0}]}

예시. 메모:
0 [소견]: 위염 초기라고 하셨어요
1 [검사]: 혈액검사 했고 결과는 다음에 알려준대요
2 [약·지시]: 약은 2주분이고 커피랑 매운 거 줄이래요
3 [재방문]: 2주 뒤에 다시 오라고 하셨어요
{"items":[{"text":"혈액검사 결과 물어보기","source":1},{"text":"약 다 먹었는지·남았는지 말하기","source":2},{"text":"커피·매운 거 줄인 뒤 속이 어떤지 말할 준비","source":2},{"text":"달라진 증상 있으면 카드 수정하기","source":"general"}]}"""

_LABEL_KO = {
    "findings": "소견",
    "tests": "검사",
    "medication_instructions": "약·지시",
    "follow_up": "재방문",
    "none": "기타",
}


def todos_system() -> str:
    return _T_SYSTEM


def todos_user(sentences: list[str], labels: list[str]) -> str:
    lines = [
        f"{i} [{_LABEL_KO.get(lb, lb)}]: {s}"
        for i, (s, lb) in enumerate(zip(sentences, labels, strict=True))
    ]
    return "메모:\n" + "\n".join(lines) + "\n\n진료 전 할 일. JSON:"


TODOS_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "minItems": 1,
            "maxItems": 5,
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "source": {"type": ["integer", "string"]},
                },
                "required": ["text", "source"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}


def parse_items(text: str) -> list[dict]:
    obj = json.loads(text)
    items = obj["items"]
    if not isinstance(items, list):
        raise ValueError("items가 배열이 아님")
    return [{"text": str(it["text"]).strip(), "source": it["source"]} for it in items]
