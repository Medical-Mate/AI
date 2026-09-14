"""웹 데모 폴백용 녹화 세션을 만든다 — `docs/examples/demo-session.json`.

서버가 죽어도 프론트가 화면을 그릴 수 있게, **완주한 세션 하나의 요청·응답 전문**을 남긴다.

두 가지를 지킨다.
1. **LLM을 부르지 않는다.** 추출·질문 후보·메모 분류 자리에 `evals/`의 저장된 결과를 주입한다
2. **실제 API를 그대로 태운다.** 손으로 쓴 JSON이 아니라 TestClient로 진짜 엔드포인트를 부르고
   응답을 그대로 저장한다. 필드명이 한 글자라도 다르면 프론트가 그걸로 개발했다가 라이브에서
   깨진다 — `docs/examples/body-map.json`에서 배운 것과 같은 자리다

  uv run python scripts/gen_demo_session.py          # 파일 갱신
  uv run python scripts/gen_demo_session.py --check  # 드리프트만 확인(쓰지 않음)
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from fastapi.testclient import TestClient  # noqa: E402

from medimate.api.app import create_app  # noqa: E402
from medimate.dialog.memo import MemoLabels  # noqa: E402
from medimate.llm.base import AxisUpdate, TurnExtraction  # noqa: E402
from medimate.schema.card import Axis, FieldStatus  # noqa: E402

OUT = Path("docs/examples/demo-session.json")

# 진료 전 — `evals/previsit_cards.jsonl` PC01의 축 값을 그대로 쓴다(기준선 카드).
# 발화는 그 값이 나오게 되는 환자 말이고, 추출 결과는 축 값 그대로다. 식별자는 없다.
#
# **부위는 인체도로 미리 채워진 채 둔다.** 첫 턴을 부위 답으로 쓰면 선택으로 들어온 값이
# 발화로 덮이고, 그러면 `axes.site.source`가 `selection`이 아니라 `ai_extraction`이 된다 —
# 데모에서 인체도 경로가 통째로 사라진다.
#
# **답은 "그 턴에 실제로 물은 축"에 맞춘다.** 대본을 순서대로 흘리면 서버가 퍼짐을 묻는데
# 환자가 동반 증상을 답하는 대화가 녹화된다. 질문 순서는 엔진이 정하므로 여기서 고정하지 않는다.
OPENING = "2주 전쯤 등산 다음 날부터 시큰하고 가끔 찌릿해요"
OPENING_UPDATES = [
    (Axis.ONSET, "2주 전부터요, 등산 다음 날부터"),
    (Axis.CHARACTER, "시큰하고 가끔 찌릿해요"),
]
ANSWERS = {
    Axis.TIME_COURSE: ("처음보다 조금 심해진 것 같아요", "처음보다 조금 심해진 것 같아요"),
    Axis.EXACERBATING_RELIEVING: (
        "계단 내려갈 때, 오래 앉았다 일어날 때요",
        "계단 내려갈 때, 오래 앉았다 일어날 때",
    ),
    Axis.ASSOCIATED: ("아침에 좀 뻣뻣해요", "아침에 좀 뻣뻣해요"),
    Axis.RADIATION: ("옆으로 퍼지진 않아요", "옆으로 퍼지지 않음"),
    Axis.SEVERITY: ("10점에 3점쯤이요", "3"),
    Axis.SITE: ("왼쪽 무릎 안쪽이요", "왼쪽 무릎 안쪽"),
    Axis.ONSET: ("2주 전부터요", "2주 전부터요"),
    Axis.CHARACTER: ("시큰해요", "시큰해요"),
}
PROFILE = {"medications": ["진통제(가끔)"], "conditions": [], "allergies": []}

# 통증 강도는 **묻지 않고 화면에서 고른다**(1d 슬라이더). `ASK_ORDER`에 `severity`가 없는 이유다 —
# NRS를 말로 물으면 환자가 숫자를 지어내고, 그 값은 근거가 없다.
# 슬라이더 값은 발화가 아니라 `selections`로 오고 evidence에 `[선택]`이 붙는다.
# 녹화에 이 자리가 없으면 프론트가 **슬라이더 값을 어디에 실어야 하는지 모른다.**
SEVERITY_SELECTION = {"axis": "severity", "value": "3 (꽤 아파요)"}

# 질문 후보 — 저장된 형식 그대로. 랭커가 정렬해 위에서 3개를 자른다
CANDIDATES = json.dumps(
    {
        "items": [
            {"text": "진통제를 계속 먹어도 되나요?", "source": "medications"},
            {"text": "계단 내려갈 때만 아픈 건 어떤 상태인가요?", "source": "exacerbating"},
            {"text": "아침에 뻣뻣한 것도 관련이 있나요?", "source": "associated"},
            {"text": "2주째 비슷한데 언제까지 이러면 다시 와야 하나요?", "source": "time_course"},
        ]
    },
    ensure_ascii=False,
)

# 진료 후 — `evals/postvisit_cases.jsonl` PM01
MEMO = "위염 초기라고 하셨어요. 혈액검사 했고 결과는 다음에 알려준대요. 약은 2주분이고 커피랑 매운 거 줄이래요. 2주 뒤에 다시 오라고 하셨어요."
MEMO_LABELS = ["findings", "tests", "medication_instructions", "follow_up"]
VISIT_DATE = "2026-09-12"


class StoredExtractor:
    """저장된 축 값을 돌려준다. 네트워크를 타지 않는다.

    **그 턴에 물은 축을 보고 답한다** — 서버의 질문 순서가 바뀌어도 녹화가 어긋나지 않는다.
    """

    model_id = "demo/stored"
    prompt_version = "demo-fixture"

    def __init__(self):
        self.turns = 0

    def answer_for(self, asked_axis) -> str:
        """이번 턴에 보낼 환자 발화."""
        if asked_axis is None:
            return OPENING
        return ANSWERS[asked_axis][0]

    def extract(self, utterance, asked_axis, history=()):
        self.turns += 1
        if asked_axis is None:
            updates = OPENING_UPDATES
        else:
            updates = [(asked_axis, ANSWERS[asked_axis][1])]
        return TurnExtraction(
            chief_complaint=utterance if self.turns == 1 else None,
            updates=[
                AxisUpdate(axis=a, status=FieldStatus.FILLED, value=v, evidence=utterance)
                for a, v in updates
            ],
        )

    def complete_json(self, system, user, schema):
        return CANDIDATES, 0, 0


class StoredMemoClassifier:
    model_id = "demo/stored"
    prompt_version = "demo-fixture"

    def classify(self, sentences):
        return MemoLabels.from_keyed(
            {str(i): MEMO_LABELS[i] for i in range(len(sentences))}, len(sentences)
        )


def build() -> dict:
    ex = StoredExtractor()
    client = TestClient(create_app(lambda: ex, memo_factory=lambda: StoredMemoClassifier()))
    steps = []

    start_body = {"site_node_id": "SUR:091", "side": "left", "request_id": "demo-0"}
    r = client.post("/v1/previsit/sessions", json=start_body)
    r.raise_for_status()
    steps.append({"step": "session_start", "request": start_body, "response": r.json()})
    state = r.json()["state"]

    # 서버가 이번 턴에 무엇을 물었는지 보고 답한다. 세션이 끝나면 멈춘다.
    i = 0
    while not steps[-1]["response"].get("ended", False):
        i += 1
        asked = Axis(state["asked_axis"]) if state.get("asked_axis") else None
        body = {"state": state, "utterance": ex.answer_for(asked), "request_id": f"demo-{i}"}
        r = client.post("/v1/previsit/turns", json=body)
        r.raise_for_status()
        steps.append({"step": f"turn_{i}", "request": body, "response": r.json()})
        state = r.json()["state"]
        if i > 20:  # 엔진 상한과 같다. 무한 루프 방지
            raise RuntimeError("세션이 끝나지 않았다")

    # **문답이 끝난 뒤에 슬라이더와 후보 요청이 온다.** 와이어프레임 순서가
    # 2 문답 → 3 통증 슬라이더 → 4 물어볼 것이고, **실제 클라이언트는 어느 턴이 마지막인지
    # 모른다.** 앞서 이걸 마지막 문답 턴에 실어 녹화했더니 `ended` 뒤 selections가 버려지는
    # 버그를 못 잡았다 — 녹화기만 미래를 알고 있었던 셈이다.
    body = {
        "state": state,
        "selections": [SEVERITY_SELECTION],
        "question_candidates": True,
        "patient_profile": PROFILE,
        "request_id": "demo-after",
    }
    r = client.post("/v1/previsit/turns", json=body)
    r.raise_for_status()
    steps.append({"step": "after_end_slider_and_candidates", "request": body, "response": r.json()})

    memo_body = {
        "memo": MEMO,
        "visit_date": VISIT_DATE,
        "clinic": "서울OO병원 내과",
        "request_id": "demo-memo",
    }
    r = client.post("/v1/postvisit/memo", json=memo_body)
    r.raise_for_status()
    steps.append({"step": "postvisit_memo", "request": memo_body, "response": r.json()})

    return {
        "note": (
            "웹 데모 폴백용 녹화 세션. 실제 API를 태워 받은 응답 전문이고 손으로 고치지 않는다. "
            "다시 만들려면 `uv run python scripts/gen_demo_session.py`. "
            "개인정보·식별자 없음 — 환자 발화는 evals/previsit_cards.jsonl PC01의 축 값에서 왔다."
        ),
        "generated_by": "scripts/gen_demo_session.py",
        "steps": steps,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="쓰지 않고 드리프트만 확인")
    args = ap.parse_args()
    fresh = build()
    text = json.dumps(fresh, ensure_ascii=False, indent=2) + "\n"
    if args.check:
        if not OUT.exists():
            print(f"{OUT} 없음")
            return 1
        same = OUT.read_text(encoding="utf-8") == text
        print("일치" if same else f"{OUT}가 지금 API 응답과 다릅니다 — 다시 생성하세요")
        return 0 if same else 1
    OUT.write_text(text, encoding="utf-8")
    turns = sum(1 for s in fresh["steps"] if s["step"].startswith("turn_"))
    last = fresh["steps"][-2]["response"]["card"]
    print(f"{OUT} · 단계 {len(fresh['steps'])}개(문진 {turns}턴)")
    print(f"  제목 {last['title']} · 후보 {len(last['question_candidates'] or [])}개")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
