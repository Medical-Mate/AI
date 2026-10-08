"""질문 후보 템플릿을 카드 100장에 돌려 run_assist와 같은 형식으로 저장한다(호출 0).

uv run python scripts/eval_question_templates.py
uv run python -m medimate.evals.run_assist --task questions --report evals/results/assist/questions-template-v1.jsonl --show
"""

import json
import sys
from pathlib import Path

from medimate.dialog.question_templates import template_candidates

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evals" / "results" / "assist" / "questions-template-v1.jsonl"
cards = [
    json.loads(x)
    for x in (ROOT / "evals" / "previsit_cards.jsonl").read_text(encoding="utf-8").splitlines()
    if x
]
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8") as f:
    for c in cards:
        items = template_candidates(c.get("axes") or {}, c.get("profile"))
        f.write(
            json.dumps(
                {
                    "case_id": c["id"],
                    "task": "questions",
                    "model_id": "template",
                    "prompt_version": "questions-template-v1",
                    "text": json.dumps({"items": items}, ensure_ascii=False),
                    "items": items,
                    "error": None,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "latency_s": 0.0,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
print("저장:", OUT, len(cards))
