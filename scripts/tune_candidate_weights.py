"""저장된 eval 결과로 질문 후보 가중치를 튜닝한다. 호출 0, $0.

가중치를 바꿔도 LLM을 다시 부르지 않는다 — 이게 랭커를 프롬프트 밖에 둔 이유다.

    uv run python scripts/tune_candidate_weights.py
    uv run python scripts/tune_candidate_weights.py --top 3 --set flat,v1,med-heavy

읽는 것: evals/results/assist/questions-<model>-<ver>.jsonl (gitignore, 로컬)
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from medimate.evals.run_assist import load_inputs, reparse  # noqa: E402
from medimate.llm.assist_rank import WEIGHTS, top_candidates  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
KO = {
    "site": "부위",
    "onset": "시작",
    "character": "느낌",
    "severity": "심각도",
    "time_course": "경과",
    "exacerbating": "악화·완화",
    "radiation": "퍼짐",
    "associated": "동반",
    "medications": "복용약",
    "conditions": "기저질환",
    "allergies": "알러지",
    "patient_message": "전할 말",
    "general": "일반",
}

# 비교용 가중치 묶음. WEIGHTS(rank-v1)가 기준
SETS: dict[str, dict[str, float] | None] = {
    "flat": {},  # 전부 같은 값 = 모델이 낸 순서대로 앞에서 자르기(현행)
    "v1": None,  # assist_rank.WEIGHTS
    "med-heavy": {**WEIGHTS, "medications": 6.0, "allergies": 6.0, "conditions": 5.0},
}


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    a = argparse.ArgumentParser()
    a.add_argument(
        "--report",
        type=Path,
        default=ROOT / "evals/results/assist/questions-apac.amazon.nova-pro-v1_0-v6.jsonl",
    )
    a.add_argument("--top", type=int, default=3)
    a.add_argument("--set", default="flat,v1,med-heavy")
    args = a.parse_args()

    rows = [json.loads(ln) for ln in args.report.read_text(encoding="utf-8").splitlines() if ln]
    reparse(rows)
    cards = {c["id"]: c for c in load_inputs("questions")}
    names = [s.strip() for s in args.set.split(",") if s.strip()]

    print(f"{args.report.name} · 카드 {len(rows)}장 · 상위 {args.top}개까지\n")

    # 재료가 있는 카드 수(기회) — 실현율의 분모
    opp: collections.Counter = collections.Counter()
    for r in rows:
        c = cards.get(r["case_id"])
        if not c:
            continue
        for k, v in c["axes"].items():
            if v:
                opp[k] += 1
        for p in ("medications", "conditions", "allergies"):
            if c["profile"].get(p):
                opp[p] += 1
        if c.get("patient_message"):
            opp["patient_message"] += 1

    tables = {}
    for name in names:
        w = SETS.get(name)
        picked: collections.Counter = collections.Counter()
        cards_with: collections.Counter = collections.Counter()
        n_items = 0
        for r in rows:
            items = r["items"] or []
            top = top_candidates(items, top=args.top, weights=w)
            n_items += len(top)
            srcs = {it["source"] for it in top}
            for it in top:
                picked[it["source"]] += 1
            for s in srcs:
                cards_with[s] += 1
        tables[name] = (picked, cards_with, n_items)

    keys = sorted(set().union(*[t[0] for t in tables.values()]) | set(opp), key=lambda k: -opp[k])
    head = "".join(f"{n:>16}" for n in names)
    print(f"{'카테고리':12}{'재료':>6}{head}")
    print("-" * (18 + 16 * len(names)))
    for k in keys:
        if not opp[k] and not any(tables[n][0][k] for n in names):
            continue
        cells = ""
        for n in names:
            picked, cards_with, _ = tables[n]
            rate = cards_with[k] / opp[k] if opp[k] else 0.0
            cells += f"{cards_with[k]:>8}장{rate:>7.0%}"
        print(f"{KO.get(k, k):12}{opp[k]:>6}{cells}")
    print("-" * (18 + 16 * len(names)))
    print(f"{'상위 항목 합':12}{'':>6}" + "".join(f"{tables[n][2]:>16}" for n in names))
    print("\n(각 칸: 그 카테고리가 상위에 든 카드 수 / 재료 있는 카드 대비 비율)")
    print("가중치를 바꿔도 호출 0. 생성에 없는 카테고리는 랭커가 만들어 내지 못한다.")


if __name__ == "__main__":
    main()
