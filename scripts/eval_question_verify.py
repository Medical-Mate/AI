"""질문 후보 검증기 실측 — 저장된 생성 결과에 검증기·템플릿 폴백을 걸어 본다(호출 0, 2026-10-08).

- 놓침: LoRA v1 출력(평가 카드 100장)에서 사람이 읽고 찾은 실패를 잡는가
- 오거절: Nova v8 출력(평가 100장 + 학습 카드 정답 395장)을 얼마나 버리는가 — 운영 웹에서 문제 삼지
  않은 출력이라 많이 버리면 검증기가 너무 좁다

  uv run --no-sync python scripts/eval_question_verify.py
  uv run --no-sync python scripts/eval_question_verify.py --show "LoRA v1" [--kept]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
sys.stdout.reconfigure(encoding="utf-8")

from medimate.dialog.question_templates_v2 import natural_candidates  # noqa: E402
from medimate.dialog.question_verify import finalize  # noqa: E402
from medimate.evals.run_assist import TEST_DRUG_TERMS  # noqa: E402
from medimate.evals.score import load_lexicon  # noqa: E402
from medimate.text.tokenize import base_kiwi  # noqa: E402

A = Path("evals/results/assist")
RUNS = {
    "Nova v8": A / "questions-apac.amazon.nova-pro-v1_0-v8.jsonl",
    "LoRA v1": A / "questions-local-qwen3-1.7b-q4_0-lora-v1-noex.jsonl",
    "1.7B v8": A / "questions-local-qwen3-1.7b-q4_0-cpu-v8.jsonl",
    "1.7B v8-noex": A / "questions-local-qwen3-1.7b-q4_0-cpu-noex.jsonl",
    "템플릿 v1": A / "questions-template-v1.jsonl",
    "템플릿 v2": A / "questions-template-v2.jsonl",
}
TEMPLATES = None
TRAIN_TARGETS = Path("evals/results/qtrain/targets-nova-v8.jsonl")


def eval_cards() -> dict[str, dict]:
    return {
        json.loads(x)["id"]: json.loads(x)
        for x in open("evals/previsit_cards.jsonl", encoding="utf-8")
        if x.strip()
    }


def train_rows() -> list[tuple[dict, list[dict]]]:
    import gen_qtrain_targets as gt

    cards = {c["id"]: c for c in gt.load_cards()}
    rows = []
    for x in TRAIN_TARGETS.read_text(encoding="utf-8").splitlines():
        r = json.loads(x)
        if r.get("items") and r["id"] in cards:
            rows.append((cards[r["id"]], r["items"]))
    return rows


def run(
    name: str, rows: list[tuple[str, dict, list[dict]]], lex, kiwi, show: bool, kept_too: bool
) -> None:
    n_items = n_kept = 0
    why: Counter = Counter()
    full_model = need_fill = short = 0
    origin: Counter = Counter()
    for cid, card, items in rows:
        out, dropped = finalize(card, items, lex, kiwi, TEMPLATES)
        n_items += len(items)
        n_kept += len(items) - len(dropped)
        for d in dropped:
            why[d["reasons"][0].split(":")[0]] += 1
        model_n = sum(o["origin"] == "model" for o in out)
        full_model += model_n >= 3
        need_fill += model_n < 3
        short += len(out) < 3
        origin.update(o["origin"] for o in out)
        if show and (dropped or kept_too):
            print(
                f"\n{cid} | {' / '.join(v for v in (card.get('axes') or {}).values() if isinstance(v, str))[:150]}"
            )
            for d in dropped:
                print(f"   ✗ {d['text']}   ← {';'.join(d['reasons'])}")
            if kept_too:
                for o in out:
                    print(f"   {'○' if o['origin'] == 'model' else '△'} {o['text']}")
    print(
        f"{name:14} 카드 {len(rows):3} | 항목 {n_items:3} → 통과 {n_kept:3} ({n_kept / max(1, n_items):.0%}) | "
        f"모델 3개 이상 {full_model:3}장 · 템플릿 채움 {need_fill:3}장 · 3개 미만 {short} | 최종 모델:템플릿 {origin['model']}:{origin['template']}"
    )
    print(f"{'':14} 버린 이유(첫 이유) {dict(why.most_common())}")


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--show", help="이 실행의 버린 항목을 출력")
    a.add_argument(
        "--kept", action="store_true", help="--show와 함께: 최종 세트도 출력(○ 모델, △ 템플릿)"
    )
    a.add_argument("--fill", choices=["v1", "v2"], default="v2", help="빈자리를 채울 템플릿")
    args = a.parse_args()
    global TEMPLATES
    TEMPLATES = natural_candidates if args.fill == "v2" else None
    lex = load_lexicon() + TEST_DRUG_TERMS
    kiwi = base_kiwi()
    cards = eval_cards()
    for name, path in RUNS.items():
        if not path.exists():
            print(name, "결과 없음")
            continue
        rows = []
        for x in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(x)
            rows.append((r["case_id"], cards[r["case_id"]], r.get("items") or []))
        run(name, rows, lex, kiwi, args.show == name, args.kept)
    if TRAIN_TARGETS.exists():
        rows = [(c["id"], c, items) for c, items in train_rows()]
        run("Nova 학습정답", rows, lex, kiwi, args.show == "Nova 학습정답", args.kept)


if __name__ == "__main__":
    main()
