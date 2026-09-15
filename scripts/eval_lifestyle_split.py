"""약/지침 분리(memo-small-v5) 축소 실측 — 생활 지시가 섞인 벡터만 골라 Nova에 태운다.

    uv run python scripts/eval_lifestyle_split.py          # 12개, 상한 $0.05

34개 전부 다시 돌리지 않는다(시간·비용). 약 칸에 생활 지시가 섞인 벡터 12개만 고르고,
그 문장들의 정답을 지침으로 다시 매긴 표(`OVERRIDE`)로 채점한다. 나머지 문장의 정답은 원래대로.
원본은 evals/results/lifestyle-<model>.jsonl (gitignore).
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from medimate.llm.memo_classifier import LLMMemoClassifier  # noqa: E402

VEC = Path("evals/ondevice/vectors-memo-qwen3-1.7b-q4_0-memo-small-v4.jsonl")
PICK = ["PM01", "PM02", "PM05", "PM06", "PM10", "PM12", "PM14", "PM18", "PM19", "PN06", "PN08", "PM22"]

# 지침으로 다시 매긴 문장. 약과 생활 지시가 한 문장에 같이 있으면(PM01·PM05) 약 그대로.
OVERRIDE = {
    "계단은 피하라고.": "lifestyle_instructions",
    "짜게 먹지 말라고 하셨어요.": "lifestyle_instructions",
    "얼음찜질하라고.": "lifestyle_instructions",
    "먼지 많은 곳 피하라고": "lifestyle_instructions",
    "운동 주 3회 이상 하라고.": "lifestyle_instructions",
    "식사량 줄이라고.": "lifestyle_instructions",
    "수영 금지.": "lifestyle_instructions",
    "무거운 거 들지 말라고.": "lifestyle_instructions",
    "요오드 많은 음식은 피하라고.": "lifestyle_instructions",
    "카페인 오후 금지.": "lifestyle_instructions",
    "무리하지 말라고 하셨어요.": "lifestyle_instructions",
}


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    a = argparse.ArgumentParser()
    a.add_argument("--provider", default="bedrock")
    a.add_argument("--model", default="apac.amazon.nova-pro-v1:0")
    a.add_argument("--budget", type=float, default=0.05)
    args = a.parse_args()

    vectors = {json.loads(ln)["id"]: json.loads(ln) for ln in VEC.open(encoding="utf-8") if ln.strip()}
    clf = LLMMemoClassifier(args.provider, args.model, budget_usd=args.budget)
    out_path = Path("evals/results") / f"lifestyle-{args.model.replace(':', '_')}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = ok = split_n = split_ok = 0
    rows = []
    print(f"모델 {args.model}  프롬프트 {clf.prompt_version}  벡터 {len(PICK)}개\n")
    for vid in PICK:
        v = vectors[vid]
        sents = v["sentences"]
        expect = [OVERRIDE.get(s, lab) for s, lab in zip(sents, v["expected_labels"], strict=False)]
        got = clf.classify(sents).labels
        misses = []
        for s_, e, g in zip(sents, expect, got, strict=False):
            total += 1
            is_split_case = s_ in OVERRIDE or e == "medication_instructions"
            if is_split_case:
                split_n += 1
            if g == e:
                ok += 1
                if is_split_case:
                    split_ok += 1
            else:
                misses.append((s_, e, g))
        mark = " OK " if not misses else "FAIL"
        print(f"{mark} {vid}  {len(sents) - len(misses)}/{len(sents)}")
        for s_, e, g in misses:
            print(f"        {s_!r}\n          정답 {e}   LLM {g}")
        rows.append({"id": vid, "sentences": sents, "expect": expect, "got": got, "misses": misses})

    u = clf.usage
    cost = u.cost_usd(args.model)
    print(
        f"\n전체 라벨 {ok}/{total}   약·지침 관련 문장 {split_ok}/{split_n}   "
        f"호출 {u.calls}  ${cost:.4f}"
    )
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.write(
            json.dumps(
                {
                    "_summary": True,
                    "model": args.model,
                    "prompt_version": clf.prompt_version,
                    "ok": ok,
                    "total": total,
                    "split_ok": split_ok,
                    "split_n": split_n,
                    "calls": u.calls,
                    "cost_usd": round(cost, 6),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    print(f"원본 {out_path}")


if __name__ == "__main__":
    main()
