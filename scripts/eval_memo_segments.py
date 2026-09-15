"""memo-v5(조각내기 + 라벨) 실측 — 메모 벡터 34개를 Nova에 태워 덮음·라벨을 잰다.

    uv run python scripts/eval_memo_segments.py                 # Nova Pro, 상한 $0.10

재는 것
- 덮음: 조각을 순서대로 이으면 원문인가(공백 무시). **이게 0이면 채택 자체가 안 된다**
- 조각 일치: LLM 조각이 사람 문장(벡터의 sentences)과 같은 개수·같은 경계인가
- 라벨: 경계가 같은 조각에서 사람 정답(expected_labels)과 같은가
원본은 evals/results/memo-v5-<model>.jsonl (gitignore).
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "src")

from medimate.dialog.memo import slice_by_coverage  # noqa: E402
from medimate.llm.memo_classifier import LLMMemoSegmenter  # noqa: E402

VEC = Path("evals/ondevice/vectors-memo-qwen3-1.7b-q4_0-memo-small-v4.jsonl")
_WS = re.compile(r"\s+")


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    a = argparse.ArgumentParser()
    a.add_argument("--provider", default="bedrock")
    a.add_argument("--model", default="apac.amazon.nova-pro-v1:0")
    a.add_argument("--budget", type=float, default=0.10)
    a.add_argument("--limit", type=int, default=0)
    args = a.parse_args()

    vectors = [json.loads(ln) for ln in VEC.open(encoding="utf-8") if ln.strip()]
    if args.limit:
        vectors = vectors[: args.limit]
    seg = LLMMemoSegmenter(args.provider, args.model, budget_usd=args.budget)
    out_path = Path("evals/results") / f"memo-v5-{args.model.replace(':', '_')}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cover_ok = same_bounds = label_ok = label_n = 0
    rows = []
    print(f"모델 {args.model}  벡터 {len(vectors)}개\n")
    for v in vectors:
        memo = " ".join(v["sentences"])  # 원문은 안 실려 있어 사람 문장을 공백으로 잇는다
        try:
            out = seg.segment(memo)
        except Exception as e:  # noqa: BLE001
            print(f"ERR  {v['id']}  {type(e).__name__}: {e}")
            rows.append({"id": v["id"], "error": f"{type(e).__name__}: {e}"})
            continue
        segs = out.get("segments") or []
        texts = [s_.get("text", "") for s_ in segs]
        labs = [s_.get("label") for s_ in segs]
        sliced = slice_by_coverage(memo, texts)
        covered = sliced is not None
        cover_ok += covered
        exp_s = [_WS.sub("", x) for x in v["sentences"]]
        exp_l = list(v["expected_labels"])
        got_s = [_WS.sub("", x) for x in (sliced or texts)]
        bounds = got_s == exp_s
        same_bounds += bounds
        hits = misses = 0
        if bounds:
            for g, e in zip(labs, exp_l, strict=False):
                label_n += 1
                if g == e:
                    label_ok += 1
                    hits += 1
                else:
                    misses += 1
        mark = " OK " if covered and bounds and misses == 0 else ("cov " if covered else "FAIL")
        print(
            f"{mark} {v['id']:6s} 조각 {len(texts)}/{len(exp_s)}  덮음 {'O' if covered else 'X'}  "
            f"경계 {'O' if bounds else 'X'}  라벨 {hits}/{hits + misses if bounds else '-'}"
        )
        if not covered or not bounds:
            print(f"        사람: {v['sentences']}")
            print(f"        LLM : {texts}")
        rows.append(
            {
                "id": v["id"],
                "memo": memo,
                "expected": v["sentences"],
                "expected_labels": exp_l,
                "segments": segs,
                "covered": covered,
                "same_bounds": bounds,
                "label_hits": hits,
                "label_misses": misses,
            }
        )

    u = seg.usage
    cost = u.cost_usd(args.model)
    n = len(vectors)
    print(
        f"\n덮음 {cover_ok}/{n}  경계일치 {same_bounds}/{n}  "
        f"라벨(경계 같은 것만) {label_ok}/{label_n}   호출 {u.calls}  ${cost:.4f}"
    )
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.write(
            json.dumps(
                {
                    "_summary": True,
                    "model": args.model,
                    "prompt_version": seg.prompt_version,
                    "n": n,
                    "covered": cover_ok,
                    "same_bounds": same_bounds,
                    "label_ok": label_ok,
                    "label_n": label_n,
                    "calls": u.calls,
                    "input_tokens": u.input_tokens,
                    "output_tokens": u.output_tokens,
                    "cost_usd": round(cost, 6),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    print(f"원본 {out_path}")


if __name__ == "__main__":
    main()
