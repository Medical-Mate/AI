"""검증 가능한 생성(B안) 시험 — 모델이 값을 쓰고 검증기가 거르면 EM이 얼마나 나오나 (#122).

    uv run python scripts/eval_value_gen.py --dry-run                       # 조각 수·비용만
    uv run python scripts/eval_value_gen.py --yes                           # Nova 실호출, 저장
    uv run python scripts/eval_value_gen.py --report evals/results/valgen-<model>.jsonl  # 재채점(호출 0)
    uv run python scripts/eval_value_gen.py --dry-run --misses              # (재채점과 함께) 틀린 조각
    uv run python scripts/eval_value_gen.py --sheet unseen_r4 --yes         # 한 시트만(결과 파일에 시트 이름이 붙는다)

기본은 시트 다섯 장을 한 번에 돈다(프롬프트를 고치면 전부). 그룹은 시트별로 나눠 센다:
- unseen_r1  evals/span_unseen_cases.jsonl     Codex 1라운드(코드가 이 메모를 보고 고쳐졌음 — 비교선 76/53)
- unseen_r3  evals/span_unseen_r3_cases.jsonl  Codex 3라운드(v2 프롬프트를 고칠 때 봤음 — 비교선 rule 29%, Nova 선택 31%)
- unseen_r4  evals/span_unseen_r4_cases.jsonl  Codex 4라운드(코드·프롬프트 미접촉, 동결 2d3f064 — 비교선 rule 37%)
- raw        evals/span_raw_cases.jsonl        날것 메모 43조각
- normal     evals/span_cases.jsonl E0_normal   정상 메모(회귀 — rule 98%)

지표
- EM        최종 값 == gold (검증 통과값 또는 폴백값). gold NONE이면 최종도 None이어야 맞다
- GEN-EM    모델이 쓴 값(검증 전) == gold. 검증기가 없었다면의 성적
- PASS      검증 통과 수. 통과했지만 틀린 값(PASS✗)은 검증기가 못 잡은 것 — 유형을 눈으로 본다
- REJ       검증 거절 수와 이유. 거절했는데 gold였던 것(REJ✗)은 검증기가 과하게 잡은 것
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "src")

from medimate.llm.prompt_value_gen import PROMPT_VERSION  # noqa: E402
from medimate.llm.providers import require_price  # noqa: E402
from medimate.span.candidates import compact  # noqa: E402
from medimate.span.generate import ValueGenerator  # noqa: E402

RESULTS = Path("evals/results")
SHEETS = [
    ("unseen_r1", "evals/span_unseen_cases.jsonl", None),
    ("unseen_r3", "evals/span_unseen_r3_cases.jsonl", None),
    ("unseen_r4", "evals/span_unseen_r4_cases.jsonl", None),
    ("raw", "evals/span_raw_cases.jsonl", None),
    ("normal", "evals/span_cases.jsonl", {"E0_normal"}),
]


def segments(only: set[str] | None = None) -> list[dict]:
    out: list[dict] = []
    for name, path, groups in SHEETS:
        if only and name not in only:
            continue
        for ln in Path(path).read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            c = json.loads(ln)
            if groups and c["group"] not in groups:
                continue
            for i, s in enumerate(c["segments"]):
                if s.get("skip") or not s.get("gold"):
                    continue
                out.append(
                    {
                        "sheet": name,
                        "case_id": c["id"],
                        "idx": i,
                        "axis": s["label"],
                        "segment": s["text"],
                        "gold": s["gold"],
                    }
                )
    return out


def same(a, b) -> bool:
    if a is None or b is None:
        return False
    return compact(str(a)) == compact(str(b))


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    a = argparse.ArgumentParser()
    a.add_argument("--provider", default="bedrock")
    a.add_argument("--model", default="apac.amazon.nova-pro-v1:0")
    a.add_argument("--budget", type=float, default=1.0)
    a.add_argument("--limit", type=int, default=0)
    a.add_argument("--dry-run", action="store_true")
    a.add_argument("--yes", action="store_true")
    a.add_argument("--report", type=Path)
    a.add_argument("--misses", action="store_true")
    a.add_argument("--sheet", action="append", choices=[n for n, _, _ in SHEETS], help="이 시트만(여러 번 가능)")
    args = a.parse_args()

    only = set(args.sheet or [])
    segs = segments(only)
    if args.limit:
        segs = segs[: args.limit]
    per_sheet = Counter(s["sheet"] for s in segs)

    if args.report:
        rows = [json.loads(ln) for ln in args.report.read_text(encoding="utf-8").splitlines() if ln.strip()]
        model = rows[0].get("model_id", args.model) if rows else args.model
        gen = ValueGenerator.__new__(ValueGenerator)  # 호출 0 — check()만 쓴다
        from medimate.span.candidates import CandidateGenerator
        from medimate.span.select import RuleSelector
        from medimate.text.lexicon import load_lexicon

        gen.lexicon, gen.gen, gen.rule, gen.model_id = load_lexicon(), CandidateGenerator(), RuleSelector(), model
        cost = None
        # gold는 지금 시트에서 읽는다 — 시트를 고친 뒤(중이염 → 귀지 때문) 재채점이 옛 gold로 세지 않게
        now = {(s["sheet"], s["case_id"], s["idx"]): s["gold"] for s in segments()}
        changed = 0
        for r in rows:
            g = now.get((r["sheet"], r["case_id"], r["idx"]))
            if g is not None and g != r["gold"]:
                r["gold"], changed = g, changed + 1
        if changed:
            print(f"저장 뒤 시트에서 gold가 바뀐 조각 {changed}개 — 지금 gold로 센다")
    else:
        i, o = require_price(args.model)
        est = (len(segs) * 1300 * i + len(segs) * 20 * o) / 1e6
        print(
            f"조각 {len(segs)} ({dict(per_sheet)}) · 호출 {len(segs)} · 예상 ${est:.3f} (상한 ${args.budget})"
        )
        if args.dry_run or not args.yes:
            print("실행하려면 --yes")
            return
        try:
            from dotenv import load_dotenv

            load_dotenv(".env")
        except ImportError:
            pass
        from medimate.obs import tracing

        gen = ValueGenerator(args.provider, args.model, budget_usd=args.budget)
        # 시트를 골랐으면 이름을 붙인다 — 전체 결과 파일을 덮어쓰지 않게
        suffix = "".join(f"-{n}" for n, _, _ in SHEETS if n in only) + (f"-first{args.limit}" if args.limit else "")
        out = RESULTS / f"valgen-{args.model.replace(':', '_')}{suffix}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        with out.open("w", encoding="utf-8") as f:
            for s in segs:
                with tracing.context(
                    trace_name="value-gen-eval",
                    session_id=f"valgen-eval:{args.model}:{PROMPT_VERSION}",
                    tags=["eval", "valgen", s["sheet"], s["axis"]],
                    metadata={**{k: s[k] for k in ("sheet", "case_id", "idx", "axis")}, "gold": s["gold"]},
                ):
                    g = gen.generate(s["segment"], s["axis"])
                    em = (g.value is None) if s["gold"].upper() == "NONE" else same(g.value, s["gold"])
                    tracing.score("em", 1.0 if em else 0.0, comment=f"gold={s['gold']} value={g.value}")
                    tracing.score("verified", 1.0 if g.source != "fallback" else 0.0, comment=g.dropped)
                row = {
                    **s,
                    "model_value": g.model_value if g.source != "none" else None,
                    "model_none": g.source == "none",
                    "value": g.value,
                    "source": g.source,
                    "dropped": g.dropped,
                    "raw_response": gen.last.get("text"),
                    "input_tokens": gen.last.get("input_tokens"),
                    "output_tokens": gen.last.get("output_tokens"),
                    "prompt_version": PROMPT_VERSION,
                    "model_id": args.model,
                }
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(".", end="", flush=True)
        tracing.flush()
        cost = gen.usage.cost_usd(args.model)
        print(f"\n저장 {out} · ${cost:.4f}")

    # 재채점 — 검증기·폴백은 코드라 저장된 model_value에 다시 적용한다
    class T:
        def __init__(self):
            self.c = Counter()

    by = defaultdict(T)
    by_axis = defaultdict(T)
    total = T()
    reasons = Counter()
    misses: list[str] = []
    for r in rows:
        got = None if r.get("model_none") else r.get("model_value")
        g = gen.check(got, r["segment"], r["axis"])
        gold_none = r["gold"].strip().upper() == "NONE"
        em = (g.value is None) if gold_none else same(g.value, r["gold"])
        gen_em = (got is None) if gold_none else same(got, r["gold"])
        for t in (total, by[r["sheet"]], by_axis[r["axis"]]):
            t.c["n"] += 1
            t.c["em"] += em
            t.c["gen_em"] += gen_em
            t.c["none"] += g.source == "none"
            if g.source == "generated":
                t.c["pass"] += 1
                t.c["pass_wrong"] += not em
            elif g.source == "fallback":
                t.c["rej"] += 1
                t.c["rej_gold"] += gen_em  # 거절했는데 모델 값이 gold였다
                t.c["rej_saved"] += em and not gen_em  # 거절이 틀린 값을 잡고 폴백이 맞았다
        if g.source == "fallback":
            reasons[g.verdict.reasons[0].split(":")[0] if g.verdict.reasons else "?"] += 1
        if args.misses and not em:
            misses.append(
                f"{r['sheet']}/{r['case_id']}.{r['idx']} [{r['axis']}] {r['segment']}\n"
                f"      gold={r['gold']!r} model={got!r} final={g.value!r} ({g.source}) {g.dropped}"
            )

    def row(name, t):
        c = t.c
        n = c["n"] or 1
        return (
            f"| {name} | {c['n']} | {c['em']} ({100 * c['em'] / n:.0f}%) | {c['gen_em']} ({100 * c['gen_em'] / n:.0f}%) "
            f"| {c['none']} | {c['pass']} | {c['pass_wrong']} | {c['rej']} | {c['rej_gold']} | {c['rej_saved']} |"
        )

    print(f"\n{PROMPT_VERSION} · {getattr(gen, 'model_id', args.model)} · 조각 {total.c['n']}")
    print(
        "\n| 시트 | n | EM(최종) | GEN-EM(검증 전) | 모델 NONE | 통과 | 통과✗ | 거절 | 거절이 gold | 거절이 구함 |"
        "\n|---|---|---|---|---|---|---|---|---|---|"
    )
    for k in (n for n, _, _ in SHEETS):
        if k in by:
            print(row(k, by[k]))
    print(row("전체", total))
    print("\n| 축 | n | EM | GEN-EM | NONE | 통과 | 통과✗ | 거절 | 거절이 gold | 거절이 구함 |\n|---|---|---|---|---|---|---|---|---|---|")
    for k in ("findings", "tests", "medication_instructions", "follow_up"):
        if k in by_axis:
            print(row(k, by_axis[k]))
    print(f"\n거절 이유: {dict(reasons)}")
    if cost is not None:
        print(f"비용 ${cost:.4f}")
    if misses:
        print("\n## 틀린 조각\n" + "\n".join(misses))


if __name__ == "__main__":
    main()
