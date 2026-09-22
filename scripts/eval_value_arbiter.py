"""심판 실험 — 생성값(B안)과 rule 값이 다를 때 Jev가 둘 중 하나(또는 NONE)를 고르면 EM이 얼마나 오르나 (#122).

    uv run python scripts/eval_value_arbiter.py --dry-run          # 심판이 필요한 조각 수·비용만
    uv run python scripts/eval_value_arbiter.py --yes              # Jev 실호출(Nova 재호출 없음), 저장
    uv run python scripts/eval_value_arbiter.py --report evals/results/valgen-arbiter-jev-1.13.0.jsonl

입력은 저장된 생성 결과(valgen-<model>.jsonl). 조각마다 최신 검증기·폴백으로 생성값을 다시 만들고 rule 값을 옆에 둔다.
- 둘이 같으면 심판 없음(그 값)
- 생성값이 NONE이고 rule도 NONE이면 심판 없음(NONE)
- 그 외(둘이 다름 · 한쪽만 값)에는 Jev Choice: {A: 생성값, B: rule 값, NONE}

지표: 심판 전 EM(생성 우선) · 심판 후 EM · 상한(둘 중 정답이 있으면 맞힌 것으로) · confidence 구간별 정답률.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")

from medimate.llm.prompt_span_select import AXIS_KO, JEV_INSTRUCTIONS  # noqa: E402
from medimate.span.candidates import CandidateGenerator, compact  # noqa: E402
from medimate.span.generate import ValueGenerator  # noqa: E402
from medimate.span.select import RuleSelector, resolve  # noqa: E402
from medimate.text.lexicon import load_lexicon  # noqa: E402

RESULTS = Path("evals/results")
GEN = RESULTS / "valgen-apac.amazon.nova-pro-v1_0.jsonl"
JEV = "jev-1.13.0"

ARBITER_NOTE = (
    " 두 후보는 같은 조각에서 나온 서로 다른 값이다. 조각의 뜻을 바꾸지 않고 칸 모양에 더 맞는 쪽을 고른다. "
    "둘 다 틀리면 NONE."
)


def same(a, b) -> bool:
    return a is not None and b is not None and compact(str(a)) == compact(str(b))


def pairs() -> list[dict]:
    g = ValueGenerator.__new__(ValueGenerator)
    g.lexicon, g.gen, g.rule = load_lexicon(), CandidateGenerator(), RuleSelector()
    out = []
    for ln in GEN.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        got = None if r["model_none"] else r["model_value"]
        res = g.check(got, r["segment"], r["axis"])
        cset = g.gen.generate(r["segment"], r["axis"])
        rule = resolve(cset, g.rule.select(cset, r["axis"]))
        gen_val = res.value if res.source == "generated" else None
        # 폴백은 곧 rule 값이라 심판 대상이 아니다. 생성값이 검증을 통과한 것만 A로 둔다
        need = not same(gen_val, rule) and not (gen_val is None and rule is None)
        out.append(
            {
                **{k: r[k] for k in ("sheet", "case_id", "idx", "axis", "segment", "gold")},
                "gen": gen_val,
                "rule": rule,
                "source": res.source,
                "need": need,
            }
        )
    return out


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    a = argparse.ArgumentParser()
    a.add_argument("--budget", type=float, default=0.05)
    a.add_argument("--dry-run", action="store_true")
    a.add_argument("--yes", action="store_true")
    a.add_argument("--report", type=Path)
    args = a.parse_args()

    rows = pairs()
    need = [r for r in rows if r["need"]]
    print(f"조각 {len(rows)} · 심판 필요 {len(need)} · Jev 예상 ${len(need) * 400 * 0.042 / 1e6:.4f}")
    if args.report:
        saved = {
            (x["case_id"], x["idx"]): x
            for x in (
                json.loads(ln)
                for ln in args.report.read_text(encoding="utf-8").splitlines()
                if ln.strip()
            )
        }
    else:
        if args.dry_run or not args.yes:
            print("실행하려면 --yes")
            return
        from dotenv import load_dotenv

        load_dotenv(".env")
        from typesafe_sdk import Choice, TypeSafeClient

        from medimate.llm.providers import BudgetExceeded, Usage
        from medimate.obs import tracing

        client = TypeSafeClient(model=JEV)
        usage = Usage()
        saved = {}
        out = RESULTS / f"valgen-arbiter-{JEV}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for r in need:
                if usage.cost_usd(JEV) >= args.budget:
                    raise BudgetExceeded(f"{JEV}: ${args.budget} 상한 도달")
                crit = {}
                if r["gen"] is not None:
                    crit["A"] = r["gen"]
                if r["rule"] is not None:
                    crit["B"] = r["rule"]
                crit["NONE"] = "둘 다 맞지 않음 · 값 없음"
                state = {"칸": AXIS_KO.get(r["axis"], r["axis"]), "조각": r["segment"]}
                t0 = time.perf_counter()
                with tracing.context(
                    trace_name="value-arbiter-eval",
                    session_id=f"valgen-arbiter:{JEV}",
                    tags=["eval", "valgen", "arbiter", r["sheet"], r["axis"]],
                    metadata={k: r[k] for k in ("sheet", "case_id", "idx", "axis", "gold")},
                ):
                    with tracing.generation(
                        "jev-arbiter",
                        model=JEV,
                        input={"state": state, "criteria": crit},
                        version="value-arbiter-v1",
                    ) as gobs:
                        resp = client.system_one(
                            state=state,
                            questions={
                                "value": Choice(
                                    instructions=JEV_INSTRUCTIONS.get(r["axis"], "") + ARBITER_NOTE,
                                    criteria=crit,
                                )
                            },
                        )
                        ans = resp.answers["value"]
                        choice = str(getattr(ans, "choice", "")).strip()
                        conf = getattr(ans, "confidence", None)
                        probs = getattr(ans, "probabilities", None)
                        ti = getattr(resp.usage, "input_tokens", 0) or 0
                        gobs.done(
                            output={"choice": choice, "confidence": conf},
                            input_tokens=ti,
                            output_tokens=0,
                            cost_usd=ti * 0.042 / 1e6,
                        )
                    usage.calls += 1
                    usage.input_tokens += ti
                    picked = {"A": r["gen"], "B": r["rule"]}.get(choice)
                    em = (
                        (picked is None)
                        if r["gold"].upper() == "NONE"
                        else same(picked, r["gold"])
                    )
                    tracing.score("em", 1.0 if em else 0.0)
                    if conf is not None:
                        tracing.score("confidence", float(conf))
                row = {
                    **r,
                    "choice": choice,
                    "picked": picked,
                    "confidence": conf,
                    "probabilities": dict(probs) if probs else None,
                    "input_tokens": ti,
                    "latency_s": round(time.perf_counter() - t0, 3),
                }
                saved[(r["case_id"], r["idx"])] = row
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(".", end="", flush=True)
        tracing.flush()
        print(f"\n저장 {out} · 호출 {usage.calls} · ${usage.cost_usd(JEV):.4f}")

    # 채점
    tot = Counter()
    by_sheet: dict[str, Counter] = {}
    conf_bins: dict[str, Counter] = {}
    misses = []
    for r in rows:
        gold, gn = r["gold"], r["gold"].upper() == "NONE"
        ok = lambda v: (v is None) if gn else same(v, gold)  # noqa: E731
        before = r["gen"] if r["gen"] is not None else r["rule"]  # 생성 우선, 없으면 폴백(rule)
        s = by_sheet.setdefault(r["sheet"], Counter())
        for c in (tot, s):
            c["n"] += 1
            c["before"] += ok(before)
            c["ceiling"] += ok(r["gen"]) or ok(r["rule"]) or (gn and (r["gen"] is None or r["rule"] is None))
        if not r["need"]:
            for c in (tot, s):
                c["after"] += ok(before)
            continue
        j = saved.get((r["case_id"], r["idx"]))
        if j is None:
            for c in (tot, s):
                c["after"] += ok(before)
            continue
        after = j["picked"]
        for c in (tot, s):
            c["after"] += ok(after)
            c["judged"] += 1
            c["j_right"] += ok(after)
        conf = j.get("confidence") or 0.0
        b = "≥0.9" if conf >= 0.9 else "0.5~0.9" if conf >= 0.5 else "<0.5"
        cb = conf_bins.setdefault(b, Counter())
        cb["n"] += 1
        cb["right"] += ok(after)
        if not ok(after):
            misses.append(
                f"{r['sheet']}/{r['case_id']}.{r['idx']} [{r['axis']}] {r['segment']}\n"
                f"      gold={gold!r} A(gen)={r['gen']!r} B(rule)={r['rule']!r} → {j['choice']} conf={conf:.2f}"
            )

    def row(name, c):
        n = c["n"] or 1
        return (
            f"| {name} | {c['n']} | {c['before']} ({100 * c['before'] / n:.0f}%) | {c['after']} ({100 * c['after'] / n:.0f}%) "
            f"| {c['ceiling']} ({100 * c['ceiling'] / n:.0f}%) | {c['judged']} | {c['j_right']} |"
        )

    print("\n| 시트 | n | 심판 전 EM | **심판 후 EM** | 상한(둘 중 정답 있음) | 심판 건수 | 심판 정답 |")
    print("|---|---|---|---|---|---|---|")
    for k in ("unseen_r1", "unseen_r3", "raw", "normal"):
        if k in by_sheet:
            print(row(k, by_sheet[k]))
    print(row("전체", tot))
    print("\n| confidence | 심판 건수 | 정답 |\n|---|---|---|")
    for k in ("≥0.9", "0.5~0.9", "<0.5"):
        if k in conf_bins:
            c = conf_bins[k]
            print(f"| {k} | {c['n']} | {c['right']} ({100 * c['right'] / c['n']:.0f}%) |")
    if misses:
        print("\n## 심판이 틀린 조각\n" + "\n".join(misses))


if __name__ == "__main__":
    main()
