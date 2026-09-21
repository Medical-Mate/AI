"""저장 결과로 선택기 비교 + cascade 시뮬레이션. 호출 0.

    uv run python scripts/span_cascade.py

읽는 것: evals/results/span-jev-1.13.0.jsonl, span-apac.amazon.nova-pro-v1_0*.jsonl (정상 243 + 날것 43).
내는 것: 선택기별 EM · Jev confidence 구간별 EM · cascade 조합 EM · Nova/Jev 일치율 · 비용·지연.
cascade는 저장된 답을 조합할 뿐 새 호출이 없다.
"""

from __future__ import annotations

import io
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "src")

from medimate.evals.run_span import load_sheet  # noqa: E402
from medimate.span.candidates import CandidateGenerator, compact  # noqa: E402
from medimate.span.select import RuleSelector, resolve  # noqa: E402

R = Path("evals/results")
JEV = R / "span-jev-1.13.0.jsonl"
NOVA = [R / "span-apac.amazon.nova-pro-v1_0.jsonl", R / "span-apac.amazon.nova-pro-v1_0-E3_raw+E4_abbrev+E5_casual.jsonl"]


def rows(*paths: Path) -> dict[tuple[str, int], dict]:
    out = {}
    for p in paths:
        if p.exists():
            for ln in p.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    r = json.loads(ln)
                    out[(r["case_id"], r["idx"])] = r
    return out


def hit(value: str | None, gold: str) -> bool:
    if gold.strip().upper() == "NONE":
        return value is None
    return value is not None and compact(value) == compact(gold)


def pct(a: int, b: int) -> str:
    return f"{a}/{b} = {100 * a / b:.0f}%" if b else "-"


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    jev, nova = rows(JEV), rows(*NOVA)
    gen, rule = CandidateGenerator(), RuleSelector()

    recs = []
    for case in load_sheet():
        for idx, seg in enumerate(case["segments"]):
            if seg.get("skip") or not seg.get("gold"):
                continue
            key = (case["id"], idx)
            if key not in jev or key not in nova:
                continue
            cset = gen.generate(seg["text"], seg["label"])
            r_sel = rule.select(cset, seg["label"])
            recs.append(
                {
                    "group": case["group"],
                    "axis": seg["label"],
                    "gold": seg["gold"],
                    "rule": resolve(cset, r_sel),
                    "rule_none": r_sel.candidate_id == "NONE",
                    "jev": jev[key]["choice_text"],
                    "conf": jev[key].get("confidence") or 0.0,
                    "nova": nova[key]["choice_text"],
                    "jev_lat": jev[key].get("latency_s", 0),
                    "nova_lat": nova[key].get("latency_s", 0),
                    "jev_in": jev[key].get("input_tokens", 0),
                    "nova_in": nova[key].get("input_tokens", 0),
                }
            )
    n = len(recs)
    print(f"조각 {n}개 (세 선택기 결과가 모두 있는 것)\n")

    # 선택기별
    print("| 선택기 | EM | 비용(저장분) | 지연 중앙값 |\n|---|---|---|---|")
    em_rule = sum(hit(r["rule"], r["gold"]) for r in recs)
    em_jev = sum(hit(r["jev"], r["gold"]) for r in recs)
    em_nova = sum(hit(r["nova"], r["gold"]) for r in recs)
    jev_cost = sum(r["jev_in"] for r in recs) * 0.042 / 1e6
    nova_cost = sum(r["nova_in"] for r in recs) * 0.80 / 1e6
    print(f"| rule | {pct(em_rule, n)} | $0 | 0 |")
    print(f"| Jev 1.13 | {pct(em_jev, n)} | ${jev_cost:.4f} | {statistics.median(r['jev_lat'] for r in recs):.2f}s |")
    print(f"| Nova Pro | {pct(em_nova, n)} | ${nova_cost:.3f} | {statistics.median(r['nova_lat'] for r in recs):.2f}s |")

    # 그룹별 세 선택기
    print("\n| 그룹 | n | rule | Jev | Nova |\n|---|---|---|---|---|")
    by_g = defaultdict(list)
    for r in recs:
        by_g[r["group"]].append(r)
    for g in sorted(by_g):
        rs = by_g[g]
        print(
            f"| {g} | {len(rs)} | {pct(sum(hit(r['rule'], r['gold']) for r in rs), len(rs))} "
            f"| {pct(sum(hit(r['jev'], r['gold']) for r in rs), len(rs))} "
            f"| {pct(sum(hit(r['nova'], r['gold']) for r in rs), len(rs))} |"
        )

    # confidence 보정
    print("\n| Jev confidence | n | EM |\n|---|---|---|")
    for lo, hi in ((0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)):
        rs = [r for r in recs if lo <= r["conf"] < hi]
        print(f"| {lo:.1f}~{min(hi, 1.0):.1f} | {len(rs)} | {pct(sum(hit(r['jev'], r['gold']) for r in rs), len(rs))} |")

    # cascade
    print("\n| cascade | EM | 모델 호출 비율 |\n|---|---|---|")
    print(f"| rule → Jev (rule이 NONE일 때만) | {pct(sum(hit(r['jev'] if r['rule_none'] else r['rule'], r['gold']) for r in recs), n)} | {pct(sum(r['rule_none'] for r in recs), n)} |")
    print(f"| rule → Nova (rule이 NONE일 때만) | {pct(sum(hit(r['nova'] if r['rule_none'] else r['rule'], r['gold']) for r in recs), n)} | {pct(sum(r['rule_none'] for r in recs), n)} |")
    for th in (0.5, 0.7, 0.9):
        em = sum(hit(r["jev"] if r["conf"] >= th else r["rule"], r["gold"]) for r in recs)
        print(f"| Jev(conf≥{th}) → rule | {pct(em, n)} | Jev 채택 {pct(sum(r['conf'] >= th for r in recs), n)} |")
    for th in (0.5, 0.7, 0.9):
        em = sum(hit(r["jev"] if r["conf"] >= th else r["nova"], r["gold"]) for r in recs)
        print(f"| Jev(conf≥{th}) → Nova | {pct(em, n)} | Nova 호출 {pct(sum(r['conf'] < th for r in recs), n)} |")

    # 일치
    agree = sum(compact(r["jev"] or "NONE") == compact(r["nova"] or "NONE") for r in recs)
    both_wrong = sum(not hit(r["jev"], r["gold"]) and not hit(r["nova"], r["gold"]) for r in recs)
    jev_only = sum(hit(r["jev"], r["gold"]) and not hit(r["nova"], r["gold"]) for r in recs)
    nova_only = sum(hit(r["nova"], r["gold"]) and not hit(r["jev"], r["gold"]) for r in recs)
    print(f"\nJev·Nova 같은 답 {pct(agree, n)} · 둘 다 오답 {both_wrong} · Jev만 맞음 {jev_only} · Nova만 맞음 {nova_only}")
    rule_miss = [r for r in recs if not hit(r["rule"], r["gold"])]
    print(f"rule 오답 {len(rule_miss)} 중 Jev 맞음 {sum(hit(r['jev'], r['gold']) for r in rule_miss)} · Nova 맞음 {sum(hit(r['nova'], r['gold']) for r in rule_miss)}")


if __name__ == "__main__":
    main()
