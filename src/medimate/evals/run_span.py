"""값 span eval — Candidate Recall · Baseline EM(tidy_value) · 선택기 EM. 결정론, LLM judge 없음.

  uv run python -m medimate.evals.run_span                    # rule 선택기, 호출 0
  uv run python -m medimate.evals.run_span --proposed         # gold 대신 rule 제안으로(연기 확인)
  uv run python -m medimate.evals.run_span --misses           # 틀린 조각을 전부 보인다
  uv run python -m medimate.evals.run_span --selector llm --provider bedrock \\
      --model apac.amazon.nova-pro-v1:0 --budget 0.40 --yes            # 실호출, 결과 저장
  uv run python -m medimate.evals.run_span --report evals/results/span-<model>.jsonl  # 재채점

시트 evals/span_cases.jsonl (scripts/build_span_sheet.py). gold가 빈 조각은 채점에서 뺀다.

지표 (docs/candidate-selection.md §4.2)
- CR   gold가 후보 안에 있는가 (공백·끝 구두점 무시). 채택 게이트: E0에서 ≥ 95%
- BASE 지금 코드 tidy_value(조각, 축) == gold
- EM   선택기가 고른 값 == gold. gold가 NONE(값 없음)이면 선택기도 NONE이어야 맞다
- SUB✗ gold가 조각의 부분 문자열이 아닌 조각 수(공백 무시). "부분 문자열 원칙"이 안 맞는 자리
- 그룹별·축별로 나눠 낸다. 비율은 같은 시트 안에서만 비교한다

gold는 후보와 무관하게 사람이 적는다. 그래야 CR이 후보 생성기를 검증한다.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

from medimate.dialog.memo import tidy_value
from medimate.schema.postvisit import PostAxis
from medimate.span.candidates import CandidateGenerator, compact
from medimate.span.select import (
    NONE,
    LLMSelector,
    ReplaySelector,
    RuleSelector,
    Selector,
    resolve,
)

ROOT = Path(__file__).resolve().parents[3]
SHEET = ROOT / "evals" / "span_cases.jsonl"
RESULTS = ROOT / "evals" / "results"


def load_sheet(path: Path = SHEET) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def same(a: str | None, b: str | None) -> bool:
    return a is not None and b is not None and compact(a) == compact(b)


class Tally:
    def __init__(self):
        self.n = 0
        self.cr = 0
        self.base = 0
        self.em = 0
        self.none = 0
        self.nonsub = 0  # gold가 부분 문자열이 아닌 조각

    def row(self, name: str) -> str:
        if not self.n:
            return f"| {name} | 0 | | | | |"
        p = lambda k: f"{k}/{self.n} = {100 * k / self.n:.0f}%"  # noqa: E731
        return (
            f"| {name} | {self.n} | {p(self.cr)} | {p(self.base)} | {p(self.em)} "
            f"| {self.none} | {self.nonsub} |"
        )


def run(
    selector: Selector,
    *,
    use_proposed: bool = False,
    show_misses: bool = False,
    save: Path | None = None,
) -> dict:
    gen = CandidateGenerator()
    saved: list[dict] = []
    by_group: dict[str, Tally] = defaultdict(Tally)
    by_axis: dict[str, Tally] = defaultdict(Tally)
    total = Tally()
    misses: list[str] = []
    skipped = 0

    for case in load_sheet():
        for idx, seg in enumerate(case["segments"]):
            if seg.get("skip"):
                continue
            gold = seg.get("proposed") if use_proposed else seg.get("gold")
            if not gold:
                skipped += 1
                continue
            axis = seg["label"]
            cset = gen.generate(seg["text"], axis)
            sel = selector.select(cset, axis, {"case_id": case["id"], "idx": idx})
            picked = resolve(cset, sel)
            if save is not None:
                saved.append(
                    {
                        "case_id": case["id"],
                        "idx": idx,
                        "group": case["group"],
                        "axis": axis,
                        "segment": cset.segment,
                        "candidates": [c.text for c in cset.candidates],
                        "choice": sel.candidate_id,
                        "choice_text": picked,
                        "note": sel.note,
                        "gold": gold,
                        "selector": selector.name,
                        "model_id": getattr(selector, "model_id", ""),
                        **(getattr(selector, "last", {}) or {}),
                    }
                )
            base = tidy_value(seg["text"], PostAxis(axis))
            gold_none = gold.strip().upper() == NONE
            if gold_none:
                # 값이 없는 조각. 선택기가 기권해야 맞고, 후보 목록에는 언제나 NONE이 있다고 본다
                hit_cr = True
                hit_base = False
                hit_em = sel.candidate_id == NONE
                nonsub = False
            else:
                hit_cr = cset.contains(gold) is not None
                hit_base = same(base, gold)
                hit_em = same(picked, gold)
                nonsub = compact(gold) not in compact(seg["text"])
            for t in (total, by_group[case["group"]], by_axis[axis]):
                t.n += 1
                t.cr += hit_cr
                t.base += hit_base
                t.em += hit_em
                t.none += sel.candidate_id == NONE
                t.nonsub += nonsub
            if show_misses and not (hit_cr and hit_em):
                tag = ("CR✗ " if not hit_cr else "") + ("EM✗" if not hit_em else "")
                if nonsub:
                    tag += " 부분문자열아님"
                misses.append(
                    f"{case['id']}  {tag:7} [{axis}] {seg['text']}\n"
                    f"      gold={gold!r} picked={picked!r} base={base!r}\n"
                    f"      후보: {[c.text for c in cset.candidates]}"
                )

    print(f"선택기 {selector.name} · 조각 {total.n}개 (gold 없음 {skipped}개 제외)\n")
    print("| 그룹 | n | CR | BASE(tidy_value) | EM | NONE | SUB✗ |\n|---|---|---|---|---|---|---|")
    for g in sorted(by_group):
        print(by_group[g].row(g))
    print(total.row("전체"))
    print("\n| 축 | n | CR | BASE | EM | NONE | SUB✗ |\n|---|---|---|---|---|---|---|")
    for a in ("findings", "tests", "medication_instructions", "follow_up"):
        if a in by_axis:
            print(by_axis[a].row(a))
    if misses:
        print("\n## 틀린 조각\n")
        print("\n".join(misses))
    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        with save.open("w", encoding="utf-8") as f:
            for r in saved:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        ti = sum(r.get("input_tokens", 0) for r in saved)
        to = sum(r.get("output_tokens", 0) for r in saved)
        usage = getattr(selector, "usage", None)
        cost = usage.cost_usd(selector.model_id) if usage else 0.0
        print(f"\n저장 {save} · 호출 {len(saved)} · 입력 {ti} tok · 출력 {to} tok · ${cost:.4f}")
    return {"n": total.n, "cr": total.cr, "base": total.base, "em": total.em}


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    a = argparse.ArgumentParser()
    a.add_argument("--selector", default="rule", choices=["rule", "llm"])
    a.add_argument("--provider", default="bedrock")
    a.add_argument("--model", default="apac.amazon.nova-pro-v1:0")
    a.add_argument("--budget", type=float, default=0.40)
    a.add_argument("--yes", action="store_true", help="비용 확인 없이 실행")
    a.add_argument("--report", type=Path, help="저장 결과로 재채점(호출 0)")
    a.add_argument("--proposed", action="store_true", help="gold 대신 rule 제안으로 CR 확인")
    a.add_argument("--misses", action="store_true")
    args = a.parse_args()

    if args.report:
        rows = [
            json.loads(ln)
            for ln in args.report.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        run(ReplaySelector(rows), show_misses=args.misses)
        return
    if args.selector == "rule":
        run(RuleSelector(), use_proposed=args.proposed, show_misses=args.misses)
        return

    from medimate.llm.providers import require_price

    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    n = sum(
        1 for c in load_sheet() for s_ in c["segments"] if not s_.get("skip") and s_.get("gold")
    )
    i, o = require_price(args.model)
    # 입력 ~1100 tok(한국어 시스템 프롬프트 포함), 출력 ~15 tok. 실측은 저장 파일에 남는다
    est = (n * 1100 * i + n * 15 * o) / 1e6
    print(f"{args.model}: 호출 {n}회, 예상 비용 약 ${est:.3f} (상한 ${args.budget:.2f})")
    if not args.yes:
        print("실행하려면 --yes")
        return
    sel = LLMSelector(args.provider, args.model, budget_usd=args.budget)
    out = RESULTS / f"span-{args.model.replace(':', '_')}.jsonl"
    run(sel, show_misses=args.misses, save=out)


if __name__ == "__main__":
    main()
