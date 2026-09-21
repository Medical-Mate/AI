"""값 span eval — Candidate Recall · Baseline EM(tidy_value) · 선택기 EM. 결정론, LLM judge 없음.

  uv run python -m medimate.evals.run_span                    # rule 선택기, 호출 0
  uv run python -m medimate.evals.run_span --proposed         # gold 대신 rule 제안으로(연기 확인)
  uv run python -m medimate.evals.run_span --misses           # 틀린 조각을 전부 보인다

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
from medimate.span.select import NONE, RuleSelector, Selector, resolve

ROOT = Path(__file__).resolve().parents[3]
SHEET = ROOT / "evals" / "span_cases.jsonl"


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


def run(selector: Selector, *, use_proposed: bool = False, show_misses: bool = False) -> dict:
    gen = CandidateGenerator()
    by_group: dict[str, Tally] = defaultdict(Tally)
    by_axis: dict[str, Tally] = defaultdict(Tally)
    total = Tally()
    misses: list[str] = []
    skipped = 0

    for case in load_sheet():
        for seg in case["segments"]:
            if seg.get("skip"):
                continue
            gold = seg.get("proposed") if use_proposed else seg.get("gold")
            if not gold:
                skipped += 1
                continue
            axis = seg["label"]
            cset = gen.generate(seg["text"], axis)
            sel = selector.select(cset, axis)
            picked = resolve(cset, sel)
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
    return {"n": total.n, "cr": total.cr, "base": total.base, "em": total.em}


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    a = argparse.ArgumentParser()
    a.add_argument("--selector", default="rule", choices=["rule"])
    a.add_argument("--proposed", action="store_true", help="gold 대신 rule 제안으로 CR 확인")
    a.add_argument("--misses", action="store_true")
    args = a.parse_args()
    run(RuleSelector(), use_proposed=args.proposed, show_misses=args.misses)


if __name__ == "__main__":
    main()
