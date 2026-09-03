"""Extractor eval 러너.

  uv run python -m medimate.evals.run --dry-run                       # 호출 0, 파이프라인 점검
  uv run python -m medimate.evals.run --provider google --model gemini-3.5-flash
  uv run python -m medimate.evals.run --report evals/results/claude-haiku-4-5.jsonl   # 재채점만

원본 응답은 evals/results/<model>.jsonl 에 전부 저장한다. 채점 로직을 고쳐도 재호출하지 않는다.
모델당 지출 상한(--budget, 기본 $0.50)을 넘으면 호출 전에 멈춘다.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from medimate.evals.score import ROOT, load_cases, load_lexicon, score
from medimate.llm.base import TurnExtraction
from medimate.llm.providers import PRICES, BudgetExceeded, LLMExtractor, RawResult, _parse_json
from medimate.schema.card import Axis

RESULTS = ROOT / "evals" / "results"


class DryRunExtractor:
    """호출 없이 러너를 끝까지 돌려보는 가짜. 기대값을 그대로 돌려준다 (전부 통과해야 정상)."""

    model_id = "dry-run"
    prompt_version = "dry"

    def __init__(self, cases):
        self._by_utt = {c["utterance"]: c for c in cases}

    def extract_raw(self, utterance, asked_axis) -> RawResult:
        c = self._by_utt[utterance]
        must = c["expect"].get("value_must_contain", {})
        ext = TurnExtraction(
            chief_complaint=utterance if c["first_turn"] else None,
            updates=[
                {"axis": a, "status": st, "value": must.get(a, "값"), "evidence": utterance}
                for a, st in c["expect"]["axes"].items()
            ],
            wants_to_stop=c["expect"]["wants_to_stop"],
        )
        return RawResult(ext.model_dump_json(), ext, None, 0, 0)


def estimate(cases, model_id: str, in_tok: int = 800, out_tok: int = 150) -> tuple[int, float]:
    calls = sum(c["k"] for c in cases)
    i, o = PRICES.get(model_id, (0.0, 0.0))
    return calls, (calls * in_tok * i + calls * out_tok * o) / 1_000_000


def _is_rate_limit(e: Exception) -> bool:
    msg = str(e)
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg or "RateLimit" in type(e).__name__


def _retry_delay(e: Exception, default: float = 60.0) -> float:
    m = re.search(r"retry in ([\d.]+)s", str(e), re.I) or re.search(r"retryDelay': '(\d+)s", str(e))
    return float(m.group(1)) + 2 if m else default


def _call_with_retry(extractor, utterance, axis, max_wait: float = 600.0):
    waited = 0.0
    while True:
        try:
            return extractor.extract_raw(utterance, axis)
        except BudgetExceeded:
            raise
        except Exception as e:  # noqa: BLE001
            if not _is_rate_limit(e) or waited >= max_wait:
                raise
            d = _retry_delay(e)
            print(f"\n  429 — {d:.0f}s 대기", end="", flush=True)
            time.sleep(d)
            waited += d


def run(extractor, cases, out_path: Path) -> list[dict]:
    """결과 파일이 있으면 이어서 실행한다(같은 case_id/rep는 건너뜀). 호출을 아낀다."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if out_path.exists():
        rows = [json.loads(ln) for ln in out_path.read_text(encoding="utf-8").splitlines() if ln]
        if rows:
            print(f"이어서 실행: {len(rows)}건 저장됨")
    done = {(r["case_id"], r["rep"]) for r in rows}
    with out_path.open("a", encoding="utf-8") as f:
        for c in cases:
            axis = Axis(c["asked_axis"]) if c["asked_axis"] else None
            for rep in range(c["k"]):
                if (c["id"], rep) in done:
                    continue
                try:
                    r = _call_with_retry(extractor, c["utterance"], axis)
                except BudgetExceeded as e:
                    print(f"\n!! {e} — 중단. 지금까지 결과는 저장됨", file=sys.stderr)
                    return rows
                row = {
                    "case_id": c["id"],
                    "rep": rep,
                    "model_id": extractor.model_id,
                    "prompt_version": extractor.prompt_version,
                    "text": r.text,
                    "parse_error": r.error,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)
                print(".", end="", flush=True)
    print()
    return rows


def report(rows: list[dict], cases) -> None:
    lexicon = load_lexicon()
    by_id = {c["id"]: c for c in cases}
    per_case: dict[str, list] = {}
    check_fail = Counter()
    for row in rows:
        c = by_id[row["case_id"]]
        # 저장된 원문을 항상 다시 파싱한다 — 파서를 고치면 재호출 없이 재채점된다
        parsed, err = None, None
        try:
            parsed = _parse_json(row["text"])
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {str(e)[:120]}"
        sc = score(c, row["text"], parsed, err, lexicon)
        per_case.setdefault(c["id"], []).append(sc)
        for k, v in sc.checks.items():
            if not v:
                check_fail[k] += 1

    n_calls = len(rows)
    safety_viol = sum(1 for scs in per_case.values() for s in scs if not s.safety_ok)
    quality_pass = sum(1 for scs in per_case.values() for s in scs if s.passed)
    model = rows[0]["model_id"] if rows else "?"
    ti = sum(r["input_tokens"] for r in rows)
    to = sum(r["output_tokens"] for r in rows)
    i, o = PRICES.get(model, (0.0, 0.0))

    print(f"\n== {model} ==")
    print(f"호출 {n_calls}  입력 {ti} tok  출력 {to} tok  비용 ${(ti * i + to * o) / 1e6:.3f}")
    print(f"안전 위반(D4/D5/D6) 건수: {safety_viol}   ← 0이어야 한다")
    print(f"전체 통과: {quality_pass}/{n_calls}")
    print("검사별 실패:", dict(check_fail) or "없음")
    print("\n케이스별 (k회 중 통과 / 사유):")
    for cid, scs in per_case.items():
        ok = sum(1 for s in scs if s.passed)
        mark = "  " if ok == len(scs) else "X "
        print(f"{mark}{cid} {by_id[cid]['category']:<15} {ok}/{len(scs)}")
        for s in scs:
            for r in s.reasons:
                print(f"       - {r}")


def main() -> None:
    # Windows 콘솔(cp949)에서 한글·기호 출력이 깨지지 않게
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")  # 없으면 조용히 넘어간다
    except ImportError:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["anthropic", "openai", "google"])
    ap.add_argument("--model")
    ap.add_argument(
        "--budget",
        type=float,
        default=float(os.environ.get("MEDIMATE_EVAL_BUDGET_USD", "0.50")),
        help="모델당 지출 상한 USD",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", type=Path, help="저장된 결과를 재채점만")
    ap.add_argument("--yes", action="store_true", help="예상 비용 확인 생략")
    a = ap.parse_args()

    cases = load_cases()

    if a.report:
        rows = [json.loads(ln) for ln in a.report.read_text(encoding="utf-8").splitlines() if ln]
        report(rows, cases)
        return

    if a.dry_run:
        rows = run(DryRunExtractor(cases), cases, RESULTS / "dry-run.jsonl")
        report(rows, cases)
        return

    if not (a.provider and a.model):
        ap.error("--provider 와 --model 필요 (또는 --dry-run / --report)")

    calls, est = estimate(cases, a.model)
    print(f"{a.model}: 호출 {calls}회, 예상 비용 약 ${est:.3f} (상한 ${a.budget})")
    if not a.yes and input("진행? [y/N] ").strip().lower() != "y":
        return
    ex = LLMExtractor(a.provider, a.model, budget_usd=a.budget)
    rows = run(ex, cases, RESULTS / f"{a.model}.jsonl")
    report(rows, cases)
    print(f"\n실제 비용 ${ex.usage.cost_usd(a.model):.3f}")


if __name__ == "__main__":
    main()
