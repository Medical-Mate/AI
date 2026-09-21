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

from medimate.evals.score import ROOT, load_cases, load_lexicon, safe_model_name, score
from medimate.llm.base import TurnExtraction
from medimate.llm.providers import (
    PRICES,
    BudgetExceeded,
    LLMExtractor,
    RawResult,
    _parse_json,
    require_price,
)
from medimate.schema.card import Axis

RESULTS = ROOT / "evals" / "results"


class DryRunExtractor:
    """호출 없이 러너를 끝까지 돌려보는 가짜. 기대값을 그대로 돌려준다 (전부 통과해야 정상)."""

    model_id = "dry-run"
    prompt_version = "dry"

    def __init__(self, cases):
        self._by_utt = {c["utterance"]: c for c in cases}

    def extract_raw(self, utterance, asked_axis, history=()) -> RawResult:
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
    """실호출 직전 견적. 가격표에 없으면 여기서 멈춘다(require_price)"""
    calls = sum(c["k"] for c in cases)
    i, o = require_price(model_id)
    return calls, (calls * in_tok * i + calls * out_tok * o) / 1_000_000


def _is_rate_limit(e: Exception) -> bool:
    msg = str(e)
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg or "RateLimit" in type(e).__name__


def _retry_delay(e: Exception, default: float = 60.0) -> float:
    m = re.search(r"retry in ([\d.]+)s", str(e), re.I) or re.search(r"retryDelay': '(\d+)s", str(e))
    return float(m.group(1)) + 2 if m else default


def _call_with_retry(extractor, utterance, axis, history=(), max_wait: float = 600.0):
    waited = 0.0
    while True:
        try:
            return extractor.extract_raw(utterance, axis, history)
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
    """결과 파일이 있으면 이어서 실행한다(같은 case_id/rep는 건너뜀). 호출을 아낀다.

    **이어받기 전에 프롬프트 버전을 본다.** 파일 이름은 모델로만 정해지므로, 프롬프트를 바꾸고
    같은 모델로 돌리면 예전 프롬프트의 결과를 그대로 이어받는다 — 호출이 0회 나가고 점수는
    예전 것과 같은데 아무 표시가 없다. 2026-09-14에 v4를 돌렸다고 생각하며 v3 결과를 봤다.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if out_path.exists():
        rows = [json.loads(ln) for ln in out_path.read_text(encoding="utf-8").splitlines() if ln]
        stale = {r.get("prompt_version") for r in rows} - {extractor.prompt_version}
        if stale:
            raise SystemExit(
                f"!! {out_path}에 다른 프롬프트의 결과가 있습니다: "
                f"{', '.join(sorted(map(str, stale)))}\n"
                f"   지금 돌리려는 것은 {extractor.prompt_version}입니다. "
                "섞이면 점수를 읽을 수 없습니다.\n"
                "   --out 으로 다른 파일에 쓰거나, 기존 파일을 옮기세요."
            )
        if rows:
            print(f"이어서 실행: {len(rows)}건 저장됨 ({extractor.prompt_version})")
    done = {(r["case_id"], r["rep"]) for r in rows}
    from medimate.obs import tracing

    lexicon = load_lexicon()
    session = f"extract-eval:{extractor.model_id}:{extractor.prompt_version}"
    with out_path.open("a", encoding="utf-8") as f:
        for c in cases:
            axis = Axis(c["asked_axis"]) if c["asked_axis"] else None
            for rep in range(c["k"]):
                if (c["id"], rep) in done:
                    continue
                try:
                    hist = [tuple(t) for t in c.get("history", [])]
                    t0 = time.perf_counter()
                    with tracing.context(
                        trace_name="extract-eval",
                        session_id=session,
                        tags=["eval", c["category"]],
                        metadata={
                            "case_id": c["id"],
                            "rep": rep,
                            "category": c["category"],
                            "asked_axis": c["asked_axis"],
                            "prompt_version": extractor.prompt_version,
                        },
                        version=extractor.prompt_version,
                    ):
                        r = _call_with_retry(extractor, c["utterance"], axis, hist)
                        # 채점을 trace 점수로 — 대시보드에서 프롬프트 버전·카테고리별 통과율
                        sc = score(c, r.text, r.parsed, r.error, lexicon)
                        tracing.score("passed", 1.0 if sc.passed else 0.0)
                        tracing.score("safety_ok", 1.0 if sc.safety_ok else 0.0)
                        for k, v in sc.checks.items():
                            if not v:
                                tracing.score(f"fail:{k}", 1.0, comment="; ".join(sc.reasons)[:500])
                    latency = round(time.perf_counter() - t0, 3)
                except BudgetExceeded as e:
                    print(f"\n!! {e} — 중단. 지금까지 결과는 저장됨", file=sys.stderr)
                    tracing.flush()
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
                    "latency_s": latency,  # 호출 왕복 시간. 온디바이스 후보 속도 비교용
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)
                print(".", end="", flush=True)
    print()
    tracing.flush()
    return rows


def report(rows: list[dict], cases, guard=None, normalize_chat: bool = False) -> None:
    lexicon = load_lexicon()
    by_id = {c["id"]: c for c in cases}
    per_case: dict[str, list] = {}
    check_fail = Counter()
    for row in rows:
        c = by_id.get(row["case_id"])
        if c is None:
            continue  # --limit 으로 좁힌 집합 밖의 행은 건너뛴다
        # 저장된 원문을 항상 다시 파싱한다 — 파서를 고치면 재호출 없이 재채점된다
        parsed, err = None, None
        try:
            parsed = _parse_json(row["text"])
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {str(e)[:120]}"
        if guard is not None and parsed is not None:
            # 런타임 가드를 거친 결과로 채점한다 — 온디바이스 프로필에서 카드에 실제로 들어가는 것
            from medimate.dialog.guard import guard_extraction

            axis = Axis(c["asked_axis"]) if c["asked_axis"] else None
            normalized = None
            if normalize_chat:
                from medimate.text.chatnorm import normalize

                n = normalize(c["utterance"])
                normalized = n.text if n.changed else None
            parsed = guard_extraction(
                parsed, c["utterance"], axis, guard, normalized=normalized
            ).extraction
        sc = score(c, row["text"], parsed, err, lexicon)
        per_case.setdefault(c["id"], []).append(sc)
        for k, v in sc.checks.items():
            if not v:
                check_fail[k] += 1

    n_calls = sum(len(v) for v in per_case.values())  # 채점된 행만 (--limit 밖 행 제외)
    safety_viol = sum(1 for scs in per_case.values() for s in scs if not s.safety_ok)
    quality_pass = sum(1 for scs in per_case.values() for s in scs if s.passed)
    model = rows[0]["model_id"] if rows else "?"
    scored = [r for r in rows if r["case_id"] in by_id]
    ti = sum(r["input_tokens"] for r in scored)
    to = sum(r["output_tokens"] for r in scored)
    i, o = PRICES.get(model, (0.0, 0.0))

    print(f"\n== {model} ==")
    print(f"호출 {n_calls}  입력 {ti} tok  출력 {to} tok  비용 ${(ti * i + to * o) / 1e6:.3f}")
    lat = sorted(r["latency_s"] for r in scored if r.get("latency_s") is not None)
    if lat:
        p50, p90 = lat[len(lat) // 2], lat[int(len(lat) * 0.9)]
        per_call = to / max(n_calls, 1)
        print(f"지연 p50 {p50:.2f}s  p90 {p90:.2f}s  최대 {lat[-1]:.2f}s", end="  ")
        print(f"(출력 {per_call:.0f} tok/호출)")
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
    ap.add_argument("--provider", choices=["anthropic", "openai", "google", "local", "bedrock"])
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
    ap.add_argument(
        "--prompt",
        choices=["auto", "v3", "small", "v4-nova", "v5-nova"],
        default="auto",
        help="프롬프트 계열. auto는 모델 id로 고른다(providers.prompt_family_for)",
    )
    ap.add_argument("--limit", type=int, help="앞에서 N케이스만 (기기 실측처럼 속도만 볼 때)")
    ap.add_argument("--out", type=Path, help="결과 파일 경로. 기본은 evals/results/<model>.jsonl")
    ap.add_argument(
        "--cases",
        help="케이스 id만 골라 돌린다(쉼표 구분: J02,A01). 실패 몇 건만 다시 볼 때 — 전체를 "
        "돌리면 한 번에 $0.16이 나간다",
    )
    ap.add_argument(
        "--guard", choices=["server", "ondevice"], help="재채점 시 런타임 가드 적용(호출 0)"
    )
    ap.add_argument(
        "--normalize",
        action="store_true",
        help="가드에 채팅 표기 복원문을 준다(MEDIMATE_CHAT_NORMALIZE=on과 같음, #117). --guard와",
    )
    ap.add_argument("--category", help="이 카테고리만(쉼표 구분). 예: chat_abbrev,chat_casual")
    a = ap.parse_args()

    cases = load_cases()
    if a.category:
        cats = {x.strip() for x in a.category.split(",") if x.strip()}
        cases = [c for c in cases if c["category"] in cats]
    if a.cases:
        want = [x.strip() for x in a.cases.split(",") if x.strip()]
        by_id = {c["id"]: c for c in cases}
        missing = [x for x in want if x not in by_id]
        if missing:
            ap.error(f"없는 케이스 id: {', '.join(missing)}")
        cases = [by_id[x] for x in want]
    if a.limit:
        cases = cases[: a.limit]

    if a.report:
        rows = [json.loads(ln) for ln in a.report.read_text(encoding="utf-8").splitlines() if ln]
        guard = None
        if a.guard:
            from medimate.dialog.guard import GuardConfig

            guard = GuardConfig.ondevice() if a.guard == "ondevice" else GuardConfig()
        report(rows, cases, guard, normalize_chat=a.normalize)
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
    ex = LLMExtractor(a.provider, a.model, budget_usd=a.budget, prompt_family=a.prompt)
    out = a.out or RESULTS / f"{safe_model_name(a.model)}.jsonl"
    print(f"결과 파일: {out}  (프롬프트 {ex.prompt_version})")
    rows = run(ex, cases, out)
    report(rows, cases)
    print(f"\n실제 비용 ${ex.usage.cost_usd(a.model):.3f}")


if __name__ == "__main__":
    main()
