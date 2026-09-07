"""진료 후 메모 분류 eval — 문장 라벨 정확도 + 재방문 날짜 + 원문 보존. LLM judge 없음.

  uv run python -m medimate.evals.run_memo --dry-run
  uv run python -m medimate.evals.run_memo --provider local --model local/Qwen3-1.7B-Q4_0 --yes
  uv run python -m medimate.evals.run_memo --report evals/results/memo/<model>.jsonl

채점
- L  문장별 라벨 일치율 (핵심 지표)
- F  follow_up_date 일치 (결정론 계산이라 라벨이 맞으면 따라온다)
- V  카드 값이 메모 문장 원문 그대로인가 (구조상 항상 참이어야 한다 — 깨지면 코드 버그)
- P  파싱·스키마 통과
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path

from medimate.dialog.memo import MemoLabels, classify_memo, split_sentences
from medimate.llm import prompt_memo_small
from medimate.llm.memo_classifier import LLMMemoClassifier
from medimate.llm.providers import PRICES

ROOT = Path(__file__).resolve().parents[3]
CASES = ROOT / "evals" / "postvisit_cases.jsonl"
RESULTS = ROOT / "evals" / "results" / "memo"


def load_cases() -> list[dict]:
    return [json.loads(ln) for ln in CASES.read_text(encoding="utf-8").splitlines() if ln.strip()]


class DryRunClassifier:
    model_id = "dry-run"
    prompt_version = prompt_memo_small.PROMPT_VERSION

    def __init__(self):
        self.last_text = ""
        self.last_tokens = (0, 0)
        self._expect: list[str] = []

    def classify(self, sentences) -> MemoLabels:
        labs = [self._expect[i] if i < len(self._expect) else "none" for i in range(len(sentences))]
        self.last_text = json.dumps({"labels": labs}, ensure_ascii=False)
        return MemoLabels.model_validate({"labels": labs})


def score(case: dict, labels_text: str, card_dump: dict | None, err: str | None) -> dict:
    exp = case["expect"]
    sents = split_sentences(case["memo"])
    out = {"P": err is None, "L": 0.0, "F": None, "V": True, "n": len(sents)}
    if err is not None or card_dump is None:
        return out
    got = card_dump["_labels"]
    if len(sents) != len(exp):
        out["split_mismatch"] = True  # 분리 규칙이 기대와 다르다 — 케이스나 규칙 중 하나를 고친다
    hits = sum(1 for i, e in enumerate(exp) if got.get(str(i), got.get(i, "none")) == e)
    out["L"] = hits / max(len(exp), 1)
    fu = card_dump.get("follow_up_date")
    out["F"] = (fu["date"] if fu else None) == case.get("follow_up_date")
    memo_norm = case["memo"].replace(" ", "")
    for a in card_dump["axes"].values():
        for ev in a["evidence"]:
            if ev.replace(" ", "") not in memo_norm:
                out["V"] = False
    return out


def run(clf, cases: list[dict], out_path: Path) -> list[dict]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if out_path.exists():
        rows = [json.loads(ln) for ln in out_path.read_text(encoding="utf-8").splitlines() if ln]
        print(f"이어서 실행: {len(rows)}건 저장됨")
    done = {r["case_id"] for r in rows}
    with out_path.open("a", encoding="utf-8") as f:
        for c in cases:
            if c["id"] in done:
                continue
            if isinstance(clf, DryRunClassifier):
                clf._expect = c["expect"]
            t0 = time.perf_counter()
            err = None
            card_dump = None
            try:
                res = classify_memo(c["memo"], clf, visit_date=date.fromisoformat(c["visit_date"]))
                card_dump = res.card.model_dump(mode="json")
                card_dump["_labels"] = {str(k): v for k, v in res.labels.items()}
                card_dump["_dropped"] = res.dropped
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {str(e)[:160]}"
            row = {
                "case_id": c["id"],
                "model_id": clf.model_id,
                "prompt_version": clf.prompt_version,
                "text": clf.last_text,
                "card": card_dump,
                "error": err,
                "input_tokens": clf.last_tokens[0],
                "output_tokens": clf.last_tokens[1],
                "latency_s": round(time.perf_counter() - t0, 3),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            print(".", end="", flush=True)
    print()
    return rows


def report(rows: list[dict], cases: list[dict]) -> None:
    by_id = {c["id"]: c for c in cases}
    tot_sent = hit_sent = 0
    f_ok = f_n = 0
    v_bad = p_bad = split_bad = 0
    lat = []
    per_label = Counter()
    per_label_hit = Counter()
    fails = []
    for r in rows:
        c = by_id.get(r["case_id"])
        if not c:
            continue
        s = score(c, r["text"], r["card"], r["error"])
        n = len(c["expect"])
        tot_sent += n
        hit_sent += round(s["L"] * n)
        if s.get("split_mismatch"):
            split_bad += 1
        if not s["P"]:
            p_bad += 1
        if not s["V"]:
            v_bad += 1
        if c.get("follow_up_date") is not None or (r["card"] and r["card"].get("follow_up_date")):
            f_n += 1
            f_ok += int(bool(s["F"]))
        if r.get("latency_s"):
            lat.append(r["latency_s"])
        if r["card"]:
            got = r["card"]["_labels"]
            for i, e in enumerate(c["expect"]):
                per_label[e] += 1
                g = got.get(str(i), "none")
                if g == e:
                    per_label_hit[e] += 1
                else:
                    fails.append(
                        (
                            c["id"],
                            i,
                            e,
                            g,
                            split_sentences(c["memo"])[i]
                            if i < len(split_sentences(c["memo"]))
                            else "?",
                        )
                    )
    model = rows[0]["model_id"] if rows else "?"
    ti = sum(r["input_tokens"] for r in rows)
    to = sum(r["output_tokens"] for r in rows)
    i, o = PRICES.get(model, (0.0, 0.0))
    print(f"\n== {model} ({rows[0]['prompt_version'] if rows else '?'}) ==")
    print(f"메모 {len(rows)}  문장 {tot_sent}  비용 ${(ti * i + to * o) / 1e6:.3f}")
    print(f"문장 라벨 정확도 L: {hit_sent}/{tot_sent} = {hit_sent / max(tot_sent, 1):.0%}")
    print(f"재방문 날짜 F: {f_ok}/{f_n}   원문 보존 V 위반: {v_bad}", end="   ")
    print(f"파싱 실패 P: {p_bad}   분리 불일치: {split_bad}")
    if lat:
        lat = sorted(lat)
        print(f"지연 p50 {lat[len(lat) // 2]:.2f}s  p90 {lat[int(len(lat) * 0.9)]:.2f}s")
    print("라벨별:", {k: f"{per_label_hit[k]}/{per_label[k]}" for k in per_label})
    for cid, idx, e, g, sent in fails[:30]:
        print(f"  X {cid}[{idx}] want={e} got={g} | {sent[:50]}")


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["anthropic", "openai", "google", "local"])
    ap.add_argument("--model")
    ap.add_argument(
        "--budget", type=float, default=float(os.environ.get("MEDIMATE_EVAL_BUDGET_USD", "0.50"))
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", type=Path)
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    cases = load_cases()
    if a.report:
        rows = [json.loads(ln) for ln in a.report.read_text(encoding="utf-8").splitlines() if ln]
        report(rows, cases)
        return
    if a.dry_run:
        rows = run(DryRunClassifier(), cases, RESULTS / "dry-run.jsonl")
        report(rows, cases)
        return
    if not (a.provider and a.model):
        ap.error("--provider 와 --model 필요 (또는 --dry-run / --report)")
    est = len(cases)
    i, o = PRICES.get(a.model, (0.0, 0.0))
    print(f"{a.model}: 호출 {est}회, 예상 비용 약 ${(est * 900 * i + est * 80 * o) / 1e6:.3f}")
    if not a.yes and input("진행? [y/N] ").strip().lower() != "y":
        return
    clf = LLMMemoClassifier(a.provider, a.model, budget_usd=a.budget)
    rows = run(clf, cases, RESULTS / f"{a.model.replace('/', '-')}.jsonl")
    report(rows, cases)
    print(f"\n실제 비용 ${clf.ex.usage.cost_usd(a.model):.3f}")


if __name__ == "__main__":
    main()
