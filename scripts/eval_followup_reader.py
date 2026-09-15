"""재방문 표현 읽기(`followup-v1`) 실측 — 규칙 파서가 모르는 표현을 Nova가 읽는지 본다.

    uv run python scripts/eval_followup_reader.py                 # Nova Pro, 상한 $0.05
    uv run python scripts/eval_followup_reader.py --model apac.amazon.nova-lite-v1:0

읽은 결과는 코드가 검증한다(`memo.followup_from_reader`) — 표현이 문장에 글자 그대로 있어야
하고 날짜는 코드가 센다. 여기서 재는 것은 "LLM이 읽은 일수가 사람 정답과 같은가"다.
원본은 evals/results/followup-<model>.jsonl (gitignore).
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, "src")

from medimate.dialog.memo import followup_date, followup_from_reader  # noqa: E402
from medimate.llm.memo_classifier import LLMFollowUpReader  # noqa: E402

VISIT = date(2026, 9, 15)

# (문장, 앞 문장, 기대 일수 집합 | 기대 날짜 문자열 | None=없어야 함)
CASES: list[tuple[str, str | None, object]] = [
    ("담주에 보자고", None, {7}),
    ("담주 화요일에 오라고 하셨어요", None, {7, 8, 9}),
    ("보름쯤 있다가 다시 오래요", None, {15}),
    ("한 열흘 뒤에 오라고", None, {10}),
    ("열흘 정도 있다가 봅시다", None, {10}),
    ("다다음주에 재방문", None, {14}),
    ("이틀 뒤에 다시 오세요", None, {2}),
    ("사나흘 뒤에 경과 보자고", None, {3, 4}),
    ("한 달 반 뒤에 오라고", None, {45}),
    ("두 달 후 재검", None, {60}),
    ("10월 3일에 오라고", None, "2026-10-03"),
    ("이주뒤 재방문", "일주일치 약처방받고", {14}),
    ("담주에 보자고", "일주일치 약처방받고", {7}),  # 앞 절의 약 기간에 끌리면 안 된다
    ("안 좋아지면 바로 오래요", None, None),
    ("다시 오세요", "3일치 약 먹고", None),  # 현재 문장에 표현이 없다 → 리더는 비워야 한다
]


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    a = argparse.ArgumentParser()
    a.add_argument("--provider", default="bedrock")
    a.add_argument("--model", default="apac.amazon.nova-pro-v1:0")
    a.add_argument("--budget", type=float, default=0.05)
    args = a.parse_args()

    reader = LLMFollowUpReader(args.provider, args.model, budget_usd=args.budget)
    out_path = Path("evals/results") / f"followup-{args.model.replace(':', '_')}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ok = 0
    rows = []
    print(f"모델 {args.model}  기준일 {VISIT}\n")
    for sent, prev, expect in CASES:
        rule = followup_date(sent, VISIT, prev_text=prev)
        got = followup_from_reader(reader, sent, VISIT, prev_text=prev)
        raw = reader.last_text
        if expect is None:
            good = got is None
            shown = "없음" if got is None else f"{got.text!r} → {got.date}"
        elif isinstance(expect, str):
            good = got is not None and got.date == expect
            shown = got.date if got else "없음"
        else:
            days = (date.fromisoformat(got.date) - VISIT).days if got else None
            good = days in expect
            shown = f"{got.text!r} = {days}d" if got else "없음"
        ok += good
        mark = " OK " if good else "FAIL"
        rule_s = f"규칙 {rule.date}" if rule else "규칙 없음"
        if rule and rule.basis.startswith("앞 절"):
            rule_s += "(앞 절)"
        print(f"{mark}  {sent!r:26s} prev={prev!r:18s} LLM {shown:24s} [{rule_s}]")
        rows.append(
            {
                "sentence": sent,
                "prev": prev,
                "expect": sorted(expect) if isinstance(expect, set) else expect,
                "raw": raw,
                "got": got.model_dump() if got else None,
                "rule": rule.model_dump() if rule else None,
                "ok": good,
            }
        )

    u = reader.usage
    cost = u.cost_usd(args.model)
    print(f"\n{ok}/{len(CASES)}   호출 {u.calls}  입력 {u.input_tokens} tok  출력 {u.output_tokens} tok  ${cost:.4f}")
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.write(
            json.dumps(
                {
                    "_summary": True,
                    "model": args.model,
                    "prompt_version": reader.prompt_version,
                    "ok": ok,
                    "n": len(CASES),
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
