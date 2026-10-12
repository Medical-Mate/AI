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

from medimate.dialog.memo import (  # noqa: E402
    followup_date,
    followup_from_reader,
    followup_from_reading,
)
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
    ("두 달 후 재검", None, {60, 61}),  # 달력: 9/15 + 2달 = 11/15 = 61일
    ("10월 3일에 오라고", None, "2026-10-03"),
    ("이주뒤 재방문", "일주일치 약처방받고", {14}),
    ("담주에 보자고", "일주일치 약처방받고", {7}),  # 앞 절의 약 기간에 끌리면 안 된다
    ("안 좋아지면 바로 오래요", None, None),
    ("다시 오세요", "3일치 약 먹고", None),  # 현재 문장에 표현이 없다 → 리더는 비워야 한다
]


# 2차 — followup-v1 예시에 없는 표현(2026-10-12). 1차 15문장 중 4개가 예시와 같은 표현이라 폰 1.7B가
# 예시를 베꼈을 수 있어서, 예시·규칙 표에 없는 말로 다시 잰다. `--set new`
NEW_CASES: list[tuple[str, str | None, object]] = [
    ("모레 다시 오라고 하셨어요", None, {2}),
    ("글피에 보자고", None, {3}),
    ("삼 주 뒤에 오래요", None, {21}),
    ("한 달 뒤에 다시 오세요", None, {30}),
    ("석 달 후에 검사 다시 하자고", None, {90, 91}),
    ("엿새 뒤에 실밥 뽑으러 오라고", None, {6}),
    ("닷새 후 재방문", None, {5}),
    ("스무날 뒤쯤 보자고", None, {20}),
    ("일주일 있다가 다시 와보래요", None, {7}),
    ("11월 20일에 재진", None, "2026-11-20"),
    ("증상 심해지면 바로 오라고만 하셨어요", None, None),
    ("약 다 먹고 나서 오래요", "닷새치 약 받고", None),  # 기간은 앞 문장에만 있다 → 비워야 한다
]


# 3차 — 조합 표현(2026-10-12 저녁). 고유어 날짜 낱말은 열거할 수 없는 집합이라 **조합**으로 잰다.
# 조립 파서(text/kodate)와 폰 1.7B를 같은 문장으로 나란히 본다. `--set open`
OPEN_CASES: list[tuple[str, str | None, object]] = [
    ("열하루 뒤에 오라고", None, {11}),
    ("열사흘 후 재방문", None, {13}),
    ("스무이틀 뒤에 보자고", None, {22}),
    ("그글피에 다시 오래요", None, {4}),
    ("내일모레 보자고", None, {2, 3}),
    ("다담주에 오라고", None, {14}),
    ("다다다음 주에 재방문", None, {21}),
    ("담달에 다시 오세요", None, {30}),
    ("다다음 달에 경과 보자고", None, {60, 61}),
    ("달포 뒤에 오라고", None, {30, 31, 35, 40, 45}),
    ("보름 남짓 지나서 오래요", None, {15, 16, 17}),
    ("반년 뒤에 재검", None, {180, 181, 182, 183}),
    ("내년에 다시 보자고", None, {365}),
    ("삼일 뒤에 오라고", None, {3}),
    ("오일 후 재방문", None, {5}),
    ("일주일 반 뒤에 오래요", None, {10, 11}),
    ("두 주 반 뒤에 보자고", None, {17, 18}),
    ("석 달 반 뒤에 재검", None, {105, 106}),
    ("열두 주 뒤에 오라고", None, {84}),
    ("스무 날 뒤에 보자고", None, {20}),
    ("열흘쯤 있다가 오래요", None, {10}),
    ("한 주 정도 있다가 다시 와보래요", None, {7}),
    ("사십일 뒤 재방문", None, {40}),
    ("보름 뒤 아침에 오라고", None, {15}),
    ("모레쯤 다시 오라고", None, {2}),
    ("일 년 뒤에 보자고", None, {365}),
    ("여드레 뒤에 실밥 뽑으러 오라고", None, {8}),
    ("아흐레 후 재방문", None, {9}),
    ("열엿새 뒤 보자고", None, {16}),
    ("스무하루 뒤에 오래요", None, {21}),
    ("다음 주 수요일에 오라고", None, {7, 8, 9}),
    ("닷새 정도 약 먹고 보름 뒤에 오라고", None, {15}),
    ("하루 두 번 먹고 이틀 뒤에 오라고", None, {2}),
    ("한달뒤 재방문", None, {30}),
    ("열흘뒤에보자고", None, {10}),
    ("12월 24일에 오라고", None, "2026-12-24"),
    ("증상 없으면 안 와도 된대요", None, None),
    ("약 떨어지면 오라고", None, None),
    ("다 먹고 오래요", "열흘치 약 받고", None),
]

# 폰 1.7B에게 세는 법을 알려 주는 한 줄(`--hint`). 1.7B는 줄 하나에 흔들리니 변형은 이것 하나만 잰다
HINT = (
    "\n\n고유어 수: 하루1 이틀2 사흘3 나흘4 닷새5 엿새6 이레7 여드레8 아흐레9 열흘10 보름15 스무날20, "
    "열+n=10+n(열사흘=13), 스무+n=20+n(스무이틀=22). 모레2 글피3 그글피4. 다음 주7 다다음 주14 담주7 다담주14. "
    "다음 달30 담달30 달포30 반년180 내년365. 한1 두2 세3 석3 네4 넉4 다섯5 … 열10 스무20. 주=7일, 달=30일, 반=절반."
)


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    a = argparse.ArgumentParser()
    a.add_argument("--provider", default="bedrock")
    a.add_argument("--model", default="apac.amazon.nova-pro-v1:0")
    a.add_argument("--budget", type=float, default=0.05)
    a.add_argument(
        "--set",
        choices=["v1", "new", "open"],
        default="v1",
        help="v1 = 1차 15문장, new = 예시에 없는 12문장, open = 조합 표현 39문장",
    )
    a.add_argument("--hint", action="store_true", help="폰 프롬프트에 고유어 수 힌트 한 줄(변형 측정)")
    args = a.parse_args()
    cases = {"v1": CASES, "new": NEW_CASES, "open": OPEN_CASES}[args.set]
    if args.hint:
        from medimate.llm import prompt_followup as _pf

        _base = _pf.system_prompt()
        _pf.system_prompt = lambda: _base + HINT  # type: ignore[assignment]

    reader = LLMFollowUpReader(args.provider, args.model, budget_usd=args.budget)
    tag = ("" if args.set == "v1" else f"-{args.set}") + ("-hint" if args.hint else "")
    out_path = Path("evals/results") / f"followup-{args.model.replace(':', '_')}{tag}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ok = hy_ok = span_n = 0
    rows = []
    print(f"모델 {args.model}  기준일 {VISIT}  세트 {args.set} ({len(cases)}문장)\n")
    for sent, prev, expect in cases:
        rule = followup_date(sent, VISIT, prev_text=prev)
        got = followup_from_reader(reader, sent, VISIT, prev_text=prev)
        raw = reader.last_text
        # 폰 경로(표현 → 규칙 → 폰 일수)도 같은 응답으로 같이 채점한다 — 호출을 더 하지 않는다
        try:
            reading = json.loads(raw) if raw and raw.strip().startswith("{") else {}
        except ValueError:
            reading = {}
        hy = followup_from_reading(reading, sent, VISIT)
        span = str(reading.get("text") or "").strip()
        span_ok = (span in sent and bool(span)) if expect is not None else (not span or span not in sent)
        span_n += span_ok
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
        if expect is None:
            hy_good = hy is None
        elif isinstance(expect, str):
            hy_good = hy is not None and hy.date == expect
        else:
            hy_good = hy is not None and (date.fromisoformat(hy.date) - VISIT).days in expect
        hy_ok += hy_good
        mark = (" OK " if good else "FAIL") + " / 폰경로 " + ("OK " if hy_good else "FAIL")
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
                "hybrid": hy.model_dump() if hy else None,
                "hybrid_ok": hy_good,
            }
        )

    u = reader.usage
    cost = u.cost_usd(args.model)
    print(f"\nLLM 일수 그대로 {ok}/{len(cases)}   표현 자르기 {span_n}/{len(cases)}   폰 경로(표현→규칙→일수) {hy_ok}/{len(cases)}   호출 {u.calls}  입력 {u.input_tokens} tok  출력 {u.output_tokens} tok  ${cost:.4f}")
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
                    "hybrid_ok": hy_ok,
                    "n": len(cases),
                    "set": args.set,
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
