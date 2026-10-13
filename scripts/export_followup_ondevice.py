"""폰 짝 자산 followup-v1 — 재방문 표현 읽기의 시스템 프롬프트 · 스키마 · 테스트 벡터(호출 0).

폰은 재방문 문장에서 **표현만 잘라** `follow_up_readings`로 보내고, 날짜는 서버가 센다(조립 파서 `text/kodate`).
폰이 센 일수는 서버 규칙이 아무것도 못 읽었을 때만 쓴다(`memo.followup_from_reading`). 그래서 벡터의 판정은
`expected_output` 바이트가 아니라 **`text`가 문장에 글자 그대로 있는가**와 서버가 낸 날짜(`expect_days`)다.

입력은 `scripts/eval_followup_reader.py --provider local --set v1|new|open` 원본(gitignore).

  uv run --no-sync python scripts/export_followup_ondevice.py --model-tag qwen3.5-2b-q4_0-ssmq8 \\
      --runs evals/results/followup-local/qwen3.5-2b-ssmq8-gpu.jsonl \\
             evals/results/followup-local/qwen3.5-2b-ssmq8-gpu-new.jsonl \\
             evals/results/followup-local/qwen3.5-2b-ssmq8-gpu-open.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8")

from datetime import date  # noqa: E402

from medimate.dialog.memo import followup_date, followup_from_reading  # noqa: E402
from medimate.llm import prompt_followup  # noqa: E402

OD = Path("evals/ondevice")
VERSION = prompt_followup.PROMPT_VERSION
VISIT = date(2026, 9, 15)  # eval_followup_reader.VISIT


def server_date(sentence: str, prev: str | None, raw: str) -> str | None:
    """운영 서버 순서(`memo._assemble`)로 낸 날짜 — 규칙 먼저, 못 읽었거나 앞 절이면 폰 읽기."""
    fu = followup_date(sentence, VISIT, prev_text=prev)
    if fu is None or fu.basis.startswith("앞 절"):
        try:
            reading = json.loads(raw) if raw else {}
        except ValueError:
            reading = {}
        read = followup_from_reading(reading, sentence, VISIT, allow_phone_days=fu is None)
        if read is not None:
            fu = read
    return fu.date if fu else None


def date_ok(got: str | None, expect, sentence: str, prev: str | None) -> bool:
    # null = "이 문장에서 읽을 날짜가 없다" — 서버는 앞 절 규칙 날짜(또는 없음)를 그대로 둬야 한다
    if expect is None:
        rule = followup_date(sentence, VISIT, prev_text=prev)
        return got == (rule.date if rule else None)
    if isinstance(expect, str):
        return got == expect
    return got is not None and (date.fromisoformat(got) - VISIT).days in expect


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--model-tag", required=True)
    a.add_argument("--runs", nargs="+", type=Path, required=True)
    a.add_argument("--engine", default="llama.cpp b10819 (PC winget, GPU Vulkan) — 폰 엔진과 다른 빌드·장치")
    args = a.parse_args()

    system = prompt_followup.system_prompt()
    (OD / f"prompt-{VERSION}.system.txt").write_text(system, encoding="utf-8", newline="\n")
    (OD / f"{VERSION}.schema.json").write_text(
        json.dumps(prompt_followup.SCHEMA, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    rows = []
    for run in args.runs:
        set_name = run.stem.rsplit("-", 1)[-1] if run.stem.endswith(("-new", "-open")) else "v1"
        for i, x in enumerate(run.read_text(encoding="utf-8").splitlines()):
            if not x.strip():
                continue
            r = json.loads(x)
            if r.get("_summary"):  # 실행 파일 끝의 요약 줄
                continue
            rows.append(
                {
                    "id": f"FU-{set_name}-{i + 1:02d}",
                    "system": system,
                    "user": prompt_followup.user_message(r["sentence"], r["prev"]),
                    "sentence": r["sentence"],
                    "prev": r["prev"],
                    "settings": {
                        "temperature": 0,
                        "seed": 42,
                        "max_tokens": 1024,
                        "reasoning": "off",
                        "json_schema": f"{VERSION}.schema.json",
                    },
                    "expected_output": r["raw"],
                    # 사람 정답: 진료일(2026-09-15)부터의 일수 후보 · 날짜 문자열 · null(재방문 날짜 없음)
                    "expect_days": r["expect"],
                    "pc_text_in_sentence": bool(
                        (json.loads(r["raw"]) if r["raw"] else {}).get("text", "")
                        and json.loads(r["raw"]).get("text", "") in r["sentence"]
                    ),
                    "pc_server_date": server_date(r["sentence"], r["prev"], r["raw"]),
                    "pc_server_date_ok": date_ok(
                        server_date(r["sentence"], r["prev"], r["raw"]),
                        r["expect"],
                        r["sentence"],
                        r["prev"],
                    ),
                    "engine": args.engine,
                }
            )
    out = OD / f"vectors-followup-{args.model_tag}-{VERSION}.jsonl"
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = sum(r["pc_server_date_ok"] for r in rows)
    print(f"{out} · 벡터 {len(rows)} · PC 서버 날짜 맞음(운영 순서) {ok}/{len(rows)}")


if __name__ == "__main__":
    main()
