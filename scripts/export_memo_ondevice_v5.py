"""폰 짝 자산 memo-small-v5 — 시스템 프롬프트 · 라벨 스키마 · 테스트 벡터 34개를 v4 파일 옆에 새로 쓴다(호출 0).

입력은 `scripts/eval_memo_ondevice.py`가 저장한 1.7B 실행 원본. v4 파일은 그대로 둔다 — 폰은 아직 v4다.
정답(`expected_labels`)은 사람 정답에 생활 지시 문장만 지침으로 다시 매긴 것(`eval_memo_ondevice.RELABEL`).

**엔진 주의**: 이 `expected_output`은 PC winget판 llama.cpp(b10819)로 냈다. v4 벡터를 낸 빌드와도, 폰 엔진
(`pkg-android-2092353`)과도 버전이 다르다. 라벨 대조(사람 정답)는 그대로 쓸 수 있지만, 폰 출력과의 **바이트 대조**는
폰 엔진과 같은 빌드로 다시 내야 한다.

  uv run --no-sync python scripts/export_memo_ondevice_v5.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
sys.stdout.reconfigure(encoding="utf-8")

from eval_memo_ondevice import RELABEL, VEC  # noqa: E402

from medimate.dialog.memo import MemoLabels  # noqa: E402
from medimate.llm import prompt_memo_small  # noqa: E402

OD = Path("evals/ondevice")
RUN = Path("evals/results/memo/ondevice-memo-small-v5-local-qwen3-1.7b-q4_0-cpu.jsonl")
VERSION = prompt_memo_small.PROMPT_VERSION
ENGINE = "llama.cpp b10819 (PC, winget) — 폰 엔진과 다른 빌드"


def main() -> None:
    assert VERSION == "memo-small-v5", VERSION
    system = prompt_memo_small.system_prompt()
    # 폰 자산은 LF — 윈도우 기본(CRLF)으로 쓰면 프롬프트 바이트가 달라진다
    (OD / f"prompt-{VERSION}.system.txt").write_text(system, encoding="utf-8", newline="\n")

    schema = MemoLabels.keyed_schema_for(4)
    schema = {
        "$comment": "예시(N=4). 요청마다 문장 수 N에 맞춰 키 \"0\"..\"N-1\"을 전부 required로 생성한다"
        " (src/medimate/dialog/memo.py MemoLabels.keyed_schema_for). memo-small-v5부터 라벨에"
        " lifestyle_instructions(지침)가 있다 — v4 스키마(memo_labels.schema.json)는 그대로 둔다.",
        **schema,
    }
    (OD / f"memo_labels-{VERSION}.schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    vecs = [json.loads(x) for x in VEC.read_text(encoding="utf-8").splitlines() if x.strip()]
    run = {json.loads(x)["id"]: json.loads(x) for x in RUN.read_text(encoding="utf-8").splitlines() if x.strip()}
    out = OD / f"vectors-memo-qwen3-1.7b-q4_0-{VERSION}.jsonl"
    ok = total = 0
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for v in vecs:
            sents = v["sentences"]
            r = run[v["id"]]
            gold = [RELABEL.get(s, g) for s, g in zip(sents, v["expected_labels"], strict=True)]
            ok += sum(a == b for a, b in zip(r["got"], gold, strict=True))
            total += len(sents)
            f.write(
                json.dumps(
                    {
                        "id": v["id"],
                        "system": system,
                        "user": prompt_memo_small.user_message(sents),
                        "sentences": sents,
                        "settings": {
                            "temperature": 0,
                            "seed": 42,
                            "reasoning": "off",
                            "json_schema": f"memo_labels-{VERSION}.schema.json (N={len(sents)})",
                            "engine": ENGINE,
                        },
                        "expected_output": r["raw"],
                        "expected_labels": gold,
                        "pc_latency_s": r.get("latency_s"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"{out} · 벡터 {len(vecs)} · PC 라벨 {ok}/{total} = {ok / total:.1%}")
    print(f"{OD / f'prompt-{VERSION}.system.txt'} · {len(system)}자")
    print(f"{OD / f'memo_labels-{VERSION}.schema.json'} · 라벨 {schema['properties']['0']['enum']}")


if __name__ == "__main__":
    main()
