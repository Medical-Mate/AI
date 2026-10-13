"""폰 실측 요청용 Qwen3.5-2B 후보 묶음 — 프롬프트 · 스키마 · 테스트 벡터를 1.7B 파일 옆에 새로 쓴다(호출 0).

입력은 PC에서 낸 2B 실행 원본(gitignore):
- 진료 전 추출: `evals/results/local/gpu-qwen3.5-2b-q4_0-small-2b-v1.jsonl` (108문항, 케이스당 첫 반복만)
- 질문 후보: `evals/results/ondevice-chain/chain-qwen3.5-2b-q4_0-small-2b-v1.jsonl` (카드 100장)

**엔진 주의**: `expected_output`은 PC winget판 llama.cpp(b10819) **GPU(Vulkan)**로 냈다. 폰 엔진
(`pkg-android-2092353`, NPU)과 빌드·장치가 다르므로 바이트 일치는 기대하지 않는다. 폰 출력은 AI 쪽이 같은
채점기(가드 포함)로 다시 판정한다. 이번 묶음의 목적은 **NPU에서 도는가 · 속도 · 발열**이다.

  uv run --no-sync python scripts/export_ondevice_2b.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["MEDIMATE_SMALL_PROMPT"] = "2b-v1"  # prompt_small을 import하기 전에
sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8")

from medimate.llm import assist_prompts as ap  # noqa: E402
from medimate.llm import prompt_small as ps  # noqa: E402
from medimate.schema.card import Axis  # noqa: E402

OD = Path("evals/ondevice")
EXTRACT_RUN = Path("evals/results/local/gpu-qwen3.5-2b-q4_0-ssmq8-small-2b-v1.jsonl")
CHAIN_RUN = Path("evals/results/ondevice-chain/chain-qwen3.5-2b-q4_0-ssmq8-small-2b-v1.jsonl")
# 질문 출력은 40자 상한 스키마로 다시 낸 것(같은 카드·같은 메시지, `scripts/ondevice_device_run.py` PC GPU).
# 상한 없는 원본과 85/100이 같고, 나머지는 길이 때문에 갈린 것이다
QUESTIONS_RUN = Path("evals/results/ondevice-chain/questions-2b-ssmq8-maxlen40.jsonl")
# 2026-10-13 폰 실측 뒤: unsloth Q4_0에서 ssm_out(Q5_K, NPU 미지원)만 Q8_0으로 바꾼 파일
# (`scripts/gguf_swap_tensor_type.py`, hyos0415/Qwen3.5-2B-NPU-GGUF@1c069ab6)
MODEL = "Qwen3.5-2B-Q4_0-ssmout-q8.gguf"
ENGINE = "llama.cpp b10819 (PC winget, GPU Vulkan) — 폰 엔진과 다른 빌드·장치"
Q_VERSION = "questions-small4"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    assert ps.PROMPT_VERSION == "extract-small-2b-v1", ps.PROMPT_VERSION
    system = ps.system_prompt()
    # 폰 자산은 LF — 윈도우 기본(CRLF)으로 쓰면 프롬프트 바이트가 달라진다
    (OD / f"prompt-{ps.PROMPT_VERSION}.system.txt").write_text(system, encoding="utf-8", newline="\n")
    # 폰 스키마는 글자 수 상한을 단다(2026-10-13 2차 실측): PC22에서 2B가 "석 달쯤"을 되풀이하다
    # max_tokens 1024에 걸려 JSON이 깨지고 66초가 걸렸다. 40자는 서버 질문 검증기(MAX_CHARS)와 같아서
    # 넘는 문장은 어차피 버려진다. 문법이 문자열을 닫게 만들어 폭주가 생길 수 없다
    phone_schema = json.loads(json.dumps(ap.QUESTIONS_SCHEMA))
    item = phone_schema["properties"]["items"]["items"]["properties"]
    item["text"]["maxLength"] = 40
    item["source"]["maxLength"] = 24
    (OD / "questions.schema.json").write_text(
        json.dumps(phone_schema, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    cases = {
        json.loads(x)["id"]: json.loads(x)
        for x in open("evals/cases.jsonl", encoding="utf-8")
        if x.strip()
    }
    settings = {
        "temperature": 0,
        "seed": 42,
        "max_tokens": 1024,
        "reasoning": "off",
        "json_schema": "turn_extraction.small.schema.json",
    }
    rows = []
    for x in EXTRACT_RUN.read_text(encoding="utf-8").splitlines():
        r = json.loads(x)
        if r["rep"] != 0:
            continue
        c = cases[r["case_id"]]
        asked = Axis(c["asked_axis"]) if c.get("asked_axis") else None
        rows.append(
            {
                "id": f"{r['case_id']}#0",
                # 앱은 첫 발화 없이 축 질문부터 연다(skip_open_ended) — 첫 발화 벡터는 참고용
                "app_path": asked is not None,
                # 이 케이스가 보는 것과 지켜야 할 것(채점기 `evals/cases.jsonl` 그대로). 폰 출력이 PC와
                # 바이트가 달라도 `expect`를 지키면 통과다 — 판정은 AI 쪽 채점기(가드 포함)가 한다
                "category": c["category"],
                "purpose": c.get("from", ""),
                "expect": c["expect"],
                "system": system,
                "user": ps.user_message(c["utterance"], asked),
                "settings": settings,
                "expected_output": r["text"],
                "pc_latency_s": r["latency_s"],
                "engine": ENGINE,
            }
        )
    out = OD / f"vectors-qwen3.5-2b-q4_0-ssmq8-{ps.PROMPT_VERSION}.jsonl"
    _write_jsonl(out, rows)
    print(f"{out} {len(rows)}개 (앱 경로 {sum(r['app_path'] for r in rows)})")

    qsettings = {**settings, "json_schema": "questions.schema.json"}
    qrows = []
    capped = {
        json.loads(x)["id"]: json.loads(x)
        for x in QUESTIONS_RUN.read_text(encoding="utf-8").splitlines()
        if x.strip()
    }
    for x in CHAIN_RUN.read_text(encoding="utf-8").splitlines():
        r = json.loads(x)
        messages = [{"role": "system", "content": ap.questions_system(Q_VERSION)}]
        messages += ap.QUESTIONS_FEWSHOT.get(Q_VERSION, [])
        messages.append({"role": "user", "content": ap.questions_user(r["phone_card"])})
        qrows.append(
            {
                "id": r["id"],
                "messages": messages,
                "settings": qsettings,
                "expected_output": capped[r["id"]]["output"],
                "pc_latency_s": capped[r["id"]]["latency_s"],
                "engine": ENGINE,
            }
        )
    qout = OD / f"vectors-questions-qwen3.5-2b-q4_0-ssmq8-{Q_VERSION}.jsonl"
    _write_jsonl(qout, qrows)
    print(f"{qout} {len(qrows)}개")


if __name__ == "__main__":
    main()
