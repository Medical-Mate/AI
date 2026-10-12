"""진료 후 메모 분류 · 온디바이스(로컬 llama-server) — 지금 프롬프트(memo-small-v5)를 폰 짝 벡터 34개로 잰다.

폰 자산은 아직 `memo-small-v4`다(`evals/ondevice/vectors-memo-…-memo-small-v4.jsonl`, PC 123/132 = 93.2%).
v5는 라벨 `lifestyle_instructions`(지침)가 하나 늘었다. 1.7B가 늘어난 라벨을 버티는지 본다.

정답 두 벌
- 새 정답: 사람 정답(`expected_labels`)에 생활 지시 문장만 지침으로 다시 매긴 것(`eval_lifestyle_split.OVERRIDE`, Nova 실측과 같은 표)
- 옛 정답: 사람 정답 그대로. v5 출력의 지침을 약으로 접어(`MEDIMATE_LIFESTYLE_AXIS` 꺼짐과 같다) 비교한다
v4 기준선은 벡터에 저장된 PC 출력(`expected_output`)을 옛 정답에 대 본다(호출 0).

  llama-server -m evals/ondevice/models/Qwen3-1.7B-Q4_0.gguf --port 8080 -c 4096 --temp 0 --seed 42 --reasoning-budget 0
  uv run --no-sync python scripts/eval_memo_ondevice.py --model local/qwen3-1.7b-q4_0-cpu
  uv run --no-sync python scripts/eval_memo_ondevice.py --report evals/results/memo/ondevice-…jsonl   # 재채점, 호출 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
sys.stdout.reconfigure(encoding="utf-8")

from eval_lifestyle_split import OVERRIDE  # noqa: E402

from medimate.llm.memo_classifier import LLMMemoClassifier  # noqa: E402

VEC = Path("evals/ondevice/vectors-memo-qwen3-1.7b-q4_0-memo-small-v4.jsonl")
# 9/15 OVERRIDE는 Nova 실측에 고른 12개 벡터만 다시 매겼다. 나머지 22개에도 생활 지시가 있어서
# memo-small-v5 프롬프트의 정의(약 얘기 없는 "~하라고/~하지 말라고/금지/줄이라고" = 지침,
# 약과 섞인 문장·처치는 약)로 같은 기준을 끝까지 적용한다(2026-10-08). 9/15 표는 Nova 기록이라 건드리지 않는다
OVERRIDE_REST = {
    "식후에 바로 눌지 말고.": "lifestyle_instructions",  # 원문 오타 그대로(눕지)
    "치실 매일 쓰라고 함.": "lifestyle_instructions",
    "긁지 말 것.": "lifestyle_instructions",
    "이어폰 오래 끼지 말라고.": "lifestyle_instructions",
    "탈수 안 되게 물 자주 먹이라고": "lifestyle_instructions",
    "검사 날 렌즈 끼고 오지 말라고.": "lifestyle_instructions",
    "손 높이 들고 있으라고": "lifestyle_instructions",
    "물 하루 2리터 이상.": "lifestyle_instructions",
    "전날 저녁 9시부터 금식.": "lifestyle_instructions",
    "커피랑 매운 거 줄이라고.": "lifestyle_instructions",
}
RELABEL = {**OVERRIDE, **OVERRIDE_REST}
FOLD = {"lifestyle_instructions": "medication_instructions"}


def score(vecs: list[dict], got_by_id: dict[str, list[str]], lat: list[float]) -> None:
    n = new_ok = old_ok = v4_ok = life_n = life_ok = 0
    confusion: Counter = Counter()
    for v in vecs:
        sents, gold4 = v["sentences"], v["expected_labels"]
        gold5 = [RELABEL.get(s, g) for s, g in zip(sents, gold4, strict=True)]
        v4 = json.loads(v["expected_output"])
        got = got_by_id[v["id"]]
        misses = []
        for i, (s, g5, g4, x) in enumerate(zip(sents, gold5, gold4, got, strict=True)):
            n += 1
            new_ok += x == g5
            old_ok += FOLD.get(x, x) == g4
            v4_ok += v4.get(str(i)) == g4
            if s in RELABEL:
                life_n += 1
                life_ok += x == g5
            if x != g5:
                confusion[(g5, x)] += 1
                misses.append((s, g5, x))
        print(("  OK " if not misses else "FAIL") + f" {v['id']}  {len(sents) - len(misses)}/{len(sents)}")
        for s, g5, x in misses:
            print(f"        {s!r}  정답 {g5} · 모델 {x}")
    print(f"\nv5 · 새 정답(지침 포함)        {new_ok}/{n} = {new_ok / n:.1%}")
    print(f"v5 · 지침 접어 옛 정답        {old_ok}/{n} = {old_ok / n:.1%}")
    print(f"v4 기준선(벡터 PC 출력)       {v4_ok}/{n} = {v4_ok / n:.1%}")
    print(f"생활 지시 문장(다시 매긴 것)  {life_ok}/{life_n}")
    print("틀린 자리(정답→모델):", ", ".join(f"{g}→{x} {c}" for (g, x), c in confusion.most_common()))
    if lat:
        lat = sorted(lat)
        print(f"지연 p50 {lat[len(lat) // 2]:.2f}s · 최대 {lat[-1]:.2f}s")


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--provider", default="local")
    a.add_argument("--model", default="local/qwen3-1.7b-q4_0-cpu")
    a.add_argument("--report", type=Path, help="저장된 결과를 재채점만(호출 0)")
    args = a.parse_args()
    vecs = [json.loads(x) for x in VEC.read_text(encoding="utf-8").splitlines() if x.strip()]

    if args.report:
        rows = [json.loads(x) for x in args.report.read_text(encoding="utf-8").splitlines() if x.strip()]
        score(vecs, {r["id"]: r["got"] for r in rows}, [r.get("latency_s", 0) for r in rows if "latency_s" in r])
        return

    clf = LLMMemoClassifier(args.provider, args.model, budget_usd=0.0 if args.provider == "local" else 0.05)
    name = args.model.replace("/", "-").replace(":", "_")
    out = Path("evals/results/memo") / f"ondevice-{clf.prompt_version}-{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"모델 {args.model} · 프롬프트 {clf.prompt_version} · 벡터 {len(vecs)}개\n")
    rows, lat = [], []
    for v in vecs:
        t0 = time.perf_counter()
        got = clf.classify(v["sentences"]).labels
        lat.append(time.perf_counter() - t0)
        rows.append({"id": v["id"], "got": got, "raw": clf.last_text, "latency_s": round(lat[-1], 3)})
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    score(vecs, {r["id"]: r["got"] for r in rows}, lat)
    print("원본", out)


if __name__ == "__main__":
    main()
