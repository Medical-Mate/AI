"""재시도 진단 — Nova `\\u` 이스케이프는 입력이 조금 바뀌면 사라지는 사고인가, 그 입력의 성질인가 (#122).

    uv run python scripts/eval_value_retry.py --dry-run        # 대상 조각 수·비용만
    uv run python scripts/eval_value_retry.py --yes            # Nova 실호출, 저장
    uv run python scripts/eval_value_retry.py --report evals/results/valgen-retry-<model>.jsonl

대상: 저장된 생성 결과에서 응답이 이스케이프·잘림으로 걸러진 조각(generate.screen). 각 조각을 **원래 프롬프트
버전으로** 다시 부르되, 사용자 메시지에 한 줄(RETRY_LINE)만 더한다. temperature 0이라 같은 입력은 같은
폭주를 내므로 입력을 바꿔야 재시도가 뜻이 있다. 운영에서 재시도한다면 이 줄을 그대로 쓴다.
v2 프롬프트는 git 이력(V2_REV)에서 꺼낸다 — 코드의 기본은 v3다.

지표
- 깨끗함   재시도 응답이 screen을 통과
- 재시도 EM  재시도 값(다듬기·검증·폴백 뒤) == gold
- 폴백 EM   지금(재시도 없이) rule 폴백 == gold. 재시도가 이것보다 나아야 넣을 이유가 있다
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import types
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from eval_value_gen import same, segments  # noqa: E402

from medimate.llm.providers import require_price  # noqa: E402
from medimate.span.candidates import CandidateGenerator  # noqa: E402
from medimate.span.generate import MAX_OUTPUT_TOKENS, ValueGenerator, screen  # noqa: E402
from medimate.span.select import RuleSelector  # noqa: E402
from medimate.text.lexicon import load_lexicon  # noqa: E402

RESULTS = Path("evals/results")
MODEL = "apac.amazon.nova-pro-v1:0"
SOURCES = {
    "value-gen-v2": [
        RESULTS / "valgen-apac.amazon.nova-pro-v1_0.jsonl",
        RESULTS / "valgen-apac.amazon.nova-pro-v1_0-unseen_r4.jsonl",
    ],
    "value-gen-v3": [
        RESULTS / "valgen-apac.amazon.nova-pro-v1_0-value-gen-v3-unseen_r1-unseen_r3-unseen_r4-raw.jsonl"
    ],
}
V2_REV = "b1f393d~1"  # v3 커밋 직전 = value-gen-v2
RETRY_LINE = "한글은 글자 그대로 쓴다(\\u 이스케이프 금지)."


def prompt_module(version: str):
    """프롬프트 모듈. v3는 지금 코드, v2는 git 이력에서."""
    if version == "value-gen-v3":
        from medimate.llm import prompt_value_gen as P

        assert P.PROMPT_VERSION == version
        return P
    src = subprocess.run(
        ["git", "show", f"{V2_REV}:src/medimate/llm/prompt_value_gen.py"],
        capture_output=True,
        check=True,
    ).stdout.decode("utf-8")
    m = types.ModuleType("prompt_value_gen_v2")
    exec(compile(src, "prompt_value_gen_v2", "exec"), m.__dict__)  # noqa: S102 — 저장소 자체 이력
    assert m.PROMPT_VERSION == version
    return m


def retry_message(P, segment: str, axis: str, norm: str) -> str:
    base = P.user_message(segment, axis, norm)
    head, _, tail = base.rpartition("\n")  # 마지막 줄 "JSON:" 앞에 넣는다
    return f"{head}\n{RETRY_LINE}\n{tail}"


def offline() -> ValueGenerator:
    g = ValueGenerator.__new__(ValueGenerator)
    g.lexicon, g.gen, g.rule, g.model_id = load_lexicon(), CandidateGenerator(), RuleSelector(), MODEL
    return g


def targets() -> list[dict]:
    gold = {(s["sheet"], s["case_id"], s["idx"]): s["gold"] for s in segments()}
    out = []
    for version, paths in SOURCES.items():
        for p in paths:
            for ln in p.read_text(encoding="utf-8").splitlines():
                if not ln.strip():
                    continue
                r = json.loads(ln)
                bad = screen(r.get("raw_response"), r.get("output_tokens"))
                if bad:
                    key = (r["sheet"], r["case_id"], r["idx"])
                    out.append({**r, "gold": gold.get(key, r["gold"]), "prompt_version": version, "was": bad})
    return out


def em(value, gold: str) -> bool:
    return (value is None) if gold.strip().upper() == "NONE" else same(value, gold)


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    a = argparse.ArgumentParser()
    a.add_argument("--budget", type=float, default=0.5)
    a.add_argument("--dry-run", action="store_true")
    a.add_argument("--yes", action="store_true")
    a.add_argument("--report", type=Path)
    args = a.parse_args()

    gen = offline()
    if args.report:
        rows = [json.loads(ln) for ln in args.report.read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        ts = targets()
        i, o = require_price(MODEL)
        est = (len(ts) * 3500 * i + len(ts) * 40 * o) / 1e6
        print(f"대상 {len(ts)} ({dict(Counter(t['prompt_version'] for t in ts))}) · 호출 {len(ts)} · 예상 ${est:.3f}")
        if args.dry_run or not args.yes:
            print("실행하려면 --yes")
            return
        try:
            from dotenv import load_dotenv

            load_dotenv(".env")
        except ImportError:
            pass
        from medimate.llm.base import parse_json_text
        from medimate.llm.providers import LLMExtractor
        from medimate.obs import tracing
        from medimate.text.chatnorm import normalize

        ex = LLMExtractor("bedrock", MODEL, budget_usd=args.budget)
        mods = {v: prompt_module(v) for v in SOURCES}
        out = RESULTS / f"valgen-retry-{MODEL.replace(':', '_')}.jsonl"
        rows = []
        with out.open("w", encoding="utf-8") as f:
            for t in ts:
                P = mods[t["prompt_version"]]
                norm = normalize(t["segment"]).text
                with tracing.context(
                    trace_name="value-gen-retry",
                    session_id=f"valgen-retry:{MODEL}",
                    tags=["eval", "valgen-retry", t["prompt_version"], t["sheet"]],
                    metadata={k: t[k] for k in ("sheet", "case_id", "idx", "axis", "was")},
                ):
                    text, ti, to = ex.complete_json(
                        P.system_prompt(),
                        retry_message(P, t["segment"], t["axis"], norm),
                        P.SCHEMA,
                        max_tokens=MAX_OUTPUT_TOKENS,
                    )
                try:
                    value = parse_json_text(text).get("value")
                    parsed = True
                except Exception:  # noqa: BLE001
                    value, parsed = None, False
                row = {
                    **{k: t[k] for k in ("sheet", "case_id", "idx", "axis", "segment", "gold", "prompt_version", "was")},
                    "raw_response": text,
                    "input_tokens": ti,
                    "output_tokens": to,
                    "parsed": parsed,
                    "model_value": value,
                }
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(".", end="", flush=True)
        tracing.flush()
        print(f"\n저장 {out} · ${ex.usage.cost_usd(MODEL):.4f}")

    by: dict[str, Counter] = {}
    lines = []
    for r in rows:
        c = by.setdefault(r["prompt_version"], Counter())
        fb = gen.rejected_response("retry-baseline", r["segment"], r["axis"])  # 지금: 폴백
        bad = screen(r["raw_response"], r["output_tokens"])
        if bad:
            res = gen.rejected_response(bad, r["segment"], r["axis"])
        elif not r["parsed"]:
            res = gen.rejected_response("parse", r["segment"], r["axis"])
        else:
            res = gen.check(r["model_value"], r["segment"], r["axis"])
        c["n"] += 1
        c["clean"] += bad is None
        c["retry_em"] += em(res.value, r["gold"])
        c["fallback_em"] += em(fb.value, r["gold"])
        won, was_ok = em(res.value, r["gold"]), em(fb.value, r["gold"])
        c["gained"] += won and not was_ok
        c["lost"] += was_ok and not won
        mark = "+" if won and not was_ok else ("-" if was_ok and not won else " ")
        lines.append(
            f"{mark} {r['prompt_version'][-2:]} {r['sheet']}/{r['case_id']}.{r['idx']} [{r['axis'][:5]}] {r['segment']}\n"
            f"      gold={r['gold']!r} 폴백={fb.value!r} 재시도={res.value!r} ({res.source}{', ' + bad if bad else ''})"
        )

    print("\n| 프롬프트 | 대상 | 재시도 깨끗함 | 폴백 EM(지금) | 재시도 EM | 얻음 | 잃음 |\n|---|---|---|---|---|---|---|")
    tot = Counter()
    for v, c in sorted(by.items()):
        tot.update(c)
        print(
            f"| {v} | {c['n']} | {c['clean']} | {c['fallback_em']} | {c['retry_em']} | +{c['gained']} | -{c['lost']} |"
        )
    print(
        f"| 합계 | {tot['n']} | {tot['clean']} | {tot['fallback_em']} | {tot['retry_em']} | +{tot['gained']} | -{tot['lost']} |"
    )
    print("\n## 조각\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
