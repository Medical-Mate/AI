"""맞춤법 교정(Nova 생성 + 가드) 시험 — 오타 문장은 고쳐지나, 정상 문장은 건드리지 않나 (#122).

    uv run python scripts/eval_spell.py --dry-run                 # 문장 수·비용만
    uv run python scripts/eval_spell.py --yes                      # Nova 실호출, 결과 저장
    uv run python scripts/eval_spell.py --report evals/results/spell-<model>.jsonl   # 재채점(호출 0)

문장
- 오타: span_raw_cases + span_unseen_cases 중 `chatnorm`이 바꾸는 조각(관찰된 오타·초성·구어) + E2 파생 조각 일부
- 정상: span_cases E0_normal 조각. **여기서 바뀌면 감점**(띄어쓰기만 바뀐 것은 따로 센다)

지표: 오타 문장 중 가드 통과·채택 수 / 정상 문장 중 바뀐 수(띄어쓰기 제외) / 가드 거절 이유 분포 / 비용.
정답 교정문은 두지 않는다 — 채택된 교정문을 표로 보이고 사람이 훑는다. 그게 이 단계의 목적이다.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")

from medimate.llm.providers import require_price  # noqa: E402
from medimate.text.chatnorm import normalize  # noqa: E402
from medimate.text.lexicon import load_lexicon  # noqa: E402
from medimate.text.spell import PROMPT_VERSION, LLMSpeller, guard  # noqa: E402

RESULTS = Path("evals/results")
_WS = re.compile(r"\s+")


def sentences() -> list[dict]:
    out: list[dict] = []
    for path, kind in (
        ("evals/span_raw_cases.jsonl", "typo"),
        ("evals/span_unseen_cases.jsonl", "typo"),
        ("evals/span_cases.jsonl", "clean"),
    ):
        for ln in Path(path).read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            c = json.loads(ln)
            if kind == "clean" and c["group"] != "E0_normal":
                continue
            for i, s in enumerate(c["segments"]):
                t = s["text"]
                if kind == "typo" and not normalize(t).changed and not re.search(r"[ㄱ-ㅎ]{2}", t):
                    continue  # 오타·초성·구어 표기가 없는 조각은 오타 집합에서 뺀다
                out.append({"id": f"{c['id']}.{i}", "kind": kind, "text": t})
    return out


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    a = argparse.ArgumentParser()
    a.add_argument("--provider", default="bedrock")
    a.add_argument("--model", default="apac.amazon.nova-pro-v1:0")
    a.add_argument("--budget", type=float, default=0.40)
    a.add_argument("--dry-run", action="store_true")
    a.add_argument("--yes", action="store_true")
    a.add_argument("--report", type=Path)
    args = a.parse_args()

    sents = sentences()
    n_typo = sum(1 for s in sents if s["kind"] == "typo")
    n_clean = sum(1 for s in sents if s["kind"] == "clean")
    if args.report:
        rows = [json.loads(ln) for ln in args.report.read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        i, o = require_price(args.model)
        est = (len(sents) * 700 * i + len(sents) * 60 * o) / 1e6
        print(f"오타 문장 {n_typo} · 정상 문장 {n_clean} · 호출 {len(sents)} · 예상 ${est:.3f} (상한 ${args.budget})")
        if args.dry_run or not args.yes:
            print("실행하려면 --yes")
            return
        try:
            from dotenv import load_dotenv

            load_dotenv(".env")
        except ImportError:
            pass
        from medimate.obs import tracing

        protected = [t.surface for t in load_lexicon().terms]
        sp = LLMSpeller(args.provider, args.model, budget_usd=args.budget, protected=protected)
        rows = []
        out = RESULTS / f"spell-{args.model.replace(':', '_')}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for s in sents:
                with tracing.context(
                    trace_name="spell-eval", session_id=f"spell-eval:{args.model}:{PROMPT_VERSION}",
                    tags=["eval", "spell", s["kind"]], metadata={"id": s["id"], "kind": s["kind"]},
                ):
                    r = sp.correct(s["text"])
                    tracing.score("accepted", 1.0 if r.accepted else 0.0, comment=r.reason)
                row = {**s, "model_text": r.model_text, "accepted": r.accepted, "reason": r.reason,
                       "corrected": r.text, "ops": r.ops, **sp.last, "prompt_version": PROMPT_VERSION}
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(".", end="", flush=True)
        tracing.flush()
        print(f"\n저장 {out} · ${sp.usage.cost_usd(args.model):.3f}")

    # 재채점(가드는 코드라 저장된 model_text에 다시 적용한다)
    protected = [t.surface for t in load_lexicon().terms]
    by_kind = {"typo": Counter(), "clean": Counter()}
    shown: list[str] = []
    for r in rows:
        g = guard(r["text"], r.get("model_text") or "", protected)
        k = r["kind"]
        by_kind[k]["n"] += 1
        if g.accepted:
            spacing_only = _WS.sub("", g.text) == _WS.sub("", r["text"])
            by_kind[k]["spacing_only" if spacing_only else "changed"] += 1
            if not spacing_only:
                shown.append(f"  [{k}] {r['text']}  →  {g.text}")
        else:
            by_kind[k][f"reject:{g.reason.split(':')[0]}"] += 1
    print("\n| 집합 | n | 표기 바뀜(채택) | 띄어쓰기만 | 거절(이유별) |\n|---|---|---|---|---|")
    for k in ("typo", "clean"):
        c = by_kind[k]
        rej = ", ".join(f"{x[7:]} {v}" for x, v in c.items() if x.startswith("reject:")) or "-"
        print(f"| {k} | {c['n']} | {c['changed']} | {c['spacing_only']} | {rej} |")
    print("\n채택된 교정(띄어쓰기만 바뀐 것 제외):")
    print("\n".join(shown))


if __name__ == "__main__":
    main()
