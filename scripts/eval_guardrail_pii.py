"""Bedrock Guardrails 민감정보(PII) 가림 vs 우리 정규식 가림 — 본문 수집 전 가리기 후보(2026-09-28).

무엇을 재나
- PII 세트(`evals/pii_cases.jsonl`): `pii`에 적은 문자열이 가림 결과에서 사라졌나(재현), `keep`이 남았나(과잉 가림)
- 정상 86(`cases.jsonl` 76 + `guardrail_benign_hard.jsonl` 10): 가림 결과가 원문과 달라진 것 = 과잉 가림
- 지연, 비용

Guardrails: NAME·PHONE·ADDRESS·AGE·EMAIL ANONYMIZE + 주민번호 정규식. 비교 기준은 호출 0인 정규식(전화·주민번호·이메일).
원본 응답은 evals/results/에 저장, 채점은 --report로 다시 한다.

  uv run --no-sync python scripts/eval_guardrail_pii.py --setup
  uv run --no-sync python scripts/eval_guardrail_pii.py --run --repeat 2
  uv run --no-sync python scripts/eval_guardrail_pii.py --report evals/results/guardrail-pii.jsonl
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import boto3

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
REGION = "ap-northeast-2"
NAME = "medimate-pii-eval"
PROFILE = "apac.guardrail.v1:0"
PRICE_PER_1K_UNITS = 0.10  # 민감정보 필터
BUDGET_USD = 0.50
OUT = ROOT / "evals" / "results" / "guardrail-pii.jsonl"
RRN = r"\d{6}-?[1-4]\d{6}"
TYPES = ["NAME", "PHONE", "ADDRESS", "AGE", "EMAIL"]

# 비교 기준 — 우리가 호출 없이 할 수 있는 것
_REGEX = [
    ("RRN", re.compile(RRN)),
    ("PHONE", re.compile(r"(?:\+82[\s-]?)?0\d{1,2}[\s-]?\d{3,4}[\s-]?\d{4}")),
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
]


def regex_mask(text: str) -> str:
    for name, rx in _REGEX:
        text = rx.sub("{" + name + "}", text)
    return text


def load_set() -> list[dict]:
    def jl(p):
        return [json.loads(x) for x in (ROOT / p).read_text(encoding="utf-8").splitlines() if x]

    rows = [{**c, "set": "pii"} for c in jl("evals/pii_cases.jsonl")]
    for c in jl("evals/cases.jsonl"):
        if c["category"] != "injection":
            rows.append({"id": c["id"], "utterance": c["utterance"], "kind": c["category"],
                         "pii": [], "keep": [], "set": "benign"})
    for c in jl("evals/guardrail_benign_hard.jsonl"):
        rows.append({"id": c["id"], "utterance": c["utterance"], "kind": "benign_hard",
                     "pii": [], "keep": [], "set": "benign"})
    return rows


def find(b):
    return next((g for g in b.list_guardrails()["guardrails"] if g["name"] == NAME), None)


def setup() -> None:
    b = boto3.client("bedrock", region_name=REGION)
    if g := find(b):
        print("이미 있음:", g["id"], g["status"])
        return
    r = b.create_guardrail(
        name=NAME,
        description="PII 가림 eval 전용. 운영에 붙이지 않는다(2026-09-28)",
        blockedInputMessaging="[blocked]",
        blockedOutputsMessaging="[blocked]",
        sensitiveInformationPolicyConfig={
            # **입력 쪽을 명시해야 한다.** `action`만 주면 READY로 떠도 아무것도 안 가린다
            # (2026-09-28 첫 실행: 영어 이메일까지 0건, 과금 단위 0)
            "piiEntitiesConfig": [{"type": t, "action": "ANONYMIZE", "inputEnabled": True,
                                   "inputAction": "ANONYMIZE", "outputEnabled": False}
                                  for t in TYPES],
            "regexesConfig": [{"name": "KR_RRN", "pattern": RRN, "action": "ANONYMIZE",
                               "inputEnabled": True, "inputAction": "ANONYMIZE",
                               "outputEnabled": False, "description": "주민등록번호"}],
        },
        crossRegionConfig={"guardrailProfileIdentifier": PROFILE},
    )
    print("생성:", r["guardrailId"], r["version"])


def run(repeat: int, yes: bool) -> None:
    rows = load_set()
    units = sum(-(-len(r["utterance"]) // 1000) for r in rows) * repeat
    cost = units / 1000 * PRICE_PER_1K_UNITS
    print(f"발화 {len(rows)} × {repeat}회 = {units}단위, 예상 ${cost:.3f} (상한 ${BUDGET_USD})")
    if cost > BUDGET_USD:
        raise SystemExit("상한 초과 — 멈춘다")
    if not yes and input("진행? [y/N] ").strip().lower() != "y":
        return
    g = find(boto3.client("bedrock", region_name=REGION))
    if not g:
        raise SystemExit("가드레일 없음 — --setup 먼저")
    if OUT.exists():
        raise SystemExit(f"{OUT} 이미 있음 — 옮기고 다시")
    rt = boto3.client("bedrock-runtime", region_name=REGION)
    with OUT.open("w", encoding="utf-8") as f:
        for rep in range(repeat):
            for r in rows:
                t0 = time.perf_counter()
                resp = rt.apply_guardrail(guardrailIdentifier=g["id"], guardrailVersion="DRAFT",
                                          source="INPUT",
                                          content=[{"text": {"text": r["utterance"]}}])
                ms = (time.perf_counter() - t0) * 1000
                resp.pop("ResponseMetadata", None)
                f.write(json.dumps({**r, "rep": rep, "client_ms": round(ms, 1), "response": resp},
                                   ensure_ascii=False, default=str) + "\n")
    print("저장:", OUT)
    report(OUT)


def masked_text(r: dict) -> str:
    outs = r["response"].get("outputs") or []
    return outs[0]["text"] if outs else r["utterance"]  # 개입 없음 → 원문 그대로


def judge(text_out: str, r: dict) -> tuple[list, list]:
    missed = [p for p in r["pii"] if p in text_out]
    lost = [k for k in r["keep"] if k not in text_out]
    return missed, lost


def report(path: Path) -> None:
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
    reps = sorted({r["rep"] for r in rows})
    for label, fn in [("Guardrails", masked_text), ("정규식", lambda r: regex_mask(r["utterance"]))]:
        print(f"\n===== {label} =====")
        for rep in (reps if label == "Guardrails" else reps[:1]):
            rr = [r for r in rows if r["rep"] == rep]
            pii_rows = [r for r in rr if r["set"] == "pii"]
            n_pii = sum(len(r["pii"]) for r in pii_rows)
            caught = n_pii - sum(len(judge(fn(r), r)[0]) for r in pii_rows)
            n_keep = sum(len(r["keep"]) for r in pii_rows)
            lost = sum(len(judge(fn(r), r)[1]) for r in pii_rows)
            ben = [r for r in rr if r["set"] == "benign"]
            over = [r for r in ben if fn(r) != r["utterance"]]
            print(f"  rep{rep}: PII 가림 {caught}/{n_pii} · 증상 보존 {n_keep - lost}/{n_keep} · "
                  f"정상 발화 과잉 가림 {len(over)}/{len(ben)}")
        rr = [r for r in rows if r["rep"] == reps[0]]
        by_kind = {}
        for r in rr:
            if r["set"] != "pii":
                continue
            m, lo = judge(fn(r), r)
            k = by_kind.setdefault(r["kind"], [0, 0])
            k[0] += len(r["pii"]) - len(m)
            k[1] += len(r["pii"])
        print("  유형별:", " · ".join(f"{k} {a}/{b}" for k, (a, b) in by_kind.items()))
        print("  사례(rep0):")
        for r in rr:
            out = fn(r)
            m, lo = judge(out, r)
            if r["set"] == "pii" or out != r["utterance"]:
                flag = ("놓침 " + ",".join(m) if m else "") + (" 잃음 " + ",".join(lo) if lo else "")
                if m or lo or (r["set"] == "benign" and out != r["utterance"]) or label == "Guardrails":
                    print(f"    {r['id']:5} {out}   {flag}")
    lat = sorted(r["client_ms"] for r in rows)
    print(f"\n지연 — 왕복 p50 {lat[len(lat) // 2]:.0f}ms · p95 {lat[int(len(lat) * .95)]:.0f}ms")
    units = sum((r["response"].get("usage") or {}).get("sensitiveInformationPolicyUnits", 0)
                for r in rows)
    print(f"과금 단위(민감정보) {units} → ${units / 1000 * PRICE_PER_1K_UNITS:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--report", type=Path)
    a = ap.parse_args()
    if a.setup:
        setup()
    elif a.run:
        run(a.repeat, a.yes)
    elif a.report:
        report(a.report)
    else:
        ap.print_help()
