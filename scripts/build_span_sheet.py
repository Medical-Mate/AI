"""값 span 정답 시트 만들기 — 기존 조각·라벨에 후보 목록과 rule 제안을 붙여 **사람이 gold를 정할 표**를 낸다.

    uv run python scripts/build_span_sheet.py                 # evals/span_cases.jsonl + evals/span_review.md
    uv run python scripts/build_span_sheet.py --import        # span_review.md의 gold 열을 jsonl에 되쓴다

호출 0. 조각과 라벨은 정답으로 준다(memo-v6가 검증된 층). 여기서 재는 것은 **값**이다.
gold는 사람이 정한다(docs/candidate-selection.md §4.1). 이 스크립트는 후보와 제안만 채운다.

그룹
- E0_normal   기존 벡터 34개(사람 문장). 라벨은 지침 분리표(eval_lifestyle_split.OVERRIDE) 반영
- E0_boundary 배경 메모 §15 경계 케이스
- E1_spacing  E0_boundary의 공백을 지운 것(같은 gold)
- E7_term     의료 전문용어
지침(lifestyle)·none 조각은 시트에 남기되 `skip: true` — 값이 부분 문자열이 아니라 이번 범위 밖.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from eval_lifestyle_split import OVERRIDE  # noqa: E402

from medimate.span.candidates import CandidateGenerator  # noqa: E402
from medimate.span.select import RuleSelector, resolve  # noqa: E402

VEC = Path("evals/ondevice/vectors-memo-qwen3-1.7b-q4_0-memo-small-v4.jsonl")
OUT = Path("evals/span_cases.jsonl")
REVIEW = Path("evals/span_review.md")
SKIP_LABELS = {"lifestyle_instructions", "none"}

EXTRA = [
    {
        "id": "SB01",
        "group": "E0_boundary",
        "memo": "위염이래요. 위산약 2주치 받고, 3주 후에 재방문 하래요",
        "segments": [
            ("위염이래요.", "findings"),
            ("위산약 2주치 받고,", "medication_instructions"),
            ("3주 후에 재방문 하래요", "follow_up"),
        ],
    },
    {
        "id": "SB02",
        "group": "E1_spacing",
        "memo": "위염이래요위산약2주치받고3주후에재방문하래요",
        "segments": [
            ("위염이래요", "findings"),
            ("위산약2주치받고", "medication_instructions"),
            ("3주후에재방문하래요", "follow_up"),
        ],
    },
    {
        "id": "ST01",
        "group": "E7_term",
        "memo": "헬리코박터균 검사를 했대요",
        "segments": [("헬리코박터균 검사를 했대요", "tests")],
    },
    {
        "id": "ST02",
        "group": "E7_term",
        "memo": "역류성식도염이라고 하셨어요",
        "segments": [("역류성식도염이라고 하셨어요", "findings")],
    },
    {
        "id": "ST03",
        "group": "E7_term",
        "memo": "프로톤펌프억제제를 처방받았어요",
        "segments": [("프로톤펌프억제제를 처방받았어요", "medication_instructions")],
    },
]


def cases_from_vectors() -> list[dict]:
    out = []
    for ln in VEC.open(encoding="utf-8"):
        if not ln.strip():
            continue
        v = json.loads(ln)
        segs = [(s, OVERRIDE.get(s, lab)) for s, lab in zip(v["sentences"], v["expected_labels"])]
        out.append(
            {
                "id": v["id"],
                "group": "E0_normal",
                "memo": " ".join(v["sentences"]),
                "segments": segs,
            }
        )
    return out


def build() -> None:
    gen = CandidateGenerator()
    rule = RuleSelector()
    existing: dict[tuple[str, int], str | None] = {}
    if OUT.exists():  # 이미 정한 gold는 지키고 후보·제안만 새로 채운다
        for ln in OUT.open(encoding="utf-8"):
            if ln.strip():
                c = json.loads(ln)
                for i, s in enumerate(c["segments"]):
                    existing[(c["id"], i)] = s.get("gold")

    rows = []
    for case in cases_from_vectors() + EXTRA:
        segs = []
        for i, (text, label) in enumerate(case["segments"]):
            entry: dict = {"text": text, "label": label}
            if label in SKIP_LABELS:
                entry["skip"] = True
            else:
                cset = gen.generate(text, label)
                entry["candidates"] = [
                    {"id": c.id, "text": c.text, "kinds": list(c.kinds), "derived": c.derived}
                    for c in cset.candidates
                ]
                if cset.overflow:
                    entry["overflow"] = cset.overflow
                entry["proposed"] = resolve(cset, rule.select(cset, label))
            entry["gold"] = existing.get((case["id"], i))
            segs.append(entry)
        rows.append(
            {"id": case["id"], "group": case["group"], "memo": case["memo"], "segments": segs}
        )

    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_seg = sum(1 for r in rows for s in r["segments"] if not s.get("skip"))
    n_skip = sum(1 for r in rows for s in r["segments"] if s.get("skip"))
    with REVIEW.open("w", encoding="utf-8") as f:
        f.write("# 값 span 정답 검토표\n\n")
        f.write(
            "`gold` 열을 채운다. **후보를 보지 말고 정답이라 생각하는 값을 그대로** 적는다. 후보에 없어도"
            " 된다(그게 후보 누락 기록이다). 값이 없는 조각은 `NONE`. 후보 ID(`C03`)도 받는다."
            " 비우면 채점에서 빠진다. 다 채우면\n"
            "`uv run python scripts/build_span_sheet.py --import`.\n\n"
        )
        f.write(f"조각 {n_seg}개 (지침·none {n_skip}개는 제외)\n\n")
        for r in rows:
            f.write(f"## {r['id']} · {r['group']}\n\n{r['memo']}\n\n")
            f.write("| # | 조각 | 축 | 후보 | rule 제안 | gold |\n|---|---|---|---|---|---|\n")
            for i, s in enumerate(r["segments"]):
                if s.get("skip"):
                    f.write(f"| {i} | {s['text']} | {s['label']} | (제외) | | |\n")
                    continue
                cands = "<br>".join(
                    f"{c['id']} {c['text']}{' *' if c['derived'] else ''}" for c in s["candidates"]
                )
                f.write(
                    f"| {i} | {s['text']} | {_short(s['label'])} | {cands} | {s['proposed'] or 'NONE'} | {s['gold'] or ''} |\n"
                )
            f.write("\n")
    print(f"조각 {n_seg}개 → {OUT}  검토표 {REVIEW}")


_SHORT = {
    "findings": "소견",
    "medication_instructions": "약",
    "tests": "검사",
    "follow_up": "재방문",
}


def _short(label: str) -> str:
    return _SHORT.get(label, label)


_ROW = re.compile(r"^\|\s*(\d+)\s*\|(.*)\|\s*$")


def import_gold() -> None:
    """검토표의 gold 열 → jsonl. 후보 ID면 그 후보의 text로 바꾼다."""
    rows = {json.loads(ln)["id"]: json.loads(ln) for ln in OUT.open(encoding="utf-8") if ln.strip()}
    cur = None
    n = 0
    for line in REVIEW.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            cur = line[3:].split(" · ")[0].strip()
            continue
        m = _ROW.match(line)
        if not m or cur not in rows:
            continue
        cells = [c.strip() for c in m.group(2).split("|")]
        if len(cells) < 5 or cells[-1] in ("", "gold"):
            continue
        i = int(m.group(1))
        seg = rows[cur]["segments"][i]
        if seg.get("skip"):
            continue
        gold = cells[-1]
        if re.fullmatch(r"C\d{2}", gold):
            by_id = {c["id"]: c["text"] for c in seg["candidates"]}
            gold = by_id.get(gold, gold)
        seg["gold"] = gold
        n += 1
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"gold {n}개 반영 → {OUT}")


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    a = argparse.ArgumentParser()
    a.add_argument("--import", dest="do_import", action="store_true")
    args = a.parse_args()
    import_gold() if args.do_import else build()
