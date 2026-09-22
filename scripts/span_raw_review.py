"""날것 메모 시트 검토표 — evals/span_raw_cases.jsonl → evals/span_raw_review.md. 호출 0.

    uv run python scripts/span_raw_review.py            # 표 만들기
    uv run python scripts/span_raw_review.py --import   # md의 gold 열 → jsonl

열: 조각 · 축 · 후보 · rule · Nova(저장 결과가 있으면) · gold. gold를 고치면 --import로 되쓴다.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "src")

from medimate.span.candidates import CandidateGenerator  # noqa: E402
from medimate.span.select import RuleSelector, resolve  # noqa: E402

JEV: Path | None = None
SHEET = Path("evals/span_raw_cases.jsonl")
REVIEW = Path("evals/span_raw_review.md")
NOVA = Path("evals/results/span-apac.amazon.nova-pro-v1_0-E3_raw+E4_abbrev+E5_casual.jsonl")
# --sheet/--out/--nova 로 바꿀 수 있다(처음 보는 메모 시트 등)
_SHORT = {"findings": "소견", "medication_instructions": "약", "tests": "검사", "follow_up": "재방문"}
_ROW = re.compile(r"^\|\s*(\d+)\s*\|(.*)\|\s*$")


def load() -> list[dict]:
    return [json.loads(ln) for ln in SHEET.read_text(encoding="utf-8").splitlines() if ln.strip()]


def build() -> None:
    gen, rule = CandidateGenerator(), RuleSelector()
    nova: dict[tuple[str, int], dict] = {}
    if NOVA.exists():
        for ln in NOVA.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                nova[(r["case_id"], r["idx"])] = r
    jev: dict[tuple[str, int], dict] = {}
    if JEV and JEV.exists():
        for ln in JEV.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                jev[(r["case_id"], r["idx"])] = r
    rows = load()
    with REVIEW.open("w", encoding="utf-8") as f:
        f.write("# 날것 메모 검토표 (E3 구어 · E4 초성 · E5 붙여쓰기)\n\n")
        f.write(
            "메모와 gold는 2026-09-21에 AI가 썼다. **gold 열을 검토한다.** 고치면 그 칸만 바꾸고\n"
            "`uv run python scripts/span_raw_review.py --import`. 값이 없으면 `NONE`.\n"
            "카드 값은 표준형으로 낸다(`담주 화욜` → `다음 주 화요일`).\n\n"
        )
        for c in rows:
            f.write(f"## {c['id']} · {c['group']}\n\n{c['memo']}\n\n_{c.get('note', '')}_\n\n")
            f.write("| # | 조각 | 축 | 후보 | rule | Nova | Jev | gold |\n|---|---|---|---|---|---|---|---|\n")
            for i, s in enumerate(c["segments"]):
                if s.get("skip"):
                    f.write(f"| {i} | {s['text']} | 지침 | (제외) | | | | |\n")
                    continue
                cset = gen.generate(s["text"], s["label"])
                cands = "<br>".join(f"{x.id} {x.text}{' *' if x.derived else ''}" for x in cset.candidates)
                r = resolve(cset, rule.select(cset, s["label"])) or "NONE"
                n = nova.get((c["id"], i))
                nv = (n["choice_text"] or "NONE") if n else "-"
                j = jev.get((c["id"], i))
                jv = (j["choice_text"] or "NONE") if j else "-"
                g = s.get("gold") or ""
                mark = lambda v: v if v.replace(" ", "") == g.replace(" ", "") else f"~~{v}~~"  # noqa: E731
                f.write(
                    f"| {i} | {s['text']} | {_SHORT[s['label']]} | {cands} | {mark(r)} "
                    f"| {mark(nv) if n else '-'} | {mark(jv) if j else '-'} | {g} |\n"
                )
            f.write("\n")
    print(f"→ {REVIEW}")


def import_gold() -> None:
    rows = {c["id"]: c for c in load()}
    cur, n = None, 0
    for line in REVIEW.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            cur = line[3:].split(" · ")[0].strip()
            continue
        m = _ROW.match(line)
        if not m or cur not in rows:
            continue
        cells = [x.strip() for x in m.group(2).split("|")]
        if len(cells) < 6 or cells[-1] in ("", "gold"):
            continue
        seg = rows[cur]["segments"][int(m.group(1))]
        if seg.get("skip"):
            continue
        if seg.get("gold") != cells[-1]:
            seg["gold"] = cells[-1]
            n += 1
    with SHEET.open("w", encoding="utf-8") as f:
        for c in rows.values():
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"gold {n}개 바뀜 → {SHEET}")


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    a = argparse.ArgumentParser()
    a.add_argument("--import", dest="do_import", action="store_true")
    a.add_argument("--sheet", type=Path)
    a.add_argument("--out", type=Path)
    a.add_argument("--nova", type=Path)
    a.add_argument("--jev", type=Path)
    args = a.parse_args()
    if args.sheet:
        SHEET = args.sheet
    if args.out:
        REVIEW = args.out
    if args.nova:
        NOVA = args.nova
    JEV = args.jev
    import_gold() if args.do_import else build()
