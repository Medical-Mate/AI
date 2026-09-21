"""파생 그룹 만들기 — 확정된 gold를 물려받아 노이즈 표면만 바꾼 케이스를 기계로 만든다. 라벨 비용 0.

    uv run python scripts/derive_span_groups.py            # gold 있는 E0 조각에서 파생 → span_cases.jsonl에 덧붙임
    uv run python scripts/derive_span_groups.py --dry-run  # 몇 개가 생기는지만

그룹 (docs/candidate-selection.md §4.1)
- E1_spacing  공백 전부 제거. gold 동일(비교가 공백을 무시한다)
- E2_typo     실제 관찰된 오타 표로 치환(괜찮→괜찬, 됐→됬, 래요→레요 …). gold 동일.
              gold 안에 오타가 떨어지면 `gold_touched: true` — 정규화가 되돌려야 맞는 자리라 따로 센다
- E6_marker   조각 끝에 표지(ㅋㅋ · ㅠㅠ · ~~). gold 동일
- E7_term     어휘집 용어를 틀 문장에 넣는다. gold = 용어. 어휘집 타입 → 축

파생 케이스 id는 `<원본id>-E1` 식. 원본 gold가 비어 있으면 만들지 않는다.
다시 돌리면 파생분을 지우고 새로 만든다(원본·손 케이스는 건드리지 않는다).
"""

from __future__ import annotations

import argparse
import io
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, "src")

from medimate.text.lexicon import TYPE_TO_AXIS, load_lexicon  # noqa: E402

SHEET = Path("evals/span_cases.jsonl")
DERIVED = re.compile(r"-E\d$")

# 관찰된 오타. 왼쪽이 원문에 있으면 오른쪽으로. 한 조각에 하나만 넣는다(노이즈를 셀 수 있게)
TYPOS: list[tuple[str, str]] = [
    ("괜찮", "괜찬"),  # PM10·PM15 실데이터
    ("됐", "됬"),
    ("래요", "레요"),
    ("하셨어요", "하셨어여"),
    ("했어요", "햇어요"),
    ("이라고", "이라구"),
    ("다고", "다구"),
    ("었", "엇"),
]
MARKERS = ["ㅋㅋ", "ㅠㅠ", "~~", "ㅎㅎ", "ㄷㄷ"]

# E7 틀 문장. {t}에 용어. 축은 어휘집 타입에서
TEMPLATES = {
    "finding": ["{t}이라고 하셨어요", "{t}래요", "{t} 같다고", "{t} 초기라고 함"],
    "medication": ["{t} 처방받았어요", "{t} 2주치 받고", "{t}는 하루 두 번", "{t} 먹으라고"],
    "test": ["{t} 했대요", "{t}는 다음에 하자고", "{t} 결과는 다음 주에", "{t} 찍었어요"],
    "procedure": ["{t} 받으라고 하셨어요", "오늘은 {t}만 했음", "{t} 6회 처방"],
}
# 틀별 gold. 검토된 gold의 모양을 따른다(2026-09-21): 수식어·용법은 붙이고, 시점은 앞으로,
# 처방은 전부 뗀다(`연고`, `물리치료 6회`), `다음에`는 `추후`
TEMPLATE_GOLD = {
    "{t} 초기라고 함": "{t} 초기",
    "{t} 처방받았어요": "{t}",  # 결정 3(2026-09-21): 처방은 전부 뗀다
    "{t} 2주치 받고": "{t} 2주치",
    "{t}는 하루 두 번": "{t} 하루 두 번",
    "{t}는 다음에 하자고": "추후 {t}",
    "{t} 결과는 다음 주에": "다음 주 {t} 결과 안내",
    "오늘은 {t}만 했음": "NONE",  # 처치만 한 것은 약 칸에 값 없음(스케일링·귀지 검토)
    "{t} 6회 처방": "{t} 6회",
}


def load() -> list[dict]:
    return [json.loads(ln) for ln in SHEET.read_text(encoding="utf-8").splitlines() if ln.strip()]


def save(rows: list[dict]) -> None:
    with SHEET.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _clone(case: dict, suffix: str, group: str, transform) -> dict | None:
    segs = []
    for s in case["segments"]:
        if s.get("skip") or not s.get("gold"):
            return None  # 한 조각이라도 gold가 없으면 이 메모는 파생하지 않는다
        text, touched = transform(s["text"], s["gold"])
        seg = {"text": text, "label": s["label"], "gold": s["gold"], "from": case["id"]}
        if touched:
            seg["gold_touched"] = True
        segs.append(seg)
    return {
        "id": f"{case['id']}{suffix}",
        "group": group,
        "memo": " ".join(x["text"] for x in segs),
        "segments": segs,
    }


def t_spacing(text: str, gold: str):
    return re.sub(r"\s+", "", text), False


def t_typo(text: str, gold: str):
    for a, b in TYPOS:
        if a in text:
            new = text.replace(a, b, 1)
            # gold 안의 글자를 건드렸나 — gold를 공백 무시로 새 텍스트에서 찾을 수 있으면 안 건드린 것
            touched = re.sub(r"\s+", "", gold) not in re.sub(r"\s+", "", new)
            return new, touched
    return text, False  # 오타 넣을 자리가 없으면 그대로(파생은 되지만 노이즈 0)


def t_marker(rng: random.Random):
    def f(text: str, gold: str):
        return text.rstrip() + " " + rng.choice(MARKERS), False

    return f


def derive_terms(rng: random.Random, per_type: int = 6) -> list[dict]:
    lex = load_lexicon()
    out = []
    for type_, tmpls in TEMPLATES.items():
        terms = list(lex.by_type(type_))
        rng.shuffle(terms)
        for k, term in enumerate(terms[:per_type]):
            tmpl = tmpls[k % len(tmpls)]
            text = tmpl.format(t=term.surface)
            gold = TEMPLATE_GOLD.get(tmpl, "{t}").format(t=term.surface)
            out.append(
                {
                    "id": f"ST{type_[:1].upper()}{k + 1:02d}",
                    "group": "E7_term",
                    "memo": text,
                    "segments": [
                        {
                            "text": text,
                            "label": TYPE_TO_AXIS[type_],
                            "gold": gold,
                            "from": "lexicon",
                        }
                    ],
                }
            )
    return out


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    a = argparse.ArgumentParser()
    a.add_argument("--dry-run", action="store_true")
    a.add_argument("--seed", type=int, default=20260921)
    args = a.parse_args()
    rng = random.Random(args.seed)

    # 파생분(-E1 …, STF/STM/STT/STP)은 지우고 새로 만든다. 원본·손 케이스(ST01~03 포함)는 그대로
    rows = [
        r
        for r in load()
        if not DERIVED.search(r["id"]) and not re.fullmatch(r"ST[FMTP]\d{2}", r["id"])
    ]
    base = [r for r in rows if r["group"].startswith("E0")]

    derived: list[dict] = []
    for case in base:
        for suffix, group, tf in (
            ("-E1", "E1_spacing", t_spacing),
            ("-E2", "E2_typo", t_typo),
            ("-E6", "E6_marker", t_marker(rng)),
        ):
            c = _clone(case, suffix, group, tf)
            if c:
                derived.append(c)
    derived += derive_terms(rng)

    n_gold_base = sum(1 for r in base for s in r["segments"] if not s.get("skip") and s.get("gold"))
    print(f"gold 있는 E0 조각 {n_gold_base}개 → 파생 케이스 {len(derived)}개")
    from collections import Counter

    for g, n in sorted(Counter(r["group"] for r in derived).items()):
        segs = sum(len(r["segments"]) for r in derived if r["group"] == g)
        print(f"  {g:12} 케이스 {n:3} · 조각 {segs}")
    touched = sum(1 for r in derived for s in r["segments"] if s.get("gold_touched"))
    print(f"  E2에서 gold를 건드린 조각 {touched}개")
    if args.dry_run:
        return
    save(rows + derived)
    print(f"→ {SHEET}")


if __name__ == "__main__":
    main()
