"""Kiwi 사용자 사전 전후 비교 — 어휘집 용어가 한 토큰으로 남는가, 문장 띄어쓰기·토큰은 어떻게 바뀌나.

    uv run python scripts/kiwi_compare.py              # 표만 출력
    uv run python scripts/kiwi_compare.py --sentences  # 케이스 문장별 전후 토큰까지

호출 0. 결과를 보고 어휘집에 무엇을 더 넣을지, 무엇을 뺄지 정한다(docs/candidate-selection.md §8 ②).
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from medimate.text.lexicon import load_lexicon  # noqa: E402
from medimate.text.tokenize import base_kiwi, build_kiwi, needs_user_dict, tokens  # noqa: E402

VEC = Path("evals/ondevice/vectors-memo-qwen3-1.7b-q4_0-memo-small-v4.jsonl")
CASES = [Path("evals/postvisit_cases.jsonl"), Path("evals/postvisit_cases_v2.jsonl")]

# 배경 메모 §15 우선 테스트 케이스. 시트에 아직 없어 여기 직접 둔다
EXTRA = [
    "위염이래요. 위산약 2주치 받고, 3주 후에 재방문 하래요",
    "위염이래요위산약2주치받고3주후에재방문하래요",
    "헬리코박터균 검사를 했대요",
    "역류성식도염이라고 하셨어요",
    "프로톤펌프억제제를 처방받았어요",
    "ㅁㄹ 기억안남",
]


def corpus() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if VEC.exists():
        for ln in VEC.open(encoding="utf-8"):
            if ln.strip():
                v = json.loads(ln)
                for i, s in enumerate(v["sentences"]):
                    out.append((f"{v['id']}.{i}", s))
    seen = {s for _, s in out}
    for p in CASES:
        if p.exists():
            for ln in p.open(encoding="utf-8"):
                if ln.strip():
                    v = json.loads(ln)
                    if v["memo"] not in seen:
                        out.append((v["id"], v["memo"]))
    out += [(f"X{i + 1}", s) for i, s in enumerate(EXTRA)]
    return out


def fmt(toks) -> str:
    return " ".join(f"{t.form}/{t.tag}{'*' if t.user else ''}" for t in toks)


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    a = argparse.ArgumentParser()
    a.add_argument("--sentences", action="store_true")
    args = a.parse_args()

    lex = load_lexicon()
    base = base_kiwi()
    kiwi, added = build_kiwi(lex)

    print(f"어휘집 {len(lex)}개 · 기본 분석이 쪼개는 것 {len(added)}개 → 사용자 사전에 넣음\n")
    print("| 용어 | 타입 | 기본 | 사용자 사전 후 |")
    print("|---|---|---|---|")
    still_split = 0
    for t in lex.terms:
        if not needs_user_dict(base, t):
            continue
        after = tokens(kiwi, t.compact)
        ok = len(after) == 1
        still_split += 0 if ok else 1
        mark = "" if ok else " ⚠"
        print(f"| {t.surface} | {t.type} | {fmt(tokens(base, t.compact))} | {fmt(after)}{mark} |")
    print(f"\n사용자 사전 후에도 쪼개지는 것: {still_split}개\n")

    # 공백 변형: `역류성 식도염`(공백 있음)도 한 토큰으로 잇는가
    print("| 공백형 | 사용자 사전 후 |")
    print("|---|---|")
    for t in lex.terms:
        if t.surface != t.compact:
            print(f"| {t.surface} | {fmt(tokens(kiwi, t.surface))} |")

    # 문장: 토큰이 바뀐 것만
    changed = 0
    rows = []
    for cid, s in corpus():
        b, k = fmt(tokens(base, s)), fmt(tokens(kiwi, s))
        if b != k:
            changed += 1
            rows.append((cid, s, b, k))
        elif args.sentences:
            rows.append((cid, s, b, None))
    print(f"\n문장 {len(corpus())}개 중 토큰이 바뀐 것 {changed}개\n")
    for cid, s, b, k in rows:
        print(f"## {cid}  {s}")
        print(f"   기본: {b}")
        if k is not None:
            print(f"   후  : {k}")
        sp_b, sp_k = base.space(s), kiwi.space(s)
        if sp_b != s or sp_k != s:
            print(f"   띄어쓰기 기본: {sp_b}")
            if sp_k != sp_b:
                print(f"   띄어쓰기 후  : {sp_k}")


if __name__ == "__main__":
    main()
