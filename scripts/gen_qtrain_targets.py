"""질문 후보 파인튜닝 파일럿 — 학습 카드에 정답 질문(Nova v8)을 붙이고 걸러 학습 파일을 만든다(2026-10-08).

단계
1. 카드 거르기(호출 0): 병명 · 요청 힌트를 그대로 옮긴 값 · 씨앗 값 복사(흔한 답 제외) · 평가 카드 100장과 겹침(3-gram 자카드 ≥ 0.5) ·
   느낌 칸 비움 · 같은 카드 중복
2. 정답 질문: 운영 웹과 같은 Nova Pro + `questions-v8`(이어서 실행 가능 — 이미 붙인 카드는 건너뛴다)
3. 정답 거르기: 채점기 `run_assist.score`(병명 V · 검사·약 T · 추측 G · 중복 D · 형식 F · 예시 X · 어투 반복 R)에 더해
   "왜 그런가요"류 · 묻는 방향 뒤집힘 · 어미 깨짐 · 40자 초과 · 물음표 없음. **하나라도 걸리면 그 카드는 통째로 버린다**
4. 학습 파일: system = `questions-v8-noex`(예시 카드 없음 — 1.7B가 예시를 끌어온 출처를 학습·추론 둘 다에서 뺀다),
   user = 운영과 같은 카드 형식, assistant = 걸러진 질문 JSON

  uv run --no-sync python scripts/gen_qtrain_targets.py --dry-run     # 1단계만, 예상 비용
  uv run --no-sync python scripts/gen_qtrain_targets.py --yes
  uv run --no-sync python scripts/gen_qtrain_targets.py --report      # 저장된 정답으로 3·4단계만(호출 0)
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
sys.stdout.reconfigure(encoding="utf-8")

import gen_qtrain_cards as g  # noqa: E402

from medimate.evals.run_assist import WHY_PATTERN, score  # noqa: E402
from medimate.evals.score import load_lexicon  # noqa: E402
from medimate.llm import assist_prompts as ap  # noqa: E402
from medimate.llm.base import parse_json_text  # noqa: E402
from medimate.llm.providers import LLMExtractor  # noqa: E402

QT = Path("evals/results/qtrain")
CARD_GLOB = "evals/results/qtrain/cards-*-s10[1-5]-*.jsonl"
TARGETS = QT / "targets-nova-v8.jsonl"
TRAIN = QT / "train-v1.jsonl"
TEACHER = "apac.amazon.nova-pro-v1:0"
TEACHER_PROMPT = "questions-v8"
TRAIN_PROMPT = "questions-v8-noex"
PER_CARD_USD = 0.0024  # Nova v8 실측(카드 100장 $0.233)
AX = ["onset", "character", "time_course", "exacerbating", "radiation", "associated"]
GENERIC = {"없음", "없어요", "비슷함", "비슷해요", "안 퍼짐", "모름"}
HINTS = set(g.SITUATIONS) | set(g.COURSES) | set(g.FEELS)
# 1.7B·Nova 실측에서 본 실패(evals/RESULTS.md 2026-10-08 · 7건 씨앗 관찰)
VOICE = re.compile(r"(복용\s*중이에요\?|복용\s*중이시죠|하시나요\?|이시죠\?|습관이 있나요|갔나요\?|드셨나요|있으세요\?|있다는데)")
GRAM = re.compile(r"(아파요는|아파는|프는|져는|된가요|되는가요|는가요\?|풀려는|아프는)")
WHY_MORE = re.compile(r"왜\s*그런\s*걸까요|왜\s*그런\s*거예요")
MAX_LEN = 40  # 백엔드 검증선


MISPLACED: Counter = Counter()


def load_cards() -> list[dict]:
    cards = []
    for f in sorted(glob.glob(CARD_GLOB)):
        cards += [json.loads(x) for x in open(f, encoding="utf-8") if x.strip()]
    for c in cards:
        # gpt-6-luna 7장이 profile을 axes 안에 넣었다. 정답 질문을 만들 때 카드 형식(questions_user)은 axes의
        # 증상 칸만 읽어 그 약 정보를 못 봤다 — 학습 입력도 정답이 본 그대로 두려고 잘못 놓인 값만 뺀다
        for k in [k for k, v in (c.get("axes") or {}).items() if not (v is None or isinstance(v, str))]:
            c["axes"].pop(k)
            MISPLACED[c.get("generator")] += 1
    return cards


def seed_values() -> set[str]:
    vals = set()
    for s in json.loads(g.SEEDS.read_text(encoding="utf-8")):
        for line in s["user"].splitlines():
            if ": " in line and not line.endswith(")"):
                vals.add(line.split(": ", 1)[1].strip())
    return vals - GENERIC


def grams(axes: dict) -> set[str]:
    t = "".join(str(axes.get(k) or "") for k in AX)
    return {t[i : i + 3] for i in range(len(t) - 2)}


def filter_cards(cards: list[dict]) -> tuple[list[dict], Counter]:
    lex = load_lexicon()
    eval_grams = [grams(json.loads(x)["axes"]) for x in open("evals/previsit_cards.jsonl", encoding="utf-8") if x.strip()]
    sv = seed_values()
    why: Counter = Counter()
    kept, seen = [], set()
    for c in cards:
        a = c.get("axes") or {}
        vals = [a.get(k) for k in AX if isinstance(a.get(k), str) and a.get(k).strip()]
        key = json.dumps({k: a.get(k) for k in AX}, ensure_ascii=False, sort_keys=True)
        reason = None
        if not a.get("character"):
            reason = "느낌 비움"
        elif any(t in v for v in vals for t in lex):
            reason = "병명"
        # 값 하나가 같으면 흔한 짧은 답("3일 전" 11장, "뻐근해" 6장, "콕콕")이 걸린다 — 실제 입력처럼 짧은 카드만
        # 골라 버리게 된다(2026-10-08 실측 37장). 한 카드에서 둘 이상 겹칠 때만 베낀 것으로 본다
        elif sum(v in HINTS for v in vals) >= 2:
            reason = "힌트 그대로"
        elif sum(v in sv for v in vals) >= 2:
            reason = "씨앗 값 복사"
        elif key in seen:
            reason = "중복"
        else:
            gr = grams(a)
            if gr and max(len(gr & e) / len(gr | e) for e in eval_grams if e) >= 0.5:
                reason = "평가 카드와 겹침"
        if reason:
            why[reason] += 1
            continue
        seen.add(key)
        kept.append(c)
    return kept, why


def target_reason(card: dict, items: list[dict] | None, lex: list[str]) -> str | None:
    if not items:
        return "질문 없음"
    s = score("questions", card, items, None, lex, TEACHER_PROMPT)
    for k, label in (("V", "병명"), ("T", "검사·약"), ("G", "추측"), ("D", "중복"), ("F", "형식"), ("X", "예시 복사"), ("R", "어투 반복")):
        if s[k]:
            return label
    if not s["P"]:
        return "개수"
    for it in items:
        t = it["text"]
        if WHY_PATTERN.search(t) or WHY_MORE.search(t):
            return "왜 그런가요"
        if VOICE.search(t):
            return "방향 뒤집힘"
        if GRAM.search(t):
            return "어미 깨짐"
        if len(t) > MAX_LEN:
            return "40자 초과"
        if not t.rstrip().endswith("?"):
            return "물음표 없음"
    return None


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--dry-run", action="store_true")
    a.add_argument("--report", action="store_true")
    a.add_argument("--yes", action="store_true")
    a.add_argument("--budget", type=float, default=2.0)
    args = a.parse_args()

    cards = load_cards()
    kept, why = filter_cards(cards)
    gen_in = Counter(c.get("generator") for c in cards)
    gen_kept = Counter(c.get("generator") for c in kept)
    print(f"카드 {len(cards)}장 → 거른 뒤 {len(kept)}장 · 버린 이유 {dict(why)}")
    for gname in sorted(gen_in):
        print(f"   {gname:28} {gen_kept[gname]}/{gen_in[gname]}")

    done = {}
    if TARGETS.exists():
        done = {json.loads(x)["id"]: json.loads(x) for x in TARGETS.read_text(encoding="utf-8").splitlines() if x.strip()}
    todo = [c for c in kept if c["id"] not in done]
    print(f"정답 질문: 이미 {len(done)} · 남은 {len(todo)}장 → 예상 ${len(todo) * PER_CARD_USD:.2f} (상한 ${args.budget}, {TEACHER} · {TEACHER_PROMPT})")
    if args.dry_run:
        return
    if todo and not args.report:
        if not args.yes and input("진행? [y/N] ").strip().lower() != "y":
            return
        ex = LLMExtractor("bedrock", TEACHER, budget_usd=args.budget)
        with TARGETS.open("a", encoding="utf-8", newline="\n") as f:
            for i, c in enumerate(todo, 1):
                try:
                    text, _, _ = ex.complete_json(ap.questions_system(TEACHER_PROMPT), ap.questions_user(c), ap.QUESTIONS_SCHEMA, max_tokens=512)
                    items = parse_json_text(text).get("items")
                    row = {"id": c["id"], "items": items, "raw": text}
                except Exception as e:  # noqa: BLE001
                    row = {"id": c["id"], "items": None, "error": f"{type(e).__name__}: {str(e)[:200]}"}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                done[c["id"]] = row
                if i % 50 == 0:
                    print(f"   {i}/{len(todo)} · ${ex.usage.cost_usd(TEACHER):.3f}")
        print(f"정답 질문 실제 ${ex.usage.cost_usd(TEACHER):.3f}")

    lex = load_lexicon()
    tr_why: Counter = Counter()
    train, by_gen = [], Counter()
    system = ap.questions_system(TRAIN_PROMPT)
    for c in kept:
        row = done.get(c["id"])
        if not row:
            continue
        r = target_reason(c, row.get("items"), lex)
        if r:
            tr_why[r] += 1
            continue
        items = [{"text": it["text"], "source": it["source"]} for it in row["items"]]
        train.append({
            "id": c["id"],
            "generator": c.get("generator"),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": ap.questions_user(c)},
                {"role": "assistant", "content": json.dumps({"items": items}, ensure_ascii=False)},
            ],
        })
        by_gen[c.get("generator")] += 1
    with TRAIN.open("w", encoding="utf-8", newline="\n") as f:
        for t in train:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"정답 거르기: {len(train)}/{sum(1 for c in kept if c['id'] in done)} 통과 · 버린 이유 {dict(tr_why)}")
    print("생성기별 학습 예시:", dict(by_gen))
    print("학습 파일", TRAIN)


if __name__ == "__main__":
    main()
