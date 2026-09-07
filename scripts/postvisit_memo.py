"""진료 후 메모 → 4묶음 카드 — 터미널에서 LLM 분류를 직접 돌려보는 스크립트.

  # 로컬(폰과 같은 모델, $0). 먼저 서버를 띄운다:
  #   llama-server -m evals/ondevice/models/Qwen3-1.7B-Q4_0.gguf --port 8080 -c 4096 --temp 0 --seed 42 --reasoning-budget 0
  uv run python scripts/postvisit_memo.py "위염 초기래요. 피검사 했어요. 2주 뒤 오라고."
  uv run python scripts/postvisit_memo.py                 # 인자 없으면 메모를 입력받는다(빈 줄로 끝)
  uv run python scripts/postvisit_memo.py --provider openai --model gpt-5.6-terra "..."   # 서버 폴백, 1회 약 $0.003

실호출은 메모 1건에 LLM 호출 1회. 출력은 문장별 라벨과 4묶음 카드.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medimate.dialog.memo import classify_memo  # noqa: E402
from medimate.llm.memo_classifier import LLMMemoClassifier  # noqa: E402

LABEL_KO = {
    "findings": "들은 소견",
    "tests": "검사",
    "medication_instructions": "약·지시",
    "follow_up": "다시 오기",
}


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("memo", nargs="?")
    ap.add_argument("--provider", default="local", choices=["local", "openai", "anthropic", "google"])
    ap.add_argument("--model", default="local/Qwen3-1.7B-Q4_0")
    ap.add_argument("--visit-date", default=date.today().isoformat())
    ap.add_argument("--json", action="store_true", help="카드 JSON 전체 출력")
    a = ap.parse_args()

    memo = a.memo
    if not memo:
        print("메모를 입력하세요 (빈 줄로 끝):")
        lines = []
        while True:
            ln = sys.stdin.readline()
            if not ln or not ln.strip():
                break
            lines.append(ln.rstrip("\n"))
        memo = "\n".join(lines)
    if not memo.strip():
        sys.exit("메모가 비어 있습니다")

    clf = LLMMemoClassifier(a.provider, a.model)
    res = classify_memo(memo, clf, visit_date=date.fromisoformat(a.visit_date))

    print(f"\n모델 {clf.model_id} · 프롬프트 {clf.prompt_version}")
    print("문장별 라벨:")
    for i, s in enumerate(res.sentences):
        print(f"  {i}: [{LABEL_KO.get(res.labels.get(i, 'none'), '분류 안 됨')}] {s}")
    print("\n카드:")
    for axis, entry in res.card.axes.items():
        name = LABEL_KO.get(axis.value, axis.value)
        print(f"  {name}: {entry.value or '-'}")
    if res.card.unsorted:
        print(f"  분류 안 됨: {' · '.join(res.card.unsorted)}")
    fu = res.card.follow_up_date
    if fu:
        print(f"  재방문 날짜: {fu.date} ({'전후' if fu.approximate else '확정'}, {fu.basis})")
    if res.dropped:
        print(f"  가드가 무시한 라벨: {res.dropped}")
    u = clf.usage
    print(f"\n토큰 {clf.last_tokens[0]}+{clf.last_tokens[1]}  비용 ${u.cost_usd(a.model):.4f}")
    if a.json:
        print(json.dumps(res.card.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
