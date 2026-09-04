"""진료 전 카드 — 터미널에서 직접 시나리오를 돌려보는 대화형 클라이언트.

서버를 따로 띄우지 않고 앱을 프로세스 안에서 부른다. 백엔드가 하는 일(state 보관·왕복)을
이 스크립트가 대신한다. 턴마다 실제 LLM을 부른다 — 턴당 약 $0.004 (Terra).

  uv run python scripts/previsit_chat.py            # 기본 모델(.env 또는 gpt-5.6-terra)
  uv run python scripts/previsit_chat.py --verbose  # 턴마다 추출 결과·비용 표시
  uv run python scripts/previsit_chat.py --site 허리  # 인체도에서 부위를 먼저 짚은 상황

입력 중 명령:  /card 현재 카드   /state 상태 JSON   /quit 종료(카드 저장)
끝나면 evals/results/chat-<시각>.json 에 전체 기록을 남긴다 (gitignore).
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from medimate.api.app import app  # noqa: E402


def show_card(card: dict) -> None:
    print(f"\n  주 호소: {card['chief_complaint']}")
    for axis, e in card["axes"].items():
        mark = {"filled": "●", "unknown": "?", "skipped": "－", "ambiguous": "~", "not_asked": "○"}[
            e["status"]
        ]
        val = e["value"] or ""
        ev = f"  ← {e['evidence']}" if e["evidence"] else ""
        print(f"  {mark} {axis:<24} {val}{ev}")
    print(f"  완성도 {card['completeness']:.0%}  최소 완성 {card['minimally_complete']}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--site", help="인체도에서 미리 짚은 부위 (예: 허리). SITE 축 사전 채움")
    a = ap.parse_args()

    c = TestClient(app)
    r = c.post("/v1/previsit/sessions", json={"site_label": a.site} if a.site else None)
    r.raise_for_status()
    state = r.json()["state"]
    print(f"[모델 {state['card']['provenance']['model_id']}] /card /state /quit")
    print(f"\nAI: {r.json()['reply']}")

    total = 0.0
    transcript: list[dict] = []
    last = None
    while True:
        try:
            u = input("나: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if u == "/quit":
            break
        if u == "/card":
            if last:
                show_card(last["card"])
            else:
                print("  아직 턴이 없어요")
            continue
        if u == "/state":
            print(json.dumps(state, ensure_ascii=False, indent=2))
            continue

        t = time.time()
        r = c.post("/v1/previsit/turns", json={"state": state, "utterance": u})
        dt = time.time() - t
        if r.status_code != 200:
            print(f"  [오류 {r.status_code}] {r.text}")
            continue
        last = r.json()
        state = last["state"]
        transcript.append({"utterance": u, "response": last, "latency_s": round(dt, 2)})
        audit = last["audit"]
        if audit:
            total += audit["usage"]["cost_usd"]
            if a.verbose:
                for up in audit["extraction"]["updates"]:
                    print(f"    · {up['axis']}: [{up['status']}] {up['value']!r} ← {up['evidence']!r}")
                print(f"    · {dt:.1f}s  ${audit['usage']['cost_usd']:.4f}")
        print(f"AI: {last['reply']}")
        if last["ended"]:
            print(f"\n[종료: {last['end_reason']}]")
            break

    if last:
        show_card(last["card"])
    print(f"턴 {len(transcript)}  누적 ${total:.4f}")
    out = ROOT / "evals" / "results" / f"chat-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"기록: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
