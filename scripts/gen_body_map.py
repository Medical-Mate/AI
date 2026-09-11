"""`docs/examples/body-map.json`을 지금 온톨로지로 다시 만든다.

이 파일은 **백엔드와 안드로이드가 각자 저장소에 박아 둔** 부위 마스터다. 온톨로지를 고치면
같이 갱신해야 하는데, 지금까지 생성기가 없어서 손으로 맞춰 왔다. 그러면 조용히 낡는다 —
`tests/test_ontology.py`가 드리프트를 잡지만, 잡힌 뒤 **어떻게 고치는지**가 없으면 그 실패가
막힌 길이 된다.

    uv run python scripts/gen_body_map.py            # 다시 만든다
    uv run python scripts/gen_body_map.py --check    # 커밋본이 지금 동작과 같은지만 본다

값은 `GET /v1/ontology/body-map` 응답과 **같은 함수에서 나온다**(`api.app`의 핸들러를
그대로 호출한다). 생성기와 엔드포인트가 따로 놀면 파일이 응답과 달라지고, 그건 이 파일을
믿고 박아 둔 두 팀에 그대로 간다.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from fastapi.testclient import TestClient  # noqa: E402

from medimate.api.app import create_app  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "examples" / "body-map.json"

def build() -> dict:
    """엔드포인트를 그대로 불러서 만든다. **응답을 한 글자도 손대지 않는다.**

    처음에 `note`·`image_key_note`를 여기서 덮어쓰게 짰다가, 커밋본과 안 맞아 드리프트가
    난 줄 알았다. 실제로는 파일이 멀쩡했고 **생성기가 응답을 바꾸고 있었다.**
    생성기가 응답에 무언가를 더하면 파일이 `GET /v1/ontology/body-map`과 달라지고,
    그 차이가 이 파일을 믿고 박아 둔 두 팀에 그대로 간다. 설명문도 엔드포인트가 낸다.
    """
    client = TestClient(create_app())
    r = client.get("/v1/ontology/body-map")
    r.raise_for_status()
    return r.json()


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    a = argparse.ArgumentParser()
    a.add_argument("--check", action="store_true", help="다시 쓰지 않고 차이만 본다")
    args = a.parse_args()

    fresh = build()
    if args.check:
        if not OUT.exists():
            raise SystemExit(f"{OUT} 없음")
        saved = json.loads(OUT.read_text(encoding="utf-8"))
        if saved == fresh:
            print(f"OK — {OUT.relative_to(ROOT)}가 지금 온톨로지와 같다 (snapshot {fresh['ontology_snapshot']})")
            return
        if saved.get("ontology_snapshot") != fresh.get("ontology_snapshot"):
            print(f"스냅샷 다름: 파일 {saved.get('ontology_snapshot')} / 지금 {fresh['ontology_snapshot']}")
        raise SystemExit("어긋남. `uv run python scripts/gen_body_map.py`로 다시 만들고 함께 커밋할 것")

    OUT.write_text(json.dumps(fresh, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    n_zone = sum(len(x["zones"]) for x in fresh["anchors"])
    print(
        f"{OUT.relative_to(ROOT)} — 앵커 {len(fresh['anchors'])} · 구역 {n_zone} "
        f"· snapshot {fresh['ontology_snapshot']}"
    )


if __name__ == "__main__":
    main()
