"""HMAC 서명 테스트 벡터를 만든다. 백엔드가 자기 구현을 바이트 단위로 대조하는 용도.

안드로이드에 준 88개 추출 벡터와 같은 방식이다 — **말로 규격을 설명하는 것보다 벡터 하나가
확실하다.** 서명이 안 맞을 때 어디서 갈렸는지(메서드 대소문자? 경로? 본문 직렬화? 인코딩?)를
한 줄씩 좁혀 갈 수 있다.

    uv run python scripts/gen_hmac_vectors.py            # 벡터 재생성
    uv run python scripts/gen_hmac_vectors.py --verify   # 파일의 기대값이 지금 구현과 같은지

산출물: docs/examples/hmac-vectors.json (저장소에 커밋한다 — 계약의 일부다)
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from medimate.api.auth import sign  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "examples" / "hmac-vectors.json"

SECRET = "test-secret-do-not-use-in-production"

# 각 벡터는 **하나씩만 다르게** 만들었다. 서명이 안 맞을 때 어느 요소에서 갈렸는지 좁히라고.
CASES: list[dict] = [
    {
        "id": "V01",
        "note": "기본형. 빈 본문 POST",
        "method": "POST",
        "path": "/v1/previsit/sessions",
        "timestamp": "1789000000",
        "request_id": "req-0001",
        "body": "{}",
    },
    {
        "id": "V02",
        "note": "한글 본문. UTF-8 바이트로 서명한다 — 이스케이프(\\uXXXX)하면 서명이 달라진다",
        "method": "POST",
        "path": "/v1/previsit/turns",
        "timestamp": "1789000001",
        "request_id": "req-0002",
        "body": '{"utterance":"오른쪽 무릎이 계단 내려갈 때 아파요"}',
    },
    {
        "id": "V03",
        "note": "V02와 본문이 같고 경로만 다르다. 경로가 서명에 들어가는지 확인",
        "method": "POST",
        "path": "/v1/previsit/sessions",
        "timestamp": "1789000001",
        "request_id": "req-0002",
        "body": '{"utterance":"오른쪽 무릎이 계단 내려갈 때 아파요"}',
    },
    {
        "id": "V04",
        "note": "V02와 전부 같고 메서드만 다르다. 메서드가 서명에 들어가는지 확인",
        "method": "PUT",
        "path": "/v1/previsit/turns",
        "timestamp": "1789000001",
        "request_id": "req-0002",
        "body": '{"utterance":"오른쪽 무릎이 계단 내려갈 때 아파요"}',
    },
    {
        "id": "V05",
        "note": "GET + 빈 본문. GET도 서명이 필요하다(2026-09-11부터)",
        "method": "GET",
        "path": "/v1/ontology/body-map",
        "timestamp": "1789000002",
        "request_id": "req-0003",
        "body": "",
    },
    {
        "id": "V06",
        "note": "쿼리가 붙은 GET. **쿼리는 서명에서 제외한다** — path만 쓴다",
        "method": "GET",
        "path": "/v1/ontology/search",
        "query_not_signed": "?q=무릎&limit=8",
        "timestamp": "1789000003",
        "request_id": "req-0004",
        "body": "",
    },
    {
        "id": "V07",
        "note": "request_id 헤더가 없을 때. 그 자리를 빈 문자열로 계산한다(백엔드는 항상 보내기로 했다)",
        "method": "POST",
        "path": "/v1/previsit/turns",
        "timestamp": "1789000004",
        "request_id": "",
        "body": '{"state":{}}',
    },
    {
        "id": "V08",
        "note": "공백이 든 본문. **직렬화된 그대로**의 바이트로 서명한다 — 재직렬화하면 달라진다",
        "method": "POST",
        "path": "/v1/previsit/turns",
        "timestamp": "1789000005",
        "request_id": "req-0005",
        "body": '{"utterance": "무릎", "selections": []}',
    },
    {
        "id": "V09",
        "note": "selections만 보내는 턴(3단계 통증 강도 슬라이더)",
        "method": "POST",
        "path": "/v1/previsit/turns",
        "timestamp": "1789000006",
        "request_id": "req-0006",
        "body": '{"selections":[{"axis":"severity","value":"4 (매우 심함)"}]}',
    },
    {
        "id": "V10",
        "note": "진료 후 메모. 줄바꿈이 든 본문",
        "method": "POST",
        "path": "/v1/postvisit/memo",
        "timestamp": "1789000007",
        "request_id": "req-0007",
        "body": '{"memo":"위염 초기래요.\\n약은 2주분 받았어요."}',
    },
]


def build() -> dict:
    out = []
    for c in CASES:
        body_bytes = c["body"].encode("utf-8")
        out.append(
            {
                **{k: v for k, v in c.items() if k != "body"},
                "body": c["body"],
                "body_utf8_bytes": len(body_bytes),
                "signature": sign(
                    SECRET, c["method"], c["path"], c["timestamp"], c["request_id"], body_bytes
                ),
            }
        )
    return {
        "note": (
            "HMAC 서명 테스트 벡터. 아래 secret으로 각 요청을 서명하면 signature가 나와야 한다. "
            "안 맞으면 어느 벡터에서 갈리는지 보고 원인을 좁힌다."
        ),
        "algorithm": "HMAC-SHA256, hex digest (소문자)",
        "message": 'f"{METHOD}.{path}.{timestamp}.{request_id}." + body_utf8_bytes',
        "rules": [
            "METHOD는 대문자",
            "path는 쿼리스트링 제외, 앞 / 포함. 프록시를 지난 뒤의 경로",
            "구분자는 '.' 이고 request_id 뒤에도 하나 붙는다. 그 뒤에 바로 본문 바이트",
            "본문은 직렬화된 그대로의 UTF-8 바이트. 재직렬화하면 공백·키 순서가 달라져 서명이 틀어진다",
            "한글을 \\uXXXX로 이스케이프하지 않는다(UTF-8 원바이트)",
            "request_id 헤더가 없으면 그 자리는 빈 문자열",
            "timestamp는 epoch 초 정수 문자열. 허용 폭 ±300초(벡터에서는 검증 대상 아님)",
        ],
        "headers": {
            "signature": "X-Signature",
            "timestamp": "X-Timestamp",
            "request_id": "X-Request-Id",
        },
        "exempt_paths": ["/health", "/docs", "/openapi.json", "/redoc"],
        "secret": SECRET,
        "vectors": out,
    }


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    a = argparse.ArgumentParser()
    a.add_argument("--verify", action="store_true", help="파일의 기대값이 지금 구현과 같은지만 확인")
    args = a.parse_args()

    fresh = build()
    if args.verify:
        if not OUT.exists():
            raise SystemExit(f"{OUT} 없음. 먼저 생성하세요")
        saved = json.loads(OUT.read_text(encoding="utf-8"))
        bad = [
            (s["id"], s["signature"], f["signature"])
            for s, f in zip(saved["vectors"], fresh["vectors"], strict=True)
            if s["signature"] != f["signature"]
        ]
        if bad:
            for vid, want, got in bad:
                print(f"불일치 {vid}\n  파일 {want}\n  구현 {got}")
            raise SystemExit("서명 규격이 바뀌었다. 바꾼 것이 맞으면 재생성하고 백엔드에 알릴 것")
        print(f"OK — 벡터 {len(saved['vectors'])}개가 지금 구현과 일치")
        return

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fresh, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{OUT.relative_to(ROOT)} — 벡터 {len(fresh['vectors'])}개")
    for v in fresh["vectors"]:
        print(f"  {v['id']} {v['method']:5} {v['path']:28} {v['signature'][:16]}…  {v['note'][:40]}")


if __name__ == "__main__":
    main()
