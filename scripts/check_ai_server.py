"""떠 있는 AI 서버를 HMAC 서명해서 찔러 본다. **Bedrock이 실제로 답하는지까지 확인한다.**

백엔드가 붙이기 전에 "우리 쪽이 준비됐나"를 한 번에 보는 용도다. `/health`만 보면 컨테이너가
떴다는 것뿐이고, 모델이 실제로 도는지·서명이 맞는지는 모른다.

    # 서버 안에서(백엔드 내부망)
    uv run python scripts/check_ai_server.py --base http://ai:8000

    # HMAC을 켠 뒤
    uv run python scripts/check_ai_server.py --base http://ai:8000 --secret "$MEDIMATE_HMAC_SECRET"

    # 모델을 부르지 않고 경로·서명만 확인(비용 0)
    uv run python scripts/check_ai_server.py --base http://ai:8000 --no-llm

무엇을 보나

1. `/health` — 컨테이너가 떴는가
2. `GET /v1/ontology/body-map` — 온톨로지가 로드됐는가. **GET도 서명이 필요하다**(2026-09-11부터)
3. `POST /v1/previsit/sessions` — 세션이 열리는가
4. `POST /v1/previsit/turns` — **여기서 Bedrock을 부른다.** `audit.source`·`usage`·`model_id`로
   실제 호출을 확인한다. `--no-llm`이면 `selections`만 보내 LLM 없이 카드가 채워지는지 본다

키를 안 주면 서명 없이 보낸다. 서버가 `MEDIMATE_HMAC_SECRET`을 갖고 있으면 401이 나는데,
그것도 정보다 — **검증이 켜져 있다는 뜻**이다.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, "src")

from medimate.api.auth import sign  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OK, NO = "  OK ", "  X  "


def call(base: str, method: str, path: str, body: dict | None, secret: str | None, rid: str):
    raw = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else b""
    headers = {"Content-Type": "application/json"} if body is not None else {}
    if secret:
        ts = str(int(time.time()))
        headers |= {
            "X-Signature": sign(secret, method, path, ts, rid, raw),
            "X-Timestamp": ts,
            "X-Request-Id": rid,
        }
    req = urllib.request.Request(base.rstrip("/") + path, data=raw or None, headers=headers)
    req.get_method = lambda: method
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read() or b"{}"), time.perf_counter() - t0
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}"), time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001 — 연결 실패도 결과다
        return 0, {"detail": f"{type(e).__name__}: {e}"}, time.perf_counter() - t0


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    a = argparse.ArgumentParser()
    a.add_argument("--base", default="http://127.0.0.1:8000")
    a.add_argument("--secret", default=None, help="MEDIMATE_HMAC_SECRET. 없으면 서명 없이 보낸다")
    a.add_argument("--no-llm", action="store_true", help="selections만 보내 LLM을 부르지 않는다")
    a.add_argument(
        "--utterance", default="오른쪽 무릎이 계단 내려갈 때만 아파요. 2주쯤 됐어요."
    )
    args = a.parse_args()

    print(f"대상 {args.base}   서명 {'켬' if args.secret else '끔'}\n")
    fails = 0

    # 1. health (서명 면제 경로)
    st, b, dt = call(args.base, "GET", "/health", None, None, "chk-health")
    ok = st == 200 and b.get("status") == "ok"
    fails += not ok
    print(f"{OK if ok else NO}/health  {st}  {dt * 1000:.0f}ms  {b}")

    # 2. 온톨로지 (GET도 서명 필요)
    st, b, dt = call(args.base, "GET", "/v1/ontology/body-map", None, args.secret, "chk-map")
    ok = st == 200 and b.get("anchors")
    fails += not ok
    extra = f"앵커 {len(b.get('anchors', []))}개" if ok else b.get("detail", "")
    print(f"{OK if ok else NO}GET /v1/ontology/body-map  {st}  {dt * 1000:.0f}ms  {extra}")
    if st == 401:
        print("       → 서명 검증이 켜져 있다. --secret 을 주고 다시 돌리면 된다")

    # 3. 세션 시작
    st, start, dt = call(
        args.base, "POST", "/v1/previsit/sessions", {"site_label": "무릎"}, args.secret, "chk-start"
    )
    ok = st == 200 and "state" in start
    fails += not ok
    print(f"{OK if ok else NO}POST /v1/previsit/sessions  {st}  {dt * 1000:.0f}ms")
    if not ok:
        print(f"       {start}")
        raise SystemExit(1 if fails else 0)
    print(f'       첫 질문: "{start["reply"][:60]}"')

    # 4. 턴 — 여기서 Bedrock을 부른다
    payload: dict = {"state": start["state"]}
    if args.no_llm:
        payload["selections"] = [{"axis": "severity", "value": "4 (매우 심함)"}]
        label = "POST /v1/previsit/turns (selections만, LLM 0)"
    else:
        payload["utterance"] = args.utterance
        label = "POST /v1/previsit/turns (Bedrock 호출)"
    st, turn, dt = call(args.base, "POST", "/v1/previsit/turns", payload, args.secret, "chk-turn")
    ok = st == 200
    fails += not ok
    print(f"{OK if ok else NO}{label}  {st}  {dt * 1000:.0f}ms")
    if not ok:
        print(f"       {turn}")
        raise SystemExit(1)

    audit = turn.get("audit") or {}
    src, usage = audit.get("source"), audit.get("usage") or {}
    print(f'       다음 질문: "{turn["reply"][:60]}"')
    print(f"       audit.source = {src}   usage = {usage}")
    prov = (turn.get("card") or {}).get("provenance") or {}
    print(f"       model_id = {prov.get('model_id')}   prompt_version = {prov.get('prompt_version')}")

    if args.no_llm:
        sev = ((turn.get("card") or {}).get("axes") or {}).get("severity") or {}
        good = sev.get("status") == "filled" and src == "none"
        fails += not good
        print(f"{OK if good else NO}selections가 LLM 없이 카드에 들어갔다 — severity={sev.get('value')}")
    else:
        good = src == "server" and usage.get("input_tokens", 0) > 0
        fails += not good
        print(f"{OK if good else NO}**Bedrock이 실제로 응답했다** (source=server, 토큰 > 0)")
        if not good:
            print("       source가 server가 아니거나 토큰이 0이다 — 모델을 안 불렀다는 뜻")
        axes = (turn.get("card") or {}).get("axes") or {}
        filled = [k for k, v in axes.items() if v.get("status") == "filled"]
        print(f"       채워진 축: {', '.join(filled) or '(없음)'}")

    print()
    print("전부 통과" if not fails else f"실패 {fails}건")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
