"""폰 llama-server에 테스트 벡터를 흘려 출력·지연·발열을 기록한다. **표준 라이브러리만** — 저장소 환경 없이 돈다.

폰에서 `llama-server`(pkg-android-2092353)를 NPU로 띄우고 `adb forward tcp:8080 tcp:8080` 한 뒤:

  python scripts/ondevice_device_run.py --vectors evals/ondevice/vectors-qwen3.5-2b-q4_0-extract-small-2b-v1.jsonl \\
      --out device-2b-extract.jsonl --adb-temp
  python scripts/ondevice_device_run.py --vectors evals/ondevice/vectors-qwen3.5-2b-q4_0-extract-small-2b-v1.jsonl \\
      --out device-2b-heat.jsonl --minutes 5 --adb-temp          # 5분 연속(발열)

결과 jsonl을 이슈 #35에 올리면 AI 쪽이 같은 채점기(가드 포함)로 판정한다. 여기서는 숫자만 요약한다.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def _temp_c() -> float | None:
    """배터리 온도(°C). 폰 발열의 대용치 — 9/7 실측도 같은 값을 썼다."""
    try:
        out = subprocess.run(
            ["adb", "shell", "dumpsys", "battery"], capture_output=True, text=True, timeout=5
        ).stdout
    except Exception:  # noqa: BLE001
        return None
    for line in out.splitlines():
        if "temperature" in line:
            try:
                return int(line.split(":")[1]) / 10
            except ValueError:
                return None
    return None


def _call(base: str, vec: dict, schema_dir: Path) -> tuple[str, float, dict]:
    s = vec["settings"]
    if "messages" in vec:
        messages = vec["messages"]
    else:
        messages = [{"role": "system", "content": vec["system"]}, {"role": "user", "content": vec["user"]}]
    schema = json.loads((schema_dir / s["json_schema"]).read_text(encoding="utf-8"))
    body = {
        "messages": messages,
        "temperature": s["temperature"],
        "seed": s["seed"],
        "max_tokens": s["max_tokens"],
        "response_format": {"type": "json_schema", "json_schema": {"name": "response", "schema": schema}},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read().decode("utf-8"))
    return d["choices"][0]["message"]["content"] or "", time.perf_counter() - t0, d.get("timings") or {}


def _same(a: str, b: str) -> bool:
    try:
        return json.loads(a) == json.loads(b)
    except json.JSONDecodeError:
        return a == b


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # 윈도우 콘솔(evals/run.py 방식)
    p = argparse.ArgumentParser()
    p.add_argument("--vectors", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--base", default="http://127.0.0.1:8080")
    p.add_argument("--schema-dir", type=Path, help="스키마 파일 폴더(기본: 벡터 파일과 같은 폴더)")
    p.add_argument("--minutes", type=float, help="이 시간 동안 벡터를 반복(발열 측정)")
    p.add_argument("--adb-temp", action="store_true", help="요청마다 adb로 배터리 온도를 읽는다")
    p.add_argument("--limit", type=int, help="앞에서 N개만(연결 확인용)")
    a = p.parse_args()
    schema_dir = a.schema_dir or a.vectors.parent
    vecs = [json.loads(x) for x in a.vectors.read_text(encoding="utf-8").splitlines() if x.strip()]
    if a.limit:
        vecs = vecs[: a.limit]

    rows = []
    start = time.time()
    with a.out.open("w", encoding="utf-8", newline="\n") as f:
        i = 0
        while True:
            v = vecs[i % len(vecs)]
            if a.minutes is None and i >= len(vecs):
                break
            if a.minutes is not None and time.time() - start > a.minutes * 60:
                break
            try:
                text, lat, tm = _call(a.base, v, schema_dir)
                err = None
            except Exception as e:  # noqa: BLE001 — 실패도 기록한다
                text, lat, tm, err = "", 0.0, {}, f"{type(e).__name__}: {e}"
            row = {
                "id": v["id"],
                "n": i,
                "t": round(time.time() - start, 1),
                "output": text,
                "error": err,
                "latency_s": round(lat, 3),
                "prompt_n": tm.get("prompt_n"),
                "prompt_per_second": tm.get("prompt_per_second"),
                "predicted_n": tm.get("predicted_n"),
                "predicted_per_second": tm.get("predicted_per_second"),
                "temp_c": _temp_c() if a.adb_temp else None,
                "same_as_pc": _same(text, v["expected_output"]) if not err else False,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            rows.append(row)
            print(f"{i:4d} {v['id']:<10} {lat:6.2f}s  {row['predicted_per_second'] or 0:5.1f} tok/s"
                  f"  {row['temp_c'] or '-'}°C  {'=' if row['same_as_pc'] else '≠'}{'  ' + err if err else ''}")
            i += 1

    ok = [r for r in rows if not r["error"]]
    lats = [r["latency_s"] for r in ok]
    tps = [r["predicted_per_second"] for r in ok if r["predicted_per_second"]]
    temps = [r["temp_c"] for r in rows if r["temp_c"] is not None]
    print("\n요약")
    print(f"  요청 {len(rows)} · 오류 {len(rows) - len(ok)} · PC와 같은 출력 {sum(r['same_as_pc'] for r in ok)}")
    if lats:
        print(f"  첫 호출 {lats[0]:.2f}s · 지연 p50 {statistics.median(lats):.2f}s"
              f" · p90 {sorted(lats)[int(len(lats) * 0.9)]:.2f}s")
    if tps:
        print(f"  생성 속도 중앙값 {statistics.median(tps):.1f} tok/s")
    if temps:
        print(f"  배터리 온도 {temps[0]}°C → {temps[-1]}°C (최고 {max(temps)}°C)")


if __name__ == "__main__":
    sys.exit(main())
