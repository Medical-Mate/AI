"""llama.cpp 스케줄러 디버그 로그에서 **CPU로 배치된 연산**만 뽑는다. 표준 라이브러리만.

폰에서 graph splits가 1.7B(3)보다 많을 때(Qwen3.5-2B 39) 무엇이 NPU 밖으로 나갔는지 이름으로 확인한다.
서버를 `-v`와 `GGML_SCHED_DEBUG=2`로 띄워야 노드별 배치가 찍힌다(`-lv 4`로는 안 찍힘, PC에서 확인):

  adb shell "cd /data/local/tmp/llama.cpp && GGML_SCHED_DEBUG=2 LD_LIBRARY_PATH=lib ADSP_LIBRARY_PATH=lib \\
    bin/llama-server -m Qwen3.5-2B-Q4_0-ssmout-q8.gguf --port 8080 -c 4096 --reasoning off \\
    --device HTP0 -ngl 99 -v" 2>&1 | tee sched-2b.log      # "graph splits" 줄이 나오면 Ctrl+C
  python scripts/sched_cpu_nodes.py sched-2b.log

첫 번째 예약 그래프(로그의 첫 SPLIT 묶음)만 본다. 층 번호는 텐서 이름 끝(`-0`, `-17`)에서 읽는다.
"""

from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict

NODE = re.compile(r"node #\s*(\d+) \(\s*([A-Z_]+)\):\s+(\S+(?: \([a-z]+\))?)\s+\(\s*[\d.]+[KMG]?\)\s+\[\s*(\S+)")
SPLIT = re.compile(r"## SPLIT #(\d+): (\S+)")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
    # 첫 그래프: SPLIT #0이 두 번째로 나오기 전까지
    starts = [m.start() for m in re.finditer(r"## SPLIT #0:", text)]
    if not starts:
        sys.exit("SPLIT 줄이 없다 — 서버를 -v 와 GGML_SCHED_DEBUG=2 로 띄웠는지 확인")
    first = text[starts[0] : starts[1] if len(starts) > 1 else len(text)]

    splits = SPLIT.findall(first)
    print(f"분할 {len(splits)}개: " + " ".join(f"{b}" for _, b in splits[:12]) + (" …" if len(splits) > 12 else ""))
    cpu = []
    for m in NODE.finditer(first):
        idx, op, name, backend = int(m.group(1)), m.group(2), m.group(3), m.group(4)
        if backend.upper().startswith("CPU"):
            cpu.append((idx, op, name))
    print(f"CPU에 놓인 연산 {len(cpu)}개")
    by_op = Counter(op for _, op, _ in cpu)
    for op, n in by_op.most_common():
        print(f"  {op:<12} {n}")
    layers = defaultdict(list)
    for idx, op, name in cpu:
        m = re.search(r"-(\d+)$", name)
        layers[m.group(1) if m else "-"].append(f"#{idx} {op} {name}")
    print("층별(앞 3개 층):")
    for k in sorted(layers, key=lambda x: (x == "-", int(x) if x.isdigit() else 0))[:4]:
        print(f"  층 {k}: " + " · ".join(layers[k][:6]))


if __name__ == "__main__":
    main()
