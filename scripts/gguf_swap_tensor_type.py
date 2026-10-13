"""GGUF 하나에서 **지정한 텐서만** 다른 양자화로 바꿔 끼운다. 나머지 텐서·메타데이터는 바이트 그대로 복사.

왜: unsloth `Qwen3.5-2B-Q4_0.gguf`는 혼합 양자화라 `ssm_out`(DeltaNet 층마다 하나, 18개)이 Q5_K다.
Hexagon NPU(pkg-android-2092353)는 Q4_0·Q4_1·Q8_0·IQ4_NL·MXFP4만 계산하므로 이 18개가 CPU로 떨어져
graph splits가 3 → 39가 됐다(2026-10-13 폰 실측, 이슈 #35). BF16 원본에서 그 텐서만 Q8_0으로 다시 만들어
끼우면 나머지 모델은 unsloth와 바이트까지 같다 — 결과가 달라지면 원인은 바꾼 텐서뿐이다.
(BF16에서 전체를 다시 양자화하면 같은 imatrix로도 Q4_0 텐서 132개의 바이트가 달라져 비교가 흐려진다)

llama.cpp `gguf-py` 필요(저장소 의존성 아님):
  PYTHONPATH=<llama.cpp>/gguf-py uv run --no-project --with pyyaml --with numpy --with sentencepiece \\
    python scripts/gguf_swap_tensor_type.py BASE.gguf SOURCE-BF16.gguf OUT.gguf --suffix ssm_out.weight --type Q8_0
"""

from __future__ import annotations

import argparse
import sys

import gguf
from gguf.quants import dequantize, quantize


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("base")
    p.add_argument("source", help="바꿀 텐서를 가져올 고정밀 원본(BF16·F16·F32)")
    p.add_argument("out")
    p.add_argument("--suffix", required=True, help="이 이름으로 끝나는 텐서만 바꾼다(예: ssm_out.weight)")
    p.add_argument("--type", required=True, help="새 양자화(예: Q8_0)")
    p.add_argument("--note", default="", help="medimate.requant_note 메타데이터로 남길 설명")
    a = p.parse_args()

    qtype = gguf.GGMLQuantizationType[a.type]
    base = gguf.GGUFReader(a.base)
    src = {t.name: t for t in gguf.GGUFReader(a.source).tensors}
    arch = base.fields[gguf.Keys.General.ARCHITECTURE].contents()
    w = gguf.GGUFWriter(a.out, arch=arch, endianess=base.endianess)

    for field in base.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        vt = field.types[0]
        sub = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(field.name, field.contents(), vt, sub_type=sub)
    if a.note:
        w.add_key_value("medimate.requant_note", a.note, gguf.GGUFValueType.STRING)

    datas = []
    swapped = 0
    for t in base.tensors:
        if t.name.endswith(a.suffix):
            s = src[t.name]
            f32 = dequantize(s.data, s.tensor_type)
            data, ttype = quantize(f32, qtype), qtype
            swapped += 1
        else:
            data, ttype = t.data, t.tensor_type
        w.add_tensor_info(t.name, data.shape, data.dtype, data.nbytes, ttype)
        datas.append(data)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for d in datas:
        w.write_tensor_data(d, tensor_endianess=base.endianess)
    w.close()
    print(f"{swapped}개 텐서를 {a.type}로 바꿔 {a.out}에 썼다", file=sys.stderr)


if __name__ == "__main__":
    main()
