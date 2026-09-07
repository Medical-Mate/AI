# 온디바이스 후보 스크리닝 (2026-09-07 시작)

목적: 갤럭시 S24 울트라(RAM 12GB)에서 llama.cpp로 돌릴 **초경량 추출 모델**을 우리 eval(88케이스, D1~D10)로 고른다.
PC와 기기가 같은 엔진(llama.cpp)·같은 GGUF·온도 0·시드 고정이므로 **PC 결과 = 기기 결과**. 실험은 전부 PC에서, 안드로이드에는 고정 묶음(모델·프롬프트·문법·테스트 벡터)만 넘긴다.

후보 조사 원본: `candidates-2026-09-07.md`. 스키마: `turn_extraction.schema.json` (디코딩 단계에서 스키마 위반 D2를 구조로 차단).

## 사다리 (작은 것부터. 위반 0이 나오는 최소 모델을 고른다)

| # | 모델 | Q4 크기 | 출시 | 한국어 | 라이선스 | GGUF |
|---|---|---|---|---|---|---|
| 1 | LFM2.5-230M | 0.15GB | 2026-06 | 공식 목록 포함 | LFM1.0 (상업 조건 확인 필요) | `LiquidAI/LFM2.5-230M-GGUF` |
| 2 | Qwen3.5-0.8B | 0.53GB | 2026-03 | 201개 언어 표기 | Apache-2.0 | `unsloth/Qwen3.5-0.8B-GGUF` |
| 3 | LFM2.5-1.2B-Instruct (참고: 2026-01, 6개월 초과) | 0.70GB | 2026-01 | 공식 목록 포함 | LFM1.0 | `LiquidAI/LFM2.5-1.2B-Instruct-GGUF` |
| 4 | Qwen3.5-2B | 1.28GB | 2026-03 | 201개 언어 | Apache-2.0 | `unsloth/Qwen3.5-2B-GGUF` |
| 5 | LFM2.5-2.6B | 1.52GB (IQ4_XS) | 2026-08 | 공식 목록 포함 | LFM1.0 | `LiquidAI/LFM2.5-2.6B-GGUF` |

보류: Kanana-2-1.3B(한국어 특화·상업 허용이지만 공식 GGUF 없음, SWA 구조의 llama.cpp 재현 미확인), Gemma 4 E2B(Q4 3.1GB, 상한 2배), EXAONE-4.0-1.2B(NC 라이선스), MiniCPM5-1B(한국어 미확인).

## 실행

```
# 1) 서버 (HF에서 직접 받는다. 첫 실행만 다운로드)
llama-server -hf unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M --port 8080 -c 4096 --temp 0 --seed 42 \
  --json-schema-file evals/ondevice/turn_extraction.schema.json --reasoning-budget 0

# 2) eval (비용 0. model_id는 "local/<이름>"; 이름은 결과 파일명에만 쓰인다)
uv run python -m medimate.evals.run --provider local --model local/qwen3.5-0.8b-q4 --yes
uv run python -m medimate.evals.run --report evals/results/local/qwen3.5-0.8b-q4.jsonl
```

- 사고(thinking) 모델은 서버에서 끈다(`--reasoning-budget 0` 또는 모델별 `/no_think`). 켜 두면 출력 한도를 사고가 먹는다(Gemini에서 본 문제)
- `--json-schema-file`을 켠 결과와 끈 결과를 둘 다 남긴다. 켰을 때 D2가 0이 되는 것은 당연하고, **D4(근거 그대로)·D5(숫자 출처)·D7(축 집합)**이 진짜 지표다
- 결과 원본은 `evals/results/local/` (gitignore). 판정은 `RESULTS.md`에 적는다

## 판정 기준

- 채택: 88케이스 안전 위반(D4/D5/D6) 0, 전체 통과 ≥ Terra − 5. 폼·칩으로 대체되는 축(심각도·경과·짧은 선택형)에서만 실패하면 허용
- 이 기준을 만족하는 **가장 작은** 모델을 고른다. 없으면 온디바이스는 접고 서버 Terra 유지. 이 문서가 그 근거

## 안드로이드에 넘기는 묶음 (채택 시)

GGUF 파일(정확한 양자화 이름), 프롬프트 파일(버전), `turn_extraction.schema.json`(또는 GBNF), 추론 설정(온도 0·시드·max_tokens·reasoning off), 테스트 벡터 88개(입력·기대 출력). 안드로이드는 기기에서 벡터 일치만 확인한다.

## 노트북 폴백

같은 `llama-server`를 핫스팟 LAN에 띄우고 앱이 `MEDIMATE_LOCAL_BASE_URL`만 바꾸면 심사장 오프라인 데모가 된다. 제품 기본 경로는 아니다(온디바이스의 장점이 없고 Terra보다 품질이 낮다).
