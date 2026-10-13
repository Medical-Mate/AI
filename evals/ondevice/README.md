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
| N1 | Qwen3-0.6B **Q4_0** (NPU 트랙) | ~0.4GB | 2025-04 | 119개 언어 | Apache-2.0 | `unsloth/Qwen3-0.6B-GGUF` |
| N2 | Qwen3-1.7B **Q4_0** (NPU 트랙) | ~1.0GB | 2025-04 | 119개 언어 | Apache-2.0 | `unsloth/Qwen3-1.7B-GGUF` |

보류: Kanana-2-1.3B(한국어 특화·상업 허용이지만 공식 GGUF 없음, SWA 구조의 llama.cpp 재현 미확인), Gemma 4 E2B(Q4 3.1GB, 상한 2배), EXAONE-4.0-1.2B(NC 라이선스), MiniCPM5-1B(한국어 미확인).

## NPU 트랙 (2026-09-07 추가) — llama.cpp Hexagon 백엔드

llama.cpp 공식 저장소에 Snapdragon 백엔드(CPU · Adreno GPU · **Hexagon NPU**)가 들어 있다
(`docs/backend/snapdragon/README.md`). 빌드 산출물에 `libggml-htp-v73/v75/v79/v81.so`가 포함되고
**8 Gen 3(S24 울트라)은 v75**이므로 지원 범위 안이다. JSON 스키마 강제는 샘플링 단계라 백엔드와 무관하게 그대로 동작한다.
→ 모델·프롬프트·문법·테스트 벡터는 CPU 트랙과 같고, **안드로이드 빌드 옵션 하나(Hexagon 포함)만 달라진다.**

- 퀄컴 AI Hub(Genie) 경로는 8 Gen 3 미지원(8 Elite 이상). S24 울트라에서는 닫혀 있다. 검토 종료
- NPU에는 **표준 트랜스포머(Qwen3)**가 유리하다. Qwen3.5의 DeltaNet 혼합 어텐션은 NPU 커널이 초기 단계라
  연산이 CPU로 떨어질 수 있다. 그래서 NPU 트랙 후보는 **Qwen3-0.6B / Qwen3-1.7B** (2025-04, 6개월 조건 밖이지만 NPU 때문에 예외)
- 백엔드 예시가 전부 **Q4_0**이다. K-quant(Q4_K_M)는 오프로드가 안 될 수 있으니 NPU 트랙은 Q4_0 파일로 돈다
- 미확인: 8 Gen 3에서 실제 오프로드 비율·속도, Qwen3의 모든 연산이 NPU에 올라가는지. 기기에서만 확인 가능(`GGML_HEXAGON_VERBOSE=1`, `GGML_HEXAGON_PROFILE=1`)
- 안드로이드 빌드: `scripts/snapdragon/build.py` (퀄컴 도커 툴체인, adb 푸시까지)

CPU 트랙과 NPU 트랙은 **같은 88케이스**로 채점한다. NPU 트랙은 Q4_0라 CPU 트랙(Q4_K_M)과 결과가 다를 수 있으므로 각각 기록한다.

## 기기 NPU 실험 — PC에서, 앱 없이 (2026-09-07 준비)

에뮬레이터로는 불가(Hexagon은 실물 DSP). 대신 **S24 울트라를 USB로 이 PC에 꽂고 명령줄 바이너리로 돈다.**
안드로이드 팀 손을 빌리지 않는다. 앱 통합은 결과가 좋을 때만.

준비물: Docker Desktop(있음), adb(`Google.PlatformTools`, 설치됨), llama.cpp 소스(`C:/Users/user/orca/tools/llama.cpp`),
폰에서 개발자 옵션 → USB 디버깅 켜기.

```
# 0) 폰 연결 확인
adb devices

# 1) 안드로이드용 빌드 (퀄컴 도커 툴체인 ghcr.io/snapdragon-toolchain/arm64-android:v0.7 — 첫 실행에 이미지 수 GB 다운로드)
cd C:/Users/user/orca/tools/llama.cpp
python scripts/snapdragon/build.py --target android --push        # 빌드 후 /data/local/tmp/llama.cpp 로 adb 푸시

# 2) 모델 푸시 (NPU 트랙은 Q4_0)
adb push evals/ondevice/models/Qwen3-1.7B-Q4_0.gguf /data/local/tmp/llama.cpp/

# 3) 기기에서 서버 실행 — NPU(HTP0) 지정, 프로파일 켜기
python scripts/snapdragon/run.py --target android --devices HTP0 --hex-profile 1 --   llama-server -m /data/local/tmp/llama.cpp/Qwen3-1.7B-Q4_0.gguf --port 8080 -c 4096 --temp 0 --seed 42   --json-schema-file /data/local/tmp/llama.cpp/turn_extraction.schema.json --reasoning off

# 4) PC 포트 연결 후 우리 러너가 기기 NPU를 부른다
adb forward tcp:8080 tcp:8080
uv run python -m medimate.evals.run --provider local --model local/npu-qwen3-1.7b-q4_0 --yes
```

- `--devices HTP0`이 NPU, 빼면 CPU. 같은 기기에서 둘을 다 돌려 **속도 차이와 채점 차이**를 나란히 기록한다
- `--hex-profile 1`로 연산별 NPU/CPU 배분이 로그에 남는다. 오프로드 비율이 낮으면 이득이 없다
- 미확인: v75(8 Gen 3) 실측. 문서에 지원 라이브러리는 있으나 이 기기 실측은 우리가 처음이다

## 실행

```
# 1) 서버 (HF에서 직접 받는다. 첫 실행만 다운로드)
llama-server -hf unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M --port 8080 -c 4096 --temp 0 --seed 42 \
  --json-schema-file evals/ondevice/turn_extraction.schema.json --reasoning off

# 2) eval (비용 0. model_id는 "local/<이름>"; 이름은 결과 파일명에만 쓰인다)
uv run python -m medimate.evals.run --provider local --model local/qwen3.5-0.8b-q4 --yes
uv run python -m medimate.evals.run --report evals/results/local/qwen3.5-0.8b-q4.jsonl
```

- 사고(thinking) 모델은 서버에서 끈다 — **`--reasoning off`**(= 템플릿 `enable_thinking=false`, 폰 번들 렌더와 동일). 켜 두면 출력 한도를 사고가 먹는다(Gemini에서 본 문제).
  **`--reasoning-budget 0`만으로는 안 된다**(llama.cpp 0.4.0-dev build 10819): 템플릿이 `<think>`를 연 채라 문법이 `{`를 강제하면 모델이 `{}`로 닫는다. 2026-10-12에 108행 중 51행이 `{}`로 나와 회귀 63→43으로 보였다. `RESULTS.md` 2026-10-12 절
- `--json-schema-file`을 켠 결과와 끈 결과를 둘 다 남긴다. 켰을 때 D2가 0이 되는 것은 당연하고, **D4(근거 그대로)·D5(숫자 출처)·D7(축 집합)**이 진짜 지표다
- 결과 원본은 `evals/results/local/` (gitignore). 판정은 `RESULTS.md`에 적는다

## 판정 기준

- 채택: 88케이스 안전 위반(D4/D5/D6) 0, 전체 통과 ≥ Terra − 5. 폼·칩으로 대체되는 축(심각도·경과·짧은 선택형)에서만 실패하면 허용
- 이 기준을 만족하는 **가장 작은** 모델을 고른다. 없으면 온디바이스는 접고 서버 Terra 유지. 이 문서가 그 근거

## 안드로이드에 넘기는 묶음 (채택 시)

GGUF 파일(정확한 양자화 이름), 프롬프트 파일(버전), `turn_extraction.schema.json`(또는 GBNF), 추론 설정(온도 0·시드·max_tokens·reasoning off), 테스트 벡터 88개(입력·기대 출력). 안드로이드는 기기에서 벡터 일치만 확인한다.

## 노트북 폴백

같은 `llama-server`를 핫스팟 LAN에 띄우고 앱이 `MEDIMATE_LOCAL_BASE_URL`만 바꾸면 심사장 오프라인 데모가 된다. 제품 기본 경로는 아니다(온디바이스의 장점이 없고 Terra보다 품질이 낮다).

## 안드로이드 대조용 산출물 (2026-09-14, 이슈 #35 #77)

앱이 `llama_chat_apply_template`·손으로 옮긴 GBNF로 돌린 결과를 서버 경로와 대조하기 위한 파일들.

| 파일 | 무엇 | 어떻게 만들었나 |
|---|---|---|
| `turn_extraction.small.gbnf` | 진료 전 추출 스키마의 GBNF | llama.cpp 업스트림 `examples/json_schema_to_grammar.py` ← `turn_extraction.small.schema.json` |
| `qwen3-1.7b.chat_template.jinja` | 모델 안의 채팅 템플릿 원문 | `Qwen3-1.7B-Q4_0.gguf`(sha256 `c876f159…81c5`)의 `tokenizer.chat_template`을 gguf-py로 추출 |
| `rendered/A01-0.*.txt` | 벡터 A01#0을 렌더한 전문 두 벌 | 위 템플릿을 jinja2(`trim_blocks`·`lstrip_blocks`)로 렌더. `enable_thinking=false`는 끝에 `<think>

</think>

`만 더 붙는다. **`llama-server` 로그 캡처가 아니다** — 같은 템플릿의 재렌더다 |
| `device-runs/2026-09-13-s24u-npu.jsonl` | 앱이 폰(S24U, HTP0)에서 낸 88건 raw, case id 포함 | 이슈 #35 코멘트를 결과 형식으로 옮김. 모델·프롬프트·설정은 벡터와 동일 |
| `device-runs/…raw.report.txt` · `…guard.report.txt` | 그 88건을 `score.py`로 채점한 결과 (가드 없이 / 온디바이스 가드) | `uv run python -m medimate.evals.run --report <jsonl> --prompt small [--guard ondevice]` |

### Qwen3.5-2B 후보 (2026-10-13, 폰 실측 요청 — `docs/android-ondevice-handoff.md` 같은 이름의 절)

| 파일 | 무엇 | 어떻게 만들었나 |
|---|---|---|
| `prompt-extract-small-2b-v1.system.txt` | 2B용 진료 전 추출 시스템 프롬프트 | `MEDIMATE_SMALL_PROMPT=2b-v1`의 `prompt_small.system_prompt()` (LF) |
| `vectors-qwen3.5-2b-q4_0-ssmq8-extract-small-2b-v1.jsonl` | 추출 벡터 79(앱 경로 65). 줄마다 `category`·`purpose`·`expect`(채점 기준)·`expected_output`(PC 출력) | `scripts/export_ondevice_2b.py` ← PC GPU 실행 원본 |
| `vectors-questions-qwen3.5-2b-q4_0-ssmq8-questions-small4.jsonl` | 질문 후보 벡터 100(`messages`에 예시 차례 포함) | 같은 스크립트 ← 이어 돌리기 원본(카드는 2B가 만든 것) |
| `questions.schema.json` · `questions.gbnf` | 질문 후보 출력 스키마와 그 GBNF | `assist_prompts.QUESTIONS_SCHEMA` · 업스트림 `examples/json_schema_to_grammar.py` |
| `vectors-memo-qwen3.5-2b-q4_0-ssmq8-memo-small-v5.jsonl` | 메모 분류 벡터 34(프롬프트·스키마는 1.7B v5와 같은 파일), PC 라벨 124/132 | `scripts/export_memo_ondevice_v5.py --model-tag …` |
| `prompt-followup-v1.system.txt` · `followup-v1.schema.json` · `vectors-followup-qwen3.5-2b-q4_0-ssmq8-followup-v1.jsonl` | 재방문 표현 자르기(폰) 66. 줄마다 `expect_days`·운영 순서 서버 날짜 `pc_server_date`(66/66) | `scripts/export_followup_ondevice.py` ← `eval_followup_reader.py --provider local` 원본 |
| `qwen3.5-2b.chat_template.jinja` | 모델 안의 채팅 템플릿 원문 | `llama-server`(b10819) `/props`의 `chat_template` |
| `rendered/2b-A01-0.…txt` · `rendered/2b-questions-PC01.…txt` | 벡터 하나씩을 `enable_thinking=false`로 렌더한 전문 | 같은 서버의 `/apply-template` — 앱의 `llama_chat_apply_template` 결과와 글자까지 같아야 한다 |

모델은 `hyos0415/Qwen3.5-2B-NPU-GGUF@1c069ab6` `Qwen3.5-2B-Q4_0-ssmout-q8.gguf`(SHA256 `c8e057df…dc53f`) — unsloth Q4_0에서 Hexagon이 못 하는 `ssm_out` Q5_K 18개만 Q8_0으로 바꿈(`scripts/gguf_swap_tensor_type.py`, 나머지 302개 텐서 바이트 동일). 1차 폰 실측(unsloth 원본)에서 graph splits 39가 이것 때문이었다.

PC 기준(폰 가드 재채점): 108문항 78(앱 경로 86 중 70) · 위반 0. unsloth 원본 77(68), 1.7B 82(68).

같은 채점기로 PC 실행(`evals/results/local/Qwen3-1.7B-Q4_0-small4b.jsonl`)과 나란히 보면:
PC 61/88·위반 0, 폰 60/88·위반 0 (둘 다 가드 적용). raw는 PC 59/3, 폰 58/2.
"축·상태가 다른" 7건 중 5건(A01×2·NG02·NG03·PD02)은 폰 쪽이 채점기 기준에 맞는다.
