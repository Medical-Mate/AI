# 안드로이드 전달 — 온디바이스 묶음 (v2 Qwen3.5-2B · 2026-10-13 확정)

안드로이드에는 **고정된 묶음**과 **일치 확인 절차**만 넘깁니다. 모델·프롬프트·스키마를 앱에서 바꾸지 않습니다.
바뀌면 AI 쪽이 버전을 올려 다시 전달합니다. 추론만 폰에서 하고 가드·검증·날짜 계산·템플릿은 AI 서버가 합니다.

## 묶음 v2 — Qwen3.5-2B (지금 기준)

2026-10-13 폰 실측 두 번(#35)으로 확정. 같은 조건에서 1.7B 대비 앱 경로 턴 지연 p50 2.80초(1.7B 2.86) ·
5분 발열 같음(38.6°C) · 진료 전 추출 같음(폰 53/65, 위반 0) · 질문 후보 검증 통과 46 → 88 ·
메모 분류 89.4% → 93.9%. 첫 호출만 1.7초 느림(5.9초 — 화면을 열 때 시스템 프롬프트를 먼저 보내 두면 체감 없음).

| 항목 | 값 / 위치 | v1에서 바뀐 것 |
|---|---|---|
| 엔진 | `pkg-android-2092353` 그대로 | 없음 |
| 모델 | `Qwen3.5-2B-Q4_0-ssmout-q8.gguf` · 1,243,185,536 bytes · SHA256 `c8e057dfc224b87b2fce69690ee6b53ca828f67543d01d088e409cb8315dc53f` | **교체** |
| 받는 곳 | `https://huggingface.co/hyos0415/Qwen3.5-2B-NPU-GGUF/resolve/1c069ab6a03d5840d81ad29b2476752a13a407c0/Qwen3.5-2B-Q4_0-ssmout-q8.gguf` (계정 불필요, 이어받기 됨, Apache-2.0) | **교체** |
| 추론 설정 | temperature 0 · seed 42 · max_tokens 1024 · **reasoning off** · `-c 4096` · NPU `--device HTP0 -ngl 99` · **`n_seq_max = 1`**(llama-server는 `-np 1`) | **`n_seq_max = 1` 필수** |
| 왜 `n_seq_max = 1` | 2B의 `ssm_out` 행렬곱은 입력이 `[2048, 토큰 수, 시퀀스 수]` 3차원이고, Hexagon 행렬곱은 세 번째 차원이 1이 아니면 받지 않는다(no broadcasting). 시퀀스를 여럿 잡으면(llama-server 기본 `n_parallel` auto = 4) DeltaNet 18층의 이 행렬곱이 CPU로 가서 graph splits 39 · 호출마다 가중치 4MB 복사. `-np 1`로 splits 3(1.7B와 같음), 2026-10-13 폰 확인(#35). 앱은 환자 한 명이 한 번에 한 요청이라 1이면 충분하다 | 2B에서 생김 |
| 채팅 템플릿 | 모델 안 템플릿 그대로. 렌더 전문 `evals/ondevice/rendered/2b-*.enable_thinking_false.txt`와 글자까지 같아야 함 | 모델과 함께 바뀜 |

| 기능 | 언제 부르나 | 프롬프트 | 스키마 | 테스트 벡터 (PC 기준) |
|---|---|---|---|---|
| 1. 진료 전 추출 | 문답 중 환자가 답할 때마다 | `prompt-extract-small-2b-v1.system.txt` **(교체)** | `turn_extraction.small.schema.json` (그대로) | `vectors-qwen3.5-2b-q4_0-ssmq8-extract-small-2b-v1.jsonl` 79 · 줄마다 `expect` |
| 2. 질문 후보 | 문답이 끝난 뒤 한 번 | `questions-small4` (벡터 `messages`에 예시 차례까지) | `questions.schema.json` · `questions.gbnf` **(text 40자 상한)** | `vectors-questions-qwen3.5-2b-q4_0-ssmq8-questions-small4.jsonl` 100 |
| 3. 진료 후 기록(메모 분류) | 메모를 저장할 때 | `prompt-memo-small-v5.system.txt` **(v4 → v5: 라벨 `lifestyle_instructions` 추가)** | `memo_labels-memo-small-v5.schema.json` (문장 수 N에 맞춰 번호 키 생성) | `vectors-memo-qwen3.5-2b-q4_0-ssmq8-memo-small-v5.jsonl` 34 · 라벨 124/132 |
| └ 재방문 표현 **(선택)** | 메모에 재방문 문장이 있을 때만 — 안 붙이면 서버 규칙이 날짜를 센다 | `prompt-followup-v1.system.txt` **(새로)** | `followup-v1.schema.json` | `vectors-followup-qwen3.5-2b-q4_0-ssmq8-followup-v1.jsonl` 66 · 서버 날짜 66/66 |

폰 모델이 하는 일은 셋(진료 전 추출 · 질문 후보 · 진료 후 기록)이고 재방문 표현은 진료 후 기록 안의 선택 단계다.
전부 `evals/ondevice/` 아래. **케이스마다 무엇을 넣고 무엇이 나와야 하는지는 [`docs/android-ondevice-testcases.md`](android-ondevice-testcases.md)**
(벡터에서 자동 생성, PC 판정·PC도 틀리는 케이스 표시).

**앱에서 바꿀 것**
1. 모델 파일(URL·해시)과 진료 전 추출 시스템 프롬프트 교체
2. 질문 후보를 붙일 때 새 질문 스키마(40자 상한) 사용 — 같은 말 되풀이로 1024토큰까지 가는 일이 없어진다
3. 메모 분류를 폰으로 옮길 때 **v5** 프롬프트·스키마 사용(v4로 붙이면 생활 지침 문장이 다른 칸으로 간다).
   메모 API 응답의 `lifestyle_axis`가 true면 지침 칸이 따로 나간다(`docs/api-postvisit.md`)
4. 재방문 문장은 폰이 **표현만 잘라** `follow_up_readings`로 보낸다. 날짜는 서버가 센다 — 폰이 센 `days`는
   서버 규칙이 아무것도 못 읽었을 때만 쓴다(2026-10-13부터 앞 절 날짜를 덮지 않는다)
5. 추론 설정·엔진은 그대로

### llama-server 설정 → 앱(llama.cpp C API) 대응표

우리 숫자는 전부 `llama-server`로 쟀습니다. 앱은 C API를 직접 부르니 이름이 다릅니다. 아래 이름은 엔진 빌드
`2092353`의 `include/llama.h` 기준이에요(2026-10-13 확인).

| llama-server에서 쓴 것 | 앱에서 같은 것 | 주의 |
|---|---|---|
| `-m <gguf>` | `llama_model_load_from_file(path, mparams)` | |
| `--device HTP0` | `mparams.devices = {ggml_backend_dev_by_name("HTP0"), NULL}` | |
| `-ngl 99` | `mparams.n_gpu_layers = 99` | |
| 환경 `LD_LIBRARY_PATH` · `ADSP_LIBRARY_PATH` | 백엔드를 올리기 **전에** `setenv("ADSP_LIBRARY_PATH", <lib 경로>, 1)` | 빠뜨리면 에러 없이 CPU로 돈다 |
| `-c 4096` | `cparams.n_ctx = 4096` | |
| `-np 1` | **`cparams.n_seq_max = 1`** | 2B 필수(위 "왜 `n_seq_max = 1`") |
| (기본) | `cparams.n_batch = 2048` · `cparams.n_ubatch = 512` | 실측 때 값 |
| (기본 auto) | `cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO` | |
| `--temp 0 --seed 42` | 샘플러 체인 끝에 `llama_sampler_init_greedy()` | 온도 0이라 greedy. seed는 쓰이지 않는다 |
| 요청별 `response_format: json_schema` | `llama_sampler_init_grammar(vocab, <GBNF>, "root")` — **체인에 넣지 말고 따로 들고**, greedy로 고른 토큰 **하나만** 문법에 대어 본다(후보 1개짜리 배열에 `llama_sampler_apply`). 막히면(logit `-INFINITY`) 그때만 전체 후보에 문법을 걸고 다시 greedy. 고른 토큰은 `llama_sampler_accept`로 문법에도 알린다 | llama-server(`common_sampler`)가 이렇게 한다. 체인 앞에 걸면 매 토큰 어휘 전체를 문법으로 걸러 **추출 p50 6.0초 → 3.0초** 차이(2026-10-14 앱 실측, 출력 79개 글자까지 같음, #35). GBNF는 아래 표. 메모는 문장 수 N마다 만든다 |
| `--reasoning off` | 프롬프트 끝에 **`<think>\n\n</think>\n\n`을 직접 붙인다** | `llama_chat_apply_template`(chatml)은 `<\|im_start\|>assistant\n`에서 끝나고 이 블록을 안 붙인다. 렌더 전문 `rendered/2b-*.txt`와 글자 대조 |
| `max_tokens 1024` | 생성 루프 상한 1024토큰 | 질문 후보는 40자 문법이라 여기 닿지 않는다 |
| 프롬프트 캐시(서버가 자동) | 시스템 프롬프트를 decode한 직후 `llama_state_seq_get_data`로 상태를 떠 두고, 턴마다 `llama_state_seq_set_data`로 되돌린 뒤 사용자 메시지만 decode | **2B는 부분 삭제가 안 된다**(로그 "does not support partial sequence removal" — DeltaNet 상태). 1.7B처럼 `llama_memory_seq_rm`으로 턴 부분만 지우는 방식은 안 통한다. 이걸 안 하면 매 턴 시스템 프롬프트(약 2,000토큰)를 다시 처리한다 |

| 기능 | GBNF |
|---|---|
| 진료 전 추출 | `evals/ondevice/turn_extraction.small.gbnf` |
| 질문 후보 | `evals/ondevice/questions.gbnf` (text 40자) |
| 메모 분류 | `evals/ondevice/memo_labels-memo-small-v5.n4.gbnf`(N=4 예시) — 키 `"0"`…`"N-1"`이 순서대로 전부 나오게 N마다 같은 모양으로 만든다 |
| 재방문 표현 | `evals/ondevice/followup-v1.gbnf` |

## (이전) 묶음 v1 — Qwen3-1.7B (2026-09-07 1차 · 2026-09-09 진료 후 추가)

아래 v1 절들은 기록으로 둡니다. 일치 확인 절차·통합 시 알아둘 것은 v2에도 그대로 적용됩니다.

| 항목 | 값 / 위치 | 비고 |
|---|---|---|
| 엔진 | llama.cpp (Snapdragon 빌드, Hexagon 포함) | **llama.cpp 업스트림의** `scripts/snapdragon/build.py --target android` — 도커 툴체인 `ghcr.io/snapdragon-toolchain/arm64-android:v0.7`. 산출물 `pkg-android/llama.cpp/{bin,lib}`; lib에 `libggml-hexagon.so`, `libggml-htp-v75.so`(8 Gen 3) |
| 모델 | **Qwen3-1.7B-Q4_0.gguf** (1,056,782,912 bytes) | 받는 방법·해시는 아래 [모델 파일 받기](#모델-파일-받기). K-quant(Q4_K_M)는 NPU 오프로드가 안 될 수 있어 Q4_0 고정 |
| 프롬프트 | `extract-small-v4` — `evals/ondevice/prompt-small-v4.system.txt` | 시스템 프롬프트 전문. 사용자 메시지 형식은 벡터 파일의 `user` 참고 |
| 스키마 | `evals/ondevice/turn_extraction.small.schema.json` | 요청별 `response_format: json_schema`로 강제(서버 전역 `--json-schema-file`은 템플릿 토큰과 충돌해 사용 금지) |
| 추론 설정 | temperature 0 · seed 42 · max_tokens 1024 · reasoning off · `-c 4096 -t 6` | NPU: `--device HTP0 -ngl 99`, 환경 `LD_LIBRARY_PATH=<lib> ADSP_LIBRARY_PATH=<lib>` |
| 테스트 벡터 | `evals/ondevice/vectors-qwen3-1.7b-q4_0-small-v4.jsonl` (88) | 각 줄: `system`, `user`, `settings`, `expected_output`(PC에서 같은 설정으로 얻은 원문), `pc_latency_s` |

## 모델 파일 받기

**파일을 주고받을 필요가 없습니다.** 아래 URL로 직접 받고 해시만 맞추면 됩니다 —
AI 쪽이 eval을 돌린 파일과 **바이트 단위로 같은 것**임을 확인했습니다(2026-09-11).

```bash
curl -L -o Qwen3-1.7B-Q4_0.gguf "https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/d7f544eead698dbd1f15126ef60b45a1e1933222/Qwen3-1.7B-Q4_0.gguf"

sha256sum Qwen3-1.7B-Q4_0.gguf
# c876f159707a4e4f70e045106c69db15bfc935a4981706fd4f65c6e7ea1e81c5
```

윈도우: `Get-FileHash .\Qwen3-1.7B-Q4_0.gguf -Algorithm SHA256`
폰에 올릴 때: `adb push Qwen3-1.7B-Q4_0.gguf /data/local/tmp/`

| | 값 |
|---|---|
| SHA256 | `c876f159707a4e4f70e045106c69db15bfc935a4981706fd4f65c6e7ea1e81c5` |
| 크기 | 1,056,782,912 bytes |
| 저장소·커밋 | `unsloth/Qwen3-1.7B-GGUF` @ `d7f544eead698dbd1f15126ef60b45a1e1933222` |
| 라이선스 | Apache-2.0 (Qwen3) |

**URL을 `main`이 아니라 커밋으로 고정한 이유** — `main`으로 두면 업스트림이 파일을 갈아 끼울 때
앱이 다른 모델을 받고, 그러면 아래 88개 벡터 검증이 조용히 깨집니다. 커밋 고정 URL에서도
같은 해시가 오는 것을 확인했습니다.

**해시가 위 값과 같으면** AI 쪽 eval 결과(진료 전 추출 61/88 위반 0, 진료 후 메모 라벨 93%)가
그 기기에서도 성립하는 것으로 봅니다. 다르면 다른 파일이니 그 상태로 벡터 검증을 하지 마세요.

받는 경로는 Range 요청을 지원하므로(`accept-ranges: bytes`, CloudFront) 모바일에서 끊겨도
이어받을 수 있습니다. 구글 드라이브는 1GB 넘는 파일에 확인 페이지가 붙고 다운로드 쿼터 차단이
있어 앱 배포 경로로 쓰지 않는 게 좋습니다. 정식 배포 때는 Play Asset Delivery가 안드로이드
정석이지만(install-time 상한이 1GB라 `fast-follow`/`on-demand` 필요) 지금 단계에서 할 일은 아닙니다.

## 일치 확인 (안드로이드가 할 유일한 검증)

1. 위 설정으로 기기에서 88개 `user`를 순서대로 넣고 출력을 저장한다.
2. `expected_output`과 **JSON 파싱 후 비교**(공백·키 순서 무시). 온도 0·시드 고정이라 같은 GGUF·같은 엔진이면 대부분 일치한다.
3. 다르면 다른 줄의 `id`와 기기 출력을 AI 쪽에 보낸다. 하드웨어 수치 차이로 몇 건은 갈릴 수 있고(PC vs 폰 실측에서 45회 중 5건), 그건 AI 쪽 채점기로 안전 위반인지 판정한다.

이 절차만 통과하면 앱 안 모델과 AI 쪽 eval 결과가 같은 것이고, 이후 프롬프트·모델 갱신도 같은 방식이다.

## 통합 시 알아둘 것 (S24 울트라 실측)

- 첫 호출은 시스템 프롬프트 처리(2,241토큰)가 붙는다. **NPU 1.7초, CPU 14.9초.** 문진 화면을 열 때 시스템 프롬프트만 먼저 보내 캐시를 만들어 두면 첫 턴 체감이 사라진다(llama.cpp는 프롬프트 캐시를 자동으로 재사용).
- 턴당 지연 ≈ 0.2초 + 출력 토큰 ÷ 13. 출력 45토큰이면 3.4초, 25토큰이면 2초.
- CPU 연속 추론은 발열로 5분 안에 무너졿다(실측). **NPU 기본, CPU는 폴백.** 미지원 기기면 서버 추출로 폴백(백엔드 `profile: server`).
- 모델 파일은 앱 첫 실행 시 다운로드(1.0GB). **기준 해시는 아래에서 드립니다** — 저장 위치와 검증 실행은 앱 몫.

## 앱 흐름에서 모델이 받는 것 (화면·문구 결정은 이슈 #14)

- 인체도 탭 목록은 `GET /v1/ontology/body-map`으로 만든다(백엔드 경유). 앵커·구역마다 **`image_key`**(예 `head`, `lower-back-and-hip`)가 있으니 이미지 파일명을 이 키로 맞추면 id→파일 매핑을 앱에 하드코딩하지 않는다. 이미지 자체는 앱 자산. 응답 예시 전체: `docs/examples/body-map.json`
- 부위를 폼(검색)으로 입력받는 화면은 body-map의 `label` + `aliases`로 로컬 매칭한다(공백 제거 후 포함 검색). 복부→배, 옆구리→허리 옆, 뒷목→목 뒤·옆. 서버 `GET /v1/ontology/search?q=`도 있는데 **2026-09-10부터 규칙이 같지 않다**(PR #59): 서버는 ①한/영 자판 오타를 복원하고(`qo`→배, `duvrnfl`→옆구리) ②앵커가 걸리면 그 아래 구역을 `score: 0`으로 딸려 보낸다(`q=다리` → 무릎·종아리·발목·발). 한글 오타(무릅)는 양쪽 다 안 잡는다. 로컬 매칭만 쓰면 이 둘이 빠지고, 서버를 쓰면 `score: 0` 항목을 화면에서 어떻게 다룰지 정해야 한다. 한 글자마다 호출해도 서버는 문제없다(약 0.2ms). 단 한글 조합 중간(`옆굴`, `아랫ㅂ`)은 0건으로 오니 직전 결과를 유지하는 쪽이 좋다. **2026-09-11 결정: 앱은 로컬로 가고 한/영 오타 복원은 넣지 않는다.** 규칙·데이터·검증 벡터는 `docs/android-body-search.md`에 따로 정리했다
- 인체도로 부위 선택 → 백엔드 세션 시작(`site_node_id`, `profile: ondevice`) → 첫 축 질문부터.
- **문답은 전부 자유 입력이다. 칩(선택지)은 없다.** 전에 이 자리에 "선택지로 답하면 모델을 부르지 않고
  `selections`로 보낸다"고 적혀 있었으나 **구현된 적이 없다** — 턴 응답에 나가는 것은 `reply` 문자열
  하나뿐이고 선택지 목록을 내보내는 자리가 없다. 칩을 넣을지는 화면 결정이므로 #14에서 정한다
  (2026-09-11 확인: 없는 것이 맞는 설계다).
- 그래서 **모든 발화가 폰 모델에 들어간다.** 결과 JSON을 `extraction`으로, 근거 검증용 원문을
  `utterance`로 함께 백엔드에 보낸다. 발화 원문은 폰을 떠나지 않는다.
- `selections`의 용처는 **통증 강도 슬라이더(화면 `1d`, 문답 전) 하나뿐**이고 그 값은 백엔드가
  문답이 끝난 뒤 질문 후보 요청에 실어 보낸다.
  챗 화면의 칩이 아니다. **NRS가 아니라 1~5 서열척도 + 단계별 라벨**이다(`{level: 3, label: "꽤 아파요"}`).
- 수정은 "이전 답 고치기" 버튼으로 해당 축을 다시 묻는다. 모델이 이력을 읽고 고치는 방식은 쓰지 않는다(소형 모델 0/8).

API 필드 정의: `docs/api-previsit.md` "온디바이스 프로필 — 예정". 백엔드 쪽 동일 내용: 이슈 #7.

## 진료 후 메모 분류 (2026-09-09 결정 — 이슈 #35)

같은 엔진·같은 모델 파일에 **프롬프트와 스키마만 하나씩 더** 싣습니다. 폰은 문장에 라벨만 붙이고, 분리·조립·날짜 계산은 서버가 합니다.

| 항목 | 값 / 위치 | 비고 |
|---|---|---|
| 모델·엔진·추론 설정 | 위 표와 동일 (Qwen3-1.7B-Q4_0, NPU, temperature 0 · seed 42 · reasoning off) | 추가 다운로드 없음 |
| 프롬프트 | `memo-small-v4` — `evals/ondevice/prompt-memo-small-v4.system.txt` | 시스템 프롬프트 전문(약 1,000토큰). 사용자 메시지 형식은 벡터의 `user` |
| 스키마 | `evals/ondevice/memo_labels.schema.json` | **번호 키 객체** `{"0":라벨,"1":라벨,…}`. 파일은 N=4 예시이고, **요청마다 문장 수 N에 맞춰 키 "0"…"N-1"을 전부 required로 생성**해 `response_format`으로 넘긴다. 라벨 enum 5개: findings / tests / medication_instructions / follow_up / none |
| 테스트 벡터 | `evals/ondevice/vectors-memo-qwen3-1.7b-q4_0-memo-small-v4.jsonl` (34) | 각 줄: `system`, `user`, `sentences`, `settings`, `expected_output`(PC 원문), `expected_labels`(사람 정답), `pc_latency_s`. PC 라벨 정확도 **123/132 = 93.2%**(2026-09-14, `split-v3`. PM24 추가로 34개) |

흐름 (`docs/api-postvisit.md`):

1. 앱이 메모 원문을 백엔드 → AI 서버 `POST /v1/postvisit/memo`에 **`classify: false`**로 보낸다. 서버는 LLM을 부르지 않고 `sentences`(번호 붙은 문장 목록)만 돌려준다. **폰이 문장을 나누지 않는다** — 폰·서버가 같은 번호를 가리키게 하는 규칙.
2. 폰이 `sentences`로 `user` 메시지를 만들어(벡터 `user` 형식 그대로) 모델을 부른다. 출력은 번호 키 객체.
3. 그 객체를 `labels`로 넣어 같은 엔드포인트를 다시 부른다. AI 서버가 LLM 없이 카드를 조립하고 재방문 날짜를 계산한다.
4. 폰 모델이 실패하면(미로드·시간 초과) `classify` 없이 같은 엔드포인트를 부른다. 서버가 분류한다. 앱은 실패 판단만 한다.
   **서버 모델은 지금 Nova Pro다**(`MEDIMATE_MODEL`). 진료 전 추출 88케이스 기준 Nova 76/88 · Terra 85 · Sonnet 83,
   안전 위반은 셋 다 0이다(2026-09-14, `evals/RESULTS.md`). 폴백 품질이 폰보다 낮아지는 구간은 아니다.

일치 확인은 위 88개와 같은 방식으로 30개 `user`를 넣고 `expected_output`과 JSON 비교. 지연은 첫 호출에 시스템 프롬프트 처리가 붙고(NPU 약 1초), 문장 5개 메모의 출력은 약 30토큰이라 턴당 2~3초 예상.

## 후보 모델 폰 실측 요청 — Qwen3.5-2B (2026-10-13)

**앱 코드는 바꾸지 않습니다.** 위 묶음(1.7B)이 계속 기준이고, 이번 것은 adb로 폰에서 재 보는 실측입니다.
결과가 좋으면 그때 묶음 버전을 올려 다시 전달합니다.

왜: PC에서 재 보니 2B가 **질문 후보를 훨씬 잘 만듭니다**(서버 검증기를 통과한 모델 질문 46 → 87,
템플릿만으로 채운 카드 17 → 8). 진료 전 추출은 앱 경로 기준 1.7B와 같습니다(86문항 중 68 · 68,
안전 위반 0). 남은 질문은 **폰 NPU에서 도는가**뿐입니다 — Qwen3.5는 DeltaNet 혼합 구조라 9/7에 "NPU에서
CPU로 떨어질 수 있다"며 뺐는데, 넘겨드린 엔진 빌드(`pkg-android-2092353`)에 그 연산의 Hexagon 커널이 들어 있습니다.

### 받을 것

| | 값 |
|---|---|
| 모델 | `Qwen3.5-2B-Q4_0-ssmout-q8.gguf` · 1,243,185,536 bytes · Apache-2.0 (2차, 아래 "2차 실측" 참고) |
| 받는 곳 | `https://huggingface.co/hyos0415/Qwen3.5-2B-NPU-GGUF/resolve/1c069ab6a03d5840d81ad29b2476752a13a407c0/Qwen3.5-2B-Q4_0-ssmout-q8.gguf` (계정·토큰 없이 받음, 이어받기 됨) |
| SHA256 | `c8e057dfc224b87b2fce69690ee6b53ca828f67543d01d088e409cb8315dc53f` |
| 엔진 | 그대로 `pkg-android-2092353` (새로 받을 것 없음) |
| 추출 프롬프트 | `evals/ondevice/prompt-extract-small-2b-v1.system.txt` (1.7B의 small-v4에서 값 규칙 한 줄과 예시 둘만 다름) |
| 추출 벡터 | `evals/ondevice/vectors-qwen3.5-2b-q4_0-ssmq8-extract-small-2b-v1.jsonl` (79개, `app_path: true` 65개) |
| 질문 후보 벡터 | `evals/ondevice/vectors-questions-qwen3.5-2b-q4_0-ssmq8-questions-small4.jsonl` (100개, `messages` 그대로 보냄) · 스키마 `evals/ondevice/questions.schema.json` · GBNF `questions.gbnf` |
| 질문 스키마 상한 | **`text` 40자 · `source` 24자** (2026-10-13 2차 실측 PC22: 2B가 "석 달쯤"을 되풀이하다 max_tokens에 걸려 JSON이 깨지고 66초). 40자는 서버 질문 검증기 상한과 같아 넘는 문장은 어차피 버려진다. 문법이 문자열을 닫게 해 폭주가 생길 수 없다. PC 100장: 파싱 실패 2 → 0, 검증 통과 170 → 171 |
| 추론 설정 | 1.7B와 같음 — temperature 0 · seed 42 · max_tokens 1024 · **reasoning off** · `-c 4096` · NPU `--device HTP0 -ngl 99` |

`expected_output`은 PC(GPU)에서 낸 것이라 폰과 바이트가 같을 필요는 없습니다. 판정은 AI 쪽 채점기가 합니다.

**벡터 읽는 법 — 폰에서 무엇이 나와야 하는가**

- **프롬프트가 먼저 같아야 합니다.** 모델에 들어가는 전문은 `evals/ondevice/rendered/2b-A01-0.enable_thinking_false.txt`와
  글자까지 같아야 합니다(끝이 `<|im_start|>assistant
<think>

</think>

` — 생각 끄기). 템플릿 원문은
  `evals/ondevice/qwen3.5-2b.chat_template.jinja`. 여기가 다르면 아래 숫자는 의미가 없습니다
- **출력은 JSON 하나**(스키마 강제). `<think>` 내용이나 설명문이 섞이면 실패입니다
- **줄마다 `expect`가 판정 기준**입니다. 예: `{"axes": {"onset": "filled"}, "value_must_contain": {"onset": "4일"}}` —
  물은 칸이 채워지고 값에 "4일"이 들어 있으면 통과. 바이트가 PC와 달라도 `expect`를 지키면 통과입니다.
  `category`·`purpose`는 그 케이스가 무엇을 보는지(경계값·고쳐 말하기·채팅체·인젝션 등)
- **`app_path: false`는 첫 발화 케이스**라 앱 흐름(축 질문부터)에는 없습니다. 참고용
- 질문 후보 벡터는 정답이 하나가 아닙니다. 서버 검증기가 거르고 템플릿이 채우므로, 폰에서는 **JSON이 나오고 3~5개**인지만 봅니다

### 2차 실측 (2026-10-13) — 1차 결과와 바꾼 것

1차(unsloth `Qwen3.5-2B-Q4_0.gguf`, 이슈 #35): 프롬프트 렌더 일치, 추출 품질 폰 = PC(앱 경로 65문항 중 53, 위반 0),
앱 경로 턴 지연 p50 3.64초 · p90 4.67초, 생성 12.7 tok/s. 대신 **graph splits 39**(1.7B 3), 첫 호출 9.0초,
긴 프롬프트 360 tok/s(1.7B 1,283).

원인: unsloth의 "Q4_0"은 혼합 양자화라 `ssm_out` 18개(DeltaNet 층마다 하나)가 **Q5_K**였고, Hexagon은 Q5_K를
계산하지 않아 층마다 CPU로 갔다 왔습니다(CPU 쪽 가중치 +49.5MiB = 그 18개 크기). 그 18개만 BF16 원본에서
**Q8_0**으로 바꿔 끼웠습니다(나머지 302개 텐서·메타데이터는 unsloth와 바이트까지 같음,
`scripts/gguf_swap_tensor_type.py`). PC 품질은 그대로입니다(108문항 77 → 78, 위반 0). 채팅 템플릿도 같아서
렌더 전문은 그대로 맞습니다.

2차로 보고 싶은 것: **graph splits가 3 근처로 내려오는지**, 첫 호출·긴 프롬프트 속도, 5분 발열. 명령은 아래와 같고
모델 파일 이름만 바뀌었습니다. 그리고 발열은 조건이 달라 1.7B와 비교할 수가 없어서(9/7은 선풍기, 1차는 USB 충전 중),
**1.7B도 같은 조건으로 5분** 한 번 부탁드립니다:

```bash
python scripts/ondevice_device_run.py --vectors evals/ondevice/vectors-qwen3-1.7b-q4_0-small-v4.jsonl --out device-17-heat.jsonl --minutes 5 --adb-temp
```

### 재는 법 (PC + USB, 30분 정도)

```bash
adb push Qwen3.5-2B-Q4_0-ssmout-q8.gguf /data/local/tmp/llama.cpp/
# 폰에서 서버 — 1.7B 때와 같은 명령, 모델 파일만 다름. VERBOSE로 연산 배치가 로그에 남는다
adb shell "cd /data/local/tmp/llama.cpp && GGML_HEXAGON_VERBOSE=1 LD_LIBRARY_PATH=lib ADSP_LIBRARY_PATH=lib   bin/llama-server -m Qwen3.5-2B-Q4_0-ssmout-q8.gguf --port 8080 -c 4096 --temp 0 --seed 42 --reasoning off   --device HTP0 -ngl 99 -np 1" 2>&1 | tee server-2b.log
adb forward tcp:8080 tcp:8080

# PC에서 (파이썬 표준 라이브러리만, 저장소 환경 불필요)
python scripts/ondevice_device_run.py --vectors evals/ondevice/vectors-qwen3.5-2b-q4_0-ssmq8-extract-small-2b-v1.jsonl --out device-2b-extract.jsonl --adb-temp
python scripts/ondevice_device_run.py --vectors evals/ondevice/vectors-questions-qwen3.5-2b-q4_0-ssmq8-questions-small4.jsonl --out device-2b-questions.jsonl --adb-temp --limit 30
python scripts/ondevice_device_run.py --vectors evals/ondevice/vectors-qwen3.5-2b-q4_0-ssmq8-extract-small-2b-v1.jsonl --out device-2b-heat.jsonl --minutes 5 --adb-temp
```

### 보고 싶은 것 넷 (1.7B 9/7 실측과 나란히 봅니다)

| | 1.7B NPU (9/7) | 2B에서 확인 |
|---|---|---|
| NPU에서 도는가 | 전부 HTP0 | `server-2b.log`에서 CPU로 떨어진 연산이 있는지(특히 `GATED_DELTA_NET` · `SSM_CONV`) |
| 첫 호출 · 턴 지연 p50 | 1.7초 · 3.4초 | 스크립트 요약 줄 |
| 생성 속도 | 12~13 tok/s | 스크립트 요약 줄 |
| 5분 연속 발열 | 34.9°C 안정 | `--minutes 5` 요약 줄(배터리 온도) |

**올려 주실 것**: `device-2b-*.jsonl` 셋과 `server-2b.log`(앞 200줄이면 충분) → 이슈 #35.
ADSP_LIBRARY_PATH를 빠뜨리면 에러 없이 CPU로 돕니다 — 첫 호출이 10초 넘게 걸리면 이것부터 확인해 주세요.

## 연락

- 벡터 불일치, 빌드 실패, 지연 이상은 이슈 #35에 코멘트. AI 쪽이 재현·판정한다.
- 이 묶음의 버전: **v2** — 모델 Qwen3.5-2B-Q4_0-ssmout-q8 · 진료 전 extract-small-2b-v1 · 질문 questions-small4 + 40자 스키마 · 진료 후 memo-small-v5 · 재방문 followup-v1 (2026-10-13). 가드 규칙은 서버 쪽(앱 무관).
- 이전: v1 — Qwen3-1.7B-Q4_0 · extract-small-v4 · memo-small-v4 (2026-09-07 · 09-09).
