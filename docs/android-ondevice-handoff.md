# 안드로이드 전달 — 온디바이스 추출 묶음 (2026-09-07 1차 · 2026-09-09 진료 후 추가)

AI 쪽 실험은 끝났고(이슈 #24), 안드로이드에는 **고정된 묶음**과 **일치 확인 절차**만 넘깁니다.
모델·프롬프트·스키마를 앱에서 바꾸지 않습니다. 바뀌면 AI 쪽이 버전을 올려 다시 전달합니다.

## 묶음

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
- `selections`의 용처는 **3단계 통증 강도 슬라이더(화면 `1d`) 하나뿐**이고 그 값은 백엔드가 보낸다.
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
4. 폰 모델이 실패하면(미로드·시간 초과) `classify` 없이 같은 엔드포인트를 부른다. 서버가 Terra로 분류한다. 앱은 실패 판단만 한다.

일치 확인은 위 88개와 같은 방식으로 30개 `user`를 넣고 `expected_output`과 JSON 비교. 지연은 첫 호출에 시스템 프롬프트 처리가 붙고(NPU 약 1초), 문장 5개 메모의 출력은 약 30토큰이라 턴당 2~3초 예상.

## 연락

- 벡터 불일치, 빌드 실패, 지연 이상은 이슈 #35에 코멘트. AI 쪽이 재현·판정한다.
- 이 묶음의 버전: 모델 Qwen3-1.7B-Q4_0 · 진료 전 프롬프트 extract-small-v4 · 스키마 small(2026-09-07) · 진료 후 프롬프트 memo-small-v4 · 스키마 번호 키(2026-09-09) · 가드 규칙은 서버 쪽(앱 무관).
