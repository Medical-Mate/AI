# 안드로이드 전달 — 온디바이스 추출 묶음 (2026-09-07, 1차)

AI 쪽 실험은 끝났고(이슈 #24), 안드로이드에는 **고정된 묶음**과 **일치 확인 절차**만 넘깁니다.
모델·프롬프트·스키마를 앱에서 바꾸지 않습니다. 바뀌면 AI 쪽이 버전을 올려 다시 전달합니다.

## 묶음

| 항목 | 값 / 위치 | 비고 |
|---|---|---|
| 엔진 | llama.cpp (Snapdragon 빌드, Hexagon 포함) | `scripts/snapdragon/build.py --target android` — 도커 툴체인 `ghcr.io/snapdragon-toolchain/arm64-android:v0.7`. 산출물 `pkg-android/llama.cpp/{bin,lib}`; lib에 `libggml-hexagon.so`, `libggml-htp-v75.so`(8 Gen 3) |
| 모델 | **Qwen3-1.7B-Q4_0.gguf** (1.0GB) | `unsloth/Qwen3-1.7B-GGUF` 의 Q4_0. K-quant(Q4_K_M)는 NPU 오프로드가 안 될 수 있어 Q4_0 고정 |
| 프롬프트 | `extract-small-v4` — `evals/ondevice/prompt-small-v4.system.txt` | 시스템 프롬프트 전문. 사용자 메시지 형식은 벡터 파일의 `user` 참고 |
| 스키마 | `evals/ondevice/turn_extraction.small.schema.json` | 요청별 `response_format: json_schema`로 강제(서버 전역 `--json-schema-file`은 템플릿 토큰과 충돌해 사용 금지) |
| 추론 설정 | temperature 0 · seed 42 · max_tokens 1024 · reasoning off · `-c 4096 -t 6` | NPU: `--device HTP0 -ngl 99`, 환경 `LD_LIBRARY_PATH=<lib> ADSP_LIBRARY_PATH=<lib>` |
| 테스트 벡터 | `evals/ondevice/vectors-qwen3-1.7b-q4_0-small-v4.jsonl` (88) | 각 줄: `system`, `user`, `settings`, `expected_output`(PC에서 같은 설정으로 얻은 원문), `pc_latency_s` |

## 일치 확인 (안드로이드가 할 유일한 검증)

1. 위 설정으로 기기에서 88개 `user`를 순서대로 넣고 출력을 저장한다.
2. `expected_output`과 **JSON 파싱 후 비교**(공백·키 순서 무시). 온도 0·시드 고정이라 같은 GGUF·같은 엔진이면 대부분 일치한다.
3. 다르면 다른 줄의 `id`와 기기 출력을 AI 쪽에 보낸다. 하드웨어 수치 차이로 몇 건은 갈릴 수 있고(PC vs 폰 실측에서 45회 중 5건), 그건 AI 쪽 채점기로 안전 위반인지 판정한다.

이 절차만 통과하면 앱 안 모델과 AI 쪽 eval 결과가 같은 것이고, 이후 프롬프트·모델 갱신도 같은 방식이다.

## 통합 시 알아둘 것 (S24 울트라 실측)

- 첫 호출은 시스템 프롬프트 처리(2,241토큰)가 붙는다. **NPU 1.7초, CPU 14.9초.** 문진 화면을 열 때 시스템 프롬프트만 먼저 보내 캐시를 만들어 두면 첫 턴 체감이 사라진다(llama.cpp는 프롬프트 캐시를 자동으로 재사용).
- 턴당 지연 ≈ 0.2초 + 출력 토큰 ÷ 13. 출력 45토큰이면 3.4초, 25토큰이면 2초.
- CPU 연속 추론은 발열로 5분 안에 무너졿다(실측). **NPU 기본, CPU는 폴백.** 미지원 기기면 서버 추출로 폴백(백엔드 `profile: server`).
- 모델 파일은 앱 첫 실행 시 다운로드(1.0GB). 저장 위치와 무결성 확인(해시)은 앱 몫.

## 앱 흐름에서 모델이 받는 것 (UX 결정 대기 — 이슈 #24)

- 인체도로 부위 선택 → 백엔드 세션 시작(`site_node_id`, `profile: ondevice`) → 첫 축 질문부터.
- 선택지(칩)로 답하면 모델을 부르지 않고 `selections`로 백엔드에 보낸다.
- "직접 입력"만 폰 모델에 넣고, 결과 JSON을 `extraction`으로 백엔드에 보낸다. 발화 원문은 폰을 떠나지 않는다.
- 수정은 "이전 답 고치기" 버튼으로 해당 축을 다시 묻는다. 모델이 이력을 읽고 고치는 방식은 쓰지 않는다(소형 모델 0/8).

API 필드 정의: `docs/api-previsit.md` "온디바이스 프로필 — 예정". 백엔드 쪽 동일 내용: 이슈 #7.

## 연락

- 벡터 불일치, 빌드 실패, 지연 이상은 이슈 #24에 코멘트. AI 쪽이 재현·판정한다.
- 이 묶음의 버전: 모델 Qwen3-1.7B-Q4_0 · 프롬프트 extract-small-v4 · 스키마 small(2026-09-07) · 가드 규칙은 서버 쪽(앱 무관).
