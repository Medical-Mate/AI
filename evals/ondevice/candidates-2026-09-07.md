# 온디바이스 초경량 LLM 후보 (갤럭시 S24 울트라 · llama.cpp · GGUF)

조사일: 2026-09-07. 용도: 한국어 300자 이내 발화 → 고정 JSON 스키마 추출(생성·추론 불필요, 원문 구간 글자 그대로 복사가 핵심).
조건: 0.3B~2.5B, Q4 GGUF 1.5GB 이하 목표, 2026-03-07 이후 출시 우선.
"미확인"은 WebSearch/WebFetch로 확인하지 못한 항목. 추정치는 "(추정)"으로 표기.

## 1. 표

### A. 6개월 조건 충족 (2026-03-07 이후 출시)

| 모델 | 출시일 | 파라미터 | Q4 GGUF 크기(대략) | GGUF 출처 | llama.cpp 지원 | 한국어 | 라이선스(원문) | 구조화 출력 관련 특징 | 근거 URL |
|---|---|---|---|---|---|---|---|---|---|
| LFM2.5-230M (Liquid AI) | 2026-06-25 | 0.23B | Q4_K_M ≈ 153MB, Q8_0 ≈ 247MB | 공식 https://huggingface.co/LiquidAI/LFM2.5-230M-GGUF | 지원(공식 "day-one llama.cpp") | 공식 10개 언어 목록에 Korean 포함 | LFM1.0 (LFM Open License v1.0, 상세 조건 미확인) | 함수 호출 학습(기본은 Pythonic 형식, 시스템 프롬프트로 JSON 요구 가능). 공식 블로그가 "data extraction" 특화를 표방, BFCLv4 21.03. 추론·코딩은 비권장으로 명시 | https://huggingface.co/LiquidAI/LFM2.5-230M , https://www.liquid.ai/blog/lfm2-5-230m |
| MiniCPM5-1B (OpenBMB) | 2026-05-19 | 1.08B | Q4_K_M ≈ 688MB | 공식 https://huggingface.co/openbmb/MiniCPM5-1B-GGUF | 지원(표준 LlamaForCausalLM, "vanilla llama.cpp") | 미확인 — 공식 문서는 English·Chinese만 명시 | Apache-2.0 | XML 스타일 툴콜 학습, think/no-think 모드, 131K 컨텍스트 | https://huggingface.co/openbmb/MiniCPM5-1B , https://github.com/openbmb/minicpm |
| Kanana-2-1.3B-instruct (카카오) | 2026-07-27~28 | 1.3B | Q8_0 = 1.38GB(커뮤니티) → Q4_K_M ≈ 0.8GB(추정) | 공식 GGUF 없음. 커뮤니티 https://huggingface.co/dummy9996/kanana-2-1.3b-instruct-GGUF (README 비어 있음) | 부분 확인 — 커뮤니티 GGUF가 qwen3 아키텍처로 변환됨. 원 모델은 Qwen3 백본 + 3:1 SWA/풀어텐션 혼합 구조로, SWA 동작이 llama.cpp에서 그대로 재현되는지 미확인 | 공식 한국어·영어. KMMLU 42.79, HAE-RAE 44.89. 한국어 토크나이저 효율 30%↑ | Kanana Open License (상업 이용 허용 명시) | 툴콜 학습(qwen3_coder 파서), BFCL-v3 Live 69.64 pass@1 | https://huggingface.co/kakaocorp/kanana-2-1.3b-instruct , https://www.kakaocorp.com/page/detail/12089 |
| LFM2.5-2.6B (Liquid AI) | 2026-08-04 | 2.6B | IQ4_XS 1.52GB, Q4_K_S 1.61GB, Q4_K_M 1.68GB — 1.5GB 목표 약간 초과 | 공식 https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF , bartowski https://huggingface.co/bartowski/LiquidAI_LFM2.5-2.6B-GGUF | 지원(day-one) | 공식 16개 언어 목록에 Korean 포함 | LFM1.0 | 함수 호출 학습(Pythonic 기본, JSON 요구 가능), 128K, think 단계 있음 | https://huggingface.co/LiquidAI/LFM2.5-2.6B , https://www.liquid.ai/blog/lfm2-5-2-6b |
| Gemma 4 E2B-it (Google) | 2026-04-02 | 2.3B 유효(임베딩 포함 5.1B) | Q4_K_M ≈ 3.1GB — 목표 초과(PLE 임베딩이 양자화되지 않음) | 공식 ggml-org, unsloth https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF | 지원(출시일 포함, 4/11 템플릿 픽스) | 140+ 언어 사전학습, 35+ 언어 즉시 지원. 모델 카드 언어 목록에 Korean 포함 | Apache 2.0 | "native support for structured tool use" 명시. 멀티모달(텍스트·이미지·오디오) | https://huggingface.co/google/gemma-4-E2B-it , https://ai.google.dev/gemma/docs/core/model_card_4 |

### B. 참고 (2026-03-07 이전 출시)

| 모델 | 출시일 | 파라미터 | Q4 GGUF 크기(대략) | GGUF 출처 | llama.cpp 지원 | 한국어 | 라이선스(원문) | 구조화 출력 관련 특징 | 근거 URL |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B (Alibaba) | 2026-03-02 (기준일 5일 전) | 0.8B | Q4_K_M 533MB, Q4_K_XL 559MB | unsloth https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF (공식 GGUF 미확인) | 지원(공식 문서 명시) | 201개 언어·방언 공식 표기. 한국어 개별 언급은 미확인 | Apache 2.0 | "excels in tool calling", think/no-think, 비전-언어 통합 모델 | https://huggingface.co/Qwen/Qwen3.5-0.8B , https://github.com/QwenLM/Qwen3.8 |
| Qwen3.5-2B (Alibaba) | 2026-03-02 | 2B | Q4_K_M 1.28GB, Q4_K_XL 1.34GB | unsloth https://huggingface.co/unsloth/Qwen3.5-2B-GGUF | 지원 | 201개 언어 표기 | Apache 2.0 | 툴콜 학습(qwen3_coder 파서) | https://github.com/QwenLM/Qwen3.8 |
| LFM2.5-1.2B-Instruct (Liquid AI) | 2026-01-05 | 1.17B | Q4_K_M 696MB (QAD Q4_0도 제공) | 공식 https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF | 지원 | 공식 8개 언어 목록에 Korean 포함 | LFM1.0 | 함수 호출 학습(Pythonic 기본, JSON 요구 가능). 8월 DSpark 드래프트 모델 추가 | https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct , https://www.liquid.ai/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai |
| Granite-4.0-1B (IBM) | 2025-10-28 | 1.6B(dense) / H-1B 하이브리드 1.5B | 미확인(≈1GB 추정) | 공식 GGUF 미확인. 커뮤니티 다수(HF "Browse Quantizations") | 지원(dense 변형이 llama.cpp 용도로 별도 제공됨) | 공식 12개 언어 목록에 Korean 포함 | Apache 2.0 | 툴콜 강화, OpenAI 함수 스키마 JSON 응답 예시 제공, BFCL-v3 | https://huggingface.co/ibm-granite/granite-4.0-1b , https://huggingface.co/blog/ibm-granite/granite-4-nano |
| EXAONE-4.0-1.2B (LG) | 2025-07-15 | 1.2B | Q4_K_M 812MB | 공식 https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B-GGUF | 지원("officially supported by llama.cpp") | 공식 영어·한국어·스페인어 | EXAONE AI Model License Agreement 1.2 - NC (비상업) | agentic tool use, BFCL-v3·Tau-Bench 평가 | https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B-GGUF |
| Midm-2.0-Mini-Instruct (KT) | 2025-07-04 | 2.3B | 미확인(≈1.4GB 추정) | 공식 GGUF 미확인. 커뮤니티 | 지원 여부 미확인(모델 카드가 llama.cpp 양자화 언급) | 공식 한국어·영어 | MIT License | 함수 호출은 2025-10-29 vLLM 전용 파서로 추가 | https://huggingface.co/K-intelligence/Midm-2.0-Mini-Instruct |
| HyperCLOVAX-SEED-Text-Instruct-0.5B / 1.5B (NAVER) | 2025-04-24 | 0.57B / 1.5B | 미확인(1.5B Q4 ≈ 1GB 추정) | 공식 없음. naver-ellm(네이버 계열 추정) https://huggingface.co/naver-ellm/HyperCLOVAX-SEED-Text-Instruct-1.5B-GGUF , 커뮤니티 rippertnt 등 | 커뮤니티 GGUF 존재로 간접 확인. 공식 아키텍처 등록 여부 미확인 | 한국어 특화. KMMLU 0.39, KoBEST 0.65(1.5B) | HyperCLOVA X SEED Model License (MAU 1천만 초과·경쟁 서비스 시 별도 라이선스) | 툴콜·JSON 학습 언급 미확인 | https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B |
| kanana-nano-2.1b-instruct (카카오) | 2025-02-27 | 2.1B | 미확인(≈1.3GB 추정) | 커뮤니티(mradermacher, DevQuasar 등) | 지원(LLaMA 아키텍처) | 한국어·영어 | CC-BY-NC-4.0 (비상업) | 함수 호출 별도 변형 언급 | https://huggingface.co/kakaocorp/kanana-nano-2.1b-instruct |
| Qwen3-0.6B / 1.7B (Alibaba) | 2025-04-29 | 0.6B / 1.7B | 미확인(≈0.4GB / ≈1.1GB 추정) | 공식 ggml-org https://huggingface.co/ggml-org/Qwen3-0.6B-GGUF , https://huggingface.co/ggml-org/Qwen3-1.7B-GGUF | 지원 | 100+ 언어 표기 | Apache 2.0 | 툴콜 학습(Qwen-Agent) | https://huggingface.co/Qwen/Qwen3-1.7B |
| Gemma 3n E2B-it (Google) | 2025-06 | 2B 유효(총 ~5B) | Q4_K_M 2.79GB — 목표 초과 | 공식 ggml-org, bartowski | 지원 | 140개 언어(모델 카드 기준) | Gemma Terms of Use | 함수 호출 지원 | https://huggingface.co/bartowski/google_gemma-3n-E2B-it-GGUF |
| Tri-1.9B-Base (Trillion Labs) | 2025-08 | 1.9B | 해당 없음 | GGUF 미확인 | 미확인 | 공식 영어·한국어·일본어·중국어 | Apache 2.0 | instruct 변형 없음(base only) → 추출 용도 부적합 | https://huggingface.co/trillionlabs/Tri-1.9B-Base |

### 조사했으나 조건 밖(참고만)
- Qwen3.6 / Qwen3.8 (2026-04 / 2026-08): 27B·35B-A3B·177B(Flash-Next)만 공개. 3B 이하 신형 없음. https://github.com/QwenLM/Qwen3.8
- EXAONE 4.5 (2026-04-09): 33B VLM만. 소형 신형 없음. "EXAONE 4.1" 존재 미확인. https://github.com/LG-AI-EXAONE/EXAONE-4.5
- SKT A.X: A.X-K1 (2026-01, 519B MoE, Apache 2.0)만 신형. A.X-4.0-Light는 7B로 범위 밖. https://huggingface.co/skt/A.X-K1
- HyperCLOVA X SEED 2026 신형은 Think-14B/32B, Omni-8B — 범위 밖.
- Kanana-2-0.9B-instruct: 카카오 보도자료 요약에 언급되나 HF 조직 페이지에 없음(401). 미확인.
- SmolLM3-3B (2025-07): 3B로 범위 밖, SmolLM4 미확인. Phi-4-mini 3.8B: 범위 밖, Phi-5 미확인.
- Ministral 3 (3B+), Nemotron Nano (4B+): 범위 밖. HRM-Text-1B (2026-05): 영어 전용·instruct 아님.
- Gemma 4 소형은 E2B가 최소. 1B/270M급 신형 미확인.

## 2. 6개월 조건 그룹 요약
- 충족(2026-03-07 이후): LFM2.5-230M, MiniCPM5-1B, Kanana-2-1.3B-instruct, LFM2.5-2.6B(크기 약간 초과), Gemma 4 E2B(크기 초과)
- 참고(이전): Qwen3.5-0.8B/2B(5일 차이), LFM2.5-1.2B, Granite-4.0-1B, EXAONE-4.0-1.2B, Midm-2.0-Mini, HyperCLOVAX-SEED 0.5B/1.5B, kanana-nano-2.1b, Qwen3-0.6B/1.7B, Gemma 3n E2B, Tri-1.9B-Base

## 3. 미확인 목록(정직 표기)
- Kanana-2-1.3B: 공식 GGUF, llama.cpp에서의 SWA 재현 여부, 0.9B 변형 존재
- MiniCPM5-1B: 한국어 지원(공식은 영·중만)
- HyperCLOVAX-SEED-Text: llama.cpp 공식 아키텍처 지원, Q4 크기
- Granite-4.0-1B, Midm-2.0-Mini, kanana-nano, Qwen3-0.6B/1.7B: 정확한 Q4 파일 크기
- LFM1.0 라이선스의 상업 이용 조건 세부
- Qwen3.5의 한국어 개별 벤치마크
- 모든 모델 공통: "원문 구간 글자 그대로 복사" 능력은 벤치마크 부재 → 자체 eval(D1~D10)로만 검증 가능

## 4. 요약 5줄 — 추천 사다리(작은 것부터)
1. LFM2.5-230M (153MB, 2026-06) — 한국어 공식 지원 + 공급사가 "데이터 추출" 용도를 직접 표방, 함수 호출 학습. 가장 싸게 "JSON 스키마 채우기가 0.2B로도 되는가"를 판정하는 바닥선.
2. Qwen3.5-0.8B (533MB, 2026-03-02, 참고 그룹이지만 5일 차) — 201개 언어·툴콜 학습·Apache 2.0·llama.cpp 공식. 신형 사다리에서 사실상 기본 비교 대상.
3. Kanana-2-1.3B-instruct (Q4 ≈ 0.8GB 추정, 2026-07) — 유일한 6개월 이내 한국어 특화 소형. 상업 허용 라이선스·BFCL 69.6. 단 GGUF가 커뮤니티 1개(qwen3 아키텍처 변환)라 SWA 재현 검증이 선행 과제.
4. LFM2.5-2.6B IQ4_XS (1.52GB, 2026-08) 또는 Qwen3.5-2B Q4_K_M (1.28GB) — 1.5GB 상한에서 최대 성능 후보. 한국어 명시는 LFM2.5-2.6B가 확실, 크기 준수는 Qwen3.5-2B.
5. Gemma 4 E2B는 한국어·툴콜·Apache 2.0 모두 좋지만 Q4가 3.1GB로 목표 2배 → 12GB RAM에서는 돌아가므로 "상한 재검토 시" 후보. EXAONE-4.0-1.2B는 NC 라이선스라 서비스 배포 부적합.
