# 진료 메이트 — AI

환자는 진료실에서 증상을 두서없이 말하고, 진료가 끝나면 들은 말을 잊습니다. 진료 메이트는 **진료 전**에 증상을 문답으로 받아
의사에게 보여 줄 **브리핑 카드**를 만들고, **진료 후**에는 환자가 적은 메모를 **소견·검사·약·생활 지침·재방문**으로 나눠 정리합니다.

이 저장소는 그 **AI 계층**입니다. 백엔드·웹·안드로이드는 `Medical-Mate` 조직의 별도 저장소이고, 협업 규칙은
[`GIT_CONVENTION.md`](./GIT_CONVENTION.md)에 있습니다.

> **현재 공개 경로(웹 데모)의 모델: AWS Bedrock `apac.amazon.nova-pro-v1:0`(Nova Pro, 서울 출발 · 아시아 6개 리전 추론 프로파일).**
> 진료 전 추출·진료 후 메모 분류·질문 후보가 모두 이 모델로 돕니다. API 서버의 코드 기본값도 이것입니다(`src/medimate/api/app.py` `DEFAULT_MODEL`, `.env.example`).

---

## 지금 운영에서 도는 것

| | |
|---|---|
| 요청 경로 | 웹 데모 → 백엔드 `/api/demo/*` 프록시 → 이 저장소의 무상태 FastAPI → Bedrock Nova Pro. 앱 경로는 미배포 |
| 배포 | AWS Lightsail 1대 + Docker Compose. 이미지는 `main` 병합 시 GHCR에 발행되고 서버가 태그를 고정해 받습니다([`docs/deploy-requirements.md`](./docs/deploy-requirements.md)) |
| 프롬프트 | 진료 전 추출 `extract-v5-nova` · 진료 후 분류 `memo-small-v5`(붙여 쓴 메모는 `memo-v6` 조각 경로) · 질문 후보 `questions-v8` |
| 결정론 층 | 질문 문장은 템플릿, 대화 흐름은 상태 기계, 채팅 표기 복원(`ㄴㄴ`→아니 등), 런타임 가드, 재방문 날짜 계산 |
| 관측 | Langfuse 트레이싱. 본문을 보낼 때는 이름·전화번호 등 식별자를 가려서 보냅니다(트레이스에만 적용, [`docs/observability.md`](./docs/observability.md)) |

### 모델의 역할

| 모델 | 역할 | 근거 |
|---|---|---|
| **Bedrock Nova Pro** | **운영 모델.** 웹 데모의 모든 LLM 호출 | `api/app.py` `DEFAULT_MODEL`, `providers._MODEL_PROMPT`, `docs/deploy-requirements.md` |
| GPT-5.6 Terra · Claude Sonnet 5 | **비교 기준.** 모델 선정(9/03)과 프롬프트 비교(9/14)에서 같은 케이스로 잰 대상. 운영에 자동 폴백으로 붙어 있지 않습니다(환경변수로 공급자를 바꾸면 쓸 수 있을 뿐) | [`docs/decisions/2026-09-03-extractor-model.md`](./docs/decisions/2026-09-03-extractor-model.md), `evals/RESULTS.md` |
| Qwen3-1.7B Q4_0 (llama.cpp) | **온디바이스 실험.** 갤럭시 S24 울트라에서 동작을 확인했지만 앱 화면 흐름에는 연결되지 않았습니다(앱 #142 미병합) | [`docs/android-ondevice-handoff.md`](./docs/android-ondevice-handoff.md) |

---

## 무엇을 하나 (현재 구현)

| 흐름 | 입력 | 결과 |
|---|---|---|
| **① 진료 전 브리핑 카드** | 인체도에서 부위 하나 선택 → 축별 질문에 **자유 입력**으로 답함 → 통증 강도는 화면 슬라이더 값(`selections`) | 8축 카드(부위·시작·양상·퍼짐·동반 증상·경과·악화 요인·강도) + 부위 속성 기반 진료과 **안내** + 종료 시 질문 후보(요청할 때만) |
| **② 진료 후 기록** | 환자가 들은 말을 자유 메모로 적음 → 문장(또는 조각) 단위 분류 | 소견 · 검사 · 약 · 생활 지침 · 재방문 묶음 + 재방문 날짜 |

두 흐름 모두 **LLM은 추출·분류만** 합니다. 카드 값은 발화·메모 원문에서 오고, 모든 값에 근거 원문(`evidence`)이 붙습니다.

---

## 평가 — 수치마다 무엇을 잰 것인지

**서로 다른 평가입니다. 하나의 "성능"으로 합치거나 나란히 비교하지 마세요.** 전부 결정론 채점(LLM judge 없음)이고,
원본 응답은 저장해 두고 채점을 고치면 재채점합니다. 상세·재현 방법은 [`evals/RESULTS.md`](./evals/RESULTS.md).

### 운영 구성의 수치

| 무엇을 | 무엇으로 쟀나 | 결과 | 언제 |
|---|---|---|---|
| **진료 전 추출** — 발화 한 개를 받아 어느 축을 어떤 상태·값으로 채우는가 | 회귀 세트 59케이스 · 88호출, 검사 D1~D10(형식·근거 원문 일치·숫자·병명 창작·축 배정 등) | **85/88**, 안전 위반 0 (Nova Pro · `extract-v5-nova` · 런타임 가드 · 표기 복원) | 2026-09-22 |
| 진료 전 추출 — 채팅체 답(`ㄴㄴ` `ㅁㄹ` `ㅠㅠ` 붙여 쓰기) | 채팅 케이스 20 | **20/20**, 안전 위반 0 (같은 구성) | 2026-09-22 |
| **진료 후 분류** — 메모 문장에 묶음 라벨을 맞게 붙이는가 | 메모 벡터 34개의 문장 라벨 | 붙여 쓴 입력용 조각 경로 `memo-v5` 116/122 = 95.1%(경계가 같은 조각만). 정상 입력 경로 `memo-small-v4`는 98.5%로 비교값만 기록돼 있고 실행 절은 없습니다 | 2026-09-15 |
| 진료 후 분류 — 생활 지침 분리 뒤(현재 `memo-small-v5`) | 약 칸에 생활 지시가 섞였던 메모 12개만 재측정 | 라벨 48/50. **34개 전체는 v5로 다시 재지 않았습니다** | 2026-09-15 |
| **질문 후보** — 진료 전 카드에서 환자가 의사에게 물을 질문을 만들 때 규칙을 어기는가 | 카드 100장(기존 20 + 새 80) · `questions-v8` | 병명·검사·약 새 언급 0, 추측 어투 1건, 40자 초과 2% (정확도가 아니라 **위반 건수**) | 2026-09-11 |

진료 전 수치는 **발화 단위 추출**의 정확도입니다. 대화 전체나 완성된 카드의 품질을 잰 것이 아닙니다.
진료 후 수치는 **라벨** 정확도이고, 카드에 담기는 값은 문장 원문 그대로라 원문 보존 위반은 구조적으로 생기지 않습니다.

### 비교·실험 기록 (운영과 다른 조건)

| 무엇을 | 결과 | 주의 |
|---|---|---|
| 모델 선정 — 진료 전 추출, `extract-v2`, 32케이스 · 51호출 | Terra 51/51 · Sonnet 51/51 · Haiku 49 · Luna 49 · Gemini 46 | 2026-09-03. 지금 세트·채점기와 다름 |
| 같은 프롬프트(`extract-v3`)로 세 모델 — 88호출 | Terra 85 · Sonnet 83 · Nova 76 | 2026-09-14 채점기 기준. 운영 구성의 85/88(9/22)은 그 뒤 케이스·채점기가 바뀌고 Nova 전용 프롬프트만 돌린 결과라 **이 줄과 직접 비교할 수 없습니다** |
| 진료 후 분류 — Terra, 20케이스 69문장 | 68/69 | 2026-09-07. 서버 비교 확인용 |
| 온디바이스 — Qwen3-1.7B, 진료 전 추출 88호출(PC) · 진료 후 69문장 | 61/88 · 64/69(93%), 안전 위반 0 | 2026-09-07. 폰(S24 NPU)에서는 30케이스로 동작만 확인 |
| Bedrock Guardrails 프롬프트 공격 필터 vs 우리 방어 — 공격 28 · 정상 86 | Guardrails 탐지 9~10/28 · 오탐 0 / 우리 26/28 | 2026-09-28. **인젝션 방어로는 도입하지 않음** |
| Bedrock Guardrails 민감정보 필터 + 정규식 — 식별자 25 · 정상 86 | 가림 23/25 · 과잉 가림 2/86 | 2026-09-28. **트레이스 가림에만 도입** |

---

## 실험 중이거나 보류한 것 (운영에 없음)

- **진료 후 짧은 값**(`장염이래 ㄷㄷ` → `장염`): 생성 + 검증기 방식(`span/`)을 재고 있습니다. 처음 보는 메모에서 규칙보다 낫지만 운영 연결 기준에 못 미쳐 **운영은 문장 원문 그대로**입니다
- **온디바이스 추출·분류**: 폰에서 동작 확인까지. 앱 연결 전
- **진료 전 할 일(todos)**: 프로토타입만 있고 제품에서 보류(2026-09-11)
- **Guardrails 인젝션 필터**: 비교 후 기각(위 표)

## 다음에 할 것

- 가드 빈틈: 환자가 "추가해"라고 **지시하거나 추측한** 병명이 값에 들어가는 경우(인젝션 세트 K20). 의사가 한 말을 옮긴 병명은 받아야 해서 둘을 가르는 규칙을 설계 중
- 진료 전 값의 띄어쓰기 복원(붙여 쓴 답이 카드에도 붙은 채 나옴)
- 진료 후 짧은 값의 운영 연결 여부 판단

---

## 절대 지키는 선

기능이 아니라 **제약**이고, 프롬프트가 아니라 **구조로** 보장합니다.

1. **진단하지 않는다.** 온톨로지에 병명 노드가 없고, 출력 스키마에 진단 자리가 없다
2. **의사가 말한 병명을 환자가 옮긴 것은 기록 가능, AI가 추론한 병명은 불가**
3. **설명문을 생성하지 않고 인용한다.** 우리가 보증하는 것은 "출처와 일치한다"뿐
4. **증상으로 진료과를 추천하지 않는다.** 부위 노드에 적힌 과를 "안내"만 한다 ([결정 문서](./docs/decisions/2026-09-04-department-guidance.md))
5. **모든 값에 근거 원문이 붙는다.** 근거 없는 값은 만들지 않는다

예외로 **응급 안내는 합니다.**

개인정보: 타인의 진료기록은 받지 않고, 사진은 서버로 보내지 않으며, 필요한 필드만 뽑습니다(화이트리스트).

---

## API (무상태)

서버는 상태를 갖지 않습니다. 백엔드가 `state`를 들고 다니며 매 턴 보내고 받습니다. 인증은 HMAC-SHA256.

| 엔드포인트 | 용도 | 계약 |
|---|---|---|
| `POST /v1/previsit/sessions` | 진료 전 세션 시작(부위 선택, 서버·온디바이스 프로필) | [`docs/api-previsit.md`](./docs/api-previsit.md) |
| `POST /v1/previsit/turns` | 턴 처리(발화 또는 폰 추출 결과, 화면 선택값) | 〃 |
| `GET /v1/ontology/body-map` | 인체도 부위 마스터 | 〃 |
| `GET /v1/ontology/search` | 폼 입력으로 부위 찾기(유의어·한영 오타 복원, LLM 없음) | 〃 |
| `POST /v1/postvisit/memo` | 진료 후 메모 → 묶음 카드. `labels`를 주면 LLM 없이 재조립 | [`docs/api-postvisit.md`](./docs/api-postvisit.md) |
| `GET /health` | 모델·프롬프트 버전, 서명 검증, 일일 상한, 트레이싱 상태 | 〃 |

**요청이 실제로 어떻게 생겼는지는 [`docs/examples/demo-session.json`](./docs/examples/demo-session.json)을 보면 된다** —
완주 세션 하나의 요청·응답 전문이다(실제 API를 태워 저장한 것이고, 드리프트는 테스트가 잡는다).
웹 데모가 서버 없이 화면을 그릴 때 쓰는 폴백이기도 하다.

| 화면 입력 | 어디에 실나 | 어느 턴에 |
|---|---|---|
| 인체도 부위 | `POST /sessions`의 `site_node_id` + `side` | 세션 시작 |
| 자유 발화 | 턴의 `utterance` | 매 턴 |
| **통증 강도 슬라이더(1d)** | 턴의 `selections: [{axis: "severity", value: "3 (꽤 아파요)"}]` | **아무 턴이나. 단 `ended` 전에** |
| 온보딩 프로필 | 턴의 `patient_profile` | **마지막 턴에만**(`question_candidates: true`와 같이) |

`severity`는 **문답으로 묻지 않는 축이다.** 통증 점수를 말로 물으면 환자가 숫자를 지어내고 그 값은
근거가 없어서, 화면에서 고른 값만 받는다(`evidence`가 `[선택] …`이 되고 `source`는 `selection`).
`selections`를 안 보내면 카드의 심각도는 끝까지 빈다. `ended`인 턴은 카드를 바꾸지 않으므로
**종료 전에 보내야 한다.**

배포 사양·요구사항은 [`docs/deploy-requirements.md`](./docs/deploy-requirements.md) (컨테이너 하나, 1 vCPU · 512MB~1GB, 무상태, GPU 없음).

---

## 시작하기

```bash
uv sync --group dev --group providers --group api
cp .env.example .env                              # 공급자 키. 실호출 전에만 필요

uv run pytest -q                                  # 커밋 전 필수
uv run ruff format . && uv run ruff check .
uv run uvicorn medimate.api.app:app --reload      # API 로컬 실행, 문서 /docs
```

LLM 호출 없이 배관을 확인하려면:

```bash
uv run python -m medimate.evals.run --dry-run       # 진료 전 채점기
uv run python -m medimate.evals.run_memo --dry-run  # 진료 후 채점기
```

직접 써 보려면:

```bash
uv run python scripts/previsit_chat.py --site "허리 가운데" --verbose   # 진료 전 문진. 앱을 프로세스 안에서 불러 .env의 MEDIMATE_MODEL(기본 Nova Pro)을 실호출
uv run python scripts/postvisit_memo.py "위염 초기래요. 2주 뒤 오라고."   # 진료 후 분류 (기본은 로컬 llama-server, $0)
```

실호출이 들어가는 명령은 **호출 수·비용을 먼저 계산해 보여주고** 모델당 지출 상한(`--budget`, 기본 $0.50)을 지킵니다. 사비로 운영합니다.

온디바이스 모델을 PC나 폰에서 돌리는 절차는 [`evals/ondevice/README.md`](./evals/ondevice/README.md).

---

## 디렉터리

```text
src/medimate/schema/    카드 모델. card.py 진료 전 8축, postvisit.py 진료 후 묶음, export.py 백엔드 형식 어댑터
src/medimate/dialog/    문진 엔진(engine·spec·questions), 런타임 가드(guard), 메모 분류·문장 분리·날짜(memo), 용어 부위 병기(widening)
src/medimate/api/       무상태 FastAPI(app), HMAC 미들웨어(auth), 일일 지출 상한(budget)
src/medimate/llm/       프롬프트(버전 명시), 공급자 어댑터(Bedrock 외 비교용), 가격표·지출 가드
src/medimate/text/      채팅 표기 복원(chatnorm — 운영 엔진이 씀), 어휘집·형태소 보조
src/medimate/span/      진료 후 짧은 값 후보·생성·검증 (실험, API에 연결 안 됨)
src/medimate/obs/       Langfuse 트레이싱(tracing), 트레이스 식별자 가림(pii)
src/medimate/ontology/  부위 그래프 로더(조상·LCA·넓히기)
src/medimate/evals/     결정론 채점기(D1~D10)와 러너. LLM judge 없음
evals/                  케이스(진료 전 79 = 회귀 59 + 채팅 20, 인젝션 25, 식별자 35, 진료 후 20+10 등), 병명 사전, RESULTS.md. results/는 gitignore
data/ontology/          부위 온톨로지 (UBERON 추출 + 수동 보강)
docs/                   설계(ai-design), API 계약, 배포·관측, 결정 기록(decisions/), 의료인 확인서, 안드로이드 전달 문서
scripts/                대화형 클라이언트, eval 보조 스크립트, 온톨로지 생성 스크립트
```

커밋 scope는 `prompt` `rag` `embedding` `model` `eval` `pipeline` 중 하나입니다 (`GIT_CONVENTION.md` §5).

---

## 문서 지도

| 알고 싶은 것 | 문서 |
|---|---|
| 평가 전체 기록과 재현 방법 | [`evals/RESULTS.md`](./evals/RESULTS.md) |
| 왜 이렇게 설계했나 | [`docs/ai-design.md`](./docs/ai-design.md) |
| 배포 구성·환경변수·IAM·비용 | [`docs/deploy-requirements.md`](./docs/deploy-requirements.md) |
| 트레이싱과 본문 가림 | [`docs/observability.md`](./docs/observability.md) |
| 진료 후를 문답이 아닌 메모 분류로 바꾼 이유 | [`docs/postvisit-b1.md`](./docs/postvisit-b1.md) |
| 모델 선택·진료과 안내 결정 근거 | [`docs/decisions/`](./docs/decisions) |
| 안드로이드가 받는 온디바이스 묶음 | [`docs/android-ondevice-handoff.md`](./docs/android-ondevice-handoff.md) |
| 의료인에게 확인받을 항목 | [`docs/advisor-checklist.md`](./docs/advisor-checklist.md) |
| 온톨로지 재생성 | [`data/ontology/README.md`](./data/ontology/README.md) |
| 2026-09-09 시점 작업 인계 (모델 등 일부 서술이 지금과 다름) | [`docs/HANDOFF.md`](./docs/HANDOFF.md) |
