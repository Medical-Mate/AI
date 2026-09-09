# 진료 전 카드 API — 백엔드 연동 계약 (v1, 초안)

AI 서버는 무상태다. 인증·세션 식별·저장은 백엔드가 한다(`docs/ai-design.md` §7).
백엔드는 사용자 세션마다 `state` 한 덩어리를 보관하고, 환자 발화가 올 때마다
그 `state`를 AI 서버에 보내고 돌아온 `state`로 덮어쓴다. AI 서버는 아무것도 기억하지 않는다.

로컬 실행: `uv run uvicorn medimate.api.app:app --reload` → OpenAPI 문서 `http://127.0.0.1:8000/docs`

## 흐름

```
앱 ──발화──▶ 백엔드 ──{state, utterance}──▶ AI
앱 ◀─reply── 백엔드 ◀─{reply, state, card, audit}── AI
                 │
                 └─ state 덮어쓰기, card 저장, audit 적재
```

## 1. 세션 시작 — `POST /v1/previsit/sessions`

본문 없음. LLM 호출 없음.

```json
{
  "reply": "어디가 어떻게 불편해서 오셨는지 편하게 말씀해 주세요.",
  "state": { "...": "불투명하게 보관. 다음 요청에 그대로 넣는다" }
}
```

## 2. 턴 처리 — `POST /v1/previsit/turns`

```json
{ "state": { "...": "직전 응답의 state 그대로" }, "utterance": "오른쪽 무릎이 계단 내려갈 때 아파요" }
```

응답:

| 필드 | 뜻 |
|---|---|
| `reply` | 환자에게 보여줄 다음 문장(질문 또는 마무리) |
| `ended`, `end_reason` | 종료 여부. `stop`(환자가 그만) · `complete`(8축 모두 닫힘) · `max_turns` · `budget` |
| `state` | 다음 턴에 그대로 보낼 것. 내부 필드에 의존하지 않는다 |
| `card` | 그 시점 브리핑 카드(`schema/export.py` 형식). 매 턴 갱신되므로 마지막 것만 저장해도 된다. `patient_message`는 8축 뒤 "의사 선생님께 전하고 싶은 말" 질문에 환자가 답한 원문(없으면 null) |
| `audit` | 판정 로그 한 줄(물은 축, 실제 입력 발화, 추출 결과, 토큰·비용). 감사 기록으로 적재. LLM을 부르지 않은 턴은 `null` |

`ended`가 true여도 종료를 강제하지 않는다 — 환자가 언제 나가도 마지막 `card`가 결과다.
`ended` 뒤에 또 턴을 보내면 마무리 문장만 돌아오고 상태는 바뀌지 않는다.

### 실제 응답 예 (2026-09-04, gpt-5.6-terra, 1턴, 8.4초)

입력: `오른쪽 무릎이 계단 내려갈 때 시큰하게 아파요. 한 달쯤 됐어요`

```json
{
  "reply": "가장 심할 때를 10점이라고 하면 지금은 몇 점 정도인가요?",
  "ended": false,
  "end_reason": null,
  "state": {
    "...": "불투명. 그대로 다음 요청에"
  },
  "card": {
    "chief_complaint": "오른쪽 무릎이 계단 내려갈 때 시큰하게 아파요.",
    "axes": {
      "site": {
        "status": "filled",
        "value": "오른쪽 무릎",
        "evidence": [
          "오른쪽 무릎"
        ]
      },
      "onset": {
        "status": "filled",
        "value": "한 달쯤 됐어요",
        "evidence": [
          "한 달쯤 됐어요"
        ]
      },
      "severity": {
        "status": "not_asked",
        "value": null,
        "evidence": []
      }
    },
    "...": "나머지 5축 동일 구조",
    "minimally_complete": true,
    "completeness": 0.5,
    "provenance": {
      "prompt_version": "extract-v3",
      "model_id": "gpt-5.6-terra",
      "ontology_snapshot": null
    }
  },
  "audit": {
    "turn": 1,
    "asked_axis": null,
    "utterance": "오른쪽 무릎이 계단 내려갈 때 시큰하게 아파요. 한 달쯤 됐어요",
    "extraction": {
      "chief_complaint": "오른쪽 무릎이 계단 내려갈 때 시큰하게 아파요.",
      "updates": [
        {
          "axis": "site",
          "status": "filled",
          "value": "오른쪽 무릎",
          "evidence": "오른쪽 무릎"
        },
        {
          "axis": "character",
          "status": "filled",
          "value": "시큰하게",
          "evidence": "시큰하게"
        }
      ],
      "...": "..."
    },
    "usage": {
      "input_tokens": 1140,
      "output_tokens": 153,
      "cost_usd": 0.004116
    }
  }
}
```

지연은 턴당 1.4~8.4초(3턴 스모크). 앱은 대기 표시가 필요하다.

## 상한 (앱·백엔드·AI 동일값, 팀 합의 임시)

| 항목 | 값 | 넘으면 |
|---|---|---|
| 발화 길이 | 300자 | AI가 잘라서 반영하고 `reply` 앞에 안내 문구를 붙인다. 백엔드는 4000자 초과만 422 |
| 세션 턴 | 20 | `end_reason: max_turns` 로 종료. 정상 문진은 최대 18턴(8축 + 확인 + 전할 말 1) |
| 세션 토큰 | 40,000 | `end_reason: budget` 로 종료 |

## 오류

| 코드 | 상황 | 백엔드 대응 |
|---|---|---|
| 422 | 본문 스키마 위반(알 수 없는 필드 포함) | 버그. 재시도 무의미 |
| 502 | 모델 출력을 파싱하지 못함. 상태 미변경 | 같은 `state`로 1회 재시도 가능 |
| 503 | 모델 지출 상한 도달 | 운영 알림 |

## 지키는 선

- 출력 어디에도 진단·병명 자리가 없다. `card.axes`는 환자가 말한 것을 정리한 값과 그 근거 발화만 담는다
- `state`나 요청에 정의되지 않은 필드가 오면 422로 거부한다(`extra="forbid"`)
- 카드의 모든 값에는 `evidence`(환자 발화 원문)가 붙는다. 근거 없는 값은 만들어지지 않는다
- **외부 LLM(Terra)에 보내는 프롬프트에 식별자를 넣지 않는다.** 발화·카드 축 값·복용약·기저질환·알러지·환자가 덧붙인 말만. 이름·나이·성별·병원명·진료일·사용자 ID·request_id는 프롬프트에 들어가지 않는다(2026-09-09 확인. 질문 후보·할 일 생성도 같은 규칙). 건강 정보는 민감정보이므로 앱 온보딩에 "입력 내용이 외부 AI 서비스(해외)로 전송됩니다" 동의 문구가 필요하다(앱·백엔드 몫). 온디바이스 프로필 사용자에게는 옵트인 버튼으로만 보낸다

## 미정 (백엔드가 정하면 `schema/export.py` 한 곳만 바뀐다)

- `card`의 최종 필드명·구조. 지금은 내부 모델을 거의 그대로 낸 임시 형식
- 부위 여러 개(호소 목록) 구조 — 대화 방식 결정 대기
- "추천 진료과" — 설계와 충돌, 결정 대기. 자리 없음

## 온디바이스 프로필 (2026-09-07 구현. 이슈 #24 #7)

추출 모델을 폰 안에서 돌린다. **무상태 구조·응답 형태·카드 형식은 그대로**이고, 아래는 전부 선택 필드라 기존 호출을 깨지 않는다.

| 어디 | 필드 | 뜻 |
|---|---|---|
| 세션 시작 | `profile`: `"server"`(기본) \| `"ondevice"` | ondevice면 첫 자유 발화 없이 첫 축 질문부터, 가드를 "물은 축만"으로 |
| 턴 요청 | `extraction` (+ `extraction_meta`: model_id·prompt_version) | 폰이 만든 추출 JSON. 있으면 서버는 LLM을 부르지 않고 가드만 통과시킨다. **근거 검증을 위해 `utterance`(원문)도 함께 보낸다** — 발화는 외부 LLM 업체에 가지 않는 것이고, 우리 서버·백엔드에는 온다(카드에 근거 인용이 들어가므로 새 노출은 아님) |
| 턴 요청 | `selections`: `[{axis, value}]` | 선택지(칩)·폼으로 고른 값. LLM 없이 카드에 바로, 근거는 `[선택] 값`. `utterance` 없이 selections만 와도 한 턴으로 처리해 다음 질문을 준다. 같은 턴에 `utterance`를 섞어도 된다 |
| 요청 공통 | `request_id` | 응답·audit에 그대로 되돌린다. 중복 응답 캐시는 백엔드가 원하면 백엔드에서 |
| 응답 `audit` | `dropped[]`, `raw_extraction`, `source` | 가드가 버린 갱신과 이유(not_asked_axis / evidence_not_in_utterance / number_not_in_utterance / note_not_in_utterance), 모델 원본, 출처(server / device / none) |
| 응답 `card.provenance` | `model_id`, `prompt_version` | `extraction_meta`가 오면 그 값으로 갱신된다(폰 모델·프롬프트 버전) |
| 새 엔드포인트 | `GET /v1/ontology/body-map` | 부위 마스터: `anchors[]{id,label,image_key,laterality,view,region,departments,zones[]}` + `ontology_snapshot`. 앱·백엔드 공용, 읽기 전용. 진료과 안내는 팀 콘텐츠(인용 아님). **`image_key`**(2026-09-09 추가)는 앱 이미지 자산 파일명 키(영문 이름 슬러그, 예 `head`, `lower-back-and-hip`) — 앱이 id→파일을 하드코딩하지 않게. 이미지는 앱 자산이고 서버는 키만 낸다. 응답 예시 전체: `docs/examples/body-map.json` |

백엔드가 할 일: 위 필드를 그대로 통과시키기, (선택) request_id 캐시, HMAC 서명.

### 인증 — HMAC (구현됨, 키 없으면 검증 생략)
- 헤더: `X-Signature`, `X-Timestamp`(epoch 초), `X-Request-Id`. 이름은 환경변수로 바꿀 수 있다
- 서명: `hex(HMAC_SHA256(secret, f"{timestamp}.{request_id}." + body_bytes))`
- 허용 시각 폭 ±300초(환경변수). `GET`·`/health`·문서 경로는 예외
- 실패 시 401 `{"detail": "hmac: <이유>"}`
- 서버 설정: `MEDIMATE_HMAC_SECRET`, `MEDIMATE_HMAC_HEADER_*`, `MEDIMATE_HMAC_MAX_SKEW_S`

### 배포 — Dockerfile (저장소 루트)
`docker build -t medimate-ai . && docker run -p 8000:8000 --env-file .env medimate-ai`. 모델 없음, 1 vCPU·1GB. 폰이 뽑았든 서버가 뽑았든 백엔드에게는 같은 `state`·`card`다.
발화 원문은 폰에서 추출 JSON으로 바뀌어 오므로 백엔드를 거치지 않는다.

가드 규칙과 근거: `evals/RESULTS.md` "엔진 가드", 코드 `src/medimate/dialog/guard.py`.
