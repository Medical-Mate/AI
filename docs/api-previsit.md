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
| `card.axes.*.status` | **`not_asked` · `filled` · `unknown` · `skipped` · `ambiguous` 다섯 값이다.** `not_asked`가 8축의 기본값이라 세 값으로 검증하면 카드 대부분이 떨어진다. `ambiguous`는 답이 모호해 한 번 되물은 상태다 |
| `card.title` | 카드 제목. **`{부위 라벨} · {기간}`을 결정론으로 조합한다** — LLM을 부르지 않는다(2026-09-11 합의). 20자 이내, `evidence` 없음. 기간은 **숫자+단위만** 뽑는다("2주 전부터요, 등산 다음 날부터" → `2주`). 순우리말 날짜(이틀·닷새·열흘)와 상대 표현(어제·오늘)도 잡는다. 기간을 못 뽑으면 부위만, 20자를 넘으면 기간을 빼고 부위만(**자르지 않는다** — 자르면 부위 이름이 깨진다). **부위가 없으면 `null`** → 앱이 제목 줄을 숨긴다. **증상어("통증")를 붙이지 않는다** — 부위 라벨에도 축에도 없는 말이라 붙이면 없는 사실을 넣는 것이고, 느낌이 "먹먹해요"인 카드에 "통증"이라 쓰면 틀린다. 카드 100장 실측: 100장 생성, 기간 포함 73장, 20자 초과 0 |
| `card` | 그 시점 브리핑 카드(`schema/export.py` 형식). 매 턴 갱신되므로 마지막 것만 저장해도 된다. `patient_message`는 8축 뒤 "의사 선생님께 전하고 싶은 말" 질문에 환자가 답한 원문(없으면 null) |
| `audit` | 판정 로그 한 줄(물은 축, 실제 입력 발화, 추출 결과, 토큰·비용). 감사 기록으로 적재. **LLM을 안 부른 턴에도 `audit`은 있다** — `source`가 `none`(선택지만 온 턴)이나 `device`(폰 추출)이고 `usage`는 0이다. `audit`이 `null`인 것은 **턴 처리 자체가 없을 때**뿐이다(빈 입력, 상한 종료). 2026-09-11 정정 |

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
| 새 엔드포인트 | `GET /v1/ontology/body-map` | 부위 마스터: `anchors[]{id,label,image_key,laterality,view,region,departments,zones[]}` + `ontology_snapshot`. 앱·백엔드 공용, 읽기 전용. 진료과 안내는 팀 콘텐츠(인용 아님). **`image_key`**(2026-09-09 추가)는 앱 이미지 자산 파일명 키(영문 이름 슬러그, 예 `head`, `lower-back-and-hip`) — 앱이 id→파일을 하드코딩하지 않게. 이미지는 앱 자산이고 서버는 키만 낸다. 응답 예시 전체: `docs/examples/body-map.json`. **`aliases`**(2026-09-10 추가)는 폼 검색용 유의어(복부→배, 옆구리→허리 옆, 뒷목→목 뒤·옆). 앱이 로컬에서 이름+aliases 포함 매칭하면 된다 |
| 새 엔드포인트 | `GET /v1/ontology/search?q=복부&limit=8` | 위 유의어 표로 서버가 매칭한 결과 `results[]{id,label,kind,anchor_id,matched,score}`. 점수 **3** 정확 / **2** 접두 / **1** 포함 / **0** 직접 매칭이 아니라 **앵커에 딸려 온 구역**(2026-09-10 추가) — `score > 0`으로 거르면 구역이 전부 사라진다. 앵커가 점수 2 이상으로 걸리면 그 아래 구역을 CSV 순서(디자이너 배치 순서)로 뒤에 붙인다: `q=다리` → 다리(3)·허벅지(1)·무릎(0)·종아리(0)·발목(0)·발(0). **한/영 자판 오타를 복원한다**(2026-09-10 추가): `q=qo` → 배, `q=duvrnfl` → 옆구리. 원문 결과가 0건일 때만 복원하므로 한글 질의 결과는 이전과 같다. 한글 오타(무릅)·임베딩은 없다. 조합 중간 입력(`옆굴`, `아랫ㅂ`)은 0건 — 앱이 직전 결과를 유지하는 쪽으로 처리한다. 앵커·구역만, 증상·병명은 매칭하지 않는다. `q`는 300자에서 자른다(발화 상한과 같은 값). 표: `data/ontology/aliases.csv` |

백엔드가 할 일: 위 필드를 그대로 통과시키기, (선택) request_id 캐시, HMAC 서명.

### 인증 — HMAC (구현됨, 키 없으면 검증 생략. 규격 확정 2026-09-11 #7)
- 헤더: `X-Signature`, `X-Timestamp`(epoch 초), `X-Request-Id`. 이름은 환경변수로 바꿀 수 있다
- 서명: `hex(HMAC_SHA256(secret, f"{METHOD}.{path}.{timestamp}.{request_id}." + body_bytes))`
  - `METHOD`는 **대문자**(`POST`)
  - `path`는 **쿼리스트링 제외**, **앞 `/` 포함** → `/v1/previsit/turns`.
    쿼리를 넣으면 인코딩 차이로 깨진다. POST에 쿼리가 붙으면 규격을 다시 정한다
  - `path`는 **프록시를 지난 뒤의 경로**다. 게이트웨이가 경로를 다시 쓰면 양쪽이 다른 문자열을
    보게 되고, 그게 서명 불일치의 제일 흔한 원인이다
  - 구분자는 `.`이고 `request_id` 뒤에도 하나 붙는다. 그 뒤에 **직렬화된 그대로의 본문 바이트**
    (재직렬화하면 공백·키 순서가 달라져 서명이 틀어진다)
  - `X-Request-Id`를 안 보내면 그 자리를 **빈 문자열**로 계산한다. 백엔드는 항상 보내기로 했다
- 허용 시각 폭 ±300초(환경변수)
- **면제는 경로 화이트리스트뿐이다** — `/health` `/docs` `/openapi.json` `/redoc`.
  **`GET`도 서명이 필요하다**(2026-09-11 변경). 이전에는 메서드 단위로 GET 전체를 면제해서
  화이트리스트가 무의미했고, 그 사이 `GET /v1/ontology/search`가 생겼다. GET 본문은 비어 있다
- 실패 시 401 `{"detail": "hmac: <이유>"}`
- 서버 설정: `MEDIMATE_HMAC_SECRET`, `MEDIMATE_HMAC_HEADER_*`, `MEDIMATE_HMAC_MAX_SKEW_S`.
  **한쪽만 채우면 AI가 전 요청을 거부한다** — 양쪽을 같은 값으로 동시에 채운다

### 배포 — Dockerfile (저장소 루트)
`docker build -t medimate-ai . && docker run -p 8000:8000 --env-file .env medimate-ai`. 모델 없음, 1 vCPU·1GB. 폰이 뽑았든 서버가 뽑았든 백엔드에게는 같은 `state`·`card`다.
발화 원문은 폰에서 추출 JSON으로 바뀌어 오므로 백엔드를 거치지 않는다.

가드 규칙과 근거: `evals/RESULTS.md` "엔진 가드", 코드 `src/medimate/dialog/guard.py`.
