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
| `card` | 그 시점 브리핑 카드(`schema/export.py` 형식). 매 턴 갱신되므로 마지막 것만 저장해도 된다 |
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
| 세션 턴 | 20 | `end_reason: max_turns` 로 종료. 정상 문진은 최대 17턴 |
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

## 미정 (백엔드가 정하면 `schema/export.py` 한 곳만 바뀐다)

- `card`의 최종 필드명·구조. 지금은 내부 모델을 거의 그대로 낸 임시 형식
- 부위 여러 개(호소 목록) 구조 — 대화 방식 결정 대기
- "추천 진료과" — 설계와 충돌, 결정 대기. 자리 없음
