# 진료 후 기록 API — 백엔드 연동 계약 (v1, 초안 · 2026-09-07)

와이어프레임 1p(진료 후 메모) → 1q(자동 분류 결과) → 1q-2(수정) → 1r(재방문 일정).
AI 서버는 무상태다. 메모 하나를 받아 **4묶음 카드**로 돌려주고 아무것도 기억하지 않는다. 저장은 백엔드.

## 한 번에 끝나는 요청 — `POST /v1/postvisit/memo`

```json
{
  "memo": "위염 초기라고 하셨어요. 혈액검사 했고 결과는 다음에 알려준대요. 약은 2주분이고 커피랑 매운 거 줄이래요. 2주 뒤에 다시 오라고 하셨어요.",
  "visit_date": "2026-09-12",
  "clinic": "서울OO병원 내과",
  "previsit_anchor_id": "ANC:004",
  "request_id": "r-77"
}
```

| 필드 | 뜻 |
|---|---|
| `memo` | 환자가 적은 원문(타이핑·음성). 최대 2000자 |
| `visit_date` | 진료일(ISO). 재방문 날짜 계산의 기준. 없으면 날짜 계산 생략 |
| `clinic` | 병원·과. 앱(1s·1r)이 준다. AI가 만들지 않는다 |
| `labels` (선택) | `{"0":"findings","1":"tests",…}`. **있으면 서버는 LLM을 부르지 않고** 카드만 조립한다. 폰이 분류했거나 1q-2에서 사용자가 고친 라벨 |
| `labels_meta` (선택) | labels가 폰 모델에서 왔으면 `{model_id, prompt_version}` → provenance에 기록 |
| `previsit_anchor_id` (선택) | 진료 전 카드의 부위 앵커. 소견 용어의 부위와 **대조만** 한다(same/different) |
| `request_id` | 응답에 그대로 |

응답:

```json
{
  "card": { "card_type": "postvisit", "axes": { "findings": {"status":"filled","value":"위염 초기라고 하셨어요.","evidence":["위염 초기라고 하셨어요."]}, "tests": {...}, "medication_instructions": {...}, "follow_up": {...} },
            "memo": "...", "unsorted": [], "follow_up_date": {"text":"2주 뒤","date":"2026-09-26","approximate":true,"basis":"visit_date 2026-09-12 + 14d"},
            "visit_date": "2026-09-12", "clinic": "서울OO병원 내과", "widening": [], "site_comparison": null, "provenance": {...} },
  "sentences": ["위염 초기라고 하셨어요.", "혈액검사 했고 결과는 다음에 알려준대요.", "약은 2주분이고 커피랑 매운 거 줄이래요.", "2주 뒤에 다시 오라고 하셨어요."],
  "labels": {"0":"findings","1":"tests","2":"medication_instructions","3":"follow_up"},
  "dropped": [],
  "source": "server",
  "usage": {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0},
  "request_id": "r-77"
}
```

- `card.axes.*.value`는 그 묶음 문장들을 ` · `로 이은 **원문**, `evidence`는 문장 목록. AI가 고쳐 쓴 말이 없다
- 묶음에 문장이 없으면 `status: unknown`(메모에 그 얘기가 없었다)
- `unsorted`: 어느 묶음에도 안 들어간 문장. **버리지 않는다.** 1q 화면 하단에 "분류되지 않은 메모"로
- `follow_up_date`: 결정론 계산. `approximate: true`면 "전후"로 표시. 못 읽으면 `null`(앱 달력에서 직접)
- `widening`: 소견 문장의 해부 용어에 부위 병기(설명 아님). `site_comparison`: 진료 전 부위와 같은가(판정 아님)
- `sentences`·`labels`는 **1q-2 수정 화면의 입력**이다. 사용자가 라벨을 바꾸면 같은 엔드포인트에 `labels`를 넣어 다시 부른다(LLM 없음, 즉시)

## 흐름

```
1p 메모 작성 ─▶ 백엔드 ─{memo, visit_date, clinic}─▶ AI ─▶ 서버 LLM 분류 → 카드
1q 결과 표시 ◀─ card, sentences, labels ─┘
1q-2 수정   ─▶ 백엔드 ─{memo, labels(고침)}────▶ AI ─▶ LLM 없이 카드 재조립
1r 일정     ◀─ card.follow_up_date ─┘
```

온디바이스: 폰이 문장 분류를 끝냈으면 `labels` + `labels_meta`로 보낸다. 서버 LLM 호출 0. 문장 분리 규칙은 폰과 서버가 같아야 하므로, 폰은 `sentences`를 먼저 받거나 같은 규칙(`dialog/memo.py split_sentences`)을 쓴다.

## 오류

| 코드 | 상황 | 대응 |
|---|---|---|
| 422 | 스키마 위반, 빈 메모, 2000자 초과 | 버그 |
| 502 | 분류기 출력 파싱 실패 | 같은 요청 1회 재시도 |
| 503 | 모델 지출 상한 | 운영 알림 |

## 지키는 선
- 값은 문장 원문. 요약·정리·설명 없음. 병명은 환자가 적은 그대로만
- 약 용법·검사 결과를 일반 지식으로 채우지 않는다(분류만)
- 날짜는 LLM이 아니라 계산. 병원·과는 앱이 준다

## 근거·수치
- 분류 정확도(Qwen3-1.7B, 20메모/69문장): 93%, 원문 보존 위반 0 — `evals/RESULTS.md` "챗봇② B1"
- 프롬프트 `memo-small-v4`, 출력 스키마 `evals/ondevice/memo_labels.schema.json`(번호 키 객체는 요청별 생성)

## 미정
- 서버 모델(Terra) 정확도 확인 1회(약 $0.05)
- 카드 최종 필드명은 백엔드가 정한다(`schema/export.py` 한 곳)
- 문장 분리 규칙의 폰·서버 공유 방식
