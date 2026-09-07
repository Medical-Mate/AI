# 진료 후 기록(챗봇② B1) — 설계 (2026-09-07 개정)

> **2026-09-07 개정**: 와이어프레임 1p·1q를 확인한 결과 진료 후는 **문답이 아니라 자유 메모 하나 → 4묶음 분류**다
> ("진료실에서 들은 말을 그대로 남겨두세요" → "AI가 메모를 4가지로 나눴어요"). 아래 6축 문답 설계는 **보조 경로**로 내리고,
> 주 경로를 `dialog/memo.py`로 바꿨다. 4묶음: 소견(findings) · 검사(tests) · 약·지시(medication_instructions) · 재방문(follow_up).
>
> **주 경로 요약**
> - 메모를 결정론으로 문장 분리 → LLM은 **문장마다 라벨 하나**(4묶음 + none) → 카드에 문장 **원문 그대로** → none은 `unsorted`에 보존
> - 다축 추출이 아니라 단일 선택 문제라 소형 모델에 유리하고, 값이 원문이라 근거 검증이 구조로 통과한다
> - 재방문 날짜는 LLM이 아니라 진료일 기준 결정론 계산(`followup_date`): "2주 뒤"→+14d, "다음 달 15일", "9월 30일"
> - 병원·과·진료일은 앱이 준다(1s·1r). AI가 만들지 않는다
> - 프롬프트 `memo-small-v1`, 스키마 `evals/ondevice/memo_labels.schema.json`, eval `medimate.evals.run_memo`(문장 라벨 정확도·재방문 날짜·원문 보존)
> - 넓히기 병기(`widening.py`)는 `findings` 문장에 그대로 적용
>
> 아래는 개정 전 문답 설계(보조 경로) 기록.

# (기록) 진료 후 문답 — 뼈대 설계 (2026-09-04)

이슈 #9의 "지금 시작 가능" 여섯 항목을 코드와 문서로 옮긴 것. LLM 호출 0.
프롬프트(`postvisit-v1`)와 eval 실호출은 축 확정(자문 확인서 회신) 뒤에 한다.

## 핵심 한 줄

진료실에서 나온 직후, 환자가 **들은 말을 문답으로 받아 환자 표현 그대로 기록**한다. 요약이 아니라 기록이다.

## 무엇이 어디에 있나

| 항목 | 위치 | 상태 |
|---|---|---|
| 카드 스키마 `PostVisitCard`(6축 + 넓히기 + 대조 + 문서 코드 자리) | `src/medimate/schema/postvisit.py` | 완료 |
| 공통 뼈대 `InterviewCard`, 진료 전 카드는 그 하위 | `src/medimate/schema/card.py` | 완료 |
| 질문 템플릿·순서·확인 문구 | `src/medimate/dialog/postvisit_questions.py` | 초안. 자문 확인 대상 |
| 엔진 재사용 — 문진 명세 `InterviewSpec`으로 매개변수화 | `src/medimate/dialog/spec.py`, `engine.py` | 완료. 진료 전 동작·테스트 불변 |
| 넓히기 병기 + 부위 대조 | `src/medimate/dialog/widening.py` | 완료. 사전 매칭, LLM 무관 |
| export (`card_type` 구분) | `src/medimate/schema/export.py` | 임시 형식. 백엔드 확정 후 한 곳만 수정 |
| 채점 규칙 | 아래 §채점 | D1~D10 그대로 적용됨. 추가 규칙은 프롬프트와 함께 |
| 자문 확인서 | `docs/advisor-checklist.md` | 초안 |

API 엔드포인트(`/v1/postvisit/...`)는 B5. 엔진·상태(`SessionState.spec = "postvisit"`)는 이미 준비돼 있어
프롬프트가 생기면 `api/app.py`에 명세 분기만 더한다.

## 축 (6)

| 축 | 값의 성격 |
|---|---|
| `heard_diagnosis` | 환자가 옮긴 의사의 말 그대로. AI가 고치거나 채우지 않는다 |
| `medication` | 이름·횟수·기간, 환자가 말한 것만 |
| `tests_procedures` | 받았거나 하기로 한 것 |
| `follow_up` | 다음 방문 |
| `instructions` | 주의사항·생활 지시 |
| `open_questions` | 못 물어본 것·헷갈리는 것. 원문 |

공통: 상태 5종, 확인 질문 축당 1회, 마지막 자유 발화 1턴 → `patient_message`, 축 밖 말 → `patient_notes`.

## 넓히기 병기 (온톨로지)

`heard_diagnosis`의 값·근거에서 구조 노드 한국어 이름(괄호 별칭 포함, 2글자 이상)을 찾아
`widen()`으로 앵커까지 올린다. 결과는 `widening[]`에 (용어, 노드, 앵커) 병기.

- "앞십자인대가 좀 늘어났다고" → 앞십자인대(전방십자인대)(인대) · 부위: 다리
- 매칭 없음("감기라고 하셨어요") → 아무것도 붙이지 않는다
- 진료 전 카드의 앵커와 비교해 `site_comparison` = same / different / null. **판정이 아니라 표시**

## 지키는 선 — 구조로

| 선 | 어디서 막나 |
|---|---|
| AI 추론 병명 없음 | 병명이 들어갈 축이 `heard_diagnosis` 하나이고 정의가 "환자가 옮긴 말". 추출 스키마 `extra="forbid"` |
| 약 용법 일반 지식 금지 | 채점 D5 — 출력의 숫자는 발화에 있어야 한다. "하루 3번"을 모델이 만들면 실패 |
| 설명·해석 없음 | 카드에 설명 필드가 없다. B3(인용)은 라이선스 회신 후 별도 |
| 진료 전 축 혼입 | 엔진이 명세에 없는 축 갱신을 버린다(`_apply`) |
| 근거 없는 값 없음 | 챗봇①과 동일 |

## 채점 (D1~D10 그대로 + 진료 후 관찰)

결정론 채점기 `evals/score.py`는 축 이름에 의존하지 않으므로 그대로 쓴다.
- D4 근거 부분 문자열, D5 숫자 출처, D6 병명 사전(**주의**: 진료 후에는 발화에 병명이 있으므로 "발화에 있으면 통과"가 그대로 맞다), D7 축 집합, D8 상태, D9 stop, D10은 진료 후에서 `chief_complaint_required: false`
- 추가 관찰(안전 조건은 아님): 용법 표현이 값에서 정규화됐는지(D4S와 같은 성격), `heard_diagnosis` 값이 발화의 병명 표현을 바꿨는지(VC로 원문 포함 검사)

## 다음 (결정 후)

1. 자문 확인서 회신 → 축·문구 확정
2. 프롬프트 `postvisit-v1`, 케이스 30, Terra·Luna 실호출(약 $0.3)
3. B5 엔드포인트 + `docs/api-postvisit.md`
4. B2 자유 메모 분류 — 같은 추출기, asked_axis 없이
