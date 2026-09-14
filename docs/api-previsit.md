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

**LLM 호출 없음.** 본문은 전부 선택이고, 없으면 빈 카드로 시작한다.

```json
{
  "site_node_id": "SUR:042",
  "side": "left",
  "profile": "server",
  "request_id": "req-0001"
}
```

| 필드 | 뜻 |
|---|---|
| `site_node_id` | 인체도에서 짚은 **앵커 또는 구역** ID(`ANC:012` · `SUR:042`). 구조 노드는 받지 않는다. SITE 축이 채워지고 첫 질문이 그 부위로 앵커되며 진료과 안내가 카드에 붙는다 |
| `side` | `left` · `right` · `both`. **대소문자를 가리지 않는다**(2026-09-11) — `LEFT`·`Left`도 받고 앞뒤 공백도 접는다. 카드에는 소문자로 남는다. **좌우가 있는 부위(34곳 중 21곳)에만** 붙일 수 있고 없는 부위에 보내면 422 |
| `site_label` | 온톨로지 없이 라벨만 넘길 때(테스트·임시). `site_node_id`가 있으면 무시된다 |
| `profile` | `server`(기본) · `ondevice` |
| `request_id` | 응답에 그대로 되돌린다 |

```json
{
  "reply": "왼쪽 허리 옆 쪽이 어떻게 불편하신지 편하게 말씀해 주세요.",
  "state": { "...": "불투명하게 보관. 다음 요청에 그대로 넣는다" },
  "request_id": "req-0001"
}
```

> 이 절이 오래 **"본문 없음"**으로 되어 있었다. 그래서 `side` 값 규약이 어디에도 없었고,
> 백엔드가 `LEFT` 대문자로 구현해 422가 났다(2026-09-11). 받는 쪽을 넓혀 해결했다.

## 2. 턴 처리 — `POST /v1/previsit/turns`

```json
{ "state": { "...": "직전 응답의 state 그대로" }, "utterance": "오른쪽 무릎이 계단 내려갈 때 아파요" }
```

응답:

| 필드 | 뜻 |
|---|---|
| `reply` | 환자에게 보여줄 다음 문장(질문 또는 마무리) |
| `ended`, `end_reason` | 종료 여부. `stop`(환자가 그만) · `complete`(8축 모두 닫힘) · `max_turns` · `budget` |
| `state` | 다음 턴에 그대로 보낼 것. **불투명하게 보관하고 열어보지 않는다** |
| ⚠️ `card` vs `state.card` | **둘은 다르다.** 최상위 `card`가 백엔드가 읽을 것이다 |
| `card.axes.*.status` | **`not_asked` · `filled` · `unknown` · `skipped` · `ambiguous` 다섯 값이다.** `not_asked`가 8축의 기본값이라 세 값으로 검증하면 카드 대부분이 떨어진다. `ambiguous`는 답이 모호해 한 번 되물은 상태다 |
| `card.title` | 카드 제목. **`{부위 라벨} · {기간}`을 결정론으로 조합한다** — LLM을 부르지 않는다(2026-09-11 합의). 20자 이내, `evidence` 없음. 기간은 **숫자+단위만** 뽑는다("2주 전부터요, 등산 다음 날부터" → `2주`). 순우리말 날짜(이틀·닷새·열흘)와 상대 표현(어제·오늘)도 잡는다. 기간을 못 뽑으면 부위만, 20자를 넘으면 기간을 빼고 부위만(**자르지 않는다** — 자르면 부위 이름이 깨진다). **부위가 없으면 `null`** → 앱이 제목 줄을 숨긴다. **증상어("통증")를 붙이지 않는다** — 부위 라벨에도 축에도 없는 말이라 붙이면 없는 사실을 넣는 것이고, 느낌이 "먹먹해요"인 카드에 "통증"이라 쓰면 틀린다. 카드 100장 실측: 100장 생성, 기간 포함 73장, 20자 초과 0 |
| `card` | 그 시점 브리핑 카드(`schema/export.py` 형식). 매 턴 갱신되므로 마지막 것만 저장해도 된다. `patient_message`는 8축 뒤 "의사 선생님께 전하고 싶은 말" 질문에 환자가 답한 원문(없으면 null) |
| `card.question_candidates` | **"의사에게 물어볼 것" 후보**(앱 4단계 화면). `[{text, source, rank}]` 최대 3개. **요청에 `question_candidates: true`를 보낸 종료 턴에만** 값이 오고, 그 외에는 `null`이다 |
| `audit` | 판정 로그 한 줄(물은 축, 실제 입력 발화, 추출 결과, 토큰·비용). 감사 기록으로 적재. **LLM을 안 부른 턴에도 `audit`은 있다** — `source`가 `none`(선택지만 온 턴)이나 `device`(폰 추출)이고 `usage`는 0이다. `audit`이 `null`인 것은 **턴 처리 자체가 없을 때**뿐이다(빈 입력, 상한 종료). 2026-09-11 정정 |

> **`state.card`가 아니라 최상위 `card`를 읽으세요.**
>
> ```
> 최상위 card  = to_backend_payload() 결과.
>                title · department_guidance · question_candidates · axes.*.source 가 여기에만 있다
> state.card   = 엔진 내부 표현. 위 필드들이 없다. 열어보지 않는다
> ```
>
> 2026-09-11에 `state.card`를 읽어 `title`·`department_guidance`가 `null`로 보이는 일이 있었다.
> `state`는 왕복용이고 내부 구조가 바뀐다.

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
| 422 | 본문 스키마 위반(알 수 없는 필드 포함). **`question_candidates`가 꺼진 턴의 `patient_profile`도 여기다.** `detail`은 `loc`·`msg`·`type`만 — **요청 값은 되돌려주지 않는다** | 버그. 재시도 무의미 |
| 400 | **스키마는 맞는데 엔진이 만들 수 없는 `state`** | 같은 상태로 재시도해도 같다. 아래 참고 |
| 502 | 모델 출력을 파싱하지 못함. 상태 미변경 | 같은 `state`로 1회 재시도 가능 |
| 503 | 모델 지출 상한 도달 | 운영 알림 |

> **422 `detail`에는 값이 없다**(2026-09-14)
>
> ```json
> {"detail": [{"loc": ["body", "patient_profile", "oops"], "msg": "Extra inputs are not permitted", "type": "extra_forbidden"}]}
> ```
>
> pydantic 기본 동작은 문제가 된 값을 `input`에 담아 되돌려준다 — 잘못된 요청 하나가
> **복용약·기저질환·알러지와 환자 발화 원문을 응답에 실어 보낸다.** 웹이 우리를 직접 부르는
> 구성이 되면 그 본문이 브라우저 콘솔·에러 리포트에 남으므로 벗겼다.
> `loc`은 그대로다 — 어느 필드가 왜 틀렸는지는 그것으로 안다.

> **400 — 상태 불변식**
>
> `state`는 왕복용이라 **그대로 돌려보내는 것이 전제**다. 재조립하거나 컬럼으로 펴서 저장했다
> 합치면 스키마는 통과하지만 우리가 만들지 않는 조합이 된다. 지금 막는 것은 둘뿐이다.
>
> | 메시지 | 뜻 | 어디를 보나 |
> |---|---|---|
> | `axes.<축>: status가 not_asked인데 value가 있음` | 안 물은 축에 값이 들어갔다 | 값을 넣을 때 `status`도 같이 옮겼는지 |
> | `site_selection 없음 — site 축은 인체도 선택으로 채워져 있음` | 인체도 노드로 시작한 세션인데 선택 기록이 없다 | `state.card.site_selection`을 떨어뜨리지 않았는지 |
>
> 둘째가 특히 조용하다 — `site_selection`이 비면 **`title`과 `department_guidance`가 `null`이 되고**
> 200에 카드도 멀쩡해 보인다. 그래서 400으로 세웠다.
>
> **막지 않는 것**: `ambiguous`에 값이 있는 것(되물으면서 잠정 값을 든 정상 상태),
> `unknown`·`skipped`에 앞 턴 값이 남은 것, 말로 답한 `site`에 `site_selection`이 없는 것,
> `site_label`로 시작해 선택 기록이 없는 것.
> 문답 중인 환자의 요청을 우리 추측으로 끊지 않는다.

## 지키는 선

- 출력 어디에도 진단·병명 자리가 없다. `card.axes`는 환자가 말한 것을 정리한 값과 그 근거 발화만 담는다
- `state`나 요청에 정의되지 않은 필드가 오면 422로 거부한다(`extra="forbid"`)
- 카드의 모든 값에는 `evidence`(환자 발화 원문)가 붙는다. 근거 없는 값은 만들어지지 않는다
- **외부 LLM(Terra)에 보내는 프롬프트에 식별자를 넣지 않는다.** 발화·카드 축 값·복용약·기저질환·알러지(요청의 `patient_profile`)·환자가 덧붙인 말만. 이름·나이·성별·병원명·진료일·사용자 ID·request_id는 프롬프트에 들어가지 않는다(2026-09-09 확인. 질문 후보·할 일 생성도 같은 규칙). 건강 정보는 민감정보이므로 앱 온보딩에 "입력 내용이 외부 AI 서비스(해외)로 전송됩니다" 동의 문구가 필요하다(앱·백엔드 몫). 온디바이스 프로필 사용자에게는 옵트인 버튼으로만 보낸다

## 확정된 것 (2026-09-11 백엔드 회신 · 이슈 #7)

- **`card` 구조는 우리 구조 그대로 간다.** 중첩(`axes: {}`), 빈 값은 `null`, `status`는 값과 따로,
  날짜는 ISO-8601, `evidence` 유지. `patient_message`는 뺐다
- **부위는 한 번에 하나만.** 배열이 아니다. `docs/decisions/2026-09-11-single-site.md`.
  호소 목록 구조는 만들지 않는다(이슈 #8 닫음)
- **"추천 진료과"는 없다.** 백엔드가 `suggested_department`와 enum 검증을 폐기했고,
  부위 속성 기반 `department_guidance`(배열 + `source`)를 그대로 받는다

### 남은 구현 (계약은 합의됨)

- `question_candidates` — 종료 턴에만. `[{text, source, rank}]`. **요청 필드로 켠다**
  (`question_candidates: true`, 기본 `false`). 온디바이스 프로필에서 카드가 외부로 나가는 것을
  환자가 모르게 두지 않기 위한 옵트인이다
- `patient_profile` — 앱 온보딩의 복용약·기저질환·알러지. 위 필드가 켜진 턴에만 보낼 수 있고
  (아니면 422), 카드에 저장하지 않는다. **이게 없으면 와이어프레임 3개 중 약 질문이 안 나온다**
- `axes.*.source` — `ai_extraction` / `selection` / `patient_edit`. 환자가 편집 화면에서 고친 값을
  의사가 구별할 수 있게 한다. `patient_edit`은 백엔드가 채운다

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

### 질문 후보 — `question_candidates`

요청 필드 **`question_candidates: true`**(기본 `false`). **안 보내면 안 돕니다.**

옵트인인 이유가 둘이다. ① **온디바이스 프로필은 "발화가 외부로 안 나간다"가 전제**인데
후보 생성은 카드 값을 외부 LLM으로 보낸다 — 환자 동의(옵트인 버튼) 없이 자동으로 돌면
그 전제가 깨진다 ② Bedrock 비용이 한정된 크레딧에서 나가고, 무조건 도는 구조면 끌 방법이 없다.

- **종료 턴에만**(`ended: true`). 중간 턴은 `null`이고 LLM을 부르지 않는다
- 최대 **3개**. 생성은 제한하지 않고 **카테고리 가중치로 정렬해 위에서 자른다**.
  `rank`는 1부터. 앞에서부터 쓰면 된다
- 정렬 원칙: **카드를 보면 의사가 바로 아는 것은 낮추고, 카드에 있어도 의사가 먼저 안
  물어볼 수 있는 것(복용약·알러지·기저질환)은 올린다**
- **생성이 실패해도 카드는 그대로 나간다.** `question_candidates`만 `null`이 되고 200이다 —
  문답을 다 마친 환자의 카드가 후보 생성 실패로 날아가면 안 된다
- **최종 목록은 앱·백엔드가 만든다.** 우리는 후보만 낸다(환자가 지우거나 직접 더한 결과가 최종)
- 프롬프트에 **식별자를 넣지 않는다** — 이름·나이·성별·병원·진료일·ID·`request_id`.
  카드에 그 값이 있어도 프롬프트에는 안 들어간다

#### 복용약·기저질환·알러지 — `patient_profile` (2026-09-11 확정, #7)

```json
{
  "state": { "...": "" },
  "utterance": "네 다 말한 것 같아요",
  "question_candidates": true,
  "patient_profile": {
    "medications": ["혈압약"],
    "conditions": ["고혈압"],
    "allergies": []
  }
}
```

이 셋이 없으면 **와이어프레임 3개 중 하나인 약 질문이 안 나옵니다.** eval에서 그 세 카테고리를
3% → 65%로 올린 규칙(`questions-v8`)이 재료가 없으면 걸리지 않습니다. 카드에는 이 자리가
없습니다 — 축이 아니고 이 문진에서 환자가 말한 것도 아니라서, **앱 온보딩에 적은 것을 요청으로
받습니다.**

- **필드는 선택입니다.** 생략하면 지금까지와 같습니다(빈 값). 이름이 `profile`이 아닌 이유는
  `state.profile`(`server`·`ondevice`)과 헷갈리기 때문입니다
- 원소는 문자열만. **환자가 적은 그대로 씁니다** — 정규화·해석하지 않고(표준 약품명으로 바꾸는
  순간 우리가 만든 값이 됩니다), **카드에 저장하지도 않습니다.** 프롬프트 재료로 쓰고 버립니다
- 빈 문자열·공백만 있는 원소는 버립니다. 앱이 빈 입력칸을 그대로 보내면 프롬프트에 빈 줄이 됩니다
- **`[]`는 "없음"이 아니라 "확인 안 됨"입니다.** 앱 온보딩에 "없어요" 버튼이 없어 빈 답이 전부
  미확인으로 저장되기 때문입니다(2026-09-14 안드로이드 확인). 필드 생략도 같습니다. 프롬프트에
  `(확인 안 됨)`으로 들어갑니다 — **알러지에서 "없음"과 "모름"은 처방이 달라지는 값**이라,
  "없음"이라고 적으면 모델이 확인된 사실로 읽습니다. 앱이 "없어요"를 받게 되면 그때 값에
  상태를 얹습니다
- 원소는 **항목 하나씩** 넣어 주세요. 저장된 값이 `", "`로 이어진 한 줄이면 백엔드에서 나눠
  배열로 보내면 됩니다(우리 쪽 변경 없음)
- 정의되지 않은 하위 필드가 오면 422입니다(`extra="forbid"`)

> **`question_candidates`가 `true`가 아닌 턴에 보내면 422입니다.**
>
> 프롬프트에 들어가는 것이 문제가 아니라 **요청 본문에 실려 오는 것**이 문제입니다. 편의상 매 턴
> 붙여 보내기 시작하면 20턴짜리 문진에서 건강정보가 20번 오가고, 서버 로그·에러 리포트에 남는
> 표면이 그만큼 늘어납니다. 아무도 모르게. 그래서 부탁하지 않고 거부합니다.
>
> 메시지: `patient_profile은 question_candidates: true인 턴에만 보낼 수 있습니다`

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
- **`GET /health`가 서버가 실제로 검증하는 계산식을 낸다**(2026-09-11 추가).

  ```json
  { "status": "ok", "hmac_enforced": true, "hmac_required": true,
    "signing": { "template": "{method}.{path}.{timestamp}.{request_id}.", "algorithm": "hmac-sha256-hex" } }
  ```

  `signing`은 `auth.sign()`이 쓰는 템플릿에서 나온다 — 손으로 관리하는 버전 번호가 아니라서
  형식을 바꾸면 이 값이 자동으로 따라간다. **붙이기 전에 이 값과 자기 계산식을 대조하면**
  배포본이 옛 형식인 상태를 바로 잡을 수 있다(2026-09-11에 그게 안 잡혀서 벡터 10 pass인데
  운영만 401이 났다). `scripts/check_ai_server.py`가 이 대조를 자동으로 한다.

  `hmac_enforced`가 `false`면 **서명 없이 모든 요청이 통과하는 상태**다. 노출이 아니다 —
  꺼져 있으면 어차피 아무 요청이나 통과하니 이미 알 수 있는 사실이다.
- 서버 설정: `MEDIMATE_HMAC_SECRET`, `MEDIMATE_HMAC_HEADER_*`, `MEDIMATE_HMAC_MAX_SKEW_S`,
  **`MEDIMATE_REQUIRE_HMAC`**. 키가 없으면 기동 로그에 경고를 남긴다(2026-09-11).
  이전에는 조용히 꺼져서 운영이 무검증으로 떠 있는 걸 아무도 몰랐다.
- **`MEDIMATE_REQUIRE_HMAC=1`이면 시크릿 없이 기동하지 않는다**(2026-09-11 추가, 백엔드 요청).
  `1` `true` `yes` `on`을 켜는 값으로 받는다(대소문자·공백 무관). **기본은 꺼짐** —
  로컬 개발과 테스트가 시크릿 없이 돌아야 하므로, 운영 env에서 켜는 구성이다.
  env에 값을 넣는 것은 배포하는 쪽 몫이다.

  `/health`의 **`hmac_required`가 그 플래그를 서버가 실제로 읽었는지** 알려준다.
  넣었는데 `false`로 보이면 값이 켜는 값이 아니거나 컨테이너가 그 env를 못 받은 것이다.
  (`hmac_enforced`는 시크릿 유무만 보므로 이 구분을 못 한다 — 2026-09-11에 백엔드가
  env를 넣고 켠 줄 알았는데 우리 코드에 변수가 없어 아무 일도 안 일어난 적이 있다.)
  **한쪽만 채우면 AI가 전 요청을 거부한다** — 양쪽을 같은 값으로 동시에 채운다

### 배포 — Dockerfile (저장소 루트)
`docker build -t medimate-ai . && docker run -p 8000:8000 --env-file .env medimate-ai`. 모델 없음, 1 vCPU·1GB. 폰이 뽑았든 서버가 뽑았든 백엔드에게는 같은 `state`·`card`다.
발화 원문은 폰에서 추출 JSON으로 바뀌어 오므로 백엔드를 거치지 않는다.

가드 규칙과 근거: `evals/RESULTS.md` "엔진 가드", 코드 `src/medimate/dialog/guard.py`.
