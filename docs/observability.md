# 관측 — Langfuse 트레이싱 (2026-09-22)

LLM 호출 하나 = generation 하나. 붙는 자리는 둘이고 그 위를 지나는 호출은 전부 남는다.

| 자리 | 덮는 것 |
|---|---|
| `llm/providers.py: LLMExtractor._call` | 진료 전 추출(v3·v4·v5), memo 조각내기·분류·재방문 리더, 후보 선택 Nova, 질문 후보. 모든 공급자 |
| `span/select.py: JevSelector.select` | Jev(SDK가 달라 따로) |

러너·API는 `obs/tracing.context()`로 요청 단위 trace를 열고 속성을 얹는다.

| 어디 | trace 이름 | session_id | tags | metadata | score |
|---|---|---|---|---|---|
| `evals/run.py` (진료 전 eval) | `extract-eval` | `extract-eval:<model>:<prompt_version>` | eval, 카테고리 | case_id·rep·asked_axis·prompt_version | `passed`, `safety_ok`, `fail:D4` 같은 실패 항목 |
| `evals/run_span.py` (후보 선택 eval) | `span-select-eval` | `span-eval:<model>` | eval, 그룹, 축, 선택기 | case_id·idx·group·axis·selector·gold | `em`, `confidence`(Jev) |
| `api/app.py` 진료 전 턴 | `previsit-turn` | request_id | api, previsit | turn·asked_axis | — |
| `api/app.py` 진료 후 메모 | `postvisit-memo` | request_id | api, postvisit | — | — |

generation에는 model·입력(system/user)·출력 원문·토큰·비용(가격표 기준)·version(prompt_version)이 실린다.
대시보드에서 **프롬프트 버전(version)·카테고리(tag)·선택기(tag)별로 통과율(score)·비용·지연**을 나눠 볼 수 있다.

## 환경변수

```
LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL   # 없으면 트레이싱이 조용히 꺼진다
MEDIMATE_TRACING=off        # 키가 있어도 끈다
MEDIMATE_TRACE_CONTENT=on   # 입력·출력 본문을 보낸다. 기본은 가림(길이만)
MEDIMATE_ENV=eval|prod      # Langfuse environment. 기본 local
```

## 본문 가림 — 왜 기본이 가림인가

호스트가 `us.cloud.langfuse.com`이다. 환자 발화 본문을 보내면 **미국으로 나간다.** 국외 이전 안내(#111)는 아시아
6개 리전(Bedrock)만 적혀 있고, 실사용 수집 동의(#113)도 트레이싱을 전제하지 않았다. 그래서 운영 API는 본문을
가린 채 토큰·지연·버전·request_id만 남기고, eval 러너는 우리가 만든 케이스라 `MEDIMATE_TRACE_CONTENT=on`으로 켠다.
운영에서 본문을 켜려면 동의문·안내문에 Langfuse(미국)를 넣고 팀이 결정한 뒤에.

## 프로젝트 둘 — eval과 운영 (2026-09-23, #127)

| 프로젝트 | 무엇이 쌓이나 | 키가 있는 곳 |
|---|---|---|
| eval(기존) | 우리 케이스로 돌린 eval·스모크. 본문 켬 | 로컬 `.env`의 `LANGFUSE_PUBLIC_KEY`/`SECRET_KEY` |
| `medical-mate-prod` | 운영 서버의 실제 문답·메모 | 운영 서버 env(백엔드) · 로컬 조회용은 `.env`의 `LANGFUSE_PROD_PUBLIC_KEY`/`SECRET_KEY` |

- 트레이싱 코드는 `_PROD_` 이름을 **읽지 않는다.** 로컬에서 운영 키를 기본 이름에 넣으면 eval이 운영 프로젝트에 섞인다
- 운영이 켜졌는지·본문이 나가는지는 `/health.tracing: {enabled, content, env, host}`(#128)
- 운영 본문 수집 순서: 웹 안내문 반영 → 앱 개인정보처리방침 → 백엔드 `MEDIMATE_TRACE_CONTENT=on`. 이전받는 자는 **ClickHouse, Inc.**(Langfuse 운영사,
  무료 요금제 기준. 유료 Pay-as-you-go면 약관상 Langfuse GmbH)
- 2027-01-15에 `medical-mate-prod` 프로젝트째 삭제(#115)

`user_id`는 쓰지 않는다. 개인 식별자를 밖으로 보내지 않는다. 세션은 request_id까지만.

## 확인 방법 (호출 0)

```
uv run python -c "from dotenv import load_dotenv; load_dotenv('.env'); from medimate.obs import tracing; print(tracing.client().auth_check())"
```
`True`면 키가 맞다. 스모크는 `run_span --selector jev --limit 2 --yes`(비용 0) 뒤 Langfuse에서 `span-select-eval`을 본다.
점수는 수집이 몇 초 늦는다 — 바로 안 보이면 잠시 뒤 다시.

## 안 한 것

- 데이터셋·실험(Langfuse Datasets) 연동 — 우리 시트(`evals/*.jsonl`)를 올리면 UI에서 실행 비교가 된다. 다음
- 프롬프트 관리(Langfuse Prompts) — 프롬프트는 저장소가 버전을 들고 있다. 옮길 이유가 생기면
- 사용자 피드백 점수 — 앱이 카드 값을 고치는 이벤트(1q-2)를 score로 보내면 실사용 정답이 쌓인다. 백엔드 합의 후
