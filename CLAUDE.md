# CLAUDE.md — 진료 메이트 AI 저장소

## 이 저장소가 하는 일
진료 메이트의 AI 계층. 진료 전 카드 챗봇(S2 증상 문답 → 브리핑 카드), 부위 온톨로지,
검색, 가드레일 판정, 평가. 백엔드·안드로이드는 별도 저장소.
설계 근거는 `docs/ai-design.md`, 협업 규칙은 `GIT_CONVENTION.md`. 둘 다 이 문서보다 우선한다.

## 절대 지키는 선 (구조로 보장, 프롬프트로 부탁하지 않는다)
- 진단하지 않는다. 온톨로지에 병명 노드를 두지 않는다. 출력 스키마에 진단 자리를 만들지 않는다
- 문서·의사가 말한 병명을 환자가 옮긴 것은 기록 가능, AI가 추론한 병명은 불가
- 설명문을 생성하지 않고 인용한다. 우리가 보증하는 것은 "출처와 일치한다"뿐
- 예외: 응급 안내는 한다
- 타인의 진료기록은 받지 않는다. 사진은 서버로 보내지 않는다. 필요한 필드만 뽑는다(화이트리스트)
- 증상 → 진료과 추천은 감별이므로 하지 않는다. 부위 노드 속성으로만 진료과를 "안내"한다(증상 축 미참조, 안내 어투). `docs/decisions/2026-09-04-department-guidance.md`

## 출력 형식
- 한국어를 포함한 모든 non-ASCII 문자는 코드 변환 없이 글자 그대로 출력한다.
  질문/선택지 UI의 라벨과 설명에도 동일하게 적용한다.
- 코드 주석·커밋 본문·문서는 한국어. 식별자·커밋 제목은 영어.
- Windows 콘솔 출력 스크립트는 UTF-8을 명시한다 (`evals/run.py` 방식).

## 구조
```
src/medimate/schema/   카드 내부 모델. card.py 공통 뼈대+진료 전 8축, postvisit.py 진료 후 6축. export.py가 백엔드 형식 어댑터
src/medimate/dialog/   문진 상태 기계(진료 전·후 공통). spec.py가 축·질문·카드 명세, questions/postvisit_questions.py 템플릿, widening.py 용어 부위 병기, state.py 직렬화 상태
src/medimate/api/      무상태 FastAPI(세션 시작·턴 처리). 백엔드가 state를 들고 다닌다. docs/api-previsit.md
src/medimate/llm/      Extractor 프로토콜, 프롬프트(버전 명시), 공급자 어댑터, 가격표·지출 가드
src/medimate/evals/    결정론 채점기(D1~D10)와 러너
evals/                 확정 시트, cases.jsonl, 병명 사전, RESULTS.md. results/는 gitignore
data/ontology/         무릎·어깨 부위 온톨로지(UBERON 추출 + 수동 보강)
docs/decisions/        팀 공유용 결정 리포트
```

## 명령
```
uv sync --group dev --group providers --group api
uv run pytest -q                                  # 커밋 전 필수
uv run ruff format . && uv run ruff check .
uv run python -m medimate.evals.run --dry-run     # 호출 0
uv run uvicorn medimate.api.app:app --reload      # API 로컬 실행, 문서 /docs
uv run python -m medimate.evals.run --report evals/results/<model>.jsonl   # 재채점, 호출 0
uv run python -m medimate.evals.run --provider openai --model gpt-5.6-terra  # 실호출
```

## 비용 — 사비로 운영한다
- LLM 호출이 들어가는 작업은 호출 수·토큰·달러를 먼저 계산해 보여주고 진행한다
- 러너의 모델당 지출 상한(`--budget`, 기본 $0.50)을 끄지 않는다
- LLM judge, 대규모 k, 케이스 수십 개 확장은 먼저 제안하고 승인 후에
- 결과 원본은 저장하고 채점 로직 수정은 재채점으로 해결한다. 재호출하지 않는다
- 프롬프트를 바꾸면 비교 대상 모델을 전부 다시 돌린다. 한 모델만 v2로 두지 않는다

## 모델
- Extractor 기본: `gpt-5.6-terra`. 동등 대안 `claude-sonnet-5`. 근거: `docs/decisions/2026-09-03-extractor-model.md`
- 모델 ID는 `src/medimate/llm/providers.py` 가격표의 문자열을 그대로 쓴다. 날짜 접미사 임의 추가 금지
- Gemini는 사고 토큰이 출력 한도에 포함된다. max_output_tokens 8192 유지

## 스키마 소유
- 백엔드 전달 JSON 스키마는 백엔드가 정한다. AI는 채운다. 맞추는 곳은 `schema/export.py` 한 곳
- 카드 완성을 강제하지 않는다. 환자가 언제 끝내도 그 시점 카드가 결과
- 모든 값에는 evidence(발화 원문)가 붙는다. 근거 없는 값은 만들지 않는다

## Git
- main 직접 push 금지. 브랜치 `<kind>/<desc>` → PR → Squash merge
- 커밋은 논리 단위로 나눈다. scope는 `prompt` `rag` `embedding` `model` `eval` `pipeline`
- 커밋은 로컬까지. push·PR은 사용자 확인 후
- `.env`, `evals/results/`, `*.obo`, 진료기록 원본은 커밋하지 않는다
- ruff는 `scripts/`를 제외한다 (기존 추출 스크립트는 별도 정리)

## 결정 대기 (착수 근거로 쓰지 말 것)
- 디자이너 인체도 부위 단위 → 앵커 노드. 지금은 무릎·어깨 임시
- 부위 여러 개 선택 시 대화 방식 (순차 진행안 제안 중)
- 국가건강정보포털 라이선스 회신, 의료인 자문 회신
- OCR·화이트리스트 파서는 챗봇 완료 후
