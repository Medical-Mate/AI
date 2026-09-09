# 진료 메이트 — AI

진료 메이트의 **AI 계층**입니다. 진료 전 문진으로 브리핑 카드를 만들고, 진료 후 메모를 네 묶음으로 정리합니다.
백엔드·안드로이드는 `Medical-Mate` 조직의 별도 저장소입니다. 협업 규칙은 **[`GIT_CONVENTION.md`](./GIT_CONVENTION.md)**.

지금 상태와 다음 할 일은 **[`docs/HANDOFF.md`](./docs/HANDOFF.md)** 한 장에 있습니다.

---

## 무엇을 하나

| 챗봇 | 흐름 | 결과 |
|---|---|---|
| **① 진료 전 브리핑 카드** | 인체도에서 부위 선택 → 축별 질문(칩 또는 직접 입력) → 카드 | 8축 카드 + 부위 기반 진료과 안내 + 의사에게 전하는 말 |
| **② 진료 후 기록** | 환자가 들은 말을 자유 메모로 적음 → 문장 단위 분류 | 들은 소견 · 검사 · 약·지시 · 다시 오기 4묶음 + 재방문 날짜 |

두 흐름 모두 **LLM은 추출·분류만** 합니다. 질문은 템플릿, 카드 값은 발화·메모 원문 그대로, 날짜는 결정론 계산입니다.

추출 모델은 두 경로입니다.

- **서버**: `gpt-5.6-terra` (폴백). 진료 전 85/88, 진료 후 99%
- **온디바이스**: Qwen3-1.7B Q4_0, llama.cpp Hexagon NPU(갤럭시 S24 울트라 실측). 발화가 폰을 떠나지 않습니다. 진료 전 61/88, 진료 후 93%, 안전 위반 0

수치의 근거는 [`evals/RESULTS.md`](./evals/RESULTS.md).

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
| `POST /v1/previsit/turns` | 턴 처리(발화 또는 폰 추출 결과, 칩 선택) | 〃 |
| `GET /v1/ontology/body-map` | 인체도 부위 마스터 | 〃 |
| `POST /v1/postvisit/memo` | 진료 후 메모 → 4묶음 카드. `labels`를 주면 LLM 없이 재조립 | [`docs/api-postvisit.md`](./docs/api-postvisit.md) |

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
uv run python scripts/previsit_chat.py --site "허리 가운데" --verbose   # 진료 전 문진 (Terra 실호출, 턴당 약 $0.004)
uv run python scripts/postvisit_memo.py "위염 초기래요. 2주 뒤 오라고."   # 진료 후 분류 (로컬 llama-server 필요, $0)
```

실호출이 들어가는 명령은 **호출 수·비용을 먼저 계산해 보여주고** 모델당 지출 상한(`--budget`, 기본 $0.50)을 지킵니다. 사비로 운영합니다.

온디바이스 모델을 PC나 폰에서 돌리는 절차는 [`evals/ondevice/README.md`](./evals/ondevice/README.md).

---

## 디렉터리

```text
src/medimate/schema/    카드 모델. card.py 진료 전 8축, postvisit.py 진료 후 4묶음, export.py 백엔드 형식 어댑터
src/medimate/dialog/    문진 엔진(engine·spec·questions), 가드(guard), 메모 분류·문장 분리·날짜(memo), 용어 부위 병기(widening)
src/medimate/api/       무상태 FastAPI(app), HMAC 미들웨어(auth)
src/medimate/llm/       프롬프트(버전 명시), 공급자 어댑터, 가격표·지출 가드, 메모 분류기
src/medimate/ontology/  부위 그래프 로더(조상·LCA·넓히기)
src/medimate/evals/     결정론 채점기(D1~D10)와 러너. LLM judge 없음
evals/                  케이스(진료 전 88, 진료 후 20+10), 병명 사전, RESULTS.md, 온디바이스 절차. results/는 gitignore
data/ontology/          부위 온톨로지 (UBERON 추출 + 수동 보강). 앵커 9 · 구역 25 · 구조 40
docs/                   설계(ai-design), API 계약, 결정 기록(decisions/), 의료인 확인서, 안드로이드 전달 문서, HANDOFF
scripts/                대화형 클라이언트, 온톨로지 생성 스크립트
```

커밋 scope는 `prompt` `rag` `embedding` `model` `eval` `pipeline` 중 하나입니다 (`GIT_CONVENTION.md` §5).

---

## 문서 지도

| 알고 싶은 것 | 문서 |
|---|---|
| 지금 어디까지 됐고 뭘 기다리나 | [`docs/HANDOFF.md`](./docs/HANDOFF.md) |
| 왜 이렇게 설계했나 | [`docs/ai-design.md`](./docs/ai-design.md) |
| 진료 후를 문답이 아닌 메모 분류로 바꾼 이유 | [`docs/postvisit-b1.md`](./docs/postvisit-b1.md) |
| 모델 선택·진료과 안내 결정 근거 | [`docs/decisions/`](./docs/decisions) |
| 안드로이드가 받는 온디바이스 묶음 | [`docs/android-ondevice-handoff.md`](./docs/android-ondevice-handoff.md) |
| 의료인에게 확인받을 항목 | [`docs/advisor-checklist.md`](./docs/advisor-checklist.md) |
| 온톨로지 재생성 | [`data/ontology/README.md`](./data/ontology/README.md) |
