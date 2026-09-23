# AI 서버 배포 요구사항

2026-09-11 갱신. 배포 구성이 확정된 뒤의 최종본이다. 관련 이슈 #7, 백엔드 PR #45.

## 한 줄

**컨테이너 하나, 1 vCPU · 512MB~1GB, GPU 없음, 세션 무상태.** 외부 호출은 Bedrock 하나뿐이다.

## 확정된 구성

| | |
|---|---|
| 실행 | AWS Lightsail 1대(`small_3_0` · 2 vCPU · 2GB · Ubuntu 24.04 · x86_64)에 Docker Compose |
| 이미지 | `ghcr.io/medical-mate/ai` — `:latest` `:main` `:sha-<커밋>` |
| 모델 | Bedrock **`apac.amazon.nova-pro-v1:0`** (서울) |
| 인증 | Bedrock은 IAM. Lightsail은 역할을 못 붙여 **액세스 키를 서버에 둔다** |

이미지는 main 머지 시 GitHub Actions가 굽는다(`.github/workflows/publish-image.yml`).
**컨테이너를 실제로 띄워 `/health`가 200인지 확인한 뒤에만 푸시한다.** 서버는 `pull`만 한다.

저장소가 private이라 GHCR 패키지도 private이다. 서버 토큰에 `medical-mate/ai` 패키지 읽기 권한이 필요하다
(`docker compose pull`에서 401이 나면 이것이다).

## 실측 (2026-09-11)

| 항목 | 값 |
|---|---|
| 메모리(RSS) | **70MB** — 인터프리터 13.4 + 앱 30.5 + 온톨로지 1.8 + boto3 9.9 + Bedrock 클라이언트 14.7 |
| 기동 | 0.35초 + 첫 요청 때 온톨로지 로드 19ms |
| 온톨로지 요청 처리 | 0.2ms (검색), LLM 호출 없음 |
| 베이스 이미지 | `python:3.12-slim` |
| 모델 가중치 | **없음.** 추출은 폰, 생성은 Bedrock |

백엔드가 잡은 `mem_limit: 192m`이면 2.7배 여유다. **그보다 조이지 않는 게 좋다** —
2026-09-10 시점 46MB는 boto3 이전 값이고, Bedrock 전환으로 24MB 늘었다.

## 세션은 무상태다 — 운영에서 빠지는 것들

세션 상태를 **백엔드가 들고 다닌다**(`state`를 요청·응답으로 주고받음). AI 서버는 아무것도 기억하지 않는다.

- 세션 스토어(Redis 등) **불필요** · 스티키 세션 **불필요** · DB **불필요**
- 재시작·재배포에 세션이 안 끊긴다. 아무 인스턴스나 받으면 된다

예외는 선택적인 일일 예산 상태 파일 하나다. `MEDIMATE_BUDGET_STATE_FILE`을 켜면 그 파일의
디렉터리에 작은 영속 볼륨이 필요하다. 이것은 세션 데이터가 아니라 날짜와 누적 추정액만 담는다.

`GET /health` 있고 Dockerfile에 HEALTHCHECK가 들어 있다.

## 모델 — Nova Pro

프롬프트 `questions-v5`를 고정하고 모델만 바꿔 비교했다(`evals/RESULTS.md`).

| 모델 | 통과 | 병명 위반 | 추측 어투 | 유효 항목 | S 카드 고유 | 지연 p50 | 세션당 |
|---|---|---|---|---|---|---|---|
| gpt-5.6-terra (이전 기준) | 20/20 | 0 | 0 | 79/79 | 82% | 2.21s | $0.0038 |
| **apac.amazon.nova-pro-v1:0** | **20/20** | **0** | **0** | 74/74 | **85%** | **1.08s** | **$0.0025** |
| apac.amazon.nova-lite-v1:0 | 20/20 | 0 | 0 | 89/91 | 76% | 1.20s | $0.0002 |
| global.anthropic.claude-haiku-4-5 | 20/20 | 0 | **6** | 76/76 | 80% | 2.05s | $0.0034 |

`gpt-5.6-terra`·`claude-sonnet-5`·`opus-5`는 Bedrock 카탈로그에 있지만 **신규 계정 등급으로 호출이 막힌다**
(`AccessDeniedException: not available for this account`). 사용 사례 양식 제출과 모델 계약 생성까지 해도 열리지 않았다.
Nova Pro가 이전 기준보다 나은 수치라 되돌릴 이유는 없다.

Lite·Haiku도 채점했다(2026-09-11). **Haiku만 추측 어투 6건**이 나와서 우리 선에 걸린다.
**Nova Lite는 76%로 문턱(30%)을 크게 넘고 값이 Nova Pro의 1/12다** — 비용이 문제가 되면 갈아탈 자리가 있다.
지금 바꾸지 않는 이유는 중복 2건과 고유 비율 열세뿐이다. 설정값 하나만 바꾸면 된다.

## 네트워크

### 아웃바운드

`bedrock-runtime.ap-northeast-2.amazonaws.com:443` 하나면 된다. **`api.openai.com`은 더 이상 필요 없다.**

### 인바운드

백엔드에서만 들어온다. 같은 호스트의 Docker 네트워크에 있으므로 호스트 포트를 열지 않는다.

## 환경변수 · 시크릿

> **`MEDIMATE_MODEL`은 문자열이 정확히 일치해야 한다.** 프롬프트 계열을 모델 id로 고르기
> 때문에 `apac.` 누락·대소문자 하나가 어긋나면 **조용히 `extract-v3`로 떨어진다** — 88케이스
> 84 → 76, 인젝션 방어 3/3 → 0/3이 되는데 응답은 200이고 카드도 멀쩡하다.
>
> **배포하면 `/health`의 `extractor`를 본다.** 턴을 안 태우고 확인된다.
>
> ```
> "extractor": { "provider": "bedrock",
>                "model_id": "apac.amazon.nova-pro-v1:0",
>                "prompt_version": "extract-v4-nova",   ← v3면 모델 id가 어긋난 것
>                "pricing_known": true }                ← false면 지출 상한이 못 센다
> ```
>
> 앞뒤 공백은 서버가 떼므로(`extractor_env()`) 그것만으로 떨어지지는 않는다.
> 여기 나오는 것은 **설정값**이다. 그 턴이 실제로 무엇으로 돌았는지는 `card.provenance`가 낸다.


```
MEDIMATE_PROVIDER=bedrock
MEDIMATE_MODEL=apac.amazon.nova-pro-v1:0
MEDIMATE_BEDROCK_REGION=ap-northeast-2
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
MEDIMATE_HMAC_SECRET=          # 비워둠

MEDIMATE_DAILY_BUDGET_USD=     # 웹 데모에서는 채운다. 비우면 상한 없음
MEDIMATE_BUDGET_RESET_TZ=Asia/Seoul
MEDIMATE_BUDGET_STATE_FILE=    # 선택. 예: /var/lib/medimate/budget.json
MEDIMATE_LIFESTYLE_AXIS=       # 비움=꺼짐. 앱이 "지침" 행을 그리면 1로. 켜기 전엔 지침이 약 칸에 접혀 나간다
MEDIMATE_CORS_ORIGINS=         # 웹이 우리를 직접 부를 때만. 쉼표 구분, `*` 금지
```

- **`OPENAI_API_KEY`는 필요 없다.** Bedrock은 IAM 인증이다
- `MEDIMATE_HMAC_SECRET`이 비면 검증 미들웨어를 아예 붙이지 않는다(`api/auth.py`).
  **한쪽만 채우면 AI가 백엔드 요청을 전부 거부한다.** 같은 사설망이므로 양쪽 다 비워 둔다
- `MEDIMATE_DAILY_BUDGET_USD` — 리셋 시간대(기본 KST) 자정 기준 당일 누적이 이 값에 닿으면 **LLM을 부르는
  경로만** 503으로 거절한다. 선택지 턴·폰 추출 턴·온톨로지 조회·`/health`는 계속 된다.
  비우면 상한이 없다. 값이 숫자가 아니거나 0 이하면 **기동에 실패한다**(오타로 조용히 꺼지면
  켠 줄 알고 4주를 떠 있게 된다)
- `MEDIMATE_BUDGET_RESET_TZ` — 생략하면 `Asia/Seoul`. IANA 시간대 이름을 쓰며 오타는 기동 실패한다
- `MEDIMATE_BUDGET_STATE_FILE` — 경로를 줄 때만 날짜와 누적액을 파일에 남긴다. 컨테이너에서
  쓰려면 그 경로의 디렉터리를 영속 볼륨으로 마운트해야 한다. 읽기·쓰기 실패는 문진을 막지 않고
  메모리 카운터로 계속하며 `/health`의 `persisted`가 `false`가 된다
- `MEDIMATE_CORS_ORIGINS` — 비면 미들웨어를 붙이지 않는다. `*`는 기동 실패.
  브라우저에 HMAC 시크릿을 둘 수 없어 **출처 목록이 사실상 유일한 문지기**가 되기 때문이다.
  백엔드가 프록시하면 요청이 서버에서 오므로 이 변수는 필요 없다
- **켜졌는지는 `/health`로 확인한다** — `llm_budget: {cap_usd, spent_today_usd, resets_at, persisted}`,
  `cors_origins`(개수). 꺼져 있으면 둘 다 `null`. `hmac_required`와 같은 이유로 낸다
- **트레이싱(Langfuse)** — `LANGFUSE_PUBLIC_KEY`·`LANGFUSE_SECRET_KEY`·`LANGFUSE_BASE_URL`, `MEDIMATE_ENV`(`prod`),
  `MEDIMATE_TRACE_CONTENT`(`on`이면 **본문이 Langfuse 호스트로 나간다** — 국외 이전 안내 #111과 함께), `MEDIMATE_TRACING=off`(끔).
  `.env`와 compose `environment:` **양쪽에**. 켜졌는지는 `/health.tracing: {enabled, content, env, host}` —
  `enabled`는 클라이언트가 실제로 만들어졌는지, `content`는 본문이 실제로 나가는지다(스위치만 켜고 키가 없으면 false)

## IAM 정책 — 서울 ARN만 넣으면 깨진다

`apac` 추론 프로파일은 **아시아 6개 리전 모델로 분산된다.** 서울 것만 허용하면 도쿄·싱가포르로
라우팅될 때 거부된다. `iam simulate-custom-policy`로 확인했다.

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
    "Resource": [
      "arn:aws:bedrock:ap-northeast-2:<AWS_ACCOUNT_ID>:inference-profile/apac.amazon.nova-pro-v1:0",
      "arn:aws:bedrock:ap-northeast-2::foundation-model/amazon.nova-pro-v1:0",
      "arn:aws:bedrock:ap-northeast-1::foundation-model/amazon.nova-pro-v1:0",
      "arn:aws:bedrock:ap-northeast-3::foundation-model/amazon.nova-pro-v1:0",
      "arn:aws:bedrock:ap-southeast-1::foundation-model/amazon.nova-pro-v1:0",
      "arn:aws:bedrock:ap-southeast-2::foundation-model/amazon.nova-pro-v1:0",
      "arn:aws:bedrock:ap-south-1::foundation-model/amazon.nova-pro-v1:0"
    ]
  }]
}
```

`<AWS_ACCOUNT_ID>`는 실제 계정 ID로 바꿔 넣는다. 저장소가 공개라 값을 적지 않는다
(`aws sts get-caller-identity --query Account --output text`로 확인). 계정 ID가 필요한 것은
**inference-profile ARN 하나뿐**이고, foundation-model ARN 6개는 그 자리가 비어 있다.

모델을 바꾸면 이 목록도 바뀐다. `global.` 프로파일은 분산 범위가 다르다(예: haiku는 3개).

## 비용

> **정정 (2026-09-15).** 아래 "폰에서 돈다 → $0" 전제가 **앱에서 성립하지 않는다.**
> 안드로이드 확인 결과, 사용자에게 보이는 AI 기능은 **전부 서버**가 처리한다 —
> 문답·축 추출, 질문 후보, 진료 후 메모 분류 셋 다 백엔드 API를 거쳐 우리에게 온다.
> 폰에서 도는 것은 음성 받아쓰기(ML Kit STT)뿐이고, **받아쓴 텍스트는 서버로 온다.**
> 온디바이스 LLM은 골격·엔진 검증까지 와 있고 화면 흐름에는 안 붙어 있다(앱 #142, 미머지).
>
> 그래서 **앱의 비용 구조는 웹 데모와 같다.** 아래 "웹 데모는 전제가 다르다" 절의 표가
> 지금은 **앱에도 그대로 적용된다.**
>
> | 세션당(실측 단가) | 6턴 | 11턴(실측) |
> |---|---|---|
> | 진료 전 추출 `extract-v4-nova` | $0.0165 | $0.0302 |
> | 질문 후보 1회 | $0.0025 | $0.0025 |
> | 진료 후 메모 분류 1회 | ≤ $0.0025 | ≤ $0.0025 |
> | **합** | **≈ $0.0215** | **≈ $0.0353** |
>
> Bedrock 몫 $48이면 **19,000세션이 아니라 1,400~2,200세션**이다. 한 자릿수가 아니라
> **한 자리 수 단위가 틀렸다** — 일일 상한을 이 값으로 다시 잡아야 한다.
>
> 징후는 이미 있었다. 아래 "문답은 6턴이 아니라 11턴이었다"에서 **앱 세션 하나가 통째로
> 예산 카운터에 남았다.** 폰에서 돌았다면 카운터에 안 남는다. 그때 알아챘어야 했다.

아래는 온디바이스 추출을 전제로 쓴 원본이다. **앱이 `OnDeviceExtractor`를 실제로 부르기
전까지는 위 정정이 맞다.**

추출이 폰으로 내려가면서 외부 호출이 세션당 한 번으로 줄었다. **인스턴스도 토큰도 작다.**

| | |
|---|---|
| 질문 후보 1회 | 입력 2,288 tok · 출력 200 tok → **세션당 $0.0025** |
| 진료 전 추출 | **$0** — 폰에서 돈다(Qwen3-1.7B) ← **지금은 아니다. 위 정정** |
| 진료 후 메모 분류 | **$0** — 폰. 서버는 실패 시 폴백만 ← **지금은 아니다. 위 정정** |

### 웹 데모는 전제가 다르다 (2026-09-14) — 앱도 같다 (2026-09-15)

해커톤이 APK를 안 받아 **웹 데모를 9/20까지 내고 심사 기간 동안 공개로 둔다.** 심사는 두 단계다(2026-09-21 확인) —
**1차 공개 심사 9/21~10/5**(실사용자가 많이 붙을 수 있는 구간), **2차 심사 ~10/17**(심사위원 중심, 트래픽 적음).
일일 상한은 1차 기준으로 `$12`(#110). 잔액은 중간중간 `scripts/bedrock_spend.py --credit 196`으로 직접 보고 필요하면 새 이슈로 조정한다.
웹에는 폰 모델이 없다. 그래서 위 표의 **"추출 $0"이 웹에서는 성립하지 않는다** — 서버 프로필로
돌아 매 턴 Nova를 부른다.

| 세션 하나 | 호출 | 단가 |
|---|---|---|
| 진료 전 추출 | 턴당 1회 × 6턴 | **$0.00275/턴** (실측, `extract-v4-nova`) → $0.0165 |
| 질문 후보 | 종료 턴 1회 | $0.0025 (실측) |
| 진료 후 메모 분류 | 1회 | ≤ $0.0025 (Nova 미측정. 후보와 프롬프트 크기가 비슷해 그 값으로 잡음) |
| **합** | | **≈ $0.021/세션**(6턴 가정) · 아래 실측 참고 |

**어떻게 쟀나**: 저장된 eval 원본(88호출)의 토큰 수에 가격표를 곱했다. 재호출 없이
`--report`로 다시 계산할 수 있다.

운영은 **`extract-v4-nova`**로 돈다(Nova 전용 프롬프트. 모델 id로 고른다 —
`providers.prompt_family_for`. 카드의 `provenance.prompt_version`에 그 이름이 찍힌다).
**v3보다 입력 토큰이 1.59배**다 — 인젝션 방어 규칙과 예시가 매 호출에 실린다.

| 프롬프트 | 88케이스 입력 tok | 호출당 | 88케이스 통과(가드) |
|---|---|---|---|
| extract-v3 | 163,913 | $0.00185 | 76/88 |
| **extract-v4-nova** | **261,074** | **$0.00275** | **84/88** |

지연은 둘 다 p50 0.72s로 같다. `evals/RESULTS.md`의 두 절을 보라.

### 실측 한 점 — 문답은 6턴이 아니라 11턴이었다 (2026-09-14, 백엔드 #7)

상한을 켜고 배포를 확인하는 사이에 **앱 세션 하나가 통째로 카운터에 남았다.** 일부러 잰 것이
아니라 그 창에 그 세션밖에 없었다.

```
세션 시작 1 + 턴 11 + 질문 후보 1 + 세션 시작 2(통로 확인)  →  spent_today_usd $0.019967
```

**여기서 고칠 것은 턴 수다.** 위 표는 6턴을 가정했는데 실제 문답은 **11턴**이었다. 후보
$0.0025를 빼면 추출이 $0.0175이고, v3 단가($0.00185)로 나누면 **LLM을 부른 턴이 약 9.4회**다
(선택지만 온 턴은 호출이 없다).

| | 세션당 | $2/일이면 |
|---|---|---|
| 실측 (v3, 앱 세션) | $0.020 | 100세션 |
| 같은 세션을 **v4**로 환산 | **≈ $0.029** | **≈ 70세션** |
| 위 표의 6턴 가정 (v4) | $0.021 | 95세션 |

**계획 숫자는 실측 쪽을 쓴다** — 6턴은 우리가 녹화한 이상적인 흐름이고, 사람은 더 많이 말한다.

> **이 값은 웹의 값이 아니라 하한이다.** 그 세션이 온디바이스 추출을 썼는지 확인되지 않았다
> (백엔드 로그가 재배포로 날아갔다). 썼다면 **웹은 폰 모델이 없어 추출도 서버로 도니 더 높다.**

**일일 상한 $2**(`MEDIMATE_DAILY_BUDGET_USD=2`)는 **하루 70세션 남짓**으로 읽는다. 심사 4주면 상한에
매일 닿아도 $56이고, 실제로는 그만큼 안 온다. 상한은 "실수로 새는 것"을 막는 값이지
예상 사용량이 아니다.

방어는 두 층이다. 한 층이 죽어도 다른 층이 남게 **일부러 다른 곳에 둔다.**

| | |
|---|---|
| 우리 쪽 | `MEDIMATE_DAILY_BUDGET_USD`. 우리 가격표 기준 **추정치**를 세고, 상태 파일을 설정하지 않으면 재시작 시 0이 된다 |
| 클라우드 쪽 | AWS 예산 알림 $30(백엔드). 실제 청구 기준이고 프로세스와 무관하다 |

> **개발이 활발한 동안에는 이중 방어의 순서가 사실상 뒤집혀 있다**(백엔드 관찰, 2026-09-14).
> 하루에 AI 두 번·백엔드 네 번 배포했고 **올릴 때마다 카운터가 0이 됐다.** 그래서 지금 실질
> 상한은 예산 알림 쪽이다. 심사 기간에 들어가 배포가 뜸해지면 제 값을 하지만,
> **"켜 뒀으니 하루 $2를 넘지 않는다"로 읽으면 안 된다.**

우리 쪽 한계 둘을 알고 둔다. **상태 파일을 설정하지 않거나 볼륨 쓰기가 실패하면 재시작 시
카운터가 0이 되고**, **세는 값이 청구서와 다를 수 있다**(가격표 기준). 그래서 `/health`로
`persisted`를 포함해 내보낸다.

rate limit은 우리가 넣지 않는다. 백엔드 프록시 뒤면 IP가 전부 백엔드라 무의미하고,
공개 엣지(Caddy·백엔드)에서 거는 것이 맞다.

**서버가 멎어도 화면은 나와야 한다.** 완주 세션 하나를 `docs/examples/demo-session.json`에
녹화해 뒀다(실제 API 응답 전문, 식별자 없음). 프론트 폴백용이고, 드리프트는 테스트가 잡는다.

> **정정(2026-09-11).** 이전 판의 `$0.0013`은 Terra의 토큰수(입력 1,223)에 Nova 단가를 곱한 추정값이었다.
> 저장된 원본으로 실측하니 **Nova는 같은 한국어 프롬프트를 2,288 토큰으로 센다 — Terra의 1.9배.**
> 회당 $0.0025가 맞다. 결론(Terra보다 싸다, 규모에서 문제없다)은 그대로다.

## 지연

**LLM 왕복이 전부다.** Nova Pro 실측 p50 1.08초, p90 1.62초. 우리 처리 자체는 ms 단위다.
Caddy의 90초 타임아웃이면 충분하다.

## 남은 확인

- [ ] GHCR 패키지 접근 권한 — 서버 토큰이 `medical-mate/ai`를 읽을 수 있는지
- [ ] 서버에서 `uname -m` → `x86_64` 확인 (이미지가 amd64 단일이다)
- [ ] 첫 `docker compose up -d` 후 `/health` 200
