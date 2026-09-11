# AI 서버 배포 요구사항

2026-09-11 갱신. 배포 구성이 확정된 뒤의 최종본이다. 관련 이슈 #7, 백엔드 PR #45.

## 한 줄

**컨테이너 하나, 1 vCPU · 512MB~1GB, GPU 없음, 무상태.** 외부 호출은 Bedrock 하나뿐이다.

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

## 무상태다 — 운영에서 빠지는 것들

세션 상태를 **백엔드가 들고 다닌다**(`state`를 요청·응답으로 주고받음). AI 서버는 아무것도 기억하지 않는다.

- 세션 스토어(Redis 등) **불필요** · 스티키 세션 **불필요** · 영속 볼륨 **불필요** · DB **불필요**
- 재시작·재배포에 세션이 안 끊긴다. 아무 인스턴스나 받으면 된다

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

```
MEDIMATE_PROVIDER=bedrock
MEDIMATE_MODEL=apac.amazon.nova-pro-v1:0
MEDIMATE_BEDROCK_REGION=ap-northeast-2
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
MEDIMATE_HMAC_SECRET=          # 비워둠
```

- **`OPENAI_API_KEY`는 필요 없다.** Bedrock은 IAM 인증이다
- `MEDIMATE_HMAC_SECRET`이 비면 검증 미들웨어를 아예 붙이지 않는다(`api/auth.py`).
  **한쪽만 채우면 AI가 백엔드 요청을 전부 거부한다.** 같은 사설망이므로 양쪽 다 비워 둔다

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
      "arn:aws:bedrock:ap-northeast-2:169523632526:inference-profile/apac.amazon.nova-pro-v1:0",
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

모델을 바꾸면 이 목록도 바뀐다. `global.` 프로파일은 분산 범위가 다르다(예: haiku는 3개).

## 비용

추출이 폰으로 내려가면서 외부 호출이 세션당 한 번으로 줄었다. **인스턴스도 토큰도 작다.**

| | |
|---|---|
| 질문 후보 1회 | 입력 2,288 tok · 출력 200 tok → **세션당 $0.0025** |
| 진료 전 추출 | **$0** — 폰에서 돈다(Qwen3-1.7B) |
| 진료 후 메모 분류 | **$0** — 폰. 서버는 실패 시 폴백만 |

Bedrock 몫으로 $48을 남긴다면 **약 19,000세션**이다. 발표 규모에서 토큰 비용은 문제가 되지 않는다.

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
