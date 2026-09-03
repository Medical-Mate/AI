# Git Convention

백엔드 개발자, AI 엔지니어, 안드로이드 개발자가 하나의 GitHub Organization에서 각 담당 저장소를 운영하며 협업하기 위한 규칙입니다.

이 문서는 3인 소규모 팀이 바로 사용할 수 있도록 꼭 필요한 규칙만 정의합니다.

## 1. 저장소 구성

Organization 아래에 제품 영역별 저장소를 둡니다.

```text
organization/
├── backend-repo
├── ai-repo
└── android-repo
```

실제 저장소 이름은 프로젝트에 맞게 정합니다. 사람이나 담당자 이름보다 저장소의 역할이 드러나는 이름을 사용합니다.

- `backend-repo`: API, 서버 비즈니스 로직, 데이터베이스, 인증
- `ai-repo`: 모델 호출, 프롬프트, RAG, 임베딩, 평가 및 추론 파이프라인
- `android-repo`: Android UI, 앱 로직, 로컬 저장소, 모바일 네트워크 연동

저장소는 **담당자가 아니라 독립적으로 개발·배포·버전 관리할 수 있는 제품 영역**을 기준으로 분리합니다. 담당자가 바뀌어도 저장소 구조는 유지될 수 있어야 합니다.

API 명세처럼 여러 저장소가 공유하는 파일은 처음에는 주 소유 저장소에서 관리합니다. 독립적인 버전 관리가 실제로 필요해지면 `api-spec` 또는 `shared` 저장소로 분리합니다.

---

## 2. 브랜치 전략

각 저장소에서 단순한 GitHub Flow를 사용합니다.

### 기본 브랜치

- 각 저장소의 기본 브랜치는 `main`입니다.
- `main`은 항상 빌드·실행 가능한 상태로 유지합니다.
- `main`에 직접 push하지 않고 Pull Request(PR)를 통해 병합합니다.
- 가능하면 CI 통과와 팀원 1명의 승인을 병합 조건으로 설정합니다.

### 작업 브랜치

저장소가 이미 영역별로 분리되어 있으므로 브랜치명에 `backend`, `ai`, `android`를 반복하지 않습니다.

```text
<branch-kind>/<short-description>
```

예시:

```text
feat/user-login
fix/empty-model-response
refactor/network-layer
docs/local-setup
hotfix/payment-timeout
```

- 영문 소문자와 하이픈(`-`)을 사용합니다.
- `branch-kind`에는 보통 `feat`, `fix`, `refactor`, `docs`를 사용하고 긴급 수정에만 `hotfix`를 사용합니다.
- 하나의 브랜치는 하나의 Issue 또는 하나의 명확한 작업만 다룹니다.
- 직군별로 장기간 유지되는 브랜치나 `develop` 브랜치는 두지 않습니다.
- 작업 브랜치는 짧게 유지하고 병합 후 삭제합니다.

저장소와 브랜치의 역할은 다릅니다. 저장소는 제품 영역을 나누고, 작업 브랜치는 해당 저장소 안에서 기능·수정 단위를 안전하게 분리합니다.

---

## 3. 커밋 메시지 규칙

Conventional Commits 형식을 사용합니다.

```text
<type>(<scope>): <summary>
```

scope가 불필요하면 생략할 수 있습니다.

```text
<type>: <summary>
```

예시:

```text
feat(auth): add refresh token API
fix(rag): handle empty search results
refactor(network): separate API client
docs: update local setup guide
```

### 작성 원칙

- `type`은 변경의 성격을 나타내며 필수입니다.
- `scope`는 현재 저장소 안에서 변경된 모듈 또는 기능 영역을 나타내며 선택입니다.
- `backend`, `ai`, `android`는 이미 저장소로 구분되므로 일반적으로 scope에 사용하지 않습니다.
- 의미 없는 scope를 억지로 만들기보다 생략합니다.
- summary는 변경 결과를 짧고 구체적으로 작성합니다.
- 한글과 영어 모두 사용할 수 있지만 하나의 PR 안에서는 한 언어로 통일합니다.
- summary 끝에 마침표를 붙이지 않습니다.
- 커밋 하나에는 한 가지 논리적 변경만 담습니다.

상세한 배경이나 주의점은 빈 줄 뒤의 본문에 작성합니다.

```text
fix(auth): prevent duplicate signup

Reject a signup request when the email already exists.
Refs: #42
```

호환성을 깨는 변경은 `!` 또는 본문의 `BREAKING CHANGE:`로 표시합니다.

```text
feat(api)!: replace legacy login response
```

---

## 4. 공통 type 정의

| type | 용도 |
| --- | --- |
| `feat` | 사용자 또는 시스템에 새로운 기능 추가 |
| `fix` | 버그 수정 |
| `refactor` | 동작 변경 없이 코드 구조 개선 |
| `test` | 테스트 추가 또는 수정 |
| `docs` | 문서만 변경 |
| `build` | 빌드 도구, 의존성, 패키징 변경 |
| `ci` | GitHub Actions 등 CI/CD 설정 변경 |
| `perf` | 성능 개선 |
| `style` | 포맷, 공백 등 동작에 영향 없는 코드 변경 |
| `chore` | 위 항목에 속하지 않는 유지보수 작업 |
| `revert` | 이전 변경 되돌리기 |

`backend`, `ai`, `android`, `hotfix`는 변경의 성격이 아니므로 type으로 사용하지 않습니다.

---

## 5. 저장소별 scope 권장안

scope는 고정된 직군 태그가 아니라 저장소 내부 구조에 맞게 사용합니다. 아래 목록은 예시이며 실제 모듈과 디렉터리 이름을 우선합니다.

### Backend 저장소

| scope | 대상 예시 |
| --- | --- |
| `auth` | 로그인, 인증, 권한 |
| `user` | 사용자 및 프로필 |
| `payment` | 결제 |
| `api` | API 계약, 컨트롤러, 응답 형식 |
| `db` | 데이터베이스, 스키마, 마이그레이션 |
| `infra` | 서버 배포 및 실행 환경 |

```text
feat(auth): add social login endpoint
fix(payment): prevent duplicate payment requests
```

### AI 저장소

| scope | 대상 예시 |
| --- | --- |
| `prompt` | 프롬프트 및 출력 형식 |
| `rag` | 검색 증강 생성 파이프라인 |
| `embedding` | 임베딩 생성 및 저장 |
| `model` | 모델 호출 및 추론 설정 |
| `eval` | 평가 데이터와 평가 코드 |
| `pipeline` | 전처리·후처리 및 전체 처리 흐름 |

```text
feat(rag): add metadata filtering
fix(prompt): handle missing user context
```

### Android 저장소

| scope | 대상 예시 |
| --- | --- |
| `ui` | 공통 UI 및 디자인 시스템 |
| `login` | 로그인 화면과 흐름 |
| `profile` | 프로필 화면과 기능 |
| `network` | API 클라이언트 및 통신 |
| `storage` | 로컬 저장소 및 캐시 |
| `navigation` | 화면 이동 및 딥링크 |

```text
feat(login): add social login screen
refactor(network): separate authentication interceptor
```

scope가 지나치게 늘어나지 않도록 다음 원칙을 적용합니다.

- 실제 모듈이나 기능 영역을 나타내는 이름을 사용합니다.
- 비슷한 scope를 중복 생성하지 않습니다.
- 새 scope가 자주 필요하다면 저장소의 모듈 구조와 함께 팀에서 정리합니다.
- 단순한 저장소에서는 scope를 생략해도 됩니다.

---

## 6. 파일·디렉터리 단위 scope 선택법

scope는 작성자의 직군이 아니라 **현재 커밋에서 직접 변경하는 모듈 또는 기능 영역**을 기준으로 선택합니다.

```text
# Android 저장소
feature/login/       -> login
feature/profile/     -> profile
core/network/        -> network
core/designsystem/   -> ui

# AI 저장소
src/rag/             -> rag
src/prompts/         -> prompt
src/embeddings/      -> embedding
evals/               -> eval
```

선택 순서:

1. 한 모듈만 변경했다면 해당 모듈명을 사용합니다.
2. 여러 파일을 변경해도 하나의 기능이 중심이라면 그 기능의 scope를 사용합니다.
3. 저장소 전체 설정이나 문서 변경처럼 대표 모듈이 없다면 scope를 생략합니다.
4. 서로 독립적인 모듈 변경이라면 커밋을 나눕니다.
5. 한 커밋에 여러 scope를 쉼표로 나열하지 않습니다.

```text
# 권장
feat(login): add biometric authentication
test(login): cover biometric authentication failure

# 비권장
feat(login,network): add biometric authentication and clean up API client
```

한 기능을 구현하면서 보조 모듈도 변경해야 한다면 핵심 기능의 scope를 사용하고 PR 본문에 전체 영향 범위를 적습니다.

---

## 7. 좋은 예시와 나쁜 예시

### 좋은 예시

```text
# backend-repo
feat(auth): add email verification endpoint
fix(db): correct user migration constraint

# ai-repo
fix(model): retry requests after rate limit
test(rag): cover empty retrieval results

# android-repo
refactor(network): extract token authenticator
feat(profile): add profile image upload

# 어느 저장소에서나 가능
docs: update local setup guide
ci: run tests on pull requests
```

### 나쁜 예시

```text
backend: add login API
ai-fix: fix prompt
feat(backend): add login API
feat: update stuff
fix(android): bug fix
feat(login,network): add login and refactor API client
chore: add API, fix tests, and update README
```

문제점:

- `backend`, `ai-fix`를 type처럼 사용했습니다.
- 저장소가 이미 역할을 나타내는데 `backend`, `android`를 scope로 반복했습니다.
- `update stuff`, `bug fix`는 무엇이 바뀌었는지 알 수 없습니다.
- 서로 다른 목적의 변경이 한 커밋에 섞였습니다.
- 실제 기능 추가를 의미 없는 `chore`로 표현했습니다.

---

## 8. Pull Request 규칙

### PR 제목

PR 제목도 커밋과 같은 형식을 사용합니다.

```text
<type>(<scope>): <summary>
```

scope가 필요하지 않다면 생략할 수 있습니다.

```text
feat(auth): add social login API
fix(rag): prevent duplicate document retrieval
refactor(network): simplify token refresh flow
docs: document development environment
```

### PR 본문

각 저장소에 다음 템플릿을 `.github/pull_request_template.md`로 추가하는 것을 권장합니다.

```markdown
## 변경 내용
- 무엇을 변경했는지 작성

## 변경 이유
- 왜 필요한지 작성

## 영향 범위
- 영향받는 모듈, API 또는 다른 저장소를 작성

## 확인 방법
- 실행한 테스트 또는 수동 확인 절차 작성

## 관련 작업
- 관련 Issue와 다른 저장소의 PR 링크 작성

## 체크리스트
- [ ] 커밋 전 diff를 직접 확인했다
- [ ] 관련 없는 변경이 포함되지 않았다
- [ ] 테스트 또는 수동 검증을 완료했다
- [ ] 필요한 문서와 API 명세를 업데이트했다

Closes #이슈번호
```

- PR 하나에는 기능 하나 또는 버그 하나만 포함합니다.
- 리뷰어는 최소 1명으로 합니다.
- 작성자는 리뷰 반영 후 CI 결과와 최종 diff를 다시 확인합니다.
- 다른 저장소에 영향을 주면 해당 저장소와 변경 내용을 본문에 명시합니다.

---

## 9. 여러 저장소가 함께 변경되는 작업

하나의 기능 때문에 Backend, AI, Android 저장소가 함께 바뀌더라도 저장소마다 별도의 브랜치와 PR을 만듭니다.

```text
backend-repo: feat(chat): expose streaming response API
ai-repo:      feat(model): support streaming generation
android-repo: feat(chat): render streaming messages
```

협업 순서:

1. 대표 Issue 하나를 만들거나 GitHub Project에서 작업을 묶습니다.
2. 각 저장소에 필요한 Issue와 PR을 만들고 서로 링크합니다.
3. PR 본문에 의존 관계와 권장 병합 순서를 작성합니다.
4. API 명세 변경을 먼저 합의한 뒤 관련 작업을 진행합니다.
5. 호환되지 않는 변경은 동시에 배포하지 말고 이전 버전과의 호환 기간을 둡니다.

PR 본문에는 다음과 같이 관련 작업을 남깁니다.

```markdown
## 관련 저장소
- API 제공: organization/backend-repo#123
- AI 처리: organization/ai-repo#45
- Android 적용: organization/android-repo#67

## 병합 순서
1. backend-repo
2. ai-repo
3. android-repo
```

---

## 10. Issue 라벨 권장안

저장소 자체가 Backend, AI, Android 영역을 구분하므로 `area: backend` 같은 직군 라벨은 기본적으로 사용하지 않습니다.

### 종류

- `type: feature`
- `type: bug`
- `type: refactor`
- `type: docs`

### 우선순위

- `priority: high`
- `priority: normal`
- `priority: low`

### 필요한 경우에만 사용하는 상태

- `blocked`
- `help wanted`

저장소 안에 모듈이 많아졌을 때만 `area: auth`, `area: rag`, `area: network` 같은 모듈 라벨을 추가합니다. 담당자 표시는 라벨 대신 GitHub Assignee를 사용합니다.

---

## 11. Claude 사용 시 지켜야 할 규칙

Claude가 코드를 작성했더라도 최종 책임은 커밋 작성자에게 있습니다.

1. **커밋 전 diff 확인**  
   변경된 모든 파일과 diff를 직접 확인합니다. 삭제, 대규모 포맷 변경, 설정 및 의존성 변경을 특히 주의합니다.

2. **관련 없는 변경 금지**  
   요청 범위 밖의 리팩터링, 이름 변경, 포맷 수정, 주석 정리를 같은 커밋에 넣지 않습니다. 필요하면 별도 Issue와 커밋으로 분리합니다.

3. **자동 생성 메시지 검토**  
   Claude가 제안한 커밋 메시지와 PR 설명을 그대로 사용하지 않습니다. 실제 diff와 일치하는지 확인하고 과장되거나 누락된 내용을 고칩니다.

4. **작은 단위 커밋**  
   Claude에게 한 번에 큰 작업을 맡겼더라도 논리적 변경 단위로 나눠 커밋합니다. 각 커밋은 가능하면 독립적으로 이해하고 되돌릴 수 있어야 합니다.

5. **현재 저장소 범위 준수**  
   다른 저장소의 변경이 필요하면 자동으로 함께 수정하지 않습니다. 필요한 작업을 정리한 뒤 해당 저장소에서 별도 Issue와 PR로 처리합니다.

6. **검증 후 커밋**  
   관련 테스트, 빌드, 린트를 실행합니다. 실행하지 못했다면 PR 본문에 이유와 미확인 범위를 적습니다.

7. **민감정보 확인**  
   API 키, 토큰, 개인정보, 로컬 설정 파일이 diff에 포함되지 않았는지 확인합니다.

Claude 요청 예시:

```text
이 Issue와 현재 저장소 범위만 수정해 줘.
관련 없는 리팩터링이나 포맷 변경은 하지 마.
다른 저장소의 변경이 필요하면 직접 수정하지 말고 필요한 작업만 알려 줘.
수정 후 변경 파일, 테스트 결과, 남은 위험을 정리해 줘.
커밋은 실행하지 말고 내가 먼저 diff를 확인할 수 있게 해 줘.
```

---

## 12. 긴급 hotfix 예외

운영 장애나 보안 문제처럼 즉시 조치해야 하는 경우에는 문제가 발생한 저장소의 `main`에서 hotfix 브랜치를 만듭니다.

```text
hotfix/payment-timeout
hotfix/model-fallback
```

- 브랜치에는 `hotfix/`를 사용하지만 커밋 type은 변경 성격에 맞는 `fix`를 사용합니다.
- 커밋 또는 PR 제목 예: `fix(payment): prevent request timeout`
- 긴급 상황에서는 사전 리뷰를 생략할 수 있지만 가능한 범위에서 테스트와 diff 확인은 반드시 수행합니다.
- 리뷰를 생략했다면 병합 후 가능한 한 빨리 사후 리뷰를 받습니다.
- 원인과 재발 방지 조치를 Issue에 기록합니다.
- 긴급 수정에 리팩터링이나 기능 개선을 함께 넣지 않습니다.
- 여러 저장소에 긴급 수정이 필요하면 저장소별 hotfix PR을 만들고 적용 순서를 명시합니다.

---

## 13. 병합 방식

기본 병합 방식으로 **Squash and merge**를 권장합니다.

이유:

- 작업 중 생긴 `fix typo`, `review 반영` 같은 중간 커밋을 정리할 수 있습니다.
- 각 저장소의 `main`에서 커밋 하나가 PR 하나와 대응하므로 변경 추적과 되돌리기가 쉽습니다.
- PR 제목을 Conventional Commits 형식으로 관리하면 squash 커밋 메시지도 일관되게 유지할 수 있습니다.

병합 전 확인 사항:

- PR 제목이 `<type>(<scope>): <summary>` 또는 `<type>: <summary>` 형식인지 확인합니다.
- squash 커밋 메시지에서 임시 문구와 불필요한 자동 생성 문구를 제거합니다.
- 여러 독립 변경이 한 PR에 들어갔다면 squash로 숨기지 말고 PR을 나눕니다.

개별 커밋 보존이 꼭 필요한 대규모 마이그레이션 등은 팀 합의로 rebase merge를 사용할 수 있습니다. 일반 작업에서는 merge commit을 만들지 않습니다.

---

## 14. 빠른 체크리스트

### 커밋 전

- [ ] 한 가지 논리적 변경만 포함했는가?
- [ ] `type`은 변경 성격을 나타내는가?
- [ ] `scope`를 썼다면 현재 저장소의 모듈이나 기능을 나타내는가?
- [ ] 전체 diff를 직접 확인했는가?
- [ ] Claude가 만든 관련 없는 변경과 민감정보가 없는가?
- [ ] 필요한 테스트를 실행했는가?

### PR 병합 전

- [ ] PR 제목이 Conventional Commits 형식인가?
- [ ] 본문에 변경 이유, 영향 범위, 확인 방법이 있는가?
- [ ] 다른 저장소에 미치는 영향과 관련 PR을 표시했는가?
- [ ] CI를 통과하고 팀원 1명이 승인했는가?
- [ ] 최종 diff와 squash 커밋 메시지를 확인했는가?
