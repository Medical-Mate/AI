"""진료 후 기록 카드 — 환자가 읽는다 (챗봇②, 이슈 #9).

와이어프레임 1p·1q(2026-09-07 확인): 문답이 아니라 **자유 메모 하나**를 받아
AI가 **4묶음으로 나눈다**.
"진료실에서 들은 말을 그대로 남겨두세요" → "AI가 메모를 4가지로 나눴어요".
그래서 축은 4개이고, 값은 환자가 쓴 **문장 원문 그대로**다. 요약·정리를 하지 않는다.

지키는 선 (구조로):
- 병명은 `findings`(들은 소견)에만, 그것도 환자가 적은 문장 그대로. AI가 덧붙이지 않는다
- 약 용법·검사 결과를 일반 지식으로 채우지 않는다. 문장 분류만 한다
- 재방문 날짜는 LLM이 아니라 진료일 기준 결정론 계산(dialog/followup.py)
- 어느 묶음에도 안 들어가는 문장은 `unsorted`에 남긴다. 버리지 않는다
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from medimate.schema.card import AxisEntry, InterviewCard


class PostAxis(StrEnum):
    """1q의 4묶음. 이름은 화면 라벨을 따른다."""

    FINDINGS = "findings"  # 소견 — 의사가 말한 병명·상태. 환자가 옮긴 문장 그대로
    TESTS = "tests"  # 검사 — 받았거나 하기로 한 검사, 결과 안내 시점
    MEDICATION_INSTRUCTIONS = "medication_instructions"  # 약 — 처방, 복용법, 주사·처치
    # 지침 — 식이·활동·자세·금지 같은 생활 지시(2026-09-15 분리). 전에는 약 칸에 접혀 있어서
    # 약 칸이 길어지고 "약"이라는 라벨 아래 약이 아닌 말이 찍혔다(백엔드 #78도 같은 지적).
    # `MEDIMATE_LIFESTYLE_AXIS`가 꺼져 있으면 예전처럼 약 칸에 접어 낸다. 앱은 모르는 축도
    # 뒤에 붙여 그리지만(라벨은 키 그대로), 켜는 시점은 운영이 정한다 — 켜면 끝까지 유지
    LIFESTYLE_INSTRUCTIONS = "lifestyle_instructions"
    FOLLOW_UP = "follow_up"  # 재방문 — 다음 방문 시점·조건


class WideningNote(BaseModel):
    """들은 용어에 부위를 **병기**한 것 (docs/ai-design.md §3 — 대체가 아니라 병기).

    설명이 아니라 위치 표시다. 온톨로지 구조 층과 사전 매칭으로 만들며 LLM은 관여하지 않는다.
    """

    term: str  # 환자 말에서 매칭된 표현 그대로
    node_id: str  # 구조 노드
    node_label: str  # 노드의 한국어 이름 (구조 유형 병기)
    anchor_id: str | None  # 올라가서 닿은 앵커
    anchor_label: str | None  # "무릎", "허리·엉덩이"
    ontology_snapshot: str


class FollowUpDate(BaseModel):
    """재방문 문장에서 결정론으로 계산한 날짜. LLM 무관. 앱 달력(1r)으로 넘어간다."""

    text: str  # "2주 뒤"
    date: str  # ISO. 진료일 + 상대 기간
    approximate: bool = True  # "전후"로 표시
    basis: str  # 계산 근거: "visit_date 2026-09-12 + 14d"


class PostVisitCard(InterviewCard):
    """진료 후 기록. axes 값은 해당 묶음의 문장들을 ' · '로 이은 원문, evidence는 문장 목록."""

    axes: dict[PostAxis, AxisEntry] = Field(
        default_factory=lambda: {a: AxisEntry() for a in PostAxis}
    )
    visit_date: str | None = None  # ISO. 재방문 날짜 계산의 기준. 앱이 준다
    clinic: str | None = None  # "서울OO병원 내과". 앱이 준다(1s·1r), AI가 만들지 않는다
    memo: str | None = None  # 환자가 적은 메모 원문 전체. 항상 보존
    unsorted: list[str] = Field(default_factory=list)  # 어느 묶음에도 못 넣은 문장. 버리지 않는다
    follow_up_date: FollowUpDate | None = None
    widening: list[WideningNote] = Field(default_factory=list)  # findings 용어의 부위 병기
    # 진료 전 카드의 부위와 대조한 결과. "same" / "different" / None(비교 불가). 판정이 아니라 표시
    site_comparison: str | None = None
    # 문서 코드(진단서·처방전 KCD) → 부위. 백로그 #22. 자리만 둔다
    document_codes: list[dict] = Field(default_factory=list)

    def is_minimally_complete(self) -> bool:
        # 한 묶음이라도 채워지면 기록이다. 병명이 없어도 성립한다(약만 들었을 수도)
        return any(e.status.value == "filled" for e in self.axes.values())

    def unfilled_axes(self) -> list[PostAxis]:  # type: ignore[override]
        from medimate.schema.card import FieldStatus

        open_ = (FieldStatus.NOT_ASKED, FieldStatus.AMBIGUOUS)
        return [a for a in PostAxis if self.axes[a].status in open_]

    def completeness(self) -> float:
        """지침 축이 접혀 있으면(플래그 꺼짐) 분모에서 뺀다 — 묻지도 채우지도 않는 축이다."""
        from medimate.dialog.memo import lifestyle_axis_enabled  # 순환 import 회피
        from medimate.schema.card import _CLOSED

        axes = {
            a: e
            for a, e in self.axes.items()
            if a != PostAxis.LIFESTYLE_INSTRUCTIONS or lifestyle_axis_enabled()
        }
        if not axes:
            return 0.0
        return sum(1 for e in axes.values() if e.status in _CLOSED) / len(axes)
