"""진료 후 요약 카드 — 환자가 읽는다 (챗봇②, 이슈 #9).

핵심은 "의사가 한 말 메모". 진료실에서 나온 직후 환자가 들은 말을 문답으로 받아
**환자 표현 그대로** 정리해 둔다. 요약이 아니라 기록이다.

지키는 선 (구조로):
- 병명은 `heard_diagnosis` 한 축에만 들어가고, 그 축의 정의가 "환자가 옮긴 말"이다.
  AI가 추론한 병명이 들어올 자리가 없다.
- 약 용법을 일반 지식으로 채우지 않는다. 채점 D5(발화에 없는 숫자 금지)가 잡는다.
- 설명·해석 자리는 없다. 설명문 인용(B3)은 라이선스 회신 후 별도 필드로.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from medimate.schema.card import AxisEntry, InterviewCard


class PostAxis(StrEnum):
    HEARD_DIAGNOSIS = "heard_diagnosis"  # 들은 병명·소견. 환자가 옮긴 말 그대로
    MEDICATION = "medication"  # 약 — 이름·횟수·기간, 환자가 말한 것만
    TESTS_PROCEDURES = "tests_procedures"  # 받았거나 하기로 한 검사·시술
    FOLLOW_UP = "follow_up"  # 다음 방문
    INSTRUCTIONS = "instructions"  # 주의사항·생활 지시
    OPEN_QUESTIONS = "open_questions"  # 못 물어본 것·헷갈리는 것. 다음 진료 때 쓸 메모


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


class PostVisitCard(InterviewCard):
    axes: dict[PostAxis, AxisEntry] = Field(
        default_factory=lambda: {a: AxisEntry() for a in PostAxis}
    )
    widening: list[WideningNote] = Field(default_factory=list)  # heard_diagnosis 용어의 부위 병기
    # 진료 전 카드의 부위와 대조한 결과. "same" / "different" / None(비교 불가). 판정이 아니라 표시
    site_comparison: str | None = None
    # 문서 코드(진단서·처방전 KCD) → 부위. 백로그 #22. 자리만 둔다
    document_codes: list[dict] = Field(default_factory=list)

    def is_minimally_complete(self) -> bool:
        # 들은 말이 하나라도 기록되면 카드다. 병명이 없어도 성립한다(약만 들었을 수도)
        return any(e.status.value == "filled" for e in self.axes.values())

    def unfilled_axes(self) -> list[PostAxis]:  # type: ignore[override]
        from medimate.schema.card import FieldStatus

        open_ = (FieldStatus.NOT_ASKED, FieldStatus.AMBIGUOUS)
        return [a for a in PostAxis if self.axes[a].status in open_]
