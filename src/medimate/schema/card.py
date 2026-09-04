"""문진 카드 — AI 내부 상태 모델.

백엔드에 넘기는 형식이 아니다. 백엔드 스키마는 백엔드가 정하고,
`medimate.schema.export`가 이 모델을 그 형식으로 변환한다.

원칙 (docs/ai-design.md §1, §5):
- 값은 환자가 말한 것을 정리한 것이다. 진단·감별 어휘를 넣지 않는다.
- 모든 값에는 근거(환자 발화 원문)가 붙는다. 근거 없는 값은 만들지 않는다.
- 비어 있는 축은 "왜 비었는지"를 구분한다. 의사가 볼 때 값이 있는 정보다.

카드는 두 종류다. 진료 전(PreVisitCard, SOCRATES 8축)과 진료 후(schema/postvisit.py, 6축).
공통 뼈대는 InterviewCard에 있고, 축 목록만 다르다.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Axis(StrEnum):
    """진료 전 — SOCRATES 8축 + 부위는 환자 표현 그대로 둔다(온톨로지 매핑은 부위 확정 후)."""

    SITE = "site"  # 어디가
    ONSET = "onset"  # 언제부터, 어떻게 시작
    CHARACTER = "character"  # 어떤 느낌 (욱신·찌르는·조이는)
    RADIATION = "radiation"  # 퍼지는 곳
    ASSOCIATED = "associated"  # 함께 나타나는 증상
    TIME_COURSE = "time_course"  # 시간에 따른 변화
    EXACERBATING_RELIEVING = "exacerbating_relieving"  # 악화·완화 요인
    SEVERITY = "severity"  # 심각도


class FieldStatus(StrEnum):
    NOT_ASKED = "not_asked"  # 아직 묻지 않았다
    FILLED = "filled"  # 환자가 답했고 값이 있다
    UNKNOWN = "unknown"  # 환자가 "모르겠다"고 했다
    SKIPPED = "skipped"  # 환자가 답하지 않고 넘겼다
    AMBIGUOUS = "ambiguous"  # 발화가 이 축을 언급했지만 값을 확정할 수 없다 → 엔진이 확인 질문


class AxisEntry(BaseModel):
    status: FieldStatus = FieldStatus.NOT_ASKED
    value: str | None = None  # 환자 표현을 정리한 한 줄
    evidence: list[str] = Field(default_factory=list)  # 환자 발화 원문 인용


class Provenance(BaseModel):
    """재현성 — 응답에 반드시 박는 것 (docs/ai-design.md §7)."""

    prompt_version: str
    model_id: str
    ontology_snapshot: str | None = None  # 온톨로지 매핑을 붙이기 전까지 None


class SiteSelectionRecord(BaseModel):
    """인체도에서 짚은 부위의 기록 (dialog/site.py가 만든다).

    진료과 안내는 노드에 적힌 값 그대로. 고르지 않는다.
    """

    node_id: str
    anchor_id: str
    side: str | None = None
    label: str
    departments: list[str] = Field(default_factory=list)
    departments_source: str = ""
    ontology_snapshot: str


_CLOSED = (FieldStatus.FILLED, FieldStatus.UNKNOWN, FieldStatus.SKIPPED)
_OPEN = (FieldStatus.NOT_ASKED, FieldStatus.AMBIGUOUS)


class InterviewCard(BaseModel):
    """진료 전·후 카드의 공통 뼈대. 축 키는 문자열(각 카드의 StrEnum 값).

    extra="allow": 상태(SessionState)로 왕복할 때 하위 카드의 필드(site_selection, widening)가
    사라지지 않게 한다. 엔진이 명세의 카드 타입으로 다시 읽는다.
    """

    model_config = ConfigDict(extra="allow")

    chief_complaint: str | None = None  # 주 호소 한 문장. 환자 표현 그대로 (진료 후 카드는 비움)
    axes: dict[str, AxisEntry] = Field(default_factory=dict)
    red_flags: list[str] = Field(default_factory=list)  # 자문 전. 자리만 둔다
    patient_notes: list[str] = Field(default_factory=list)  # 축에 안 들어가는 환자 말(자동)
    patient_message: str | None = None  # 마지막 질문에 환자가 의사에게 남긴 말. 원문 그대로
    provenance: Provenance | None = None

    # --- 종료 조건 -------------------------------------------------------
    # 강제하지 않는다. 환자가 나가면 그 시점 카드를 그대로 낸다.

    def is_minimally_complete(self) -> bool:
        return any(e.status == FieldStatus.FILLED for e in self.axes.values())

    def unfilled_axes(self) -> list[str]:
        """아직 묻지 않았거나 확인이 필요한 축. 되묻기 후보."""
        return [a for a, e in self.axes.items() if e.status in _OPEN]

    def completeness(self) -> float:
        if not self.axes:
            return 0.0
        answered = sum(1 for e in self.axes.values() if e.status in _CLOSED)
        return answered / len(self.axes)


class PreVisitCard(InterviewCard):
    """진료 전 브리핑 카드 — 의사가 읽는다."""

    axes: dict[Axis, AxisEntry] = Field(default_factory=lambda: {a: AxisEntry() for a in Axis})
    site_selection: SiteSelectionRecord | None = None  # 인체도 선택. 없으면 문답으로 SITE를 묻는다

    # "카드로 낼 수 있는 최소"만 정의한다: 부위 + 주 호소.
    def is_minimally_complete(self) -> bool:
        site = self.axes[Axis.SITE]
        return site.status == FieldStatus.FILLED and bool(self.chief_complaint)

    def unfilled_axes(self) -> list[Axis]:  # type: ignore[override]
        return [a for a in Axis if self.axes[a].status in _OPEN]
