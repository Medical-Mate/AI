"""내부 카드 → 백엔드 스키마 어댑터.

백엔드 스키마는 백엔드가 정한다. 확정되면 이 파일만 고친다.
지금은 확정 전이므로 내부 모델을 거의 그대로 내보내는 임시 형식이다.
진료 전(PreVisitCard)·진료 후(PostVisitCard) 둘 다 여기서 낸다. `card_type`으로 구분한다.
"""

from __future__ import annotations

from typing import Any

from medimate.schema.card import InterviewCard, PreVisitCard
from medimate.schema.postvisit import PostVisitCard


def to_backend_payload(card: InterviewCard) -> dict[str, Any]:
    """임시 형식. TODO: 백엔드 스키마 확정 후 필드명·구조 맞추기."""
    out: dict[str, Any] = {
        "card_type": _card_type(card),
        "chief_complaint": card.chief_complaint,
        "axes": {
            str(getattr(axis, "value", axis)): {
                "status": entry.status.value,
                "value": entry.value,
                "evidence": entry.evidence,
            }
            for axis, entry in card.axes.items()
        },
        "red_flags": card.red_flags,
        "patient_notes": card.patient_notes,
        "patient_message": card.patient_message,  # 마지막 질문에 남긴 말. 원문. 없으면 null
        "minimally_complete": card.is_minimally_complete(),
        "completeness": card.completeness(),
        "provenance": card.provenance.model_dump() if card.provenance else None,
    }
    if isinstance(card, PreVisitCard):
        # 진료과 안내 — 부위 노드 속성. 증상 축과 무관. 추천 아님(안내 어투는 앱이 붙인다)
        # docs/decisions/2026-09-04-department-guidance.md
        out["department_guidance"] = _department_guidance(card)
    if isinstance(card, PostVisitCard):
        # 넓히기 병기 — 들은 용어에 부위를 붙인 것. 설명이 아니라 위치 표시 (§3)
        out["widening"] = [w.model_dump() for w in card.widening]
        out["site_comparison"] = card.site_comparison  # same / different / null. 판정 아님
        out["document_codes"] = card.document_codes  # 백로그 #22. 지금은 항상 []
        # 메모 원문·분류되지 않은 문장·재방문 날짜(결정론)·진료 메타. 앱 1q 화면이 그대로 그린다
        out["memo"] = card.memo
        out["unsorted"] = card.unsorted
        out["follow_up_date"] = card.follow_up_date.model_dump() if card.follow_up_date else None
        out["visit_date"] = card.visit_date
        out["clinic"] = card.clinic
    return out


def _card_type(card: InterviewCard) -> str:
    if isinstance(card, PostVisitCard):
        return "postvisit"
    if isinstance(card, PreVisitCard):
        return "previsit"
    return "unknown"


def _department_guidance(card: PreVisitCard) -> dict[str, Any] | None:
    sel = card.site_selection
    if sel is None or not sel.departments:
        return None
    return {
        "site_label": sel.label,
        "departments": list(sel.departments),  # 순서에 의미 없음. 전부 보여준다
        "source": sel.departments_source,  # 인용이 아님을 함께 내보낸다
        "note": "접수 시 확인해 주세요",
    }
