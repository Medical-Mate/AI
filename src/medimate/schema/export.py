"""내부 카드 → 백엔드 스키마 어댑터.

백엔드 스키마는 백엔드가 정한다. 확정되면 이 파일만 고친다.
지금은 확정 전이므로 내부 모델을 거의 그대로 내보내는 임시 형식이다.
"""

from __future__ import annotations

from typing import Any

from medimate.schema.card import PreVisitCard


def to_backend_payload(card: PreVisitCard) -> dict[str, Any]:
    """임시 형식. TODO: 백엔드 스키마 확정 후 필드명·구조 맞추기."""
    return {
        "chief_complaint": card.chief_complaint,
        "axes": {
            axis.value: {
                "status": entry.status.value,
                "value": entry.value,
                "evidence": entry.evidence,
            }
            for axis, entry in card.axes.items()
        },
        "red_flags": card.red_flags,
        "patient_notes": card.patient_notes,
        "patient_message": card.patient_message,  # 의사에게 전하는 말. 원문. 없으면 null
        "minimally_complete": card.is_minimally_complete(),
        "completeness": card.completeness(),
        "provenance": card.provenance.model_dump() if card.provenance else None,
        # 진료과 안내 — 부위 노드 속성. 증상 축과 무관. 추천 아님(안내 어투는 앱이 붙인다)
        # docs/decisions/2026-09-04-department-guidance.md
        "department_guidance": _department_guidance(card),
    }


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
