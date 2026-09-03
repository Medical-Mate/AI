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
        "minimally_complete": card.is_minimally_complete(),
        "completeness": card.completeness(),
        "provenance": card.provenance.model_dump() if card.provenance else None,
    }
