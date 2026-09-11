"""내부 카드 → 백엔드 스키마 어댑터.

백엔드 스키마는 백엔드가 정한다. 확정되면 이 파일만 고친다.
지금은 확정 전이므로 내부 모델을 거의 그대로 내보내는 임시 형식이다.
진료 전(PreVisitCard)·진료 후(PostVisitCard) 둘 다 여기서 낸다. `card_type`으로 구분한다.
"""

from __future__ import annotations

import re
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
        "minimally_complete": card.is_minimally_complete(),
        "completeness": card.completeness(),
        "provenance": card.provenance.model_dump() if card.provenance else None,
    }
    if isinstance(card, PreVisitCard):
        # 진료과 안내 — 부위 노드 속성. 증상 축과 무관. 추천 아님(안내 어투는 앱이 붙인다)
        # docs/decisions/2026-09-04-department-guidance.md
        out["department_guidance"] = _department_guidance(card)
        out["title"] = _title(card)
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


# 기간 표현. 숫자(아라비아·한글) + 단위, 또는 상대 표현. **원문을 자르지 않고 이만 뽑는다.**
# 긴 단위를 먼저 둬야 "일주일"이 일+주일로 잡히고 "3일"이 3+일로 잡힌다.
#
# ── 백로그: 알려진 한계 (2026-09-11 실측) ──────────────────────────────────
# 지금은 제목 한 줄이 유일한 소비자이고 카드 본문에 `onset` 원문이 그대로 있어서, 못 뽑아도
# 정보가 깨지지 않는다. 그래서 미뤘다. 착수 기준은 **기간 파싱의 소비자가 하나 더 생길 때**다
# (재방문 날짜 계산, 시간축 변화 요약 등). 지금 손대면 제목 하나를 위해 의존성을 늘리는 셈이다.
#
# 못 잡는 것
#   이형태      "석 달"·"넉 달"  → None. 세/네가 달 앞에서 교체되는데 목록에 없다.
#                                  실제로 "세 달"보다 "석 달"이 더 흔하다
#   복합수      "열두 시간" → "두 시간", "이십 일" → "십 일", "열하루" → "하루" (앞자리 손실)
#   어림수      "사나흘" → "나흘", "대엿새" → "엿새" (substring이라 뒷부분만 먹는다)
#               **이게 제일 나쁘다 — 없는 답보다 틀린 답이다.** 3~4일을 4일로 단정한다
#   문맥        "이틀에 한 번씩 아파요" → "이틀" (빈도인데 기간으로), "하루 종일" → "하루"
#
# 형태소 분석기(Kiwi 등)를 쓰면 **위 셋은 풀린다.** 수관형사·수사는 닫힌 부류라 사전이
# 이형태(석/넉/서/너·스무)와 복합수(열두)를 이미 갖고 있고, 품사 태그로 수(MM/NR) + 단위
# 의존명사(NNB) 구성을 바로 잡는다. "수치라 변화가 무궁무진하다"는 걱정은 이 층에는 해당하지
# 않는다 — 무한한 건 값이지 형태가 아니다.
#
# **풀리지 않는 건 네 번째, 의미 판별이다.** "이틀에 한 번"과 "이틀 됐어요"는 형태가 같고
# 품사도 같다. 기간이냐 빈도냐는 뒤따르는 조사·용언을 봐야 갈리므로 규칙이 따로 필요하다.
# 즉 형태소 분석기는 재료를 정확히 주지만 판단은 여전히 우리 몫이다. 갈아탈 때 이 점을 기억할 것.
# ───────────────────────────────────────────────────────────────────────
_DURATION = re.compile(
    r"(\d+|[한두세네]|[일이삼사오육칠팔구십]|다섯|여섯|일곱|여덟|아홉|열)\s*(주일|개월|시간|일|주|달|년)"
)
_RELATIVE = ("엊그제", "그저께", "그제", "어제", "오늘", "방금")
# 순우리말 날짜. 숫자+단위가 아니라 한 낱말이라 위 정규식이 못 잡는다.
# 실제 카드에서 "이틀 전부터"·"닷새 전에"·"열흘쯤 전에"가 나왔다. 긴 것부터 본다
_PLAIN = (
    "여드레",
    "아흐레",
    "엿새",
    "이레",
    "닷새",
    "나흘",
    "사흘",
    "이틀",
    "하루",
    "열흘",
    "보름",
)
TITLE_MAX = 20


def _title(card: PreVisitCard) -> str | None:
    """카드 제목 — `{좌우} {부위 라벨} · {기간}`. **LLM을 부르지 않는다.**

    2026-09-11 백엔드 합의(#7). 처음 제안은 `chief_complaint`에서 뽑는 것이었는데 철회했다 —
    그건 환자 첫 발화라 길고 구어체여서 20자에 맞추려면 잘라야 하고, **자르는 순간 환자 말이
    아니게 된다.** 카드 값의 조합이면 병명이 들어갈 경로가 없어 백엔드 검증도 필요 없어진다.

    **증상어를 붙이지 않는다.** 와이어프레임 제목이 `복부 통증 · 3주`라 "통증"이 어디서 오는지
    백엔드가 물었는데, 부위 라벨에도 축에도 없다. 붙이면 우리가 만들지 않은 사실을 넣는 것이고,
    느낌 축이 "먹먹해요"인 카드에 "통증"이라고 쓰면 틀린다. 부위 라벨만 쓴다.

    기간은 **숫자+단위만** 뽑는다("2주 전부터요, 등산 다음 날부터" → "2주"). 못 뽑으면 부위만,
    부위도 없으면 `None` — 백엔드가 제목 줄을 숨긴다(`department_guidance`와 같은 처리).
    """
    site = _site_label(card)
    if not site:
        return None
    dur = _duration(card)
    title = f"{site} · {dur}" if dur else site
    # 20자를 넘으면 기간을 뺀다. **자르지 않는다** — 자르면 부위 이름이 깨진다
    return title if len(title) <= TITLE_MAX else site[:TITLE_MAX]


def _site_label(card: PreVisitCard) -> str | None:
    """인체도에서 짚은 라벨이 우선(좌우가 이미 붙어 있다). 없으면 site 축 값."""
    if card.site_selection and card.site_selection.label:
        return card.site_selection.label.strip()
    from medimate.schema.card import Axis, FieldStatus

    entry = card.axes.get(Axis.SITE)
    if entry and entry.status == FieldStatus.FILLED and entry.value:
        return entry.value.strip()
    return None


def _duration(card: PreVisitCard) -> str | None:
    """**시작(onset) 축에서만** 기간을 뽑는다.

    경과(time_course)는 쓰지 않는다(2026-09-11, 백엔드 합의 #7). "심해졌어요"는 기간이 아니고,
    숫자가 들어 있어도 뜻이 다르다. 카드 100장에서 경과가 기여하는 건 PC31 한 장인데 그게
    정확히 틀린 예다 — 경과가 "한 번 생기면 **열흘**쯤 있다 아물어요"라 `열흘`이 뽑히지만,
    그건 **삽화 하나의 지속 기간**이지 발병 후 경과가 아니다. 제목에 붙이면
    "열흘 전에 시작됐다"로 읽힌다.
    """
    from medimate.schema.card import Axis, FieldStatus

    for axis in (Axis.ONSET,):
        entry = card.axes.get(axis)
        if not entry or entry.status != FieldStatus.FILLED or not entry.value:
            continue
        v = entry.value
        m = _DURATION.search(v)
        if m:
            return f"{m.group(1)}{m.group(2)}" if m.group(1).isdigit() else m.group(0).strip()
        for word in _PLAIN + _RELATIVE:
            if word in v:
                return word
    return None
