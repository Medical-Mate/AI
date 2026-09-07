"""추출 결과 런타임 가드 — 모델이 무엇을 내든 카드에는 규칙에 맞는 것만 들어간다.

프롬프트는 부탁이고 가드는 강제다 (CLAUDE.md "구조로 보장, 프롬프트로 부탁하지 않는다").
채점기의 D4·D5·D7을 런타임 코드로 옮긴 것이다. 소형(온디바이스) 모델은 이 셋에서 흔들리는데,
크기·프롬프트로는 0이 되지 않았다(2026-09-07 스크리닝 1·2차). 그래서 엔진이 지킨다.

1. 축 필터: 물은 축이 있으면 그 축의 갱신만 받는다. 다른 축은 버리되 근거는 patient_notes에 남긴다
   (환자 말을 버리지 않는다). 첫 발화(물은 축 없음)에는 적용하지 않는다
2. 근거 필터: evidence가 발화의 부분 문자열(공백 무시)이 아니면 그 갱신을 버린다. 모델이 글자를
   바꿔 쓴 것("찌릿"→"치릿")은 근거가 아니다
3. 숫자 필터: value의 숫자가 발화에 없으면 버린다. "3일 전이라고 적어"에 3일을 채우는 것을 막는다

버린 것은 전부 `dropped`에 남아 감사 로그(audit)로 나간다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from medimate.llm.base import AxisUpdate, TurnExtraction


@dataclass(frozen=True)
class GuardConfig:
    only_asked_axis: bool = (
        False  # 소형 모델 프로필에서 True. 서버 Terra는 곁들인 축을 잘 뽑으므로 False
    )
    evidence_substring: bool = True  # 설계 원칙(근거는 발화 원문). 모든 프로필에서 켠다
    numbers_from_utterance: bool = True  # 같음

    @classmethod
    def ondevice(cls) -> GuardConfig:
        return cls(only_asked_axis=True)


@dataclass
class GuardResult:
    extraction: TurnExtraction
    dropped: list[dict] = field(default_factory=list)  # {"axis","reason","value","evidence"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def guard_extraction(
    ext: TurnExtraction,
    utterance: str,
    asked_axis: StrEnum | None,
    cfg: GuardConfig | None = None,
) -> GuardResult:
    cfg = cfg or GuardConfig()
    kept: list[AxisUpdate] = []
    dropped: list[dict] = []
    nu = _norm(utterance)
    utt_nums = set(re.findall(r"\d+", utterance))
    # notes는 "축에 안 들어간 환자 말"이다. 발화에 없는 문장(예시 베끼기, 이력 안내문 복사)은 버린다
    notes: list[str] = []
    for n in ext.notes:
        if cfg.evidence_substring and _norm(n) not in nu:
            dropped.append({"axis": "notes", "reason": "note_not_in_utterance", "value": n})
        else:
            notes.append(n)

    for u in ext.updates:
        reason = None
        if cfg.only_asked_axis and asked_axis is not None and u.axis != asked_axis:
            reason = "not_asked_axis"
        elif cfg.evidence_substring and (not u.evidence.strip() or _norm(u.evidence) not in nu):
            reason = "evidence_not_in_utterance"
        elif cfg.numbers_from_utterance and u.value:
            bad = [n for n in re.findall(r"\d+", u.value) if n not in utt_nums]
            if bad:
                reason = f"number_not_in_utterance:{','.join(bad)}"
        if reason is None:
            kept.append(u)
            continue
        dropped.append(
            {"axis": u.axis.value, "reason": reason, "value": u.value, "evidence": u.evidence}
        )
        # 물은 축이 아니어서 버린 것은 환자 말이므로 notes에 남긴다(근거가 진짜일 때만)
        if reason == "not_asked_axis" and u.evidence.strip() and _norm(u.evidence) in nu:
            if u.evidence not in notes:
                notes.append(u.evidence)

    # chief_complaint에도 숫자 규칙을 적용한다 (첫 발화 요약에 숫자를 만들어 넣는 것 방지)
    cc = ext.chief_complaint
    if cfg.numbers_from_utterance and cc:
        if any(n not in utt_nums for n in re.findall(r"\d+", cc)):
            dropped.append(
                {"axis": "chief_complaint", "reason": "number_not_in_utterance", "value": cc}
            )
            cc = None

    guarded = TurnExtraction(
        chief_complaint=cc, updates=kept, notes=notes, wants_to_stop=ext.wants_to_stop
    )
    return GuardResult(extraction=guarded, dropped=dropped)
