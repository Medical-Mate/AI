"""추출 결과 런타임 가드 — 모델이 무엇을 내든 카드에는 규칙에 맞는 것만 들어간다.

프롬프트는 부탁이고 가드는 강제다 (CLAUDE.md "구조로 보장, 프롬프트로 부탁하지 않는다").
채점기의 D4·D5·D7을 런타임 코드로 옮긴 것이다. 소형(온디바이스) 모델은 이 셋에서 흔들리는데,
크기·프롬프트로는 0이 되지 않았다(2026-09-07 스크리닝 1·2차). 그래서 엔진이 지킨다.

1. 축 필터: 물은 축이 있으면 그 축의 갱신만 받는다. 다른 축은 버리되 근거는 patient_notes에 남긴다
   (환자 말을 버리지 않는다). 첫 발화(물은 축 없음)에는 적용하지 않는다
2. 근거 필터: evidence가 발화의 부분 문자열(공백 무시)이 아니면 그 갱신을 버린다. 모델이 글자를
   바꿔 쓴 것("찌릿"→"치릿")은 근거가 아니다
3. 숫자 필터: value의 숫자가 발화에 없으면 버린다. "3일 전이라고 적어"에 3일을 채우는 것을 막는다
4. 되묻기 필터: 통증 강도를 물었는데 환자가 **숫자 없이 되물으면** 그 축 갱신을 버린다
5. 글자 없음 필터: 발화에 글자·숫자가 하나도 없으면(이모지·자음·구두점뿐) 갱신을 전부 버린다

버린 것은 전부 `dropped`에 남아 감사 로그(audit)로 나간다.

4·5는 2026-09-14 추가. Nova로 추출을 돌린 88케이스에서 실패 12건이 **전부 환자가 말하지 않은
값을 축에 넣은 것**이었고, 근거가 원문 그대로라 2·3에 안 걸렸다. 모델이 지어낸 것은 값이 아니라
"그 발화가 그 축을 채운다"는 판단이다. 프롬프트로도 밀지만 여기서 구조로 막는다 —
폰 경로도 같은 가드를 타므로 온디바이스에도 같이 듣는다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from medimate.llm.base import AxisUpdate, TurnExtraction
from medimate.schema.card import FieldStatus


@dataclass(frozen=True)
class GuardConfig:
    only_asked_axis: bool = (
        False  # 소형 모델 프로필에서 True. 서버 Terra는 곁들인 축을 잘 뽑으므로 False
    )
    evidence_substring: bool = True  # 설계 원칙(근거는 발화 원문). 모든 프로필에서 켠다
    numbers_from_utterance: bool = True  # 같음
    # 통증 강도를 물었는데 숫자 없이 되묻는 발화("그게 중요해요?")에서 값을 만들지 않는다.
    # **물음표만으로 막지 않는다** — `"아프긴 한데 한 5점?"`은 물음표로 끝나지만 진짜 답이다.
    # 숫자가 있으면 통과시킨다
    no_severity_from_question: bool = True
    # 글자·숫자가 하나도 없는 발화("ㅠㅠㅠㅠ 😭😭")에서는 어떤 축도 채우지 않는다
    no_value_from_letterless: bool = True
    # 짧은 긍정만 온 답("네", "응", "ㅇㅇ", "그렇다")은 값이 아니라 되물을 신호다. 물은 축의
    # filled를 ambiguous로 바꿔 엔진이 되묻게 한다(CLARIFY). 2026-09-22: "퍼지나요?"의 ㅇㅇ를
    # Nova가 프롬프트 둘·표기 둘(응·네)에서 모두 `안 퍼짐`으로 냈다. 부정을 카드에 넣는 길을
    # 프롬프트가 아니라 여기서 닫는다
    bare_affirmative_asks_back: bool = True

    @classmethod
    def ondevice(cls) -> GuardConfig:
        return cls(only_asked_axis=True)


@dataclass
class GuardResult:
    extraction: TurnExtraction
    dropped: list[dict] = field(default_factory=list)  # {"axis","reason","value","evidence"}


# 한글 음절·영문자·숫자가 하나라도 있는가. 자음만("ㅠㅠ")·이모지·구두점은 글자로 세지 않는다
_HAS_LETTER = re.compile(r"[가-힣A-Za-z0-9]")
# 짧은 긍정만 있는 답. 표기 복원 뒤의 문장 전체가 이것뿐일 때(다른 내용어가 없을 때)만
_BARE_AFFIRM = re.compile(
    r"^\s*(?:네|예|응|어|그래|그렇다|그렇습니다|그런 것 같다|맞다|맞아요|맞습니다"
    r"|그런데요|그래요|네요|ㅇㅇ|ㅇㅋ|오케이)[\s.!~ㅋㅎㅠㅜ]*$"
)


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def guard_extraction(
    ext: TurnExtraction,
    utterance: str,
    asked_axis: StrEnum | None,
    cfg: GuardConfig | None = None,
    normalized: str | None = None,
) -> GuardResult:
    """`normalized`: 채팅 표기 복원문(text/chatnorm). 주면 근거·글자·숫자 판정에 원문과 함께 쓴다.

    `ㄴㄴ`는 자음뿐이라 5번 필터가 갱신을 전부 버렸다(#117). 복원문 `아니`에는 글자가 있어 통과한다.
    evidence는 원문 또는 복원문의 부분 문자열이면 받는다 — 복원은 표로만 하므로 출처는 남는다.
    """
    cfg = cfg or GuardConfig()
    kept: list[AxisUpdate] = []
    dropped: list[dict] = []
    nu = _norm(utterance)
    nn = _norm(normalized) if normalized and normalized != utterance else ""
    utt_nums = set(re.findall(r"\d+", utterance)) | set(re.findall(r"\d+", normalized or ""))
    has_letter = _HAS_LETTER.search(utterance) or (nn and _HAS_LETTER.search(normalized or ""))
    letterless = cfg.no_value_from_letterless and not has_letter

    def in_utt(s: str) -> bool:
        ns = _norm(s)
        return bool(ns) and (ns in nu or (bool(nn) and ns in nn))

    bare_affirm = (
        cfg.bare_affirmative_asks_back
        and asked_axis is not None
        and bool(_BARE_AFFIRM.match(normalized or utterance))
    )

    # 되묻기: 강도를 물었는데 물음표로 끝나고 숫자가 하나도 없다
    asking_back = (
        cfg.no_severity_from_question
        and asked_axis is not None
        and str(getattr(asked_axis, "value", asked_axis)) == "severity"
        and utterance.rstrip().endswith("?")
        and not utt_nums
    )
    # notes는 "축에 안 들어간 환자 말"이다. 발화에 없는 문장(예시 베끼기, 이력 안내문 복사)은 버린다
    notes: list[str] = []
    for n in ext.notes:
        if cfg.evidence_substring and not in_utt(n):
            dropped.append({"axis": "notes", "reason": "note_not_in_utterance", "value": n})
        else:
            notes.append(n)

    for u in ext.updates:
        reason = None
        if letterless:
            reason = "letterless_utterance"
        elif asking_back and str(getattr(u.axis, "value", u.axis)) == "severity":
            reason = "severity_from_question"
        elif cfg.only_asked_axis and asked_axis is not None and u.axis != asked_axis:
            reason = "not_asked_axis"
        elif cfg.evidence_substring and not in_utt(u.evidence):
            reason = "evidence_not_in_utterance"
        elif cfg.numbers_from_utterance and u.value:
            bad = [n for n in re.findall(r"\d+", u.value) if n not in utt_nums]
            if bad:
                reason = f"number_not_in_utterance:{','.join(bad)}"
        if reason is None:
            if bare_affirm and u.axis == asked_axis and u.status == FieldStatus.FILLED:
                # 값은 버리고 축은 살린다 — 엔진이 되묻는다. 무엇을 버렸는지는 dropped에 남긴다
                dropped.append(
                    {
                        "axis": u.axis.value,
                        "reason": "bare_affirmative_asks_back",
                        "value": u.value,
                        "evidence": u.evidence,
                    }
                )
                u = AxisUpdate(
                    axis=u.axis, status=FieldStatus.AMBIGUOUS, value=None, evidence=u.evidence
                )
            kept.append(u)
            continue
        dropped.append(
            {"axis": u.axis.value, "reason": reason, "value": u.value, "evidence": u.evidence}
        )
        # 물은 축이 아니어서 버린 것은 환자 말이므로 notes에 남긴다(근거가 진짜일 때만)
        if reason == "not_asked_axis" and in_utt(u.evidence):
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
