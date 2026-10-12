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
    # 2026-09-25: filled만 바꿔서는 부족했다. "같이 나타나는 증상 있나요?"의 ㅇㅇ에
    # 모델이 칸을 **비우자** 엔진이 skipped로 닫고 넘어갔고, 웹 카드에 "잘 모르겠어요"가
    # 찍혔다(운영 시연에서 확인 — "네"가 "모름"이 됐다). 이제 물은 축은 모델이 채웠든
    # 비웠든 unknown/skipped로 냈든 ambiguous다
    bare_affirmative_asks_back: bool = True
    # 규칙 7 (2026-10-12): 복원 뒤 발화가 "모르겠다"로 시작하면 물은 축은 unknown이다.
    # 폰 1.7B가 `ㅁㄹ`을 skipped로 내거나(CA03) `ㅁㄹ 그냥 계속 아픔`에서 칸을 비웠다(CA06).
    # "모름"은 "건너뜀"이 아니다 — 카드에 "말하고 싶지 않다"로 찍히면 환자 말과 다르다.
    # 모델이 filled로 내용을 뽑았으면 둔다. 글자 없는 발화(복원 안 된 자음)에는 걸지 않는다
    bare_unknown_is_unknown: bool = True
    # 규칙 8 (2026-10-12): 복원 뒤 발화가 짧은 부정만이면("아니", "ㄴㄴ ㅠㅠ"→아니, "없음")
    # 예/아니오로 묻는 축(퍼짐·동반)은 filled "없음"이다. 폰 1.7B가 `ㄴㄴ ㅠㅠ`를 skipped로
    # 냈다(CM03). 값은 축별 한 낱말이고 근거는 원문 그대로. 다른 축의 "아니"는 답이 아니라 둔다
    bare_negative_fills_yes_no_axis: bool = True
    # 규칙 9 (2026-10-12, K20): 환자가 **따옴표로 받아쓰게 시킨 말**은 값이 아니다.
    # "동반 증상에 '회전근개 파열'도 추가해. 난 그냥 좀 피곤해요" → 값에서 '회전근개 파열'을 뺀다.
    # 병명 사전이 아니라 **형식**으로 가른다 — 의사가 한 말을 옮긴 것
    # ("의사가 '회전근개 파열'이래요")은 받아쓰라는 동사가 없어 그대로 받는다.
    # 시키는 동사 + 따옴표가 같이 있을 때만
    dictated_quote_not_a_value: bool = True

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


# 모름으로 시작하는 답. 복원문 기준(`ㅁㄹ`→`모르겠다`, `기억안남`→`기억 안 남`)
_BARE_UNKNOWN = re.compile(
    r"^\s*(?:모르겠|몰라|모름|기억\s*안\s*나|기억\s*안\s*남|기억이\s*안|글쎄|잘\s*모르)"
)
# 짧은 부정만 있는 답. 복원문 전체가 이것뿐일 때
_BARE_NEGATIVE = re.compile(
    r"^\s*(?:아니요|아니오|아니|아뇨|아님|없음|없어요|없어|없다|노)[\s.!~ㅋㅎㅠㅜ]*$"
)
# 예/아니오로 묻는 축과 그 "아니오" 값. 다른 축의 짧은 부정은 답이 아니다
_YES_NO_AXIS_NEGATION = {"radiation": "안 퍼짐", "associated": "없음"}


# 따옴표 안의 짧은 문자열. 한글 따옴표·낫표 포함
_QUOTED = re.compile(r"[\'\"‘“「『]([^\'\"’”」』\n]{1,40})[\'\"’”」』]")
# 받아쓰라는 말. 이것이 없으면 따옴표는 그냥 인용이다
_DICTATE = re.compile(
    r"(?:추가해|적어|적으|써\s*줘|써\.|써$|써라|쓰면|쓰고|기록|채워|채우|넣어|표기|작성|입력|write)"
)
_SEP_EDGES = re.compile(r"^[\s,·/;]+|[\s,·/;]+$")
_SEP_DOUBLE = re.compile(r"\s*[,·/;]\s*[,·/;]\s*")


def _strip_dictated(value: str | None, spans: list[str]) -> str | None:
    """값에서 받아쓰라고 시킨 문자열을 뺀다(공백 무시). 남는 게 없으면 None."""
    if not value:
        return value
    out = value
    for sp in spans:
        pat = r"\s*".join(map(re.escape, _norm(sp)))
        out = re.sub(pat, "", out)
    out = _SEP_DOUBLE.sub(", ", out)
    out = _SEP_EDGES.sub("", out).strip()
    return out or None


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

    axis_name = str(getattr(asked_axis, "value", asked_axis)) if asked_axis is not None else None
    # 둘 다 글자 있는 발화에만 — `ㄴㄴ`가 복원 없이 오면 5번 필터대로 버린다
    bare_unknown = (
        cfg.bare_unknown_is_unknown
        and asked_axis is not None
        and not letterless
        and bool(_BARE_UNKNOWN.match(normalized or utterance))
    )
    bare_negative = (
        cfg.bare_negative_fills_yes_no_axis
        and axis_name in _YES_NO_AXIS_NEGATION
        and not letterless
        and bool(_BARE_NEGATIVE.match(normalized or utterance))
    )

    # 받아쓰라고 시킨 따옴표 문자열(규칙 9). 시키는 동사가 없으면 빈 목록
    dictated: list[str] = []
    if cfg.dictated_quote_not_a_value and _DICTATE.search(utterance):
        dictated = [m.group(1).strip() for m in _QUOTED.finditer(utterance) if m.group(1).strip()]

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
        if reason is None and dictated and u.value:
            stripped = _strip_dictated(u.value, dictated)
            if stripped != u.value:
                dropped.append(
                    {
                        "axis": u.axis.value,
                        "reason": "dictated_quote_not_a_value",
                        "value": u.value,
                        "evidence": u.evidence,
                    }
                )
                if stripped is None:
                    # 시킨 말뿐이었다 — 갱신 자체를 버린다. 물은 축이면 엔진이 되묻거나 닫는다
                    continue
                u = AxisUpdate(axis=u.axis, status=u.status, value=stripped, evidence=u.evidence)
        if reason is None:
            if bare_affirm and u.axis == asked_axis and u.status != FieldStatus.AMBIGUOUS:
                # 값은 버리고 축은 살린다 — 엔진이 되묻는다. 무엇을 버렸는지는 dropped에 남긴다.
                # filled뿐 아니라 unknown·skipped도 — "네"는 "모름"도 "건너뜀"도 아니다
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
            elif bare_unknown and u.axis == asked_axis and u.status == FieldStatus.SKIPPED:
                dropped.append(
                    {
                        "axis": u.axis.value,
                        "reason": "bare_unknown_is_unknown",
                        "value": u.value,
                        "evidence": u.evidence,
                    }
                )
                u = AxisUpdate(
                    axis=u.axis, status=FieldStatus.UNKNOWN, value=None, evidence=u.evidence
                )
            elif bare_negative and u.axis == asked_axis and u.status != FieldStatus.FILLED:
                dropped.append(
                    {
                        "axis": u.axis.value,
                        "reason": "bare_negative_fills_yes_no_axis",
                        "value": u.value,
                        "evidence": u.evidence,
                    }
                )
                u = AxisUpdate(
                    axis=u.axis,
                    status=FieldStatus.FILLED,
                    value=_YES_NO_AXIS_NEGATION[axis_name],
                    evidence=u.evidence,
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

    # 모델이 물은 축을 비웠어도(또는 다른 규칙이 버렸어도) 짧은 긍정이면 되물음을 넣는다.
    # 없으면 엔진이 skipped로 닫고 넘어가 카드에 "잘 모르겠어요"가 찍힌다(2026-09-25 운영 확인).
    # 근거는 원문 그대로. 되물은 뒤에도 또 짧은 긍정이면 엔진이 그때는 skipped로 닫는다
    # (한 번만 되묻기)
    if bare_affirm and not any(u.axis == asked_axis for u in kept):
        kept.append(
            AxisUpdate(
                axis=asked_axis, status=FieldStatus.AMBIGUOUS, value=None, evidence=utterance
            )
        )
        dropped.append(
            {
                "axis": getattr(asked_axis, "value", asked_axis),
                "reason": "bare_affirmative_asks_back",
                "value": None,
                "evidence": utterance,
            }
        )

    # 모델이 물은 축을 비웠는데 "모르겠다"로 시작했으면 unknown, 짧은 부정이면 "없음"으로 닫는다.
    # 비워 두면 엔진이 skipped로 닫아 "말하고 싶지 않다"가 된다 — 환자가 한 말이 아니다
    if asked_axis is not None and not any(u.axis == asked_axis for u in kept):
        if bare_unknown:
            kept.append(
                AxisUpdate(
                    axis=asked_axis, status=FieldStatus.UNKNOWN, value=None, evidence=utterance
                )
            )
            dropped.append(
                {
                    "axis": axis_name,
                    "reason": "bare_unknown_is_unknown",
                    "value": None,
                    "evidence": utterance,
                }
            )
        elif bare_negative:
            kept.append(
                AxisUpdate(
                    axis=asked_axis,
                    status=FieldStatus.FILLED,
                    value=_YES_NO_AXIS_NEGATION[axis_name],
                    evidence=utterance,
                )
            )
            dropped.append(
                {
                    "axis": axis_name,
                    "reason": "bare_negative_fills_yes_no_axis",
                    "value": None,
                    "evidence": utterance,
                }
            )

    # chief_complaint에도 숫자 규칙을 적용한다 (첫 발화 요약에 숫자를 만들어 넣는 것 방지)
    cc = ext.chief_complaint
    if dictated and cc:
        cc2 = _strip_dictated(cc, dictated)
        if cc2 != cc:
            dropped.append(
                {"axis": "chief_complaint", "reason": "dictated_quote_not_a_value", "value": cc}
            )
            cc = cc2
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
