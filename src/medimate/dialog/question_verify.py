"""질문 후보 검증기 — 모델이 쓴 질문이 **카드에 없는 말을 하지 않았나**를 기계로
확인한다(2026-10-08).

앱 경로는 추론만 폰(Qwen3-1.7B)에서 하고, 검증·폴백은 AI 서버에 둔다. 폰이 만든 후보를 여기서
거르고, 빈자리는 템플릿(`question_templates`)으로 채운다. 환자가 후보를 고르거나 고쳐 쓰는 구조라
자연스러움은 100%일 필요가 없다. **여기서 막는 것은 틀린 말이다.**

진료 후 값 검증기(`span/verify.py`, B안)와 태스크가 다르다 — 그쪽 출력은 조각을 **줄여 쓴 값**이라
새 낱말이 거의 없어야 하고, 이쪽 출력은 카드를 재료로 쓴 **새 문장**이라 질문 틀 말("해도 되나요",
"다시 와야")이 반드시 붙는다. Kiwi 형태소 추출과 부정 절 자르기만 가져다 쓴다.

항목 하나하나에
1. 형식 — 물음표로 끝남, 40자(백엔드 검증선) 안, 한자·중국어 없음
2. 낱말 출처 — 내용 형태소가 카드(축 값·복용약 등·덧붙인 말)에 있거나 질문 틀 허용 표에 있어야 한다.
   "술을 마시면", "깃털이랑"이 여기서 떨어진다
3. 숫자 — 카드에 있는 숫자만
4. 부정 — (a) 카드에 없는 부정을 더하지 않는다(질문 틀의 "~하면 안 되나요"는 뺀다)
   (b) 부정이 있는 카드 값에서 가져왔으면 그 부정을 지킨다. "안 쓸 수가 없어요" → "안 쓸 수가
   있는데"
5. 용어 — 병명 사전·검사·약 이름은 카드에 글자 그대로 있어야 한다. 카드에 있어도 **병명이 맞는지
   되묻거나**(진단 요청) **다른 사람이 들은 병명**이면 버린다
6. 어투 — 원인 짐작("가능성", "~일 수도"), "왜"(v8부터 금지), 환자에게 묻는 꼴, 깨진 어미
세트에서
7. 반복 — 앞서 통과한 질문과 카드 재료가 거의 같으면 뒤의 것을 버린다
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from medimate.dialog.question_templates import MIN_ITEMS, template_candidates
from medimate.span.verify import _clauses, _content
from medimate.span.verify import _neg as _span_neg
from medimate.text.tokenize import base_kiwi, tokens

MAX_CHARS = 40  # 백엔드 검증선
MAX_ITEMS = 5

# 질문 틀 낱말 — 카드에 없어도 쓸 수 있다. **환자의 사실을 새로 말하지 않는 말만** 넣는다.
# 넣지 않은 것: `낫`(쥐가 "더 나요"를 "더 나은"으로 뒤집은 자리, 2026-10-08 LoRA v1 PC99),
# `운동`·`술`·`음식`처럼 화제를 새로 여는 낱말
FRAME = {
    # 되다·하다·있다류
    "되",
    "하",
    "있",
    "괜찮",
    "같",
    "다르",
    "보",
    "알",
    "모르",
    "받",
    "두",
    "놓",
    # 묻는 말
    "어떻",
    "어떠",
    "어떤",
    "얼마나",
    "얼마",
    "언제",
    "어디",
    "무슨",
    "뭐",
    "무엇",
    "혹시",
    "꼭",
    # 진료·처치 일반
    "병원",
    "진료",
    "의사",
    "선생",
    "선생님",
    "말씀",
    "치료",
    "관리",
    "조심",
    "주의",
    "필요",
    "처방",
    "반영",
    "확인",
    "설명",
    # 상태·관계 일반
    "상관",
    "관련",
    "영향",
    "원인",
    "이유",
    "상태",
    "증상",
    "정상",
    "이상",
    "문제",
    "심하",
    "심각",
    "걱정",
    "위험",
    "괜히",
    "그냥",
    # 시간·정도
    "계속",
    "다시",
    "또",
    "자주",
    "오래",
    "동안",
    "기간",
    "정도",
    "지금",
    "요즘",
    "이번",
    "앞으로",
    "나중",
    "아직",
    "이미",
    "먼저",
    "바로",
    "빨리",
    "점점",
    "더",
    "덜",
    "좀",
    "많이",
    "조금",
    "그대로",
    # 행동 일반
    "오",
    "가",
    "쓰",
    "참",
    "쉬",
    "피하",
    "기다리",
    "지내",
    "생기",
    "나타나",
    "지속",
    "좋",
    "변하",
    "달라지",
    "바뀌",
    "줄",
    "버티",
    "나빠지",
    # 지시·의존
    "이렇",
    "그렇",
    "이러",
    "그러",
    "이런",
    "그런",
    "이",
    "그",
    "저",
    "제",
    "것",
    "거",
    "수",
    "때",
    "데",
    "경우",
    "쪽",
    "번",
    "등",
    "만",
    "중",
    # 아래는 학습 카드 Nova 정답(490장)에서 걸린 낱말 중 **환자의 사실을 새로 말하지 않는
    # 것**(2026-10-08).
    # 평가 카드 100장이 아니라 학습 카드로 맞췄다. 걸렸지만 넣지 않은 것:
    # 무리·다치·기억나·그때·이전·계기
    # (환자에게 사건을 묻는 말), 두통·위장·음식·옷·부작용(새 증상·새 화제·추론), 진통제·검사(사전이
    # 본다),
    # 간지럽·가려움증·메슥거리처럼 카드 낱말을 **다른 낱말로 바꾼 것**(같은 뜻인지 기계가 모른다 —
    # 템플릿으로 폴백)
    # 증상 일반·축 이름
    "느낌",
    "느끼",
    "아프",
    "아픔",
    "통증",
    "불편",
    "부위",
    "곳",
    "부분",
    "동반",
    "악화",
    "완화",
    "경과",
    "시작",
    "퍼지",
    "질환",
    "기저",
    # 관계·의미
    "때문",
    "탓",
    "의미",
    "현상",
    "도움",
    "변화",
    "다른",
    "함께",
    "같이",
    "동시",
    "특별히",
    "특이",
    "점",
    "방법",
    "신경",
    "미치",
    "심각하",
    "징조",
    # 시간·흐름
    "갑자기",
    "완전히",
    "이어지",
    "유지",
    "반복",
    "이대로",
    "지켜보",
    "걸리",
    "사라지",
    "줄어들",
    "나아지",
    "일어나",
    "전",
    "후",
    "시",
    "적",
    "없이",
    "지나",
    "시간",
    "여전히",
    # 묻는 방식·조언 일반 — "그런 동작을 피해야 하나요"의 `동작`은 카드 행동을 가리키는 말이다
    "동작",
    "상황",
    "궁금하",
    "효과",
    "흔하",
    "조치",
    "취하",
    "똑같",
    "다루",
    "치료법",
    "특별",
    "줄이",
    # 묻는 꼴의 일반 말 — 부정 자체는 4번이 따로 센다
    "왜",
    "안",
    "못",
    "없",
    "나",
    "들",
    "나오",
    "뿐",
}
# 카드에 근거가 있을 때만 허용하는 틀 낱말
FRAME_IF = {
    "약": r"약|복용|알러지|알레르기",
    "먹": r"약|먹|복용",
    "복용": r"약|먹|복용",
    # 알러지 칸 값은 알레르겐 이름(`꽃가루`)이라 카테고리명이 카드에 없다 — 프로필로 판단
    "알러지": r".",
    "알레르기": r".",
    "기저질환": r".",
    "지병": r".",
}

_DIGITS = re.compile(r"\d+")
_WS = re.compile(r"\s+")
_CJK = re.compile(r"[㐀-鿿豈-﫿]")
# 질문 틀의 부정 — 환자의 사실이 아니라 묻는 방식이다
_FRAME_NEG = re.compile(
    r"(?:으?면|해도|어도|아도|봐도|가도|와도)\s*안\s*(?:되|돼)"
    r"|안\s*(?:하|가|와|먹|써|해|봐)도\s*(?:되|돼|괜찮)"
    r"|(?:지|하지|가지|먹지)\s*않아도\s*(?:되|돼|괜찮)"
    r"|안\s*(?:되나요|될까요|돼요|되는)"
    r"|(?:문제|걱정할\s*(?:건|게|거)|상관|지장)\s*(?:는|은)?\s*없"
    r"|지\s*말아야"
)
_GUESS = re.compile(r"가능성|일\s*수도|일\s*수\s*있|인\s*것\s*같|아닐까|때문인\s*것|탓인\s*것")
_WHY = re.compile(r"왜")
_HANGUL_SYL = re.compile(r"[가-힣]")
_VOICE = re.compile(
    r"복용\s*중이에요\?|복용\s*중이시죠|하시나요\?|이시죠\?|습관이 있나요|갔나요\?|드셨나요|"
    r"있으세요\?|"
    r"중이에요\?|물어보세요|확인해\s*주세요|알려\s*주세요"
)
# LoRA v1이 만든 말이 안 되는 꼬리(2026-10-08 통과분 사람 판독) — 낱말 출처로는 못 잡는다
_AWKWARD = re.compile(r"지속해야\s*하나요|걱정할\s*수\s*있나요|설쳐야|궁금해요\?$")
_GRAM = re.compile(
    r"아파요는|아파는|프는|져는|된가요|되는가요|는가요\?|있는가요|풀려는|아프는|돼요는|나요는"
)
_DX_ASK = re.compile(r"(?:인가요|맞나요|아닌가요|일까요|인지|이에요|예요)\s*\??\s*$|진단")
_OTHER_PERSON = re.compile(
    r"엄마|아빠|어머니|아버지|부모|가족|형|언니|누나|오빠|동생|할머니|할아버지|남편|아내|와이프|"
    r"친구|지인|동료|아이|애가|딸|아들|이웃|인터넷|유튜브|검색|찾아보|"
    # 들은 말·찾아본 말 — "찾아보니 협착증이라는 말이 나오는데" → "협착증과 관련이 있나요"(PC89,
    # LoRA v1)
    r"라고\s*(?:하|들|했)|라는\s*(?:말|얘기)|얘기를\s*들"
)
_GENERIC = {"하", "있", "되", "것", "거", "수", "좀", "다시", "더"}


@dataclass
class QVerdict:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    unsourced: list[str] = field(default_factory=list)

    @property
    def reason(self) -> str:
        return ";".join(self.reasons)


@dataclass
class Card:
    """검증용 카드 — 축 값·프로필·덧붙인 말을 조각으로 쪼개 형태소를 미리 뽑아 둔다."""

    segments: list[str]
    text: str
    compact: str
    content: set[str]
    seg_forms: list[list[tuple[str, str]]]
    has_profile: dict[str, bool]


def _neg(forms: list[tuple[str, str]]) -> list[str]:
    """span/verify의 부정 세기에서 동사 `안다`(`아이를 안을 때`, 안/VV)를 뺀다 — 부정 `안`은 MAG다.
    그대로 두면 "아이를 안아야 해서 손을 안 쓸 수가 있는데"가 부정 둘로 세져 PC80 뒤집기가
    통과했다."""
    return _span_neg([(f, t) for f, t in forms if not (f == "안" and t.startswith("VV"))])


def _forms(text: str, kiwi) -> list[tuple[str, str]]:
    return [(t.form, t.tag) for t in tokens(kiwi, text)]


def _usable(v: Any) -> bool:
    return isinstance(v, str) and bool(v.strip()) and not v.strip().startswith("(")


def prepare_card(card: dict, kiwi=None) -> Card:
    kiwi = kiwi or base_kiwi()
    segs: list[str] = []
    for v in (card.get("axes") or {}).values():
        if _usable(v):
            segs.append(v.strip())
    prof = card.get("profile") or {}
    for key in ("medications", "conditions", "allergies"):
        segs += [x for x in prof.get(key) or [] if _usable(x)]
    if _usable(card.get("patient_message")):
        segs.append(card["patient_message"].strip())
    seg_forms = [_forms(s, kiwi) for s in segs]
    content = {f for fs in seg_forms for f, _ in _content(fs)}
    text = " / ".join(segs)
    return Card(
        segs,
        text,
        _WS.sub("", text),
        content,
        seg_forms,
        {
            k: bool([x for x in prof.get(k) or [] if _usable(x)])
            for k in ("medications", "conditions", "allergies")
        },
    )


def _sourced(f: str, card: Card) -> bool:
    if f in card.content:
        return True
    # 어간·분절 차이(`간수치`↔`간수/치`, `저려`↔`저리`): 글자 포함으로 본다 — span/verify와 같은
    # 규칙
    if len(f) >= 2 and (
        f in card.compact or any(f in s or s in f for s in card.content if len(s) >= 2)
    ):
        return True
    # 마지막 음절 받침만 다른 경우(`저리`↔카드 `저림`, `찔리`↔`찔림`) — 명사형 `-ㅁ`이 붙은 카드 값
    if len(f) >= 2 and _HANGUL_SYL.fullmatch(f[-1]):
        head, last = f[:-1], (ord(f[-1]) - 0xAC00) // 28
        for i in range(len(card.compact) - len(f) + 1):
            c = card.compact[i + len(head)]
            if (
                card.compact.startswith(head, i)
                and _HANGUL_SYL.fullmatch(c)
                and (ord(c) - 0xAC00) // 28 == last
            ):
                return True
    return False


def _frame_ok(f: str, card: Card) -> bool:
    if f in FRAME:
        return True
    cond = FRAME_IF.get(f)
    if cond is None:
        return False
    if f in ("알러지", "알레르기"):
        return card.has_profile["allergies"] or bool(re.search("알러지|알레르기", card.text))
    if f in ("기저질환", "지병"):
        return card.has_profile["conditions"]
    return bool(re.search(cond, card.text)) or card.has_profile["medications"]


def verify_question(text: str, card: Card, lexicon: list[str] | None = None, kiwi=None) -> QVerdict:
    kiwi = kiwi or base_kiwi()
    t = (text or "").strip()
    reasons: list[str] = []

    # 1. 형식
    if not t.endswith("?"):
        reasons.append("not_question")
    if len(t) > MAX_CHARS:
        reasons.append("too_long")
    if _CJK.search(t) and not _CJK.search(card.text):
        reasons.append("cjk")

    # 2. 낱말 출처
    q_forms = _forms(t, kiwi)
    q_content = [f for f, _ in _content(q_forms)]
    unsourced = [f for f in q_content if not _sourced(f, card) and not _frame_ok(f, card)]
    if unsourced:
        reasons.append("unsourced:" + ",".join(dict.fromkeys(unsourced)))

    # 3. 숫자
    if set(_DIGITS.findall(t)) - set(_DIGITS.findall(card.text)):
        reasons.append("digits")

    # 4. 부정 — 질문 틀의 부정은 빼고 센다
    # 더한 부정은 질문 틀의 부정("~하면 안 되나요")을 빼고 센다. 지운 부정은 빼지 않고 센다 —
    # 카드 값 "집중이 안 돼요"를 인용한 질문에서 틀 패턴이 그 부정까지 지워 negation_dropped가 났다
    q_neg = _neg(_forms(_FRAME_NEG.sub(" 되", t), kiwi))
    q_neg_full = _neg(q_forms)
    drawn_content = {f for f in q_content if f in card.content} - _GENERIC - FRAME
    touched = []  # 낱말이 하나라도 겹치는 카드 절 — (a) 더한 부정의 기준
    drawn = []  # 실제로 가져온 절 — 겹침 2개 이상, 또는 절의 내용 전부와 겹침. (b)의 기준
    for fs in card.seg_forms:
        for c in _clauses(fs):
            cc = {f for f, _ in _content(c)} - _GENERIC - FRAME
            ov = drawn_content & cc
            if ov:
                touched.append(c)
                # "낮에는 아무 문제 없어요"의 `낮` 하나만 겹친 질문까지 부정을 요구하면 안 된다(PC99
                # 오거절)
                if len(ov) >= 2 or ov == cc:
                    drawn.append(c)
    if len(q_neg) > max((len(_neg(c)) for c in touched), default=0):
        reasons.append("negation_added")
    # 가져온 절 중 부정이 가장 많은 절 기준(span/verify와 같다). 최소로 보면 부정 없는 다른 절과
    # 겹치는 것만으로 면제된다 — PC80 "손을 안 쓸 수가 없어요" → "안 쓸 수가 있는데"가 "아이를 안을
    # 때"와
    # 겹쳐 통과했다(2026-10-08)
    elif len(q_neg_full) < max((len(_neg(c)) for c in drawn), default=0):
        reasons.append("negation_dropped")

    # 5. 용어
    compact = _WS.sub("", t)
    for term in lexicon or []:
        tc = _WS.sub("", term)
        if tc and tc in compact:
            if tc not in card.compact:
                reasons.append(f"new_term:{term}")
            elif _DX_ASK.search(t):
                reasons.append(f"diagnosis_ask:{term}")
            elif any(tc in _WS.sub("", s) and _OTHER_PERSON.search(s) for s in card.segments):
                reasons.append(f"other_person_dx:{term}")
            break

    # 6. 어투
    if _GUESS.search(t):
        reasons.append("guess")
    if _VOICE.search(t):
        reasons.append("voice")
    if _GRAM.search(t) or _AWKWARD.search(t):
        reasons.append("grammar")

    return QVerdict(not reasons, reasons, unsourced)


def _key(text: str, card: Card, kiwi) -> set[str]:
    """반복 판정용 — 질문이 카드에서 가져온 내용 형태소."""
    return {f for f, _ in _content(_forms(text, kiwi)) if _sourced(f, card)} - _GENERIC - FRAME


def verify_items(
    card: dict, items: list[dict] | None, lexicon: list[str] | None = None, kiwi=None
) -> tuple[list[dict], list[dict]]:
    """(통과, 버림). 버린 항목에는 `reasons`가 붙는다."""
    kiwi = kiwi or base_kiwi()
    c = prepare_card(card, kiwi)
    kept: list[dict] = []
    dropped: list[dict] = []
    keys: list[set[str]] = []
    n_why = 0
    for it in items or []:
        text = str(it.get("text") or "")
        v = verify_question(text, c, lexicon, kiwi)
        reasons = list(v.reasons)
        if not reasons:
            k = _key(text, c, kiwi)
            if not k:
                reasons.append("no_card_content")
            elif any(len(k & p) / len(k | p) >= 0.6 for p in keys if p):
                reasons.append("duplicate")
            elif _WHY.search(text) and n_why >= 1:
                reasons.append("why_repeat")
            else:
                keys.append(k)
                n_why += bool(_WHY.search(text))
        if reasons:
            dropped.append({**it, "reasons": reasons})
        else:
            kept.append(dict(it))
    return kept[:MAX_ITEMS], dropped


def finalize(
    card: dict,
    items: list[dict] | None,
    lexicon: list[str] | None = None,
    kiwi=None,
    templates=None,
) -> tuple[list[dict], list[dict]]:
    """검증 통과분 + 모자라면 템플릿으로 채운다(MIN_ITEMS까지). 각 항목에 `origin`(model|template).

    `templates(axes, profile)` — 기본은 v1(인용). v2는
    `question_templates_v2.natural_candidates`."""
    kept, dropped = verify_items(card, items, lexicon, kiwi)
    out = [{**it, "origin": "model"} for it in kept]
    used = {it.get("source") for it in out}
    if len(out) < MIN_ITEMS:
        for tpl in (templates or template_candidates)(card.get("axes") or {}, card.get("profile")):
            if len(out) >= MIN_ITEMS:
                break
            if tpl["source"] in used:
                continue
            out.append({**tpl, "origin": "template"})
            used.add(tpl["source"])
    return out, dropped
