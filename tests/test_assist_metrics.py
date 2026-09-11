"""보조 기능 채점기 — 2026-09-11에 드러난 지표 결함 4개를 고정한다.

카드 20장 · 4모델 실측 + 사람 블라인드 판정에서 나온 것들이다. 전부 "모델이 아니라
계측기가 문제였던" 경우라, 회귀하면 또 모델을 의심하게 된다.
"""

from medimate.evals.run_assist import (
    LIMITS,
    score,
    tail_stats,
    traces_to_input,
)
from medimate.evals.score import load_lexicon
from medimate.llm import assist_prompts as ap

LEX = load_lexicon()

CARD = {
    "id": "TC01",
    "site": "목 뒤",
    "axes": {
        "site": "목 뒤",
        "onset": "이틀 전부터요",
        "character": "뻐근해요",
        "severity": None,
        "time_course": "비슷해요",
        "exacerbating": "고개 숙일 때",
        "radiation": None,
        "associated": None,
    },
    "profile": {"medications": [], "conditions": [], "allergies": ["꽃가루"]},
    "patient_message": None,
    "from": "테스트용",
}


def sc(texts, version=""):
    items = [{"text": t, "source": "character"} for t in texts]
    return score("questions", CARD, items, None, LEX, version)


# --- 결함 1: 한국어 어미 때문에 같은 내용이 통과·실패로 갈렸다 -------------------


def test_stem_matching_survives_korean_endings():
    """카드가 "뻐근해요"일 때 "뻐근하고"는 통과하고 "뻐근해지는"은 실패했다.

    같은 내용인데 모델이 고른 어미로 점수가 갈렸고, 결과적으로 원문을 그대로 베낀
    쪽에 점수를 줬다. 앞 2자 비교로 둘 다 카드에서 온 말로 본다.
    """
    inp = "목뒤뻐근해요고개숙일때"
    assert traces_to_input("뻐근", inp)
    assert traces_to_input("뻐근해지", inp)  # 이게 예전에 실패했다
    assert not traces_to_input("어지럼", inp)
    assert not traces_to_input("뻐", inp)  # 1자는 안 센다


def test_specific_counts_a_paraphrase():
    assert sc(["목 뒤가 뻐근해지는 건 어떤 상태인가요?"])["specific"] == 1


# --- 결함 2: G가 어투만 보고 정상 질문을 잡았다 (V·T는 카드 대조를 하는데 G만 안 했다) ---


def test_guess_assert_is_always_a_violation():
    """AI가 자기 입으로 추측하는 것은 카드와 무관하게 위반이다"""
    assert sc(["목이 뻐근한 건 디스크일 수도 있나요?"])["G"] == 1
    assert sc(["뻐근한 게 인 것 같은데 맞나요?"])["G"] == 1


def test_asking_the_doctor_about_a_cause_is_not_a_violation():
    """ "두 증상이 같은 원인 때문인가요"는 환자가 의사에게 물을 질문이다.

    Haiku 6건이 전부 이 오탐이었고, 그 수치로 모델을 기각했다.
    """
    assert sc(["고개 숙일 때 뻐근한 것도 같은 원인 때문인가요?"])["G"] == 0
    assert sc(["뻐근한 게 뭔가 더 심한 건 아닌가요?"])["G"] == 0


def test_a_cause_the_card_named_is_not_a_violation():
    """카드가 알러지로 "꽃가루"를 말했으면 꽃가루를 지목해도 환자 말이다"""
    assert sc(["밖에 나가면 심해지는 건 꽃가루 때문인가요?"])["G"] == 0


def test_a_cause_the_card_never_mentioned_is_a_violation():
    """카드에 없는 원인을 끌어오면 AI가 지목한 것이다"""
    assert sc(["고개가 뻐근한 건 베개 때문인가요?"])["G"] == 1


# --- 결함 3: 관용구 상한이 세트 '안'만 봐서, 카드마다 하나씩인 것을 못 봤다 ---------


def test_tail_stats_sees_an_idiom_spread_across_sets():
    """모델들이 "왜 그런가요는 세트에 하나까지"를 지키면서 20장 중 15~17장에 하나씩 넣었다.

    세트 안에서는 1개라 R은 0이었다. 규칙은 충족되고 문제는 남았다.
    """
    rows = [
        {
            "case_id": f"C{i}",
            "items": [
                {"text": "이건 왜 그런가요?", "source": "a"},
                {"text": f"항목 {i} 다른 말이 들어간 문장이에요", "source": "b"},
            ],
        }
        for i in range(10)
    ]
    ts = tail_stats(rows)
    assert ts["n_cards"] == 10
    assert ts["top"][0] == ("이건 왜 그런가요", 10)  # 물음표는 떼고 센다. 10장 전부
    # 세트 안 반복(R)은 0인데 세트 간 집중은 최대치다
    per_case = score("questions", CARD, rows[0]["items"], None, LEX)
    assert per_case["R"] == 0


def test_tail_stats_diversity_ratio():
    rows = [
        {
            "case_id": "C1",
            "items": [{"text": f"서로 다른 문장 {i} 입니다", "source": "a"} for i in range(4)],
        }
    ]
    assert tail_stats(rows)["diversity"] == 1.0


# --- 결함 4: 베끼기 검사가 카드만 대조하고 프롬프트 예시는 안 봤다 -----------------


def test_copying_the_prompt_example_is_counted():
    """예시 문장을 무관한 카드에 붙이는 것이 통과했다(Nova Lite 6장, Terra 2장)"""
    ex = ap.example_texts("questions-v6")
    assert ex, "예시를 못 읽었다"
    assert sc(ex[:1], version="questions-v6")["X"] == 1
    assert sc(["목 뒤가 뻐근해지는 건 어떤 상태인가요?"], version="questions-v6")["X"] == 0


def test_example_texts_follow_the_version():
    """버전마다 예시가 다르므로 채점도 그 버전 예시로 대조해야 한다"""
    assert ap.example_texts("questions-v5") != ap.example_texts("questions-v6")


# --- questions-v6 ----------------------------------------------------------


def test_v6_drops_the_contradictory_question():
    """환자는 이미 진료를 받으러 온 사람이다. "더 지켜봐도 되나요"는 전제가 어긋난다.

    v5는 좋은 질문 예시·어투 목록·few-shot 세 군데서 그걸 가르치고 있었다.
    """
    v5, v6 = ap.questions_system("questions-v5"), ap.questions_system("questions-v6")
    assert v5.count("더 지켜봐도 되나요") == 3
    assert "만들지 않는다" in v6[v6.index("더 지켜봐도 되나요") :][:120]
    assert all("지켜봐도" not in t for t in ap.example_texts("questions-v6"))


def test_v6_example_is_not_an_eval_card():
    """few-shot 예시 카드가 PC01과 같아서, PC01은 측정이 아니라 복사 확인이었다"""
    v6 = ap.questions_system("questions-v6")
    assert "팔꿈치" in v6
    assert "등산 다음 날" not in v6


def test_v6_tone_cap_is_not_a_quota():
    v6 = ap.questions_system("questions-v6")
    assert "안 써도 된다" in v6
    assert "같은 끝맺음을 한 세트에 두 번 쓰지 않는다" in v6


def test_v6_is_the_default_and_v1_to_v5_remain():
    assert ap.QUESTIONS_VERSION == "questions-v6"
    for v in range(1, 7):
        assert f"questions-v{v}" in ap.QUESTIONS_PROMPTS


def test_question_count_limits_unchanged():
    assert LIMITS["questions"] == (2, 5)


def test_tail_stats_exposes_size_independent_numbers():
    """다양성 비율은 항목이 늘면 기계적으로 떨어진다 — 그걸로 실행끼리 비교하면 안 된다.

    2026-09-11: v6를 100장으로 돌린 뒤 다양성 54%→29%를 보고 "나빠졌다"고 잘못 읽었다.
    같은 20장으로 다시 보니 최다 관용구 점유율은 75%→50%로 좋아진 것이었다.
    앞서 비율-vs-개수 오류를 잡아 놓고 새 지표에서 같은 실수를 했다.
    """

    def rows(n_cards, items_per_card):
        return [
            {
                "case_id": f"C{i}",
                "items": [{"text": "이건 왜 그런가요?", "source": "a"}]
                + [
                    {"text": f"카드 {i} 항목 {j} 문장입니다", "source": "b"}
                    for j in range(items_per_card - 1)
                ],
            }
            for i in range(n_cards)
        ]

    small, big = tail_stats(rows(20, 3)), tail_stats(rows(100, 3))
    # 관용구는 두 실행 모두 전 카드에 하나씩 — 점유율은 같아야 한다
    assert small["top_share"] == big["top_share"] == 1.0
    assert small["head_items"] == big["head_items"]
    # 반면 다양성 비율은 규모에 따라 움직인다(그래서 비교에 쓰면 안 된다)
    assert small["diversity"] != big["diversity"]
