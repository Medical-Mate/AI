"""질문 후보 — 앱 4단계 화면("의사에게 물어볼 것"). 요청으로 켜고, 종료 턴에만 나간다."""

import json

from fastapi.testclient import TestClient

from medimate.api.app import create_app
from medimate.llm import AxisUpdate, TurnExtraction
from medimate.schema import Axis, FieldStatus
from tests.fakes import ScriptedExtractor

FOUR = json.dumps(
    {
        "items": [
            {"text": "혈압약을 계속 먹어도 되나요?", "source": "medications"},
            {"text": "계단 내려갈 때만 아픈 건 어떤 상태인가요?", "source": "exacerbating"},
            {"text": "아침에 뻣뻣한 것도 관련이 있나요?", "source": "associated"},
            {"text": "2주째 비슷한데 언제까지 이러면 다시 와야 하나요?", "source": "time_course"},
        ]
    },
    ensure_ascii=False,
)


class Fake(ScriptedExtractor):
    """추출은 대본대로, 후보 생성은 `out`을 돌려준다. `out=None`이면 터진다."""

    def __init__(self, out=FOUR):
        super().__init__(
            [
                TurnExtraction(
                    chief_complaint="무릎",
                    updates=[
                        AxisUpdate(axis=a, status=FieldStatus.FILLED, value="x", evidence="x")
                        for a in Axis
                    ],
                )
            ]
        )
        self.out = out
        self.prompts: list[tuple[str, str]] = []

    def complete_json(self, system, user, schema):
        self.prompts.append((system, user))
        if self.out is None:
            raise RuntimeError("모델이 죽었다")
        return self.out, 100, 20


def finish(flag: bool, out=FOUR):
    ex = Fake(out)
    c = TestClient(create_app(lambda: ex))
    st = c.post("/v1/previsit/sessions", json={"site_node_id": "SUR:042", "side": "left"}).json()[
        "state"
    ]
    body = {"state": st, "utterance": "x 다 말했어요"}
    if flag:
        body["question_candidates"] = True
    return c.post("/v1/previsit/turns", json=body).json(), ex


def test_off_by_default_and_no_llm_call():
    """안 보내면 안 돈다. 온디바이스 프로필은 "발화가 외부로 안 나간다"가 전제이고,
    후보 생성은 카드 값을 외부 LLM으로 보낸다. 크레딧도 한정이다."""
    b, ex = finish(flag=False)
    assert b["ended"] is True
    assert b["card"]["question_candidates"] is None
    assert ex.prompts == []  # 후보 생성 호출이 아예 없다


def test_on_returns_ranked_top_three():
    b, ex = finish(flag=True)
    cands = b["card"]["question_candidates"]
    assert len(ex.prompts) == 1
    assert len(cands) == 3  # 생성은 제한하지 않고 위에서 3개를 자른다(모델은 4개를 냈다)
    assert [c["rank"] for c in cands] == [1, 2, 3]
    # 랭커 원칙: 환자가 꺼내야 처방에 반영되는 것을 올린다
    assert cands[0]["source"] == "medications"
    assert set(cands[0]) == {"text", "source", "rank"}


def test_generation_failure_never_loses_the_card():
    """문답을 다 마친 환자의 카드가 후보 생성 실패로 날아가면 안 된다.

    카드 완성을 강제하지 않는다는 우리 원칙과 같다 — 환자가 언제 끝내도 그 시점 카드가 결과다.
    """
    b, _ = finish(flag=True, out=None)
    assert b["ended"] is True
    assert b["card"]["question_candidates"] is None
    assert sum(1 for a in b["card"]["axes"].values() if a["value"]) == 8  # 카드는 멀쩡하다


def test_not_generated_before_the_session_ends():
    """중간 턴에는 `null`. 끝나야 재료가 다 모인다"""
    ex = Fake()
    # 축 하나만 채워 세션이 안 끝나게 한다
    ex._script = [
        TurnExtraction(
            chief_complaint="무릎",
            updates=[
                AxisUpdate(
                    axis=Axis.SITE, status=FieldStatus.FILLED, value="무릎", evidence="무릎이"
                )
            ],
        )
    ]
    c = TestClient(create_app(lambda: ex))
    st = c.post("/v1/previsit/sessions").json()["state"]
    b = c.post(
        "/v1/previsit/turns",
        json={"state": st, "utterance": "무릎이 아파요", "question_candidates": True},
    ).json()
    assert b["ended"] is False
    assert b["card"]["question_candidates"] is None
    assert ex.prompts == []  # 끝나기 전에는 부르지 않는다


def test_prompt_carries_no_identifiers():
    """이름·나이·성별·병원·진료일·ID는 카드에 있어도 프롬프트에 안 들어간다.

    이게 이 기능을 외부 LLM으로 돌리는 근거다(`docs/api-previsit.md` 지키는 선).
    """
    _, ex = finish(flag=True)
    _, user = ex.prompts[0]
    for banned in ("request_id", "req-", "세", "님", "병원", "진료일"):
        if banned in ("세", "님"):  # 축 값에 우연히 섞일 수 있는 글자는 제외
            continue
        assert banned not in user, banned
    assert user.startswith("카드:")


# --- patient_profile (2026-09-11 백엔드 합의 #7) ------------------------------
#
# 카드에 복용약·기저질환·알러지 자리가 없어서 프롬프트가 항상 "없음"을 봤다. eval에서 그 세
# 카테고리를 3% → 65%로 올린 규칙(questions-v8)이 제품에서 안 걸리는 상태였다.


def _turn(client, state, **extra):
    return client.post("/v1/previsit/turns", json={"state": state, "utterance": "x 끝", **extra})


def _client():
    ex = Fake()
    c = TestClient(create_app(lambda: ex))
    st = c.post("/v1/previsit/sessions", json={"site_node_id": "SUR:042", "side": "left"}).json()[
        "state"
    ]
    return c, ex, st


def test_profile_reaches_the_prompt():
    c, ex, st = _client()
    r = _turn(
        c,
        st,
        question_candidates=True,
        patient_profile={"medications": ["혈압약"], "conditions": ["고혈압"], "allergies": []},
    )
    assert r.status_code == 200
    _, user = ex.prompts[0]
    assert "혈압약" in user and "고혈압" in user


def test_profile_is_not_stored_on_the_card():
    """축이 아니고 이 문진에서 환자가 말한 것도 아니다. 재료로 쓰고 버린다"""
    c, _, st = _client()
    # 후보 대본에 안 나오는 값이라야 카드에 샌 것과 구분된다
    b = _turn(c, st, question_candidates=True, patient_profile={"medications": ["와파린"]}).json()
    assert "와파린" not in json.dumps(b["card"], ensure_ascii=False)
    assert "patient_profile" not in b["card"]
    assert "patient_profile" not in json.dumps(b["state"], ensure_ascii=False)


def test_omitting_the_field_is_the_same_as_before():
    """필드 없음 = 온보딩 안 거침. 빈 배열과 같은 프롬프트가 나간다"""
    c1, ex1, st1 = _client()
    _turn(c1, st1, question_candidates=True)
    c2, ex2, st2 = _client()
    _turn(c2, st2, question_candidates=True, patient_profile={})
    assert ex1.prompts[0] == ex2.prompts[0]


def test_blank_elements_are_dropped():
    """앱이 빈 입력칸을 그대로 보내면 프롬프트에 빈 줄이 들어간다"""
    c, ex, st = _client()
    _turn(
        c,
        st,
        question_candidates=True,
        patient_profile={"medications": ["", "  ", " 혈압약 "], "allergies": [""]},
    )
    _, user = ex.prompts[0]
    line = next(ln for ln in user.splitlines() if "복용약" in ln)
    assert "혈압약" in line and ", ," not in line


def test_rejected_when_candidates_are_off():
    """프롬프트가 아니라 **요청 본문**이 문제다.

    편의상 매 턴 붙여 보내면 20턴짜리 문진에서 건강정보가 20번 오가고, 서버 로그·에러
    리포트에 남는 표면이 그만큼 늘어난다. 규칙으로 부탁하지 않고 구조로 막는다.
    """
    c, ex, st = _client()
    r = _turn(c, st, patient_profile={"medications": ["혈압약"]})
    assert r.status_code == 422
    assert "question_candidates" in json.dumps(r.json(), ensure_ascii=False)
    assert ex.prompts == []


def test_unknown_profile_field_is_rejected():
    """`extra="forbid"` — 요청에 정의되지 않은 필드가 오면 422 (계약 문서 지키는 선)"""
    c, _, st = _client()
    r = _turn(
        c, st, question_candidates=True, patient_profile={"medications": [], "birth_year": 1990}
    )
    assert r.status_code == 422
