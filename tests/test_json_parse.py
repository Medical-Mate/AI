"""코드펜스를 두르는 모델이 있다 — 파서가 못 벗기면 점수가 0으로 잡힌다.

2026-09-11 실제 사고: questions eval에서 Haiku·Nova Lite가 20/20을 ```json으로 감쌌고
`parse_items`가 펜스를 안 벗겨 전부 JSONDecodeError로 집계됐다. 모델 실력 차가 아니라 파서 차였다.
추출(providers)·보조(assist_prompts)·메모(memo_classifier) 세 경로가
같은 파서를 쓰는지도 함께 고정한다.
"""

from medimate.llm import assist_prompts as ap
from medimate.llm.base import parse_json_text

ITEMS = '{"items": [{"text": "무릎이 언제부터 아팠나요?", "source": "onset"}]}'


def test_bare_json():
    assert parse_json_text(ITEMS)["items"][0]["source"] == "onset"


def test_fenced_json():
    for wrapped in (
        f"```json\n{ITEMS}\n```",
        f"```\n{ITEMS}\n```",
        f"  ```json\n{ITEMS}\n```  ",
    ):
        assert parse_json_text(wrapped)["items"][0]["source"] == "onset"


def test_parse_items_accepts_fence():
    """보조 프롬프트 경로. 한글은 글자 그대로 보존된다"""
    items = ap.parse_items(f"```json\n{ITEMS}\n```")
    assert items == [{"text": "무릎이 언제부터 아팠나요?", "source": "onset"}]


def test_extraction_path_shares_the_parser():
    from medimate.llm.providers import _parse_json

    ext = _parse_json(
        '```json\n{"updates": [{"axis": "onset", "status": "filled",'
        ' "value": "3일 전", "evidence": "3일 전부터요"}]}\n```'
    )
    assert ext.updates[0].value == "3일 전"


def test_memo_path_shares_the_parser():
    from medimate.llm.memo_classifier import parse_json_text as memo_parser

    assert memo_parser is parse_json_text


def test_safe_model_name_strips_colon():
    """Bedrock 추론 프로파일 ID의 `:`가 윈도우 대체 데이터 스트림을 만들었다"""
    from medimate.evals.score import safe_model_name

    assert safe_model_name("apac.amazon.nova-pro-v1:0") == "apac.amazon.nova-pro-v1_0"
    assert safe_model_name("gpt-5.6-terra") == "gpt-5.6-terra"
    assert safe_model_name("a|b?c*d") == "a_b_c_d"


def test_safe_model_name_keeps_the_existing_local_convention():
    """`/`는 `-`로 간다. 쌓인 결과 파일이 그 이름이라 바꾸면 재채점이 끊긴다"""
    from medimate.evals.score import safe_model_name

    assert safe_model_name("local/Qwen3-1.7B-Q4_0") == "local-Qwen3-1.7B-Q4_0"


def test_budget_guard_refuses_an_unpriced_model():
    """PRICES에 없으면 비용이 0으로 계산돼 --budget이 영원히 안 걸린다"""
    import pytest

    from medimate.llm.providers import BudgetExceeded, require_price

    assert require_price("gpt-5.6-terra") == (2.00, 12.00)
    assert require_price("local/Qwen3-1.7B-Q4_0") == (0.0, 0.0)  # 온디바이스는 0이 맞다
    with pytest.raises(BudgetExceeded, match="PRICES에 없다"):
        require_price("gpt-5.6-terra-2026-09-11")  # 날짜 접미사 임의 추가 같은 오타
