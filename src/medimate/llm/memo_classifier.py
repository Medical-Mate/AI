"""진료 후 메모 문장 분류기 — LLM 어댑터 (챗봇② B1/B5).

LLMExtractor의 공급자 호출을 빌려 쓴다(같은 서버·같은 설정·같은 지출 가드).
프롬프트와 출력 스키마만 다르다. 출력은 번호 키 객체 {"0": 라벨, …}를 스키마로 강제한다
(memo-small-v4). 폰(llama.cpp)과 서버(OpenAI)가 같은 스키마를 받는다.
"""

from __future__ import annotations

from collections.abc import Sequence

from medimate.dialog.memo import MemoLabels
from medimate.llm import prompt_memo_small
from medimate.llm.providers import LLMExtractor, _parse_json_text


class LLMMemoClassifier:
    def __init__(self, provider: str, model_id: str, budget_usd: float = 0.5, client=None):
        self.ex = LLMExtractor(provider, model_id, budget_usd=budget_usd, _client=client)
        self.model_id = model_id
        self.prompt_version = prompt_memo_small.PROMPT_VERSION
        self.last_text = ""
        self.last_tokens = (0, 0)

    @property
    def usage(self):
        return self.ex.usage

    def classify(self, sentences: Sequence[str]) -> MemoLabels:
        self.ex.response_schema = MemoLabels.keyed_schema_for(len(sentences))
        text, i, o = self.ex._call(
            prompt_memo_small.system_prompt(), prompt_memo_small.user_message(sentences)
        )
        self.ex.usage.calls += 1
        self.ex.usage.input_tokens += i
        self.ex.usage.output_tokens += o
        self.last_text, self.last_tokens = text, (i, o)
        return MemoLabels.from_keyed(_parse_json_text(text), len(sentences))


class FixedLabels:
    """LLM을 부르지 않는다. 폰이 이미 분류했거나(온디바이스), 앱이 1q-2에서 라벨을 고쳤을 때."""

    model_id = "client"
    prompt_version = "client-labels"
    last_text = ""
    last_tokens = (0, 0)

    def __init__(self, keyed: dict[str, str]):
        self._keyed = keyed

    def classify(self, sentences: Sequence[str]) -> MemoLabels:
        return MemoLabels.from_keyed(self._keyed, len(sentences))
