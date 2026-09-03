"""테스트용 Extractor — 미리 정한 추출 결과를 순서대로 돌려준다."""

from medimate.llm.base import TurnExtraction
from medimate.schema.card import Axis


class ScriptedExtractor:
    model_id = "fake"
    prompt_version = "test"

    def __init__(self, script: list[TurnExtraction]):
        self._script = list(script)
        self.calls: list[tuple[str, Axis | None]] = []
        self.histories: list[list] = []

    def extract(self, utterance: str, asked_axis: Axis | None, history=()) -> TurnExtraction:
        self.calls.append((utterance, asked_axis))
        self.histories.append(list(history))
        return self._script.pop(0) if self._script else TurnExtraction()
