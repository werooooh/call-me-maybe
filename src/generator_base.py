from abc import ABC, abstractmethod
from typing import Any

from llm_sdk import Small_LLM_Model

from src.tokens import TokenVocabulary, pick_best


class ValueGenerator(ABC):
    """Generates one JSON value, token by token, under a constraint.

    A subclass only has to say which tokens keep the value valid for
    its type at each step; the model never chooses outside that set.
    When the set holds a single token the model is not queried at all:
    the choice is already made, and skipping that forward pass is what
    makes writing a known value free.
    """

    PREFIX = ""
    SUFFIX = ""
    MAX_STEPS = 1024

    def __init__(self, model: Small_LLM_Model, vocab: TokenVocabulary) -> None:
        self._model = model
        self._vocab = vocab
        self._source_text = ""

    def generate(
        self, context_ids: list[int], source_text: str = ""
    ) -> tuple[list[int], Any]:
        """Return the tokens written for the value and the value."""
        self._source_text = source_text
        generated: list[int] = []

        for _ in range(self.MAX_STEPS):
            allowed_ids = self._allowed_next_ids(generated)
            if not allowed_ids:
                break

            chosen_id = self._choose(context_ids + generated, allowed_ids)
            if self._is_stop_signal(generated, chosen_id):
                break

            generated.append(chosen_id)
            if self._is_complete(generated):
                break

        return generated, self._decode(generated)

    def _choose(self, input_ids: list[int], allowed_ids: list[int]) -> int:
        if len(allowed_ids) == 1:
            return allowed_ids[0]
        logits = self._model.get_logits_from_input_ids(input_ids)
        return pick_best(logits, allowed_ids)

    def _is_stop_signal(self, generated: list[int], chosen_id: int) -> bool:
        """True when the token ends the value without belonging to it."""
        return False

    @abstractmethod
    def _allowed_next_ids(self, generated: list[int]) -> list[int]:
        """Tokens that keep the value valid at this step."""

    @abstractmethod
    def _is_complete(self, generated: list[int]) -> bool:
        """True when the value is finished."""

    @abstractmethod
    def _decode(self, generated: list[int]) -> Any:
        """Turn the generated tokens into a Python value."""
