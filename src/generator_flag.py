from typing import Any

from src.generator_base import ValueGenerator


class FlagGenerator(ValueGenerator):
    """Generates a JSON boolean: one token, either "true" or "false"."""

    def _allowed_next_ids(self, generated: list[int]) -> list[int]:
        """Only the two boolean literals are ever valid."""
        return [self._vocab.true_id, self._vocab.false_id]

    def _is_complete(self, generated: list[int]) -> bool:
        """A boolean is one token long."""
        return len(generated) == 1

    def _decode(self, generated: list[int]) -> Any:
        """True when the written token is the "true" literal."""
        return bool(generated) and generated[0] == self._vocab.true_id
