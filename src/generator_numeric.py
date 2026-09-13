from typing import Any

from src.generator_base import ValueGenerator


class NumericGenerator(ValueGenerator):
    """Generates a JSON number, digit by digit.

    There is no "end of number" token, so the tokens that would
    naturally follow the value (comma, closing brace) are offered at
    every step where the number is already valid. Picking one is a
    stop signal: it ends the value without being consumed, since the
    builder writes the real separator itself.

    A sign or a dot is never the last token written, so the value is
    always parseable.
    """

    MAX_TOKENS = 12

    def _allowed_next_ids(self, generated: list[int]) -> list[int]:
        vocab = self._vocab
        if not generated:
            return vocab.digits + [vocab.minus]
        if generated[-1] in (vocab.minus, vocab.dot):
            return list(vocab.digits)

        allowed = vocab.digits + [vocab.comma, vocab.rbrace]
        if vocab.dot not in generated:
            allowed.append(vocab.dot)
        return allowed

    def _is_stop_signal(self, generated: list[int], chosen_id: int) -> bool:
        return chosen_id in (self._vocab.comma, self._vocab.rbrace)

    def _is_complete(self, generated: list[int]) -> bool:
        if generated[-1] in (self._vocab.minus, self._vocab.dot):
            return False
        return len(generated) >= self.MAX_TOKENS

    def _decode(self, generated: list[int]) -> Any:
        text = self._model.decode(generated)
        try:
            return float(text) if "." in text else int(text)
        except ValueError:
            return 0
