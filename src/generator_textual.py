from llm_sdk import Small_LLM_Model

from src.generator_base import ValueGenerator
from src.tokens import TokenVocabulary


class TextualGenerator(ValueGenerator):
    """Generates the content of a JSON string.

    Two regimes share the same loop. While what has been written is
    the beginning of a passage of the request, the value is a copy:
    the only allowed token is the next one of that passage, so the
    copy cannot drift and costs no forward pass. Otherwise the value
    is written freely, restricted to tokens that cannot break out of
    the string, with a length cap and a repetition guard.

    The quotes themselves are written by the builder (PREFIX and
    SUFFIX), so the model sees the natural tokenisation of `: "`.
    Choosing a token that starts with a quote is a stop signal: it
    ends the value without being consumed, exactly like a comma ends
    a number.
    """

    PREFIX = '"'
    SUFFIX = '"'
    MAX_FREE_TOKENS = 16
    LOOP_WINDOW = 3

    def __init__(
        self, model: Small_LLM_Model, vocab: TokenVocabulary
    ) -> None:
        super().__init__(model, vocab)
        self._spans_cache: tuple[str, list[str]] = ("", [])

    def _allowed_next_ids(self, generated: list[int]) -> list[int]:
        vocab = self._vocab
        content = self._model.decode(generated)

        if not content:
            return vocab.string_tokens

        matches = [s for s in self._spans() if s.startswith(content)]
        if matches:
            return self._copy_ids(matches, len(content))

        if len(generated) >= self.MAX_FREE_TOKENS:
            return [vocab.quote]
        if self._is_looping(generated):
            return [vocab.quote]
        return vocab.string_or_closing

    def _copy_ids(self, matches: list[str], position: int) -> list[int]:
        """Next token of each passage still being copied."""
        vocab = self._vocab
        allowed: list[int] = []

        for span in matches:
            remaining = span[position:]
            if not remaining:
                if vocab.quote not in allowed:
                    allowed.append(vocab.quote)
                continue
            token_id = vocab.token_sequence_for(remaining)[0]
            if token_id not in allowed:
                allowed.append(token_id)

        return allowed or [vocab.quote]

    def _spans(self) -> list[str]:
        """Passages of the request a value may be copied from: quoted
        passages when the request has any, individual words otherwise.
        Never both -- falling back to words while quotes exist would
        let a stray word hijack a value meant to be invented, such as
        a regex.
        """
        source = self._source_text
        if self._spans_cache[0] == source:
            return self._spans_cache[1]

        spans = self._quoted(source) or self._words(source)
        self._spans_cache = (source, spans)
        return spans

    @staticmethod
    def _quoted(source: str) -> list[str]:
        # Double quotes first: a request quoted with " may itself
        # contain an apostrophe ("I'm"), which splitting on ' would
        # cut in half.
        for delimiter in ('"', "'"):
            parts = source.split(delimiter)
            spans = [parts[i] for i in range(1, len(parts), 2) if parts[i]]
            if spans:
                return spans
        return []

    @staticmethod
    def _words(source: str) -> list[str]:
        words: list[str] = []
        current = ""
        for char in source:
            if char.isalnum():
                current += char
                continue
            if current:
                words.append(current)
            current = ""
        if current:
            words.append(current)
        return words

    def _is_looping(self, generated: list[int]) -> bool:
        """True when the last tokens already appeared in that order."""
        window = self.LOOP_WINDOW
        if len(generated) < 2 * window:
            return False
        tail = tuple(generated[-window:])
        earlier = {
            tuple(generated[i:i + window])
            for i in range(len(generated) - window)
        }
        return tail in earlier

    def _is_stop_signal(self, generated: list[int], chosen_id: int) -> bool:
        return chosen_id in self._vocab.closing_ids

    def _is_complete(self, generated: list[int]) -> bool:
        return False

    def _decode(self, generated: list[int]) -> str:
        return self._model.decode(generated)
