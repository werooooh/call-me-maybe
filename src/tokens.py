import json

import numpy as np
import numpy.typing as npt

from llm_sdk import Small_LLM_Model

SPACE_MARKER = "Ġ"
SPECIAL_MARKER = "<|"


class TokenVocabulary:
    """Token ids the constrained decoder works with.

    Every id is read from the vocabulary file of the model, so nothing
    is hardcoded for one particular tokenizer.
    """

    def __init__(self, model: Small_LLM_Model) -> None:
        self._model = model
        self._token_to_id = self._read_vocab(model)
        self._encoded: dict[str, list[int]] = {}

        self.lbrace = self._require("{")
        self.rbrace = self._require("}")
        self.quote = self._require('"')
        self.colon = self._require(":")
        self.comma = self._require(",")
        self.minus = self._require("-")
        self.dot = self._require(".")
        self.digits = [self._require(str(d)) for d in range(10)]
        self.true_id = self._require("true")
        self.false_id = self._require("false")
        self.string_tokens = self._safe_string_tokens()
        self.closing_tokens = self._closing_tokens()
        self.closing_ids = frozenset(self.closing_tokens)
        self.string_or_closing = self.string_tokens + self.closing_tokens

    @staticmethod
    def _read_vocab(model: Small_LLM_Model) -> dict[str, int]:
        try:
            path = model.get_path_to_vocab_file()
            with open(path, encoding="utf-8") as vocab_file:
                data: dict[str, int] = json.load(vocab_file)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Vocabulary unavailable: {exc}") from exc
        return data

    def _require(self, text: str) -> int:
        try:
            return self._token_to_id[text]
        except KeyError as exc:
            raise RuntimeError(
                f"Expected token {text!r} not found in vocab"
            ) from exc

    def _safe_string_tokens(self) -> list[int]:
        """Ids of every token that may appear inside a JSON string.

        A token qualifies when its text is printable ASCII (the byte
        level marker of a space included) and holds neither a double
        quote nor a backslash. Writing any of them can therefore never
        close the string early nor open an invalid escape sequence,
        whatever the model chooses.
        """
        safe = {chr(code) for code in range(32, 127)} - set('"\\')
        safe.add(SPACE_MARKER)
        return sorted(
            token_id
            for token, token_id in self._token_to_id.items()
            if token and SPECIAL_MARKER not in token and set(token) <= safe
        )

    def _closing_tokens(self) -> list[int]:
        """Ids of every token that closes a JSON string.

        There is no "end of string" token: the model ends a value by
        writing the closing quote, and in real JSON that quote comes
        glued to what follows it -- `",` or `"}` are single tokens.
        Offering only the lone quote would hide the natural way out
        and push the model to keep writing instead.
        """
        followers = set(",}]:") | {SPACE_MARKER}
        return sorted(
            token_id
            for token, token_id in self._token_to_id.items()
            if token.startswith('"') and set(token[1:]) <= followers
        )

    def token_sequence_for(self, text: str) -> list[int]:
        """Exact ids the real tokenizer produces for `text`.

        Needed for anything written without asking the model: function
        names, parameter keys, copied passages.
        """
        cached = self._encoded.get(text)
        if cached is None:
            cached = [int(i) for i in self._model.encode(text)[0].tolist()]
            self._encoded[text] = cached
        return cached


def build_mask(
    allowed_ids: list[int], vocab_size: int
) -> npt.NDArray[np.float32]:
    """Bias vector: 0.0 on the allowed ids, -inf everywhere else.

    Added to the raw logits, it removes every token that would break
    the structure before the best one is picked.
    """
    mask = np.full(vocab_size, -np.inf, dtype=np.float32)
    valid = np.asarray(
        [i for i in allowed_ids if 0 <= i < vocab_size], dtype=np.int64
    )
    mask[valid] = 0.0
    return mask


def pick_best(logits: list[float], allowed_ids: list[int]) -> int:
    """Id of the best token once the constraint mask is applied."""
    scores = np.asarray(logits, dtype=np.float32)
    scores = scores + build_mask(allowed_ids, int(scores.size))
    return int(np.argmax(scores))
