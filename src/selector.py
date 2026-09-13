from llm_sdk import Small_LLM_Model

from src.schemas import FunctionDefinition
from src.tokens import TokenVocabulary, pick_best


class FunctionNameSelector:
    """Picks the function name, constrained to the declared names.

    The names are walked token by token: at each step only the tokens
    that continue at least one candidate are allowed, so the model can
    only ever spell an existing name. When a single token continues
    the remaining candidates, it is written without querying the
    model.
    """

    def __init__(
        self, model: Small_LLM_Model, vocab: TokenVocabulary
    ) -> None:
        self._model = model
        self._vocab = vocab
        self.last_token_count = 0

    def select(
        self, prompt_ids: list[int], candidates: list[FunctionDefinition]
    ) -> tuple[str, list[int]]:
        """Return the chosen name and the tokens that spell it."""
        remaining = [
            (fn.name, self._vocab.token_sequence_for(fn.name))
            for fn in candidates
        ]
        generated: list[int] = []

        while remaining:
            step = len(generated)
            allowed_ids: list[int] = []
            for _, seq in remaining:
                if step < len(seq) and seq[step] not in allowed_ids:
                    allowed_ids.append(seq[step])

            if not allowed_ids:
                break

            chosen_id = self._choose(prompt_ids + generated, allowed_ids)
            generated.append(chosen_id)
            remaining = [
                (name, seq)
                for name, seq in remaining
                if step < len(seq) and seq[step] == chosen_id
            ]

        self.last_token_count = len(generated)
        return (remaining[0][0] if remaining else ""), generated

    def _choose(self, input_ids: list[int], allowed_ids: list[int]) -> int:
        if len(allowed_ids) == 1:
            return allowed_ids[0]
        logits = self._model.get_logits_from_input_ids(input_ids)
        return pick_best(logits, allowed_ids)
