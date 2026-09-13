from llm_sdk import Small_LLM_Model

from src.builder import ParametersBuilder
from src.prompting import PromptBuilder
from src.schemas import FunctionCallResult, FunctionDefinition, TestPrompt
from src.selector import FunctionNameSelector
from src.tokens import TokenVocabulary


class FunctionCaller:
    """Turns one request into a complete function call."""

    def __init__(
        self,
        model: Small_LLM_Model,
        vocab: TokenVocabulary,
        functions: list[FunctionDefinition],
    ) -> None:
        self._model = model
        self._vocab = vocab
        self._functions = functions
        self._prompt_builder = PromptBuilder(functions)
        self._selector = FunctionNameSelector(model, vocab)
        self._builder = ParametersBuilder(model, vocab)

    @property
    def last_name_tokens(self) -> int:
        return self._selector.last_token_count

    @property
    def last_value_tokens(self) -> int:
        return self._builder.last_token_count

    def process(self, test_prompt: TestPrompt) -> FunctionCallResult | None:
        """Return the call for this request, or None if impossible."""
        if not self._functions:
            return None

        prompt_text = self._prompt_builder.build(test_prompt.prompt)
        prompt_ids = self._vocab.token_sequence_for(prompt_text)

        name, name_ids = self._selector.select(prompt_ids, self._functions)
        function = next(
            (fn for fn in self._functions if fn.name == name), None
        )
        if function is None:
            return None

        # The name tokens stay in the context: the values are written
        # right after the name the model has just committed to, which
        # is what makes them consistent with the chosen function.
        parameters = self._builder.build(
            prompt_ids + name_ids, function, test_prompt.prompt
        )

        return FunctionCallResult(
            prompt=test_prompt.prompt, name=name, parameters=parameters
        )
