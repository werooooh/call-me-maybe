from typing import Any

from llm_sdk import Small_LLM_Model

from src.generator_base import ValueGenerator
from src.generator_flag import FlagGenerator
from src.generator_numeric import NumericGenerator
from src.generator_textual import TextualGenerator
from src.schemas import FunctionDefinition, ParameterType
from src.tokens import TokenVocabulary


class ParametersBuilder:
    """Writes the parameters object of the call.

    Keys, braces, commas and the opening quote of a string are written
    by this class: they are imposed by the schema, so the model is
    never asked about them. Only the values are generated, each one by
    the generator of its declared type.
    """

    def __init__(
        self, model: Small_LLM_Model, vocab: TokenVocabulary
    ) -> None:
        self._vocab = vocab
        self._generators: dict[ParameterType, ValueGenerator] = {
            ParameterType.NUMBER: NumericGenerator(model, vocab),
            ParameterType.STRING: TextualGenerator(model, vocab),
            ParameterType.BOOLEAN: FlagGenerator(model, vocab),
        }
        self.last_token_count = 0

    def build(
        self,
        context_ids: list[int],
        function: FunctionDefinition,
        source_text: str,
    ) -> dict[str, Any]:
        """Return every parameter of `function` with a valid value."""
        vocab = self._vocab
        written: list[int] = []
        values: dict[str, Any] = {}
        self.last_token_count = 0

        pending = ""
        for index, (name, schema) in enumerate(function.parameters.items()):
            generator = self._generators[schema.type]
            opening = "{" if index == 0 else ", "
            key = f'{pending}{opening}"{name}": {generator.PREFIX}'
            written.extend(vocab.token_sequence_for(key))

            value_ids, value = generator.generate(
                context_ids + written, source_text
            )
            written.extend(value_ids)
            values[name] = value
            pending = generator.SUFFIX
            self.last_token_count += len(value_ids)

        return values
