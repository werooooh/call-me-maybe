from src.schemas import FunctionDefinition


class PromptBuilder:
    """Builds the chat prompt fed to the model for one request."""

    def __init__(self, functions: list[FunctionDefinition]) -> None:
        self._functions_block = self._list_functions(functions)

    def build(self, request: str) -> str:
        """Return the full prompt text for `request`."""
        system = (
            "You are a function-calling assistant. Available functions:\n"
            f"{self._functions_block}\n\n"
            "Rules:\n"
            "- Copy values exactly as written in the request, invent "
            "nothing; when the request quotes a passage, that passage "
            "is the text the function operates on.\n"
            "- Write a symbol as the symbol itself, never as its "
            "name.\n"
            "- Keep a pattern as short as possible: one character "
            "class, no alternation, no group, no repetition.\n"
            "- Answer with the function name then the JSON "
            "parameters, nothing else.\n\n"
            "User: send bob a message that says hi\n"
            'Assistant: fn_demo_send{"recipient": "bob", "text": "hi"}\n'
            "User: replace every comma in 'red, green, blue' with a dash\n"
            'Assistant: fn_demo_replace{"source": "red, green, blue", '
            '"target": ",", "replacement": "-"}\n'
            'User: replace all letters in "a1 b2" with a plus\n'
            'Assistant: fn_demo_replace{"source": "a1 b2", '
            '"target": "[a-z]", "replacement": "+"}'
        )

        # Qwen3 opens a <think> block by default; leaving it empty here
        # tells the model to skip straight to answering.
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{request}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )

    @staticmethod
    def _list_functions(functions: list[FunctionDefinition]) -> str:
        lines = []
        for fn in functions:
            params = ", ".join(
                f"{name}: {schema.type.value}"
                for name, schema in fn.parameters.items()
            )
            lines.append(f"- {fn.name}({params}): {fn.description}")
        return "\n".join(lines)
