import argparse
import sys
import time
from pathlib import Path

from llm_sdk import Small_LLM_Model

from src.files import (
    InputFileError,
    OutputFileError,
    load_function_definitions,
    load_test_prompts,
    write_results,
)
from src.orchestrator import FunctionCaller
from src.schemas import FunctionCallResult, TestPrompt
from src.tokens import TokenVocabulary


def parse_args() -> argparse.Namespace:
    """Read the optional input and output paths."""
    parser = argparse.ArgumentParser(
        description="Function calling via constrained decoding."
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=Path("data/input/functions_definition.json"),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/input/function_calling_tests.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/function_calling_results.json"),
    )
    return parser.parse_args()


def run(
    caller: FunctionCaller, prompts: list[TestPrompt]
) -> list[FunctionCallResult]:
    """Process every request, reporting time and tokens as it goes."""
    results: list[FunctionCallResult] = []
    total_start = time.perf_counter()

    for test_prompt in prompts:
        start = time.perf_counter()
        try:
            result = caller.process(test_prompt)
        except KeyboardInterrupt:
            print("\nInterrupted.", file=sys.stderr)
            break
        except (RuntimeError, ValueError) as exc:
            print(f"Skipped {test_prompt.prompt!r}: {exc}", file=sys.stderr)
            continue
        elapsed = time.perf_counter() - start

        print(
            f"[{elapsed:5.1f}s] "
            f"name={caller.last_name_tokens:2d} tok  "
            f"values={caller.last_value_tokens:2d} tok  "
            f"prompt={test_prompt.prompt!r}"
        )
        if result is not None:
            results.append(result)

    total = time.perf_counter() - total_start
    print(f"\nTotal: {total:.1f}s for {len(prompts)} prompts")
    return results


def main() -> int:
    """Entry point: load, process, write."""
    args = parse_args()

    try:
        functions = load_function_definitions(args.functions_definition)
        prompts = load_test_prompts(args.input)
    except InputFileError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not functions:
        print("Error: no usable function definition", file=sys.stderr)
        return 1
    if not prompts:
        print("Error: no usable prompt", file=sys.stderr)
        return 1

    try:
        model = Small_LLM_Model()
        vocab = TokenVocabulary(model)
    except Exception as exc:
        print(f"Error: could not load the model: {exc}", file=sys.stderr)
        return 1

    caller = FunctionCaller(model, vocab, functions)
    results = run(caller, prompts)

    try:
        write_results(args.output, results)
    except OutputFileError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Processed {len(results)}/{len(prompts)} -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
