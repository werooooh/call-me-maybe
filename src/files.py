import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.schemas import FunctionCallResult, FunctionDefinition, TestPrompt


class InputFileError(Exception):
    """Raised when an input file is missing, malformed, or invalid."""


class OutputFileError(Exception):
    """Raised when the output file cannot be written."""


def _load_json_array(path: Path) -> list[dict[str, Any]]:
    """Read a JSON array of objects, or raise InputFileError."""
    if not path.exists():
        raise InputFileError(f"File not found: {path}")

    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise InputFileError(f"Could not read {path}: {exc}") from exc

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise InputFileError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(data, list):
        raise InputFileError(f"Expected a JSON array in {path}")

    return [entry for entry in data if isinstance(entry, dict)]


def load_function_definitions(path: Path) -> list[FunctionDefinition]:
    """Load the callable functions, skipping unusable entries.

    An entry is unusable when a field is missing or when a parameter
    declares a type the generators cannot produce.
    """
    definitions: list[FunctionDefinition] = []

    for entry in _load_json_array(path):
        try:
            definitions.append(FunctionDefinition.model_validate(entry))
        except ValidationError:
            continue

    return definitions


def load_test_prompts(path: Path) -> list[TestPrompt]:
    """Load the requests to process, skipping unusable entries."""
    prompts: list[TestPrompt] = []

    for entry in _load_json_array(path):
        try:
            prompts.append(TestPrompt.model_validate(entry))
        except ValidationError:
            continue

    return prompts


def write_results(path: Path, results: list[FunctionCallResult]) -> None:
    """Write the calls as a JSON array, creating the directory."""
    payload = [result.model_dump() for result in results]

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        raise OutputFileError(f"Could not write {path}: {exc}") from exc
