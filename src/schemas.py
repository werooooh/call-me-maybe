from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict


class ParameterType(str, Enum):
    """Primitive JSON types this project knows how to generate.

    A function declaring any other type (array, object, ...) fails
    validation and is dropped from the candidates instead of crashing
    the program or producing a call with a missing argument.
    """

    NUMBER = "number"
    STRING = "string"
    BOOLEAN = "boolean"


class ParameterSchema(BaseModel):
    """Definition of a single function parameter."""

    model_config = ConfigDict(extra="ignore")

    type: ParameterType


class ReturnSchema(BaseModel):
    """Definition of a function's return type (kept for completeness)."""

    model_config = ConfigDict(extra="ignore")

    type: str


class FunctionDefinition(BaseModel):
    """One entry from functions_definition.json."""

    model_config = ConfigDict(extra="ignore")

    name: str
    description: str = ""
    parameters: dict[str, ParameterSchema] = {}
    returns: ReturnSchema | None = None


class TestPrompt(BaseModel):
    """One entry from function_calling_tests.json."""

    model_config = ConfigDict(extra="ignore")

    prompt: str


class FunctionCallResult(BaseModel):
    """One entry written to function_calling_results.json."""

    prompt: str
    name: str
    parameters: dict[str, Any]
