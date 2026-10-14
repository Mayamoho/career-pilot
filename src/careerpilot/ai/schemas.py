"""Structured-output base class.

Every AI response in this codebase is an instance of a model derived from
`StructuredOutput`. That is not ceremony: it is what lets the gateway retry on a
malformed answer, and what lets us assert in tests that no call site is free-typing.
"""

from __future__ import annotations

import json
from typing import Any, Self, TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError


class StructuredOutput(BaseModel):
    """Strict, extra-forbidding base. Unknown keys are a contract violation."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        schema = cls.model_json_schema()
        schema.pop("title", None)
        return schema

    @classmethod
    def schema_text(cls) -> str:
        return json.dumps(cls.json_schema(), indent=2, sort_keys=True)

    @classmethod
    def from_llm_json(cls, raw: str) -> Self:
        """Parse a model reply that may be fenced, padded or newline-wrapped."""
        payload = extract_json_object(raw)
        if payload is None:
            raise ValidationError.from_exception_data(
                cls.__name__,
                line_errors=[
                    {
                        "type": "json_invalid",
                        "loc": (),
                        "input": raw,
                        "ctx": {"error": "no JSON object found in model output"},
                    }
                ],
            )
        return cls.model_validate(payload)


def extract_json_object(raw: str) -> dict[str, Any] | None:
    """Best-effort recovery of the single JSON object inside a chatty reply."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


OutputT = TypeVar("OutputT", bound=StructuredOutput)
