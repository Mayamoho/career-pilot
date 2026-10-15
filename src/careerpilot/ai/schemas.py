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
        """The schema as a provider's structured-output compiler wants it.

        Pydantic leaves defaulted fields out of `required`, which is correct JSON Schema
        and rejected outright by strict response_format: Groq answers HTTP 400 "`required`
        is required to be supplied and to be an array including every key in properties".
        Nothing here relaxes validation — every field still has to arrive, and the
        defaults in the model definitions are what make an omission survivable.
        """
        schema = cls.model_json_schema()
        schema.pop("title", None)
        _make_every_key_required(schema)
        return schema

    @classmethod
    def provider_schema(cls) -> dict[str, Any]:
        """The schema sent as a strict `response_format`.

        A strict compiler enforces what it is given by *discarding the answer*: a reply
        whose requirement text is three characters instead of four comes back as HTTP 400
        with the generation thrown away, and the gateway is left with a transport failure
        instead of the text it was supposed to send back to the model for repair. Bounds
        are exactly the constraints models brush past, so they stay in the prompt and in
        our own validation — where a violation costs one repair — and are lifted here.
        Shape survives: types, required keys, enums, patterns, no extra properties.
        """
        schema = cls.json_schema()
        _drop_bounds(schema)
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

# Value-size constraints. Everything else in a JSON Schema — type, required keys, enum,
# pattern, item shape, no-extra-properties — describes the *shape* of the answer, and a
# strict compiler is a good enforcement of shape. These describe how big an individual
# value may be, which is where a model brushes the line and the compiler's response is
# to discard the whole reply.
_BOUNDS = (
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minItems",
    "maxItems",
)


def _drop_bounds(node: Any) -> None:
    """Strip size constraints from a schema in place, keeping its shape.

    Only the bounds are removed, because the bound is the part a strict compiler
    enforces by discarding the reply. A `min_length` on a requirement is a judgement
    about the data, not a structural fact about the answer, and belongs with the
    validator that can ask for a repair instead.
    """
    if isinstance(node, dict):
        for key in _BOUNDS:
            node.pop(key, None)
        for value in node.values():
            _drop_bounds(value)
    elif isinstance(node, list):
        for item in node:
            _drop_bounds(item)


def _make_every_key_required(node: Any) -> None:
    """Rewrite a generated schema in place so a strict response_format accepts it.

    A schema where a key is optional is a schema where the model may answer without it,
    and strict compilers refuse the whole document rather than infer intent. Every
    object node gets all of its declared properties listed as required, including the
    nested ones under `$defs`, which is where submodels live.
    """
    if isinstance(node, dict):
        properties = node.get("properties")
        if isinstance(properties, dict):
            node["required"] = sorted(properties)
            node["additionalProperties"] = False
        for value in node.values():
            _make_every_key_required(value)
    elif isinstance(node, list):
        for item in node:
            _make_every_key_required(item)
