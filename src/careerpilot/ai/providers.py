"""Provider adapters. Raw HTTP on purpose.

Wrapping two vendors behind one `generate()` signature keeps the gateway in control
of retries and makes the test double a drop-in. Token counts come from the providers'
own usage fields where present so cost accounting is not a guess.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

import httpx

from careerpilot.ai.errors import MissingProviderCredential, TransportError
from careerpilot.ai.schemas import OutputT, StructuredOutput
from careerpilot.config import ModelSpec, require_secret

CREDENTIAL_ENV = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY"}

HTTP_ERROR_STATUS = 400
HTTP_SERVER_ERROR = 500
RETRYABLE_STATUSES = frozenset({408, 409, 429})


class Provider(Protocol):
    def generate(
        self,
        *,
        system: str,
        messages: list[dict[str, str]],
        schema: type[OutputT],
        model: ModelSpec,
        client: httpx.Client,
    ) -> ProviderReply: ...


class ProviderReply:
    __slots__ = ("completion_tokens", "prompt_tokens", "text")

    def __init__(self, text: str, prompt_tokens: int, completion_tokens: int) -> None:
        self.text = text
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

    def parse(self, schema: type[OutputT]) -> OutputT:
        return schema.from_llm_json(self.text)


def _credentials(model: ModelSpec) -> str:
    secret = CREDENTIAL_ENV.get(model.provider)
    if secret is None:
        raise TransportError(model.provider, "unknown provider")
    try:
        return require_secret(secret)
    except Exception as exc:
        raise MissingProviderCredential(model.provider, secret) from exc


class GroqProvider:
    """OpenAI-compatible chat completions with JSON-schema response_format."""

    BASE = "https://api.groq.com/openai/v1/chat/completions"

    def generate(
        self,
        *,
        system: str,
        messages: list[dict[str, str]],
        schema: type[StructuredOutput],
        model: ModelSpec,
        client: httpx.Client,
    ) -> ProviderReply:
        key = _credentials(model)
        body = {
            "model": model.model,
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "strict": True,
                    "schema": schema.json_schema(),
                },
            },
            "messages": [{"role": "system", "content": system}, *messages],
        }
        response = client.post(self.BASE, json=body, headers={"Authorization": f"Bearer {key}"})
        return _openai_style_reply(response, model)


class GeminiProvider:
    BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    def generate(
        self,
        *,
        system: str,
        messages: list[dict[str, str]],
        schema: type[StructuredOutput],
        model: ModelSpec,
        client: httpx.Client,
    ) -> ProviderReply:
        key = _credentials(model)
        contents = [
            {"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]}
            for m in messages
        ]
        body: dict[str, Any] = {
            "contents": contents,
            "systemInstruction": {"parts": [{"text": system}]},
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
                "responseSchema": _gemini_schema(schema.json_schema()),
            },
        }
        url = f"{self.BASE}/{model.model}:generateContent"
        response = client.post(url, json=body, params={"key": key})
        if response.status_code >= HTTP_ERROR_STATUS:
            raise TransportError(
                model.provider, f"HTTP {response.status_code}: {response.text[:200]}"
            )
        payload = response.json()
        try:
            text = payload["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise TransportError(model.provider, f"malformed response: {exc}") from exc
        usage = payload.get("usageMetadata", {})
        return ProviderReply(
            text,
            int(usage.get("promptTokenCount", 0)),
            int(usage.get("candidatesTokenCount", 0)),
        )


def _openai_style_reply(response: httpx.Response, model: ModelSpec) -> ProviderReply:
    if response.status_code >= HTTP_ERROR_STATUS:
        raise TransportError(
            model.provider,
            f"HTTP {response.status_code}: {response.text[:200]}",
            retryable=response.status_code in RETRYABLE_STATUSES
            or response.status_code >= HTTP_SERVER_ERROR,
        )
    try:
        payload = response.json()
        text = payload["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise TransportError(model.provider, f"malformed response: {exc}") from exc
    usage = payload.get("usage") or {}
    return ProviderReply(
        text or "",
        int(usage.get("prompt_tokens", 0)),
        int(usage.get("completion_tokens", 0)),
    )


def _gemini_schema(node: dict[str, Any]) -> dict[str, Any]:
    """Trim JSON Schema down to what Gemini's responseSchema accepts."""
    if not isinstance(node, dict):
        return node
    keep: dict[str, Any] = {}
    for key, value in node.items():
        if key in {"$defs", "definitions", "title", "$schema", "additionalProperties"}:
            continue
        if key == "anyOf" and isinstance(value, list):
            non_null = [v for v in value if v != {"type": "null"}]
            if len(non_null) == 1:
                keep["type"] = "STRING" if non_null[0].get("type") == "string" else None
                merged = _gemini_schema(non_null[0])
                keep.update({k: v for k, v in merged.items() if k != "type"})
                keep.setdefault("nullable", True)
                continue
        if key == "type" and isinstance(value, str):
            keep["type"] = {"string": "STRING", "number": "NUMBER", "integer": "INTEGER",
                            "boolean": "BOOLEAN", "object": "OBJECT", "array": "ARRAY"}.get(
                                value.upper(), value.upper())
        elif key in {"properties", "items"} and isinstance(value, dict):
            keep[key] = {k: _gemini_schema(v) for k, v in value.items()}
        elif key == "items" and isinstance(value, list):
            keep[key] = _gemini_schema(value[0])
        elif key in {"required", "enum", "description"}:
            keep[key] = value
    return {k: v for k, v in keep.items() if v is not None}


PROVIDERS: dict[str, Provider] = {"groq": GroqProvider(), "gemini": GeminiProvider()}
