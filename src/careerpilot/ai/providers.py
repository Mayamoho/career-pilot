"""Provider adapters. Raw HTTP on purpose.

Wrapping several vendors behind one `generate()` signature keeps the gateway in control
of retries and makes the test double a drop-in. Token counts come from the providers'
own usage fields where present so cost accounting is not a guess.

Every hop here is an OpenAI-compatible `/chat/completions` endpoint except Gemini, which
is its own shape. They are in the chain because each has a free tier that needs no card,
and a candidate with no key configured is skipped before the request rather than counted
as a failure — see `credential_present`.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, ClassVar, Protocol

import httpx

from careerpilot.ai.errors import MissingProviderCredential, TransportError
from careerpilot.ai.schemas import OutputT, StructuredOutput
from careerpilot.config import ModelSpec, load_env_file, require_secret

# One env var per provider. Any of them being unset simply removes that hop.
CREDENTIAL_ENV = {
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
}

HTTP_ERROR_STATUS = 400
RETRYABLE_STATUSES = frozenset({408, 409, 429})

# Not every 429 is a minute. Measured 2026-10-16 on the free tier:
# `Rate limit reached for model openai/gpt-oss-120b … on tokens per day (TPD):
# Limit 200000, Used 197861`. A day-sized window does not close by waiting, so it is
# treated as the verdict it is rather than slept through twice.
_DAILY_WINDOW = re.compile(r"per day|per month|per week|\(TPD\)|\(RPD\)|daily", re.I)


def _http_failure(model: ModelSpec, response: httpx.Response) -> TransportError:
    """Wrap a status code in the one error type that knows what waiting can fix.

    A 429 with a minute-sized window is a schedule: the tokens free up shortly and the
    same request will succeed, so it is retryable in place. A 429 naming a day window is
    a verdict about today, and the fallback chain — not a re-ask — is the answer to it.
    A 5xx is a verdict about that model too. Conflating the two wastes the chain on a
    rate limit and re-asks a dead model; it also buys the user 90 silent seconds.
    """
    body = response.text[:200]
    return TransportError(
        model.provider,
        f"HTTP {response.status_code}: {body}",
        retryable=response.status_code in RETRYABLE_STATUSES and not _DAILY_WINDOW.search(body),
    )


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


def credential_present(provider: str) -> bool:
    """Does this operator have a key for that provider? Checked before the request.

    The chain is deliberately longer than the keys most people have. Asking a provider
    with no credential and recording the refusal as a cause would bury the one failure
    that matters under three identical ones, so an unconfigured hop is skipped silently
    and never reaches the network.
    """
    name = CREDENTIAL_ENV.get(provider)
    if name is None:
        return False
    load_env_file()
    return bool(os.environ.get(name, "").strip())


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


class OpenAICompatibleProvider:
    """Chat completions against any endpoint that speaks the OpenAI JSON body.

    Groq, Cerebras and OpenRouter all expose this exact shape, so one class covers three
    providers and the only thing that differs is the URL. The JSON-schema `response_format`
    is what every hop in the chain declares support for — a model that silently ignores it
    returns prose, which the validator then rejects as a contract failure.
    """

    def __init__(self, base: str) -> None:
        self.base = base

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
                    "schema": schema.provider_schema(),
                },
            },
            "messages": [{"role": "system", "content": system}, *messages],
        }
        response = client.post(self.base, json=body, headers={"Authorization": f"Bearer {key}"})
        return _openai_style_reply(response, model)


class GroqProvider(OpenAICompatibleProvider):
    """OpenAI-compatible chat completions with JSON-schema response_format."""

    BASE: ClassVar[str] = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self) -> None:
        super().__init__(self.BASE)


class CerebrasProvider(OpenAICompatibleProvider):
    """Cerebras inference — same OpenAI-shaped body, its own host."""

    BASE: ClassVar[str] = "https://api.cerebras.ai/v1/chat/completions"

    def __init__(self) -> None:
        super().__init__(self.BASE)


class OpenRouterProvider(OpenAICompatibleProvider):
    """OpenRouter — a router in front of many models, one OpenAI-shaped endpoint."""

    BASE: ClassVar[str] = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self) -> None:
        super().__init__(self.BASE)


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
                "responseSchema": _gemini_schema(schema.provider_schema()),
            },
        }
        url = f"{self.BASE}/{model.model}:generateContent"
        response = client.post(url, json=body, params={"key": key})
        if response.status_code >= HTTP_ERROR_STATUS:
            raise _http_failure(model, response)
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
        raise _http_failure(model, response)
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


_GEMINI_DROP = {"$defs", "definitions", "title", "$schema", "additionalProperties", "$ref"}
_GEMINI_TYPES = {
    "string": "STRING",
    "number": "NUMBER",
    "integer": "INTEGER",
    "boolean": "BOOLEAN",
    "object": "OBJECT",
    "array": "ARRAY",
}


def _deref(node: dict[str, Any], root: dict[str, Any]) -> dict[str, Any] | None:
    """The definition a `$ref` points at, if this node is a reference we can resolve."""
    ref = node.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/$defs/"):
        return None
    target = (root.get("$defs") or {}).get(ref.rsplit("/", 1)[-1])
    return target if isinstance(target, dict) else None


def _gemini_schema(node: Any, root: dict[str, Any] | None = None) -> dict[str, Any]:
    """Trim JSON Schema down to what Gemini's responseSchema accepts.

    Two shapes that pydantic generates and Gemini refuses, both seen in a live 400 on
    2026-10-16: `$ref`/`$defs`, which Gemini has no concept of, and the difference between
    `properties` (a map of schemas) and `items` (one schema). Treating `items` as a map
    passed a requirement submodel's `{"$ref": …}` through untouched, and the answer came
    back as `Unknown name "$ref" at '…response_schema.properties[0].value.items'` with the
    generation discarded — which is why the second hop of the chain never actually worked.
    """
    if root is None:
        root = node
    if not isinstance(node, dict):
        # A boolean schema node means "any value", which Gemini spells as an empty schema.
        return {}
    referenced = _deref(node, root)
    if referenced is not None:
        return _gemini_schema(referenced, root)
    keep: dict[str, Any] = {}
    for key, value in node.items():
        if key in _GEMINI_DROP:
            continue
        if key == "anyOf" and isinstance(value, list):
            non_null = [v for v in value if v != {"type": "null"}]
            if len(non_null) == 1:
                keep["type"] = None
                keep.update(_gemini_schema(non_null[0], root))
                keep["nullable"] = True
                continue
        if key == "type" and isinstance(value, str):
            keep["type"] = _GEMINI_TYPES.get(value.upper(), value.upper())
        elif key == "properties" and isinstance(value, dict):
            keep[key] = {k: _gemini_schema(v, root) for k, v in value.items()}
        elif key == "items":
            one = value[0] if isinstance(value, list) else value
            keep[key] = _gemini_schema(one, root)
        elif key in {"required", "enum", "description"}:
            keep[key] = value
    return {k: v for k, v in keep.items() if v is not None}


PROVIDERS: dict[str, Provider] = {
    "groq": GroqProvider(),
    "gemini": GeminiProvider(),
    "cerebras": CerebrasProvider(),
    "openrouter": OpenRouterProvider(),
}
