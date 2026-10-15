"""The single choke point for every AI call in CareerPilot.

Nothing else in this codebase may open a socket to a model. That rule is what makes
the rest of the guarantees enforceable:

* outputs are always Pydantic-validated, so a malformed answer is a retryable event
  rather than a `KeyError` three layers away;
* a bad answer costs at most two repairs, then raises so the caller escalates to a
  human instead of quietly shipping garbage;
* transport failures move down the fallback chain, contract failures go back to the
  model with the validator's exact complaint;
* every attempt is traced with tokens and cost, live or replayed.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import TYPE_CHECKING

from pydantic import ValidationError

from careerpilot.ai.errors import (
    ContractError,
    ExhaustedRetriesError,
    NoCredentialError,
    NoRecordingError,
    TransportError,
)
from careerpilot.ai.providers import CREDENTIAL_ENV, credential_present
from careerpilot.ai.schemas import OutputT, StructuredOutput
from careerpilot.ai.trace import CallRecord, Timer, TraceSink, default_sink
from careerpilot.config import (
    MAX_REPAIR_ATTEMPTS,
    MODEL_CHAIN,
    NETWORK_ATTEMPTS,
    NETWORK_BACKOFF_S,
    RATE_LIMIT_WAIT_S,
    REQUEST_TIMEOUT_S,
    live_mode_enabled,
)

if TYPE_CHECKING:
    import httpx

    from careerpilot.ai.cassette import CassetteStore
    from careerpilot.ai.providers import Provider, ProviderReply
    from careerpilot.config import ModelSpec

SYSTEM_PREAMBLE = """You are a structured data extraction engine, not a chat assistant.

Rules, in priority order:
1. Reply with exactly one JSON object and no other text, prose or markdown fences.
2. The object must match the JSON Schema below exactly.
3. Never invent a value that is not supported by the input text. Absence is a valid
   answer: use an empty list, `null`, or the lowest permitted value.
4. Do not add properties the schema does not declare.
5. Identifiers you reference (atom ids, job ids) must be copied verbatim from the
   input. Never paraphrase an identifier and never construct one.

JSON SCHEMA:
"""


class AIGateway:
    def __init__(
        self,
        *,
        providers: dict[str, Provider],
        cassette: CassetteStore | None = None,
        sink: TraceSink | None = None,
        timeout: float = REQUEST_TIMEOUT_S,
        network_attempts: int = NETWORK_ATTEMPTS,
        network_backoff_s: float = NETWORK_BACKOFF_S,
        rate_limit_wait_s: float = RATE_LIMIT_WAIT_S,
    ) -> None:
        self._providers = providers
        self._cassette = cassette
        self._sink = sink or default_sink
        self._timeout = timeout
        self._network_attempts = network_attempts
        self._network_backoff_s = network_backoff_s
        self._rate_limit_wait_s = rate_limit_wait_s
        self._http: httpx.Client | None = None

    @property
    def sink(self) -> TraceSink:
        return self._sink

    @staticmethod
    def cache_key(task: str, system: str, user: str, schema: type[StructuredOutput]) -> str:
        blob = "\x1f".join([task, system, user, schema.__name__, schema.schema_text()])
        return hashlib.sha256(blob.encode()).hexdigest()[:24]

    def ask(
        self,
        *,
        task: str,
        system: str,
        user: str,
        schema: type[OutputT],
    ) -> OutputT:
        full_system = f"{SYSTEM_PREAMBLE}{schema.schema_text()}\n\n{system}"
        key = self.cache_key(task, full_system, user, schema)

        if not live_mode_enabled():
            replayed = self._replay(task=task, key=key, schema=schema)
            if replayed is not None:
                return replayed
            raise NoRecordingError(task)

        return self._call_live(task=task, system=full_system, user=user, schema=schema, key=key)

    def _replay(self, *, task: str, key: str, schema: type[OutputT]) -> OutputT | None:
        if self._cassette is None:
            return None
        entry = self._cassette.get(key)
        if entry is None:
            return None
        with Timer() as timer:
            try:
                result = schema.model_validate(json.loads(entry["text"]))
            except (ValidationError, json.JSONDecodeError) as exc:
                self._trace(
                    CallRecord(
                        task=task, provider="cassette", model=entry.get("model", "recorded"),
                        attempt=0, duration_ms=timer.elapsed_ms, ok=False,
                        error_kind=type(exc).__name__, input_hash=key, source="cassette",
                    )
                )
                raise ContractError(task, f"recorded fixture no longer validates: {exc}") from exc
        self._trace(
            CallRecord(
                task=task, provider="cassette", model=entry.get("model", "recorded"),
                attempt=0, duration_ms=timer.elapsed_ms, ok=True, input_hash=key, source="cassette",
            )
        )
        return result

    def _generate(
        self,
        provider: Provider,
        *,
        system: str,
        messages: list[dict[str, str]],
        schema: type[OutputT],
        spec: ModelSpec,
    ) -> ProviderReply:
        """Ask one model, retrying only a wire that broke or a limit that will pass.

        A reset connection says nothing about the prompt or the model, so it gets its
        own budget instead of spending a repair or burning down the fallback chain. A
        rate limit says the same thing about the quota — it says *come back in a
        minute* — so it waits rather than moving on, because the next provider would be
        asked to do the same work with the same tokens. An error that is neither is a
        verdict and is left alone: that is the case the fallback chain exists for. A
        timeout is left alone too, since the generation may still be running upstream
        and re-sending it would pay for it twice.
        """
        import httpx  # noqa: PLC0415 - matched against exception types at call time

        broken_wire = (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError)
        hop = 0
        while True:
            try:
                return provider.generate(
                    system=system,
                    messages=messages,
                    schema=schema,
                    model=spec,
                    client=self._client(),
                )
            except TransportError as exc:
                if not exc.retryable:
                    raise
                hop += 1
                if hop >= self._network_attempts:
                    raise
                time.sleep(self._rate_limit_wait_s)
            except broken_wire as exc:
                if isinstance(exc, httpx.TimeoutException):
                    raise
                hop += 1
                if hop >= self._network_attempts:
                    raise
                time.sleep(self._network_backoff_s * 2**hop)

    def _call_live(
        self,
        *,
        task: str,
        system: str,
        user: str,
        schema: type[OutputT],
        key: str,
    ) -> OutputT:
        causes: list[str] = []
        total_attempts = 0
        tried = 0

        for spec in MODEL_CHAIN:
            provider = self._providers.get(spec.provider)
            if provider is None:
                causes.append(f"{spec.provider}: no adapter")
                continue
            if not credential_present(spec.provider):
                continue  # no key for this hop: skip it, do not spend it as a failure
            tried += 1
            messages: list[dict[str, str]] = [{"role": "user", "content": user}]

            for attempt in range(MAX_REPAIR_ATTEMPTS + 1):
                total_attempts += 1
                with Timer() as timer:
                    try:
                        reply = self._generate(
                            provider, system=system, messages=messages, schema=schema, spec=spec
                        )
                    except TransportError as exc:
                        self._trace(
                            CallRecord(
                                task=task,
                                provider=spec.provider,
                                model=spec.model,
                                attempt=attempt,
                                duration_ms=timer.elapsed_ms, ok=False,
                                error_kind="TransportError", input_hash=key,
                            )
                        )
                        causes.append(f"{spec.provider}: {exc}")
                        break  # transport is not fixed by re-asking the same model
                    except Exception as exc:
                        self._trace(
                            CallRecord(
                                task=task,
                                provider=spec.provider,
                                model=spec.model,
                                attempt=attempt,
                                duration_ms=timer.elapsed_ms, ok=False,
                                error_kind=type(exc).__name__, input_hash=key,
                            )
                        )
                        causes.append(f"{spec.provider}: {type(exc).__name__}: {exc}")
                        break

                try:
                    parsed = reply.parse(schema)
                except ValidationError as exc:
                    self._trace(
                        CallRecord(
                            task=task, provider=spec.provider, model=spec.model, attempt=attempt,
                            duration_ms=timer.elapsed_ms, ok=False, error_kind="ContractError",
                            prompt_tokens=reply.prompt_tokens,
                            completion_tokens=reply.completion_tokens,
                            cost_usd=spec.cost_usd(reply.prompt_tokens, reply.completion_tokens),
                            input_hash=key,
                        )
                    )
                    causes.append(
                        f"{spec.provider} attempt {attempt}: "
                        f"{exc.error_count()} validation errors"
                    )
                    messages = [
                        *messages,
                        {"role": "assistant", "content": reply.text[:2000]},
                        {"role": "user", "content": self._repair_prompt(exc)},
                    ]
                    continue
                except Exception as exc:
                    causes.append(f"{spec.provider}: unparseable output ({type(exc).__name__})")
                    continue

                self._trace(
                    CallRecord(
                        task=task, provider=spec.provider, model=spec.model, attempt=attempt,
                        duration_ms=timer.elapsed_ms, ok=True,
                        prompt_tokens=reply.prompt_tokens,
                        completion_tokens=reply.completion_tokens,
                        cost_usd=spec.cost_usd(reply.prompt_tokens, reply.completion_tokens),
                        input_hash=key,
                    )
                )
                if self._cassette is not None:
                    self._cassette.put(key, text=reply.text, model=spec.model, task=task)
                return parsed

        if tried == 0:
            configured = ", ".join(sorted(set(CREDENTIAL_ENV.values())))
            raise NoCredentialError(configured)
        raise ExhaustedRetriesError(task, total_attempts, causes)

    @staticmethod
    def _repair_prompt(exc: ValidationError) -> str:
        errors = [
            {
                "loc": list(err["loc"]),
                "msg": err["msg"],
                "type": err["type"],
            }
            for err in exc.errors(include_url=False, include_context=False)
        ]
        return (
            "Your previous reply was rejected by the validator. Fix exactly these "
            "problems and reply with only the corrected JSON object:\n"
            f"{json.dumps(errors, indent=2, default=str)}"
        )

    def _client(self) -> httpx.Client:
        """One connection, held open and reused for every call in this process.

        This is not a pooling optimization. Measured on the link this tool is built for,
        a fresh TLS handshake per call was reset roughly half the time, while eight
        requests pushed down one already-open connection all landed. Handshaking per
        call was paying the network's price for a connection that was never needed.
        """
        import httpx  # noqa: PLC0415 - resolved at call time so tests can substitute it

        if self._http is None:
            self._http = httpx.Client(timeout=self._timeout)
        return self._http

    def _trace(self, record: CallRecord) -> None:
        self._sink.record(record)


_gateway: AIGateway | None = None


def get_gateway() -> AIGateway:
    global _gateway  # noqa: PLW0603 - process-wide singleton by design
    if _gateway is None:
        from careerpilot.ai.cassette import CassetteStore  # noqa: PLC0415
        from careerpilot.ai.providers import PROVIDERS  # noqa: PLC0415 - built on first use

        _gateway = AIGateway(providers=PROVIDERS, cassette=CassetteStore())
    return _gateway


def set_gateway(gateway: AIGateway | None) -> None:
    global _gateway  # noqa: PLW0603 - tests swap the singleton, nothing else does
    _gateway = gateway


def ask(*, task: str, system: str, user: str, schema: type[OutputT]) -> OutputT:
    """Module-level convenience so call sites stay one line."""
    return get_gateway().ask(task=task, system=system, user=user, schema=schema)
