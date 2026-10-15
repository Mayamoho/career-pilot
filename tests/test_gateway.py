"""Tests for the single choke point every AI call goes through."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from pydantic import Field

from careerpilot import config
from careerpilot.ai.cassette import CassetteStore
from careerpilot.ai.errors import (
    ContractError,
    ExhaustedRetriesError,
    NoCredentialError,
    NoRecordingError,
)
from careerpilot.ai.gateway import SYSTEM_PREAMBLE, AIGateway
from careerpilot.ai.providers import PROVIDERS, GeminiProvider, GroqProvider
from careerpilot.ai.schemas import StructuredOutput
from careerpilot.ai.trace import TraceSink
from careerpilot.config import FALLBACK, live_mode_enabled, require_secret


class Answer(StructuredOutput):
    text: str = Field(min_length=1)
    score: int = Field(ge=0, le=10)


SYSTEM = "Answer with a short verdict."
USER = "How grounded is this CV line?"


def _full_system() -> str:
    return f"{SYSTEM_PREAMBLE}{Answer.schema_text()}\n\n{SYSTEM}"


def _openai_body(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": json.dumps(payload)}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }


def test_replay_from_cassette_without_any_provider(gateway: AIGateway, tmp_cassette: Any) -> None:
    key = gateway.cache_key("answer", _full_system(), USER, Answer)
    tmp_cassette.put(key, text='{"text": "grounded", "score": 8}', model="recorded", task="answer")

    result = gateway.ask(task="answer", system=SYSTEM, user=USER, schema=Answer)

    assert (result.text, result.score) == ("grounded", 8)
    assert gateway.sink.records[-1].source == "cassette"


def test_unrecorded_prompt_is_refused_in_non_live_mode(gateway: AIGateway) -> None:
    """And refused *honestly*: no model was ever asked, so the message may not claim the
    model answered in the wrong shape — that sends the reader to debug a prompt that ran
    in neither case."""
    with pytest.raises(NoRecordingError, match="record-fixtures") as excinfo:
        gateway.ask(task="answer", system=SYSTEM, user=USER, schema=Answer)

    assert "schema validation" not in str(excinfo.value)
    assert "CAREERPILOT_LIVE" in str(excinfo.value)


def test_changing_the_prompt_invalidates_the_recording(
    gateway: AIGateway,
    tmp_cassette: Any,
) -> None:
    key = gateway.cache_key("answer", _full_system(), USER, Answer)
    tmp_cassette.put(key, text='{"text": "grounded", "score": 8}', model="m", task="answer")

    with pytest.raises(NoRecordingError):
        gateway.ask(task="answer", system=SYSTEM + " ", user=USER, schema=Answer)


def test_a_recording_that_no_longer_validates_fails_loudly(
    gateway: AIGateway, tmp_cassette: Any
) -> None:
    key = gateway.cache_key("answer", _full_system(), USER, Answer)
    tmp_cassette.put(key, text='{"text": "grounded", "score": "high"}', model="m", task="answer")

    with pytest.raises(ContractError, match="no longer validates"):
        gateway.ask(task="answer", system=SYSTEM, user=USER, schema=Answer)


@pytest.fixture
def live(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AIGateway:
    monkeypatch.setenv("CAREERPILOT_LIVE", "1")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")
    live_mode_enabled.cache_clear()
    return AIGateway(
        providers=PROVIDERS,
        cassette=CassetteStore(tmp_path / "cassettes"),
        sink=TraceSink(quiet=True),
        # The retry backoff is real wall clock; tests should not wait through it.
        network_backoff_s=0.0,
        rate_limit_wait_s=0.0,
    )


GROQ_URL = GroqProvider.BASE
GEMINI_URL = f"{GeminiProvider.BASE}/{FALLBACK.model}:generateContent"


def test_a_malformed_reply_is_retried_with_the_validators_complaint(live: AIGateway) -> None:
    with respx.mock:
        route = respx.post(GROQ_URL).mock(
            side_effect=[
                httpx.Response(200, json=_openai_body({"text": "x", "score": "high"})),
                httpx.Response(200, json=_openai_body({"text": "x", "score": 4})),
            ]
        )
        result = live.ask(task="answer", system=SYSTEM, user=USER, schema=Answer)

    assert result.score == 4
    assert route.call_count == 2
    repair = json.loads(route.calls[1].request.content)["messages"][-1]["content"]
    assert "rejected by the validator" in repair
    assert "score" in repair


def test_transport_failure_moves_down_the_chain_instead_of_re_asking(live: AIGateway) -> None:
    with respx.mock:
        groq = respx.post(GROQ_URL).mock(return_value=httpx.Response(503, text="unavailable"))
        gemini = respx.post(GEMINI_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "candidates": [
                        {"content": {"parts": [{"text": '{"text": "ok", "score": 2}'}]}}
                    ],
                    "usageMetadata": {"promptTokenCount": 50, "candidatesTokenCount": 10},
                },
            )
        )
        result = live.ask(task="answer", system=SYSTEM, user=USER, schema=Answer)

    assert result.score == 2
    assert groq.call_count == 1
    assert gemini.called
    kinds = [(r.provider, r.ok) for r in live.sink.records]
    assert kinds == [("groq", False), ("gemini", True)]


def test_a_strict_schema_asks_for_every_key_and_keeps_the_bounds_in_the_prompt() -> None:
    """The split the live providers forced.

    Groq refuses a schema with a defaulted key left out of `required`, and refuses an
    answer that breaks `minLength` by throwing it away — HTTP 400, nothing left to send
    back for repair. So the compiler's copy is complete on shape and silent on size, and
    the size limits stay in the prompt text and in our own validation.
    """
    provider = Answer.provider_schema()
    prompt = Answer.schema_text()

    assert provider["required"] == ["score", "text"]
    assert provider["additionalProperties"] is False
    assert "minLength" not in json.dumps(provider) and "maximum" not in json.dumps(provider)
    assert "minLength" in prompt and "maximum" in prompt


def test_a_reset_connection_is_retried_before_the_blame_moves(live: AIGateway) -> None:
    """The failure this exists for: the socket dies before any of the answer arrives.

    Nothing about the prompt or the provider caused it, so re-sending the identical
    request is the correct response — not spending a repair on it, and not asking a
    second provider that would be handed the same broken wire.
    """
    with respx.mock:
        groq = respx.post(GROQ_URL).mock(
            side_effect=[
                httpx.ConnectError("[Errno 104] Connection reset by peer"),
                httpx.RemoteProtocolError("Server disconnected without sending a response."),
                httpx.Response(200, json=_openai_body({"text": "grounded", "score": 7})),
            ]
        )
        gemini = respx.post(GEMINI_URL).mock(
            return_value=httpx.Response(200, json={"candidates": []})
        )
        result = live.ask(task="answer", system=SYSTEM, user=USER, schema=Answer)

    assert result.score == 7
    assert groq.call_count == 3
    assert not gemini.called


def test_a_reset_that_keeps_happening_still_moves_down_the_chain(live: AIGateway) -> None:
    with respx.mock:
        groq = respx.post(GROQ_URL).mock(side_effect=httpx.ConnectError("reset"))
        gemini = respx.post(GEMINI_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "candidates": [
                        {"content": {"parts": [{"text": '{"text": "ok", "score": 3}'}]}}
                    ],
                    "usageMetadata": {"promptTokenCount": 50, "candidatesTokenCount": 10},
                },
            )
        )
        result = live.ask(task="answer", system=SYSTEM, user=USER, schema=Answer)

    assert result.score == 3
    assert groq.call_count == config.NETWORK_ATTEMPTS
    assert gemini.called


def test_a_timeout_is_not_resent_because_the_model_may_still_be_answering(
    live: AIGateway,
) -> None:
    """A reset means nothing was produced; a timeout does not, so re-sending it would
    pay for the same generation twice."""
    with respx.mock:
        groq = respx.post(GROQ_URL).mock(side_effect=httpx.ReadTimeout("timed out"))
        respx.post(GEMINI_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "candidates": [
                        {"content": {"parts": [{"text": '{"text": "ok", "score": 5}'}]}}
                    ],
                    "usageMetadata": {"promptTokenCount": 50, "candidatesTokenCount": 10},
                },
            )
        )
        result = live.ask(task="answer", system=SYSTEM, user=USER, schema=Answer)

    assert result.score == 5
    assert groq.call_count == 1


def test_the_gateway_holds_one_connection_open_across_calls(live: AIGateway) -> None:
    """The resets this network produces happen at the handshake, so handshaking per call
    pays for a connection the pipeline was going to need anyway."""
    assert live._client() is live._client()


def test_a_rate_limit_waits_for_the_window_instead_of_blaming_the_next_model(
    live: AIGateway,
) -> None:
    """Groq's free tier limits tokens per minute, so a 429 is a schedule, not a verdict.

    Falling over to Gemini here would ask a second provider to spend its quota on work
    the first will do in under a minute — and with one credential configured it would
    only turn a temporary limit into a permanent failure.
    """
    with respx.mock:
        groq = respx.post(GROQ_URL).mock(
            side_effect=[
                httpx.Response(429, text="tokens per minute"),
                httpx.Response(200, json=_openai_body({"text": "grounded", "score": 8})),
            ]
        )
        gemini = respx.post(GEMINI_URL).mock(return_value=httpx.Response(200, json={}))
        result = live.ask(task="answer", system=SYSTEM, user=USER, schema=Answer)

    assert result.score == 8
    assert groq.call_count == 2
    assert not gemini.called


def test_a_daily_quota_moves_on_instead_of_waiting_through_it(
    live: AIGateway,
) -> None:
    """A day-sized window is a verdict about today, not a minute to sit through.

    Measured 2026-10-16: `tokens per day (TPD): Limit 200000, Used 197861`. Retrying that
    in place bought 90 silent seconds and the same refusal — the user reads a tool that
    waits as a tool that hung. The chain is the answer, exactly as it is for a 5xx.
    """
    with respx.mock:
        groq = respx.post(GROQ_URL).mock(
            return_value=httpx.Response(
                429, text="Rate limit reached on tokens per day (TPD): Limit 200000"
            )
        )
        gemini = respx.post(GEMINI_URL).mock(
            return_value=httpx.Response(200, json={"candidates": [{"content": {"parts": [
                {"text": '{"text": "grounded", "score": 8}'}]}}]})
        )
        result = live.ask(task="answer", system=SYSTEM, user=USER, schema=Answer)

    assert result.score == 8
    assert groq.call_count == 1, "a daily limit must not be re-asked in place"
    assert gemini.called


def test_a_hop_with_no_key_is_skipped_and_never_mentioned(live: AIGateway) -> None:
    """The chain is four providers deep; most people have one or two keys.

    An unconfigured hop must not appear in the failure list. Listing three
    "no credential" causes under the one provider that actually refused makes the real
    answer unreadable, and it would also mean the tool is reporting a state of the
    user's `.env` as though it were a model's behaviour.
    """
    with respx.mock:
        groq = respx.post(GROQ_URL).mock(return_value=httpx.Response(400, text="bad request"))
        gemini = respx.post(GEMINI_URL).mock(return_value=httpx.Response(400, text="bad request"))

        with pytest.raises(ExhaustedRetriesError) as excinfo:
            live.ask(task="answer", system=SYSTEM, user=USER, schema=Answer)

    assert groq.called and gemini.called
    causes = " ".join(excinfo.value.causes)
    assert "cerebras" not in causes and "openrouter" not in causes
    assert len(excinfo.value.causes) == 2


def test_no_key_at_all_says_so_instead_of_reporting_zero_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live mode on with an empty `.env` is a setup state, not a model that refused."""
    monkeypatch.setenv("CAREERPILOT_LIVE", "1")
    for name in ("GROQ_API_KEY", "GEMINI_API_KEY", "CEREBRAS_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.setenv(name, "")
    live_mode_enabled.cache_clear()
    gateway = AIGateway(
        providers=PROVIDERS, cassette=CassetteStore(tmp_path / "c"), sink=TraceSink(quiet=True)
    )

    with pytest.raises(NoCredentialError) as excinfo, respx.mock:
        gateway.ask(task="answer", system=SYSTEM, user=USER, schema=Answer)

    assert "GROQ_API_KEY" in str(excinfo.value)
    assert "attempt" not in str(excinfo.value).lower()


def test_every_model_failing_raises_exhausted_instead_of_guessing(live: AIGateway) -> None:
    with respx.mock:
        respx.post(GROQ_URL).mock(
            return_value=httpx.Response(200, json=_openai_body({"text": "x", "score": -1}))
        )
        respx.post(GEMINI_URL).mock(return_value=httpx.Response(429, text="quota"))

        with pytest.raises(ExhaustedRetriesError) as excinfo:
            live.ask(task="answer", system=SYSTEM, user=USER, schema=Answer)

    assert excinfo.value.attempts == 4  # 3 repairs on groq, then one transport failure
    assert "Escalate to a human" in str(excinfo.value)


def test_a_successful_live_call_is_recorded_for_the_next_run(
    tmp_cassette: CassetteStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CAREERPILOT_LIVE", "1")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq")
    live_mode_enabled.cache_clear()
    gateway = AIGateway(providers=PROVIDERS, cassette=tmp_cassette, sink=TraceSink(quiet=True))

    with respx.mock:
        respx.post(GROQ_URL).mock(
            return_value=httpx.Response(200, json=_openai_body({"text": "x", "score": 6}))
        )
        result = gateway.ask(task="answer", system=SYSTEM, user=USER, schema=Answer)

    assert result.score == 6
    assert len(tmp_cassette) == 1


def test_a_dotenv_file_in_the_project_directory_supplies_the_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The documented way to set a key on a Windows terminal, where `export` is not a thing."""
    (tmp_path / ".env").write_text("GROQ_API_KEY=gsk_from_the_file\n", encoding="utf-8")
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    config.load_env_file.cache_clear()
    try:
        assert require_secret("GROQ_API_KEY") == "gsk_from_the_file"
    finally:
        # load_dotenv writes straight into os.environ, which monkeypatch does not undo.
        config.load_env_file.cache_clear()
        monkeypatch.delenv("GROQ_API_KEY", raising=False)


def test_a_credential_exported_in_the_shell_outranks_the_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A forgotten .env must not silently win over the key you just exported."""
    (tmp_path / ".env").write_text("GROQ_API_KEY=gsk_from_the_file\n", encoding="utf-8")
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_from_the_shell")
    config.load_env_file.cache_clear()
    try:
        assert require_secret("GROQ_API_KEY") == "gsk_from_the_shell"
    finally:
        config.load_env_file.cache_clear()


def test_an_env_file_cannot_switch_the_suite_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The README invites users to put `CAREERPILOT_LIVE=1` in `.env`. This suite must
    stay deaf to it: a developer who follows those instructions would otherwise be
    billed for a model call by every test, and the results would stop being repeatable.
    """
    (tmp_path / ".env").write_text("CAREERPILOT_LIVE=1\n", encoding="utf-8")
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    config.load_env_file.cache_clear()
    try:
        assert live_mode_enabled() is False
    finally:
        config.load_env_file.cache_clear()
        monkeypatch.delenv("CAREERPILOT_LIVE", raising=False)
