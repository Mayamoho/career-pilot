"""Tests for the single choke point every AI call goes through."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from pydantic import Field

from careerpilot.ai.cassette import CassetteStore
from careerpilot.ai.errors import ContractError, ExhaustedRetriesError
from careerpilot.ai.gateway import SYSTEM_PREAMBLE, AIGateway
from careerpilot.ai.providers import PROVIDERS, GeminiProvider, GroqProvider
from careerpilot.ai.schemas import StructuredOutput
from careerpilot.ai.trace import TraceSink
from careerpilot.config import FALLBACK, live_mode_enabled


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
    with pytest.raises(ContractError, match="record-fixtures"):
        gateway.ask(task="answer", system=SYSTEM, user=USER, schema=Answer)


def test_changing_the_prompt_invalidates_the_recording(
    gateway: AIGateway,
    tmp_cassette: Any,
) -> None:
    key = gateway.cache_key("answer", _full_system(), USER, Answer)
    tmp_cassette.put(key, text='{"text": "grounded", "score": 8}', model="m", task="answer")

    with pytest.raises(ContractError):
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
