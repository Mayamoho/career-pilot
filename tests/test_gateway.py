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
from careerpilot.ai.providers import (
    CREDENTIAL_ENV,
    PROVIDERS,
    CohereProvider,
    GeminiProvider,
    GroqProvider,
    _gemini_schema,
    _http_failure,
)
from careerpilot.ai.schemas import StructuredOutput
from careerpilot.ai.trace import TraceSink
from careerpilot.cli import _KEY_SOURCE
from careerpilot.config import (
    FALLBACK,
    MAX_REPAIR_ATTEMPTS,
    MODEL_CHAIN,
    live_mode_enabled,
    require_secret,
)
from careerpilot.ground.gate import Claim, ClaimList, EntailmentVerdict
from careerpilot.match.extract import ExtractedRequirement, RequirementSet
from careerpilot.match.score import EvidenceCitation
from careerpilot.tailor.bullets import BulletDraft, DraftedBullet
from careerpilot.tailor.letter import DraftedParagraph, LetterDraft


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
# Both Gemini hops in the chain, matched on host rather than on the pinned id, so a
# re-pinning in config.py cannot leave a second model with no registered mock.
GEMINI_CALL = rf"{GeminiProvider.BASE.replace('.', '[.]')}/.*:generateContent"


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
    """The chain is eight hops deep; most people have one or two keys.

    An unconfigured hop must not appear in the failure list. Listing three
    "no credential" causes under the one provider that actually refused makes the real
    answer unreadable, and it would also mean the tool is reporting a state of the
    user's `.env` as though it were a model's behaviour.

    Every *configured* hop is expected to speak here, so Groq answers twice — the chain
    gives it two models, and a spent day on one of them is not a spent day on the other.
    """
    with respx.mock:
        groq = respx.post(GROQ_URL).mock(return_value=httpx.Response(400, text="bad request"))
        gemini = respx.post(url__regex=GEMINI_CALL).mock(
            return_value=httpx.Response(400, text="bad request")
        )

        with pytest.raises(ExhaustedRetriesError) as excinfo:
            live.ask(task="answer", system=SYSTEM, user=USER, schema=Answer)

    assert groq.called and gemini.called
    causes = " ".join(excinfo.value.causes)
    for provider in set(CREDENTIAL_ENV) - {"groq", "gemini"}:
        assert provider not in causes
    assert {c.split(":", 1)[0] for c in excinfo.value.causes} == {"groq", "gemini"}


def test_every_hop_in_the_chain_has_an_adapter_a_variable_and_a_key_hint() -> None:
    """`config.py` can add a hop; the three tables that answer it must not be forgotten.

    A provider in MODEL_CHAIN with no adapter is reported as `groq: no adapter` inside a
    causes list — a missing dict entry wearing the clothes of a model refusal. `keys`
    would raise a KeyError instead, which is at least honest, but both are a crash or a
    lie on the command whose whole job is to tell the user what to do about quota.
    """
    chain = {spec.provider for spec in MODEL_CHAIN}
    assert chain <= set(PROVIDERS), "add the adapter before the hop"
    assert chain <= set(CREDENTIAL_ENV), "every hop needs a variable to be skipped on"
    assert chain <= set(_KEY_SOURCE), "`keys` needs to say where each free key comes from"


def test_no_key_at_all_says_so_instead_of_reporting_zero_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live mode on with an empty `.env` is a setup state, not a model that refused."""
    monkeypatch.setenv("CAREERPILOT_LIVE", "1")
    for name in CREDENTIAL_ENV.values():
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
        respx.post(url__regex=GEMINI_CALL).mock(
            return_value=httpx.Response(429, text=GEMINI_QUOTA)
        )

        with pytest.raises(ExhaustedRetriesError) as excinfo:
            live.ask(task="answer", system=SYSTEM, user=USER, schema=Answer)

    groq_hops = sum(1 for spec in MODEL_CHAIN if spec.provider == "groq")
    gemini_hops = sum(1 for spec in MODEL_CHAIN if spec.provider == "gemini")
    # Every Groq model spends its whole repair budget; every Gemini model is refused once,
    # because a day-sized quota is a verdict and re-asking the same model cannot fix it.
    assert excinfo.value.attempts == groq_hops * (MAX_REPAIR_ATTEMPTS + 1) + gemini_hops
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


CONTRACTS: list[Any] = [
    RequirementSet,
    ExtractedRequirement,
    EvidenceCitation,
    BulletDraft,
    DraftedBullet,
    LetterDraft,
    DraftedParagraph,
    ClaimList,
    Claim,
    EntailmentVerdict,
]


@pytest.mark.parametrize("contract", CONTRACTS, ids=lambda c: c.__name__)
def test_no_contract_sends_gemini_a_reference_it_cannot_resolve(
    contract: type[StructuredOutput],
) -> None:
    """Every task's schema, translated, must be self-contained.

    Gemini's `responseSchema` has no `$ref`, and pydantic puts one wherever a field holds a
    submodel — which is most of the array fields in this codebase. Measured live on
    2026-10-16: `extract_requirements` died with HTTP 400 `Unknown name "$ref"`, so the
    second hop of the chain had never worked for the tasks that needed it. A test over the
    real contracts is the only thing that catches the next submodel someone adds.
    """
    body = json.dumps(_gemini_schema(contract.provider_schema()))

    assert "$ref" not in body, f"{contract.__name__} still carries an unresolved reference"
    assert "$defs" not in body


def test_the_gemini_request_inlines_a_submodel_array(live: AIGateway) -> None:
    with respx.mock:
        respx.post(GROQ_URL).mock(
            return_value=httpx.Response(
                429, text="Rate limit reached on tokens per day (TPD): Limit 200000"
            )
        )
        gemini = respx.post(GEMINI_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "text": json.dumps(
                                            {
                                                "requirements": [
                                                    {
                                                        "text": "PHP required",
                                                        "skill": "php",
                                                        "category": "must_have",
                                                    }
                                                ],
                                                "seniority": "intern",
                                                "location_constraint": "bangladesh",
                                            }
                                        )
                                    }
                                ]
                            }
                        }
                    ]
                },
            )
        )
        result = live.ask(
            task="extract_requirements",
            system=SYSTEM,
            user=USER,
            schema=RequirementSet,
        )

    sent = json.loads(gemini.calls[0].request.content)
    schema = sent["generationConfig"]["responseSchema"]
    assert "$ref" not in json.dumps(schema)
    assert schema["properties"]["requirements"]["items"]["type"] == "OBJECT"
    assert result.requirements[0].skill == "php"


GEMINI_QUOTA = (
    '{"error":{"code":429,"message":"You exceeded your current quota, please check '
    'your plan and billing details.","status":"RESOURCE_EXHAUSTED"}}'
)

OPENROUTER_DAY = (
    '{"error":{"message":"Rate limit exceeded: Free Usage Limit Reached: 50 Requests, '
    'try again later. Per Day","type":"rate_limit"}}'
)
OPENROUTER_HOUR = OPENROUTER_DAY.replace("Per Day", "Per Hour")

OR_SPEC = next(spec for spec in MODEL_CHAIN if spec.provider == "openrouter")
COHERE_SPEC = next(spec for spec in MODEL_CHAIN if spec.provider == "cohere")


def test_a_request_metered_per_day_is_a_verdict_but_a_hourly_one_is_a_schedule() -> None:
    """OpenRouter counts requests, not tokens, and names the window in the same breath.

    Waiting through a day is never the answer; waiting 45 seconds through an hour is,
    because the same model will take the work and the next hop has no key yet.
    """
    assert not _http_failure(OR_SPEC, httpx.Response(429, text=OPENROUTER_DAY)).retryable
    assert _http_failure(OR_SPEC, httpx.Response(429, text=OPENROUTER_HOUR)).retryable

    # And the window has to survive the cut: the refusal message is truncated for display,
    # so matching it before truncating is what keeps a long body from becoming a minute.
    filler = "no useful detail here; " * 20
    late = f'{{"error":{{"message":"Rate limit exceeded: {filler} Per Day"}}}}'
    assert not _http_failure(OR_SPEC, httpx.Response(429, text=late)).retryable


@pytest.mark.parametrize(
    "payload",
    [
        # Chat V2 as documented: a list of typed blocks under `message`.
        {"message": {"content": [{"type": "text", "text": '{"text": "g", "score": 9}'}]}},
        # The same field answering with a bare string.
        {"message": {"content": '{"text": "g", "score": 9}'}},
        # V1, which put it at the top level.
        {"text": '{"text": "g", "score": 9}'},
    ],
    ids=["blocks", "string", "legacy"],
)
def test_the_cohere_hop_is_asked_plainly_and_its_answer_is_found_wherever_it_lives(
    monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]
) -> None:
    """Cohere accepts no `response_format`, so the JSON is asked for and then checked.

    NOT EXERCISED against a live endpoint — no COHERE_API_KEY exists on this machine.
    That is the reason all three reply spellings are read here: a wrong guess about where
    the answer sits would cost a user a skipped hop at the exact moment every other
    provider had already spent its day, and would be invisible until then.
    """
    monkeypatch.setenv("COHERE_API_KEY", "co-test")
    body = {**payload, "usage": {"billed_units": {"input_tokens": 40, "output_tokens": 12}}}

    with respx.mock:
        route = respx.post(CohereProvider.BASE).mock(return_value=httpx.Response(200, json=body))
        with httpx.Client() as client:
            reply = CohereProvider().generate(
                system=SYSTEM,
                messages=[{"role": "user", "content": USER}],
                schema=Answer,
                model=COHERE_SPEC,
                client=client,
            )

    assert reply.parse(Answer).score == 9
    assert (reply.prompt_tokens, reply.completion_tokens) == (40, 12)
    sent = json.loads(route.calls[0].request.content)
    assert "response_format" not in sent
    assert sent["messages"][0] == {"role": "system", "content": SYSTEM}


def test_an_unnamed_gemini_quota_moves_on_rather_than_being_guessed_at(
    live: AIGateway,
) -> None:
    """Google states no window, so we may not sleep as if it promised one.

    Measured 2026-10-16: after a Groq day spent on the same posting, Gemini answered
    `You exceeded your current quota`. Waiting 45 seconds twice per task to discover that
    a day is not a minute is the failure already fixed for Groq, in a second dialect.
    """
    with respx.mock:
        respx.post(GROQ_URL).mock(
            return_value=httpx.Response(
                429, text="Rate limit reached on tokens per day (TPD): Limit 200000"
            )
        )
        gemini = respx.post(url__regex=GEMINI_CALL).mock(
            return_value=httpx.Response(429, text=GEMINI_QUOTA)
        )
        with pytest.raises(ExhaustedRetriesError) as exc:
            live.ask(task="answer", system=SYSTEM, user=USER, schema=Answer)

    # One call per Gemini model in the chain and not one more: the refusal is a verdict
    # about today, so the answer is the next model, never a re-ask of the same one.
    gemini_hops = sum(1 for spec in MODEL_CHAIN if spec.provider == "gemini")
    assert gemini.call_count == gemini_hops, "an unnamed quota must not be re-asked in place"
    text = str(exc.value)
    assert "out of quota today" in text
    assert "groq" in text and "gemini" in text, "both ceilings must be named"


def test_the_quota_hint_points_at_the_work_that_is_still_free() -> None:
    """The reader is not done for the day: half the tool never asks a model."""
    error = ExhaustedRetriesError(
        "tailor",
        2,
        [
            "groq: groq: HTTP 429: tokens per day (TPD): Limit 200000, Used 199722",
            f"gemini: gemini: HTTP 429: {GEMINI_QUOTA}",
        ],
    )

    assert "groq, gemini" in str(error)
    assert "rank --max-new 0" in str(error)
    assert "careerpilot keys" in str(error)
