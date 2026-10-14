"""Shared fixtures. Nothing here may touch a network or a real model."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from careerpilot.ai.cassette import CassetteStore
from careerpilot.ai.fake import ScriptedGateway
from careerpilot.ai.gateway import AIGateway, set_gateway
from careerpilot.ai.trace import TraceSink
from careerpilot.config import live_mode_enabled, profile_path
from careerpilot.domain.job import JobPosting
from careerpilot.profile.model import Profile, load_profile
from support import TaskFake


@pytest.fixture(autouse=True)
def _isolated_gateway(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Every test starts with no gateway and with live mode forced off."""
    monkeypatch.delenv("CAREERPILOT_LIVE", raising=False)
    live_mode_enabled.cache_clear()
    set_gateway(None)
    yield
    set_gateway(None)
    live_mode_enabled.cache_clear()


@pytest.fixture
def profile() -> Profile:
    return load_profile(profile_path())


@pytest.fixture
def job() -> JobPosting:
    return JobPosting(
        source="greenhouse",
        source_id="123",
        company="Example AI",
        title="Backend Engineer, RAG",
        location="Remote",
        url="https://example.com/jobs/123",
        description=(
            "About the role: build retrieval services.\n"
            "Requirements:\n"
            "- 3+ years building backend services in Python or Go\n"
            "- Experience with PostgreSQL and vector search\n"
            "- Familiarity with RAG pipelines and LLM APIs\n"
            "Nice to have: experience with SSE streaming\n"
        ),
    )


@pytest.fixture
def use_fake() -> Any:
    """Install a scripted gateway so `ask()` never reaches a provider."""

    def _install(payloads: list[Any]) -> ScriptedGateway:
        gateway = ScriptedGateway(payloads)
        set_gateway(gateway)
        return gateway

    return _install


@pytest.fixture
def use_task_fake() -> Any:
    """Install a gateway that answers by task name, for multi-call flows."""

    def _install(by_task: dict[str, Any]) -> TaskFake:
        gateway = TaskFake(by_task)
        set_gateway(gateway)
        return gateway

    return _install


@pytest.fixture
def tmp_cassette(tmp_path: Any) -> CassetteStore:
    return CassetteStore(tmp_path / "cassettes")


@pytest.fixture
def gateway(tmp_cassette: CassetteStore) -> AIGateway:
    return AIGateway(providers={}, cassette=tmp_cassette, sink=TraceSink(quiet=True))
