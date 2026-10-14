"""The groundedness gate: what a bullet must survive before it can be printed.

Each case names the check it is supposed to trip, and asserts the model was consulted
only when it should have been — checks 1 to 4 are pure code and must stay that way.
"""

from __future__ import annotations

from typing import Any

import pytest

from careerpilot.ground.gate import Claim, check_bullet, gate
from careerpilot.profile.model import Profile

SUPPORTED = {"supported": True, "confidence": 0.9, "unsupported_span": ""}
REJECTED = {"supported": False, "confidence": 0.9, "unsupported_span": "the metric"}
HESITATING = {"supported": True, "confidence": 0.4, "unsupported_span": ""}

TRUTHFUL = (
    "Built the RAG assistant with pgvector similarity combined with PostgreSQL "
    "full-text search"
)


@pytest.mark.parametrize(
    ("text", "atom_ids", "judge", "expected"),
    [
        pytest.param(TRUTHFUL, [], None, {"no_citation"}, id="uncited"),
        pytest.param(
            TRUTHFUL, ["atom-that-does-not-exist"], None, {"unknown_atom"}, id="invented-id"
        ),
        pytest.param(
            "Compressed Bangla text so 40% more characters fit per segment",
            ["proj-bornomala"],
            None,
            {"unsourced_number"},
            id="invented-metric",
        ),
        pytest.param(
            "Implemented Flutter authentication for the CSxEDU assistant",
            ["proj-csxedu-rag"],
            None,
            {"misattributed_skill"},
            id="right-skill-wrong-work",
        ),
        pytest.param(TRUTHFUL, ["proj-csxedu-rag"], REJECTED, {"not_entailed"}, id="not-entailed"),
        pytest.param(
            TRUTHFUL, ["proj-csxedu-rag"], HESITATING, {"not_entailed"}, id="low-confidence"
        ),
        pytest.param(TRUTHFUL, ["proj-csxedu-rag"], SUPPORTED, set(), id="ships"),
    ],
)
def test_each_failure_mode_trips_exactly_its_check(
    use_fake: Any,
    profile: Profile,
    *,
    text: str,
    atom_ids: list[str],
    judge: Any,
    expected: set[str],
) -> None:
    gateway = use_fake([judge] if judge is not None else [])

    verdict = check_bullet(text, atom_ids, profile)

    assert {f.kind for f in verdict.failures} == expected
    assert verdict.shipped == (not expected)
    assert len(gateway.calls) == (1 if judge is not None else 0)


def test_a_code_failure_never_costs_a_model_call(use_fake: Any, profile: Profile) -> None:
    gateway = use_fake([])

    check_bullet("Cut latency by 90% with Kubernetes", [], profile)

    assert gateway.calls == []


def test_rejections_are_counted_not_silently_dropped(use_fake: Any, profile: Profile) -> None:
    use_fake([SUPPORTED, SUPPORTED])
    claims = [
        Claim(text=TRUTHFUL, atom_ids=["proj-csxedu-rag"]),
        Claim(text="Reduced cost by 60% using pgvector", atom_ids=["proj-csxedu-rag"]),
        Claim(
            text="Shipped the emergency mode with 9 national hotlines",
            atom_ids=["proj-bornomala-emergency"],
        ),
    ]

    report = gate(claims, profile)

    assert report.total == 3
    assert len(report.shipped) == 2
    assert report.pass_rate == pytest.approx(2 / 3, abs=1e-4)
    assert report.rejection_summary() == {"unsourced_number": 1}
