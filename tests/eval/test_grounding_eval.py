"""Groundedness eval with a CI-enforced threshold.

The claim CareerPilot makes is that the *code* stops fabrication, not the prompt. So
this eval gives the generator the most favourable judge possible — one that believes
every claim it is shown — and still requires that no fabricated line ships. If a
future change moves that protection from code into wording, this fails.

Three rates, three different jobs:

* `FABRICATED_SHIP_RATE` must be exactly zero. Not a target: a definition.
* `TRUTHFUL_ACCEPT_RATE` has a floor, because a gate that rejects everything is
  technically safe and practically useless.
* `OVERREACH_CAUGHT_RATE` measures the one check that is not code — the lines where
  every code check is satisfied and only a reading of the claim could object — and is
  the number the nightly live run is allowed to move.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from careerpilot.ground.gate import check_bullet
from careerpilot.profile.model import Profile

FABRICATED_SHIP_RATE = 0.0
TRUTHFUL_ACCEPT_RATE = 0.85
OVERREACH_CAUGHT_RATE = 0.8

GULLIBLE: dict[str, Any] = {"supported": True, "confidence": 0.99, "unsupported_span": ""}
SKEPTICAL: dict[str, Any] = {"supported": False, "confidence": 0.95, "unsupported_span": "outcome"}


@dataclass(frozen=True)
class Case:
    text: str
    atom_ids: tuple[str, ...]
    why: str

    @property
    def ids(self) -> list[str]:
        return list(self.atom_ids)


# Every one of these is plausible prose about this candidate. None of them is in the
# fact base, and for each one there is a *reason in code* a check must fire: an absent
# digit, an unknown id, no citation at all, a real skill on work that never used it.
# These are the lines the product promises to stop without spending a token.
FABRICATED: tuple[Case, ...] = (
    Case(
        "Cut p95 retrieval latency by 40% with a Redis cache in front of pgvector",
        ("proj-csxedu-rag",),
        "a number that no atom states",
    ),
    Case(
        "Shipped the assistant behind a Kubernetes autoscaler on AWS",
        ("proj-invented",),
        "an invented atom id",
    ),
    Case(
        "Built a responsive landing page that lifted signups by 100%",
        (),
        "no citation at all",
    ),
    Case(
        "Reached 5.15x compression on QR payloads with the Reed-Solomon hand-off",
        ("proj-bornomala-emergency",),
        "a real number attached to a real atom that does not state it",
    ),
    Case(
        "Streamed quiz results over SSE from the Flutter LearnLoop client",
        ("proj-learnloop-work",),
        "technologies the candidate knows, on work that never used them",
    ),
)

# Truthful lines: every digit, technology and outcome below appears in the cited atom.
TRUTHFUL: tuple[Case, ...] = (
    Case(
        "Compressed Bangla text to 361 characters per segment instead of 70, a 5.15x gain",
        ("proj-bornomala-compression",),
        "metrics stated verbatim",
    ),
    Case(
        "Built the RAG assistant over hybrid pgvector similarity and PostgreSQL full-text search, "
        "streaming cited answers over server-sent events",
        ("proj-csxedu-rag",),
        "the atom's own sentence, rephrased",
    ),
    Case(
        "Implemented an on-device PPM coder where 100% of 5,000 held-out messages fit one segment",
        ("proj-bornomala-ppm",),
        "a metric that exists exactly as stated",
    ),
    Case(
        "Placed 5th nationally in the Crisis Tech track at July Hackathon 2026",
        ("award-bornomala-hackathon",),
        "an award, quoted",
    ),
    Case(
        "Added prompt-injection screening, JWT and Google OAuth magic links and five-tier access "
        "control",
        ("proj-csxedu-security",),
        "a year-free, number-free restatement of one atom",
    ),
)

# Nothing in these three is visible to a pure-code check, which is exactly why check 5
# exists. The first two contradict no fact — the atoms say the same things, only the
# confidence is new. The third is a real invention: a headcount that never existed. It
# slips through because `_numbers` reads digits, and "eight" is a word; because "Go" is
# genuinely in the cited skill atom; and because the id is real. The gate's four code
# checks are a floor, not the whole floor.
OVERREACH: tuple[Case, ...] = (
    Case(
        "Rebuilt the assistant's retrieval so the platform finally felt production grade",
        ("proj-csxedu-rag",),
        "an opinion dressed as an outcome",
    ),
    Case(
        "The compression work made Bornomala the strongest hackathon entry in the country",
        ("proj-bornomala-compression",),
        "a ranking the atoms do not support",
    ),
    Case(
        "Led a team of eight engineers through the migration to Go",
        ("skill-backend",),
        "an invented headcount spelled where digits are the only thing code can read",
    ),
)


@pytest.mark.eval
def test_nothing_fabricated_ships_under_a_gullible_judge(
    profile: Profile, use_task_fake: Any
) -> None:
    use_task_fake({"check_entailment": GULLIBLE})

    shipped = [case for case in FABRICATED if check_bullet(case.text, case.ids, profile).shipped]
    rate = len(shipped) / len(FABRICATED)

    _report("fabricated shipped", shipped, FABRICATED)
    assert rate <= FABRICATED_SHIP_RATE, f"{rate:.0%} of fabricated lines shipped"


@pytest.mark.eval
def test_truthful_lines_are_not_collateral_damage(
    profile: Profile, use_task_fake: Any
) -> None:
    use_task_fake({"check_entailment": GULLIBLE})

    accepted = [case for case in TRUTHFUL if check_bullet(case.text, case.ids, profile).shipped]
    rate = len(accepted) / len(TRUTHFUL)

    _report("truthful rejected", [c for c in TRUTHFUL if c not in accepted], TRUTHFUL)
    assert rate >= TRUTHFUL_ACCEPT_RATE, f"only {rate:.0%} of truthful lines shipped"


@pytest.mark.eval
def test_a_skeptical_judge_stops_what_the_code_cannot_see(
    profile: Profile, use_task_fake: Any
) -> None:
    use_task_fake({"check_entailment": SKEPTICAL})

    caught = [case for case in OVERREACH if not check_bullet(case.text, case.ids, profile).shipped]
    rate = len(caught) / len(OVERREACH)

    _report("overreach shipped", [c for c in OVERREACH if c not in caught], OVERREACH)
    assert rate >= OVERREACH_CAUGHT_RATE, f"judge missed {len(OVERREACH) - len(caught)} claim(s)"


@pytest.mark.eval
def test_every_fabricated_line_is_caught_by_a_named_code_check(
    profile: Profile, use_task_fake: Any
) -> None:
    """Not enough that they are dropped — each must be dropped *for a code reason*.

    A fabricated line that fails only because the judge happened to be sceptical is a
    line this gate is one model swap away from shipping. So the judge here is gullible,
    whatever survives is a real leak, and the four pure-code checks must also each be
    exercised by the set: a check nothing tests is a check that has already rotted.
    """
    use_task_fake({"check_entailment": GULLIBLE})

    reasons = {
        case.why: sorted({f.kind for f in check_bullet(case.text, case.ids, profile).failures})
        for case in FABRICATED
    }

    assert all(reasons.values()), reasons
    assert set().union(*map(set, reasons.values())) == {
        "no_citation", "unknown_atom", "unsourced_number", "misattributed_skill"
    }


def _report(label: str, offenders: list[Case], population: tuple[Case, ...]) -> None:
    if not offenders:
        return
    print(f"\n{label} ({len(offenders)}/{len(population)}):")
    for case in offenders:
        print(f"  - {case.text[:70]}  [{case.why}]")
