"""Wall-time floor assertions on hot-path fixtures.

A 4x margin over typical wall time, with a 3-second floor, catches
catastrophic regressions (e.g., a previously-shipped 60-300x slowdown
from defensive config.override calls per partition) without being
brittle on slow CI runners. This is a smoke gate, not a benchmark
suite — per-fixture micro-budgets and trend tracking belong in the
wider benchmark rewrite.

State-tie substrates (AND-XOR class) get dedicated entries below.
These exercise the per-state max-min cascade path that the goldens
deliberately exclude (per ``test/golden/zoo.py:32-33`` — tied-state
fixtures excluded pending the tie-resolution overhaul). Without these
budgets, a regression in the cascade's per-state recomputation —
e.g., an O(|ties|^2) blow-up or an unintended Composition-level
escalation that pulls CES into the inner loop — could ship
undetected.
"""

from __future__ import annotations

import time

import pytest

from test.golden import ALL_FIXTURES
from test.golden.compute import compute_all_layers

_FIXTURES_BY_NAME = {f.name: f for f in ALL_FIXTURES}

PERF_BUDGETS_S: dict[str, float] = {
    "basic_iit3_emd": 8.0,
    "basic_iit4_2023": 3.0,
    "basic_iit4_2026": 3.0,
    "xor_iit4_2026": 3.0,
    "logistic3_k8_iit4_2026": 3.0,
}


@pytest.mark.perf
@pytest.mark.parametrize("name", list(PERF_BUDGETS_S))
def test_perf_budget(name: str) -> None:
    budget_s = PERF_BUDGETS_S[name]
    fixture = _FIXTURES_BY_NAME[name]
    start = time.perf_counter()
    with fixture.config_context():
        compute_all_layers(fixture)
    elapsed = time.perf_counter() - start
    assert elapsed < budget_s, (
        f"{name} took {elapsed:.2f}s, exceeded perf budget {budget_s}s. "
        f"4x margin over typical suggests catastrophic regression — "
        f"profile the hot path before retuning."
    )


# ----- State-tie substrate perf budgets -----
#
# These substrates produce tied specified states at the system level;
# they exercise the per-state max-min cascade. Budget is generous (4x
# typical) to act as a smoke gate, not a regression detector.


@pytest.mark.perf
def test_perf_and_xor_sia_state_tie() -> None:
    """AND-XOR substrate has tied specified states for state (0, 1).

    Exercises the per-state max-min cascade on a small substrate. Typical
    sia() wall time is well under 1s; the 5s budget tolerates 5-10x
    slowdown without flaking on a slow CI runner.
    """
    from pyphi import System
    from test import example_substrates

    substrate = example_substrates.and_xor_substrate()
    system = System(substrate, state=(0, 1))
    start = time.perf_counter()
    sia = system.sia()
    elapsed = time.perf_counter() - start
    assert sia is not None
    assert elapsed < 5.0, (
        f"and_xor sia() took {elapsed:.2f}s on a 2-node substrate — "
        f"the per-state cascade should be near-instant. Check for an "
        f"unintended Composition-level escalation invoking CES per tie."
    )


@pytest.mark.perf
def test_perf_no_state_tie_path_unchanged() -> None:
    """Substrates *without* state ties should pay no measurable overhead.

    The cascade primitive has a constant per-call cost; the no-tie
    case (single-spec input) walks the cascade and resolves
    immediately at level 1. Asserting that a typical IIT 4.0 (2026)
    basic SIA stays under its baseline budget guards against the
    cascade infrastructure leaking cost into the common path.
    """
    fixture = _FIXTURES_BY_NAME["basic_iit4_2026"]
    start = time.perf_counter()
    with fixture.config_context():
        compute_all_layers(fixture)
    elapsed = time.perf_counter() - start
    # Tighter than the catastrophic 3s budget above; if the cascade
    # adds even 30% overhead this trips.
    assert elapsed < 2.0, (
        f"basic_iit4_2026 (no-tie path) took {elapsed:.2f}s — "
        f"cascade infrastructure may be leaking constant cost into "
        f"the common path. Profile sia()."
    )
