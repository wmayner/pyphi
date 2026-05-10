"""Wall-time floor assertions on hot-path fixtures.

A 4x margin over typical wall time, with a 3-second floor, catches
catastrophic regressions (e.g., a previously-shipped 60-300x slowdown
from defensive config.override calls per partition) without being
brittle on slow CI runners. This is a smoke gate, not a benchmark
suite — per-fixture micro-budgets and trend tracking belong in the
wider benchmark rewrite.
"""

from __future__ import annotations

import time

import pytest

from .golden import ALL_FIXTURES
from .golden.compute import compute_all_layers

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
