# Tier 1 Inline Perf Budget — Design

**Status:** approved (2026-05-10)
**Branch base:** 2.0 head `755f65eb`
**Implementation file:** `test/test_perf_budget.py` (new)
**Scope:** ~30 lines of test code + one marker registration

---

## Motivation

During the 2.0 work a 4-day-old structural change to `IIT4_2026Formalism`
(5 nested defensive `config.override` calls per partition) interacted
catastrophically with a 3-year-old latent `atomic_write_yaml` callback
to produce a 60-300x slowdown on the 2026 hot path. The golden suite
caught nothing — goldens are correctness gates, not performance gates,
and the suite wall time crept from minutes to 13 minutes without
tripping any signal. The 2026 issue is fixed (commit `7c2e2cd2`), but
the lesson is: structural refactors in hot paths need a perf gate.

The next two scheduled projects (non-binary units and Zaeemzadeh
pruning) both touch hot paths. A cheap, well-placed gate before either
lands turns a future "spent a day profiling the regression" into a CI
failure on the originating PR.

## Goal

Catch *catastrophic* regressions (≥ 4x slowdown over typical wall time
on a hot-path fixture) with minimal CI cost and zero ongoing
maintenance overhead — until the full benchmark-suite rewrite
supersedes it.

This is explicitly **not** a benchmark suite. It is a smoke gate.
Per-fixture micro-budgets, baseline tracking, counter-based stable
measurements, nightly trend reports, and the `benchmarks/` rewrite all
belong in the follow-on benchmark-suite project.

## Non-goals

- Detecting sub-2x regressions
- Stable measurements across CI runner classes
- Trend tracking or alerting
- Rewriting `benchmarks/` (the prior-architecture ASV suite)
- Adding a nightly workflow
- Per-machine auto-calibration

## Architecture

A standalone test file parametrizes over a small canary set with
hardcoded wall-time floors. Each test calls the existing golden
harness's `compute_all_layers(fixture)` inside the fixture's own
`config_context()`, times it with `time.perf_counter()`, and asserts
elapsed < budget. The `perf` pytest marker is registered in
`test/conftest.py` alongside the existing `golden`/`robust`/`slow`
markers. Tests run in the default fast lane, so every PR is gated
without explicit invocation.

## Canary fixtures

Five fixtures, chosen to cover each active formalism's hot path with
some scaling sensitivity:

| Fixture | Formalism | Why included |
|---|---|---|
| `basic_iit3_emd` | IIT 3.0 | Hot path for the legacy formalism |
| `basic_iit4_2023` | IIT 4.0 (2023) | Hot path for the stable 4.0 |
| `basic_iit4_2026` | IIT 4.0 (2026) | The site of the motivating regression |
| `xor_iit4_2026` | IIT 4.0 (2026) | Second 2026 fixture so a partition-specific regression isn't a single-point-of-failure |
| `logistic3_k8_iit4_2026` | IIT 4.0 (2026) | Larger substrate; covers scaling-class regressions a 3-node basic might miss |

All five live in `test/golden/zoo.py` already, in the fast (non-`slow`)
lane. The perf-budget tests re-run them; that cost (~10-15s added to
the fast lane) is the price of separation between correctness and
performance signals.

## Budget policy

For each canary fixture:

```
budget_s = max(3.0, 4 * typical_wall_time_seconds)
```

- **4x multiplier** catches anything ≥ 4x slowdown. Tighter than the
  ROADMAP's exploratory "5x" without entering flake territory: typical
  CI variance (1.5-3x) + battery throttling (~2x) compounds to ~3-4x
  worst-case benign, leaving headroom but not slack.
- **3s floor** absorbs measurement noise on sub-1s fixtures, where 4x
  of a small typical produces a budget close enough to noise to flake.
- **Typical** = median of 3 sequential runs on the development machine
  (Apple Silicon, the same target as the golden fixtures).

Numbers are rounded up to clean values (3, 4, 5, 8, 10, 20, 30 s) for
readability. The calibration is part of implementation, not design.

If the gate flakes on a CI runner: re-tune that one budget upward by
50% and document; don't ratchet down on success. If real-world
variance keeps exceeding 4x, the gate moved to a flaky runner class
and the policy needs revisiting — but that is not solving here.

## Implementation

### `test/test_perf_budget.py` (new)

```python
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
    "basic_iit3_emd": 3.0,
    "basic_iit4_2023": 3.0,
    "basic_iit4_2026": 4.0,
    "xor_iit4_2026": 8.0,
    "logistic3_k8_iit4_2026": 20.0,
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
```

The numbers in `PERF_BUDGETS_S` above are placeholders. Calibration
during implementation may change them.

### `test/conftest.py` (one line added)

Inside the existing `pytest_configure(config)` function:

```python
config.addinivalue_line(
    "markers",
    "perf: Wall-time floor assertion on a hot-path fixture",
)
```

## CI integration

No new workflow files. Tests run in the existing fast lane (default
`pytest test/` invocation). The `perf` marker is *not* `slow`, so the
fast lane picks them up without opt-in. Convenience selectors:

- Run only perf budgets: `pytest -m perf`
- Skip perf budgets: `pytest -m "not perf"`

## Acceptance

1. **Canary set passes locally:** all 5 perf canaries pass on the
   development machine at the pinned budgets, with elapsed wall time
   well under (~25%) the budget — confirming headroom.
2. **Fast-lane cost contained:** Fast lane (`pytest test/ -m "not slow"`)
   wall time increases by less than 20s over baseline.
3. **Sensitivity check:** Manually doubling a fixture's typical run
   time (e.g., by inserting `time.sleep`) trips the corresponding
   budget; halving it doesn't flip a passing test to failing.
4. **Type/lint clean:** `uv run pyright test/test_perf_budget.py` and
   `uv run ruff check test/test_perf_budget.py` both clean.
5. **No ROADMAP marker leakage:** The new file contains no `P11.8`,
   `Tier 1`, `Phase A`, or similar planning artifacts. Comments
   describe the *what* (wall-time floor) without referencing the
   ROADMAP project context.

## Risks

| Risk | Mitigation |
|---|---|
| CI runner variance exceeds 4x | Single-budget retune (50% bump on the affected canary) is a 2-minute fix. Don't retune all on first flake. |
| Adds 10-15s to fast lane | Acceptable. The alternative is undetected multi-minute wall-time creep. |
| Battery-throttled dev laptop trips budget | Local-only workaround: `pytest -m "not perf"` until plugged in. |
| Budget rot over 6+ months | Tier 1 supersedes itself when the broader benchmark-suite project lands. Not solving here. |
| Sub-second-fixture noise causes spurious failure | 3s floor absorbs noise at the small end. |

## Out of scope (for the follow-on benchmark-suite project)

Items deferred to the benchmark-suite project (Tier 2):

- Counter-based stable measurements (cProfile call counts, partition
  evaluation counts)
- Trend tracking and regression-vs-baseline policy
- Nightly ASV runs on develop / 2.0
- `benchmarks/` rewrite for the 2.0 architecture and vocabulary
- Layered set of micro-benchmarks (`find_mip`, `find_mice`, single
  distinction, repertoire compute) for regression localization
- `.github/workflows/` nightly job

The Tier 1 budget is the bridge that prevents catastrophic regressions
from reaching `2.0` between now and the benchmark-suite project.

## Implementation phases (for the plan)

One commit total, with five ordered sub-steps. Budgets, marker, and
test file land together — the marker is meaningless without the test,
and budgets are meaningless without code that asserts on them.

1. **Calibrate budgets.** Run each canary 3× sequentially on the dev
   machine via `time pytest test/test_golden_regression.py -k <name>`
   (or equivalent), take median wall time, apply
   `max(3.0, 4 * median)`, round up to a clean number (3, 4, 5, 8, 10,
   20, 30 s). Record the typicals — they go in the commit message,
   not the source.
2. **Register the `perf` marker** in `test/conftest.py`.
3. **Add `test/test_perf_budget.py`** with the parametrized test and
   pinned budgets.
4. **Run acceptance checks**: canaries pass, fast lane wall time
   contained, sensitivity check (manual sleep), pyright + ruff clean.
5. **Commit** with the typicals documented in the message body.
