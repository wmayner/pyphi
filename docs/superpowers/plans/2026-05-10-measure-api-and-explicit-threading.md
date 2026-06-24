# Measure API + Explicit-Parameter Threading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce typed metric Protocols and thread measure choices through every internal call as explicit parameters, eliminating the `config.override(measure=...)` wrapper pattern and making the cap-regression class of bug impossible by construction.

**Architecture:** Three Protocol classes (`DistributionMetric`, `StateAwareMetric`, `CompositeMetric`) capture the metric shape diversity. Below the user-facing `System` class methods, every function that uses a metric receives it as an explicit parameter. The `config.formalism.iit.{mechanism,system}_phi_measure` config fields are read exactly at the `System.*` boundary; internal code is config-free. Six migration phases, each ending green.

**Tech Stack:** Python 3.13, pyright (standard mode), Protocol types from `typing`, pytest, hypothesis, existing pyphi metric registry infrastructure.

---

## Spec

See `docs/superpowers/specs/2026-05-10-measure-api-and-explicit-threading-design.md` (committed at `90ca10be`). Read it first if any decision feels ambiguous.

## Branch base

`2.0` head `90ca10be`. Land work as six commits on `2.0` (one per phase, or finer if a phase has clean sub-steps). Do **not** push without explicit per-action consent.

## File map

```
pyphi/metrics/
├── protocols.py             CREATE (Phase 1): 3 Protocol classes
└── distribution.py          MODIFY (Phases 1-2): retype, split registries, drop `measures`

pyphi/formalism/iit4/
├── __init__.py              MODIFY (Phase 3, 5): thread params; remove config reads at 79, 414, 487
└── formalism.py             MODIFY (Phase 4): add metric kwargs, delete 5 config.override blocks (lines 266, 278, 293, 300, 305)

pyphi/formalism/iit3/
└── formalism.py             MODIFY (Phase 4): thread params

pyphi/formalism/
└── queries.py               MODIFY (Phase 3-4): resolve at boundary if find_mip dispatcher reads measure config (verify in Phase 3)

pyphi/core/
└── repertoire_algebra.py    MODIFY (Phase 3): thread params; remove config reads at 297, 530

pyphi/system.py / pyphi/subsystem.py    MODIFY (Phase 4-5): resolve metric at boundary, thread to formalism layer

pyphi/actual.py              MODIFY (Phase 6): AC parallel

pyphi/conf/snapshot.py       MODIFY (Phase 6): revisit ConfigSnapshot.as_kwargs colliding-field workaround

test/
├── test_metric_protocols.py             CREATE (Phase 1)
├── test_metric_resolution.py            CREATE (Phase 2)
├── test_formalism_metric_threading.py   CREATE (Phase 4)
├── test_cap_regression_impossible.py    CREATE (Phase 5)
└── test_big_phi_robust.py               MODIFY (Phase 5): 3 raw-import sites at 441, 474, 494
```

## Project conventions (do not violate)

- **No P-number markers, "Phase A", `TODO(Px)`, "per ROADMAP" in source/comments/docstrings/changelog filenames or contents.** The spec is the only place planning context lives. Commit messages MAY reference "Phase N of the spec" since that's pointing at the spec doc.
- **Default to no comments unless WHY is non-obvious.** Module-level docstrings carry intent; inside function bodies, no commentary.
- **Pre-commit hooks (ruff + pyright) must pass on commit.** Never bypass with `--no-verify` or `SKIP=pyright`. If a hook fails, run `uv run ruff check pyphi/ test/` and `uv run pyright pyphi/` directly to see the message; fix the underlying issue.
- **GPG signing bypass is authorized for this session only.** Use `git -c commit.gpgsign=false commit ...`. Do not change git config.
- **Use `uv run` for any Python invocation.**
- **Targeted `git add` only.** Never `git add -A` or `git add .`. There are untracked detritus files in the repo root.
- **Failing tests under a tightened contract may reveal latent inconsistencies — diagnose before reverting.** Today's session caught one such (the cap regression); expect Phases 3-5 to surface more.
- **Long test runs: use `run_in_background: true`.** Golden fast lane is ~40s; full fast lane ~1min; hypothesis fast lane ~9s. Kick off slow runs in background, work on the next thing while they run.

## Per-phase gates

Every commit must pass:

1. **Goldens** — `uv run pytest test/test_golden_regression.py` → 17/17 numerical match.
2. **Fast unit lane** — `uv run pytest test/ -m "not slow"` → zero failures.
3. **Hypothesis fast lane** — `uv run pytest test/test_invariants_hypothesis.py` (or current equivalent) → green.
4. **Perf budget** — `uv run pytest test/ -m perf` → 5/5 pass with headroom.
5. **Pyright** — `uv run pyright pyphi/` → only the 2 pre-existing geometry.py errors as baseline (now at `pyphi/visualize/ces/geometry.py`).
6. **Ruff** — `uv run ruff check pyphi/ test/` + `uv run ruff format --check pyphi/ test/` → clean.

If a gate goes red mid-phase: stop, diagnose root cause, fix in the same phase before commit.

---

## Phase 1 — Protocol scaffolding + INTRINSIC_DIFFERENTIATION cleanup

**Goal:** Introduce three Protocol types. Retype each existing metric function against the right Protocol. Remove `INTRINSIC_DIFFERENTIATION`'s vestigial `q` arg (small wart that bundles cleanly).

**Estimated:** 1 day.

### Task 1.1: Create the Protocols file

**Files:**
- Create: `pyphi/metrics/protocols.py`

- [ ] **Step 1: Write the file**

```python
"""Protocol types for metric callables.

Three Protocol classes capture the shape diversity in pyphi's metric
machinery. Each registered metric satisfies exactly one of these
Protocols; the registries are typed against the corresponding Protocol.

- ``DistributionMetric``: (p, q) -> float. Distribution-to-distribution
  distance. Symmetric or asymmetric (see ``asymmetric`` attribute).
- ``StateAwareMetric``: (p, state) -> float. Pointwise probability at a
  specified state.
- ``CompositeMetric``: (forward, partitioned, selectivity, *, state)
  -> DistanceResult. Multi-input metric returning rich metadata; used
  by GID / INTRINSIC_INFORMATION / INTRINSIC_SPECIFICATION at the
  system level.
"""

from __future__ import annotations

from typing import Protocol
from typing import runtime_checkable

from numpy.typing import ArrayLike

from pyphi.data_structures.state import State


@runtime_checkable
class DistributionMetric(Protocol):
    """Distribution-to-distribution distance."""

    name: str
    asymmetric: bool

    def __call__(self, p: ArrayLike, q: ArrayLike) -> float: ...


@runtime_checkable
class StateAwareMetric(Protocol):
    """Pointwise probability at a specified state."""

    name: str

    def __call__(self, p: ArrayLike, state: State) -> float: ...


@runtime_checkable
class CompositeMetric(Protocol):
    """Multi-input metric returning DistanceResult metadata."""

    name: str

    def __call__(
        self,
        forward: ArrayLike,
        partitioned: ArrayLike,
        selectivity: ArrayLike | None = None,
        *,
        state: State | None = None,
    ) -> "DistanceResult": ...
```

The forward-string reference to `DistanceResult` avoids a circular import; the actual import lives only at the import-resolution boundary.

- [ ] **Step 2: Confirm imports resolve**

```bash
uv run python -c "from pyphi.metrics.protocols import DistributionMetric, StateAwareMetric, CompositeMetric; print('ok')"
```

Expected: `ok`. If ImportError: check the `State` import path (`pyphi.data_structures.state.State` is the post-2.0 location; verify with `grep -rn "^class State" pyphi/`).

### Task 1.2: Write tests pinning the Protocol shapes

**Files:**
- Create: `test/test_metric_protocols.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Pin every registered metric's Protocol membership.

Catches signature drift: if a metric's function signature changes such
that it no longer matches its declared Protocol, the test fails at
collection time.
"""

from __future__ import annotations

import pytest

from pyphi.metrics import distribution
from pyphi.metrics.protocols import CompositeMetric
from pyphi.metrics.protocols import DistributionMetric
from pyphi.metrics.protocols import StateAwareMetric

DISTRIBUTION_METRICS = [
    "EMD",
    "L1",
    "ENTROPY_DIFFERENCE",
    "PSQ2",
    "MP2Q",
    "KLD",
    "ID",
    "AID",
    "KLM",
    "BLD",
]

STATE_AWARE_METRICS = [
    "IIT_4.0_SMALL_PHI",
    "IIT_4.0_SMALL_PHI_NO_ABSOLUTE_VALUE",
    "INTRINSIC_DIFFERENTIATION",
    "APMI",
]

COMPOSITE_METRICS = [
    "GENERALIZED_INTRINSIC_DIFFERENCE",
    "INTRINSIC_SPECIFICATION",
    "INTRINSIC_INFORMATION",
]


@pytest.mark.parametrize("name", DISTRIBUTION_METRICS)
def test_distribution_metric_satisfies_protocol(name: str) -> None:
    metric = distribution.measures[name]
    assert isinstance(metric, DistributionMetric), (
        f"{name!r} does not satisfy DistributionMetric Protocol; "
        f"call signature mismatch."
    )


@pytest.mark.parametrize("name", STATE_AWARE_METRICS)
def test_state_aware_metric_satisfies_protocol(name: str) -> None:
    metric = distribution.measures[name]
    assert isinstance(metric, StateAwareMetric), (
        f"{name!r} does not satisfy StateAwareMetric Protocol; "
        f"call signature mismatch."
    )


@pytest.mark.parametrize("name", COMPOSITE_METRICS)
def test_composite_metric_satisfies_protocol(name: str) -> None:
    metric = distribution.measures[name]
    assert isinstance(metric, CompositeMetric), (
        f"{name!r} does not satisfy CompositeMetric Protocol; "
        f"call signature mismatch."
    )
```

The exact lists may need adjustment after Step 2 — some metrics may be missing from the registry or have different names. Use `grep "^@measures.register" pyphi/metrics/distribution.py` to enumerate.

- [ ] **Step 2: Run the tests — expect mixed failures**

```bash
uv run pytest test/test_metric_protocols.py -v 2>&1 | tail -30
```

Expected: most metrics pass (those whose signatures already match a Protocol). Failures will likely include:
- `INTRINSIC_DIFFERENTIATION` (currently `(p, q, state=None)` — doesn't match `StateAwareMetric`'s `(p, state)`)
- Any other metric whose actual signature deviates from the Protocol declared for it

For each failure, decide whether to (a) fix the metric's signature, (b) adjust the test's classification (i.e., maybe a metric is actually CompositeMetric not StateAwareMetric), or (c) extend the Protocol definitions if a fourth shape is needed.

### Task 1.3: Remove INTRINSIC_DIFFERENTIATION's vestigial `q` arg

**Files:**
- Modify: `pyphi/metrics/distribution.py:917`
- Modify: `pyphi/formalism/iit4/__init__.py:464` (caller)

- [ ] **Step 1: Update the function signature**

Find the current definition at `pyphi/metrics/distribution.py:917`:

```python
@measures.register("INTRINSIC_DIFFERENTIATION", asymmetric=True)
def intrinsic_differentiation(p, q, state=None):
    ...
```

Change to:

```python
@measures.register("INTRINSIC_DIFFERENTIATION")
def intrinsic_differentiation(p: ArrayLike, state: State) -> float:
    """Pointwise intrinsic differentiation evaluated at the given state."""
    ...
```

Note: drop `asymmetric=True` since `StateAwareMetric` doesn't have an `asymmetric` attribute (that's only on `DistributionMetric`). Verify by re-reading the metric body — `q` was unused, so removing it is safe.

Inside the body, replace any reference to `q` with whatever the function actually uses (per the spec's analysis, `q` is unused).

- [ ] **Step 2: Update the caller at iit4/__init__.py:464**

Find:

```python
return metrics.distribution.intrinsic_differentiation(
    forward_repertoire, partitioned_repertoire, state
)
```

(Exact form may differ — read the current code.) Update to:

```python
return metrics.distribution.intrinsic_differentiation(
    forward_repertoire, state
)
```

Drop the `partitioned_repertoire` arg.

- [ ] **Step 3: Run the existing metrics test suite to confirm no regressions**

```bash
uv run pytest test/test_metrics.py -v 2>&1 | tail -10
```

Expected: green. If failures: a caller other than `iit4/__init__.py:464` was relying on the old signature. Grep for `intrinsic_differentiation(` and update.

### Task 1.4: Run all gates and commit Phase 1

- [ ] **Step 1: Goldens green**

```bash
uv run pytest test/test_golden_regression.py 2>&1 | tail -5
```

Expected: 17 passed (no `slow`-marked entries).

- [ ] **Step 2: Fast unit lane green**

```bash
uv run pytest test/ -m "not slow" -q --no-header 2>&1 | tail -5
```

Expected: zero failures.

- [ ] **Step 3: Perf budget green**

```bash
uv run pytest test/ -m perf -q 2>&1 | tail -5
```

Expected: 5/5 pass.

- [ ] **Step 4: Pyright + ruff clean**

```bash
uv run pyright pyphi/ 2>&1 | tail -3
uv run ruff check pyphi/ test/
uv run ruff format --check pyphi/ test/
```

Expected: pyright shows only the 2 pre-existing `pyphi/visualize/ces/geometry.py` errors as baseline. Ruff clean.

- [ ] **Step 5: Commit**

```bash
git add pyphi/metrics/protocols.py pyphi/metrics/distribution.py \
        pyphi/formalism/iit4/__init__.py test/test_metric_protocols.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Add metric Protocol types; drop INTRINSIC_DIFFERENTIATION q arg

Introduces three Protocol classes — DistributionMetric,
StateAwareMetric, CompositeMetric — typing the metric shape diversity
already present in the registry. Every registered metric satisfies
exactly one Protocol; a parametrized test pins the classification.

INTRINSIC_DIFFERENTIATION's vestigial q argument (unused since the P5
metric API discussions) is removed; the lone caller in iit4/__init__.py
is updated. The metric now cleanly satisfies StateAwareMetric.

Spec: docs/superpowers/specs/2026-05-10-measure-api-and-explicit-threading-design.md
Phase 1 of 6.
EOF
)"
```

---

## Phase 2 — Registry split + resolvers

**Goal:** Replace `pyphi/metrics/distribution.py::measures` (mixed registry) with three typed registries. Add `resolve_*_metric` helpers.

**Estimated:** 1 day.

### Task 2.1: Add typed registries alongside existing `measures`

**Files:**
- Modify: `pyphi/metrics/distribution.py` (registry definitions + new typed registries)

- [ ] **Step 1: Add the three typed registries**

After the existing `DistributionMeasureRegistry` definition at `pyphi/metrics/distribution.py:267`, add:

```python
# Three typed registries replacing the mixed `measures` registry. Each
# registered metric is validated at registration time against the
# relevant Protocol; signature drift becomes an import-time error.

class DistributionMetricRegistry(Registry):
    desc = "distribution metrics"

    def register(self, name: str, *, asymmetric: bool = False):
        def decorator(func):
            if not isinstance(func, DistributionMetric):
                raise TypeError(
                    f"{name!r} doesn't satisfy DistributionMetric Protocol"
                )
            func.asymmetric = asymmetric  # type: ignore[attr-defined]
            self.store[name] = func
            return func
        return decorator


class StateAwareMetricRegistry(Registry):
    desc = "state-aware metrics"

    def register(self, name: str):
        def decorator(func):
            if not isinstance(func, StateAwareMetric):
                raise TypeError(
                    f"{name!r} doesn't satisfy StateAwareMetric Protocol"
                )
            self.store[name] = func
            return func
        return decorator


class CompositeMetricRegistry(Registry):
    desc = "composite metrics"

    def register(self, name: str):
        def decorator(func):
            if not isinstance(func, CompositeMetric):
                raise TypeError(
                    f"{name!r} doesn't satisfy CompositeMetric Protocol"
                )
            self.store[name] = func
            return func
        return decorator


distribution_metrics = DistributionMetricRegistry()
state_aware_metrics = StateAwareMetricRegistry()
composite_metrics = CompositeMetricRegistry()
```

Add the Protocol imports at the top of the file:

```python
from pyphi.metrics.protocols import CompositeMetric
from pyphi.metrics.protocols import DistributionMetric
from pyphi.metrics.protocols import StateAwareMetric
```

- [ ] **Step 2: Migrate each `@measures.register` decorator to the typed registry**

For each `@measures.register(NAME, ...)` in `pyphi/metrics/distribution.py`, replace with the right typed registry's decorator. Use the classification from `test_metric_protocols.py`:

```python
# Before:
@measures.register("EMD")
def emd(p, q): ...

# After:
@distribution_metrics.register("EMD")
def emd(p, q): ...
```

```python
# Before:
@measures.register("INTRINSIC_DIFFERENTIATION")
def intrinsic_differentiation(p, state): ...

# After:
@state_aware_metrics.register("INTRINSIC_DIFFERENTIATION")
def intrinsic_differentiation(p, state): ...
```

```python
# Before:
@measures.register("GENERALIZED_INTRINSIC_DIFFERENCE", asymmetric=True)
@measures.register("INTRINSIC_SPECIFICATION", asymmetric=True)
def generalized_intrinsic_difference(...): ...

# After:
@composite_metrics.register("GENERALIZED_INTRINSIC_DIFFERENCE")
@composite_metrics.register("INTRINSIC_SPECIFICATION")
def generalized_intrinsic_difference(...): ...
```

(Composite metrics don't have `asymmetric` — that attribute is only on `DistributionMetric`.)

Apply the same pattern to all ~13 registered metrics.

- [ ] **Step 3: Run the protocol tests, now against the typed registries**

Update `test/test_metric_protocols.py` to assert against the typed registries instead of `measures`:

```python
# Before:
metric = distribution.measures[name]
assert isinstance(metric, DistributionMetric)

# After:
metric = distribution.distribution_metrics[name]
# isinstance check is now redundant (registry validates at registration),
# but keep it as a defense-in-depth assertion:
assert isinstance(metric, DistributionMetric)
```

```bash
uv run pytest test/test_metric_protocols.py -v 2>&1 | tail -20
```

Expected: all green.

### Task 2.2: Add resolver helpers

**Files:**
- Modify: `pyphi/metrics/distribution.py` (append resolver functions)

- [ ] **Step 1: Add `resolve_*_metric` helpers**

Append after the registry definitions:

```python
def resolve_mechanism_metric(name: str) -> StateAwareMetric | CompositeMetric:
    """Look up a metric usable at the mechanism level.

    Mechanism-level integration uses either a state-aware pointwise
    metric (mechanism phi via IIT_4.0_SMALL_PHI) or a composite metric
    (mechanism phi via GID at the partition layer).
    """
    if name in state_aware_metrics:
        return state_aware_metrics[name]
    if name in composite_metrics:
        return composite_metrics[name]
    available = sorted(set(state_aware_metrics) | set(composite_metrics))
    raise ValueError(
        f"Unknown mechanism metric {name!r}. Available: {available}"
    )


def resolve_system_metric(name: str) -> CompositeMetric:
    """Look up a metric usable at the system level."""
    if name in composite_metrics:
        return composite_metrics[name]
    raise ValueError(
        f"Unknown system metric {name!r}. "
        f"Available: {sorted(composite_metrics)}"
    )


def resolve_alpha_measure(name: str) -> DistributionMetric:
    """Look up a metric usable for actual-causation alpha."""
    if name in distribution_metrics:
        return distribution_metrics[name]
    raise ValueError(
        f"Unknown alpha measure {name!r}. "
        f"Available: {sorted(distribution_metrics)}"
    )
```

### Task 2.3: Delete `measures` mixed registry; update callers

**Files:**
- Modify: `pyphi/metrics/distribution.py` (remove old `measures` definition)
- Modify: `pyphi/metrics/ces.py` (per-metric `asymmetric` attribute read)
- Modify: `pyphi/visualize/distribution.py` (same)
- Modify: any other caller surfaced by grep

- [ ] **Step 1: Audit existing `measures` callers**

```bash
git grep -n "metrics\.distribution\.measures\|distribution\.measures\b\|from .*distribution.*import.*measures\b" pyphi/ test/
```

Each call site needs to migrate to one of:
- `distribution_metrics[name]` if the caller expects a distribution metric
- `state_aware_metrics[name]` if expecting state-aware
- `composite_metrics[name]` if expecting composite
- `resolve_*_metric(name)` if the scope is variable

- [ ] **Step 2: Migrate each caller**

For `pyphi/metrics/ces.py:83` (the `asymmetric()` check):

```python
# Before:
if config.formalism.iit.mechanism_phi_measure in distribution.measures.asymmetric():

# After:
metric = distribution.resolve_mechanism_metric(
    config.formalism.iit.mechanism_phi_measure
)
if isinstance(metric, DistributionMetric) and metric.asymmetric:
```

For `pyphi/visualize/distribution.py:154`:

```python
# Before:
if config.formalism.iit.mechanism_phi_measure not in [...some list of asymmetric measures...]:

# After:
metric = distribution.resolve_mechanism_metric(
    config.formalism.iit.mechanism_phi_measure
)
if not (isinstance(metric, DistributionMetric) and metric.asymmetric):
```

(Read the current code for exact form; the principle is replacing the registry-side `asymmetric()` query with per-metric attribute access.)

- [ ] **Step 3: Delete the `measures` mixed registry definition**

Remove the `DistributionMeasureRegistry` class definition at `pyphi/metrics/distribution.py:267-324` AND the line `measures = DistributionMeasureRegistry()`.

Also remove the redundant `ActualCausationMeasureRegistry` at line 327 if it's only used by AC and can be cleaner — but defer to Phase 6's AC parallel work. Leave it for now.

### Task 2.4: Add resolver tests, run gates, commit Phase 2

**Files:**
- Create: `test/test_metric_resolution.py`

- [ ] **Step 1: Write resolver tests**

```python
"""Resolver helpers: name → typed metric callable."""

from __future__ import annotations

import pytest

from pyphi.metrics.distribution import (
    composite_metrics,
    distribution_metrics,
    resolve_alpha_measure,
    resolve_mechanism_metric,
    resolve_system_metric,
    state_aware_metrics,
)
from pyphi.metrics.protocols import (
    CompositeMetric,
    DistributionMetric,
    StateAwareMetric,
)


def test_resolve_mechanism_metric_returns_state_aware() -> None:
    metric = resolve_mechanism_metric("IIT_4.0_SMALL_PHI")
    assert isinstance(metric, StateAwareMetric)


def test_resolve_mechanism_metric_returns_composite() -> None:
    metric = resolve_mechanism_metric("GENERALIZED_INTRINSIC_DIFFERENCE")
    assert isinstance(metric, CompositeMetric)


def test_resolve_system_metric_returns_composite() -> None:
    metric = resolve_system_metric("INTRINSIC_INFORMATION")
    assert isinstance(metric, CompositeMetric)


def test_resolve_system_metric_rejects_state_aware() -> None:
    with pytest.raises(ValueError, match="Unknown system metric"):
        resolve_system_metric("IIT_4.0_SMALL_PHI")


def test_resolve_alpha_measure_returns_distribution_metric() -> None:
    metric = resolve_alpha_measure("EMD")
    assert isinstance(metric, DistributionMetric)


def test_unknown_metric_name_raises_with_available_list() -> None:
    with pytest.raises(ValueError, match="Available:"):
        resolve_mechanism_metric("NONSENSE")
```

- [ ] **Step 2: Run gates**

Same six gates as Phase 1's Task 1.4. Pyright + ruff clean; goldens 17/17; fast lane green; perf 5/5.

- [ ] **Step 3: Commit**

```bash
git add pyphi/metrics/distribution.py pyphi/metrics/ces.py \
        pyphi/visualize/distribution.py test/test_metric_resolution.py \
        test/test_metric_protocols.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Split metric registry into three typed registries; add resolvers

Replaces the mixed `measures` registry with `distribution_metrics`,
`state_aware_metrics`, and `composite_metrics`. Each enforces its
Protocol at registration time — signature drift becomes an import-time
error rather than a silent runtime issue.

Adds `resolve_{mechanism,system}_metric` and `resolve_alpha_measure`
helpers for the user-facing class methods to use at the config-boundary
in later phases. They return typed callables and raise descriptively on
unknown names.

The `asymmetric()` registry-level query is replaced by per-metric
`asymmetric` attribute access on `DistributionMetric`. The two call
sites in ces.py and visualize/distribution.py are updated.

Spec: docs/superpowers/specs/2026-05-10-measure-api-and-explicit-threading-design.md
Phase 2 of 6.
EOF
)"
```

---

## Phase 3 — Thread internal helpers

**Goal:** Add explicit metric params to every helper below the formalism class boundary. Remove all `config.formalism.iit.*_phi_measure` reads from `pyphi/{core,metrics}/` and `pyphi/formalism/iit4/__init__.py`'s internal helpers.

**Estimated:** 2 days.

### Task 3.1: Thread `system_metric` through `_evaluate_partition` and `integration_value`

**Files:**
- Modify: `pyphi/formalism/iit4/__init__.py` (lines 411-535 area)

- [ ] **Step 1: Read the current signatures**

```bash
grep -n "^def \|def integration_value\|def evaluate_partition\|def _evaluate_partition" pyphi/formalism/iit4/__init__.py | head -10
```

Identify the three helpers: `integration_value` (around line 411), `evaluate_partition` (around line 487), and any others that read `system_phi_measure`.

- [ ] **Step 2: Add `system_metric` kwarg to `integration_value`**

Find at `pyphi/formalism/iit4/__init__.py:411`:

```python
def integration_value(
    direction: Direction,
    system: System,
    partition: ...,
    system_state: ...,
    repertoire_distance: str | None = None,
) -> RepertoireIrreducibilityAnalysis:
    repertoire_distance = fallback(
        repertoire_distance, config.formalism.iit.system_phi_measure
    )
    ...
```

Change to:

```python
def integration_value(
    direction: Direction,
    system: System,
    partition: ...,
    system_state: ...,
    *,
    system_metric: CompositeMetric,
) -> RepertoireIrreducibilityAnalysis:
    # No config read. system_metric is required.
    ...
```

Replace `repertoire_distance` string-based dispatch in the body with direct use of `system_metric` (the callable). Add the import:

```python
from pyphi.metrics.protocols import CompositeMetric
```

- [ ] **Step 3: Update `evaluate_partition` analogously**

At `pyphi/formalism/iit4/__init__.py:487`:

```python
def evaluate_partition(
    partition: ...,
    direction: Direction,
    system: System,
    system_state: ...,
    *,
    system_metric: CompositeMetric,
) -> ...:
    # Eqs. 19-20: system-level partition integration uses GID only.
    # The ii(s) cap (Eq. 23) is applied separately below.
    if system_metric.name == "INTRINSIC_INFORMATION":
        partition_metric = composite_metrics["GENERALIZED_INTRINSIC_DIFFERENCE"]
    else:
        partition_metric = system_metric

    integration = {
        direction: integration_value(
            direction, system, partition, system_state,
            system_metric=partition_metric,
        )
        for direction in directions
    }
    ...
    if system_metric.name == "INTRINSIC_INFORMATION":
        # Eq. 23 cap branch
        for direction in directions:
            i_spec = utils.positive_part(...)
            i_diff = utils.positive_part(...)
            phi = min(phi, i_spec, i_diff)
    ...
```

The branch keys off `system_metric.name`, NOT a config string. Imports:

```python
from pyphi.metrics.distribution import composite_metrics
```

- [ ] **Step 4: Update internal callers of `evaluate_partition` and `integration_value`**

Search:

```bash
git grep -n "evaluate_partition(\|integration_value(" pyphi/formalism/iit4/
```

Each caller now needs to pass `system_metric=...` explicitly. For now, leave the OUTER callers (the `_sia` function) reading config and passing it down — they'll be threaded in Task 3.2.

- [ ] **Step 5: Run goldens to confirm no regression**

```bash
uv run pytest test/test_golden_regression.py -k iit4_2026 -v 2>&1 | tail -10
```

Expected: 2026 fixtures pass. If they fail with wrong phi values: the cap branch keys off `system_metric.name == "INTRINSIC_INFORMATION"` — verify the metric being passed has that name.

### Task 3.2: Thread `system_metric` through `_sia` and `system_intrinsic_information`

**Files:**
- Modify: `pyphi/formalism/iit4/__init__.py` (lines 79, ~582 area)

- [ ] **Step 1: Update `system_intrinsic_information`**

At `pyphi/formalism/iit4/__init__.py:60-100` area:

```python
def system_intrinsic_information(
    system: System,
    *,
    specification_metric: CompositeMetric,
    directions: ... = None,
) -> SystemStateSpecification:
    """Return the cause/effect states specified by the system."""
    directions = fallback(directions, Direction.both())
    ...
    # No config read for specification_measure. Use specification_metric.
    ii = {
        direction: ... specification_metric(...)
        for direction in directions
    }
    ...
```

Drop the `config.formalism.iit.specification_measure` read at line 79.

- [ ] **Step 2: Update `_sia` (the implementation behind `pyphi.formalism.iit4.sia`)**

The function `sia` is at `pyphi/formalism/iit4/__init__.py:582`. Add `system_metric` and `specification_metric` kwargs:

```python
def sia(
    system: System,
    *,
    system_metric: CompositeMetric,
    specification_metric: CompositeMetric,
    directions: ... = None,
    partition_scheme: str | None = None,
    partitions: ... = None,
    system_state: SystemStateSpecification | None = None,
    **kwargs,
) -> SystemIrreducibilityAnalysis:
    ...
```

Required kwargs (no defaults) ensure callers can't accidentally fall back to config.

Inside the body, replace all `evaluate_partition` and `integration_value` calls to pass `system_metric=system_metric`. Replace the `system_intrinsic_information(system)` call to pass `specification_metric=specification_metric`.

- [ ] **Step 3: Update `phi_structure` (in the same module) similarly**

Find `def phi_structure` in `pyphi/formalism/iit4/__init__.py` and apply the same threading.

- [ ] **Step 4: Update all internal callers**

```bash
git grep -n "from pyphi\.formalism\.iit4 import sia\|formalism\.iit4\.sia(\|_sia(" pyphi/
```

Each internal caller (in `pyphi/`, not `test/`) needs to pass `system_metric` + `specification_metric` explicitly. Tests that import sia directly will be updated in Phase 5.

For now, the formalism class methods (`IIT4_2023Formalism.evaluate_system` at line 227) need to resolve from config and pass down. Temporary plumbing — Phase 4 will move the resolution to the formalism's `default_*_metric`.

```python
# Temporary plumbing in IIT4_2023Formalism.evaluate_system:
def evaluate_system(self, system, **kwargs) -> ...:
    check_metric_compatible(self, config.formalism.iit.system_phi_measure)
    system_metric = resolve_system_metric(config.formalism.iit.system_phi_measure)
    specification_metric = resolve_system_metric(
        config.formalism.iit.specification_measure
    )
    return _sia(
        system,
        system_metric=system_metric,
        specification_metric=specification_metric,
        **kwargs,
    )
```

Same pattern in IIT4_2026Formalism's `evaluate_system` (which currently uses `config.override`). Phase 4 will clean this up; for Phase 3 the goal is just "internal helpers don't read config".

### Task 3.3: Thread `mechanism_metric` through `_evaluate_partition_iit4` and `_find_mip_iit4`

**Files:**
- Modify: `pyphi/formalism/iit4/__init__.py` (search for `_evaluate_partition_iit4`, `_find_mip_iit4`)

- [ ] **Step 1: Add `mechanism_metric` to `_evaluate_partition_iit4`**

```python
def _evaluate_partition_iit4(
    system: System,
    direction: Direction,
    mechanism: ...,
    purview: ...,
    partition: ...,
    *,
    mechanism_metric: StateAwareMetric | CompositeMetric,
    **kwargs,
) -> RepertoireIrreducibilityAnalysis:
    # No config read. mechanism_metric is required.
    ...
```

- [ ] **Step 2: Add `mechanism_metric` to `_find_mip_iit4`**

Same pattern: required kwarg, no config read in body, replace internal config reads with `mechanism_metric` usage.

- [ ] **Step 3: Update callers in the formalism class methods**

`IIT4_2023Formalism._find_mechanism_mip` and `evaluate_mechanism_partition` (around lines 200, 220) now resolve and pass:

```python
def _find_mechanism_mip(self, system, direction, mechanism, purview, **kwargs):
    check_metric_compatible(self, config.formalism.iit.mechanism_phi_measure)
    mechanism_metric = resolve_mechanism_metric(
        config.formalism.iit.mechanism_phi_measure
    )
    return _find_mip_iit4(
        system, direction, mechanism, purview,
        mechanism_metric=mechanism_metric, **kwargs,
    )
```

### Task 3.4: Thread through `pyphi/core/repertoire_algebra.py`

**Files:**
- Modify: `pyphi/core/repertoire_algebra.py:297, 530`

- [ ] **Step 1: Identify the consuming functions**

```bash
grep -n "config\.formalism\.iit\." pyphi/core/repertoire_algebra.py
```

Two reads:
- Line 297: `config.formalism.iit.mechanism_phi_measure`
- Line 530: `config.formalism.iit.specification_measure`

For each, find the enclosing function and add the appropriate metric kwarg.

- [ ] **Step 2: Add kwargs and remove reads**

The function around line 297 likely takes a `repertoire_distance: str | None = None` parameter today. Change to:

```python
def some_function(
    ...,
    *,
    mechanism_metric: StateAwareMetric | CompositeMetric,
) -> ...:
    # Was: repertoire_distance = fallback(repertoire_distance, config.formalism.iit.mechanism_phi_measure)
    # Now: mechanism_metric is required; use directly.
    ...
```

Same for line 530 with `specification_metric: CompositeMetric`.

- [ ] **Step 3: Update callers in iit4/__init__.py and elsewhere**

`git grep` for the affected function names; thread the metric param through.

### Task 3.5: Verify no internal config reads remain; run gates; commit Phase 3

- [ ] **Step 1: Audit for residual measure-config reads**

```bash
git grep -n "config\.formalism\.iit\.\(mechanism\|system\)_phi_measure\|config\.formalism\.iit\.\(specification\|differentiation\)_measure" pyphi/ | grep -v "system\.py\|subsystem\.py"
```

Expected: only matches in `pyphi/formalism/iit4/formalism.py` (formalism class methods — Phase 4 cleans these up) and possibly `pyphi/formalism/iit3/formalism.py` (Phase 4 same).

If matches appear in `pyphi/core/`, `pyphi/metrics/`, or `pyphi/formalism/iit4/__init__.py`: a read was missed; locate and thread.

- [ ] **Step 2: Goldens, fast lane, perf budget, pyright, ruff** (same six gates)

```bash
uv run pytest test/test_golden_regression.py 2>&1 | tail -5
uv run pytest test/ -m "not slow" -q --no-header 2>&1 | tail -5
uv run pytest test/ -m perf -q 2>&1 | tail -5
uv run pyright pyphi/ 2>&1 | tail -3
uv run ruff check pyphi/ test/
uv run ruff format --check pyphi/ test/
```

- [ ] **Step 3: Commit**

```bash
git add pyphi/formalism/iit4/__init__.py pyphi/formalism/iit4/formalism.py \
        pyphi/core/repertoire_algebra.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Thread metric params through internal helpers; remove config reads

Every helper below the formalism class boundary now receives its
metric(s) as explicit parameters. The reads of
config.formalism.iit.{mechanism,system}_phi_measure and
config.formalism.iit.{specification,differentiation}_measure that
lived inside `_evaluate_partition`, `integration_value`, `_sia`,
`system_intrinsic_information`, `_evaluate_partition_iit4`,
`_find_mip_iit4`, and the helpers in core/repertoire_algebra.py are
all gone.

Formalism class methods temporarily resolve from config at the
formalism-method boundary and pass down (Phase 4 will move resolution
to the formalism's default_*_metric ClassVars).

The `if measure == "INTRINSIC_INFORMATION"` string dispatch in
`_evaluate_partition` becomes `if system_metric.name == "..."`, keyed
off the metric object itself rather than a config string.

Spec: docs/superpowers/specs/2026-05-10-measure-api-and-explicit-threading-design.md
Phase 3 of 6.
EOF
)"
```

---

## Phase 4 — Thread formalism class methods

**Goal:** Add `*, mechanism_metric=None, system_metric=None` kwargs to all formalism class methods. Resolve from `self.default_*_metric` when not given. Delete the 5 `config.override` blocks in `IIT4_2026Formalism`.

**Estimated:** 1.5 days.

### Task 4.1: Write failing tests pinning the new contract

**Files:**
- Create: `test/test_formalism_metric_threading.py`

- [ ] **Step 1: Write the tests**

```python
"""Formalism class methods accept explicit metric kwargs.

Pin that passing a non-default metric to `evaluate_system` etc.
actually uses that metric, with no implicit override from
`default_system_metric`.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyphi import Substrate, System
from pyphi.formalism.iit4.formalism import IIT4_2026Formalism
from pyphi.metrics.distribution import (
    composite_metrics,
    resolve_system_metric,
)


@pytest.fixture
def noisy_copy_system():
    """2-node noisy COPY, p=0.8, state (1,1).

    Same fixture as TestEq23IntrinsicInformationCap so we can compare
    against the same i_diff ≈ 0.644 / GID(MIP) ≈ 0.868 ground truth.
    """
    p = 0.8
    tpm = np.array(
        [[1 - p, 1 - p], [1 - p, p], [p, 1 - p], [p, p]]
    )
    cm = np.array([[0, 1], [1, 0]])
    substrate = Substrate(tpm, cm=cm, node_labels=["A", "B"])
    return System(substrate, (1, 1))


def test_evaluate_system_uses_passed_metric_not_default(noisy_copy_system):
    """Passing GID explicitly to 2026 formalism's evaluate_system
    should produce GID(MIP) (no cap) — the formalism's
    default_system_metric of INTRINSIC_INFORMATION must not override.
    """
    formalism = IIT4_2026Formalism()
    gid_metric = resolve_system_metric("GENERALIZED_INTRINSIC_DIFFERENCE")

    result = formalism.evaluate_system(
        noisy_copy_system,
        system_metric=gid_metric,
        specification_metric=composite_metrics["INTRINSIC_SPECIFICATION"],
    )

    # With explicit GID, no cap applies → phi = GID(MIP) ≈ 0.868
    assert float(result.phi) == pytest.approx(0.868, abs=0.001)


def test_evaluate_system_resolves_from_default_when_omitted(
    noisy_copy_system,
):
    """Calling without explicit metrics uses the formalism's
    default_*_metric — INTRINSIC_INFORMATION for 2026 → cap fires.
    """
    formalism = IIT4_2026Formalism()
    result = formalism.evaluate_system(noisy_copy_system)
    # With default INTRINSIC_INFORMATION, cap fires → phi ≈ 0.644
    assert float(result.phi) == pytest.approx(0.644, abs=0.001)
```

- [ ] **Step 2: Run — expect failures**

```bash
uv run pytest test/test_formalism_metric_threading.py -v 2>&1 | tail -15
```

Expected: failures. The formalism methods don't yet accept `system_metric=` / `specification_metric=` kwargs.

### Task 4.2: Add metric kwargs to formalism class methods; delete config.override blocks

**Files:**
- Modify: `pyphi/formalism/iit4/formalism.py` (lines 179-306)
- Modify: `pyphi/formalism/iit3/formalism.py` (parallel methods)

- [ ] **Step 1: Update IIT4_2023Formalism**

Each method gains `*, mechanism_metric=None, system_metric=None, specification_metric=None` (use only the ones relevant to that method's scope):

```python
def evaluate_system(
    self,
    system: Any,
    *,
    system_metric: CompositeMetric | None = None,
    specification_metric: CompositeMetric | None = None,
    **kwargs: Any,
) -> Any:
    system_metric = system_metric or composite_metrics[self.default_system_metric]
    specification_metric = (
        specification_metric
        or composite_metrics["INTRINSIC_SPECIFICATION"]  # paper-faithful default
    )
    check_metric_compatible(self, system_metric.name)
    return _sia(
        system,
        system_metric=system_metric,
        specification_metric=specification_metric,
        **kwargs,
    )
```

Apply to all five methods: `evaluate_mechanism`, `_find_mechanism_mip`, `evaluate_mechanism_partition`, `evaluate_system`, `build_phi_structure`.

- [ ] **Step 2: Update IIT4_2026Formalism — delete the 5 config.override blocks**

At lines 266, 278, 293, 300, 305 the pattern `with config.override(...):` exists. Replace each with the explicit-kwarg pattern. Example for `evaluate_system` at line 298-301:

```python
# Before:
def evaluate_system(self, system, **kwargs):
    check_metric_compatible(self, config.formalism.iit.system_phi_measure)
    with config.override(system_phi_measure=self.default_system_metric):
        return _sia(system, **kwargs)

# After:
def evaluate_system(
    self,
    system,
    *,
    system_metric: CompositeMetric | None = None,
    specification_metric: CompositeMetric | None = None,
    **kwargs,
):
    system_metric = system_metric or composite_metrics[self.default_system_metric]
    specification_metric = (
        specification_metric or composite_metrics["INTRINSIC_SPECIFICATION"]
    )
    check_metric_compatible(self, system_metric.name)
    return _sia(
        system,
        system_metric=system_metric,
        specification_metric=specification_metric,
        **kwargs,
    )
```

All 5 `config.override` blocks become `or`-fallback resolutions from `self.default_*_metric`.

- [ ] **Step 3: Update IIT3Formalism**

Symmetric pattern in `pyphi/formalism/iit3/formalism.py`. Each method gains explicit metric kwargs; `default_mechanism_metric` and `default_system_metric` (both `"EMD"`) drive the fallback.

- [ ] **Step 4: Verify config.override blocks are gone**

```bash
git grep -n "config\.override.*\(mechanism\|system\)_phi_measure\|config\.override.*measure" pyphi/
```

Expected: zero matches in `pyphi/`. (Tests may still have these — they're fine; tests use config.override for setup.)

### Task 4.3: Update System class methods to resolve at boundary, run gates, commit Phase 4

**Files:**
- Modify: `pyphi/system.py` and/or `pyphi/subsystem.py` (the user-facing class methods)

- [ ] **Step 1: Find the System.sia / find_mip / phi_structure methods**

```bash
grep -n "def sia\|def find_mip\|def phi_structure" pyphi/system.py pyphi/subsystem.py
```

- [ ] **Step 2: Resolve at boundary**

Each public class method reads config, resolves to a metric object, and passes through:

```python
def sia(self, **kwargs):
    formalism = FORMALISM_REGISTRY[config.formalism.iit.version]
    return formalism.evaluate_system(self, **kwargs)
```

Note: `System.sia` doesn't need to resolve metrics itself — it can delegate to the formalism's evaluate_system, which now defaults to `self.default_*_metric` if no metric is passed. The cleaner approach:

- `System.*` reads `config.formalism.iit.version` to pick formalism (unchanged).
- The formalism's evaluate_X method reads config to resolve the metric if not passed, falling back to `self.default_*_metric`.

Wait — that re-introduces config reads inside formalism methods. Per the spec, config reads should happen at System.* only. Let me reconsider.

**Correct shape:** `System.sia()` reads config to get the metric, then passes to `formalism.evaluate_system(system_metric=...)`. Formalism methods themselves don't read config — they receive the metric or fall back to their `default_*_metric` ClassVar (a constant, no config involvement).

```python
def sia(self, **kwargs):
    formalism = FORMALISM_REGISTRY[config.formalism.iit.version]
    system_metric = resolve_system_metric(
        config.formalism.iit.system_phi_measure
    )
    specification_metric = composite_metrics[
        config.formalism.iit.specification_measure
    ]
    return formalism.evaluate_system(
        self,
        system_metric=system_metric,
        specification_metric=specification_metric,
        **kwargs,
    )
```

Same for `System.find_mip` (mechanism scope), `System.phi_structure` (system scope).

- [ ] **Step 3: Run gates** (same six)

Expected: Test 4.1's two tests now pass. Goldens green. Fast lane green. Perf 5/5. Pyright + ruff clean.

If any of these fail: the threading is incomplete or the metric-resolution at boundary is wrong. Diagnose before reverting.

- [ ] **Step 4: Commit Phase 4**

```bash
git add pyphi/formalism/iit4/formalism.py pyphi/formalism/iit3/formalism.py \
        pyphi/system.py pyphi/subsystem.py \
        test/test_formalism_metric_threading.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Thread metrics through formalism class methods; delete config.override blocks

Formalism class methods (IIT4_2023Formalism, IIT4_2026Formalism,
IIT3Formalism) now accept explicit `mechanism_metric`,
`system_metric`, and `specification_metric` kwargs. When unset they
fall back to `self.default_*_metric` ClassVars (constants, no config).

The 5 `config.override(...)` blocks in IIT4_2026Formalism (at lines
266, 278, 293, 300, 305 pre-refactor) are deleted. The pattern that
enabled today's cap regression class of bug is gone from pyphi/.

System.{sia,find_mip,phi_structure} now resolve config-driven metric
names to typed callables at the public boundary and pass them down
to the formalism layer.

Spec: docs/superpowers/specs/2026-05-10-measure-api-and-explicit-threading-design.md
Phase 4 of 6.
EOF
)"
```

---

## Phase 5 — Thread module-level functions + cap-regression-impossible test

**Goal:** Make `pyphi.formalism.iit4.{sia, find_mip, phi_structure}` require explicit metric kwargs (no config fallback). Update tests that import them directly. Add the cap-regression-impossible test as the headline validation.

**Estimated:** 1.5 days.

### Task 5.1: Make module-level functions require explicit metrics

**Files:**
- Modify: `pyphi/formalism/iit4/__init__.py` (the public `sia`, `find_mip`, `phi_structure`)

- [ ] **Step 1: Update signatures to make metrics required**

Currently the module-level `sia` (at line 582) has `**kwargs` and could accept metrics via the threading from Phase 3. Now make them explicit and required (no default):

```python
def sia(
    system: System,
    *,
    system_metric: CompositeMetric,
    specification_metric: CompositeMetric,
    directions: ... = None,
    ...
) -> SystemIrreducibilityAnalysis:
    ...
```

No `config.formalism.iit.system_phi_measure` read. The function raises `TypeError` (Python's built-in) if a caller omits the required kwargs.

Same for `find_mip` and `phi_structure`.

- [ ] **Step 2: Verify no caller in pyphi/ relies on the implicit-from-config behavior**

```bash
git grep -n "from pyphi.formalism.iit4 import" pyphi/
git grep -n "formalism\.iit4\.\(sia\|find_mip\|phi_structure\)(" pyphi/
```

Each internal caller should already be passing explicit metrics after Phase 3 / 4. If any is missed: thread it now.

### Task 5.2: Update tests importing raw module-level functions

**Files:**
- Modify: `test/test_big_phi_robust.py:441, 474, 494`
- Possibly: other test files surfaced by grep

- [ ] **Step 1: Audit tests importing raw module-level functions**

```bash
git grep -n "from pyphi.formalism.iit4 import \(sia\|find_mip\|phi_structure\)\|formalism\.iit4\.\(sia\|find_mip\|phi_structure\)(" test/
```

- [ ] **Step 2: Update each call site to pass explicit metrics**

Example for `test/test_big_phi_robust.py:441-461` (`test_phi_capped_by_ii`):

```python
# Before:
from pyphi.formalism.iit4 import sia
...
with config.override(**self.II_CONFIG):  # II_CONFIG sets system_phi_measure=INTRINSIC_INFORMATION
    result = sia(system)

# After:
from pyphi.formalism.iit4 import sia
from pyphi.metrics.distribution import resolve_system_metric, composite_metrics
...
system_metric = resolve_system_metric("INTRINSIC_INFORMATION")
specification_metric = composite_metrics["INTRINSIC_SPECIFICATION"]
result = sia(
    system,
    system_metric=system_metric,
    specification_metric=specification_metric,
)
```

The `config.override` block is no longer needed for these tests since they're passing the metric directly. (Other tests that DO test config-reading behavior at the boundary should still use config.override.)

Apply the same pattern to the other two raw-import sites at lines 474 and 494.

### Task 5.3: Add the cap-regression-impossible test

**Files:**
- Create: `test/test_cap_regression_impossible.py`

- [ ] **Step 1: Write the headline validation test**

```python
"""The cap-regression class of bug is impossible by construction.

Today's session caught a bug where setting `mechanism_phi_measure`
config to `INTRINSIC_INFORMATION` silently failed to activate the
Eq. 23 ii(s) cap (the cap is gated on `system_phi_measure`, not
`mechanism_phi_measure`). The fix in commit 85d8e029 was test-side;
this refactor makes the underlying architecture immune.

This test pins the new invariant: raw module-level `sia()` requires
explicit metric kwargs. Setting `mechanism_phi_measure` config and
calling raw `sia()` without explicit metrics produces a TypeError
(missing kwarg), not a silent wrong-scope read.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyphi import Substrate, System, config
from pyphi.formalism.iit4 import sia


@pytest.fixture
def noisy_copy_system():
    p = 0.8
    tpm = np.array([[1 - p, 1 - p], [1 - p, p], [p, 1 - p], [p, p]])
    cm = np.array([[0, 1], [1, 0]])
    substrate = Substrate(tpm, cm=cm, node_labels=["A", "B"])
    return System(substrate, (1, 1))


def test_raw_sia_requires_explicit_metric(noisy_copy_system):
    """The original cap-regression bug: setting config.mechanism_phi_measure
    to INTRINSIC_INFORMATION and calling raw sia() used to silently miss
    the cap. Now it raises because system_metric is required.
    """
    from dataclasses import replace

    with config.override(
        iit=replace(
            config.formalism.iit,
            mechanism_phi_measure="INTRINSIC_INFORMATION",
        )
    ):
        with pytest.raises(TypeError, match="system_metric"):
            sia(noisy_copy_system)
```

The override syntax may need adjustment for the current FormalismConfig API — verify against the current shape (see `pyphi/conf/formalism.py::IITConfig`).

- [ ] **Step 2: Run the test — expect green**

```bash
uv run pytest test/test_cap_regression_impossible.py -v 2>&1 | tail -10
```

Expected: pass. If it fails because `sia()` still falls back to config: Phase 5 / Task 5.1 didn't fully remove the fallback.

### Task 5.4: Run gates and commit Phase 5

- [ ] **Step 1: Six gates** (same as previous phases)

- [ ] **Step 2: Verify the architectural invariant**

```bash
git grep -n "config\.formalism\.iit\.\(mechanism\|system\)_phi_measure\|config\.formalism\.iit\.\(specification\|differentiation\)_measure" pyphi/
```

Expected output: only matches in `pyphi/system.py` / `pyphi/subsystem.py` (the public boundary). No internal helper or formalism class method reads measure config.

- [ ] **Step 3: Commit**

```bash
git add pyphi/formalism/iit4/__init__.py test/test_big_phi_robust.py \
        test/test_cap_regression_impossible.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Require explicit metrics on raw sia/find_mip/phi_structure; pin invariant

pyphi.formalism.iit4.{sia, find_mip, phi_structure} now require
explicit `system_metric` / `mechanism_metric` / `specification_metric`
kwargs. No config fallback. Callers must declare which metric they
want.

The three test sites in test_big_phi_robust.py that imported raw `sia`
are updated to pass explicit metrics — the TestEq23IntrinsicInformationCap
tests now read exactly what they test.

A new test_cap_regression_impossible.py pins the architectural
invariant: setting the wrong-scope config field and calling raw `sia()`
without explicit metrics raises TypeError rather than silently
observing a wrong scope. This is the headline validation of the
refactor's "impossible by construction" claim.

Spec: docs/superpowers/specs/2026-05-10-measure-api-and-explicit-threading-design.md
Phase 5 of 6.
EOF
)"
```

---

## Phase 6 — Actual Causation parallel

**Goal:** Apply the same threading pattern to AC measures (`alpha_measure`, `background_scheme`, `alpha_aggregation`, `partitioned_repertoire_scheme`). Revisit `ConfigSnapshot.as_kwargs` colliding-field workaround.

**Estimated:** 1.5 days.

### Task 6.1: Audit AC measure usage sites

**Files:**
- Read-only audit of `pyphi/actual.py`, `pyphi/metrics/distribution.py` (AC sections)

- [ ] **Step 1: Find AC config reads**

```bash
git grep -n "config\.formalism\.actual_causation\." pyphi/
```

Expected: reads of `alpha_measure`, `background_scheme`, `alpha_aggregation`, `partitioned_repertoire_scheme`, plus any others (e.g., `mechanism_partition_scheme`).

- [ ] **Step 2: Find AC public entry points and internal helpers**

```bash
grep -n "def alpha\|def evaluate\|class Transition" pyphi/actual.py
```

Identify the public-boundary methods (`Transition.alpha`, `Transition.account`, etc.) and the internal helpers that read AC config.

### Task 6.2: Thread AC metrics through helpers

**Files:**
- Modify: `pyphi/actual.py` (multiple sites)

- [ ] **Step 1: Add explicit AC metric kwargs to each internal helper**

Pattern (one helper at a time):

```python
# Before:
def probability_distance(p, q):
    metric = actual_causation_measures[config.formalism.actual_causation.alpha_measure]
    return metric(p, q)

# After:
def probability_distance(p, q, *, alpha_metric: DistributionMetric):
    return alpha_metric(p, q)
```

Update callers within `pyphi/actual.py` to pass `alpha_metric=...` (resolved from config at the `Transition.alpha` boundary).

Apply the same pattern for:
- `background_scheme` consumers (the strategy for handling unspecified inputs)
- `alpha_aggregation` consumers (how per-direction alphas combine)
- `partitioned_repertoire_scheme` consumers

Each becomes an explicit kwarg in the relevant helper signature.

- [ ] **Step 2: Update `Transition.alpha` (and parallel public methods) to resolve and pass**

```python
def alpha(self) -> ActualCause:
    alpha_metric = resolve_alpha_measure(
        config.formalism.actual_causation.alpha_measure
    )
    background_scheme = config.formalism.actual_causation.background_scheme
    aggregation = config.formalism.actual_causation.alpha_aggregation
    return self._compute_alpha(
        alpha_metric=alpha_metric,
        background_scheme=background_scheme,
        aggregation=aggregation,
    )
```

### Task 6.3: Revisit ConfigSnapshot workaround; run gates; commit Phase 6

**Files:**
- Modify: `pyphi/conf/snapshot.py` (revisit colliding-field exclusion logic)

- [ ] **Step 1: Check whether the colliding-field workaround is still needed**

The workaround at `pyphi/conf/snapshot.py:60` (the `as_kwargs` method) exists because `mechanism_partition_scheme` collides between IIT and AC sub-namespaces. With explicit threading, this collision may now be irrelevant — the formalism layer no longer round-trips through flat config kwargs.

```bash
grep -n "colliding\|exclude" pyphi/conf/snapshot.py
```

If the colliding-field logic is still load-bearing: leave it. If it's no longer needed (no code path round-trips through it post-threading): simplify or delete.

- [ ] **Step 2: Run gates** (same six)

The AC test suite (`test/test_actual.py::TestActualCausationIIT30` and related) should pass green. If any IIT 3.0 AC test fails: the explicit-threading missed a path; trace and fix.

- [ ] **Step 3: Commit**

```bash
git add pyphi/actual.py pyphi/conf/snapshot.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Thread Actual Causation metrics through internal helpers

Mirrors the IIT-side threading from Phases 3-5 for AC: alpha_measure,
background_scheme, alpha_aggregation, and partitioned_repertoire_scheme
are now explicit kwargs on every internal helper. `Transition.alpha`
and parallel public methods resolve from config at the boundary; below
that, no config reads.

Revisits the ConfigSnapshot.as_kwargs colliding-field exclusion
workaround for `mechanism_partition_scheme` — [removed / simplified /
retained, fill in based on observed state during Phase 6 audit].

Spec: docs/superpowers/specs/2026-05-10-measure-api-and-explicit-threading-design.md
Phase 6 of 6.
EOF
)"
```

---

## Final acceptance (after all six phases)

- [ ] **Cap-regression-impossible test green:** `uv run pytest test/test_cap_regression_impossible.py -v` → 1 passed.
- [ ] **Goldens 17/17:** `uv run pytest test/test_golden_regression.py` → 17 passed.
- [ ] **Fast lane zero failures:** `uv run pytest test/ -m "not slow"` → 0 failed.
- [ ] **Hypothesis fast lane green.**
- [ ] **Perf budget intact:** `uv run pytest test/ -m perf` → 5/5 pass.
- [ ] **Pyright clean baseline:** `uv run pyright pyphi/` → only the 2 pre-existing `pyphi/visualize/ces/geometry.py` errors.
- [ ] **Ruff clean:** `uv run ruff check pyphi/ test/` and `uv run ruff format --check pyphi/ test/` → both clean.
- [ ] **Zero internal config reads of measure fields:**

```bash
git grep -n "config\.formalism\.iit\.\(mechanism\|system\)_phi_measure\|config\.formalism\.iit\.\(specification\|differentiation\)_measure" pyphi/
```

Expected: only matches in `pyphi/system.py` / `pyphi/subsystem.py` (public-boundary class methods).

- [ ] **Zero `config.override(measure=...)` wrappers:**

```bash
git grep -n "config\.override(.*\(measure\|phi_measure\)" pyphi/
```

Expected: zero matches in `pyphi/` (test files may still use them for test setup — that's fine).

- [ ] **No planning artifacts in source:**

```bash
git grep -iE 'P[0-9]+\.[0-9]+|Tier [0-9]|ROADMAP|Phase [A-Z]\b' pyphi/ test/test_metric_protocols.py test/test_metric_resolution.py test/test_formalism_metric_threading.py test/test_cap_regression_impossible.py
```

Expected: zero matches.

---

## If something goes wrong

**Pyright failures mid-phase** — Run `uv run pyright pyphi/` directly to see the full error list. Common shapes:
- Missing import after adding a Protocol type → add the import.
- Protocol member not declared → add `name: str` ClassVar.
- Optional kwarg without default → set `= None` and resolve in body.
- Type narrowing failure with `or` fallback → use explicit `if ... is None` check.

**Goldens fail with wrong phi values** — A metric got threaded with the wrong name, or the `if metric.name == "X"` dispatch didn't update. Trace:
1. Identify which fixture failed.
2. Re-run that fixture with `-v -s` to see captured logs.
3. Add a temporary `print(system_metric.name)` at the cap-branch site in `_evaluate_partition`. Re-run. Confirm it matches expectations.
4. Remove the print, fix the propagation.

**Hypothesis fast lane reveals an invariant break** — A subtle behavior change snuck in. Read the falsifying example; trace back to which helper changed semantics. The cause is usually a default-metric fallback now resolving to a different name than the pre-refactor config-string path.

**Perf budget tripped** — Likely from extra Python attribute lookup overhead. If it's catastrophic (>4x typical), profile with `cProfile`. If it's marginal (~1.2x), bump the budget once and document.

**The cap-regression-impossible test passes but a different bug surfaces** — Today's session's audit was narrow. There may be other cross-scope bugs that the explicit threading reveals. Failing tests under a tightened contract are signal, not noise. Diagnose root cause before reverting.

**A phase commit gets bundled with another agent's staged work** (today's scenario) — Don't `--no-verify`. Either wait for the other session to land, or commit their work first under a proper attribution message, then commit yours. The reset-soft pattern (`git reset --soft HEAD^`) cleanly undoes if a bundled commit lands mistakenly.
