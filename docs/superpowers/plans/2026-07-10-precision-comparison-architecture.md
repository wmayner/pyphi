# Precision-Comparison Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete `PyPhiFloat`, consolidate all tolerant φ/Φ/α comparison at decision sites (`pyphi.numerics` scalar predicates + rebuilt `resolve_ties` cascades), fixing the eight B22 audit bugs.

**Architecture:** Strangler migration in four phases: (1) `pyphi.numerics` lands as a pure move; (2) decision sites convert subsystem-by-subsystem with goldens green between steps; (3) `PyPhiFloat` deleted, `DistanceResult` re-based on `float`, serialization schema changed; (4) AST lint gate on. Values are plain exact floats; tolerance lives only in `pyphi/numerics.py` (scalars) and `pyphi/resolve_ties.py` (object selection via tolerant lexicographic clustering).

**Tech Stack:** Python 3.13+, numpy, Hypothesis, pytest, msgspec.

**Spec:** `docs/superpowers/specs/2026-07-10-precision-comparison-architecture-design.md`

## Global Constraints

- Work in the worktree `.claude/worktrees/precision-architecture` (branch `precision-architecture`). All paths below are relative to the worktree root.
- **Worktree venv caveat:** `uv run` uses the worktree `.venv`, but bare `uv pip install` targets the main repo's venv via `.envrc`'s `VIRTUAL_ENV`. If dependencies are missing, install with: `env -u VIRTUAL_ENV uv pip install --python "$(uv run python -c 'import sys; print(sys.executable)')" -e ".[dev,parallel,visualize,emd,caching]"`.
- Run all Python via `uv run` (e.g. `uv run pytest`, `uv run python`).
- **Golden policy:** goldens must stay byte-identical after every task, EXCEPT these enumerated drift categories: (1) AC MIP / causal-nexus selection, (2) macro tie ordering / `is_maximal` labeling among φ-tied disjoint candidates, (3) tie-set membership gaining precision-tied members, (4) the ii-differentiation value where a certain-node surprisal artifact was selected. Any other drift is a stop-the-line bug: halt and report.
- Docstrings: NumPy style, final-state impersonal voice, Unicode symbols (`φ`, `Φ`, `α`), paper citations verified against `papers/` (never from memory). No planning artifacts (no "B22", "Phase 2", "per ROADMAP", "replaces PyPhiFloat") in source, comments, docstrings, or changelog fragments.
- Never use `git commit --no-verify`. If a pre-commit hook fails, fix the failure.
- Python 3.13+ only; no backward-compatibility shims of any kind.
- The full-suite verification command is `uv run pytest` with **no path argument** (this includes the `pyphi/` doctest sweep). Bare-path invocations skip doctests.

---

### Task 1: Create `pyphi/numerics.py` with tests

**Files:**
- Create: `pyphi/numerics.py`
- Create: `test/test_numerics.py`

**Interfaces:**
- Consumes: `pyphi.conf.config` (reads `config.numerics.precision` at call time).
- Produces: `numerics.eq(x, y) -> bool`, `numerics.is_zero(x) -> bool`, `numerics.is_positive(x) -> bool`, `numerics.is_nonpositive(x) -> bool`, `numerics.positive_mask(array) -> np.ndarray[bool]`, `numerics.round_to_precision(x) -> float`. Every later task imports these.

- [ ] **Step 1: Write the failing tests**

Create `test/test_numerics.py`:

```python
"""Tests for the tolerant scalar predicates in ``pyphi.numerics``."""

import numpy as np
import pytest

import pyphi
from pyphi import numerics


TOL = 10 ** (-13)  # default config.numerics.precision == 13


def test_eq_within_tolerance():
    assert numerics.eq(0.5, 0.5 + 1e-14)
    assert numerics.eq(0.5, 0.5 - 1e-14)
    assert not numerics.eq(0.5, 0.5 + 1e-12)


def test_eq_reads_config_at_call_time():
    with pyphi.config.override(precision=6):
        assert numerics.eq(0.5, 0.5 + 1e-9)
    assert not numerics.eq(0.5, 0.5 + 1e-9)


def test_is_zero():
    assert numerics.is_zero(0.0)
    assert numerics.is_zero(3.2e-16)   # summation-noise residue
    assert numerics.is_zero(-3.2e-16)
    assert not numerics.is_zero(1e-12)


def test_is_positive():
    assert numerics.is_positive(0.5)
    assert not numerics.is_positive(0.0)
    assert not numerics.is_positive(3.2e-16)   # noise is not positive
    assert not numerics.is_positive(-0.5)


def test_is_nonpositive():
    assert numerics.is_nonpositive(-0.5)
    assert numerics.is_nonpositive(0.0)
    assert not numerics.is_nonpositive(0.5)


def test_positive_mask_matches_elementwise_is_positive():
    rng = np.random.default_rng(20260710)
    values = np.concatenate([
        rng.uniform(-1, 1, 100),
        np.array([0.0, 3.2e-16, -3.2e-16, 1e-13, 1e-12, -1e-12]),
    ])
    mask = numerics.positive_mask(values)
    expected = np.array([numerics.is_positive(v) for v in values])
    np.testing.assert_array_equal(mask, expected)


def test_positive_mask_certain_node_surprisal():
    # -log2 of a float-noise near-1 probability is ~3e-16: mathematically
    # zero surprisal, must be masked out.
    surprisal = -np.log2(np.array([0.9999999999999998, 0.5, 0.25]))
    masked = surprisal[numerics.positive_mask(surprisal)]
    np.testing.assert_allclose(masked, [1.0, 2.0])


def test_round_to_precision():
    assert numerics.round_to_precision(0.5 + 4e-15) == round(0.5 + 4e-15, 13)
    with pyphi.config.override(precision=6):
        assert numerics.round_to_precision(0.1234567891) == pytest.approx(0.123457)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/test_numerics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyphi.numerics'` (or ImportError at collection).

- [ ] **Step 3: Write the implementation**

Create `pyphi/numerics.py`:

```python
# numerics.py
"""Tolerant scalar comparison of φ, Φ, and α values.

Floating-point results that are mathematically equal can differ by
roughly 1e-15 when computed through different code paths: distinct
algebraic routes to the same value produce different bit patterns, and
summation order is not associative. Integrated-information theory
treats ties between candidates (partitions, purviews, states, systems)
as meaningful, so detecting them requires comparison up to a tolerance
rather than exact equality.

These predicates are the only tolerant scalar comparisons in the
library. Values themselves are plain :class:`float`\\ s with exact
comparison semantics; tolerance applies where a comparison decides an
outcome. Selection among competing φ-objects goes through
:mod:`pyphi.resolve_ties`, which clusters candidates with :func:`eq`.

The tolerance is ``10**-precision`` with ``precision`` read from
``config.numerics.precision`` at call time (default 13, roughly two
orders of magnitude above the observed noise floor and far below
genuine φ differences).
"""

import math

import numpy as np

from .conf import config


def _epsilon() -> float:
    return 10 ** (-int(config.numerics.precision))


def eq(x: float, y: float) -> bool:
    """Return whether two values are equal up to ``config.numerics.precision``."""
    epsilon = _epsilon()
    return math.isclose(x, y, rel_tol=epsilon, abs_tol=epsilon)


def is_zero(x: float) -> bool:
    """Return whether ``x`` is zero up to ``config.numerics.precision``."""
    return eq(x, 0.0)


def is_positive(x: float) -> bool:
    """Return whether ``x`` is positive up to ``config.numerics.precision``."""
    # Need ``bool`` to cast from numpy to native Boolean
    return not eq(x, 0) and bool(x > 0)


def is_nonpositive(x: float) -> bool:
    """Return whether ``x`` is nonpositive (exact)."""
    # Need ``bool`` to cast from numpy to native Boolean
    return bool(x <= 0)


def positive_mask(array: np.ndarray) -> np.ndarray:
    """Return a boolean mask of the elements positive up to
    ``config.numerics.precision``.

    Equivalent to applying :func:`is_positive` elementwise. Values within
    the tolerance of zero (for example, the surprisal ``-log2(p)`` of a
    probability that is 1 up to floating-point noise) are masked out.
    """
    epsilon = _epsilon()
    a = np.asarray(array)
    return (a > 0) & ~np.isclose(a, 0.0, rtol=epsilon, atol=epsilon)


def round_to_precision(x: float) -> float:
    """Return ``x`` rounded to ``config.numerics.precision`` decimal places."""
    return round(x, int(config.numerics.precision))
```

Note on `positive_mask` equivalence: for comparison against zero, `math.isclose(x, 0, rel_tol=e, abs_tol=e)` reduces to `|x| <= e`, and `np.isclose(a, 0.0, rtol=e, atol=e)` reduces to `|a| <= e` as well; the elementwise-agreement test in Step 1 pins this.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/test_numerics.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add pyphi/numerics.py test/test_numerics.py
git commit -m "Add pyphi.numerics: tolerant scalar comparison predicates"
```

---

### Task 2: Repoint `utils.eq`/`is_positive`/`is_nonpositive` callers to `numerics`; remove from `utils`

**Files:**
- Modify: `pyphi/utils.py` (delete `eq` at line 140, `is_positive` at 148, `is_nonpositive` at 154; keep `is_falsy`, `positive_part`, `NO_DEFAULT`, `iter_with_default`)
- Modify: every `pyphi/` caller of the three predicates (~34 sites; enumerate with the grep in Step 1)
- Modify: `pyphi/models/cmp.py:120` (EQUALITY_TOLERANCE docstring: `utils.eq` → `numerics.eq`)
- Modify: `pyphi/data_structures/pyphi_float.py:7` (`from pyphi.utils import eq` → `from pyphi.numerics import eq`) — keeps the type working until Task 10 deletes it
- Modify: `CLAUDE.md` "Common Pitfalls" §1 (example becomes `from pyphi import numerics` / `numerics.is_zero(phi)`; key functions list becomes `numerics.is_zero`, `numerics.is_positive`, `numerics.eq`)
- Modify: any `test/` callers of `utils.eq`/`utils.is_positive`/`utils.is_nonpositive`

**Interfaces:**
- Consumes: `pyphi.numerics` from Task 1.
- Produces: `pyphi/utils.py` no longer defines any tolerant predicate; all callers import from `pyphi.numerics`. No re-export shims.

- [ ] **Step 1: Enumerate every caller**

Run:
```bash
grep -rn "utils\.eq(\|utils\.is_positive(\|utils\.is_nonpositive(\|_utils\.eq(\|from pyphi.utils import eq\|from .utils import eq" pyphi/ test/ --include="*.py"
```
Expected: ~34 hits in `pyphi/` (concentrated in `condensation.py`, `models/actual_causation.py`, `models/explanation.py`, `models/ria.py`, `formalism/iit4/__init__.py`, `formalism/actual_causation/compute.py`, `sweep.py`, `estimate.py`, `measures/ces.py`, `macro/criteria.py`, `models/ces.py`, `models/diff.py`) plus test callers. Record the exact list before editing.

- [ ] **Step 2: Delete the three predicates from `pyphi/utils.py`**

Remove these functions (currently at `pyphi/utils.py:140-158`):

```python
def eq(x: float, y: float) -> bool: ...
def is_positive(x: float) -> bool: ...
def is_nonpositive(x: float) -> bool: ...
```

Keep `is_falsy`, `positive_part`, `np_hashable`, and everything else. Remove the now-unused `import math` only if nothing else in `utils.py` uses it (check with `grep -n "math\." pyphi/utils.py`).

- [ ] **Step 3: Update every caller from Step 1's list**

For each file: replace `utils.eq(` → `numerics.eq(`, `utils.is_positive(` → `numerics.is_positive(`, `utils.is_nonpositive(` → `numerics.is_nonpositive(`, adding `from pyphi import numerics` (or `from . import numerics` / `from .. import numerics` matching the file's existing relative-import style) to the imports. Where a file used a local alias (`from pyphi import utils as _utils` in `condensation.py:_phi_groups`), import `from pyphi import numerics as _numerics` and call `_numerics.eq`. Update `pyphi/data_structures/pyphi_float.py` and `pyphi/models/cmp.py` docstring as listed in Files. Update `CLAUDE.md` pitfalls §1.

- [ ] **Step 4: Verify no stragglers, then run the full suite**

Run:
```bash
grep -rn "utils\.eq(\|utils\.is_positive(\|utils\.is_nonpositive(" pyphi/ test/ --include="*.py"; echo "exit: $?"
```
Expected: no matches (exit 1).

Run: `uv run pytest -x -q` (full suite, no path argument — includes doctests; `utils.eq` had doctests only via CLAUDE examples, but the sweep confirms no module references broke).
Expected: all pass, zero behavior change.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "Move tolerant scalar predicates from utils to numerics"
```

---

### Task 3: Tolerant clustering in `resolve_ties` (the selection engine)

**Files:**
- Modify: `pyphi/resolve_ties.py` (`_apply_level` at lines 170-182; `resolve()` at lines 682-699; delete the dead commented-out `resolve` draft at lines 663-679 and the stale `TODO` above it)
- Test: `test/test_resolve_ties_precision.py` (new)

**Interfaces:**
- Consumes: `numerics.eq` from Task 1.
- Produces: `_tied_with_extremum(objects, keys, extremum) -> list` module-private helper; `_apply_level` and `resolve()` cluster float keys tolerantly. Signatures of `resolve()`, `states()`, `partitions()`, `purviews()`, `sias()`, `cascade()`, and every `resolve_*_tie` function are unchanged.

- [ ] **Step 1: Write the failing tests**

Create `test/test_resolve_ties_precision.py`:

```python
"""Precision-awareness of the resolve_ties selection engine.

Every selection must (a) treat sub-tolerance float differences as ties
and (b) return the same winner set regardless of candidate order.
"""

import itertools
from dataclasses import dataclass, field
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pyphi import resolve_ties


@dataclass(frozen=True)
class FakePartition:
    key: bytes

    def lex_key(self):
        return self.key


@dataclass(frozen=True)
class FakeMIP:
    phi: float
    normalized_phi: float
    partition: FakePartition
    purview: tuple = field(default=(0,))


NOISE = 5.6e-16  # the observed mirror-isomorphic candidate gap


def _mips(*specs):
    """Build FakeMIPs from (phi, normalized_phi, lexbyte) triples."""
    return [
        FakeMIP(phi=p, normalized_phi=n, partition=FakePartition(bytes([b])))
        for p, n, b in specs
    ]


class TestResolveClustering:
    def test_noise_tied_candidates_both_survive_default_strategy(self):
        # Default mip_tie_resolution = ["NORMALIZED_PHI", "NEGATIVE_PHI"]:
        # sub-tolerance gaps on both keys must not drop either candidate.
        a, b = _mips((0.3, 0.15, 1), (0.3 + NOISE, 0.15 + NOISE, 2))
        survivors = list(resolve_ties.partitions([a, b]))
        assert set(survivors) == {a, b}

    def test_genuine_difference_still_selects(self):
        a, b = _mips((0.3, 0.15, 1), (0.4, 0.2, 2))
        survivors = list(resolve_ties.partitions([a, b]))
        assert survivors == [a]  # min normalized_phi

    def test_lexicographic_tolerance_per_component(self):
        # Component 1 tied within tolerance -> decision falls to component 2,
        # even though exact comparison of component 1 would differ.
        a, b = _mips((0.5, 0.15 + NOISE, 1), (0.2, 0.15, 2))
        # operation=min over NORMALIZED_PHI: tied -> NEGATIVE_PHI: min(-phi)
        # = max phi -> a (phi 0.5) wins.
        survivors = list(resolve_ties.partitions([a, b]))
        assert survivors == [a]

    def test_permutation_invariance_exhaustive(self):
        mips = _mips(
            (0.3, 0.15, 1),
            (0.3 + NOISE, 0.15 + NOISE, 2),
            (0.3 - NOISE, 0.15 - NOISE, 3),
            (0.7, 0.4, 4),
        )
        results = {
            frozenset(resolve_ties.partitions(list(perm)))
            for perm in itertools.permutations(mips)
        }
        assert len(results) == 1

    def test_integer_keys_compare_exactly(self):
        # PURVIEW_SIZE is an int key; exact comparison, no tolerance.
        a = FakeMIP(0.3, 0.15, FakePartition(b"\x01"), purview=(0,))
        b = FakeMIP(0.3, 0.15, FakePartition(b"\x02"), purview=(0, 1))
        survivors = list(
            resolve_ties.resolve([a, b], ["PURVIEW_SIZE"], operation=max)
        )
        assert survivors == [b]


class TestCascadeClustering:
    def test_apply_level_clusters_float_keys(self):
        a, b = _mips((0.3, 0.15, 1), (0.3 + NOISE, 0.15, 2))
        level = resolve_ties.CascadeLevel(
            postulate="Integration", op="argmax", key=lambda m: m.phi
        )
        assert set(resolve_ties._apply_level([a, b], level)) == {a, b}

    def test_apply_level_exact_for_bytes(self):
        a, b = _mips((0.3, 0.15, 1), (0.3, 0.15, 2))
        level = resolve_ties.CascadeLevel(
            postulate="Determinism", op="argmin",
            key=lambda m: m.partition.lex_key(),
        )
        assert resolve_ties._apply_level([a, b], level) == (a,)


@settings(max_examples=200, deadline=None)
@given(
    base=st.floats(min_value=0.01, max_value=10.0, allow_nan=False),
    n_twins=st.integers(min_value=2, max_value=5),
    n_others=st.integers(min_value=0, max_value=4),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_property_twins_always_coselected(base, n_twins, n_others, seed):
    """Candidates perturbed by sub-tolerance noise are always co-selected,
    under every input permutation (sampled)."""
    import random

    rng = random.Random(seed)
    twins = [
        FakeMIP(
            phi=base + i * NOISE,
            normalized_phi=(base + i * NOISE) / 2,
            partition=FakePartition(bytes([i])),
        )
        for i in range(n_twins)
    ]
    others = [
        FakeMIP(
            phi=base + 1.0 + j,
            normalized_phi=(base + 1.0 + j) / 2,
            partition=FakePartition(bytes([100 + j])),
        )
        for j in range(n_others)
    ]
    pool = twins + others
    rng.shuffle(pool)
    survivors = set(resolve_ties.partitions(pool))
    assert survivors == set(twins)  # twins are the min tier, all co-selected
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/test_resolve_ties_precision.py -v`
Expected: `test_noise_tied_candidates_both_survive_default_strategy`, `test_lexicographic_tolerance_per_component`, `test_permutation_invariance_exhaustive`, `test_apply_level_clusters_float_keys`, and the Hypothesis property FAIL (raw-float `==`/tuple comparison drops the perturbed twins). `test_genuine_difference_still_selects`, `test_integer_keys_compare_exactly`, `test_apply_level_exact_for_bytes` may already pass.

- [ ] **Step 3: Implement tolerant clustering**

In `pyphi/resolve_ties.py`, add `from . import numerics` to the imports, then add the shared helper and rewrite `_apply_level` and `resolve()`:

```python
def _tied_with_extremum[U](
    objects: Sequence[U], keys: Sequence[Any], extremum: Any
) -> tuple[U, ...]:
    """Return the objects whose key ties the extremum.

    Float keys tie up to ``config.numerics.precision`` (via
    :func:`pyphi.numerics.eq`); all other key types (integers, bytes,
    tuples) compare exactly.
    """
    if isinstance(extremum, float):
        return tuple(
            o for o, k in zip(objects, keys, strict=True) if numerics.eq(k, extremum)
        )
    return tuple(o for o, k in zip(objects, keys, strict=True) if k == extremum)


def _apply_level[U](
    candidates: Sequence[U],
    level: CascadeLevel,
) -> tuple[U, ...]:
    """Apply ``level``'s op to ``candidates`` and return the winners."""
    if level.op == "filter":
        return tuple(c for c in candidates if level.key(c))
    keys = [level.key(c) for c in candidates]
    extremum = max(keys) if level.op == "argmax" else min(keys)
    return _tied_with_extremum(candidates, keys, extremum)
```

Replace `resolve()` (and delete the commented-out draft and its `TODO` block directly above it):

```python
def resolve[T](
    objects: Iterable[T],
    strategy: str | list[str],
    operation: Callable[..., Any],
    default: Any = NO_DEFAULT,
) -> Iterator[T]:
    """Filter φ-objects to those extremal under ``strategy``.

    Strategy components apply lexicographically: for each component in
    order, the exact extremum of the component's key is computed over
    the surviving objects, and every object whose key ties it survives
    (float keys tie up to ``config.numerics.precision``; other key
    types compare exactly). Later components only see the survivors of
    earlier ones. The surviving set is independent of input order.
    """
    if strategy == "NONE":
        yield from iter_with_default(objects, default=default)
        return
    if isinstance(strategy, str):
        strategy = [strategy]
    survivors = list(objects)
    if not survivors:
        yield from iter_with_default(survivors, default=default)
        return
    for name in strategy:
        if len(survivors) == 1:
            break
        key_function = phi_object_tie_resolution_strategies[name]
        keys = [key_function(obj) for obj in survivors]
        extremum = operation(keys)
        survivors = list(_tied_with_extremum(survivors, keys, extremum))
    yield from survivors
```

Note the semantics change is exactly the designed one: the old implementation compared full key *tuples* with `==` against `operation(values)`, so a sub-tolerance difference in any component silently dropped a tied candidate; the new implementation clusters per component. The `default` is only reachable on empty input, matching the old `operation(values, default=default)` behavior.

- [ ] **Step 4: Run the new tests, then the fast lane**

Run: `uv run pytest test/test_resolve_ties_precision.py -v`
Expected: all pass.

Run: `uv run pytest test/ -x -q -m "not slow"`
Expected: pass. **Permitted drift:** tie-set membership may grow (enumerated category 3) — if a golden or fixture asserts tie sets and fails, inspect: the new members must be within `10**-13` of the old extremum; update the fixture deliberately and note it in the commit message. Any φ *value* change is stop-the-line.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "Cluster float tie keys tolerantly in resolve_ties selection"
```

---

### Task 4: Precision-aware AC mechanism-level cascades

**Files:**
- Modify: `pyphi/resolve_ties.py` (`resolve_ac_causal_link_tie` at lines 471-490)
- Modify: `pyphi/models/actual_causation.py` (delete `greater_than_zero` at lines 46-50; its three `__bool__` callers use `numerics.is_positive`)
- Test: `test/test_resolve_ties_precision.py` (extend)

**Interfaces:**
- Consumes: `numerics.eq`, `numerics.is_positive`, `_tied_with_extremum` from Task 3.
- Produces: `resolve_ac_causal_link_tie` clusters α tolerantly; `resolve_ac_partition_tie` is already fixed by Task 3's `_apply_level` (its `abs(r.alpha)` float key now clusters) — this task adds the regression test proving it.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_resolve_ties_precision.py`:

```python
@dataclass(frozen=True)
class FakeAcRIA:
    alpha: float
    purview: tuple
    partition: FakePartition


class TestAcCascades:
    def test_causal_link_tie_clusters_alpha(self):
        # Two purviews with alpha equal up to noise: both must appear in
        # the tied set (over-determination), not one silently dropped.
        a = FakeAcRIA(0.2, (0,), FakePartition(b"\x01"))
        b = FakeAcRIA(0.2 + NOISE, (1,), FakePartition(b"\x02"))
        ctx = resolve_ties.ResolutionContext(max_escalation_level="Determinism")
        outcome = resolve_ties.resolve_ac_causal_link_tie([a, b], context=ctx)
        assert set(outcome.tied_set) == {a, b}

    def test_partition_tie_escalates_to_determinism_on_noise_tie(self):
        # |alpha| tied up to noise -> the cascade must reach the
        # Determinism (lex) level instead of resolving on the noise.
        a = FakeAcRIA(0.2, (0,), FakePartition(b"\x02"))
        b = FakeAcRIA(0.2 + NOISE, (0,), FakePartition(b"\x01"))
        ctx = resolve_ties.ResolutionContext(max_escalation_level="Determinism")
        outcome = resolve_ties.resolve_ac_partition_tie([a, b], context=ctx)
        assert outcome.resolved is b          # lex-smallest partition
        assert set(outcome.tied_set) == {a, b}
```

- [ ] **Step 2: Run tests to verify the causal-link one fails**

Run: `uv run pytest test/test_resolve_ties_precision.py::TestAcCascades -v`
Expected: `test_causal_link_tie_clusters_alpha` FAILS (raw `r.alpha == max_alpha` drops `a`). `test_partition_tie_escalates_to_determinism_on_noise_tie` PASSES already (Task 3 fixed `_apply_level`) — it stays as the regression pin.

- [ ] **Step 3: Fix `resolve_ac_causal_link_tie`**

In `pyphi/resolve_ties.py:474-475`, replace:

```python
    max_alpha = max(r.alpha for r in survivors)
    info_tied = tuple(r for r in survivors if r.alpha == max_alpha)
```

with:

```python
    alphas = [r.alpha for r in survivors]
    info_tied = _tied_with_extremum(survivors, alphas, max(alphas))
```

In `pyphi/models/actual_causation.py`, delete `greater_than_zero` (lines 46-50) and replace its uses (`AcRepertoireIrreducibilityAnalysis.__bool__`, `CausalLink.__bool__`, `AcSystemIrreducibilityAnalysis.__bool__` — find with `grep -n greater_than_zero pyphi/`) with `numerics.is_positive(self.alpha)`. Also check for external importers: `grep -rn greater_than_zero pyphi/ test/` and update any.

- [ ] **Step 4: Run tests**

Run: `uv run pytest test/test_resolve_ties_precision.py test/models/ test/formalism/ -q`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "Cluster α tolerantly in the AC causal-link cascade"
```

---

### Task 5: AC system-level cascade (`resolve_ac_sia_tie`), `_sia` materialization, `causal_nexus`

**Files:**
- Modify: `pyphi/resolve_ties.py` (add `resolve_ac_sia_tie` and `resolve_ac_nexus_tie` after `resolve_ac_causal_link_tie`)
- Modify: `pyphi/models/actual_causation.py` (add `ties`/`set_ties` to `AcSystemIrreducibilityAnalysis`)
- Modify: `pyphi/formalism/actual_causation/compute.py` (`_sia` at lines 561-649: replace `reduce_func=min` with materialize → cascade; use `numerics.round_to_precision` at line 520)
- Modify: `pyphi/actual.py` (`causal_nexus` at lines 909-925: replace `max(result)`)
- Test: `test/formalism/test_ac_system_ties.py` (new)
- Modify: `test/data/perf/call_counts.json` via `uv run python scripts/gen_perf_counts.py` (AC counts shift: streaming min → materialize)

**Interfaces:**
- Consumes: `cascade`, `CascadeLevel`, `ResolutionContext`, `_tied_with_extremum` (Task 3); `AcSystemIrreducibilityAnalysis` attributes `alpha` (float), `size` (int), `partition` (has `.lex_key()`), `cause_indices`/`effect_indices` (int tuples).
- Produces: `resolve_ac_sia_tie(sias, *, context, on_unresolved="defer") -> CascadeOutcome` (argmin `alpha`, Determinism argmin `partition.lex_key()`); `resolve_ac_nexus_tie(sias, *, context, on_unresolved="defer") -> CascadeOutcome` (argmax `alpha`, argmax `size`, Determinism argmin sorted index tuples); `AcSystemIrreducibilityAnalysis.ties -> tuple[AcSystemIrreducibilityAnalysis, ...]` and `.set_ties(seq)` (shared-by-reference like `AcRIA.set_partition_ties`).

- [ ] **Step 1: Write the failing tests**

Create `test/formalism/test_ac_system_ties.py`:

```python
"""System-level AC tie handling: MIP selection and causal nexus."""

from dataclasses import dataclass, field

import pyphi
from pyphi import resolve_ties
from pyphi.examples import actual_causation_substrate


NOISE = 5.6e-16


@dataclass(frozen=True)
class FakePartition:
    key: bytes

    def lex_key(self):
        return self.key


@dataclass(frozen=True)
class FakeAcSIA:
    alpha: float
    size: int
    partition: FakePartition
    cause_indices: tuple = (0,)
    effect_indices: tuple = (0,)


class TestResolveAcSiaTie:
    def test_noise_tied_partitions_escalate_to_lex(self):
        a = FakeAcSIA(0.2 + NOISE, 2, FakePartition(b"\x02"))
        b = FakeAcSIA(0.2, 2, FakePartition(b"\x01"))
        ctx = resolve_ties.ResolutionContext(max_escalation_level="Determinism")
        outcome = resolve_ties.resolve_ac_sia_tie([a, b], context=ctx)
        assert outcome.resolved is b            # lex-smallest partition
        assert set(outcome.tied_set) == {a, b}

    def test_genuine_minimum_wins(self):
        a = FakeAcSIA(0.5, 2, FakePartition(b"\x01"))
        b = FakeAcSIA(0.2, 2, FakePartition(b"\x02"))
        ctx = resolve_ties.ResolutionContext(max_escalation_level="Determinism")
        outcome = resolve_ties.resolve_ac_sia_tie([a, b], context=ctx)
        assert outcome.resolved is b
        assert outcome.tied_set == (b,)


class TestAcSiaEndToEnd:
    def test_sia_populates_ties(self):
        # The canonical OR-AND example; ties may or may not exist, but the
        # attribute must be populated and consistent.
        substrate = actual_causation_substrate()
        with pyphi.config.override(**pyphi.conf.presets.iit3()):
            transition = pyphi.actual.Transition(
                substrate, (1, 0), (1, 0), (0, 1), (0, 1)
            )
            sia = pyphi.actual.sia(transition)
        assert isinstance(sia.ties, tuple)
        assert sia in sia.ties or sia.ties == (sia,)

    def test_causal_nexus_deterministic(self):
        substrate = actual_causation_substrate()
        with pyphi.config.override(**pyphi.conf.presets.iit3()):
            a = pyphi.actual.causal_nexus(substrate, (1, 0), (1, 0))
            b = pyphi.actual.causal_nexus(substrate, (1, 0), (1, 0))
        assert a.transition == b.transition
        assert a.alpha == b.alpha
```

Adjust the `Transition` construction / preset invocation to the actual signatures if they differ — check `test/test_actual.py` for the canonical way this example is driven (`grep -n "Transition(" test/test_actual.py | head -5`) and mirror it exactly.

- [ ] **Step 2: Run tests to verify failures**

Run: `uv run pytest test/formalism/test_ac_system_ties.py -v`
Expected: `TestResolveAcSiaTie` FAILS with `AttributeError: ... has no attribute 'resolve_ac_sia_tie'`; `test_sia_populates_ties` FAILS with `AttributeError: ... no attribute 'ties'`.

- [ ] **Step 3: Implement**

**(a)** In `pyphi/resolve_ties.py`, after `resolve_ac_causal_link_tie`, add (the `_AcSIALike` protocol mirrors `_AcRIALike`):

```python
class _AcSIALike(Protocol):
    """Structural type for an AC system-irreducibility analysis."""

    @property
    def alpha(self) -> float: ...

    @property
    def size(self) -> int: ...

    @property
    def partition(self) -> Any: ...


def resolve_ac_sia_tie[V: _AcSIALike](
    sias: "Iterable[V]",
    *,
    context: ResolutionContext,
    on_unresolved: OnUnresolved = "defer",
) -> CascadeOutcome[V]:
    """Resolve a tie among per-partition AC system analyses at min 𝒜.

    The system-level MIP is the partition of minimum 𝒜 (Albantakis et
    al. 2019, Eq. 20); tie-break behavior is unspecified by the paper.
    The cascade walks Integration (argmin 𝒜) and falls through to a
    pyphi-specific Determinism level (lex-canonical partition) so
    equal-𝒜 partitions resolve reproducibly across iteration orderings.
    """
    return cascade(
        sias,
        levels=[
            CascadeLevel(
                postulate="Integration",
                op="argmin",
                key=lambda s: s.alpha,
            ),
            CascadeLevel(
                postulate="Determinism",
                op="argmin",
                key=lambda s: s.partition.lex_key(),
            ),
        ],
        context=context,
        on_unresolved=on_unresolved,
    )


def resolve_ac_nexus_tie[V: _AcSIALike](
    sias: "Iterable[V]",
    *,
    context: ResolutionContext,
    on_unresolved: OnUnresolved = "defer",
) -> CascadeOutcome[V]:
    """Resolve a tie among candidate transitions for the causal nexus.

    The causal nexus is the transition of maximal 𝒜. Ties escalate to
    the larger transition, then to a pyphi-specific Determinism level
    (lex-smallest cause/effect index sets) for reproducibility.
    """
    return cascade(
        sias,
        levels=[
            CascadeLevel(
                postulate="Integration",
                op="argmax",
                key=lambda s: s.alpha,
            ),
            CascadeLevel(
                postulate="Integration",
                op="argmax",
                key=lambda s: s.size,
            ),
            CascadeLevel(
                postulate="Determinism",
                op="argmin",
                key=lambda s: (
                    tuple(sorted(s.cause_indices)),
                    tuple(sorted(s.effect_indices)),
                ),
            ),
        ],
        context=context,
        on_unresolved=on_unresolved,
    )
```

(The `alpha`-then-`size` order reproduces the previous `order_by() == [alpha, size]` `max()` semantics, with the α comparison now tolerance-clustered.)

**(b)** In `pyphi/models/actual_causation.py`, inside `AcSystemIrreducibilityAnalysis`, add to `__init__` (find it with `grep -n "def __init__" pyphi/models/actual_causation.py` — the AcSIA one) the line `self._ties: tuple[AcSystemIrreducibilityAnalysis, ...] = (self,)`, and add alongside the existing properties:

```python
    @property
    def ties(self) -> tuple["AcSystemIrreducibilityAnalysis", ...]:
        """System analyses tied with this one at the winning 𝒜, including
        this one. A singleton when the minimum is unique."""
        return self._ties

    def set_ties(self, ties: Sequence["AcSystemIrreducibilityAnalysis"]) -> None:
        """Attach the tied analysis set, shared by reference among peers."""
        tied = tuple(ties)
        if len(tied) <= 1:
            self._ties = (self,)
            return
        for member in tied:
            member._ties = tied
```

If `AcSystemIrreducibilityAnalysis` uses `__slots__`, add `"_ties"` to it; check with `grep -n "__slots__" pyphi/models/actual_causation.py`. Also verify serialization: `grep -n "AcSIA\|ac_sia" pyphi/serialize/schema.py pyphi/serialize/convert.py` — if the AcSIA schema round-trips constructor fields only, `_ties` resets to the singleton on load, which is acceptable (matches `AcRIA._partition_ties` behavior; confirm that precedent with `grep -n "partition_ties" pyphi/serialize/convert.py`).

**(c)** In `pyphi/formalism/actual_causation/compute.py`:

At line 520, replace `alpha=round(alpha, config.numerics.precision),` with `alpha=numerics.round_to_precision(alpha),` (import `from pyphi import numerics` if Task 2 didn't already add it).

Replace the `_sia` reduction (lines 628-646) — currently `result = map_reduce(..., reduce_func=min, reduce_kwargs={"default": _null_ac_sia(...)}, shortcircuit_func=utils.is_falsy, **parallel_kwargs)` — with the IIT 3.0 materialize-then-cascade pattern (`pyphi/formalism/iit3/__init__.py:367-408` is the reference):

```python
    candidates = map_reduce(
        _evaluate_partition,
        cuts,
        map_kwargs={
            "transition": transition,
            "direction": direction,
            "unpartitioned_account": unpartitioned_account,
            "alpha_measure": alpha_measure,
            "partitioned_repertoire_scheme": partitioned_repertoire_scheme,
        },
        shortcircuit_func=utils.is_falsy,
        **parallel_kwargs,
    )
    if not candidates:
        log.info("No partitions to evaluate; returning null AC SIA.")
        return _null_ac_sia(
            transition, direction, reasons=[NullResultReason.NO_VALID_PARTITIONS]
        )
    context = resolve_ties.ResolutionContext(max_escalation_level="Determinism")
    outcome = resolve_ties.resolve_ac_sia_tie(candidates, context=context)
    result = outcome.resolved
    assert result is not None, "AC SIA cascade returned no winner"
    if len(outcome.tied_set) > 1:
        result.set_ties(outcome.tied_set)
    log.info("Finished calculating big-ac-phi data for %s.", transition)
    log.debug("RESULT: \n%s", result)
    return result
```

Add `from pyphi import resolve_ties` to the imports if absent (check: `grep -n "import resolve_ties" pyphi/formalism/actual_causation/compute.py`). Check `NullResultReason.NO_VALID_PARTITIONS` exists (`grep -n "NO_VALID_PARTITIONS" pyphi/models/`); if the AC null path uses a different reason enum member for "no partitions", reuse whatever `_null_ac_sia` callers use for that case.

**(d)** In `pyphi/actual.py:909-925` (`causal_nexus`), replace `result = max(result)` with:

```python
        context = resolve_ties.ResolutionContext(max_escalation_level="Determinism")
        outcome = resolve_ties.resolve_ac_nexus_tie(result, context=context)
        winner = outcome.resolved
        assert winner is not None, "causal-nexus cascade returned no winner"
        if len(outcome.tied_set) > 1:
            winner.set_ties(outcome.tied_set)
        result = winner
```

Add `from pyphi import resolve_ties` to `pyphi/actual.py` imports if absent. `nexus()`'s `sorted(..., reverse=True)` stays — it is presentation ordering, exact and deterministic.

- [ ] **Step 4: Run tests; regenerate perf counters**

Run: `uv run pytest test/formalism/test_ac_system_ties.py test/test_actual.py -v` (the AC test file may live at `test/formalism/test_actual.py` — locate with `ls test/**/test_actual*.py` and run that).
Expected: new tests pass. Existing AC tests: pass, EXCEPT any that pin the exact winning partition where multiple partitions tie at min 𝒜 — enumerated drift category 1. Inspect each: the new winner must have α within `10**-13` of the old; update the assertion deliberately.

Run: `uv run pytest test/integration/test_perf_counters.py -q`
Expected: AC counter rows FAIL (materialization changes call counts). Regenerate: `uv run python scripts/gen_perf_counts.py`, re-run, verify only AC-related counts changed in the diff of `test/data/perf/call_counts.json`.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "Select the AC system MIP and causal nexus via precision-aware cascades"
```

---

### Task 6: Condensation ordering owns tier clustering; macro/substrate callers simplified

**Files:**
- Modify: `pyphi/condensation.py` (`_phi_groups` at lines 96-109 → `_phi_tiers`; `exclusion_cascade` at ~line 189; `iit3_exclusion_cascade` at ~line 242; `_resolve_clique_by_big_phi` at lines 145-186)
- Modify: `pyphi/substrate.py` (complexes region, lines ~843-881: drop the pre-sort)
- Modify: `pyphi/macro/search.py` (lines ~855-870: drop the pre-sort; the `float(phi)` downcast stays until Task 10 makes it redundant)
- Test: `test/test_condensation_ordering.py` (new)

**Interfaces:**
- Consumes: `numerics.eq`.
- Produces: `exclusion_cascade(candidates)` and `iit3_exclusion_cascade(candidates)` accept candidates in **any** order (callers pass dispatch/enumeration order); tier membership is `numerics.eq`-clustered on φₛ and within-tier order preserves the input order. `Candidate.phi` stays a plain `float`.

- [ ] **Step 1: Write the failing tests**

Create `test/test_condensation_ordering.py`:

```python
"""The exclusion cascade's φₛ-tier clustering and within-tier ordering."""

from pyphi.condensation import Candidate, exclusion_cascade


NOISE = 5.6e-16


def _candidate(footprint, phi, tag):
    return Candidate(
        footprint=frozenset(footprint),
        phi=phi,
        sia_provider=lambda tag=tag: tag,
        system_provider=lambda tag=tag: tag,
    )


class TestTierOrdering:
    def test_disjoint_noise_tied_candidates_keep_input_order(self):
        # Two disjoint candidates tied up to noise: BOTH are accepted and
        # their order (hence which is "maximal") follows input order, not
        # the bit pattern of the noise.
        a = _candidate({0, 1}, 0.3, "a")
        b = _candidate({2, 3}, 0.3 + NOISE, "b")  # bitwise larger

        accepted_ab = exclusion_cascade([a, b]).accepted
        accepted_ba = exclusion_cascade([b, a]).accepted

        assert [c.sia_provider() for c in accepted_ab] == ["a", "b"]
        assert [c.sia_provider() for c in accepted_ba] == ["b", "a"]

    def test_unsorted_input_is_tiered_correctly(self):
        # Callers no longer pre-sort; a genuinely lower-phi candidate in
        # front must still land in the later tier.
        low = _candidate({0, 1}, 0.1, "low")
        high = _candidate({0, 1}, 0.9, "high")  # overlaps low
        outcome = exclusion_cascade([low, high])
        assert [c.sia_provider() for c in outcome.accepted] == ["high"]

    def test_tier_membership_is_tolerant(self):
        # Overlapping candidates tied up to noise form ONE clique (tier
        # co-membership), so the Composition escalation runs — the tie is
        # not split into two tiers by the noise.
        a = _candidate({0, 1}, 0.3, "a")
        b = _candidate({1, 2}, 0.3 + NOISE, "b")
        outcome = exclusion_cascade([a, b])
        # Both systems have no real ces()/big_phi (providers return str),
        # so this asserts on clique formation indirectly: with distinct
        # fingerprint-less systems the clique resolves or fails at
        # Composition — either way NOT both accepted.
        assert len(outcome.accepted) <= 1
```

The third test drives `_resolve_clique_by_big_phi` with string "systems" — it relies on `_fingerprint_key` returning fresh `object()`s for fingerprint-less systems and `system.ces()` failing on `str`. Check what happens: `"a".ces()` raises `AttributeError`. If the escalation raises rather than degrading, replace the string providers with a minimal stub having `ces()` returning an object with `big_phi` (two stubs, values `0.5` and `0.5 + NOISE`) and assert `len(outcome.accepted) == 0` (Φ tie → clique fails). Write whichever version actually exercises the path; verify by reading the failure mode when the test first runs.

- [ ] **Step 2: Run tests to verify failures**

Run: `uv run pytest test/test_condensation_ordering.py -v`
Expected: `test_disjoint_noise_tied_candidates_keep_input_order` FAILS (caller-side sort assumption: unsorted input breaks `_phi_groups` contiguity → wrong tiers) and/or `test_unsorted_input_is_tiered_correctly` FAILS. Read the actual failures to confirm the diagnosis before implementing.

- [ ] **Step 3: Implement internal tier ordering**

In `pyphi/condensation.py`, replace `_phi_groups` (lines 96-109) with:

```python
def _phi_tiers(candidates: Sequence[Candidate]):
    """Yield φₛ tiers in descending order.

    Tier membership is tolerant: a candidate joins the tier when its φₛ
    equals the tier head's up to ``config.numerics.precision``. Within a
    tier, candidates keep their input order, so the caller's dispatch
    order — not floating-point noise — breaks presentation ties.
    """
    from pyphi import numerics as _numerics

    indexed = sorted(enumerate(candidates), key=lambda pair: -pair[1].phi)
    i = 0
    while i < len(indexed):
        tier_phi = indexed[i][1].phi
        j = i + 1
        while j < len(indexed) and _numerics.eq(indexed[j][1].phi, tier_phi):
            j += 1
        tier = sorted(indexed[i:j], key=lambda pair: pair[0])
        yield [candidate for _, candidate in tier]
        i = j
```

In `exclusion_cascade` and `iit3_exclusion_cascade`, change `for tier in _phi_groups(candidates):` to `for tier in _phi_tiers(candidates):`, and update both docstrings: delete the "``candidates`` must be sorted by φₛ descending (a stable sort — ties keep their input order)" sentence; state instead that candidates may arrive in any order and within-tier presentation order follows input order.

In `_resolve_clique_by_big_phi` (lines 145-186): remove `from pyphi.data_structures.pyphi_float import PyPhiFloat`, the explanatory comment block about `PyPhiFloat` (lines 163-165), and change `big_phis[key] = PyPhiFloat(system.ces().big_phi)` to `big_phis[key] = float(system.ces().big_phi)`; change the dict annotation to `dict[Any, float]`. (Task 3's `_apply_level` now clusters the `big_phi` floats tolerantly inside `resolve_complex_tie`.)

In `pyphi/substrate.py` (~line 843), replace:

```python
    sorted_sias = sorted(
        irreducible_sias(substrate, state, candidates, **kwargs), reverse=True
    )
    if not sorted_sias:
        return ()
```

with:

```python
    all_sias = list(irreducible_sias(substrate, state, candidates, **kwargs))
    if not all_sias:
        return ()
```

and rename the downstream use (`for sia in sorted_sias` / `cascade_candidates = [_as_candidate(sia) for sia in sorted_sias]` → `all_sias`).

In `pyphi/macro/search.py` (~lines 865-868), replace:

```python
    by_candidate = dict(zip(candidates, (s for s, _ in evaluated), strict=True))
    # Stable sort keeps the sweep's deterministic dispatch order within ties.
    ordered = sorted(candidates, key=lambda c: -c.phi)
    outcome = exclusion_cascade(ordered)
    records_map = exclusion_records(outcome.accepted, ordered)
```

with:

```python
    by_candidate = dict(zip(candidates, (s for s, _ in evaluated), strict=True))
    outcome = exclusion_cascade(candidates)
    records_map = exclusion_records(outcome.accepted, candidates)
```

Also drop the `float(phi)` downcast at line 858 (`phi=float(phi),` → `phi=float(phi),` stays valid while `PyPhiFloat` still exists — actually LEAVE it until Task 10, where `memo` values become plain floats; the downcast is a harmless no-op then and removing it now changes nothing. Leave as-is; Task 10 sweeps it).

Check `exclusion_records`' use of its second argument (`grep -n "def exclusion_records" -A 15 pyphi/condensation.py`): it consumes the candidate list to find overlapping non-accepted candidates; input order only affects record order, which now follows dispatch order — consistent.

- [ ] **Step 4: Run tests and the macro/condensation suites**

Run: `uv run pytest test/test_condensation_ordering.py test/macro/ -q` and `uv run pytest test/ -q -m "not slow" -k "condensation or complex or substrate"`
Expected: pass. **Permitted drift:** which of several φ-tied disjoint complexes is listed first / flagged `is_maximal` (enumerated category 2) — inspect any such failure, confirm the φ values tie within `10**-13`, update deliberately.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "Order exclusion-cascade tiers inside condensation with tolerant clustering"
```

---

### Task 7: Binding-direction and runner-up tie detection

**Files:**
- Modify: `pyphi/models/explanation.py` (`binding_direction_finding` at line 123; `runner_up_from_candidates` at lines 83-102)
- Modify: `pyphi/formalism/iit4/__init__.py` (`_binding_direction_changed` at lines 509-524)
- Modify: `pyphi/models/distinction.py` (`explain` at lines 147-157)
- Test: `test/models/test_explanation.py` (extend — file exists; confirm with `ls test/models/test_explanation.py`)

**Interfaces:**
- Consumes: `numerics.eq`.
- Produces: `binding_direction_finding(cause_phi, effect_phi)` returns `value="TIED"` (tone `"neutral"`) when the φ values tie; `_binding_direction_changed` returns `False` for tied-vs-tied and compares three-valued direction labels; `runner_up_from_candidates` breaks ties among equal runner-up candidates by `partition.lex_key()`.

- [ ] **Step 1: Write the failing tests**

Append to `test/models/test_explanation.py` (match its existing import style):

```python
NOISE = 5.6e-16


class TestBindingDirectionTies:
    def test_finding_reports_tie(self):
        finding = binding_direction_finding(0.3, 0.3 + NOISE)
        assert finding.value == "TIED"

    def test_finding_reports_cause_when_strictly_smaller(self):
        finding = binding_direction_finding(0.2, 0.3)
        assert finding.value == "CAUSE"

    def test_finding_reports_effect_when_strictly_smaller(self):
        finding = binding_direction_finding(0.3, 0.2)
        assert finding.value == "EFFECT"


class TestRunnerUpTieBreak:
    def test_equal_runner_ups_pick_lex_smallest_partition(self):
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class P:
            key: bytes

            def lex_key(self):
                return self.key

        @dataclass(frozen=True)
        class C:
            phi: float
            partition: P

        mip = C(0.1, P(b"\x00"))
        r1 = C(0.5 + NOISE, P(b"\x02"))
        r2 = C(0.5, P(b"\x01"))
        out_a = runner_up_from_candidates([mip, r1, r2], mip.phi)
        out_b = runner_up_from_candidates([mip, r2, r1], mip.phi)
        assert out_a.partition == out_b.partition == P(b"\x01")
```

Add `binding_direction_finding` and `runner_up_from_candidates` to the file's imports from `pyphi.models.explanation`.

- [ ] **Step 2: Run to verify failures**

Run: `uv run pytest test/models/test_explanation.py -v -k "BindingDirection or RunnerUp"`
Expected: `test_finding_reports_tie` FAILS (returns `"CAUSE"`); `test_equal_runner_ups_pick_lex_smallest_partition` FAILS under one of the two orders.

- [ ] **Step 3: Implement**

In `pyphi/models/explanation.py:123`, replace `binding_direction_finding`:

```python
def binding_direction_finding(cause_phi: Any, effect_phi: Any) -> Finding:
    """The Finding naming which direction binds ``min(φ_c, φ_e)``.

    Reports ``"TIED"`` when the two values are equal up to
    ``config.numerics.precision``.
    """
    if numerics.eq(float(cause_phi), float(effect_phi)):
        value, tone = "TIED", "neutral"
    elif float(cause_phi) < float(effect_phi):
        value, tone = "CAUSE", "cause"
    else:
        value, tone = "EFFECT", "effect"
    return Finding(
        kind="binding_direction",
        label="Binding direction",
        value=value,
        detail=(("φ_cause", cause_phi), ("φ_effect", effect_phi)),
        tone=tone,
    )
```

Check `Finding`'s `tone` accepted values (`grep -n "tone" pyphi/models/explanation.py pyphi/display/ -r | head`); if `"neutral"` is not an accepted tone, use whatever the neutral/default tone is (likely `None` — read the `Finding` dataclass definition and match it).

In the same file, `runner_up_from_candidates` — replace the selection loop body condition:

```python
        if (
            phi > mip
            and not numerics.eq(phi, mip)
            and (
                best is None
                or phi < float(best.phi)
                and not numerics.eq(phi, float(best.phi))
                or numerics.eq(phi, float(best.phi))
                and candidate.partition.lex_key() < best.partition.lex_key()
            )
        ):
            best = candidate
```

That compound condition is hard to read — implement it as explicit branches instead:

```python
    mip = float(mip_phi)
    best = None
    for candidate in candidates:
        phi = float(candidate.phi)
        if phi <= mip or numerics.eq(phi, mip):
            continue  # the MIP itself or a tied peer, not a runner-up
        if best is None:
            best = candidate
            continue
        best_phi = float(best.phi)
        if numerics.eq(phi, best_phi):
            if candidate.partition.lex_key() < best.partition.lex_key():
                best = candidate
        elif phi < best_phi:
            best = candidate
```

(`utils.eq` was already imported here pre-Task 2; it is `numerics.eq` now.)

In `pyphi/formalism/iit4/__init__.py:509-524`, replace `_binding_direction_changed`'s direction computation:

```python
        def _direction(cause_phi: float, effect_phi: float) -> str:
            if numerics.eq(float(cause_phi), float(effect_phi)):
                return "tied"
            return "cause" if float(cause_phi) < float(effect_phi) else "effect"

        a_dir = _direction(self.cause.phi, self.effect.phi)
        b_dir = _direction(other.cause.phi, other.effect.phi)
        return a_dir != b_dir
```

In `pyphi/models/distinction.py:153-157`, replace the binding pick:

```python
        binding = (
            self.cause
            if numerics.eq(float(self.cause.phi), float(self.effect.phi))
            or float(self.cause.phi) < float(self.effect.phi)
            else self.effect
        )
```

(On a tie the cause side is reported, matching `min(cause_phi, effect_phi)` returning its first argument on exact ties; the accompanying `binding_direction_finding` now says `TIED`.)

Add the `numerics` import to each file if absent.

- [ ] **Step 4: Run tests**

Run: `uv run pytest test/models/ test/formalism/ -q -m "not slow"`
Expected: pass. Display/explanation goldens asserting `"CAUSE"`/`"EFFECT"` text should still pass (real fixtures have genuinely unequal φ_c/φ_e); if one fails, verify its φ values actually tie within tolerance before touching it — otherwise stop-the-line.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "Report binding-direction and runner-up ties instead of resolving on noise"
```

---

### Task 8: Fix the intrinsic-differentiation positivity filter

**Files:**
- Modify: `pyphi/measures/distribution.py` (`intrinsic_differentiation` at lines ~1307-1340)
- Test: `test/measures/test_measures_distribution.py` (extend)

**Interfaces:**
- Consumes: `numerics.positive_mask`.
- Produces: `intrinsic_differentiation(p, state)` excludes surprisals that are zero up to precision. **This is enumerated drift category 4** — a genuine φ-value correction under `system_phi_measure = "INTRINSIC_INFORMATION"`.

- [ ] **Step 1: Write the failing test**

Append to `test/measures/test_measures_distribution.py` (match existing imports; the file already imports from `pyphi.measures.distribution`):

```python
class TestIntrinsicDifferentiationPrecision:
    def test_certain_node_noise_surprisal_is_excluded(self):
        # A probability of 1 up to floating-point noise has surprisal
        # ~3e-16 — mathematically zero. It must not be selected as the
        # "smallest strictly positive surprisal".
        p = np.array([0.9999999999999998, 0.5])
        result = intrinsic_differentiation(p, (slice(None),))
        assert float(result) == pytest.approx(1.0)  # -log2(0.5), not 3e-16

    def test_exactly_certain_node_still_excluded(self):
        p = np.array([1.0, 0.25])
        result = intrinsic_differentiation(p, (slice(None),))
        assert float(result) == pytest.approx(2.0)

    def test_all_certain_returns_zero(self):
        p = np.array([1.0, 0.9999999999999998])
        result = intrinsic_differentiation(p, (slice(None),))
        assert float(result) == 0.0
```

Adjust the `state` argument form to the function's actual contract: read the function and one existing caller/test first (`grep -n "intrinsic_differentiation" test/measures/test_measures_distribution.py pyphi/ -r | head`) and mirror the real call shape — the intent of each case is what matters.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest test/measures/test_measures_distribution.py -v -k IntrinsicDifferentiationPrecision`
Expected: `test_certain_node_noise_surprisal_is_excluded` FAILS (returns ~3.2e-16). The other two pass (exact-1 is caught by the existing `> 0` filter; all-certain hits the empty-branch).

- [ ] **Step 3: Implement**

In `pyphi/measures/distribution.py`, `intrinsic_differentiation` — replace:

```python
    p = p.squeeze()[state]
    positive_entries = pointwise_intrinsic_differentiation(p)[
        pointwise_intrinsic_differentiation(p) > 0
    ]
```

with:

```python
    p = p.squeeze()[state]
    surprisal = pointwise_intrinsic_differentiation(p)
    positive_entries = surprisal[numerics.positive_mask(surprisal)]
```

(Also removes the doubled `pointwise_intrinsic_differentiation` evaluation.) Add `from pyphi import numerics` to the module imports if Task 2 didn't. Update the docstring's description of the filter ("strictly positive" → "positive up to ``config.numerics.precision``").

- [ ] **Step 4: Run tests + goldens**

Run: `uv run pytest test/measures/ -q` then `uv run pytest test/ -q -m "not slow"`
Expected: pass. **Permitted drift (category 4):** any golden computed under `INTRINSIC_INFORMATION` where a certain-node artifact was previously selected — the new value must be LARGER (the artifact was a spurious minimum) or the capped φ correspondingly corrected. Diff each, document in the commit message, regenerate deliberately. Goldens under the default GID measure must be byte-identical.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "Exclude noise-level surprisals from intrinsic differentiation"
```

---

### Task 9: Tier-3 confirmation experiments (PMI zero-guard; approximate specified state)

**Files:**
- Create: `test/measures/test_precision_confirmations.py`
- Possibly modify: `pyphi/measures/distribution.py` (`pointwise_mutual_information` line ~1467, `weighted_pointwise_mutual_information` ~1495, `approximate_specified_state` ~1079) — ONLY if an experiment produces a witness

**Interfaces:**
- Consumes: the real AC probability plumbing (`pyphi.actual.Transition.probability` / `partitioned_probability`).
- Produces: either (a) a documented negative result pinned as a test, or (b) a fix + regression test per witnessed site.

- [ ] **Step 1: Write and run the PMI residue experiment**

Create `test/measures/test_precision_confirmations.py`:

```python
"""Confirmation experiments: can a mathematically-zero probability reach
the PMI zero-guard as a nonzero floating-point residue?

The guard in ``pointwise_mutual_information`` is exact (``p == 0.0``). If
transition probabilities computed by the AC plumbing can carry ~1e-16
residues where the true value is 0, the guard misfires and produces a
spurious large ``log2``. This experiment drives the real plumbing over
deterministic substrates and records the smallest nonzero probability
observed; the test pins the finding.
"""

import itertools

import numpy as np
import pytest

import pyphi
from pyphi import actual
from pyphi.examples import actual_causation_substrate


SEED = 20260710


def _iter_probabilities():
    """Yield every (partitioned and unpartitioned) transition probability
    over all mechanism/purview pairs of the canonical AC example, in both
    directions, for every reachable before/after state pair."""
    substrate = actual_causation_substrate()
    n = substrate.size
    states = list(itertools.product((0, 1), repeat=n))
    for before, after in itertools.product(states, states):
        try:
            transition = actual.Transition(
                substrate, before, after, tuple(range(n)), tuple(range(n))
            )
        except pyphi.exceptions.StateUnreachableError:
            continue
        for direction in (
            pyphi.Direction.CAUSE,
            pyphi.Direction.EFFECT,
        ):
            mechanisms = pyphi.utils.powerset(range(n), nonempty=True)
            for mechanism in mechanisms:
                for purview in pyphi.utils.powerset(range(n), nonempty=True):
                    try:
                        yield float(
                            transition.probability(direction, mechanism, purview)
                        )
                    except Exception:  # unreachable/invalid combos
                        continue


def test_no_subprecision_probability_residues():
    """Every probability from the real plumbing is either exactly 0 or
    at least the tolerance away from 0 — the exact PMI guard is safe.

    If this test ever fails, the PMI guards need
    ``numerics.is_zero(p) or numerics.is_zero(q)``.
    """
    tol = 10 ** (-13)
    values = np.array(list(_iter_probabilities()))
    assert values.size > 0
    nonzero = values[values != 0.0]
    assert nonzero.size == 0 or nonzero.min() > tol
```

Fit the driver to the real API first: check `Transition.probability`'s exact signature (`grep -n "def probability" pyphi/actual.py`) and adjust the mechanism/purview iteration accordingly — the experiment's substance is "exhaustively sample the real probability values and check the gap around zero".

Run: `uv run pytest test/measures/test_precision_confirmations.py -v`

- [ ] **Step 2: Act on the PMI result**

- **If it passes** (expected: deterministic OR/AND gates produce probabilities that are exact 0, or ≥ 2⁻ⁿ): the exact guard is confirmed safe; keep the test as the standing confirmation. Add an inline waiver comment at both PMI guards in `pyphi/measures/distribution.py`:
  ```python
    if p == 0.0 or q == 0.0:  # numerics: exact — probabilities from the
        # transition plumbing are exactly 0 or bounded away from 0; see
        # test/measures/test_precision_confirmations.py
  ```
- **If it fails** (a residue exists): change both guards to `if numerics.is_zero(p) or numerics.is_zero(q):`, keep the experiment as the regression test, and record the witness values in the test docstring.

- [ ] **Step 3: The `approximate_specified_state` site**

The `discriminant < tmp_inform` comparison (`pyphi/measures/distribution.py:1079`) sits inside a documented approximation ("results are only a good guess") whose docstring already accepts arbitrary tie-breaking. Confirm the docstring still says so (`grep -n "good guess\|arbitrar" pyphi/measures/distribution.py`), then add the waiver comment at the comparison:

```python
        if discriminant < tmp_inform:  # numerics: exact — documented
            # approximation; a sub-precision tie degrades an explicit
            # guess, not an exact result
```

No behavioral change; no test needed beyond the existing ones.

- [ ] **Step 4: Run the measures suite**

Run: `uv run pytest test/measures/ -q`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "Confirm the PMI exact zero-guard against the real probability plumbing"
```

(Adjust the message if Step 2 took the fix branch: "Guard PMI against sub-precision probability residues".)

---

### Task 10: Strip `PyPhiFloat` constructions from models, formalisms, and macro

**Files:**
- Modify: `pyphi/models/ria.py` (lines 114-130 annotations; 155-200 constructions)
- Modify: `pyphi/models/state_specification.py` (lines 107-125)
- Modify: `pyphi/formalism/iit4/__init__.py` (all 16 `PyPhiFloat` references — enumerate with grep)
- Modify: `pyphi/models/sia.py`, `pyphi/models/diff.py`, `pyphi/models/ces.py`, `pyphi/relations.py`, `pyphi/core/repertoire_algebra.py`, `pyphi/macro/search.py`, `pyphi/macro/criteria.py`
- Test: existing suites (behavior-preserving after Tasks 3-7)

**Interfaces:**
- Consumes: nothing new.
- Produces: no module outside `pyphi/data_structures/` and `pyphi/serialize/` references `PyPhiFloat`; `.phi`/`.alpha`/`.normalized_phi`/`intrinsic_information` are plain `float` (or `DistanceResult`).

- [ ] **Step 1: Enumerate all sites**

Run: `grep -rn "PyPhiFloat" pyphi/ --include="*.py" | grep -v "pyphi_float.py\|serialize/\|data_structures/__init__"`
Expected: ~50 lines across the files listed above. Work file-by-file.

- [ ] **Step 2: Convert, file by file**

The uniform transformation: `PyPhiFloat(x)` → `float(x)`; type annotations `PyPhiFloat` → `float`, `PyPhiFloat | DistanceResult` → `float | DistanceResult` (note `DistanceResult` remains a `float` subclass, so plain `float` annotations also admit it — prefer `float` alone unless the code reads `DistanceResult` metadata); remove the `PyPhiFloat` imports. Specific sites needing more than the mechanical swap:

- `pyphi/models/ria.py:155-161`: `self._phi = PyPhiFloat(clamped_phi)` → `self._phi = float(clamped_phi)`; the `DistanceResult` branch stays. Same for `_signed_phi`, `_signed_normalized_phi`, `_normalized_phi` (lines 165, 199-200). Property return annotations (lines 206, 219) → `float` / `float | DistanceResult`; update the `phi` property docstring's first line (`"PyPhiFloat: Canonical φ value"` → `"float: Canonical φ value"`).
- `pyphi/models/state_specification.py:117-125` (`__post_init__`): the wrapping branches become plain-float coercion:
  ```python
    def __post_init__(self):
        if not isinstance(self.intrinsic_information, DistanceResult):
            self.intrinsic_information = float(self.intrinsic_information)
        if self.runner_up_intrinsic_information is not None and not isinstance(
            self.runner_up_intrinsic_information, DistanceResult
        ):
            self.runner_up_intrinsic_information = float(
                self.runner_up_intrinsic_information
            )
  ```
  Annotations at lines 110/114 → `float | DistanceResult`; the class docstring's attribute types likewise.
- `pyphi/models/ces.py:203`: `big_phi` returns the plain-float sum already once operands are floats; add the docstring line "float: Φ, the sum of distinction and relation φ." if the property lacks one.
- `pyphi/models/sia.py:195` and `pyphi/formalism/iit4/__init__.py:463`: `value=PyPhiFloat(float(self.runner_up.phi) - float(self.phi))` → `value=float(self.runner_up.phi) - float(self.phi)`.
- `pyphi/formalism/iit4/__init__.py:1140-1143`: `gap = ...; PyPhiFloat(max(0.0, gap))` → `max(0.0, gap)` (plain float).
- `pyphi/relations.py:59,195-201`: `self.phi = PyPhiFloat(phi)` fallback → `float(phi)`; the `Relation.phi` cached property `PyPhiFloat(len(...) * min(...))` → `float(len(...) * min(...))`; return annotations → `float`.
- `pyphi/core/repertoire_algebra.py:652`: `intrinsic_information=PyPhiFloat(information)` → `intrinsic_information=float(information)` (check the other two references the grep finds in this file and treat identically).
- `pyphi/macro/search.py`: `memo: dict[MacroSystem, PyPhiFloat]` → `dict[MacroSystem, float]`; `_phi` helper returning `PyPhiFloat` → `float`; `evaluated: list[tuple[MacroSystem, PyPhiFloat]]` → `list[tuple[MacroSystem, float]]`; the `phi=float(phi)` in the `Candidate` construction is now redundant → `phi=phi`.
- `pyphi/macro/criteria.py`: the `float(...)` casts around φ comparisons become no-ops — simplify (`best_phi = float(competitor_phi)` → direct use) but preserve the `numerics.eq`-based logic exactly.

- [ ] **Step 3: Run the full fast suite**

Run: `uv run pytest test/ -q -m "not slow"` and `uv run pyright pyphi` (if pyright is enabled in this session's pre-commit, the commit gate runs it anyway).
Expected: pass, byte-identical goldens (Tasks 3-7 already moved all comparison semantics to decision sites; this task only removes now-inert wrappers).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "Store φ and α as plain floats in result types"
```

---

### Task 11: Re-base `DistanceResult` on `float`; delete `PyPhiFloat`; serialization schema

**Files:**
- Modify: `pyphi/measures/distribution.py` (`class DistanceResult(PyPhiFloat)` at line 53)
- Delete: `pyphi/data_structures/pyphi_float.py`
- Modify: `pyphi/data_structures/__init__.py` (drop the export)
- Modify: `pyphi/serialize/schema.py` (lines 21-32, 475), `pyphi/serialize/convert.py` (lines 55-58 + registration calls)
- Modify: `pyphi/conf/numerics.py` (docstring lines 22-26)
- Delete: `test/data_structures/test_pyphi_float.py`
- Modify: `test/serialize/test_serialize_values.py`, `test/models/test_models.py`, `test/models/test_to_pandas.py`, `test/macro/test_macro_search.py`, `test/formalism/test_iit4_sia_components.py`, `test/measures/test_measures_distribution.py` (drop `PyPhiFloat` references — enumerate with `grep -rn PyPhiFloat test/`)
- Regenerate: the 28 fixture files under `test/data/` containing `pyphi_float` tags (`grep -rl pyphi_float test/data/`)

**Interfaces:**
- Consumes: nothing new.
- Produces: `class DistanceResult(float)` with metadata, exact comparisons; `PhiSchema = float | DistanceResultSchema`; `pyphi.data_structures` no longer exports `PyPhiFloat`.

- [ ] **Step 1: Re-base `DistanceResult` and delete the type**

In `pyphi/measures/distribution.py:53`: `class DistanceResult(PyPhiFloat):` → `class DistanceResult(float):`; remove the `PyPhiFloat` import at the top of the file. Update the class docstring: the two sentences describing `PyPhiFloat` behavior ("DistanceResult extends PyPhiFloat…", "behaves like a PyPhiFloat for all mathematical operations") become a description of the actual contract — a `float` carrying auxiliary metadata, with exact float comparison semantics; comparison of φ values at decision points goes through `pyphi.numerics` / `pyphi.resolve_ties`. In `_public_aux_data`'s docstring, drop the parenthetical about "the precision snapshot inherited from PyPhiFloat" (keep the underscore-exclusion rule — `_preserve_aux_data` etc. still rely on it).

Delete `pyphi/data_structures/pyphi_float.py`. In `pyphi/data_structures/__init__.py` remove the `from .pyphi_float import PyPhiFloat as PyPhiFloat` line.

- [ ] **Step 2: Serialization**

In `pyphi/serialize/schema.py`: delete `PyPhiFloatSchema` (lines 21-22); `PhiSchema = PyPhiFloatSchema | DistanceResultSchema` → `PhiSchema = float | DistanceResultSchema`; remove `PyPhiFloatSchema` from the big union at line 475. In `pyphi/serialize/convert.py`: delete the `_register_pyphi_float` function (lines ~53-58) and its call-site registration (find with `grep -n "_register_pyphi_float" pyphi/serialize/convert.py`). Verify plain floats in `PhiSchema` positions encode/decode natively through msgspec (they do — `float` is a msgspec primitive; the tagged `DistanceResultSchema` union member disambiguates).

- [ ] **Step 3: Fix tests, delete the dead test file**

`git rm test/data_structures/test_pyphi_float.py`. For the other test files from the grep: replace `PyPhiFloat(x)` constructions with plain floats and delete assertions about tolerant comparison of bare values (those semantics are now covered by `test/test_numerics.py` and `test/test_resolve_ties_precision.py`). Read each site — a test asserting `isinstance(sia.phi, PyPhiFloat)` becomes `isinstance(sia.phi, float)`.

- [ ] **Step 4: Regenerate the 28 fixtures**

The fixtures (`test/data/relations/*.json`, `test/data/sia/*.json`, and others from `grep -rl pyphi_float test/data/`) embed `pyphi_float`-tagged values and can no longer decode. Locate their generators: `grep -rn "test/data/relations\|ces_rule110\|data/sia" scripts/ test/ --include="*.py" | grep -iv "def test" | head -20` — regen scripts or golden-harness writers. Regenerate each; then verify values survived unchanged:

```bash
uv run python - <<'PY'
"""Verify fixture regeneration changed only the encoding of phi values."""
import json, subprocess, sys
from pathlib import Path

def strip(obj):
    """Normalize old-format {'pyphi_float': {'value': x}} (msgspec tagged
    structs may also serialize as ['pyphi_float', {...}] or with a 'tag'
    key — inspect one old file first and adapt) to bare floats."""
    if isinstance(obj, dict):
        keys = set(obj.keys())
        if keys == {"value"} or ("value" in keys and obj.get("tag") == "pyphi_float"):
            return obj["value"]
        return {k: strip(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [strip(v) for v in obj]
    return obj

failures = []
for path in subprocess.run(
    ["git", "diff", "--name-only", "--", "test/data/"],
    capture_output=True, text=True, check=True,
).stdout.split():
    old = subprocess.run(["git", "show", f"HEAD:{path}"],
                         capture_output=True, text=True, check=True).stdout
    new = Path(path).read_text()
    if strip(json.loads(old)) != strip(json.loads(new)):
        failures.append(path)
print("value drift in:", failures or "none")
sys.exit(1 if failures else 0)
PY
```

Inspect one old fixture's actual tag encoding first (`head -c 500 test/data/sia/s.json`) and adapt `strip` to it. Expected: `value drift in: none`. Any value drift is stop-the-line.

- [ ] **Step 5: Update `pyphi/conf/numerics.py` docstring**

Lines 22-26: replace the `PyPhiFloat` sentences with:

```python
    precision : int
        Decimal places of agreement required when comparing φ values via
        :func:`pyphi.numerics.eq` and the other :mod:`pyphi.numerics`
        predicates. Values smaller than ``10**-precision`` are treated as
        zero.
```

- [ ] **Step 6: Full verification**

Run: `uv run pytest` (NO path argument — doctest sweep included; `DistanceResult`'s doctests must pass with exact semantics).
Expected: all pass. `grep -rn "PyPhiFloat" pyphi/ test/ docs/ --include="*.py"` → zero hits.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "Delete PyPhiFloat; DistanceResult is an exact float with metadata"
```

---

### Task 12: Enforcement lint

**Files:**
- Create: `test/test_precision_lint.py`
- Modify: waiver comments where the lint flags value-definitional sites (expected: `pyphi/formalism/queries.py` φ=min definitions, `pyphi/formalism/iit4/__init__.py:728` direction-min and `:795` ii-cap min, `pyphi/models/distinction.py:145` `min(self.cause.phi, self.effect.phi)`, `pyphi/measures/distribution.py` `min(specification, differentiation)`, plus whatever the first run surfaces)

**Interfaces:**
- Consumes: Python `ast` stdlib.
- Produces: a CI-blocking test forbidding raw φ/α comparisons outside `pyphi/numerics.py` and `pyphi/resolve_ties.py`; the waiver marker is a `# numerics: exact — <reason>` comment on the flagged line or the line above.

- [ ] **Step 1: Write the lint test**

Create `test/test_precision_lint.py`:

```python
"""Forbid raw comparisons of φ/Φ/α magnitudes outside the tolerant layer.

A comparison operator or min/max/sorted call whose operand is an
attribute named like a φ magnitude must route through pyphi.numerics or
pyphi.resolve_ties, or carry an explicit waiver comment
(``# numerics: exact — <reason>``) on its line or the line above.
"""

import ast
from pathlib import Path

PHI_ATTRS = {
    "phi",
    "alpha",
    "big_phi",
    "normalized_phi",
    "signed_phi",
    "signed_normalized_phi",
    "sum_phi",
    "intrinsic_information",
}
ALLOWED_MODULES = {"numerics.py", "resolve_ties.py"}
PYPHI = Path(__file__).parent.parent / "pyphi"


def _mentions_phi_attr(node: ast.AST) -> bool:
    return any(
        isinstance(sub, ast.Attribute) and sub.attr in PHI_ATTRS
        for sub in ast.walk(node)
    )


def _waived(lines: list[str], lineno: int) -> bool:
    for candidate in (lineno - 1, lineno - 2):  # the line and the line above
        if 0 <= candidate < len(lines) and "# numerics: exact" in lines[candidate]:
            return True
    return False


def _violations(path: Path) -> list[str]:
    source = path.read_text()
    lines = source.splitlines()
    tree = ast.parse(source)
    found = []

    def flag(node, what):
        if not _waived(lines, node.lineno):
            found.append(f"{path.relative_to(PYPHI.parent)}:{node.lineno} {what}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            if any(
                isinstance(op, ast.Attribute) and op.attr in PHI_ATTRS
                for op in operands
            ):
                flag(node, "raw comparison of a φ attribute")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"min", "max", "sorted"} and (
                any(_mentions_phi_attr(arg) for arg in node.args)
                or any(
                    kw.arg == "key" and _mentions_phi_attr(kw.value)
                    for kw in node.keywords
                )
            ):
                flag(node, f"raw {node.func.id}() over a φ attribute")
    return found


def test_no_raw_phi_comparisons():
    violations = []
    for path in sorted(PYPHI.rglob("*.py")):
        if path.name in ALLOWED_MODULES:
            continue
        violations.extend(_violations(path))
    assert not violations, (
        "Raw φ/α comparisons found. Route through pyphi.numerics or "
        "pyphi.resolve_ties, or add '# numerics: exact — <reason>':\n"
        + "\n".join(violations)
    )
```

- [ ] **Step 2: Run and triage every hit**

Run: `uv run pytest test/test_precision_lint.py -v`
Expected: FAILS with a list of hits. Triage each one:
- **Value definition** (φ = min of directions, ii-cap min, `Relation.phi` weakest-link min, display/serialization float reads that happen to sit in a comparison): add `# numerics: exact — <reason>` naming why exactness is correct there.
- **Genuine selection missed by the audit**: STOP — report it to the user before fixing; it is a new B22 finding.
- **False positive on the lint's pattern** (e.g. comparing `.phi` to a literal `0` inside an already-tolerant helper): prefer a waiver comment over weakening the lint.

- [ ] **Step 3: Re-run until green, then the full fast lane**

Run: `uv run pytest test/test_precision_lint.py test/ -q -m "not slow"`
Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "Add a lint forbidding raw φ/α comparisons outside the tolerant layer"
```

---

### Task 13: Full verification, changelog, migration guide, roadmap

**Files:**
- Create: `changelog.d/precision-architecture.change.md`, `changelog.d/ac-system-ties.feature.md`, `changelog.d/intrinsic-differentiation-noise.fix.md`
- Modify: `docs/migration/migration-2.0.md` (new subsection), `ROADMAP.md` (B22 row)

**Interfaces:** none new — verification and records.

- [ ] **Step 1: Full suite, both lanes**

Run the slow lane in background and the fast lane in foreground:
```bash
uv run pytest -m slow -q          # background (run_in_background)
uv run pytest -q                  # foreground, full default suite incl. doctests
```
Expected: both green. Also run `uv run pytest test/integration/test_golden_regression.py test/integration/test_cross_formalism_invariants.py -q` explicitly and confirm the B5 invariants and goldens are green.

- [ ] **Step 2: Changelog fragments**

```bash
cat > changelog.d/precision-architecture.change.md <<'EOF'
φ, Φ, and α values are now plain floats with exact comparison semantics.
Tolerant comparison (up to `config.numerics.precision`) lives at decision
sites: the scalar predicates in the new `pyphi.numerics` module
(`eq`, `is_zero`, `is_positive`, `positive_mask`, `round_to_precision`)
and the tie-resolution cascades in `pyphi.resolve_ties`, which now
cluster float keys tolerantly so candidates tied up to precision are
always co-selected regardless of iteration order. The `PyPhiFloat` type
is removed; `DistanceResult` remains a float carrying metadata.
`pyphi.utils.eq` / `is_positive` / `is_nonpositive` moved to
`pyphi.numerics`.
EOF
cat > changelog.d/ac-system-ties.feature.md <<'EOF'
Actual-causation system analyses now record ties: the system-level MIP
and the causal nexus are selected through precision-aware cascades with
deterministic tie-breaks, and `AcSystemIrreducibilityAnalysis.ties`
carries the tied set.
EOF
cat > changelog.d/intrinsic-differentiation-noise.fix.md <<'EOF'
`INTRINSIC_DIFFERENTIATION` no longer selects a floating-point noise
artifact as the minimum positive surprisal when a purview node is
certain up to rounding; the positivity filter now respects
`config.numerics.precision`.
EOF
```

- [ ] **Step 3: Migration guide entry**

Add to `docs/migration/migration-2.0.md`, following its existing topic/audience-tag format (read a neighboring section first and match it), a "Precision and φ comparison" subsection covering: `.phi`/`.alpha` are plain floats (direct `==`/`<` between them is exact — use `pyphi.numerics.eq` for tolerant comparison); `pyphi.utils.eq`→`pyphi.numerics.eq` (and friends, plus new `is_zero`); `PyPhiFloat` removed; tie sets may include members that earlier versions silently dropped.

- [ ] **Step 4: ROADMAP update**

In `ROADMAP.md`, update the B22 dashboard row status from `⬜ open` to `✅ landed` and rewrite its one-liner to describe what landed (audit: eight raw-float φ/α comparison sites found and fixed; `PyPhiFloat` deleted — tolerant comparison consolidated at decision sites in `pyphi.numerics` + `resolve_ties` tolerant clustering; AC gains system-level tie cascade; AST lint gates recurrence; spec `docs/superpowers/specs/2026-07-10-precision-comparison-architecture-design.md`). Update the Wave 1 work-item entry to match.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "Add changelog, migration notes, and roadmap entry for the precision architecture"
```

---

## Self-Review (completed)

- **Spec coverage:** charter+derivation → numerics docstring (Task 1); scalar layer (Task 1-2); tolerant lexicographic cascade + NEGATIVE_PHI fix (Task 3); AC mechanism cascades (Task 4); AC system cascade + ties + nexus (Task 5); condensation tier ordering + macro/substrate callers + `big_phi` re-wrap removal (Task 6); binding-direction/runner-up (Task 7); ii-filter (Task 8); Tier-3 confirmations (Task 9); type deletion + `DistanceResult` re-base + schema + fixtures (Tasks 10-11); lint (Task 12); goldens/changelog/migration/roadmap (Task 13). Waiver sites named in Task 12 match the spec's out-of-scope list (automorphism `_ROUND` untouched — not flagged by the lint since it compares rounded copies, not attributes; if flagged, waive it).
- **Type consistency:** `_tied_with_extremum` defined Task 3, consumed Tasks 4; `resolve_ac_sia_tie`/`resolve_ac_nexus_tie`/`set_ties` defined Task 5 and consumed there only; `numerics.*` signatures uniform across tasks.
- **Placeholder scan:** every code step carries the code; steps that depend on reading the live API (Transition signature, Finding tone values, fixture tag encoding, AcSIA `__init__`/`__slots__`) say exactly what to read and what property the result must satisfy.
