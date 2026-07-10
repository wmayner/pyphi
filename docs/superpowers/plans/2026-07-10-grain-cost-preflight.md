# Grain-Search Cost Pre-Flight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `SearchBounds.estimate(substrate)` returns a `SearchEstimate` with candidate counts and cost weights for a grain search, computed by pure counting — no TPM construction, no φₛ.

**Architecture:** A new `pyphi/macro/estimate.py` runs a counting walk that mirrors `_derive_units`/`complexes()` control flow, driving the real enumerators (`_decompositions`, `_apportionments`, `_assemble_systems`, `candidate_mappings`) with featherweight `_StandIn` units (footprint, grain, apportionment, multiplicity). All-pass assumption gives an exact worst case; level-1 judgments are exact; `max_depth=0` is fully exact (up to run-time unreachable-state discards).

**Tech Stack:** Python 3.13, pytest.

**Spec:** `docs/superpowers/specs/2026-07-10-grain-cost-preflight-design.md`

## Global Constraints

- `uv run` for all python. Never `--no-verify`. Stage only files the task touches. Pre-commit hooks must pass; if a hook modifies files, re-stage and re-commit.
- Docstrings NumPy-style, final-state impersonal voice, Unicode symbols (φₛ). No planning artifacts in code, comments, or changelog fragments.
- No `PyPhiFloat` (all counting is integer).
- Counts must come from the real enumerators — do not re-derive combinatorics with formulas.
- Changelog fragment per user-facing change.
- Full verification at the end: `uv run pytest` with NO path argument.

## Reference values (verified live, 2026-07-10, presets.iit4_2023)

- Partition counts under `DIRECTED_SET_PARTITION`: m = 1 → 1, 2 → 3, 3 → 22, 4 → 150, 5 → 1,061, 6 → 7,896.
- min-substrate (n = 2, state (0, 0)): `complexes()` records — defaults: **8**; `mappings="EXHAUSTIVE"`: **10**; `max_depth=0`: **3**. Hand-verified worst-case walk: defaults → 3 judgment-phase combos ({0}, {1}, {0,1} micro pair) + 5 mapped variants of the one valid decomposition = 8 (worst case achieved: the only candidate passes); EXHAUSTIVE → 3 + 7 = 10. `candidate_mappings(2, 1, ...)`: FAMILIES → 5, EXHAUSTIVE → 7.
- bu-substrate (n = 3, state (0, 0, 0)) at `max_depth=0`: records = **6**, subsets = 7 — one candidate's state is unreachable under its own TPM (`StateUnreachableError`, no record). The estimate counts 7: enumeration bounds candidates enumerated, not records.
- min defaults expected estimate: `judgments_by_level=(1,)`, final pool 2 micro + 5 variants = 7 concrete units, `assemblies_upper_bound=8`, `distinct_systems_upper_bound=8`, `systems_by_unit_count={1: 7, 2: 1}`, `partitions_by_unit_count={1: 1, 2: 3}`, `partition_sweeps_upper_bound=10`, `construction_keys_upper_bound=3` ({0}·g1, {1}·g1, {0,1}·g1).

---

### Task 1: `SearchEstimate` + the counting walk

**Files:**
- Create: `pyphi/macro/estimate.py`
- Test: `test/macro/test_macro_estimate.py` (new)

**Interfaces:**
- Consumes (all existing, `pyphi/macro/search.py`): `SearchBounds`; `_decompositions(footprint, pool, *, allow_singleton)` (reads `unit.micro_constituents`, `unit.micro_grain`); `_apportionments(n, footprint, inherited, bounds)`; `_assemble_systems(pool, background_cap)` (reads `unit.micro_constituents`, `unit.background_apportionment`); `candidate_mappings(num_constituents, update_grain, bounds)` (raises `ValueError` for EXHAUSTIVE above `exhaustive_cap`).
- Produces: `estimate_search(bounds: SearchBounds, n: int, limit: int = 1_000_000) -> SearchEstimate`; `SearchEstimate` frozen dataclass with fields `n, bounds, judgments_by_level, worst_case_pool_by_level, assemblies_upper_bound, distinct_systems_upper_bound, systems_by_unit_count, partitions_by_unit_count, partition_sweeps_upper_bound, construction_keys_upper_bound, is_exact, truncated` (Task 2 fills the two partition fields with real values; in this task they are `{}` and `0`).

- [ ] **Step 1: Write the failing tests** — create `test/macro/test_macro_estimate.py`:

```python
"""Tests for pyphi.macro.estimate: the grain-search cost pre-flight."""

import pytest

from pyphi import config
from pyphi.conf import presets
from pyphi.macro.estimate import SearchEstimate
from pyphi.macro.estimate import estimate_search
from pyphi.macro.search import SearchBounds


class TestCountingWalk:
    def test_min_defaults_worst_case(self):
        # n=2 at default bounds: one candidate decomposition ({0},{1});
        # judgment evaluates it + its two singleton competitors; its 5
        # FAMILIES-mapped variants join the pool; the sweep adds them.
        est = estimate_search(SearchBounds(), 2)
        assert est.judgments_by_level == (1,)
        assert est.worst_case_pool_by_level == (7,)  # 2 micro + 5 variants
        assert est.assemblies_upper_bound == 8
        assert est.distinct_systems_upper_bound == 8
        assert est.systems_by_unit_count == {1: 7, 2: 1}
        assert est.construction_keys_upper_bound == 3
        assert est.is_exact is False
        assert est.truncated is False

    def test_min_exhaustive_worst_case(self):
        est = estimate_search(SearchBounds(mappings="EXHAUSTIVE"), 2)
        assert est.distinct_systems_upper_bound == 10  # 3 + 7 mappings

    def test_depth_zero_is_exact(self):
        est = estimate_search(SearchBounds(max_depth=0), 2)
        assert est.is_exact is True
        assert est.judgments_by_level == ()
        assert est.distinct_systems_upper_bound == 3  # {0}, {1}, {0,1}
        assert est.assemblies_upper_bound == 3
        assert est.systems_by_unit_count == {1: 2, 2: 1}

    def test_depth_zero_counts_subsets(self):
        est = estimate_search(SearchBounds(max_depth=0), 3)
        assert est.distinct_systems_upper_bound == 7  # nonempty subsets
        assert est.systems_by_unit_count == {1: 3, 2: 3, 3: 1}

    def test_truncation(self):
        est = estimate_search(SearchBounds(), 4, limit=5)
        assert est.truncated is True
        assert est.is_exact is False
        assert 0 < est.distinct_systems_upper_bound <= 5

    def test_exhaustive_above_cap_raises_without_running(self):
        bounds = SearchBounds(
            mappings="EXHAUSTIVE", max_constituents=4, exhaustive_cap=8
        )
        with pytest.raises(ValueError, match="exhaustive_cap"):
            estimate_search(bounds, 4)

    def test_monotone_in_depth(self):
        shallow = estimate_search(SearchBounds(max_depth=0), 3)
        deep = estimate_search(SearchBounds(max_depth=1), 3)
        assert (
            deep.distinct_systems_upper_bound
            >= shallow.distinct_systems_upper_bound
        )


class TestAgainstRealSweeps:
    def test_min_defaults_estimate_equals_records(self):
        from test.macro.test_macro_criteria import min_substrate

        from pyphi.macro.search import complexes

        substrate = min_substrate()
        with config.override(**presets.iit4_2023):
            result = complexes(substrate, (0, 0), SearchBounds())
        est = SearchBounds().estimate(substrate)
        # Worst case achieved: min's only candidate decomposition passes.
        assert est.distinct_systems_upper_bound == len(result.records) == 8

    def test_min_exhaustive_estimate_equals_records(self):
        from test.macro.test_macro_criteria import min_substrate

        from pyphi.macro.search import complexes

        substrate = min_substrate()
        bounds = SearchBounds(mappings="EXHAUSTIVE")
        with config.override(**presets.iit4_2023):
            result = complexes(substrate, (0, 0), bounds)
        est = bounds.estimate(substrate)
        assert est.distinct_systems_upper_bound == len(result.records) == 10

    def test_bu_depth_zero_unreachable_gap(self):
        from test.macro.test_macro_criteria import bu_substrate

        from pyphi.macro.search import complexes

        substrate = bu_substrate()
        bounds = SearchBounds(max_depth=0)
        with config.override(**presets.iit4_2023):
            result = complexes(substrate, (0, 0, 0), bounds)
        est = bounds.estimate(substrate)
        # One candidate's state is unreachable under its own TPM and is
        # discarded at run time; the enumeration cannot predict that.
        assert est.distinct_systems_upper_bound == 7
        assert len(result.records) == 6

    def test_bu_defaults_upper_bound(self):
        from test.macro.test_macro_criteria import bu_substrate

        from pyphi.macro.search import complexes

        substrate = bu_substrate()
        with config.override(**presets.iit4_2023):
            result = complexes(substrate, (0, 0, 0), SearchBounds())
        est = SearchBounds().estimate(substrate)
        assert est.distinct_systems_upper_bound >= len(result.records)

    def test_min_depth_zero_exact_against_records(self):
        from test.macro.test_macro_criteria import min_substrate

        from pyphi.macro.search import complexes

        substrate = min_substrate()
        bounds = SearchBounds(max_depth=0)
        with config.override(**presets.iit4_2023):
            result = complexes(substrate, (0, 0), bounds)
        est = bounds.estimate(substrate)
        assert est.is_exact is True
        assert est.distinct_systems_upper_bound == len(result.records) == 3

    def test_chain_depth_zero_upper_bound(self):
        from test.macro.test_macro_search import decaying_chain_substrate

        from pyphi.macro.search import complexes

        substrate = decaying_chain_substrate()
        bounds = SearchBounds(max_depth=0)
        with config.override(**presets.iit4_2023):
            result = complexes(substrate, (0, 0, 0, 0), bounds)
        est = bounds.estimate(substrate)
        # 15 nonempty subsets enumerated; unreachable-state candidates
        # (if any) are discarded at run time, so records may be fewer.
        assert est.distinct_systems_upper_bound == 15
        assert est.distinct_systems_upper_bound >= len(result.records)
```

Note: `SearchBounds.estimate` is added in this task (Step 3) as a thin
delegator so the integration tests read naturally.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/macro/test_macro_estimate.py -x -q`
Expected: FAIL at import — `No module named 'pyphi.macro.estimate'`.

- [ ] **Step 3: Implement** — create `pyphi/macro/estimate.py`:

```python
# macro/estimate.py
"""Analytic pre-flight for the bounded intrinsic-unit search.

Counts the candidate systems a :func:`pyphi.macro.complexes` sweep would
evaluate, without constructing macro TPMs or computing φₛ. The counting
walk mirrors the search's own control flow and drives the search's own
enumeration functions with lightweight stand-in units, so the counts
reflect the implementation exactly.

Because the search prunes adaptively — only decompositions that pass the
intrinsic-unit criteria spawn mapped variants — counts beyond the first
macroing level assume every candidate passes: an exact worst case. At
``max_depth=0`` no judgment happens and the enumeration is exact. Even
then, a candidate whose state is unreachable under its own TPM is
discarded at run time, which the enumeration cannot predict, so counts
bound the candidates *enumerated*, and ``ComplexesResult.records`` can be
smaller.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from pyphi.display import Description
from pyphi.display import Displayable
from pyphi.display import Row
from pyphi.display import Section
from pyphi.macro.search import SearchBounds
from pyphi.macro.search import _apportionments
from pyphi.macro.search import _assemble_systems
from pyphi.macro.search import _decompositions
from pyphi.macro.search import candidate_mappings
from pyphi.models.pandas import ToPandasMixin

_PARTITION_COUNT_CAP = 8


@dataclass(frozen=True)
class _StandIn:
    """One unit shape standing for every mapped variant that shares it.

    Carries exactly the attributes the search's enumeration functions
    read, plus the number of concrete units it represents.
    """

    micro_constituents: tuple[int, ...]
    micro_grain: int
    background_apportionment: tuple[int, ...]
    multiplicity: int = 1


def _shape_key(unit: _StandIn):
    return (
        unit.micro_constituents,
        unit.micro_grain,
        unit.background_apportionment,
    )


def _combo_count(combo) -> int:
    total = 1
    for unit in combo:
        total *= unit.multiplicity
    return total


def _merged(units) -> list[_StandIn]:
    """Merge stand-ins sharing a shape key, summing multiplicities."""
    acc: dict[tuple, _StandIn] = {}
    for unit in units:
        key = _shape_key(unit)
        prior = acc.get(key)
        if prior is None:
            acc[key] = unit
        else:
            acc[key] = _StandIn(
                unit.micro_constituents,
                unit.micro_grain,
                unit.background_apportionment,
                prior.multiplicity + unit.multiplicity,
            )
    return list(acc.values())


def _variant_standins(V, W, bounds: SearchBounds, base: int) -> list[_StandIn]:
    """Stand-ins for the mapped and grained variants ``base`` concrete
    decompositions of shape ``V`` would spawn, mirroring the search's
    variant construction (a single-constituent decomposition is pure
    grain raising, so its variants start at update grain 2)."""
    size = len(V)
    grain = V[0].micro_grain
    footprint = tuple(
        sorted(set().union(*(set(u.micro_constituents) for u in V)))
    )
    min_grain = 2 if size == 1 else 1
    out = []
    for update_grain in range(min_grain, bounds.max_update_grain + 1):
        num_mappings = len(candidate_mappings(size, update_grain, bounds))
        if num_mappings:
            out.append(
                _StandIn(footprint, update_grain * grain, W, base * num_mappings)
            )
    return out


class _LimitReached(Exception):
    pass


class _Counter:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.spent = 0

    def charge(self, amount: int) -> None:
        self.spent += amount
        if self.spent > self.limit:
            raise _LimitReached


@dataclass(frozen=True)
class SearchEstimate(Displayable, ToPandasMixin):
    """The size and shape of a grain search, before running it.

    All quantities are exact worst cases under the assumption that every
    judged decomposition passes the intrinsic-unit criteria; the search
    itself can only do less. See the module docstring for what "exact"
    can and cannot promise.

    Attributes
    ----------
    n : int
        Micro universe size the estimate was computed for.
    bounds : SearchBounds
        The bounds the estimate describes.
    judgments_by_level : tuple[int, ...]
        Candidate decompositions judged per macroing level (the first
        level's count is exact; deeper levels are worst-case).
    worst_case_pool_by_level : tuple[int, ...]
        Concrete units in the pool after each level, all-pass.
    assemblies_upper_bound : int
        Candidate systems the final sweep assembles (Eq. 18).
    distinct_systems_upper_bound : int
        Deduplicated candidate systems across judgment candidates,
        judgment competitors, and the final sweep — the number to compare
        with ``len(ComplexesResult.records)``.
    systems_by_unit_count : dict[int, int]
        Final-sweep candidates bucketed by macro unit count m.
    partitions_by_unit_count : dict[int, int]
        Partitions per system irreducibility analysis at each m, under
        the active system partition scheme.
    partition_sweeps_upper_bound : int
        Σ over m of systems × partitions — the SIA-cost axis.
    construction_keys_upper_bound : int
        Distinct (footprint, grain) construction keys — the Θ(τ·4ⁿ)
        construction-cost axis.
    is_exact : bool
        Whether the enumeration is exact rather than an upper bound
        (``max_depth == 0`` and not truncated).
    truncated : bool
        The counting hit its ``limit`` (or a bucket exceeded the
        partition-count cap); counts are then lower bounds of the bound.
    """

    n: int
    bounds: SearchBounds
    judgments_by_level: tuple[int, ...]
    worst_case_pool_by_level: tuple[int, ...]
    assemblies_upper_bound: int
    distinct_systems_upper_bound: int
    systems_by_unit_count: dict[int, int]
    partitions_by_unit_count: dict[int, int]
    partition_sweeps_upper_bound: int
    construction_keys_upper_bound: int
    is_exact: bool
    truncated: bool

    def _qualifier(self) -> str:
        if self.truncated:
            return "≥"
        return "=" if self.is_exact else "≤"

    def _pandas_record(self) -> dict:
        return {
            "n": self.n,
            "distinct_systems_upper_bound": self.distinct_systems_upper_bound,
            "assemblies_upper_bound": self.assemblies_upper_bound,
            "partition_sweeps_upper_bound": self.partition_sweeps_upper_bound,
            "construction_keys_upper_bound": self.construction_keys_upper_bound,
            "is_exact": self.is_exact,
            "truncated": self.truncated,
        }

    def _describe(self, verbosity: int) -> Description:  # noqa: ARG002
        q = self._qualifier()
        rows = [
            Row("Candidate systems", f"{q} {self.distinct_systems_upper_bound}"),
            Row("Assembly sweep", f"{q} {self.assemblies_upper_bound}"),
            Row("Partition sweeps", f"{q} {self.partition_sweeps_upper_bound}"),
            Row("Construction keys", f"{q} {self.construction_keys_upper_bound}"),
            Row("Judgments by level", str(self.judgments_by_level)),
            Row("Pool by level", str(self.worst_case_pool_by_level)),
            Row("Exact", self.is_exact),
            Row("Truncated", self.truncated),
        ]
        bucket_rows = tuple(
            Row(
                f"m = {m}",
                f"{self.systems_by_unit_count[m]} systems × "
                f"{self.partitions_by_unit_count.get(m, '>cap')} partitions",
            )
            for m in sorted(self.systems_by_unit_count)
        )
        sections = [Section(rows=tuple(rows))]
        if bucket_rows:
            sections.append(
                Section(label="Systems by unit count", rows=bucket_rows)
            )
        return Description(
            title="SearchEstimate",
            subtitle=f"{self.n} micro units",
            sections=tuple(sections),
            compact=(
                f"SearchEstimate(n={self.n}, systems "
                f"{q} {self.distinct_systems_upper_bound})"
            ),
        )


def _partition_counts(ms) -> dict[int, int]:
    from pyphi.partition import system_partitions

    return {
        m: sum(1 for _ in system_partitions(tuple(range(m))))
        for m in ms
        if m <= _PARTITION_COUNT_CAP
    }


def estimate_search(
    bounds: SearchBounds, n: int, limit: int = 1_000_000
) -> SearchEstimate:
    """Count the candidate systems a grain search over ``n`` micro units
    would evaluate under ``bounds``, without running it.

    Notes
    -----
    The walk's own running time is proportional to the counts it
    produces; ``limit`` bounds the total counted items and stops the walk
    early (``truncated=True``) when exceeded. ``mappings="EXHAUSTIVE"``
    above ``exhaustive_cap`` raises the same ``ValueError`` the search
    itself would.
    """
    counter = _Counter(limit)
    truncated = False

    # Global registry of distinct unit-combos (the memo's worst case).
    # A key re-recorded with a larger count supersedes: concrete variants
    # only accumulate, so the later set contains the earlier one.
    combos: dict[tuple, int] = {}

    def record_combo(combo) -> None:
        key = tuple(sorted(_shape_key(u) for u in combo))
        count = _combo_count(combo)
        prior = combos.get(key, 0)
        if count > prior:
            counter.charge(count - prior)
            combos[key] = count

    indices = tuple(range(n))
    pool: list[_StandIn] = [_StandIn((i,), 1, ()) for i in indices]

    judgments_by_level: list[int] = []
    pool_by_level: list[int] = []
    # Shape-level judgment ledger: concrete decompositions already judged
    # per (decomposition shape, W). A shape re-encountered after its
    # members gained variants is re-judged for the new concretes only.
    judged: dict[tuple, int] = {}
    min_size = 1 if bounds.max_update_grain > 1 else 2
    max_size = min(n, bounds.max_constituents)
    assemblies = 0
    sweep_buckets: dict[int, int] = {}

    try:
        for unit in pool:
            record_combo((unit,))
        for _level in range(bounds.max_depth):
            pool_prev = tuple(pool)
            judged_this_level = 0
            emitted_any = False
            for size in range(min_size, max_size + 1):
                pool_at_class_start = tuple(pool)
                new_units: list[_StandIn] = []
                for footprint in itertools.combinations(indices, size):
                    for V in _decompositions(
                        footprint,
                        pool_prev,
                        allow_singleton=bounds.max_update_grain > 1,
                    ):
                        inherited = set().union(
                            *(set(u.background_apportionment) for u in V)
                        )
                        for W in _apportionments(n, footprint, inherited, bounds):
                            key = (
                                tuple(sorted(_shape_key(u) for u in V)),
                                W,
                            )
                            base = _combo_count(V)
                            delta = base - judged.get(key, 0)
                            if delta <= 0:
                                continue
                            judged[key] = base
                            counter.charge(delta)
                            judged_this_level += delta
                            record_combo(V)
                            fp = set(footprint)
                            allowed = set(W)
                            members = [
                                u
                                for u in pool_at_class_start
                                if set(u.micro_constituents) < fp
                                and set(u.background_apportionment) <= allowed
                            ]
                            for combo in _assemble_systems(
                                members, bounds.max_background
                            ):
                                record_combo(combo)
                            new_units.extend(
                                _variant_standins(V, W, bounds, delta)
                            )
                pool = _merged([*pool, *new_units])
                emitted_any = emitted_any or bool(new_units)
            judgments_by_level.append(judged_this_level)
            pool_by_level.append(sum(u.multiplicity for u in pool))
            if not emitted_any:
                break
        for combo in _assemble_systems(pool, bounds.max_background):
            record_combo(combo)
            count = _combo_count(combo)
            counter.charge(count)
            assemblies += count
            m = len(combo)
            sweep_buckets[m] = sweep_buckets.get(m, 0) + count
    except _LimitReached:
        truncated = True

    partitions = _partition_counts(sorted(sweep_buckets))
    if any(m > _PARTITION_COUNT_CAP for m in sweep_buckets):
        truncated = True
    partition_sweeps = sum(
        count * partitions[m]
        for m, count in sweep_buckets.items()
        if m in partitions
    )
    construction_keys = len(
        {(u.micro_constituents, u.micro_grain) for u in pool}
    )
    return SearchEstimate(
        n=n,
        bounds=bounds,
        judgments_by_level=tuple(judgments_by_level),
        worst_case_pool_by_level=tuple(pool_by_level),
        assemblies_upper_bound=assemblies,
        distinct_systems_upper_bound=sum(combos.values()),
        systems_by_unit_count=sweep_buckets,
        partitions_by_unit_count=partitions,
        partition_sweeps_upper_bound=partition_sweeps,
        construction_keys_upper_bound=construction_keys,
        is_exact=bounds.max_depth == 0 and not truncated,
        truncated=truncated,
    )
```

Also add the delegator method to `SearchBounds` in `pyphi/macro/search.py`
(after the `max_micro_grain` property):

```python
    def estimate(self, substrate, limit: int = 1_000_000):
        """Pre-flight estimate of the sweep these bounds define over
        ``substrate``, by pure counting; see
        :func:`pyphi.macro.estimate.estimate_search`. Only the
        substrate's size is read — no TPM is constructed and no φₛ is
        computed."""
        from pyphi.macro.estimate import estimate_search

        return estimate_search(self, substrate.size, limit=limit)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest test/macro/test_macro_estimate.py -v`
Expected: all PASS. The two partition fields are already populated by
`_partition_counts` in this task (the split with Task 2 is verification,
not implementation), so `TestCountingWalk::test_min_defaults_worst_case`
must pass in full. If a hand-verified anchor (8/10/3/7) does NOT
reproduce, do not adjust the expected number — the walk deviates from the
search's control flow; debug against `pyphi/macro/search.py` lines
504–601 (`_derive_units`) and 817–887 (`complexes`) or escalate.

- [ ] **Step 5: Commit**

```bash
git add pyphi/macro/estimate.py pyphi/macro/search.py test/macro/test_macro_estimate.py
git commit -m "Add the grain-search cost pre-flight counting walk

SearchBounds.estimate(substrate) counts the candidate systems a sweep
would evaluate by driving the search's own enumeration functions with
stand-in units -- no TPM construction, no phi. Level-1 judgment counts
are exact; deeper levels and the assembly sweep are exact worst cases
under the all-pass assumption."
```

### Task 2: Partition-weight verification

**Files:**
- Test: `test/macro/test_macro_estimate.py` (append)

**Interfaces:**
- Consumes: `estimate_search` and `SearchEstimate.partitions_by_unit_count` / `partition_sweeps_upper_bound` (Task 1); `pyphi.partition.system_partitions`.

- [ ] **Step 1: Write the tests** — append to `test/macro/test_macro_estimate.py`:

```python
class TestPartitionWeights:
    def test_partition_counts_pinned(self):
        # Measured values under DIRECTED_SET_PARTITION.
        with config.override(**presets.iit4_2023):
            est = estimate_search(SearchBounds(max_depth=0), 5)
        assert est.partitions_by_unit_count == {
            1: 1,
            2: 3,
            3: 22,
            4: 150,
            5: 1061,
        }

    def test_partition_sweeps_are_weighted_sum(self):
        with config.override(**presets.iit4_2023):
            est = estimate_search(SearchBounds(max_depth=0), 3)
        expected = sum(
            est.systems_by_unit_count[m] * est.partitions_by_unit_count[m]
            for m in est.systems_by_unit_count
        )
        assert est.partition_sweeps_upper_bound == expected
        # n=3 subsets: 3 singletons + 3 pairs + 1 triple.
        assert expected == 3 * 1 + 3 * 3 + 1 * 22
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest test/macro/test_macro_estimate.py -v -k Partition`
Expected: PASS (implementation landed in Task 1; these pin the weights
against live `system_partitions` enumeration). If a count differs,
investigate — do not adjust the pinned values.

- [ ] **Step 3: Commit**

```bash
git add test/macro/test_macro_estimate.py
git commit -m "Pin grain-estimate partition weights against live enumeration"
```

### Task 3: Display card, exports, changelog

**Files:**
- Modify: `pyphi/macro/__init__.py` (imports + `__all__`)
- Test: `test/macro/test_macro_estimate.py` (append)
- Create: `changelog.d/grain-cost-preflight.feature.md`

**Interfaces:**
- Consumes: `SearchEstimate._describe` / `_pandas_record` (implemented in Task 1).
- Produces: `pyphi.macro.SearchEstimate`, `pyphi.macro.estimate_search`.

- [ ] **Step 1: Write the tests** — append to `test/macro/test_macro_estimate.py`:

```python
class TestSurfaces:
    def test_display_card_headline_rows(self):
        est = estimate_search(SearchBounds(), 2)
        desc = est._describe(verbosity=2)
        labels = [
            row.label for section in desc.sections for row in section.rows
        ]
        assert "Candidate systems" in labels
        assert "Partition sweeps" in labels
        assert "m = 1" in labels  # bucket section

    def test_qualifier_tracks_exactness(self):
        exact = estimate_search(SearchBounds(max_depth=0), 2)
        bound = estimate_search(SearchBounds(), 2)
        assert "= 3" in exact._describe(2).compact
        assert "≤ 8" in bound._describe(2).compact

    def test_pandas_record_scalars(self):
        est = estimate_search(SearchBounds(), 2)
        record = est._pandas_record()
        assert record["distinct_systems_upper_bound"] == 8
        assert record["is_exact"] is False

    def test_package_exports(self):
        import pyphi.macro

        assert pyphi.macro.SearchEstimate is SearchEstimate
        assert pyphi.macro.estimate_search is estimate_search
```

- [ ] **Step 2: Run to verify the export test fails**

Run: `uv run pytest test/macro/test_macro_estimate.py -v -k Surfaces`
Expected: `test_package_exports` FAILS (`AttributeError`); the display and
pandas tests pass already.

- [ ] **Step 3: Wire the exports** — in `pyphi/macro/__init__.py`, add (in
alphabetical import order with the existing block):

```python
from pyphi.macro.estimate import SearchEstimate
from pyphi.macro.estimate import estimate_search
```

and add `"SearchEstimate"` and `"estimate_search"` to `__all__` (keep it
sorted).

- [ ] **Step 4: Run the surface tests**

Run: `uv run pytest test/macro/test_macro_estimate.py -v`
Expected: all PASS.

- [ ] **Step 5: Changelog fragment**

```bash
echo 'Added `SearchBounds.estimate(substrate)`: an analytic pre-flight for the grain search that counts candidate systems, assemblies, partition sweeps, and construction keys — by pure counting through the search'"'"'s own enumeration, with no TPM construction and no φₛ. Level-1 judgment counts are exact; deeper levels report an exact worst case.' > changelog.d/grain-cost-preflight.feature.md
```

- [ ] **Step 6: Commit**

```bash
git add pyphi/macro/__init__.py test/macro/test_macro_estimate.py changelog.d/grain-cost-preflight.feature.md
git commit -m "Export the grain-search estimate surfaces

pyphi.macro.SearchEstimate and estimate_search join the package
namespace; the estimate renders a display card and a pandas record."
```

### Task 4: Full verification

- [ ] **Step 1: Run the full suite with the doctest sweep**

Run: `uv run pytest` (NO path argument)
Expected: all green (recent baseline on main: 3110 passed, 284 skipped).

- [ ] **Step 2: If anything fails,** fix within the task that introduced it and re-run; do not proceed with failures.
