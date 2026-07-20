# Analysis Cost Estimate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An analytic workload pre-flight for single-system analyses (`pyphi.estimate_analysis` in new `pyphi/cost.py`), with the MCP `analyze` guard gating on estimated counts instead of node counts and a new `estimate_cost` MCP tool.

**Architecture:** Counts are produced by driving the real enumeration machinery (`system_partitions`, `Substrate.potential_purviews` via a reference-state `System`, `mechanism_partitions`) under the active configuration, memoized by scheme and size with seeded default-scheme values, budgeted by a work counter that stops the walk when exceeded. The shared counter and system-partition memo move from `pyphi/macro/estimate.py` into `pyphi/cost.py`.

**Tech Stack:** Python 3.13, dataclasses, pytest; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-20-analysis-cost-estimate-design.md`

## Global Constraints

- Formalism pinning in tests: complete presets only (`pyphi.config.override(**presets.iit4_2026, ...)`, `IIT_3_CONFIG` from `test/conftest.py`); never a hand-listed subset of `iit.*` fields.
- Docstrings: NumPy style (underlined sections), final-state impersonal voice, Unicode symbols. No planning-artifact references (item numbers, spec/plan mentions) in code, docstrings, or changelog.
- Fidelity: counts and structural weights only — never predicted seconds.
- Commit messages end with the two standard trailers (`Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and the `Claude-Session:` line). Never `--no-verify`. After every commit run `git log --oneline -1` to confirm it landed (ruff-format can abort silently; re-add and re-commit if so).
- Never pipe pytest through `tail`/`head`; redirect to a log file and read the summary line.
- The MCP thresholds are `_SIA_PARTITION_LIMIT = 4_419_572` (the `DIRECTED_SET_PARTITION` count for 9 fully connected binary units, verified by direct enumeration) and `_CES_SWEEP_LIMIT = 100_000_000` (admits the fully connected 6-unit binary CES at 31,938,830 sweeps; refuses the 7-unit one at 1,450,456,298).
- `pyphi/macro/estimate.py` keeps its exact current behavior; it only imports the moved helpers.

---

### Task 1: Shared counting utilities in `pyphi/cost.py`

**Files:**
- Create: `pyphi/cost.py`
- Modify: `pyphi/macro/estimate.py`
- Test: `test/test_cost.py` (new; seed-verification classes only in this task)

**Interfaces:**
- Consumes: `pyphi.partition.system_partitions`, `pyphi.partition.mechanism_partitions`, `pyphi.conf.config`.
- Produces (used by Task 2 and by `pyphi/macro/estimate.py`): `_Counter(limit).charge(amount)` raising `_LimitReached`; `_PARTITION_COUNT_CAP: int = 6`; `_PARTITION_COUNT_MEMO: dict[tuple[str, int], int]`; `_MECHANISM_PARTITION_COUNT_MEMO: dict[tuple[str, int, int], int]`; `_partition_counts(ms) -> dict[int, int]`; `_system_partition_count(m: int, counter: _Counter) -> int`; `_mechanism_partition_count(msize: int, psize: int, counter: _Counter) -> int`.

- [ ] **Step 1: Write the failing seed-verification tests**

Create `test/test_cost.py`:

```python
"""Tests for pyphi.cost: the single-system analysis workload pre-flight."""

import numpy as np
import pytest

import pyphi
from pyphi import examples
from pyphi.conf import presets
from pyphi.cost import _MECHANISM_PARTITION_COUNT_MEMO
from pyphi.cost import _PARTITION_COUNT_MEMO
from pyphi.partition import mechanism_partitions
from pyphi.partition import system_partitions


@pytest.fixture(autouse=True)
def _pin_formalism():
    with pyphi.config.override(**presets.iit4_2026, progress_bars=False):
        yield


def _dense3():
    return examples.basic_substrate(cm=np.ones((3, 3)))


class TestSeeds:
    def test_system_partition_seeds_match_enumeration(self):
        for m in range(1, 7):
            direct = sum(1 for _ in system_partitions(tuple(range(m))))
            assert _PARTITION_COUNT_MEMO[("DIRECTED_SET_PARTITION", m)] == direct

    @pytest.mark.slow
    def test_system_partition_seeds_match_enumeration_large(self):
        # m = 9 (240 s to enumerate) is excluded; its seed was verified by
        # one direct enumeration of the same generator.
        for m in (7, 8):
            direct = sum(1 for _ in system_partitions(tuple(range(m))))
            assert _PARTITION_COUNT_MEMO[("DIRECTED_SET_PARTITION", m)] == direct

    def test_mechanism_partition_seeds_match_enumeration(self):
        for a in range(1, 6):
            for b in range(1, 6):
                direct = sum(
                    1
                    for _ in mechanism_partitions(
                        tuple(range(a)), tuple(range(a, a + b))
                    )
                )
                assert (
                    _MECHANISM_PARTITION_COUNT_MEMO[("JOINT_PARTITION_ALL", a, b)]
                    == direct
                )

    @pytest.mark.slow
    def test_mechanism_partition_seeds_match_enumeration_large(self):
        # Pairs (7, 7) (218 s), (7, 6) (36 s), and (6, 7) (24 s) are
        # excluded; those seeds were verified by one direct enumeration of
        # the same generator.
        pairs = [
            (a, b)
            for a in range(1, 8)
            for b in range(1, 8)
            if (a >= 6 or b >= 6) and (a, b) not in {(6, 7), (7, 6), (7, 7)}
        ]
        for a, b in pairs:
            direct = sum(
                1
                for _ in mechanism_partitions(
                    tuple(range(a)), tuple(range(a, a + b))
                )
            )
            assert (
                _MECHANISM_PARTITION_COUNT_MEMO[("JOINT_PARTITION_ALL", a, b)]
                == direct
            )
```

(`examples` and `_dense3` are already used in this task; Task 2 extends this file.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest test/test_cost.py -x -q > /tmp/t1a.log 2>&1; cat /tmp/t1a.log`
Expected: collection error — `ModuleNotFoundError: No module named 'pyphi.cost'`.

- [ ] **Step 3: Create `pyphi/cost.py`**

```python
"""Analytic workload counting for single-system analyses.

Counts the work a :func:`pyphi.analyze` call would perform — system
partitions swept by the system irreducibility analysis, candidate
mechanisms, connectivity-pruned purview evaluations, and mechanism
partitions per (mechanism, purview) pair — without computing any φ.
Counts are produced by driving the same enumeration machinery the
analysis uses under the active configuration, so the partition schemes,
the connectivity, and the alphabet are all reflected exactly.

All quantities are counts and structural weights. Wall time depends on
the machine and configuration and is never predicted.
"""

from __future__ import annotations

_PARTITION_COUNT_CAP = 6


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


# Partition counts keyed by (system partition scheme name, m). Enumerating
# the partitions of m elements is the same regardless of substrate, so the
# count is memoized across calls at module scope. The default scheme's
# counts are seeded from direct enumeration of ``system_partitions``; the
# seed-verification tests re-enumerate them.
_PARTITION_COUNT_MEMO: dict[tuple[str, int], int] = {
    ("DIRECTED_SET_PARTITION", 1): 1,
    ("DIRECTED_SET_PARTITION", 2): 3,
    ("DIRECTED_SET_PARTITION", 3): 22,
    ("DIRECTED_SET_PARTITION", 4): 150,
    ("DIRECTED_SET_PARTITION", 5): 1_061,
    ("DIRECTED_SET_PARTITION", 6): 7_896,
    ("DIRECTED_SET_PARTITION", 7): 61_888,
    ("DIRECTED_SET_PARTITION", 8): 510_313,
    ("DIRECTED_SET_PARTITION", 9): 4_419_572,
}

# Mechanism-partition counts keyed by (mechanism partition scheme name,
# |mechanism|, |purview|); the count depends only on the two sizes. The
# default scheme's counts are seeded from direct enumeration of
# ``mechanism_partitions``; the seed-verification tests re-enumerate them.
_MECHANISM_PARTITION_COUNT_MEMO: dict[tuple[str, int, int], int] = {
    ("JOINT_PARTITION_ALL", 1, 1): 1,
    ("JOINT_PARTITION_ALL", 1, 2): 1,
    ("JOINT_PARTITION_ALL", 1, 3): 1,
    ("JOINT_PARTITION_ALL", 1, 4): 1,
    ("JOINT_PARTITION_ALL", 1, 5): 1,
    ("JOINT_PARTITION_ALL", 1, 6): 1,
    ("JOINT_PARTITION_ALL", 1, 7): 1,
    ("JOINT_PARTITION_ALL", 2, 1): 4,
    ("JOINT_PARTITION_ALL", 2, 2): 10,
    ("JOINT_PARTITION_ALL", 2, 3): 28,
    ("JOINT_PARTITION_ALL", 2, 4): 82,
    ("JOINT_PARTITION_ALL", 2, 5): 244,
    ("JOINT_PARTITION_ALL", 2, 6): 730,
    ("JOINT_PARTITION_ALL", 2, 7): 2_188,
    ("JOINT_PARTITION_ALL", 3, 1): 14,
    ("JOINT_PARTITION_ALL", 3, 2): 44,
    ("JOINT_PARTITION_ALL", 3, 3): 146,
    ("JOINT_PARTITION_ALL", 3, 4): 500,
    ("JOINT_PARTITION_ALL", 3, 5): 1_754,
    ("JOINT_PARTITION_ALL", 3, 6): 6_284,
    ("JOINT_PARTITION_ALL", 3, 7): 22_946,
    ("JOINT_PARTITION_ALL", 4, 1): 51,
    ("JOINT_PARTITION_ALL", 4, 2): 185,
    ("JOINT_PARTITION_ALL", 4, 3): 699,
    ("JOINT_PARTITION_ALL", 4, 4): 2_729,
    ("JOINT_PARTITION_ALL", 4, 5): 10_971,
    ("JOINT_PARTITION_ALL", 4, 6): 45_305,
    ("JOINT_PARTITION_ALL", 4, 7): 191_739,
    ("JOINT_PARTITION_ALL", 5, 1): 202,
    ("JOINT_PARTITION_ALL", 5, 2): 822,
    ("JOINT_PARTITION_ALL", 5, 3): 3_472,
    ("JOINT_PARTITION_ALL", 5, 4): 15_162,
    ("JOINT_PARTITION_ALL", 5, 5): 68_272,
    ("JOINT_PARTITION_ALL", 5, 6): 316_242,
    ("JOINT_PARTITION_ALL", 5, 7): 1_503_592,
    ("JOINT_PARTITION_ALL", 6, 1): 876,
    ("JOINT_PARTITION_ALL", 6, 2): 3_934,
    ("JOINT_PARTITION_ALL", 6, 3): 18_306,
    ("JOINT_PARTITION_ALL", 6, 4): 88_018,
    ("JOINT_PARTITION_ALL", 6, 5): 436_266,
    ("JOINT_PARTITION_ALL", 6, 6): 2_224_354,
    ("JOINT_PARTITION_ALL", 6, 7): 11_643_066,
    ("JOINT_PARTITION_ALL", 7, 1): 4_139,
    ("JOINT_PARTITION_ALL", 7, 2): 20_267,
    ("JOINT_PARTITION_ALL", 7, 3): 102_671,
    ("JOINT_PARTITION_ALL", 7, 4): 536_867,
    ("JOINT_PARTITION_ALL", 7, 5): 2_891_639,
    ("JOINT_PARTITION_ALL", 7, 6): 16_012_187,
    ("JOINT_PARTITION_ALL", 7, 7): 90_995_711,
}


def _partition_counts(ms) -> dict[int, int]:
    """System-partition counts per unit count, for m up to the cap."""
    from pyphi.conf import config
    from pyphi.partition import system_partitions

    scheme = config.formalism.iit.system_partition_scheme
    counts = {}
    for m in ms:
        if m > _PARTITION_COUNT_CAP:
            continue
        key = (scheme, m)
        count = _PARTITION_COUNT_MEMO.get(key)
        if count is None:
            count = sum(1 for _ in system_partitions(tuple(range(m))))
            _PARTITION_COUNT_MEMO[key] = count
        counts[m] = count
    return counts


def _system_partition_count(m: int, counter: _Counter) -> int:
    """Count the system partitions of ``m`` units under the active scheme.

    A memoized count is free; a fresh enumeration charges the counter one
    unit per partition, so an unmemoized (scheme, size) pair cannot exceed
    the walk's work budget.
    """
    from pyphi.conf import config
    from pyphi.partition import system_partitions

    scheme = config.formalism.iit.system_partition_scheme
    key = (scheme, m)
    count = _PARTITION_COUNT_MEMO.get(key)
    if count is None:
        count = 0
        for _ in system_partitions(tuple(range(m))):
            counter.charge(1)
            count += 1
        _PARTITION_COUNT_MEMO[key] = count
    return count


def _mechanism_partition_count(msize: int, psize: int, counter: _Counter) -> int:
    """Count the mechanism partitions of a (``msize``, ``psize``) pair
    under the active scheme, with the same budget behavior as
    :func:`_system_partition_count`.
    """
    from pyphi.conf import config
    from pyphi.partition import mechanism_partitions

    scheme = config.formalism.iit.mechanism_partition_scheme
    key = (scheme, msize, psize)
    count = _MECHANISM_PARTITION_COUNT_MEMO.get(key)
    if count is None:
        count = 0
        mechanism = tuple(range(msize))
        purview = tuple(range(msize, msize + psize))
        for _ in mechanism_partitions(mechanism, purview):
            counter.charge(1)
            count += 1
        _MECHANISM_PARTITION_COUNT_MEMO[key] = count
    return count
```

- [ ] **Step 4: Point `pyphi/macro/estimate.py` at the shared helpers**

In `pyphi/macro/estimate.py`:

1. Delete these blocks (currently at lines 36–41, 108–120, and 238–253):
   - `_PARTITION_COUNT_CAP = 6` and the `_PARTITION_COUNT_MEMO` definition with its comment,
   - `class _LimitReached(Exception)` and `class _Counter`,
   - `def _partition_counts(ms) -> dict[int, int]:` (whole function).
2. Add these imports after `from dataclasses import dataclass` (isort places `pyphi.cost` before `pyphi.display`):

```python
from pyphi.cost import _Counter
from pyphi.cost import _LimitReached
from pyphi.cost import _PARTITION_COUNT_CAP
from pyphi.cost import _partition_counts
```

Everything else in the file is unchanged (`estimate_search` and `SearchEstimate` keep using these names).

- [ ] **Step 5: Run the new tests and the grain-estimate regression**

Run: `uv run pytest test/test_cost.py test/macro/test_macro_estimate.py -q > /tmp/t1b.log 2>&1; cat /tmp/t1b.log`
Expected: all PASS (seed tests green; macro estimate behavior unchanged). Fast-lane seed tests only — the `slow`-marked ones are skipped without `--slow`.

- [ ] **Step 6: Run the slow seed verification once**

Run: `uv run pytest test/test_cost.py -m slow --slow -q > /tmp/t1c.log 2>&1; cat /tmp/t1c.log`
Expected: 2 passed (~90 s: re-enumerates system partitions m=7,8 and the large mechanism pairs).

- [ ] **Step 7: Commit**

```bash
git add pyphi/cost.py pyphi/macro/estimate.py test/test_cost.py
git commit -m "Add shared partition-counting utilities in pyphi.cost"
```

(Trailers per Global Constraints; then `git log --oneline -1`.)

---

### Task 2: `AnalysisEstimate` and `estimate_analysis`

**Files:**
- Modify: `pyphi/cost.py` (append), `pyphi/__init__.py`, `test/test_cost.py` (append)

**Interfaces:**
- Consumes: Task 1's `_Counter`, `_LimitReached`, `_system_partition_count`, `_mechanism_partition_count`.
- Produces: `estimate_analysis(substrate, subset=None, compute=None, limit=1_000_000) -> AnalysisEstimate`; frozen dataclass `AnalysisEstimate` with fields `n_units, state_space_size, compute, system_partitions, mechanisms, purview_evaluations, mechanism_partition_sweeps, relations_closed_form, possible_distinctions, possible_relations, capped`; top-level `pyphi.estimate_analysis` and `pyphi.AnalysisEstimate`.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_cost.py`:

```python
class TestCounts:
    def test_counts_match_direct_enumeration(self):
        est = estimate_analysis(_dense3())
        assert est.n_units == 3
        assert est.compute == "full"
        assert est.state_space_size == 8
        assert est.mechanisms == 7
        assert est.system_partitions == sum(1 for _ in system_partitions((0, 1, 2)))
        # Fully connected: every nonempty purview is a candidate for every
        # mechanism in both directions.
        assert est.purview_evaluations == 7 * 2 * 7
        expected = sum(
            comb(3, a)
            * comb(3, b)
            * 2
            * sum(
                1
                for _ in mechanism_partitions(
                    tuple(range(a)), tuple(range(a, a + b))
                )
            )
            for a in (1, 2, 3)
            for b in (1, 2, 3)
        )
        assert est.mechanism_partition_sweeps == expected
        assert est.relations_closed_form is True
        assert est.possible_distinctions == 7
        assert est.possible_relations == 2**7 - 1
        assert est.capped is False

    def test_dense_3unit_reference_values(self):
        est = estimate_analysis(_dense3())
        assert est.system_partitions == 22
        assert est.purview_evaluations == 98
        assert est.mechanism_partition_sweeps == 1102

    def test_sparse_connectivity_prunes_purviews(self):
        sparse = estimate_analysis(examples.basic_substrate())
        dense = estimate_analysis(_dense3())
        assert sparse.purview_evaluations == 30
        assert sparse.mechanism_partition_sweeps == 526
        assert sparse.purview_evaluations < dense.purview_evaluations
        assert sparse.mechanism_partition_sweeps < dense.mechanism_partition_sweeps

    def test_subset_restricts_the_walk(self):
        est = estimate_analysis(_dense3(), subset=(0, 1))
        assert est.n_units == 2
        assert est.state_space_size == 4
        assert est.mechanisms == 3
        assert est.purview_evaluations == 3 * 2 * 3
        assert est.system_partitions == sum(1 for _ in system_partitions((0, 1)))


class TestScope:
    def test_sia_scope(self):
        est = estimate_analysis(_dense3(), compute="sia")
        assert est.compute == "sia"
        assert est.system_partitions == 22
        assert est.mechanisms is None
        assert est.purview_evaluations is None
        assert est.mechanism_partition_sweeps is None
        assert est.relations_closed_form is None
        assert est.possible_distinctions is None
        assert est.possible_relations is None

    def test_ces_scope(self):
        est = estimate_analysis(_dense3(), compute="ces")
        assert est.compute == "ces"
        assert est.system_partitions is None
        assert est.mechanism_partition_sweeps == 1102

    def test_unknown_compute_raises(self):
        with pytest.raises(ValueError, match="compute"):
            estimate_analysis(_dense3(), compute="everything")


class TestConfigSensitivity:
    def test_system_partition_scheme_changes_the_count(self):
        with pyphi.config.override(
            system_partition_scheme="DIRECTED_BIPARTITION"
        ):
            est = estimate_analysis(_dense3(), compute="sia")
            direct = sum(1 for _ in system_partitions((0, 1, 2)))
            assert est.system_partitions == direct
        default = estimate_analysis(_dense3(), compute="sia")
        assert est.system_partitions != default.system_partitions

    def test_concrete_relations_backend_reports_enumeration(self):
        with pyphi.config.override(relation_computation="CONCRETE"):
            est = estimate_analysis(_dense3())
        assert est.relations_closed_form is False
        assert est.possible_relations == 2**7 - 1

    def test_iit3_counts_without_iit4_context(self):
        with IIT_3_CONFIG:
            est = estimate_analysis(_dense3())
            direct = sum(1 for _ in system_partitions((0, 1, 2)))
            assert est.system_partitions == direct
            assert est.mechanism_partition_sweeps is not None
            assert est.relations_closed_form is None
            assert est.possible_distinctions is None
            assert est.possible_relations is None

    def test_kary_work_axes_without_binary_context(self):
        rng = np.random.default_rng(2026)
        f0 = rng.uniform(size=(3, 3, 3))
        f0 = f0 / f0.sum(axis=-1, keepdims=True)
        f1 = rng.uniform(size=(3, 3, 3))
        f1 = f1 / f1.sum(axis=-1, keepdims=True)
        sub = pyphi.Substrate(
            marginals=[f0, f1], state_space=("LOW", "MID", "HIGH")
        )
        est = estimate_analysis(sub)
        assert est.state_space_size == 9
        assert est.mechanisms == 3
        assert est.purview_evaluations is not None
        assert est.relations_closed_form is not None
        assert est.possible_distinctions is None
        assert est.possible_relations is None


class TestBudget:
    def test_limit_truncates_the_walk(self):
        est = estimate_analysis(_dense3(), limit=10)
        assert est.capped is True
        assert est.purview_evaluations is not None
        assert est.purview_evaluations < 98
        assert est.mechanism_partition_sweeps < 1102

    def test_memoized_counts_do_not_consume_budget(self):
        # Seeded system-partition counts resolve even under a unit budget.
        est = estimate_analysis(_dense3(), compute="sia", limit=1)
        assert est.system_partitions == 22
        assert est.capped is False


class TestPresentation:
    def test_pandas_record(self):
        record = estimate_analysis(_dense3()).to_pandas()
        assert record["n_units"] == 3
        assert record["mechanisms"] == 7
        assert record["capped"] is False

    def test_card_renders(self):
        est = estimate_analysis(_dense3())
        card = str(est)
        assert "AnalysisEstimate" in card

    def test_capped_card_uses_lower_bound_qualifier(self):
        est = estimate_analysis(_dense3(), limit=10)
        assert "≥" in str(est)
```

Also add to the imports at the top of `test/test_cost.py` (with the other `pyphi.cost` imports and after them, per isort):

```python
from math import comb

from pyphi.cost import estimate_analysis
from test.conftest import IIT_3_CONFIG
```

(`from math import comb` goes at the very top of the file, first import; the other two join their groups per isort.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest test/test_cost.py -q > /tmp/t2a.log 2>&1; cat /tmp/t2a.log`
Expected: ImportError — `cannot import name 'estimate_analysis' from 'pyphi.cost'`.

- [ ] **Step 3: Implement `AnalysisEstimate` and `estimate_analysis`**

In `pyphi/cost.py`:

1. Extend the import block at the top to:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any

from pyphi.display import Description
from pyphi.display import Displayable
from pyphi.display import Row
from pyphi.display import Section
from pyphi.models.pandas import ToPandasMixin

if TYPE_CHECKING:
    from pyphi.substrate import Substrate

__all__ = ["AnalysisEstimate", "estimate_analysis"]
```

2. Append at the end of the file:

```python
def _fmt(value: int) -> str:
    """Format a count for display; huge counts as a power of ten."""
    if value < 10**15:
        return f"{value:,}"
    return f"~10^{len(str(value)) - 1}"


@dataclass(frozen=True)
class AnalysisEstimate(Displayable, ToPandasMixin):
    """The workload of a single-system analysis, before running it.

    Work axes are counted by driving the analysis's own enumeration
    machinery under the active configuration. ``None`` marks an axis
    outside the estimate's scope: excluded by ``compute``, not applicable
    under the active formalism, or not reached before the work budget
    (``capped=True``).

    Attributes
    ----------
    n_units : int
        Number of units in the candidate system.
    state_space_size : int
        Product of the candidate units' alphabet sizes — the scale of one
        repertoire evaluation. Reported as a weight, never multiplied into
        the counts.
    compute : str
        ``"full"``, ``"sia"``, or ``"ces"``.
    system_partitions : int or None
        Partitions the system irreducibility analysis sweeps, under the
        active system partition scheme.
    mechanisms : int or None
        Candidate mechanisms: 2ⁿ − 1 for n units.
    purview_evaluations : int or None
        Connectivity-pruned (mechanism, direction, purview) triples — the
        repertoire-computation axis.
    mechanism_partition_sweeps : int or None
        Mechanism partitions summed over all counted triples, under the
        active mechanism partition scheme — the dominant cost of unfolding
        a cause-effect structure.
    relations_closed_form : bool or None
        Whether the active relation backend computes relations in closed
        form (``ANALYTICAL``) rather than by enumeration (``CONCRETE``).
        ``None`` when relations are outside the estimate's scope.
    possible_distinctions : int or None
        Candidate distinctions (2ⁿ − 1) — the size ceiling of the
        cause-effect structure. Present only under an IIT 4.0 formalism
        with binary units.
    possible_relations : int or None
        Candidate relations (2^(2ⁿ−1) − 1) — the size ceiling of the
        relation set, and the enumeration worst case when
        ``relations_closed_form`` is ``False``. Present only under an
        IIT 4.0 formalism with binary units.
    capped : bool
        The counting walk hit its work budget; walked counts are lower
        bounds (rendered with a ``≥`` qualifier) and axes never reached
        are ``None``.
    """

    n_units: int
    state_space_size: int
    compute: str
    system_partitions: int | None
    mechanisms: int | None
    purview_evaluations: int | None
    mechanism_partition_sweeps: int | None
    relations_closed_form: bool | None
    possible_distinctions: int | None
    possible_relations: int | None
    capped: bool

    def _qualifier(self) -> str:
        return "≥" if self.capped else "="

    def _pandas_record(self) -> dict:
        return {
            "n_units": self.n_units,
            "state_space_size": self.state_space_size,
            "compute": self.compute,
            "system_partitions": self.system_partitions,
            "mechanisms": self.mechanisms,
            "purview_evaluations": self.purview_evaluations,
            "mechanism_partition_sweeps": self.mechanism_partition_sweeps,
            "relations_closed_form": self.relations_closed_form,
            "possible_distinctions": self.possible_distinctions,
            "possible_relations": self.possible_relations,
            "capped": self.capped,
        }

    def _describe(self, verbosity: int) -> Description:  # noqa: ARG002
        q = self._qualifier()
        rows = [
            Row("Units", str(self.n_units)),
            Row("State space", _fmt(self.state_space_size)),
        ]
        if self.system_partitions is not None:
            rows.append(
                Row("System partitions", f"{q} {_fmt(self.system_partitions)}")
            )
        if self.mechanisms is not None:
            rows.append(Row("Mechanisms", _fmt(self.mechanisms)))
        if self.purview_evaluations is not None:
            rows.append(
                Row("Purview evaluations", f"{q} {_fmt(self.purview_evaluations)}")
            )
        if self.mechanism_partition_sweeps is not None:
            rows.append(
                Row(
                    "Mechanism partition sweeps",
                    f"{q} {_fmt(self.mechanism_partition_sweeps)}",
                )
            )
        if self.relations_closed_form is not None:
            rows.append(
                Row(
                    "Relations",
                    "closed form" if self.relations_closed_form else "enumerated",
                )
            )
        if self.possible_distinctions is not None:
            rows.append(
                Row("Possible distinctions", _fmt(self.possible_distinctions))
            )
        if self.possible_relations is not None:
            rows.append(Row("Possible relations", _fmt(self.possible_relations)))
        rows.append(Row("Capped", self.capped))
        return Description(
            title="AnalysisEstimate",
            subtitle=f"{self.n_units} units, {self.compute}",
            sections=(Section(rows=tuple(rows)),),
            compact=(
                f"AnalysisEstimate(n_units={self.n_units}, "
                f"compute={self.compute!r})"
            ),
        )


def estimate_analysis(
    substrate: Substrate,
    subset: Any = None,
    compute: str | None = None,
    limit: int = 1_000_000,
) -> AnalysisEstimate:
    """Count the workload of a single-system analysis, without running it.

    Drives the same enumeration machinery :func:`pyphi.analyze` would use
    under the active configuration: the system partition scheme, the
    connectivity-pruned purview sets, and the mechanism partition scheme.
    No φ is computed and no state is needed — every counted quantity is
    state-independent.

    Parameters
    ----------
    substrate : Substrate
        The substrate to analyze.
    subset : optional
        Node indices (or labels) of the candidate system; ``None`` uses
        the whole substrate.
    compute : str or None, optional
        ``None`` estimates the full analysis; ``"sia"`` only the
        system-partition axis; ``"ces"`` only the distinction axis.
    limit : int, optional
        Work budget for the counting walk itself: purview evaluations and
        fresh partition enumerations each cost one unit, while memoized
        partition counts are free. A walk that exceeds the budget stops
        immediately and reports ``capped=True``.

    Returns
    -------
    AnalysisEstimate
        The counted workload.

    Raises
    ------
    ValueError
        If ``compute`` is not ``"sia"``, ``"ces"``, or ``None``.

    Examples
    --------
    >>> from pyphi import examples
    >>> est = estimate_analysis(examples.basic_substrate())
    >>> est.mechanisms
    7
    >>> est.system_partitions
    22
    """
    if compute not in (None, "sia", "ces"):
        raise ValueError(
            f"unknown compute: {compute!r}; expected 'sia', 'ces', or None "
            "for the full analysis"
        )
    from pyphi import utils
    from pyphi.conf import config
    from pyphi.direction import Direction
    from pyphi.system import System

    cs = System.from_substrate(substrate, (0,) * substrate.size, subset)
    indices = cs.node_indices
    m = len(indices)
    alphabet = substrate.factored_tpm.alphabet_sizes
    state_space_size = 1
    for i in indices:
        state_space_size *= int(alphabet[i])
    scope = "full" if compute is None else compute

    counter = _Counter(limit)
    capped = False
    system_partition_count = None
    mechanisms = None
    purview_evaluations = None
    sweeps = None
    try:
        if scope in ("full", "sia"):
            system_partition_count = _system_partition_count(m, counter)
        if scope in ("full", "ces"):
            mechanisms = 2**m - 1
            purview_evaluations = 0
            sweeps = 0
            for mechanism in utils.powerset(indices, nonempty=True):
                for direction in (Direction.CAUSE, Direction.EFFECT):
                    for purview in cs.potential_purviews(direction, mechanism):
                        counter.charge(1)
                        purview_evaluations += 1
                        sweeps += _mechanism_partition_count(
                            len(mechanism), len(purview), counter
                        )
    except _LimitReached:
        capped = True

    version = config.formalism.iit.version
    relations_closed_form = None
    possible_distinctions = None
    possible_relations = None
    if version.startswith("IIT_4_0") and scope in ("full", "ces"):
        relations_closed_form = (
            config.formalism.iit.relation_computation == "ANALYTICAL"
        )
        if all(int(alphabet[i]) == 2 for i in indices):
            from pyphi.formalism.iit4 import bounds

            possible_distinctions = bounds.number_of_possible_distinctions(m)
            possible_relations = bounds.number_of_possible_relations(m)

    return AnalysisEstimate(
        n_units=m,
        state_space_size=state_space_size,
        compute=scope,
        system_partitions=system_partition_count,
        mechanisms=mechanisms,
        purview_evaluations=purview_evaluations,
        mechanism_partition_sweeps=sweeps,
        relations_closed_form=relations_closed_form,
        possible_distinctions=possible_distinctions,
        possible_relations=possible_relations,
        capped=capped,
    )
```

3. In `pyphi/__init__.py`, after the two `from .core.tpm import ...` lines, add:

```python
from .cost import AnalysisEstimate as AnalysisEstimate
from .cost import estimate_analysis as estimate_analysis
```

- [ ] **Step 4: Run the tests, including the module's doctests**

Run: `uv run pytest test/test_cost.py pyphi/cost.py -q > /tmp/t2b.log 2>&1; cat /tmp/t2b.log`
Expected: all PASS (the `pyphi/cost.py` path collects the docstring examples via `--doctest-modules`).

- [ ] **Step 5: Type-check**

Run: `uv run pyright pyphi/cost.py pyphi/__init__.py > /tmp/t2c.log 2>&1; cat /tmp/t2c.log`
Expected: 0 errors.

- [ ] **Step 6: Commit**

```bash
git add pyphi/cost.py pyphi/__init__.py test/test_cost.py
git commit -m "Add estimate_analysis workload pre-flight"
```

---

### Task 3: MCP guard delegation and `estimate_cost` tool

**Files:**
- Modify: `pyphi/mcp/server.py`, `test/mcp/test_server.py`

**Interfaces:**
- Consumes: `estimate_analysis` (Task 2), `pyphi.conf.presets.by_name`.
- Produces: module constants `_SIA_PARTITION_LIMIT = 4_419_572`, `_CES_SWEEP_LIMIT = 100_000_000`, `_GUARD_COUNT_BUDGET = 3_000_000`; MCP tool `estimate_cost(handle, compute="full", formalism=None) -> dict` returning `{"card": str, "estimate": dict}`.

- [ ] **Step 1: Write the failing tests**

In `test/mcp/test_server.py`, after `test_analyze_guardrail_refuses_large_without_confirmation`, add:

```python
def test_analyze_guard_reports_estimated_counts():
    tpm = np.zeros((2**8, 8))
    handle = srv.build_substrate(tpm.tolist())["handle"]
    with pytest.raises(ValueError, match="mechanism-partition sweeps"):
        srv.analyze(handle, [0] * 8, compute="full")


def test_count_gate_admits_sparse_system_above_old_node_limit():
    # Eight disconnected units have no candidate purviews at all, so the
    # estimated workload is trivial and the guard admits the analysis
    # without confirmation, where a node-count guard refused at this size.
    tpm = np.zeros((2**8, 8))
    cm = np.zeros((8, 8))
    handle = srv.build_substrate(tpm.tolist(), cm=cm.tolist())["handle"]
    out = srv.analyze(handle, [0] * 8, compute="ces")
    assert "result_ref" in out


def test_estimate_cost_tool(basic_handle):
    out = srv.estimate_cost(basic_handle)
    assert "AnalysisEstimate" in out["card"]
    est = out["estimate"]
    assert est["n_units"] == 3
    assert est["mechanisms"] == 7
    assert est["capped"] is False


def test_estimate_cost_sia_scope(basic_handle):
    est = srv.estimate_cost(basic_handle, compute="sia")["estimate"]
    assert est["system_partitions"] == 22
    assert est["mechanism_partition_sweeps"] is None


def test_estimate_cost_unknown_formalism_is_a_clear_error(basic_handle):
    with pytest.raises(ValueError, match="unknown formalism"):
        srv.estimate_cost(basic_handle, formalism="IIT_5_0")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest test/mcp/test_server.py -q > /tmp/t3a.log 2>&1; cat /tmp/t3a.log`
Expected: the five new tests FAIL (`estimate_cost` missing; guard message does not mention sweeps; sparse case refused); pre-existing tests PASS.

- [ ] **Step 3: Implement the delegation**

In `pyphi/mcp/server.py`:

1. Add imports — `from dataclasses import asdict` as the first `from`-import of the stdlib group (before `from pathlib import Path`), and in the pyphi group:

```python
from pyphi.conf import presets
from pyphi.cost import estimate_analysis
```

(after `from pyphi import serialize` / before `from pyphi.conf.infrastructure import InfrastructureConfig`, respecting isort order.)

2. Replace the node-limit constants and their comment:

```python
# A full cause-effect structure unfolds distinctions and relations, whose count
# grows doubly-exponentially in the number of units, so a larger request is
# refused unless the caller confirms it. System integrated information alone is
# cheaper. These are soft guards against accidental hours-long runs, not hard
# limits on what PyPhi can compute.
_CES_NODE_LIMIT = 7
_SIA_NODE_LIMIT = 9
```

becomes:

```python
# Soft guards against accidentally starting an hours-long run, not hard limits
# on what PyPhi can compute. The guard counts the requested analysis's workload
# with pyphi.cost.estimate_analysis, so the partition schemes, connectivity,
# and alphabet all inform the refusal. _SIA_PARTITION_LIMIT is the
# DIRECTED_SET_PARTITION count for 9 fully connected binary units — the
# largest system-level analysis admitted without confirmation.
# _CES_SWEEP_LIMIT admits a fully connected 6-unit binary cause-effect
# structure under JOINT_PARTITION_ALL (31,938,830 sweeps) and refuses the
# 7-unit one (1,450,456,298 sweeps).
#
# _GUARD_COUNT_BUDGET bounds the guard's own counting work (purview
# evaluations plus fresh partition enumerations; memoized counts are free),
# keeping the pre-flight to a few seconds. A walk that exceeds it is
# refused conservatively: a workload too large to count cheaply is treated
# as too large to run unconfirmed.
_SIA_PARTITION_LIMIT = 4_419_572
_CES_SWEEP_LIMIT = 100_000_000
_GUARD_COUNT_BUDGET = 3_000_000
```

3. In the `analyze` tool, replace the guard block:

```python
    substrate = _get_substrate(handle)
    size = substrate.size
    unfolds_structure = compute in ("full", "ces")
    limit = _CES_NODE_LIMIT if unfolds_structure else _SIA_NODE_LIMIT
    if size > limit and not confirm_large:
        raise ValueError(
            f"This substrate has {size} nodes; a '{compute}' analysis at this "
            f"size may run for a very long time (cost grows exponentially, and "
            f"relations doubly-exponentially). Pass confirm_large=true to "
            f"proceed anyway, or use compute='sia' for a cheaper system-level "
            f"result."
        )
```

becomes:

```python
    substrate = _get_substrate(handle)
    if not confirm_large:
        unfolds_structure = compute in ("full", "ces")
        threshold = (
            _CES_SWEEP_LIMIT if unfolds_structure else _SIA_PARTITION_LIMIT
        )
        overrides = presets.by_name.get(formalism, {}) if formalism else {}
        with pyphi.config.override(**overrides):
            estimate = estimate_analysis(
                substrate,
                compute="ces" if unfolds_structure else "sia",
                limit=_GUARD_COUNT_BUDGET,
            )
        gauge = (
            estimate.mechanism_partition_sweeps
            if unfolds_structure
            else estimate.system_partitions
        )
        axis = (
            "mechanism-partition sweeps"
            if unfolds_structure
            else "system partitions"
        )
        if estimate.capped or (gauge is not None and gauge > threshold):
            if gauge is None:
                estimated = f"beyond the guard's counting budget in {axis}"
            elif estimate.capped:
                estimated = f"at least {gauge:,} {axis}"
            else:
                estimated = f"{gauge:,} {axis}"
            raise ValueError(
                f"A '{compute}' analysis of this {substrate.size}-node "
                f"substrate is estimated at {estimated} (soft limit "
                f"{threshold:,}); it may run for a very long time. Pass "
                f"confirm_large=true to proceed anyway, use the "
                f"estimate_cost tool to inspect the workload, or use "
                f"compute='sia' for a cheaper system-level result."
            )
```

(An unknown `formalism` string falls through to the estimate under the active config and is then rejected with the standard message by `pyphi.analyze` below.)

4. Update the `confirm_large` parameter docstring in `analyze`:

```
    confirm_large : bool
        Full/CES analyses are refused above a soft node-count threshold unless
        this is set, to avoid accidentally starting an hours-long computation.
        Parallelism does not lift the threshold — it divides the constants,
        not the exponents.
```

becomes:

```
    confirm_large : bool
        An analysis whose estimated workload exceeds a soft limit is refused
        unless this is set, to avoid accidentally starting an hours-long
        computation. The workload is counted by the same machinery as the
        ``estimate_cost`` tool, so the partition schemes, connectivity, and
        alphabet all inform the guard. Parallelism does not lift the
        threshold — it divides the constants, not the exponents.
```

5. After the `analyze` function (before `configure_parallel`), add the tool:

```python
@mcp.tool()
def estimate_cost(
    handle: str,
    compute: str = "full",
    formalism: str | None = None,
) -> dict[str, Any]:
    """Count the workload of an analysis before running it.

    Reports what ``analyze`` would evaluate — system partitions, candidate
    mechanisms, connectivity-pruned purview evaluations, and
    mechanism-partition sweeps — without computing any φ. The counts
    reflect the partition schemes, connectivity, and alphabet under the
    requested formalism. Wall time is machine-dependent and is not
    predicted; use the counts to compare candidate systems and settings.

    Parameters
    ----------
    handle : str
        A substrate handle from ``load_example`` or ``build_substrate``.
    compute : str
        ``"full"`` (default), ``"sia"``, or ``"ces"`` — the analysis whose
        workload to estimate, as in ``analyze``.
    formalism : str, optional
        As in ``analyze``.

    Returns
    -------
    dict
        A ``card`` (human-readable text) and an ``estimate`` mapping with
        the counts; ``capped=true`` marks counts that are lower bounds.
    """
    substrate = _get_substrate(handle)
    if formalism is not None and formalism not in presets.by_name:
        valid = ", ".join(sorted(presets.by_name))
        raise ValueError(
            f"unknown formalism {formalism!r}; expected one of: {valid}"
        )
    overrides = presets.by_name[formalism] if formalism is not None else {}
    with pyphi.config.override(**overrides):
        estimate = estimate_analysis(
            substrate, compute=None if compute == "full" else compute
        )
    return {"card": str(estimate), "estimate": asdict(estimate)}
```

- [ ] **Step 4: Run the MCP tests**

Run: `uv run pytest test/mcp/test_server.py -q > /tmp/t3b.log 2>&1; cat /tmp/t3b.log`
Expected: all PASS, including the two pre-existing guard tests (`test_analyze_guardrail_refuses_large_without_confirmation`, `test_analyze_guardrail_unchanged_with_parallel` — the dense 8-node case still matches "confirm_large" because the message contains `confirm_large=true`).

- [ ] **Step 5: Commit**

```bash
git add pyphi/mcp/server.py test/mcp/test_server.py
git commit -m "Gate MCP analyze on estimated workload counts"
```

---

### Task 4: Documentation, changelog, ROADMAP

**Files:**
- Modify: `docs/theory/computational-complexity.md`, `pyphi/mcp/content/performance.md`, `ROADMAP.md`
- Create: `changelog.d/analysis-cost-estimate.feature.md`

**Interfaces:**
- Consumes: `pyphi.estimate_analysis` (Task 2), `estimate_cost` tool (Task 3).

- [ ] **Step 1: Add the pre-flight section to the complexity page**

In `docs/theory/computational-complexity.md`, immediately before the line `## The cost of the grain search`, insert:

````markdown
## Estimating a workload before running it

Every count above is knowable before any φ is computed.
`pyphi.estimate_analysis` walks the same enumeration machinery the analysis
would use — the active system partition scheme, the connectivity-pruned
purview sets, the mechanism partitions per (mechanism, purview) size pair —
and returns the counts as an `AnalysisEstimate`, without evaluating
anything:

```{code-cell} python
import pyphi
from pyphi import examples

pyphi.estimate_analysis(examples.basic_substrate())
```

The estimate reports counts and structural weights (the state-space size is
the per-evaluation cost scale); it never predicts wall time, which is
machine- and configuration-dependent. Counting is budgeted: pass `limit` to
bound the estimate's own work, and a truncated walk reports lower bounds
with `capped=True`. The counterpart for grain searches is
`SearchBounds.estimate`, described in the next section.

````

- [ ] **Step 2: Add the pre-flight paragraph to the MCP performance topic**

In `pyphi/mcp/content/performance.md`, after the first paragraph (the one ending "…not after it hangs."), insert:

```markdown
Cost is countable before you commit: the `estimate_cost` tool reports the
workload of an `analyze` call — system partitions, purview evaluations,
mechanism-partition sweeps — without computing anything (in Python,
`pyphi.estimate_analysis`). The `analyze` guard itself runs on these
counts, so `confirm_large` is requested exactly when the workload is
actually large under the active formalism, scheme, and connectivity.
```

- [ ] **Step 3: Changelog fragment**

Create `changelog.d/analysis-cost-estimate.feature.md`:

```markdown
Added `pyphi.estimate_analysis()`: an analytic pre-flight that counts the
workload of a single-system analysis — system partitions under the active
scheme, candidate mechanisms, connectivity-pruned purview evaluations, and
mechanism-partition sweeps — without computing any φ. The MCP server's
`analyze` guard now gates on these estimated counts rather than node
counts, and a new `estimate_cost` MCP tool exposes the estimate.
```

- [ ] **Step 4: ROADMAP row**

In `ROADMAP.md`, replace:

```markdown
- **Cost pre-flight for analyze/sia/ces (M5).** The counting primitives exist in `bounds.py`;
  expose an estimate surface and make the MCP hard node-limits delegate to it.
```

with:

```markdown
- **Cost pre-flight for analyze/sia/ces (M5).** *Landed 2026-07-20:*
  `pyphi.estimate_analysis` (`pyphi/cost.py`) counts the workload of a single-system
  analysis — scheme-aware system partitions, connectivity-pruned purview evaluations,
  mechanism-partition sweeps — by driving the real enumerators with seeded memos; the
  MCP `analyze` guard now gates on those counts instead of node counts (a sparse
  system above the old node limit passes; a fully connected 7-unit CES,
  ≈1.45 × 10⁹ sweeps, now asks for confirmation), and a new `estimate_cost` MCP tool
  exposes the estimate. Spec:
  `docs/superpowers/specs/2026-07-20-analysis-cost-estimate-design.md`.
```

- [ ] **Step 5: Build the docs**

Run: `rm -rf docs/reference/_autosummary && just docs > /tmp/t4a.log 2>&1; tail -5 /tmp/t4a.log`
Expected: build succeeds (the new code-cell executes; the recursive autosummary picks up `pyphi.cost`). Read the tail for the success line — do not trust the exit code alone.

- [ ] **Step 6: Full suite**

Run: `uv run pytest -q > /tmp/t4b.log 2>&1; cat /tmp/t4b.log | grep -E "^(FAILED|ERROR)|passed|failed"`
Expected: 0 failed. (Pathless invocation — the doctest sweep over `pyphi/` is part of the gate.)

- [ ] **Step 7: Commit**

```bash
git add docs/theory/computational-complexity.md pyphi/mcp/content/performance.md changelog.d/analysis-cost-estimate.feature.md ROADMAP.md
git commit -m "Document the analysis workload pre-flight"
```
