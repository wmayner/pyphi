# B7 — Unified partition edge-set interface — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make "what edges does this partition sever?" a total, efficient interface on every partition/cut type, add the refinement partial order and an ergonomic total order, and delete the `except AttributeError: return None` fragility in distinction-φ normalization.

**Architecture:** All partition/cut types already inherit `_PartitionBase` (`pyphi/models/partitions.py`) and implement `cut_matrix(n)`. We add `removed_edges()` (canonical primitive, efficient per-type structural overrides validated against `cut_matrix`), derive `num_connections_cut()` from it on the base, add `refines()`/`coarsens()` (refinement partial order = superset of severed edges) and `functools.total_ordering` dunders keyed on the existing `lex_key` (total order), then remove the now-dead `AttributeError`/`None` paths.

**Tech Stack:** Python 3.13, numpy, pytest, `uv run` for all commands.

**Approved spec:** `docs/superpowers/specs/2026-06-15-b7-unified-partition-edge-set-design.md`

---

## File structure

- **Modify** `pyphi/models/partitions.py` — add `removed_edges`, base `num_connections_cut`, `refines`/`coarsens`, `functools.total_ordering` + `__lt__`; delete `JointPartition.num_connections_cut`.
- **Modify** `pyphi/models/state_specification.py` — remove the `except AttributeError: return None` arm; tighten return type.
- **Modify** `pyphi/models/ria.py` — remove the now-unreachable `if norm is None` branch.
- **Create** `test/test_partition_edge_set.py` — the B7 unit tests (edge-set equivalence, count preservation, partial/total order).
- **Modify** `changelog.d/` — add `b7-partition-edge-set.refactor.md`.
- **Modify** `ROADMAP.md` — B7 dashboard row + Wave-2 prose.

All `git commit` steps use `-S` (SSH signing is configured). **Do not push** unless the user asks.

---

## Task 1: `removed_edges()` — canonical primitive

**Files:**
- Modify: `pyphi/models/partitions.py`
- Test: `test/test_partition_edge_set.py`

- [ ] **Step 1: Write the failing test**

Create `test/test_partition_edge_set.py`:

```python
"""B7 — total edge-set interface on partitions/cuts."""

from __future__ import annotations

import numpy as np
import pytest

from pyphi.direction import Direction
from pyphi.models.partitions import (
    CompleteEdgeCut,
    DirectedBipartition,
    DirectedJointPartition,
    DirectedSetPartition,
    EdgeCut,
    JointBipartition,
    JointPartition,
    JointTripartition,
    NullCut,
    Part,
)

# Substrate size large enough to embed every instance's indices.
N = 8

_JP = JointPartition(Part((0,), (1,)), Part((1,), (0,)))

PARTITIONS = [
    NullCut((0, 1, 2)),
    DirectedBipartition(Direction.EFFECT, (0,), (1, 2)),
    DirectedBipartition(Direction.CAUSE, (0, 3), (1, 2)),
    EdgeCut((0, 2, 3), np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])),
    CompleteEdgeCut((1, 2, 4)),
    DirectedSetPartition(
        (0, 1, 2), np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]), [[0], [1, 2]]
    ),
    _JP,
    JointBipartition(Part((0, 2), (1,)), Part((1,), (0, 2))),
    JointTripartition(Part((0,), (1,)), Part((1,), (2,)), Part((2,), (0,))),
    DirectedJointPartition(Direction.CAUSE, _JP),
]


@pytest.mark.parametrize("p", PARTITIONS, ids=lambda p: type(p).__name__)
def test_removed_edges_matches_cut_matrix(p):
    expected = frozenset(
        (int(a), int(b)) for a, b in np.argwhere(p.cut_matrix(N))
    )
    assert p.removed_edges() == expected


def test_removed_edges_n_invariant():
    # The edge set must not change when cut_matrix is evaluated at larger n.
    p = DirectedBipartition(Direction.EFFECT, (0,), (1, 2))
    small = frozenset((int(a), int(b)) for a, b in np.argwhere(p.cut_matrix(3)))
    assert p.removed_edges() == small


def test_removed_edges_returns_python_ints():
    p = EdgeCut((0, 2), np.array([[0, 1], [0, 0]]))
    (edge,) = p.removed_edges()
    assert type(edge[0]) is int and type(edge[1]) is int
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_partition_edge_set.py -q`
Expected: FAIL — `AttributeError: 'NullCut' object has no attribute 'removed_edges'`.

- [ ] **Step 3: Add the base default + structural overrides**

In `pyphi/models/partitions.py`, add to `_PartitionBase` (after `lex_key`, around line 132):

```python
    def removed_edges(self) -> frozenset[tuple[int, int]]:
        """The set of directed edges ``(from, to)`` this partition severs.

        Default derivation from :meth:`cut_matrix`; concrete subclasses
        override with an equivalent structural form that avoids materializing
        the full ``n x n`` matrix. The two must agree (verified by
        ``test_partition_edge_set.py``).
        """
        indices = self.indices
        if not indices:
            return frozenset()
        matrix = self.cut_matrix(max(indices) + 1)
        return frozenset((int(a), int(b)) for a, b in np.argwhere(matrix))
```

Add to `NullCut` (after `cut_matrix`, around line 153):

```python
    def removed_edges(self) -> frozenset[tuple[int, int]]:
        return frozenset()
```

Add to `DirectedBipartition` (after `cut_matrix`, around line 222):

```python
    def removed_edges(self) -> frozenset[tuple[int, int]]:
        # relevant_connections sets cm[f, t] = 1 for f in from_nodes,
        # t in to_nodes (see connectivity.relevant_connections).
        return frozenset(
            (f, t) for f in self.from_nodes for t in self.to_nodes
        )
```

Add to `EdgeCut` (after `cut_matrix`, around line 350). This is inherited by `CompleteEdgeCut` and `DirectedSetPartition`:

```python
    def removed_edges(self) -> frozenset[tuple[int, int]]:
        idx = self.node_indices
        return frozenset(
            (idx[i], idx[j]) for i, j in np.argwhere(self._cut_matrix)
        )
```

Add to `DirectedJointPartition` (after `cut_matrix`, around line 296):

```python
    def removed_edges(self) -> frozenset[tuple[int, int]]:
        indices = set(self.indices)
        edges: set[tuple[int, int]] = set()
        for part in self.partition:
            from_, to = self.direction.order(part.mechanism, part.purview)
            external = indices - set(to)
            edges.update((f, e) for f in from_ for e in external)
        return frozenset(edges)
```

Add to `JointPartition` (after `cut_matrix`, around line 605):

```python
    def removed_edges(self) -> frozenset[tuple[int, int]]:
        purview = set(self.purview)
        edges: set[tuple[int, int]] = set()
        for part in self.parts:
            outside = purview - set(part.purview)
            edges.update((m, o) for m in part.mechanism for o in outside)
        return frozenset(edges)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_partition_edge_set.py -q`
Expected: PASS (13 tests).

- [ ] **Step 5: Commit**

```bash
git add pyphi/models/partitions.py test/test_partition_edge_set.py
git commit -S -m "Add total removed_edges() to partition types (B7)"
```

---

## Task 2: Lift `num_connections_cut()` to the base; delete the override

**Files:**
- Modify: `pyphi/models/partitions.py:590-598` (delete `JointPartition.num_connections_cut`)
- Test: `test/test_partition_edge_set.py`

- [ ] **Step 1: Write the failing test**

Append to `test/test_partition_edge_set.py`:

```python
@pytest.mark.parametrize("p", PARTITIONS, ids=lambda p: type(p).__name__)
def test_num_connections_cut_is_edge_count(p):
    assert p.num_connections_cut() == len(p.removed_edges())


def test_num_connections_cut_preserves_eq24_counts():
    # Values the deleted JointPartition Eq. 24 override produced.
    assert JointPartition(Part((0,), (1,)), Part((1,), (0,))).num_connections_cut() == 2
    jb = JointBipartition(Part((0, 2), (1,)), Part((1,), (0, 2)))
    assert jb.num_connections_cut() == 5


def test_nullcut_num_connections_cut_is_zero():
    assert NullCut((0, 1, 2)).num_connections_cut() == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_partition_edge_set.py -k num_connections -q`
Expected: FAIL — `NullCut`/`EdgeCut`/etc. have no `num_connections_cut` (only `JointPartition` does today).

- [ ] **Step 3: Add base method, delete the override**

In `pyphi/models/partitions.py`, add to `_PartitionBase` (right after `removed_edges`):

```python
    def num_connections_cut(self) -> int:
        """Number of directed connections severed (IIT 4.0 Eq. 24)."""
        return len(self.removed_edges())
```

Then **delete** `JointPartition.num_connections_cut` entirely (the block at lines 590-598):

```python
    def num_connections_cut(self) -> int:
        """Number of connections severed by the induced edge cut (IIT 4.0 Eq. 24)."""
        n = 0
        purview_lengths = [len(part.purview) for part in self.parts]
        for i, part in enumerate(self.parts):
            n += len(part.mechanism) * (
                sum(purview_lengths[:i]) + sum(purview_lengths[i + 1 :])
            )
        return n
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/test_partition_edge_set.py test/test_bounds.py -q`
Expected: PASS. `test_bounds.py` pins `num_connections_cut()` values (`2`, and set-partition counts) and must stay green — confirming the derived count equals the deleted Eq. 24 formula.

- [ ] **Step 5: Verify no golden / AC regression from the now-total count**

`DirectedJointPartition` (AC distinctions) previously lacked `num_connections_cut()`; it now has one. Confirm this changes no computed result:

Run: `uv run pytest test/test_golden_regression.py test/test_actual.py test/test_models.py -q`
Expected: PASS, unchanged. (Characterization note: distinction-φ normalization only reaches this method under `distinction_phi_normalization="NUM_CONNECTIONS_CUT"`, where the partition is a `JointPartition`; the AC/iit3 path uses `"NONE"`. If any golden shifts, STOP and investigate before continuing — it would mean a previously-`None` normalization path was live.)

- [ ] **Step 6: Commit**

```bash
git add pyphi/models/partitions.py test/test_partition_edge_set.py
git commit -S -m "Derive num_connections_cut from removed_edges on the base (B7)"
```

---

## Task 3: Refinement partial order — `refines()` / `coarsens()`

**Files:**
- Modify: `pyphi/models/partitions.py`
- Test: `test/test_partition_edge_set.py`

- [ ] **Step 1: Write the failing test**

Append to `test/test_partition_edge_set.py`:

```python
def _edgecut(edges, n=4):
    m = np.zeros((n, n), dtype=int)
    for a, b in edges:
        m[a, b] = 1
    return EdgeCut(tuple(range(n)), m)


def test_refines_is_superset_of_severed_edges():
    coarse = _edgecut([(0, 1)])
    fine = _edgecut([(0, 1), (1, 0)])
    assert fine.refines(coarse)        # fine severs a superset
    assert not coarse.refines(fine)
    assert coarse.coarsens(fine)
    assert not fine.coarsens(coarse)


def test_refines_is_reflexive():
    p = _edgecut([(0, 1), (2, 3)])
    assert p.refines(p) and p.coarsens(p)


def test_refines_partial_order_has_incomparable_pairs():
    a = _edgecut([(0, 1)])
    b = _edgecut([(1, 0)])
    assert not a.refines(b) and not b.refines(a)  # genuinely partial


def test_refines_is_transitive():
    a = _edgecut([(0, 1), (1, 0), (2, 3)])
    b = _edgecut([(0, 1), (1, 0)])
    c = _edgecut([(0, 1)])
    assert a.refines(b) and b.refines(c) and a.refines(c)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_partition_edge_set.py -k refines -q`
Expected: FAIL — `'EdgeCut' object has no attribute 'refines'`.

- [ ] **Step 3: Implement on the base**

In `pyphi/models/partitions.py`, add to `_PartitionBase` (after `num_connections_cut`):

```python
    def refines(self, other: _PartitionBase) -> bool:
        """Whether this is *finer-or-equal* to ``other``.

        A partition is finer when it severs more connections, so refinement
        is **superset** of :meth:`removed_edges`. This is a *partial* order:
        two partitions can be incomparable (neither refines the other). It is
        NOT a total order and must not be used as a ``sorted``/``min`` key —
        use ``<`` (the ``lex_key`` total order) for that.
        """
        return self.removed_edges() >= other.removed_edges()

    def coarsens(self, other: _PartitionBase) -> bool:
        """Whether this is *coarser-or-equal* to ``other`` (inverse of
        :meth:`refines`)."""
        return other.refines(self)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_partition_edge_set.py -k refines -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add pyphi/models/partitions.py test/test_partition_edge_set.py
git commit -S -m "Add refines/coarsens refinement partial order (B7)"
```

---

## Task 4: Total-order comparison dunders (keyed on `lex_key`)

**Files:**
- Modify: `pyphi/models/partitions.py` (imports + `_PartitionBase`)
- Test: `test/test_partition_edge_set.py`

- [ ] **Step 1: Write the failing test**

Append to `test/test_partition_edge_set.py`:

```python
def test_total_order_matches_lex_key():
    items = [
        DirectedBipartition(Direction.EFFECT, (1,), (2,)),
        NullCut((0, 1)),
        DirectedBipartition(Direction.EFFECT, (0,), (1,)),
    ]
    assert sorted(items) == sorted(items, key=lambda p: p.lex_key())


def test_total_order_operators():
    a = DirectedBipartition(Direction.EFFECT, (0,), (1,))
    b = DirectedBipartition(Direction.EFFECT, (1,), (2,))
    lo, hi = sorted([a, b], key=lambda p: p.lex_key())
    assert lo < hi and lo <= hi and hi > lo and hi >= lo
    assert not (hi < lo)


def test_nullcut_sorts_first():
    null = NullCut((0, 1))
    cut = DirectedBipartition(Direction.EFFECT, (0,), (1,))
    assert sorted([cut, null]) == [null, cut]  # lex_key("") sorts first


def test_equality_unchanged():
    a = DirectedBipartition(Direction.EFFECT, (0,), (1,))
    b = DirectedBipartition(Direction.EFFECT, (0,), (1,))
    assert a == b and hash(a) == hash(b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_partition_edge_set.py -k "total_order or nullcut_sorts" -q`
Expected: FAIL — `TypeError: '<' not supported between instances of 'DirectedBipartition'...`.

- [ ] **Step 3: Implement total ordering**

In `pyphi/models/partitions.py`, add `import functools` to the imports block (after `from itertools import chain`, around line 51):

```python
import functools
```

Decorate `_PartitionBase` and add `__lt__` (the class header at line 67 and a method inside it):

```python
@functools.total_ordering
class _PartitionBase:
    ...
    def __lt__(self, other: object) -> bool:
        """Total order by induced-cut bytes (:meth:`lex_key`).

        This is the deterministic order already used for tie-breaking
        (``PARTITION_LEX``, the SIA sort key). ``__eq__``/``__hash__`` are
        defined per subclass and unchanged; partitions with identical induced
        cuts but distinct structure sort as equal-rank. For the refinement
        relation use :meth:`refines`/:meth:`coarsens`, NOT ``<``.
        """
        if not isinstance(other, _PartitionBase):
            return NotImplemented
        return self.lex_key() < other.lex_key()
```

Place the `__lt__` method inside `_PartitionBase` (e.g. right after `lex_key`). Keep `@functools.total_ordering` immediately above `class _PartitionBase:`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_partition_edge_set.py -q`
Expected: PASS (all B7 tests).

- [ ] **Step 5: Regression-check the model/partition suites**

Run: `uv run pytest test/test_models.py test/test_partition.py -q`
Expected: PASS — equality/hashing/sorting behavior of partitions elsewhere is unaffected.

- [ ] **Step 6: Commit**

```bash
git add pyphi/models/partitions.py test/test_partition_edge_set.py
git commit -S -m "Add lex_key total-ordering dunders to partitions (B7)"
```

---

## Task 5: Delete the `except AttributeError: return None` hack

**Files:**
- Modify: `pyphi/models/state_specification.py:185-192`
- Modify: `pyphi/models/ria.py:160-173`
- Test: `test/test_partition_edge_set.py`

- [ ] **Step 1: Write the characterization test**

Append to `test/test_partition_edge_set.py`:

```python
from pyphi.models.state_specification import normalization_factor


@pytest.mark.parametrize("p", PARTITIONS, ids=lambda p: type(p).__name__)
def test_normalization_factor_never_none(p):
    # Every partition now has num_connections_cut(), so the NUM_CONNECTIONS_CUT
    # normalization always returns a real factor (1 for a zero-cut partition).
    from pyphi.conf import config

    with config.override(distinction_phi_normalization="NUM_CONNECTIONS_CUT"):
        result = normalization_factor(p)
    assert result is not None


def test_normalization_factor_zero_cut_is_one():
    from pyphi.conf import config

    with config.override(distinction_phi_normalization="NUM_CONNECTIONS_CUT"):
        assert normalization_factor(NullCut((0, 1))) == 1
```

- [ ] **Step 2: Run test to verify current behavior**

Run: `uv run pytest test/test_partition_edge_set.py -k normalization -q`
Expected: PASS already (Tasks 1-2 made `num_connections_cut` total, so `None` can no longer occur). This test locks that in *before* removing the dead arm.

- [ ] **Step 3: Remove the `AttributeError` arm**

In `pyphi/models/state_specification.py`, replace the `NUM_CONNECTIONS_CUT` registration (lines 185-192) and tighten `normalization_factor`'s return type (line 195):

```python
@distinction_phi_normalizations.register("NUM_CONNECTIONS_CUT")
def _(partition: object) -> int | float:
    try:
        return 1 / partition.num_connections_cut()  # type: ignore[attr-defined]
    except ZeroDivisionError:
        return 1


def normalization_factor(partition: object) -> int | float:
    key = config.formalism.iit.distinction_phi_normalization
    func = distinction_phi_normalizations[key]  # type: ignore[index]
    return func(partition)
```

- [ ] **Step 4: Remove the unreachable `None` branch in `ria.py`**

In `pyphi/models/ria.py`, replace lines 160-173:

```python
        norm = normalization_factor(self._partition)

        if norm is None:
            self._normalized_phi = None
            self._signed_normalized_phi = None
        else:
            # Compute the signed normalized phi (raw) first, then derive
            # the clamped canonical value.
            if isinstance(signed_phi, DistanceResult):
                signed_norm = float(signed_phi) * norm
            else:
                signed_norm = signed_phi * norm
            self._signed_normalized_phi = PyPhiFloat(signed_norm)
            self._normalized_phi = PyPhiFloat(utils.positive_part(signed_norm))
```

with:

```python
        # Every partition exposes num_connections_cut(), so normalization_factor
        # always returns a real factor (never None).
        norm = normalization_factor(self._partition)
        if isinstance(signed_phi, DistanceResult):
            signed_norm = float(signed_phi) * norm
        else:
            signed_norm = signed_phi * norm
        self._signed_normalized_phi = PyPhiFloat(signed_norm)
        self._normalized_phi = PyPhiFloat(utils.positive_part(signed_norm))
```

- [ ] **Step 5: Verify `normalized_phi` is never `None` anywhere it's read**

Run: `grep -rn "normalized_phi is None\|_normalized_phi = None" pyphi/ | grep -v __pycache__`
Expected: no remaining occurrences. If any other site special-cases `normalized_phi is None`, simplify it the same way (the value is now always present).

- [ ] **Step 6: Run the regression suites**

Run: `uv run pytest test/test_models.py test/test_golden_regression.py test/test_actual.py -q`
Expected: PASS, unchanged values.

- [ ] **Step 7: Commit**

```bash
git add pyphi/models/state_specification.py pyphi/models/ria.py test/test_partition_edge_set.py
git commit -S -m "Drop the num_connections_cut AttributeError hack (B7)"
```

---

## Task 6: Docs, changelog, ROADMAP, full verification

**Files:**
- Create: `changelog.d/b7-partition-edge-set.refactor.md`
- Modify: `ROADMAP.md`

- [ ] **Step 1: Write the changelog fragment**

Create `changelog.d/b7-partition-edge-set.refactor.md`:

```markdown
Unified the partition/cut edge-set interface (B7). Every partition type
(`pyphi/models/partitions.py`) now exposes total ``removed_edges()`` and
``num_connections_cut()`` methods, derived once on ``_PartitionBase`` and
overridden per type with an efficient structural form (no ``n x n``
materialization), validated against ``cut_matrix``. Added a refinement
partial order (``refines()``/``coarsens()`` — superset of severed edges) and
``lex_key``-keyed total-ordering comparison operators so partitions sort
deterministically. Removed the ``except AttributeError: return None`` fallback
in distinction-φ normalization (and the dead ``normalized_phi is None`` branch
it fed): every partition now yields a real normalization factor. No computed
value changes.
```

- [ ] **Step 2: Update the ROADMAP dashboard row**

In `ROADMAP.md`, locate the B7 row in the Status Dashboard table (search for `| B7`) and replace it with:

```markdown
| B7 unified PartitionAlgebra | ✅ landed | 2 | Total `removed_edges()`/`num_connections_cut()` on every partition type (efficient structural overrides validated vs `cut_matrix`), `refines()`/`coarsens()` refinement partial order + `lex_key` total-ordering dunders; deleted the `except AttributeError: return None` φ-norm hack. No value change |
```

Then locate the Wave-2 prose bullet (search for `**B7 — unified`) and replace it with:

```markdown
- **B7 — unified `PartitionAlgebra` — landed (2026-06-15).** P6 had already done the structural consolidation (every type under `_PartitionBase` with a universal `cut_matrix(n)`), so the remaining surface was narrow. Added total `removed_edges()` (efficient per-type structural overrides, validated exhaustively against `cut_matrix` nonzeros) and derived `num_connections_cut()` on the base (deleting `JointPartition`'s Eq. 24 override — the derived `len(removed_edges())` reproduces it). Added the refinement **partial** order as named methods `refines()`/`coarsens()` (superset of severed edges) and a deterministic **total** order via `functools.total_ordering` keyed on the existing `lex_key` (so `sorted(partitions)` works) — kept distinct by design so a partial order never masquerades as `<`. Deleted the `except AttributeError: return None` fragility in distinction-φ normalization (and the dead `normalized_phi is None` branch it fed). No computed value changes.
```

- [ ] **Step 3: Run the full suite (includes the `pyphi/` doctest sweep)**

Run: `PYPHI_WELCOME_OFF=yes uv run pytest -q -m "not slow"`
Expected: PASS (no path argument so `testpaths` + `--doctest-modules` run; this is the complete fast verification per CLAUDE.md).

- [ ] **Step 4: Run the perf budget and slow lane**

Run: `uv run pytest test/test_perf_budget.py -q`
Then: `uv run pytest --slow -q -m "slow"`
Expected: PASS — perf budget neutral (structural `removed_edges` avoids matrix materialization), Hypothesis invariants green (this touched a core model type).

- [ ] **Step 5: Commit**

```bash
git add changelog.d/b7-partition-edge-set.refactor.md ROADMAP.md
git commit -S -m "Document B7 unified partition edge-set (changelog + ROADMAP)"
```

---

## Notes for the implementer

- **Signing:** every commit uses `-S`. Do **not** push unless the user explicitly asks.
- **`@cmp.sametype`:** several subclasses wrap `__eq__` with this decorator. `functools.total_ordering` only generates the missing ordering operators from `__lt__`; it does not touch `__eq__`. Leave `__eq__`/`__hash__` exactly as they are.
- **n-invariance:** the base `removed_edges` default uses `max(indices) + 1`; padding `cut_matrix` to a larger `n` only adds zero rows/cols, so the edge set is unchanged (Task 1's `test_removed_edges_n_invariant`).
- **If a golden shifts in Task 2 Step 5 or Task 5 Step 6:** stop and investigate. It would mean a previously-`None` (un-normalized) distinction path was live, which is a behavior question to resolve with the user, not silently accept.
