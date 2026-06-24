# B7 — Unified partition edge-set interface

- **Date:** 2026-06-15
- **Status:** Draft (awaiting user review)
- **Roadmap item:** B7 (Wave 2, pre-freeze)

## Context

P6 already did the structural consolidation B7 was scoped against: every
partition/cut type now inherits from a single `_PartitionBase`
(`pyphi/models/partitions.py`) and implements a universal `cut_matrix(n)` (the
binary severed-edge matrix). The "directed bipartite edge-set" the roadmap
wants therefore already exists as `cut_matrix`. What remains is narrow:

- `num_connections_cut()` (IIT 4.0 Eq. 24) is implemented on **only**
  `JointPartition` (`partitions.py:590`). No other type has it.
- No type has a `removed_edges()` accessor.
- Distinction-φ normalization (`pyphi/models/state_specification.py:185`)
  reaches for `partition.num_connections_cut()` inside
  `try/except AttributeError: return None` — a fragility that exists *solely*
  because the method isn't total.
- There is no partition-refinement (sub-partition) relation, which B1/B2
  (bounds-pruning, branch-and-bound MIP search) will want.

Display is already thin (`pyphi/models/fmt.py` is plain functions; partition
objects carry no display logic), so the roadmap's "display classes become thin
views" sub-goal is already satisfied — no work there.

## Goals

1. Make `removed_edges()` and `num_connections_cut()` **total** across every
   partition type, via one efficient code path.
2. Add a partition-refinement **partial order** as named methods.
3. Add ergonomic **total-order comparison dunders** (deterministic, keyed on
   the existing `lex_key`), so `sorted(partitions)` / `min(partitions)` work.
4. Delete the `except AttributeError: return None` hack; normalization always
   receives a real count.

## Non-goals

- No first-class `DirectedEdgeSet` value type (the base class already is the
  unification point; YAGNI).
- No change to `__eq__`/`__hash__`, to `cut_matrix`, or to any computed φ
  value. This is an additive interface refactor.
- No new partition schemes; no display/`fmt.py` changes.

## Design

All changes are in `pyphi/models/partitions.py` unless noted.

### 1. `removed_edges()` — the canonical primitive

Add `removed_edges() -> frozenset[tuple[int, int]]` returning the set of
severed directed edges `(from_index, to_index)`. Implemented per type from its
own structure — **no full `n×n` matrix materialization** (the "stay efficient"
decision):

| Type | `removed_edges()` source |
|---|---|
| `NullCut` | empty set |
| `CompleteEdgeCut` | base `cut_matrix`-derived default (boundary case, not hot) |
| `EdgeCut`, `DirectedSetPartition` | `np.argwhere(self._cut_matrix)` (matrix already held) |
| `DirectedBipartition` | severed cross-part pairs in the cut direction (`from_nodes`×`to_nodes`, oriented by `direction`) |
| `JointPartition`, `JointBipartition`, `JointTripartition` | enumerated from `parts` (the same structure Eq. 24 already counts) |
| `DirectedJointPartition` | from its inner `JointPartition` + direction |

A `_PartitionBase` **default** also derives `removed_edges()` from
`cut_matrix(self._edge_index_span())` as a correctness fallback; concrete types
override with the efficient structural form above. The default guarantees any
future partition type is correct-by-default even before a bespoke override.

`_edge_index_span()` returns `max(self.indices) + 1` (the minimal `n`); the
edge count is invariant to larger `n` because padding adds only zero rows/cols.

### 2. `num_connections_cut()` — lift to the base

Define on `_PartitionBase`:

```python
def num_connections_cut(self) -> int:
    return len(self.removed_edges())
```

**Delete** `JointPartition.num_connections_cut()` (its Eq. 24 result is exactly
this `len`, guarded by the equivalence test in §5). `NullCut` returns 0
naturally (empty set).

### 3. Partition-refinement partial order — named methods

A partition is **finer** when it severs *more* connections. So refinement maps
to **superset** of severed edges:

```python
def refines(self, other) -> bool:
    "True if this severs every edge `other` does (and maybe more): finer-or-equal."
    return self.removed_edges() >= other.removed_edges()

def coarsens(self, other) -> bool:
    "True if `other` refines this: coarser-or-equal."
    return other.refines(self)
```

This is a **partial** order (severed-edge containment): two partitions can be
incomparable. Docstrings state this explicitly and warn that it is *not* a
total order, so it must not be used as a `sorted`/`min` key.

### 4. Total-order comparison dunders — keyed on `lex_key`

Apply `functools.total_ordering` to `_PartitionBase` with:

```python
def __lt__(self, other) -> bool:
    return self.lex_key() < other.lex_key()
```

This is the **deterministic total order already used everywhere** (SIA sort key
`(phi, partition.lex_key())`, the `PARTITION_LEX` tie strategy). It makes
`sorted(partitions)` / `min(partitions)` work and match existing tie-breaks.

`__eq__`/`__hash__` are unchanged (structural). Ordering is by *induced cut*:
two structurally-distinct partitions with the same `lex_key` (identical induced
cut) sort as equal-rank — consistent with the existing `(phi, lex_key)` SIA
key. Docstrings make the lex-vs-refinement distinction explicit:

- `__lt__`/`<` → total order by induced-cut bytes (for deterministic sorting).
- `refines()`/`coarsens()` → partial order by severed-edge containment (for
  bounds/B&B reasoning).

### 5. Delete the normalization hack

`pyphi/models/state_specification.py:185` becomes:

```python
@distinction_phi_normalizations.register("NUM_CONNECTIONS_CUT")
def _(partition) -> int | float:
    try:
        return 1 / partition.num_connections_cut()
    except ZeroDivisionError:
        return 1
```

The `except AttributeError: return None` arm is removed (all partitions now
have `num_connections_cut()`). The `ZeroDivisionError → 1` branch is preserved.
Downstream code that handled `None` is audited; with `None` no longer possible,
those branches become dead and are removed where they exist.

## Correctness verification (confirmation experiment — required before merge)

A test (`test/test_partition.py` or `test/test_models.py`) asserting, for every
registered mechanism and system partition scheme on small substrates
(n ≤ 4, all states where relevant):

1. **Edge-set ≡ matrix:** `set(map(tuple, np.argwhere(p.cut_matrix(n)))) ==
   p.removed_edges()` for each partition `p` — validates every structural
   override against the established `cut_matrix` primitive.
2. **Count preserved:** for `JointPartition`s, the new
   `num_connections_cut()` equals the value the deleted Eq. 24 override
   produced (pin a few golden counts, e.g. the existing `test_bounds.py`
   expectations `2`, `[4,6,8]`).
3. **n-invariance:** `removed_edges()` is unchanged when `cut_matrix` is
   evaluated at `n` larger than the minimal span.
4. **NullCut:** `removed_edges() == frozenset()` and `num_connections_cut() == 0`.
5. **Partial order:** `refines` is reflexive, antisymmetric w.r.t. edge-sets,
   transitive on a small sample; incomparable pairs exist (sanity that it's
   genuinely partial).
6. **Total order:** `sorted(partitions)` is deterministic and equals
   `sorted(partitions, key=lambda p: p.lex_key())`.

## Performance

The structural `removed_edges()` avoids materializing `cut_matrix` in the hot
path (distinction-φ normalization, bounds). `EdgeCut`-family types reuse their
held `_cut_matrix`; vertex-partition types enumerate from `parts`/`from/to`
node tuples. Expected net effect on `test_perf_budget.py`: neutral. The budget
test is run as part of verification.

## Files touched

- `pyphi/models/partitions.py` — `removed_edges`, `num_connections_cut`,
  `refines`/`coarsens`, `functools.total_ordering` + `__lt__`, docstrings.
- `pyphi/models/state_specification.py` — delete the hack; audit `None`-handling
  consumers.
- `test/test_models.py`, `test/test_partition.py` — the verification tests.
- `changelog.d/` — a `.refactor.md` fragment.
- `ROADMAP.md` — dashboard B7 row + Wave-2 prose.

## Risks & mitigations

- **Subset-as-`<` footgun (avoided by design):** the partial order is named
  methods only; `<` is the total `lex_key` order. Python's sort/min/heapq stay
  correct and deterministic.
- **Structural override diverging from `cut_matrix`:** caught by verification
  test §1 (every scheme, exhaustive on small n).
- **Hidden `None`-dependent consumers** of the normalization result: grep audit
  before deleting the `AttributeError` arm.

## Testing

`uv run pytest` (no path arg — includes the `pyphi/` doctest sweep), with the
partition/model/bounds suites and `test_perf_budget.py` green; slow lane
(`-m slow`) for the Hypothesis invariants since this touches a core model type.
