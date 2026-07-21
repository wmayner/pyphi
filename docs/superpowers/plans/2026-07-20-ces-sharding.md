# Scoped CES Sharding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Distribute one system's cause-effect structure (and SIA) across an
HTCondor campaign: declarative scope, an automatic shard-planning ladder
(mechanism → purview-range → partition-stride), tie-preserving exact merges,
and reconstruction through the existing `ces()` assembly with certified
bounds for what the scope excludes.

**Architecture:** Builds on the cycle-1 campaign infrastructure (spec
`docs/superpowers/specs/2026-07-20-ces-sharding-design.md`, companion
`2026-07-20-htcondor-campaign-design.md`). New task kinds ride the existing
directory format, runner, and status/resubmission flow; every shard executes
through existing seams (`distinction(cause_purviews=…)`,
`find_mip(partitions=…)`, `system.sia(partitions=…)`) and every merge
re-runs the existing `pyphi.resolve_ties` machinery on unions of shard tie
sets.

**Tech Stack:** Python 3.13, existing `pyphi.campaign` / `pyphi.cost` /
`pyphi.resolve_ties` / `pyphi.serialize` machinery. No new dependencies.

## Global Constraints

- Commit messages end with both trailers, each on its own line:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01PEAxNzhDCaTrntX3o1JqMV`.
- Never `git commit --no-verify`. After EVERY commit run
  `git log --oneline -1` (the ruff-format hook aborts commits silently).
- All Python invocations via `uv run`. Never pipe pytest through
  `tail`/`head`; redirect to a log file and read the summary line.
- Docstrings: NumPy style, final-state impersonal voice, Unicode symbols
  (`φ`, `Φ`, `Σφ_r`). No planning-artifact references in code, docstrings,
  comments, or changelog fragments.
- Tests that assert φ values pin formalism with complete presets
  (`config.override(**presets.by_name["IIT_4_0_2026"], ...)`); the headline
  test runs under both `IIT_4_0_2026` and `IIT_4_0_2023`.
- Every new `Displayable` type must also provide `to_pandas`
  (`test_every_displayable_has_to_pandas` enforces this).
- Do NOT touch concurrent sessions' files (`docs/whats-new-in-2.0.md`,
  `REVIEW-2026-07-13.md`, `TRIAGE-WAVE5.md`, `color-theory/`,
  `experiments/`, benchmark JSONs). Stage only files this plan names.
- Scope never reaches partition axes (a partial partition sweep would turn
  φ into an upper bound); partitions are shard-only.
- `AxisScope.explicit` is exclusive of every other constraint field
  (`ValueError` at construction).

## Verified mechanics (probed on main, 2026-07-20 — trust these)

- `find_mip(cs, direction, mechanism, purview, partitions=<slice>)` threads
  the slice through every specified-state pin
  (`_find_mip_iit4` in `pyphi/formalism/iit4/formalism.py:198`).
- `system.sia(partitions=<slice>)` works (`iit4.sia` has a
  `partitions: Iterable | None` parameter).
- **Partition-stride merge is exact** (winner identity included) when: per
  specified-state pin, the union of shard partition-tie sets is sorted back
  to global enumeration order, resolved with `resolve_ties.partitions`, and
  the per-pin winners resolved with `resolve_ties.states`. Probe: 3-stride
  split of a 10-partition sweep reproduced the full winner exactly.
- **Purview-range merge is exact**: per-purview RIAs (all of them, in the
  canonical `potential_purviews` order) wrapped in the MICE class and
  resolved with `resolve_ties.purviews` — this is literally what
  `find_mice` does internally.
- **SIA-stride merge is exact**: union of shard SIA tie sets (`.ties`)
  resolved with `resolve_ties.sias` reproduced the full `system.sia()`
  winner and partition.
- **Partition reprs depend on node labels**: enumerate with
  `mechanism_partitions(mechanism, purview, node_labels)` (as `find_mip`
  does internally) whenever using `str(partition)` as an identity key — an
  unlabeled enumeration produces different strings for the same cut.
- `system.ces(sia=…, distinctions=UnresolvedDistinctions(…))` assembles a
  `CauseEffectStructure` from precomputed pieces (congruence resolution +
  relations happen inside).
- `Concept(mechanism=…, cause=…, effect=…)` builds a distinction from a
  MIC/MIE pair (see `distinction()` in `pyphi/formalism/queries.py:385`).
- `potential_purviews(cs, direction, mechanism, purviews=<list>)`
  **intersects** the explicit list with the connectivity-pruned candidates
  (`pyphi/core/repertoire_algebra.py:721`) — scope can only narrow.
- `resolve_ties` entry points: `partitions` (min), `purviews` (max),
  `states` (max), `sias` (min) — each returns an iterator over the tie set,
  first element is the winner (`pyphi/resolve_ties.py:815-861`).
- `NodeLabels.coerce_to_indices` (`pyphi/labels.py:126`) normalizes
  label-or-index unit collections.
- Tie sets survive serialization (partition/state/purview/SIA ties;
  peers encoded explicitly in `pyphi/serialize/convert.py`).
- Bottleneck-first ordering (sort by count of present-in-cm connections
  severed, ascending) finds a φ=0 partition at position 0 on a sparse
  chain; zero-cut partitions evaluate to exactly φ=0.
- `sum_phi_relations_measured_bound(distinctions)` /
  `big_phi_measured_bound(distinctions)` in `pyphi/formalism/iit4/bounds.py`
  take a resolved distinction iterable and return an `UpperBound`.
- Merged partition margins are derivable **without extra bookkeeping**:
  global margin = min( min over non-winning shards of (shard winner's
  normalized φ) − winner's normalized φ, the winning shard's own
  `partition_margin` ); `None` if any contributing shard's margin is `None`
  (short-circuited or single-candidate slice).
- `_find_mip_single_state` materializes its `partitions` argument with
  `list(...)` — a stride slice is bounded by the per-job budget, so
  materializing the slice is fine; only the full enumeration must stay lazy.

## File Map

| File | Responsibility |
|---|---|
| `pyphi/campaign/scope.py` (create) | `AxisScope`, `CESScope`, resolution against node labels |
| `pyphi/cost.py` (modify) | `scope=` on `estimate_analysis`; `mechanism_workloads()`; `partition_sweep_count()` |
| `pyphi/campaign/shards.py` (create) | `ShardSpec`, planning ladder, stride enumeration, bottleneck ordering |
| `pyphi/campaign/__init__.py` (modify) | shard task types, `prepare(kind="ces")`, CES `collect()`, `ScopeReport` |
| `pyphi/campaign/merge.py` (create) | tie-preserving merges (strides → RIAs → MICE → distinctions; SIA) |
| `pyphi/campaign/runner.py` (modify) | dispatch on task kind; shard execution + aux bookkeeping |
| `pyphi/serialize/schema.py` + `convert.py` (modify) | scope schemas, shard-task schemas, `CellOutput.aux` |
| `pyphi/mcp/server.py` (modify) | scope on `estimate_cost`; CES args on `prepare_campaign` |
| `test/campaign/` (extend) | scope/planner/merge units + the sharded ≡ unsharded headline |
| `docs/howto/campaigns.md`, `pyphi/mcp/content/campaigns.md` (extend) | scope + CES campaign sections |
| `changelog.d/`, `ROADMAP.md` | fragment; P11 row completion |

---

### Task 1: Scope objects

**Files:**
- Create: `pyphi/campaign/scope.py`
- Modify: `pyphi/serialize/schema.py`, `pyphi/serialize/convert.py`
- Test: `test/campaign/test_scope.py`

**Interfaces:**
- Produces (used by Tasks 2, 3, 5, 6):

```python
@dataclass(frozen=True)
class AxisScope:
    explicit: tuple[tuple[int, ...], ...] | None = None
    min_order: int | None = None
    max_order: int | None = None
    containing: tuple[int, ...] | None = None
    within: tuple[int, ...] | None = None
    # .unconstrained -> bool
    # .admits(units: tuple[int, ...]) -> bool
    # .select(candidates: Iterable[tuple[int, ...]]) -> Iterator[tuple[int, ...]]

@dataclass(frozen=True)
class CESScope:
    mechanisms: AxisScope = AxisScope()
    cause_purviews: AxisScope = AxisScope()
    effect_purviews: AxisScope = AxisScope()
    # .purviews(direction) -> AxisScope   (Direction.CAUSE / Direction.EFFECT)

def resolve_scope(scope: CESScope, node_labels) -> CESScope
    # labels → indices via NodeLabels.coerce_to_indices; unit tuples sorted
```

- [ ] **Step 1: Write the failing tests**

`test/campaign/test_scope.py`:

```python
import pytest

from pyphi import examples
from pyphi.campaign.scope import AxisScope
from pyphi.campaign.scope import CESScope
from pyphi.campaign.scope import resolve_scope
from pyphi.direction import Direction
from pyphi.serialize import load
from pyphi.serialize import save

CANDIDATES = [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]


def test_unconstrained_admits_everything():
    scope = AxisScope()
    assert scope.unconstrained
    assert list(scope.select(CANDIDATES)) == CANDIDATES


def test_explicit_is_the_axis():
    scope = AxisScope(explicit=((0, 1), (2,)))
    assert list(scope.select(CANDIDATES)) == [(2,), (0, 1)]


def test_explicit_excludes_other_fields():
    with pytest.raises(ValueError, match="exclusive"):
        AxisScope(explicit=((0,),), max_order=2)


def test_constraints_intersect():
    scope = AxisScope(max_order=2, containing=(0,))
    assert list(scope.select(CANDIDATES)) == [(0,), (0, 1), (0, 2)]
    scope = AxisScope(min_order=2, within=(0, 1))
    assert list(scope.select(CANDIDATES)) == [(0, 1)]


def test_ces_scope_directions():
    scope = CESScope(cause_purviews=AxisScope(max_order=1))
    assert scope.purviews(Direction.CAUSE).max_order == 1
    assert scope.purviews(Direction.EFFECT).unconstrained


def test_resolve_scope_coerces_labels():
    substrate = examples.basic_substrate()
    labels = list(map(str, substrate.node_labels))
    scope = CESScope(mechanisms=AxisScope(containing=(labels[0],)))
    resolved = resolve_scope(scope, substrate.node_labels)
    assert resolved.mechanisms.containing == (0,)


def test_scope_roundtrips(tmp_path):
    scope = CESScope(
        mechanisms=AxisScope(explicit=((0, 1), (2,))),
        effect_purviews=AxisScope(max_order=2, within=(0, 1, 2)),
    )
    save(scope, tmp_path / "scope.json.gz")
    assert load(tmp_path / "scope.json.gz") == scope
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/campaign/test_scope.py -x -q > /tmp/c2t1a.log 2>&1`; read the log.
Expected: FAIL with `ModuleNotFoundError: No module named 'pyphi.campaign.scope'`.

- [ ] **Step 3: Implement `pyphi/campaign/scope.py`**

```python
"""Declarative feasibility surfaces for scoped cause-effect analyses.

A scope states which mechanisms and purviews a computation considers.
Exclusions are explicit and certified — scope changes *what* is computed;
it never silently approximates. Constraint fields are named data (no
callables), so scopes serialize, ship to batch jobs, and land in
provenance. Partition sweeps cannot be scoped: a partial sweep would turn
φ into an upper bound.
"""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Iterator
from dataclasses import dataclass
from dataclasses import field

from pyphi.direction import Direction


@dataclass(frozen=True)
class AxisScope:
    """A constraint on one axis of unit sets (mechanisms or purviews).

    Constraint fields combine by intersection. ``explicit`` is exclusive:
    an explicit list *is* the axis, so combining it with any other field
    raises :class:`ValueError`. The default (all fields ``None``) admits
    every candidate.
    """

    explicit: tuple[tuple[int, ...], ...] | None = None
    min_order: int | None = None
    max_order: int | None = None
    containing: tuple[int, ...] | None = None
    within: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        others = (self.min_order, self.max_order, self.containing, self.within)
        if self.explicit is not None and any(o is not None for o in others):
            raise ValueError(
                "explicit is exclusive: an explicit list is the axis and "
                "cannot combine with other constraint fields"
            )

    @property
    def unconstrained(self) -> bool:
        return (
            self.explicit is None
            and self.min_order is None
            and self.max_order is None
            and self.containing is None
            and self.within is None
        )

    def admits(self, units: tuple[int, ...]) -> bool:
        if self.explicit is not None:
            return tuple(sorted(units)) in {
                tuple(sorted(e)) for e in self.explicit
            }
        if self.min_order is not None and len(units) < self.min_order:
            return False
        if self.max_order is not None and len(units) > self.max_order:
            return False
        if self.containing is not None and not set(self.containing) <= set(units):
            return False
        if self.within is not None and not set(units) <= set(self.within):
            return False
        return True

    def select(
        self, candidates: Iterable[tuple[int, ...]]
    ) -> Iterator[tuple[int, ...]]:
        """Yield the candidates this scope admits, preserving their order."""
        if self.explicit is not None:
            allowed = {tuple(sorted(e)) for e in self.explicit}
            for candidate in candidates:
                if tuple(sorted(candidate)) in allowed:
                    yield candidate
            return
        for candidate in candidates:
            if self.admits(candidate):
                yield candidate


@dataclass(frozen=True)
class CESScope:
    """The feasibility surface of a cause-effect structure computation."""

    mechanisms: AxisScope = field(default_factory=AxisScope)
    cause_purviews: AxisScope = field(default_factory=AxisScope)
    effect_purviews: AxisScope = field(default_factory=AxisScope)

    def purviews(self, direction: Direction) -> AxisScope:
        if direction == Direction.CAUSE:
            return self.cause_purviews
        return self.effect_purviews


def _resolve_units(units: tuple | None, node_labels) -> tuple[int, ...] | None:
    if units is None:
        return None
    return tuple(sorted(node_labels.coerce_to_indices(units)))


def _resolve_axis(scope: AxisScope, node_labels) -> AxisScope:
    return AxisScope(
        explicit=None
        if scope.explicit is None
        else tuple(_resolve_units(e, node_labels) for e in scope.explicit),
        min_order=scope.min_order,
        max_order=scope.max_order,
        containing=_resolve_units(scope.containing, node_labels),
        within=_resolve_units(scope.within, node_labels),
    )


def resolve_scope(scope: CESScope, node_labels) -> CESScope:
    """Return the scope with every unit reference normalized to indices."""
    return CESScope(
        mechanisms=_resolve_axis(scope.mechanisms, node_labels),
        cause_purviews=_resolve_axis(scope.cause_purviews, node_labels),
        effect_purviews=_resolve_axis(scope.effect_purviews, node_labels),
    )
```

(`_resolve_axis` builds a new `AxisScope`, which re-runs `__post_init__`
validation — intended. `coerce_to_indices` accepts labels or indices; check
its exact call convention at `pyphi/labels.py:126` and adapt the two call
sites if it takes varargs rather than an iterable.)

- [ ] **Step 4: Register serialization**

In `pyphi/serialize/schema.py`, after `CampaignTaskOutputSchema`:

```python
class AxisScopeSchema(msgspec.Struct, frozen=True, tag="axis_scope"):
    explicit: tuple[tuple[int, ...], ...] | None
    min_order: int | None
    max_order: int | None
    containing: tuple[int, ...] | None
    within: tuple[int, ...] | None


class CESScopeSchema(msgspec.Struct, frozen=True, tag="ces_scope"):
    mechanisms: AxisScopeSchema
    cause_purviews: AxisScopeSchema
    effect_purviews: AxisScopeSchema
```

Add both to the `Schema` union (after `CampaignTaskOutputSchema`). In
`pyphi/serialize/convert.py`, extend `_register_campaign()`:

```python
    from pyphi.campaign.scope import AxisScope
    from pyphi.campaign.scope import CESScope

    _ENCODERS[AxisScope] = lambda a: schema.AxisScopeSchema(
        explicit=a.explicit,
        min_order=a.min_order,
        max_order=a.max_order,
        containing=a.containing,
        within=a.within,
    )

    def _decode_axis_scope(s: schema.AxisScopeSchema) -> Any:
        return AxisScope(
            explicit=None
            if s.explicit is None
            else tuple(tuple(e) for e in s.explicit),
            min_order=s.min_order,
            max_order=s.max_order,
            containing=None if s.containing is None else tuple(s.containing),
            within=None if s.within is None else tuple(s.within),
        )

    _DECODERS[schema.AxisScopeSchema] = _decode_axis_scope

    _ENCODERS[CESScope] = lambda c: schema.CESScopeSchema(
        mechanisms=to_schema(c.mechanisms),
        cause_purviews=to_schema(c.cause_purviews),
        effect_purviews=to_schema(c.effect_purviews),
    )

    def _decode_ces_scope(s: schema.CESScopeSchema) -> Any:
        return CESScope(
            mechanisms=from_schema(s.mechanisms),
            cause_purviews=from_schema(s.cause_purviews),
            effect_purviews=from_schema(s.effect_purviews),
        )

    _DECODERS[schema.CESScopeSchema] = _decode_ces_scope
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest test/campaign/test_scope.py test/serialize/ -q > /tmp/c2t1b.log 2>&1`; read the log.
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyphi/campaign/scope.py pyphi/serialize/schema.py pyphi/serialize/convert.py test/campaign/test_scope.py
git commit -m "Add declarative scope objects for cause-effect analyses"
git log --oneline -1
```

---

### Task 2: Scope-aware estimation

**Files:**
- Modify: `pyphi/cost.py`
- Test: `test/test_cost.py` (append)

**Interfaces:**
- Consumes: Task 1's `CESScope` (duck-typed: only `.mechanisms.select`,
  `.purviews(direction).select` are used; import deferred inside the
  function to avoid a cost→campaign import cycle).
- Produces (used by Tasks 3 and 5):

```python
def estimate_analysis(substrate, subset=None, compute=None,
                      limit=1_000_000, scope=None) -> AnalysisEstimate
def mechanism_workloads(substrate, subset=None, scope=None,
                        limit=10_000_000) -> dict[tuple[int, ...], int]
    # mechanism -> its scoped purview evaluations + partition sweeps;
    # raises ValueError when the walk exceeds `limit`
def partition_sweep_count(mechanism_size: int, purview_size: int) -> int
    # memoized count of mechanism partitions for one (m, p) pair under the
    # active scheme
```

- [ ] **Step 1: Write the failing tests**

Append to `test/test_cost.py` (mirror its existing imports; add
`from pyphi.campaign.scope import AxisScope, CESScope` and
`from pyphi.cost import mechanism_workloads, partition_sweep_count`):

```python
class TestScopedEstimation:
    def test_scope_narrows_counts(self):
        substrate = examples.basic_substrate()
        with config.override(**presets.by_name["IIT_4_0_2026"]):
            full = estimate_analysis(substrate, compute="ces")
            scoped = estimate_analysis(
                substrate,
                compute="ces",
                scope=CESScope(mechanisms=AxisScope(max_order=1)),
            )
        assert scoped.mechanisms == 3  # singletons only
        assert scoped.purview_evaluations < full.purview_evaluations
        assert scoped.mechanism_partition_sweeps < full.mechanism_partition_sweeps

    def test_purview_scope_narrows_purview_axis(self):
        substrate = examples.basic_substrate()
        scope = CESScope(
            cause_purviews=AxisScope(max_order=1),
            effect_purviews=AxisScope(max_order=1),
        )
        with config.override(**presets.by_name["IIT_4_0_2026"]):
            full = estimate_analysis(substrate, compute="ces")
            scoped = estimate_analysis(substrate, compute="ces", scope=scope)
        assert scoped.mechanisms == full.mechanisms
        assert scoped.purview_evaluations < full.purview_evaluations

    def test_mechanism_workloads_sum_matches_estimate(self):
        substrate = examples.basic_substrate()
        scope = CESScope(mechanisms=AxisScope(containing=(0,)))
        with config.override(**presets.by_name["IIT_4_0_2026"]):
            workloads = mechanism_workloads(substrate, scope=scope)
            scoped = estimate_analysis(substrate, compute="ces", scope=scope)
        assert set(workloads) == {(0,), (0, 1), (0, 2), (0, 1, 2)}
        assert sum(workloads.values()) == (
            scoped.purview_evaluations + scoped.mechanism_partition_sweeps
        )

    def test_partition_sweep_count_matches_enumeration(self):
        from pyphi.partition import mechanism_partitions

        with config.override(**presets.by_name["IIT_4_0_2026"]):
            count = partition_sweep_count(2, 2)
            enumerated = len(list(mechanism_partitions((0, 1), (0, 2))))
        assert count == enumerated
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/test_cost.py -k Scoped -x -q > /tmp/c2t2a.log 2>&1`; read the log.
Expected: FAIL with `ImportError` (`mechanism_workloads`).

- [ ] **Step 3: Implement**

In `pyphi/cost.py`:

1. Add `scope: Any | None = None` to `estimate_analysis`'s signature
   (after `limit`), document it in the docstring
   (`scope : CESScope, optional — restrict the counted mechanisms and
   purviews to the scope's feasibility surface; see
   pyphi.campaign.scope`), and thread it into the walk — replace the
   mechanism/purview loops (currently `pyphi/cost.py:399-406`) with:

```python
            mechanism_iter: Any = utils.powerset(indices, nonempty=True)
            if scope is not None:
                mechanism_iter = scope.mechanisms.select(mechanism_iter)
            mechanisms = 0
            purview_evaluations = 0
            sweeps = 0
            for mechanism in mechanism_iter:
                mechanisms += 1
                for direction in (Direction.CAUSE, Direction.EFFECT):
                    purviews = cs.potential_purviews(direction, mechanism)
                    if scope is not None:
                        purviews = list(scope.purviews(direction).select(purviews))
                    for purview in purviews:
                        counter.charge(1)
                        purview_evaluations += 1
                        sweeps += _mechanism_partition_count(
                            len(mechanism), len(purview), counter
                        )
```

   (Delete the now-redundant `mechanisms = 2**m - 1` line; with no scope
   the loop counts the same total. `possible_distinctions` /
   `possible_relations` stay unscoped — they are structural ceilings of
   the full system.)

2. Add the two public helpers at module level:

```python
def partition_sweep_count(mechanism_size: int, purview_size: int) -> int:
    """Memoized mechanism-partition count for one (mechanism, purview) pair
    under the active mechanism partition scheme."""
    counter = _Counter(None)
    return _mechanism_partition_count(mechanism_size, purview_size, counter)


def mechanism_workloads(
    substrate: "Substrate",
    subset: Any = None,
    scope: Any | None = None,
    limit: int = 10_000_000,
) -> dict[tuple[int, ...], int]:
    """Per-mechanism workload under a scope: purview evaluations plus
    mechanism-partition sweeps, keyed by mechanism.

    The raw data behind shard planning: the sum over all mechanisms equals
    the scoped ``estimate_analysis`` totals for the distinction axis.

    Raises
    ------
    ValueError
        If the counting walk exceeds ``limit`` — the workload is then too
        large to plan; narrow the scope or raise the limit.
    """
    from pyphi import utils
    from pyphi.direction import Direction
    from pyphi.system import System

    cs = System.from_substrate(substrate, (0,) * substrate.size, subset)
    counter = _Counter(limit)
    workloads: dict[tuple[int, ...], int] = {}
    mechanism_iter: Any = utils.powerset(cs.node_indices, nonempty=True)
    if scope is not None:
        mechanism_iter = scope.mechanisms.select(mechanism_iter)
    try:
        for mechanism in mechanism_iter:
            units = 0
            for direction in (Direction.CAUSE, Direction.EFFECT):
                purviews = cs.potential_purviews(direction, mechanism)
                if scope is not None:
                    purviews = list(scope.purviews(direction).select(purviews))
                for purview in purviews:
                    counter.charge(1)
                    units += 1 + _mechanism_partition_count(
                        len(mechanism), len(purview), counter
                    )
            workloads[tuple(mechanism)] = units
    except _LimitReached:
        raise ValueError(
            f"mechanism workload walk exceeded limit={limit}; narrow the "
            "scope or raise the limit"
        ) from None
    return workloads
```

   Check `_Counter(None)` handles an unlimited budget (read the `_Counter`
   class at `pyphi/cost.py:39`); if it requires an int, use a very large
   sentinel (`2**63`).

   Note `mechanism_workloads` counts `1 + partition_count` per purview
   (evaluation + sweeps) while `estimate_analysis` reports them as two
   separate fields — the workloads sum must equal
   `purview_evaluations + mechanism_partition_sweeps` (the test asserts
   this).

3. Extend `__all__` with the two new names.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/test_cost.py -q > /tmp/c2t2b.log 2>&1`; read the log.
Expected: PASS (all — including the existing unscoped tests, whose counts
must be unchanged by the loop rewrite).

- [ ] **Step 5: Commit**

```bash
git add pyphi/cost.py test/test_cost.py
git commit -m "Add scope-aware estimation and per-mechanism workloads"
git log --oneline -1
```

---

### Task 3: Shard planner, stride enumeration, bottleneck ordering

**Files:**
- Create: `pyphi/campaign/shards.py`
- Test: `test/campaign/test_shards.py`

**Interfaces:**
- Consumes: `CESScope` (Task 1), `mechanism_workloads` /
  `partition_sweep_count` (Task 2), `cost_balanced_partition`
  (`pyphi/parallel/chunking.py`), `mechanism_partitions` /
  `system_partition_types` (`pyphi/partition.py`).
- Produces (used by Tasks 5 and 6):

```python
@dataclass(frozen=True)
class ShardSpec:
    payload_kind: str  # "mechanisms" | "purview_range" | "partition_stride"
    mechanisms: tuple[tuple[int, ...], ...] = ()          # payload "mechanisms"
    mechanism: tuple[int, ...] | None = None              # range / stride
    direction: str | None = None                          # "CAUSE" | "EFFECT"
    purviews: tuple[tuple[int, ...], ...] = ()            # payload "purview_range"
    purview: tuple[int, ...] | None = None                # payload "partition_stride"
    stride: tuple[int, int] | None = None                 # (i, k)
    units: float = 0.0                                    # estimated work

def plan_ces_shards(system, scope, units_per_job, limit=10_000_000) -> list[ShardSpec]
def plan_sia_shards(system, units_per_job) -> list[ShardSpec]
    # payload "partition_stride" with mechanism=None (system partitions)
def enumerate_partition_stride(mechanism, purview, node_labels, i, k)
    -> tuple[list, list[int]]     # (partitions, their global enumeration indices)
def enumerate_system_partition_stride(system, scheme, i, k)
    -> tuple[list, list[int]]
def bottleneck_order(partitions, indices, cm, direction) -> tuple[list, list[int]]
    # both lists reordered by ascending count of present-in-cm connections cut
def cut_present_edges(partition, cm, direction) -> int
```

- [ ] **Step 1: Write the failing tests**

`test/campaign/test_shards.py`:

```python
import numpy as np

from pyphi import examples
from pyphi.campaign.scope import AxisScope
from pyphi.campaign.scope import CESScope
from pyphi.campaign.shards import bottleneck_order
from pyphi.campaign.shards import enumerate_partition_stride
from pyphi.campaign.shards import plan_ces_shards
from pyphi.campaign.shards import plan_sia_shards
from pyphi.conf import config
from pyphi.conf import presets
from pyphi.direction import Direction
from pyphi.partition import mechanism_partitions
from pyphi.system import System

PIN = dict(parallel=False, progress_bars=False)


def _system():
    return System(examples.basic_substrate(), (1, 0, 0))


def test_generous_budget_yields_mechanism_shards():
    with config.override(**presets.by_name["IIT_4_0_2026"], **PIN):
        specs = plan_ces_shards(_system(), CESScope(), units_per_job=1e9)
    assert all(s.payload_kind == "mechanisms" for s in specs)
    covered = sorted(m for s in specs for m in s.mechanisms)
    assert covered == sorted(
        [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
    )


def test_tiny_budget_descends_the_ladder():
    with config.override(**presets.by_name["IIT_4_0_2026"], **PIN):
        specs = plan_ces_shards(_system(), CESScope(), units_per_job=2.0)
    kinds = {s.payload_kind for s in specs}
    assert "partition_stride" in kinds
    # Strides for one (mechanism, direction, purview) cover disjoint indices.
    strides = [
        s for s in specs
        if s.payload_kind == "partition_stride" and s.stride is not None
    ]
    assert strides, "expected stride shards under a tiny budget"
    i, k = strides[0].stride
    assert 0 <= i < k


def test_plan_is_deterministic():
    with config.override(**presets.by_name["IIT_4_0_2026"], **PIN):
        a = plan_ces_shards(_system(), CESScope(), units_per_job=5.0)
        b = plan_ces_shards(_system(), CESScope(), units_per_job=5.0)
    assert a == b


def test_scope_restricts_the_plan():
    scope = CESScope(mechanisms=AxisScope(explicit=((0,),)))
    with config.override(**presets.by_name["IIT_4_0_2026"], **PIN):
        specs = plan_ces_shards(_system(), scope, units_per_job=1e9)
    covered = [m for s in specs for m in s.mechanisms]
    assert covered == [(0,)]


def test_stride_enumeration_partitions_the_enumeration():
    system = _system()
    mechanism, purview = (0, 1), (0, 2)
    with config.override(**presets.by_name["IIT_4_0_2026"], **PIN):
        full = list(
            mechanism_partitions(mechanism, purview, system.node_labels)
        )
        k = 3
        seen_indices = []
        seen_parts = []
        for i in range(k):
            parts, indices = enumerate_partition_stride(
                mechanism, purview, system.node_labels, i, k
            )
            assert indices == list(range(i, len(full), k))
            seen_indices.extend(indices)
            seen_parts.extend(str(p) for p in parts)
    assert sorted(seen_indices) == list(range(len(full)))
    assert sorted(seen_parts) == sorted(str(p) for p in full)


def test_bottleneck_order_finds_zero_cut_first():
    # Sparse chain 0 -> 1 -> 2 -> 3 (with self-loops): far-apart mechanism
    # and purview admit partitions that cut no present connection.
    cm = np.array(
        [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]]
    )
    with config.override(**presets.by_name["IIT_4_0_2026"], **PIN):
        parts = list(mechanism_partitions((0, 3), (1, 2)))
        ordered, indices = bottleneck_order(
            parts, list(range(len(parts))), cm, Direction.EFFECT
        )
    from pyphi.campaign.shards import cut_present_edges

    counts = [cut_present_edges(p, cm, Direction.EFFECT) for p in ordered]
    assert counts == sorted(counts)
    assert counts[0] == 0
    assert len(indices) == len(parts)


def test_sia_shards_cover_system_partitions():
    with config.override(**presets.by_name["IIT_4_0_2026"], **PIN):
        specs = plan_sia_shards(_system(), units_per_job=5.0)
    assert all(s.payload_kind == "partition_stride" for s in specs)
    assert all(s.mechanism is None for s in specs)
    ks = {s.stride[1] for s in specs}
    assert len(ks) == 1
    assert sorted(s.stride[0] for s in specs) == list(range(ks.pop()))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/campaign/test_shards.py -x -q > /tmp/c2t3a.log 2>&1`; read the log.
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `pyphi/campaign/shards.py`**

```python
"""Shard planning for scoped cause-effect campaigns.

The planner descends a three-rung ladder, splitting only where the per-job
budget requires: whole mechanisms are cost-balance-packed into shards; a
mechanism over budget splits its scoped (direction, purview) list into
cost-balanced ranges; a single (mechanism, direction, purview) pair over
budget splits its partition enumeration into interleaved strides (shard i
of k evaluates partitions i, i+k, i+2k, …), which balances any systematic
cost trend along the enumeration. Sharding never changes results — every
shard executes exact computations over a subset, and collection merges tie
sets losslessly.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from itertools import islice
from typing import Any

from pyphi.conf import config
from pyphi.cost import mechanism_workloads
from pyphi.cost import partition_sweep_count
from pyphi.direction import Direction
from pyphi.parallel.chunking import cost_balanced_partition
from pyphi.partition import mechanism_partitions
from pyphi.partition import system_partition_types
from pyphi.warnings import PyPhiWarning

__all__ = [
    "ShardSpec",
    "bottleneck_order",
    "cut_present_edges",
    "enumerate_partition_stride",
    "enumerate_system_partition_stride",
    "plan_ces_shards",
    "plan_sia_shards",
]


@dataclass(frozen=True)
class ShardSpec:
    """One shard of a scoped analysis: what to compute and how it was split."""

    payload_kind: str
    mechanisms: tuple[tuple[int, ...], ...] = ()
    mechanism: tuple[int, ...] | None = None
    direction: str | None = None
    purviews: tuple[tuple[int, ...], ...] = ()
    purview: tuple[int, ...] | None = None
    stride: tuple[int, int] | None = None
    units: float = 0.0


def enumerate_partition_stride(
    mechanism: tuple[int, ...],
    purview: tuple[int, ...],
    node_labels: Any,
    i: int,
    k: int,
) -> tuple[list, list[int]]:
    """Materialize stride ``i`` of ``k`` of the partition enumeration.

    Returns the partitions and their global enumeration indices. Only the
    stride is materialized; the full enumeration is consumed lazily. The
    enumeration must use the same ``node_labels`` the analysis uses, so
    partition identities (their string forms) agree across processes.
    """
    parts = list(
        islice(mechanism_partitions(mechanism, purview, node_labels), i, None, k)
    )
    return parts, [i + j * k for j in range(len(parts))]


def enumerate_system_partition_stride(
    system: Any, scheme: str, i: int, k: int
) -> tuple[list, list[int]]:
    """Materialize stride ``i`` of ``k`` of the system-partition enumeration."""
    generator = system_partition_types[scheme](
        system.partition_indices, node_labels=system.node_labels
    )
    parts = list(islice(generator, i, None, k))
    return parts, [i + j * k for j in range(len(parts))]


def cut_present_edges(partition: Any, cm: Any, direction: Direction) -> int:
    """Count present-in-cm connections severed by a mechanism partition."""
    parts = list(partition)
    count = 0
    for a, part_a in enumerate(parts):
        for b, part_b in enumerate(parts):
            if a == b:
                continue
            for m in part_a.mechanism:
                for p in part_b.purview:
                    src, dst = (m, p) if direction == Direction.EFFECT else (p, m)
                    if cm[src, dst]:
                        count += 1
    return count


def bottleneck_order(
    partitions: list, indices: list[int], cm: Any, direction: Direction
) -> tuple[list, list[int]]:
    """Reorder a partition slice so likely-reducible partitions come first.

    Sorts by ascending count of severed present connections: a partition
    that cuts no present connection yields φ = 0, so on sparse substrates
    the sweep's zero-φ short-circuit fires within the first evaluations.
    Ordering never affects results — the minimum is order-independent and
    tie resolution runs on the collected set — only time to short-circuit.
    The sort is stable, so equal-count partitions keep enumeration order.
    """
    keyed = sorted(
        zip(partitions, indices, strict=True),
        key=lambda pair: cut_present_edges(pair[0], cm, direction),
    )
    return [p for p, _ in keyed], [i for _, i in keyed]


def _pack_specs(items: list[ShardSpec], units_per_job: float) -> list[ShardSpec]:
    """Cost-balance whole-mechanism items into "mechanisms" shards."""
    if not items:
        return []
    weights = [s.units for s in items]
    jobs = max(1, math.ceil(sum(weights) / units_per_job))
    bins = cost_balanced_partition(weights, jobs)
    packed = []
    for indices in (sorted(b) for b in bins):
        packed.append(
            ShardSpec(
                payload_kind="mechanisms",
                mechanisms=tuple(
                    m for i in indices for m in items[i].mechanisms
                ),
                units=float(sum(items[i].units for i in indices)),
            )
        )
    return packed


def plan_ces_shards(
    system: Any,
    scope: Any,
    units_per_job: float,
    limit: int = 10_000_000,
) -> list[ShardSpec]:
    """Plan the shards of a scoped cause-effect computation.

    Descends mechanism → purview-range → partition-stride only where the
    budget requires. Deterministic for fixed inputs; every spec carries its
    estimated work units.
    """
    workloads = mechanism_workloads(
        system.substrate, subset=system.node_indices, scope=scope, limit=limit
    )
    whole: list[ShardSpec] = []
    specs: list[ShardSpec] = []
    for mechanism, units in workloads.items():
        if units <= units_per_job:
            whole.append(
                ShardSpec(
                    payload_kind="mechanisms",
                    mechanisms=(mechanism,),
                    units=float(units),
                )
            )
            continue
        # Rung 2: split this mechanism's (direction, purview) list.
        for direction in (Direction.CAUSE, Direction.EFFECT):
            purviews = system.potential_purviews(direction, mechanism)
            purviews = list(scope.purviews(direction).select(purviews))
            if not purviews:
                continue
            weights = [
                1.0 + partition_sweep_count(len(mechanism), len(p))
                for p in purviews
            ]
            oversized = [
                (p, w) for p, w in zip(purviews, weights, strict=True)
                if w > units_per_job
            ]
            fitting = [
                (p, w) for p, w in zip(purviews, weights, strict=True)
                if w <= units_per_job
            ]
            if fitting:
                jobs = max(
                    1,
                    math.ceil(sum(w for _, w in fitting) / units_per_job),
                )
                bins = cost_balanced_partition([w for _, w in fitting], jobs)
                for bin_indices in (sorted(b) for b in bins):
                    specs.append(
                        ShardSpec(
                            payload_kind="purview_range",
                            mechanism=mechanism,
                            direction=direction.name,
                            purviews=tuple(fitting[i][0] for i in bin_indices),
                            units=float(
                                sum(fitting[i][1] for i in bin_indices)
                            ),
                        )
                    )
            # Rung 3: stride each oversized pair.
            for purview, weight in oversized:
                count = partition_sweep_count(len(mechanism), len(purview))
                k = min(math.ceil(weight / units_per_job), count)
                if weight / k > units_per_job:
                    warnings.warn(
                        f"budget units_per_job={units_per_job:.3g} is "
                        f"unreachable for mechanism {mechanism} purview "
                        f"{purview} ({count} partitions); one partition per "
                        "shard is the floor",
                        PyPhiWarning,
                        stacklevel=2,
                    )
                for i in range(k):
                    specs.append(
                        ShardSpec(
                            payload_kind="partition_stride",
                            mechanism=mechanism,
                            direction=direction.name,
                            purview=purview,
                            stride=(i, k),
                            units=float(weight / k),
                        )
                    )
    return _pack_specs(whole, units_per_job) + specs


def plan_sia_shards(system: Any, units_per_job: float) -> list[ShardSpec]:
    """Plan system-partition strides for the system irreducibility analysis."""
    scheme = config.formalism.iit.system_partition_scheme
    total = sum(
        1
        for _ in system_partition_types[scheme](
            system.partition_indices, node_labels=system.node_labels
        )
    )
    k = max(1, min(math.ceil(total / units_per_job), total))
    return [
        ShardSpec(
            payload_kind="partition_stride",
            mechanism=None,
            stride=(i, k),
            units=float(total / k),
        )
        for i in range(k)
    ]
```

(`plan_sia_shards` counts the enumeration by iterating it once — fine for
schemes whose counts are modest; the count is exactly what
`estimate_analysis(compute="sia")` reports, so if profiling ever matters,
swap the loop for `_system_partition_count`. Do not do that now.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/campaign/test_shards.py -q > /tmp/c2t3b.log 2>&1`; read the log.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyphi/campaign/shards.py test/campaign/test_shards.py
git commit -m "Add shard planner with stride enumeration and bottleneck ordering"
git log --oneline -1
```

---

### Task 4: Shard task types and serialization

**Files:**
- Modify: `pyphi/campaign/__init__.py` (types), `pyphi/serialize/schema.py`,
  `pyphi/serialize/convert.py`
- Test: `test/campaign/test_types.py` (append)

**Interfaces:**
- Consumes: `ShardSpec` (Task 3), `CESScope` (Task 1), cycle-1
  `CellOutput`/`CampaignTaskOutput`.
- Produces (used by Tasks 5–8):

```python
@dataclass(frozen=True)
class CESShardTask:
    task_id: int
    kind: str                          # "ces_shard"
    substrate_label: Any               # str | int
    state: tuple[int, ...]
    subset: tuple[int, ...] | None
    scope: Any                         # resolved CESScope
    config_overrides: dict[str, Any]
    formalism: str
    spec: ShardSpec
    ordering: str | None               # None | "bottleneck_first"

@dataclass(frozen=True)
class SIAShardTask:
    task_id: int
    kind: str                          # "sia_shard"
    substrate_label: Any
    state: tuple[int, ...]
    subset: tuple[int, ...] | None
    config_overrides: dict[str, Any]
    formalism: str
    stride: tuple[int, int]
```

- `CellOutput` gains `aux: dict[str, Any] | None = None` (per-entry
  bookkeeping: tie enumeration indices, partition scheme). The cycle-1
  encoder/decoder and schema are extended accordingly (new schema field
  with default `None`).

- [ ] **Step 1: Write the failing tests**

Append to `test/campaign/test_types.py`:

```python
def test_ces_shard_task_roundtrip(tmp_path):
    from pyphi.campaign import CESShardTask
    from pyphi.campaign.scope import AxisScope
    from pyphi.campaign.scope import CESScope
    from pyphi.campaign.shards import ShardSpec

    task = CESShardTask(
        task_id=1,
        kind="ces_shard",
        substrate_label="sys",
        state=(1, 0, 0),
        subset=(0, 1, 2),
        scope=CESScope(mechanisms=AxisScope(max_order=2)),
        config_overrides={"precision": 13},
        formalism="IIT_4_0_2026",
        spec=ShardSpec(
            payload_kind="partition_stride",
            mechanism=(0, 1),
            direction="EFFECT",
            purview=(0, 2),
            stride=(1, 3),
            units=4.0,
        ),
        ordering="bottleneck_first",
    )
    save(task, tmp_path / "t.json.gz")
    assert load(tmp_path / "t.json.gz") == task


def test_sia_shard_task_roundtrip(tmp_path):
    from pyphi.campaign import SIAShardTask

    task = SIAShardTask(
        task_id=2,
        kind="sia_shard",
        substrate_label="sys",
        state=(1, 0, 0),
        subset=None,
        config_overrides={},
        formalism="IIT_4_0_2026",
        stride=(0, 2),
    )
    save(task, tmp_path / "t.json.gz")
    assert load(tmp_path / "t.json.gz") == task


def test_cell_output_aux_roundtrip(tmp_path):
    out = CampaignTaskOutput(
        task_id=0,
        pyphi_version="test",
        entries=(
            CellOutput(
                status="ok",
                result=None,
                traceback=None,
                aux={"tie_indices": {"(0, 1)": [4, 7]}, "scheme": "X"},
            ),
        ),
    )
    save(out, tmp_path / "o.json.gz")
    loaded = load(tmp_path / "o.json.gz")
    assert loaded.entries[0].aux == {
        "tie_indices": {"(0, 1)": [4, 7]},
        "scheme": "X",
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/campaign/test_types.py -x -q > /tmp/c2t4a.log 2>&1`; read the log.
Expected: FAIL (`ImportError: CESShardTask`).

- [ ] **Step 3: Implement**

1. In `pyphi/campaign/__init__.py`: add `aux: dict[str, Any] | None = None`
   as the last field of `CellOutput`; add the two dataclasses exactly as in
   the Interfaces block above (import `ShardSpec` from
   `pyphi.campaign.shards` — module-level import is safe: `shards` does
   not import the package `__init__`); extend `__all__` with
   `"CESShardTask"`, `"SIAShardTask"`.

2. In `pyphi/serialize/schema.py`: add `aux: dict[str, Any] | None = None`
   to `CellOutputSchema`, and add:

```python
class ShardSpecSchema(msgspec.Struct, frozen=True, tag="shard_spec"):
    payload_kind: str
    mechanisms: tuple[tuple[int, ...], ...]
    mechanism: tuple[int, ...] | None
    direction: "str | None"
    purviews: tuple[tuple[int, ...], ...]
    purview: tuple[int, ...] | None
    stride: tuple[int, int] | None
    units: float


class CESShardTaskSchema(msgspec.Struct, frozen=True, tag="campaign_ces_task"):
    task_id: int
    kind: str
    substrate_label: "str | int"
    state: tuple[int, ...]
    subset: tuple[int, ...] | None
    scope: CESScopeSchema
    config_overrides: dict[str, Any]
    formalism: str
    spec: ShardSpecSchema
    ordering: "str | None"


class SIAShardTaskSchema(msgspec.Struct, frozen=True, tag="campaign_sia_task"):
    task_id: int
    kind: str
    substrate_label: "str | int"
    state: tuple[int, ...]
    subset: tuple[int, ...] | None
    config_overrides: dict[str, Any]
    formalism: str
    stride: tuple[int, int]
```

   Add all three to the `Schema` union.

3. In `pyphi/serialize/convert.py` `_register_campaign()`: extend the
   `CellOutput` encoder/decoder with the `aux` field
   (`aux=None if e.aux is None else dict(e.aux)` both ways), and add
   encoders/decoders for `ShardSpec`, `CESShardTask`, `SIAShardTask`
   following the same field-by-field pattern as `CampaignTask` (tuples of
   tuples re-tupled on decode; `scope`/`spec` through
   `to_schema`/`from_schema`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/campaign/ test/serialize/ -q > /tmp/c2t4b.log 2>&1`; read the log.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyphi/campaign/__init__.py pyphi/serialize/schema.py pyphi/serialize/convert.py test/campaign/test_types.py
git commit -m "Add shard task types and per-entry aux bookkeeping"
git log --oneline -1
```

---

### Task 5: `prepare(kind="ces")`

**Files:**
- Modify: `pyphi/campaign/__init__.py`
- Test: `test/campaign/test_prepare_ces.py`

**Interfaces:**
- Consumes: Tasks 1–4; cycle-1 `prepare` internals (`_wire_overrides`,
  directory writing, `_SUBMIT_TEMPLATE`, `CampaignStatus`).
- Produces:

```python
def prepare_ces(
    substrate, *, state, subset=None, scope=None, directory,
    units_per_job, formalism=None, sia=None, resolution_state=None,
    ordering=None, infeasible_threshold=1e9, strict=False,
    container_image="pyphi.sif", request_memory="4GB",
    request_disk="4GB", seed=None,
) -> CampaignStatus
```

  (A separate entry point, not a `kind=` flag on the sweep `prepare` — the
  argument sets differ almost entirely. Both share the directory-writing
  helpers.) Manifest for CES campaigns: `kind` = `"ces"`, plus
  `formalism` (one preset name; `None` → active version), `state`,
  `subset`, `substrate_label` (always `"system"`), serialized-scope file
  reference, `sia_mode` (`"shards"` | `"precomputed"` | `"none"`),
  `ordering`, `tasks` (list of `{"task_id", "kind", "units"}`), `seed`,
  `pyphi_version`, `created`, `mechanism_workloads` (mechanism → units —
  the raw data), `units_per_job`, `partition_scheme`,
  `mechanism_partition_scheme`. Files: `scope.json.gz`; `sia.json.gz`
  (mode "precomputed"); `resolution_state.json.gz` (mode "none" with an
  explicit state); shard tasks in `tasks/` (mixed `CESShardTask` /
  `SIAShardTask` by id).

- [ ] **Step 1: Write the failing tests**

`test/campaign/test_prepare_ces.py`:

```python
import json

import pytest

from pyphi import examples
from pyphi.campaign import prepare_ces
from pyphi.campaign.scope import AxisScope
from pyphi.campaign.scope import CESScope
from pyphi.serialize import load

BASIC_STATE = (1, 0, 0)


def test_prepare_ces_writes_shard_campaign(tmp_path):
    directory = tmp_path / "camp"
    status = prepare_ces(
        examples.basic_substrate(),
        state=BASIC_STATE,
        formalism="IIT_4_0_2026",
        directory=directory,
        units_per_job=50.0,
    )
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["kind"] == "ces"
    assert manifest["sia_mode"] == "shards"
    assert manifest["formalism"] == "IIT_4_0_2026"
    assert (directory / "scope.json.gz").exists()
    assert status.n_tasks == len(manifest["tasks"])
    kinds = {t["kind"] for t in manifest["tasks"]}
    assert kinds == {"ces_shard", "sia_shard"}
    task0 = load(directory / "tasks" / "task-0000.json.gz")
    assert task0.kind in ("ces_shard", "sia_shard")
    assert (directory / "pyphi.sub").exists()


def test_precomputed_sia_skips_sia_shards(tmp_path):
    import pyphi
    from pyphi.conf import presets

    substrate = examples.basic_substrate()
    with pyphi.config.override(
        **presets.by_name["IIT_4_0_2026"], parallel=False, progress_bars=False
    ):
        sia = pyphi.System(substrate, BASIC_STATE).sia()
    directory = tmp_path / "camp"
    prepare_ces(
        substrate,
        state=BASIC_STATE,
        formalism="IIT_4_0_2026",
        directory=directory,
        units_per_job=50.0,
        sia=sia,
    )
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["sia_mode"] == "precomputed"
    assert {t["kind"] for t in manifest["tasks"]} == {"ces_shard"}
    assert (directory / "sia.json.gz").exists()


def test_scope_lands_in_manifest_and_tasks(tmp_path):
    directory = tmp_path / "camp"
    scope = CESScope(mechanisms=AxisScope(containing=(0,)))
    prepare_ces(
        examples.basic_substrate(),
        state=BASIC_STATE,
        formalism="IIT_4_0_2026",
        scope=scope,
        directory=directory,
        units_per_job=1e9,
        sia=None,
        resolution_state=None,
    )
    saved_scope = load(directory / "scope.json.gz")
    assert saved_scope.mechanisms.containing == (0,)
    manifest = json.loads((directory / "manifest.json").read_text())
    mechs = [tuple(m) for m in manifest["mechanism_workloads"]]
    assert all(0 in m for m in mechs)


def test_empty_scope_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="zero mechanisms"):
        prepare_ces(
            examples.basic_substrate(),
            state=BASIC_STATE,
            formalism="IIT_4_0_2026",
            scope=CESScope(mechanisms=AxisScope(explicit=())),
            directory=tmp_path / "camp",
            units_per_job=1.0,
        )
```

(`manifest["mechanism_workloads"]` is a JSON object keyed by
JSON-encodable mechanism keys — store keys as `"0,1"` comma-joined strings
and parse with `tuple(int(x) for x in key.split(","))`; adjust the test's
`mechs` line accordingly:
`mechs = [tuple(int(x) for x in k.split(",")) for k in manifest["mechanism_workloads"]]`.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/campaign/test_prepare_ces.py -x -q > /tmp/c2t5a.log 2>&1`; read the log.
Expected: FAIL (`ImportError: prepare_ces`).

- [ ] **Step 3: Implement `prepare_ces` in `pyphi/campaign/__init__.py`**

Refactor first: extract the directory-scaffolding tail of the sweep
`prepare` (mkdir tree, `remaining.txt`, `run_task.sh`, `pyphi.sub`
writing) into a module-level helper both entry points call:

```python
def _write_campaign_scaffold(
    directory: Path,
    n_tasks: int,
    container_image: str,
    request_memory: str,
    request_disk: str,
) -> None:
    (directory / "outputs").mkdir()
    (directory / "logs").mkdir()
    (directory / "remaining.txt").write_text(
        "".join(f"{task_id}\n" for task_id in range(n_tasks))
    )
    run_task_sh = directory / "run_task.sh"
    run_task_sh.write_text(_RUN_TASK_SH)
    run_task_sh.chmod(
        run_task_sh.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    )
    (directory / "pyphi.sub").write_text(
        _SUBMIT_TEMPLATE.format(
            container_image=container_image,
            request_memory=request_memory,
            request_disk=request_disk,
        )
    )
```

Then:

```python
def prepare_ces(
    substrate: Any,
    *,
    state: Any,
    subset: Any = None,
    scope: Any = None,
    directory: Any,
    units_per_job: float,
    formalism: str | None = None,
    sia: Any = None,
    resolution_state: Any = None,
    ordering: str | None = None,
    infeasible_threshold: float = 1e9,
    strict: bool = False,
    container_image: str = "pyphi.sif",
    request_memory: str = "4GB",
    request_disk: str = "4GB",
    seed: int | None = None,
) -> CampaignStatus:
    """Materialize one system's scoped CES analysis as a campaign.

    Plans shards for the scoped distinction computation (and, unless a
    precomputed ``sia`` or explicit ``resolution_state`` is given, for the
    system irreducibility analysis), descending mechanism → purview-range →
    partition-stride only where ``units_per_job`` requires. Shards are
    independent condor jobs on the standard campaign scaffold; collection
    merges them exactly (tie sets preserved) and assembles the
    cause-effect structure through the standard analysis path.

    Parameters
    ----------
    substrate
        The substrate of the analyzed system.
    state : tuple[int, ...]
        The system state.
    subset : optional
        Node indices (or labels) of the candidate system; ``None`` uses
        the whole substrate.
    scope : CESScope, optional
        The feasibility surface; ``None`` is the unconstrained scope.
    directory
        Target campaign directory; created, and must not already exist.
    units_per_job : float
        Target work units per shard — the ladder's budget.
    formalism : str, optional
        Preset name; ``None`` uses the active formalism version.
    sia : optional
        A precomputed system irreducibility analysis; suppresses SIA
        shards and is used at collection.
    resolution_state : optional
        An explicit congruence-resolution state; suppresses SIA shards.
        The collected structure then carries no Φₛ.
    ordering : {"bottleneck_first", None}, optional
        Reorder each partition-stride shard's slice so likely-reducible
        partitions are evaluated first (sparse substrates short-circuit
        sooner). Never affects results.
    infeasible_threshold, strict, container_image, request_memory,
    request_disk, seed
        As for the sweep campaign entry point.

    Returns
    -------
    CampaignStatus
        The freshly prepared ledger (all tasks pending).
    """
    from pyphi.campaign import shards as _shards
    from pyphi.campaign.scope import CESScope
    from pyphi.campaign.scope import resolve_scope
    from pyphi.system import System

    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(
            f"campaign directory {directory} already exists; "
            "campaign directories are never overwritten"
        )
    if sia is not None and resolution_state is not None:
        raise ValueError("pass either sia or resolution_state, not both")
    formalism_ = formalism if formalism is not None else config.formalism.iit.version
    if formalism_ not in presets.by_name:
        raise ValueError(f"unknown formalism {formalism_!r}")
    scope = scope if scope is not None else CESScope()

    with config.override(**presets.by_name[formalism_], progress_bars=False):
        system = System.from_substrate(substrate, tuple(state), subset)
        resolved = resolve_scope(scope, system.node_labels)
        ces_specs = _shards.plan_ces_shards(system, resolved, units_per_job)
        if not any(s.mechanisms or s.mechanism for s in ces_specs):
            raise ValueError("the scope admits zero mechanisms")
        sia_specs = (
            _shards.plan_sia_shards(system, units_per_job)
            if sia is None and resolution_state is None
            else []
        )
        partition_scheme = config.formalism.iit.system_partition_scheme
        mechanism_partition_scheme = config.formalism.iit.mechanism_partition_scheme
        workloads = mechanism_workloads(
            substrate, subset=system.node_indices, scope=resolved
        )

    for spec in ces_specs + sia_specs:
        if spec.units > infeasible_threshold:
            message = (
                f"shard {spec!r} estimate {spec.units:.3g} exceeds "
                f"infeasible_threshold {infeasible_threshold:.3g}"
            )
            if strict:
                raise ValueError(message)
            warnings.warn(message, PyPhiWarning, stacklevel=2)

    directory.mkdir(parents=True)
    substrates_dir = directory / "substrates"
    substrates_dir.mkdir()
    serialize.save(substrate, substrates_dir / "substrate-system.json.gz")
    serialize.save(resolved, directory / "scope.json.gz")
    if sia is not None:
        serialize.save(sia, directory / "sia.json.gz")
    if resolution_state is not None:
        serialize.save(resolution_state, directory / "resolution_state.json.gz")

    tasks_dir = directory / "tasks"
    tasks_dir.mkdir()
    overrides = _wire_overrides()
    subset_ = tuple(system.node_indices)
    task_rows = []
    task_id = 0
    for spec in ces_specs:
        task = CESShardTask(
            task_id=task_id,
            kind="ces_shard",
            substrate_label="system",
            state=tuple(state),
            subset=subset_,
            scope=resolved,
            config_overrides=overrides,
            formalism=formalism_,
            spec=spec,
            ordering=ordering,
        )
        serialize.save(task, tasks_dir / f"task-{task_id:04d}.json.gz")
        task_rows.append(
            {"task_id": task_id, "kind": "ces_shard", "units": spec.units}
        )
        task_id += 1
    for spec in sia_specs:
        task = SIAShardTask(
            task_id=task_id,
            kind="sia_shard",
            substrate_label="system",
            state=tuple(state),
            subset=subset_,
            config_overrides=overrides,
            formalism=formalism_,
            stride=spec.stride,
        )
        serialize.save(task, tasks_dir / f"task-{task_id:04d}.json.gz")
        task_rows.append(
            {"task_id": task_id, "kind": "sia_shard", "units": spec.units}
        )
        task_id += 1

    sia_mode = (
        "precomputed" if sia is not None
        else "none" if resolution_state is not None
        else "shards"
    )
    manifest = {
        "kind": "ces",
        "pyphi_version": importlib.metadata.version("pyphi"),
        "created": datetime.now(UTC).isoformat(),
        "seed": seed,
        "formalism": formalism_,
        "state": list(state),
        "subset": list(subset_),
        "substrate_label": "system",
        "sia_mode": sia_mode,
        "ordering": ordering,
        "tasks": task_rows,
        "mechanism_workloads": {
            ",".join(map(str, mechanism)): units
            for mechanism, units in workloads.items()
        },
        "units_per_job": units_per_job,
        "infeasible_threshold": infeasible_threshold,
        "partition_scheme": partition_scheme,
        "mechanism_partition_scheme": mechanism_partition_scheme,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    _write_campaign_scaffold(
        directory, len(task_rows), container_image, request_memory, request_disk
    )
    return CampaignStatus(
        directory=str(directory),
        n_tasks=len(task_rows),
        n_cells=len(task_rows),
        done=(),
        failed=(),
        pending=tuple(range(len(task_rows))),
        total_units=float(sum(row["units"] for row in task_rows)),
    )
```

Also: the sweep `prepare` now calls `_write_campaign_scaffold` instead of
its inlined tail (delete the duplicated block); `status()` reads
`manifest["tasks"]` which for CES campaigns is a list of dicts — its
`range(len(manifest["tasks"]))` loop and `n_cells=len(manifest["cells"])`
line must handle both kinds: use
`n_cells=len(manifest.get("cells", manifest["tasks"]))` and
`total_units=float(sum(manifest["weights"])) if "weights" in manifest else
float(sum(row["units"] for row in manifest["tasks"]))`. Extend `__all__`
with `"prepare_ces"`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/campaign/ -q > /tmp/c2t5b.log 2>&1`; read the log.
Expected: PASS (including all cycle-1 tests — the scaffold refactor and
`status()` generalization must not change sweep-campaign behavior).

- [ ] **Step 5: Commit**

```bash
git add pyphi/campaign/__init__.py test/campaign/test_prepare_ces.py
git commit -m "Add prepare_ces: shard-planned campaigns for one system"
git log --oneline -1
```

---

### Task 6: Runner dispatch for shard kinds

**Files:**
- Modify: `pyphi/campaign/runner.py`
- Test: `test/campaign/test_runner_shards.py`

**Interfaces:**
- Consumes: Tasks 3–5; `find_mip` / `distinction` from
  `pyphi.formalism.queries`; `enumerate_partition_stride` /
  `enumerate_system_partition_stride` / `bottleneck_order` (Task 3).
- Produces: `run_task` handles all three task kinds. Shard outputs are
  `CampaignTaskOutput` documents whose entries align 1:1 with the shard's
  items:
  - `"mechanisms"` payload → one entry per mechanism, `result` = the
    distinction (possibly falsy);
  - `"purview_range"` payload → one entry per purview, `result` = the
    purview's RIA (with tie sets);
  - `"partition_stride"` payload (CES) → one entry, `result` = the
    stride-winner RIA; `aux = {"tie_indices": {<repr of pin state>:
    [global enumeration indices aligned with that pin's partition-tie
    set]}, "scheme": <mechanism partition scheme>}`;
  - `sia_shard` → one entry, `result` = the stride-winner SIA;
    `aux = {"tie_indices": [global indices aligned with the SIA tie set],
    "scheme": <system partition scheme>}`.

- [ ] **Step 1: Write the failing tests**

`test/campaign/test_runner_shards.py`:

```python
import json

from pyphi import examples
from pyphi.campaign import prepare_ces
from pyphi.campaign.runner import run_task
from pyphi.serialize import load

BASIC_STATE = (1, 0, 0)


def _run_all(directory):
    for task_file in sorted((directory / "tasks").glob("task-*.json.gz")):
        rc = run_task(
            task_file,
            substrates_dir=directory / "substrates",
            outputs_dir=directory / "outputs",
        )
        assert rc == 0


def test_shard_outputs_align_with_items(tmp_path):
    directory = tmp_path / "camp"
    prepare_ces(
        examples.basic_substrate(),
        state=BASIC_STATE,
        formalism="IIT_4_0_2026",
        directory=directory,
        units_per_job=5.0,
    )
    _run_all(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    saw_stride_aux = False
    for row in manifest["tasks"]:
        task = load(directory / "tasks" / f"task-{row['task_id']:04d}.json.gz")
        out = load(directory / "outputs" / f"task-{row['task_id']:04d}.json.gz")
        if row["kind"] == "sia_shard":
            assert len(out.entries) == 1
            assert out.entries[0].aux is not None
            assert "tie_indices" in out.entries[0].aux
        elif task.spec.payload_kind == "mechanisms":
            assert len(out.entries) == len(task.spec.mechanisms)
        elif task.spec.payload_kind == "purview_range":
            assert len(out.entries) == len(task.spec.purviews)
        elif task.spec.payload_kind == "partition_stride":
            assert len(out.entries) == 1
            aux = out.entries[0].aux
            assert aux is not None and "tie_indices" in aux
            saw_stride_aux = True
    assert saw_stride_aux


def test_bottleneck_ordering_gives_same_results(tmp_path):
    a = tmp_path / "plain"
    b = tmp_path / "ordered"
    for directory, ordering in ((a, None), (b, "bottleneck_first")):
        prepare_ces(
            examples.basic_substrate(),
            state=BASIC_STATE,
            formalism="IIT_4_0_2026",
            directory=directory,
            units_per_job=5.0,
            ordering=ordering,
        )
        _run_all(directory)
    manifest = json.loads((a / "manifest.json").read_text())
    for row in manifest["tasks"]:
        oa = load(a / "outputs" / f"task-{row['task_id']:04d}.json.gz")
        ob = load(b / "outputs" / f"task-{row['task_id']:04d}.json.gz")
        for ea, eb in zip(oa.entries, ob.entries, strict=True):
            if ea.result is not None and hasattr(ea.result, "phi"):
                assert float(ea.result.phi) == float(eb.result.phi)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/campaign/test_runner_shards.py -x -q > /tmp/c2t6a.log 2>&1`; read the log.
Expected: FAIL — `run_task` treats a `CESShardTask` like a sweep task
(`AttributeError: 'CESShardTask' object has no attribute 'cells'`).

- [ ] **Step 3: Implement the dispatch**

In `pyphi/campaign/runner.py`, rename the existing body's per-cell loop
into `_run_sweep_task(task, substrates) -> tuple[list[CellOutput], bool]`
(returning entries and the failed flag), and add:

```python
def _shard_config(task: Any) -> dict[str, Any]:
    return {
        **task.config_overrides,
        **presets.by_name[task.formalism],
        "parallel": False,
        "progress_bars": False,
    }


def _global_tie_indices(ties: Any, slice_parts: list, indices: list[int]) -> list[int]:
    """Map a tie set's partitions back to global enumeration indices."""
    local = {str(p): g for p, g in zip(slice_parts, indices, strict=True)}
    return [local[str(t.partition)] for t in ties]


def _run_ces_shard(task: Any, substrates: dict) -> tuple[list[CellOutput], bool]:
    from pyphi.campaign import shards as _shards
    from pyphi.direction import Direction
    from pyphi.formalism.queries import distinction as _distinction
    from pyphi.formalism.queries import find_mip
    from pyphi.system import System

    entries: list[CellOutput] = []
    failed = False
    spec = task.spec
    with config.override(**_shard_config(task)):
        system = System(
            substrates[task.substrate_label], task.state, node_indices=task.subset
        )
        scheme = config.formalism.iit.mechanism_partition_scheme
        try:
            if spec.payload_kind == "mechanisms":
                for mechanism in spec.mechanisms:
                    cause_purviews = list(
                        task.scope.purviews(Direction.CAUSE).select(
                            system.potential_purviews(Direction.CAUSE, mechanism)
                        )
                    )
                    effect_purviews = list(
                        task.scope.purviews(Direction.EFFECT).select(
                            system.potential_purviews(Direction.EFFECT, mechanism)
                        )
                    )
                    result = _distinction(
                        system,
                        mechanism,
                        cause_purviews=cause_purviews,
                        effect_purviews=effect_purviews,
                    )
                    entries.append(
                        CellOutput(status="ok", result=result, traceback=None)
                    )
            elif spec.payload_kind == "purview_range":
                direction = Direction[spec.direction]
                for purview in spec.purviews:
                    ria = find_mip(system, direction, spec.mechanism, purview)
                    entries.append(
                        CellOutput(status="ok", result=ria, traceback=None)
                    )
            elif spec.payload_kind == "partition_stride":
                direction = Direction[spec.direction]
                i, k = spec.stride
                parts, indices = _shards.enumerate_partition_stride(
                    spec.mechanism, spec.purview, system.node_labels, i, k
                )
                if task.ordering == "bottleneck_first":
                    parts, indices = _shards.bottleneck_order(
                        parts, indices, system.cm, direction
                    )
                ria = find_mip(
                    system, direction, spec.mechanism, spec.purview,
                    partitions=parts,
                )
                tie_indices = {}
                for pin in getattr(ria, "_state_ties", None) or (ria,):
                    pin_ties = getattr(pin, "_partition_ties", None) or (pin,)
                    tie_indices[repr(pin.specified_state.state)] = (
                        _global_tie_indices(pin_ties, parts, indices)
                    )
                entries.append(
                    CellOutput(
                        status="ok",
                        result=ria,
                        traceback=None,
                        aux={"tie_indices": tie_indices, "scheme": scheme},
                    )
                )
            else:
                raise ValueError(f"unknown payload kind {spec.payload_kind!r}")
        except Exception:
            entries.append(
                CellOutput(
                    status="error", result=None, traceback=_traceback.format_exc()
                )
            )
            failed = True
    return entries, failed


def _run_sia_shard(task: Any, substrates: dict) -> tuple[list[CellOutput], bool]:
    from pyphi.campaign import shards as _shards
    from pyphi.system import System

    with config.override(**_shard_config(task)):
        system = System(
            substrates[task.substrate_label], task.state, node_indices=task.subset
        )
        scheme = config.formalism.iit.system_partition_scheme
        i, k = task.stride
        parts, indices = _shards.enumerate_system_partition_stride(
            system, scheme, i, k
        )
        try:
            sia = system.sia(partitions=parts)
            ties = getattr(sia, "ties", None) or (sia,)
            aux = {
                "tie_indices": _global_tie_indices(ties, parts, indices),
                "scheme": scheme,
            }
            return (
                [CellOutput(status="ok", result=sia, traceback=None, aux=aux)],
                False,
            )
        except Exception:
            return (
                [
                    CellOutput(
                        status="error",
                        result=None,
                        traceback=_traceback.format_exc(),
                    )
                ],
                True,
            )
```

and dispatch in `run_task` (replacing the direct per-cell loop):

```python
    kind = getattr(task, "kind", "sweep_cells")
    if kind == "ces_shard":
        entries, failed = _run_ces_shard(task, substrates)
    elif kind == "sia_shard":
        entries, failed = _run_sia_shard(task, substrates)
    else:
        entries, failed = _run_sweep_task(task, substrates)
```

Note `_global_tie_indices` for SIA tie sets uses `t.partition` — verify
the SIA object's partition attribute name matches (`sia.partition` was
probed and works); if the tie elements are SIA objects their partitions
are `t.partition` too.

`_load_substrates` reads labels from `task.cells` for sweep tasks; give
shard tasks the same duck by generalizing:

```python
def _task_labels(task: Any) -> set:
    if hasattr(task, "cells"):
        return {cell[0] for cell in task.cells}
    return {task.substrate_label}
```

and use it in `_load_substrates`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/campaign/ -q > /tmp/c2t6b.log 2>&1`; read the log.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyphi/campaign/runner.py test/campaign/test_runner_shards.py
git commit -m "Dispatch campaign runner on task kind and execute shards"
git log --oneline -1
```

---

### Task 7: Tie-preserving merges

**Files:**
- Create: `pyphi/campaign/merge.py`
- Test: `test/campaign/test_merge.py`

**Interfaces:**
- Consumes: `resolve_ties` entry points; `MaximallyIrreducibleCause` /
  `MaximallyIrreducibleEffect` (`pyphi/models/mice.py`); `Concept`
  (import it from wherever `pyphi/formalism/queries.py` imports it —
  check that file's imports and use the same path).
- Produces (used by Task 8):

```python
def merge_stride_rias(entries: list[tuple[Any, dict]]) -> Any
    # entries: (RIA, aux) per stride of ONE (mechanism, direction, purview);
    # exact merged RIA (per-pin union of partition ties sorted by global
    # index, resolve partitions, then resolve states); margins per the
    # verified formula
def merge_purview_rias(direction: Direction, rias: list,
                       canonical_purviews: list) -> Any
    # per-purview RIAs -> merged MICE (mirrors find_mice's tail)
def build_distinction(mechanism, mic, mie) -> Any
def merge_sia_strides(entries: list[tuple[Any, dict]]) -> Any
```

- [ ] **Step 1: Write the failing tests**

`test/campaign/test_merge.py` — merges are tested directly against full
computations (the same checks the pre-design probes ran):

```python
from pyphi import examples
from pyphi.campaign.merge import build_distinction
from pyphi.campaign.merge import merge_purview_rias
from pyphi.campaign.merge import merge_sia_strides
from pyphi.campaign.merge import merge_stride_rias
from pyphi.campaign.shards import enumerate_partition_stride
from pyphi.campaign.shards import enumerate_system_partition_stride
from pyphi.conf import config
from pyphi.conf import presets
from pyphi.direction import Direction
from pyphi.formalism.queries import find_mice
from pyphi.formalism.queries import find_mip
from pyphi.system import System

PIN = dict(parallel=False, progress_bars=False, shortcircuit_sia=False)


def _system():
    return System(examples.basic_substrate(), (1, 0, 0))


def test_stride_merge_equals_full_find_mip():
    with config.override(**presets.by_name["IIT_4_0_2026"], **PIN):
        system = _system()
        mechanism, purview = (0, 1), (0, 2)
        direction = Direction.EFFECT
        full = find_mip(system, direction, mechanism, purview)
        k = 3
        entries = []
        for i in range(k):
            parts, indices = enumerate_partition_stride(
                mechanism, purview, system.node_labels, i, k
            )
            ria = find_mip(
                system, direction, mechanism, purview, partitions=parts
            )
            local = {str(p): g for p, g in zip(parts, indices, strict=True)}
            tie_indices = {}
            for pin in ria._state_ties or (ria,):
                pin_ties = pin._partition_ties or (pin,)
                tie_indices[repr(pin.specified_state.state)] = [
                    local[str(t.partition)] for t in pin_ties
                ]
            entries.append((ria, {"tie_indices": tie_indices}))
        merged = merge_stride_rias(entries)
    assert float(merged.phi) == float(full.phi)
    assert str(merged.partition) == str(full.partition)
    assert repr(merged.specified_state.state) == repr(full.specified_state.state)


def test_purview_merge_equals_full_find_mice():
    with config.override(**presets.by_name["IIT_4_0_2026"], **PIN):
        system = _system()
        mechanism, direction = (0, 1), Direction.EFFECT
        purviews = system.potential_purviews(direction, mechanism)
        full = find_mice(system, direction, mechanism)
        rias = [
            find_mip(system, direction, mechanism, p) for p in purviews
        ]
        merged = merge_purview_rias(direction, rias, list(purviews))
    assert float(merged.phi) == float(full.phi)
    assert merged.purview == full.purview
    assert merged.purview_margin == full.purview_margin


def test_distinction_assembly_matches_direct():
    from pyphi.formalism.queries import distinction as _distinction

    with config.override(**presets.by_name["IIT_4_0_2026"], **PIN):
        system = _system()
        mechanism = (0, 1)
        direct = _distinction(system, mechanism)
        mic = find_mice(system, Direction.CAUSE, mechanism)
        mie = find_mice(system, Direction.EFFECT, mechanism)
        built = build_distinction(mechanism, mic, mie)
    assert float(built.phi) == float(direct.phi)
    assert built.mechanism == direct.mechanism


def test_sia_stride_merge_equals_full():
    with config.override(**presets.by_name["IIT_4_0_2026"], **PIN):
        system = _system()
        full = system.sia()
        scheme = config.formalism.iit.system_partition_scheme
        entries = []
        k = 2
        for i in range(k):
            parts, indices = enumerate_system_partition_stride(
                system, scheme, i, k
            )
            sia = system.sia(partitions=parts)
            local = {str(p): g for p, g in zip(parts, indices, strict=True)}
            ties = getattr(sia, "ties", None) or (sia,)
            entries.append(
                (sia, {"tie_indices": [local[str(t.partition)] for t in ties]})
            )
        merged = merge_sia_strides(entries)
    assert float(merged.phi) == float(full.phi)
    assert str(merged.partition) == str(full.partition)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/campaign/test_merge.py -x -q > /tmp/c2t7a.log 2>&1`; read the log.
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `pyphi/campaign/merge.py`**

```python
"""Tie-preserving merges of shard outputs.

Every merge re-runs the same :mod:`pyphi.resolve_ties` machinery the
single-machine code path uses, applied to the union of shard tie sets.
Exactness: the global extremum is attained inside some shard, so any
candidate within tolerance of the global extremum is within tolerance of
its own shard's extremum and therefore present in that shard's tie set —
the union loses nothing. Candidates are restored to global enumeration
order before resolution, so the selected representative is identical to a
full sweep's.
"""

from __future__ import annotations

from typing import Any

from pyphi import resolve_ties
from pyphi.direction import Direction
from pyphi.models.mice import MaximallyIrreducibleCause
from pyphi.models.mice import MaximallyIrreducibleEffect

__all__ = [
    "build_distinction",
    "merge_purview_rias",
    "merge_sia_strides",
    "merge_stride_rias",
]


def _pin_key(ria: Any) -> str:
    spec = ria.specified_state
    return repr(spec.state) if spec is not None else "none"


def merge_stride_rias(entries: list[tuple[Any, dict]]) -> Any:
    """Merge the stride winners of one (mechanism, direction, purview).

    ``entries`` pairs each stride's winning RIA with its aux record, whose
    ``tie_indices`` map each specified-state pin (by ``repr`` of its
    state) to the global enumeration indices of that pin's partition-tie
    members, in tie-set order.
    """
    per_pin: dict[str, list[tuple[int, Any]]] = {}
    shard_best: dict[str, list[Any]] = {}
    for ria, aux in entries:
        for pin in getattr(ria, "_state_ties", None) or (ria,):
            key = _pin_key(pin)
            ties = getattr(pin, "_partition_ties", None) or (pin,)
            indices = aux["tie_indices"][key]
            per_pin.setdefault(key, []).extend(
                zip(indices, ties, strict=True)
            )
            shard_best.setdefault(key, []).append(pin)
    pin_winners = []
    for key, indexed in per_pin.items():
        indexed.sort(key=lambda pair: pair[0])
        candidates = [c for _, c in indexed]
        ties = tuple(resolve_ties.partitions(candidates))
        winner = ties[0]
        for tie in ties:
            tie.set_partition_ties(ties)
        winner.partition_margin = _merged_partition_margin(
            winner, shard_best[key]
        )
        pin_winners.append(winner)
    state_ties = tuple(resolve_ties.states(pin_winners))
    # Mirror the state-tie attachment the single-machine path performs
    # (see the lines following ``resolve_ties.states`` in
    # ``pyphi/formalism/iit4/formalism.py`` around line 258 — read them and
    # replicate the attachment exactly).
    for tie in state_ties:
        tie.set_state_ties(state_ties)
    return state_ties[0]


def _merged_partition_margin(winner: Any, shard_winners: list[Any]) -> Any:
    """Global margin from per-shard winners, or None when underivable.

    The global runner-up normalized φ is the smaller of: the best
    normalized φ among non-winning shards, and the winning shard's own
    runner-up (its winner's normalized φ plus its margin). Underivable
    (None) when any shard's margin is None — a short-circuited or
    single-candidate slice.
    """
    if any(getattr(s, "partition_margin", None) is None for s in shard_winners):
        return None
    if winner.normalized_phi is None:
        return None
    winner_nphi = float(winner.normalized_phi)
    rivals = []
    for shard in shard_winners:
        nphi = float(shard.normalized_phi)
        if nphi == winner_nphi:
            # The winning value's shard: its runner-up is the rival.
            rivals.append(nphi + float(shard.partition_margin))
        else:
            rivals.append(nphi)
    # numerics: exact — reported margin, not a selection.
    return max(0.0, min(rivals) - winner_nphi) if rivals else None


def merge_purview_rias(
    direction: Direction, rias: list, canonical_purviews: list
) -> Any:
    """Merge per-purview RIAs into the MICE (mirrors ``find_mice``'s tail)."""
    mice_cls = (
        MaximallyIrreducibleCause
        if direction == Direction.CAUSE
        else MaximallyIrreducibleEffect
    )
    order = {tuple(p): i for i, p in enumerate(canonical_purviews)}
    rias = sorted(rias, key=lambda ria: order[tuple(ria.purview)])
    all_mice = [mice_cls(ria) for ria in rias]
    ties = tuple(resolve_ties.purviews(all_mice))
    for tie in ties:
        tie.set_purview_ties(ties)
    winner = ties[0]
    others = [m for m in all_mice if m is not winner]
    if others:
        # numerics: exact — reported margin, not a selection.
        best_rival = max(float(m.phi) for m in others)
        winner.purview_margin = max(0.0, float(winner.phi) - best_rival)
    return winner


def build_distinction(mechanism: Any, mic: Any, mie: Any) -> Any:
    """Assemble a distinction from a merged MIC and MIE."""
    from pyphi.formalism.queries import Concept  # same class queries.py builds

    return Concept(mechanism=tuple(mechanism), cause=mic, effect=mie)


def merge_sia_strides(entries: list[tuple[Any, dict]]) -> Any:
    """Merge SIA stride winners (union of tie sets, global order restored)."""
    indexed: list[tuple[int, Any]] = []
    for sia, aux in entries:
        ties = getattr(sia, "ties", None) or (sia,)
        indexed.extend(zip(aux["tie_indices"], ties, strict=True))
    indexed.sort(key=lambda pair: pair[0])
    candidates = [c for _, c in indexed]
    ties = tuple(resolve_ties.sias(candidates))
    winner = ties[0]
    set_ties = getattr(winner, "set_ties", None)
    if set_ties is not None:
        for tie in ties:
            tie.set_ties(ties)
    return winner
```

Before running the tests: read
`pyphi/formalism/iit4/formalism.py:258-275` (the lines after
`resolve_ties.states(mips)`) and make `merge_stride_rias` perform exactly
the same tie attachment (replace the `tie.set_state_ties(ties)` line if
the real method name differs). Likewise check the SIA tie-attachment
method name (`set_ties` vs direct assignment) against
`pyphi/serialize/convert.py:641` and the SIA model class, and fix the
`merge_sia_strides` tail to match.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/campaign/test_merge.py -q > /tmp/c2t7b.log 2>&1`; read the log.
Expected: PASS — these are exact-equality checks against full
computations.

- [ ] **Step 5: Commit**

```bash
git add pyphi/campaign/merge.py test/campaign/test_merge.py
git commit -m "Add tie-preserving shard merges"
git log --oneline -1
```

---

### Task 8: CES `collect()`, scope report, three SIA modes — the headline

**Files:**
- Modify: `pyphi/campaign/__init__.py`
- Test: `test/campaign/test_collect_ces.py`

**Interfaces:**
- Consumes: everything above; `UnresolvedDistinctions`
  (`pyphi/models/distinctions.py`); `system_intrinsic_information` and
  `NullSystemIrreducibilityAnalysis` (`pyphi/formalism/iit4/__init__.py`);
  `sum_phi_relations_measured_bound` / `big_phi_measured_bound`
  (`pyphi/formalism/iit4/bounds.py`).
- Produces:

```python
@dataclass(frozen=True)
class ScopeReport(Displayable, ToPandasMixin):
    mechanisms_computed: int
    mechanisms_admitted: int          # scope-admitted count (== computed + missing)
    mechanisms_possible: int          # 2^n - 1
    missing_groups: tuple[str, ...]   # unreconstructable groups (partial collect)
    sum_phi_r_lower: float            # Σφ_r of the computed relations (exact lower bound)
    sum_phi_r_upper: float | None     # measured certificate
    big_phi_upper: float | None       # measured certificate
    sia_mode: str

def collect(directory, partial=False, sia=None, resolution_state=None)
    # kind-dispatched: SweepResult for sweep campaigns (unchanged),
    # CauseEffectStructure for CES campaigns
def scope_report(directory) -> ScopeReport   # reads scope_report.json
```

`collect()` on a CES campaign also writes `scope_report.json` into the
campaign directory and stamps a compact JSON summary into the result's
provenance `note`.

- [ ] **Step 1: Write the failing tests**

`test/campaign/test_collect_ces.py`:

```python
import json

import pytest

from pyphi import examples
from pyphi.campaign import collect
from pyphi.campaign import prepare_ces
from pyphi.campaign import scope_report
from pyphi.campaign.runner import run_task
from pyphi.campaign.scope import AxisScope
from pyphi.campaign.scope import CESScope
from pyphi.conf import config
from pyphi.conf import presets
from pyphi.system import System
from pyphi.warnings import PyPhiWarning

BASIC_STATE = (1, 0, 0)
PIN = dict(parallel=False, progress_bars=False)


def _run_all(directory):
    for task_file in sorted((directory / "tasks").glob("task-*.json.gz")):
        assert (
            run_task(
                task_file,
                substrates_dir=directory / "substrates",
                outputs_dir=directory / "outputs",
            )
            == 0
        )


def _campaign(tmp_path, formalism, **kwargs):
    directory = tmp_path / "camp"
    prepare_ces(
        examples.basic_substrate(),
        state=BASIC_STATE,
        formalism=formalism,
        directory=directory,
        units_per_job=5.0,  # tiny: forces all three ladder rungs
        **kwargs,
    )
    _run_all(directory)
    return directory


@pytest.mark.parametrize("formalism", ["IIT_4_0_2026", "IIT_4_0_2023"])
def test_sharded_equals_unsharded(tmp_path, formalism):
    directory = _campaign(tmp_path, formalism)
    result = collect(directory)
    with config.override(**presets.by_name[formalism], **PIN):
        reference = System(examples.basic_substrate(), BASIC_STATE).ces()
    assert float(result.sia.phi) == float(reference.sia.phi)
    assert len(result.distinctions) == len(reference.distinctions)
    got = sorted(
        (d.mechanism, d.cause.purview, d.effect.purview, float(d.phi))
        for d in result.distinctions
    )
    want = sorted(
        (d.mechanism, d.cause.purview, d.effect.purview, float(d.phi))
        for d in reference.distinctions
    )
    assert got == want
    assert float(result.relations.sum_phi()) == float(
        reference.relations.sum_phi()
    )


def test_scope_report_written_and_certified(tmp_path):
    scope = CESScope(mechanisms=AxisScope(containing=(0,)))
    directory = _campaign(tmp_path, "IIT_4_0_2026", scope=scope)
    result = collect(directory)
    report = scope_report(directory)
    assert report.mechanisms_possible == 7
    assert report.mechanisms_admitted == 4
    assert report.sum_phi_r_lower == float(result.relations.sum_phi())
    assert report.sum_phi_r_upper is None or (
        report.sum_phi_r_upper >= report.sum_phi_r_lower
    )
    assert (directory / "scope_report.json").exists()


def test_precomputed_sia_mode(tmp_path):
    import pyphi

    with config.override(**presets.by_name["IIT_4_0_2026"], **PIN):
        sia = pyphi.System(examples.basic_substrate(), BASIC_STATE).sia()
    directory = _campaign(tmp_path, "IIT_4_0_2026", sia=sia)
    result = collect(directory)
    assert float(result.sia.phi) == float(sia.phi)


def test_no_sia_mode_carries_no_phi_s(tmp_path):
    from pyphi.formalism.iit4 import NullSystemIrreducibilityAnalysis

    directory = tmp_path / "camp"
    with config.override(**presets.by_name["IIT_4_0_2026"], **PIN):
        system = System(examples.basic_substrate(), BASIC_STATE)
        state = _resolution_state(system)
    prepare_ces(
        examples.basic_substrate(),
        state=BASIC_STATE,
        formalism="IIT_4_0_2026",
        directory=directory,
        units_per_job=5.0,
        resolution_state=state,
    )
    _run_all(directory)
    result = collect(directory)
    assert isinstance(result.sia, NullSystemIrreducibilityAnalysis)
    assert len(result.distinctions) >= 1


def test_version_guard_refuses_mismatched_outputs(tmp_path):
    directory = _campaign(tmp_path, "IIT_4_0_2026")
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["pyphi_version"] = "0.0.0"
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="prepared under"):
        collect(directory)


def test_partial_collect_reports_missing_groups(tmp_path):
    directory = _campaign(tmp_path, "IIT_4_0_2026")
    manifest = json.loads((directory / "manifest.json").read_text())
    victim = next(
        row["task_id"] for row in manifest["tasks"] if row["kind"] == "ces_shard"
    )
    (directory / "outputs" / f"task-{victim:04d}.json.gz").unlink()
    with pytest.raises(RuntimeError, match="incomplete"):
        collect(directory)
    with pytest.warns(PyPhiWarning):
        partial = collect(directory, partial=True)
    report = scope_report(directory)
    assert report.missing_groups
    assert len(partial.distinctions) >= 0
```

Define the test helper `_resolution_state(system)` at the top of the test
file, computing the state exactly the way `ces()`'s fallback does
(`pyphi/formalism/iit4/__init__.py` around line 1424):

```python
def _resolution_state(system):
    from pyphi.formalism.iit4 import system_intrinsic_information
    from pyphi.measures.distribution import resolve_mechanism_measure
    from pyphi.conf import config as _config

    return system_intrinsic_information(
        system,
        specification_measure=resolve_mechanism_measure(
            _config.formalism.iit.specification_measure
        ),
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/campaign/test_collect_ces.py -x -q > /tmp/c2t8a.log 2>&1`; read the log.
Expected: FAIL (`ImportError: scope_report`, or collect() raising on the
unknown manifest kind).

- [ ] **Step 3: Implement CES collection**

In `pyphi/campaign/__init__.py`:

1. `ScopeReport` dataclass as in the Interfaces block, with
   `_pandas_record` (flat dict of all fields, `missing_groups` as its
   length) and `_describe` (rows for each field; follow `CampaignStatus`'s
   pattern), plus JSON (de)serialization helpers
   (`dataclasses.asdict` → `scope_report.json`; `scope_report(directory)`
   reads the file and rebuilds the dataclass, raising `FileNotFoundError`
   with a "collect the campaign first" message when absent).

2. Rename the existing sweep-collect body to `_collect_sweep(directory,
   manifest, partial)` and dispatch in `collect`:

```python
def collect(
    directory: Any,
    partial: bool = False,
    sia: Any = None,
    resolution_state: Any = None,
) -> Any:
    """Reassemble a campaign's outputs into its result.

    Sweep campaigns return the exact local-sweep :class:`SweepResult`;
    CES campaigns return the assembled
    :class:`~pyphi.models.subsystem.CauseEffectStructure` (merging shard
    tie sets exactly, resolving congruence, and computing relations
    through the standard analysis path), writing a scope report alongside.
    """
    directory = Path(directory)
    manifest = _load_manifest(directory)
    if manifest["kind"] == "sweep_cells":
        if sia is not None or resolution_state is not None:
            raise ValueError("sia/resolution_state apply only to CES campaigns")
        return _collect_sweep(directory, manifest, partial)
    return _collect_ces(directory, manifest, partial, sia, resolution_state)
```

3. `_collect_ces`:

```python
def _collect_ces(
    directory: Path,
    manifest: dict,
    partial: bool,
    sia_override: Any,
    resolution_state_override: Any,
) -> Any:
    from pyphi.campaign import merge as _merge
    from pyphi.direction import Direction
    from pyphi.models.distinctions import UnresolvedDistinctions
    from pyphi.system import System

    st = status(directory)
    incomplete = set(st.failed) | set(st.pending)
    if incomplete:
        summary = (
            f"{len(incomplete)} of {st.n_tasks} tasks incomplete "
            f"(failed: {sorted(st.failed)}, pending: {sorted(st.pending)}); "
            "resubmit with condor_submit pyphi.sub"
        )
        if not partial:
            raise RuntimeError(summary)
        warnings.warn(summary, PyPhiWarning, stacklevel=3)

    formalism_ = manifest["formalism"]
    with config.override(
        **presets.by_name[formalism_], parallel=False, progress_bars=False
    ):
        substrate = serialize.load(
            directory / "substrates" / "substrate-system.json.gz"
        )
        system = System(
            substrate,
            tuple(manifest["state"]),
            node_indices=tuple(manifest["subset"]),
        )
        scope = serialize.load(directory / "scope.json.gz")

        # Group loaded outputs by reconstruction target.
        whole_distinctions: dict[tuple, Any] = {}
        purview_rias: dict[tuple, dict[tuple, Any]] = {}
        stride_entries: dict[tuple, list[tuple[Any, dict]]] = {}
        sia_entries: list[tuple[Any, dict]] = []
        missing_groups: set[str] = set()

        expected_schemes = {
            "sia_shard": manifest["partition_scheme"],
            "ces_shard": manifest["mechanism_partition_scheme"],
        }
        for row in manifest["tasks"]:
            task_id = row["task_id"]
            task_path = directory / "tasks" / f"task-{task_id:04d}.json.gz"
            out_path = directory / "outputs" / f"task-{task_id:04d}.json.gz"
            task = serialize.load(task_path)
            if task_id in incomplete:
                missing_groups.add(_group_name(task))
                continue
            output = serialize.load(out_path)
            # Stride semantics depend on the enumeration order, which is a
            # property of the PyPhi version and partition scheme; refuse to
            # merge outputs produced under a different one.
            if output.pyphi_version != manifest["pyphi_version"]:
                raise RuntimeError(
                    f"task {task_id} was run under pyphi "
                    f"{output.pyphi_version} but the campaign was prepared "
                    f"under {manifest['pyphi_version']}; re-run the task"
                )
            for entry in output.entries:
                if entry.aux is not None and "scheme" in entry.aux:
                    expected = expected_schemes[row["kind"]]
                    if entry.aux["scheme"] != expected:
                        raise RuntimeError(
                            f"task {task_id} ran under partition scheme "
                            f"{entry.aux['scheme']!r} but the manifest "
                            f"records {expected!r}; re-run the task"
                        )
            if row["kind"] == "sia_shard":
                entry = output.entries[0]
                sia_entries.append((entry.result, entry.aux))
                continue
            spec = task.spec
            if spec.payload_kind == "mechanisms":
                for mechanism, entry in zip(
                    spec.mechanisms, output.entries, strict=True
                ):
                    whole_distinctions[tuple(mechanism)] = entry.result
            elif spec.payload_kind == "purview_range":
                bucket = purview_rias.setdefault(
                    (tuple(spec.mechanism), spec.direction), {}
                )
                for purview, entry in zip(
                    spec.purviews, output.entries, strict=True
                ):
                    bucket[tuple(purview)] = entry.result
            elif spec.payload_kind == "partition_stride":
                stride_entries.setdefault(
                    (tuple(spec.mechanism), spec.direction, tuple(spec.purview)),
                    [],
                ).append((output.entries[0].result, output.entries[0].aux))

        # A group is unreconstructable if any of its shards is missing;
        # drop groups named in missing_groups.
        def _group_ok(name: str) -> bool:
            return name not in missing_groups

        # Bottom-up: strides -> purview RIAs.
        for (mechanism, direction, purview), entries in stride_entries.items():
            if not _group_ok(f"stride:{mechanism}:{direction}:{purview}"):
                continue
            merged = _merge.merge_stride_rias(entries)
            purview_rias.setdefault((mechanism, direction), {})[purview] = merged

        # Purview RIAs -> MICE -> distinctions for split mechanisms.
        split_mechanisms: dict[tuple, dict[str, Any]] = {}
        for (mechanism, direction), by_purview in purview_rias.items():
            if not _group_ok(f"range:{mechanism}:{direction}"):
                continue
            dir_ = Direction[direction]
            canonical = list(
                scope.purviews(dir_).select(
                    system.potential_purviews(dir_, mechanism)
                )
            )
            if set(map(tuple, canonical)) - set(by_purview):
                missing_groups.add(f"range:{mechanism}:{direction}")
                continue
            mice = _merge.merge_purview_rias(
                dir_, [by_purview[tuple(p)] for p in canonical], canonical
            )
            split_mechanisms.setdefault(mechanism, {})[direction] = mice
        for mechanism, mice_by_dir in split_mechanisms.items():
            if "CAUSE" in mice_by_dir and "EFFECT" in mice_by_dir:
                whole_distinctions[mechanism] = _merge.build_distinction(
                    mechanism, mice_by_dir["CAUSE"], mice_by_dir["EFFECT"]
                )
            else:
                missing_groups.add(f"mechanism:{mechanism}")

        distinctions = UnresolvedDistinctions(
            tuple(d for d in whole_distinctions.values() if d)
        )

        # SIA per mode.
        sia_mode = manifest["sia_mode"]
        sia = sia_override
        if sia is None and (directory / "sia.json.gz").exists():
            sia = serialize.load(directory / "sia.json.gz")
        resolution_state = resolution_state_override
        if resolution_state is None and (
            directory / "resolution_state.json.gz"
        ).exists():
            resolution_state = serialize.load(
                directory / "resolution_state.json.gz"
            )
        if sia is None and sia_mode == "shards":
            if sia_entries and len(sia_entries) == sum(
                1 for row in manifest["tasks"] if row["kind"] == "sia_shard"
            ):
                sia = _merge.merge_sia_strides(sia_entries)
            else:
                missing_groups.add("sia")

        if sia is not None:
            result = system.ces(sia=sia, distinctions=distinctions)
        else:
            result = _assemble_without_sia(
                system, distinctions, resolution_state
            )

        report = _build_scope_report(
            manifest, scope, system, result, missing_groups, sia_mode
        )
    (directory / "scope_report.json").write_text(
        json.dumps(dataclasses.asdict(report), indent=2)
    )
    with_provenance = getattr(result, "with_provenance", None)
    if with_provenance is not None:
        note = json.dumps(
            {"campaign": str(directory), "scope_report": dataclasses.asdict(report)}
        )
        with_provenance(note=note, seed=manifest["seed"])
    return result
```

with the two helpers:

```python
def _group_name(task: Any) -> str:
    if getattr(task, "kind", None) == "sia_shard":
        return "sia"
    spec = task.spec
    if spec.payload_kind == "mechanisms":
        return f"mechanisms:{spec.mechanisms}"
    if spec.payload_kind == "purview_range":
        return f"range:{tuple(spec.mechanism)}:{spec.direction}"
    return (
        f"stride:{tuple(spec.mechanism)}:{spec.direction}:{tuple(spec.purview)}"
    )


def _assemble_without_sia(system: Any, distinctions: Any, resolution_state: Any):
    """Mode-3 assembly: resolve congruence without a system Φₛ."""
    from pyphi.formalism.iit4 import NullSystemIrreducibilityAnalysis
    from pyphi.formalism.iit4 import system_intrinsic_information
    from pyphi.measures.distribution import resolve_mechanism_measure
    from pyphi.relations import relations as compute_relations
    from pyphi.models.subsystem import CauseEffectStructure

    if resolution_state is None:
        resolution_state = system_intrinsic_information(
            system,
            specification_measure=resolve_mechanism_measure(
                config.formalism.iit.specification_measure
            ),
        )
    resolved = distinctions.resolve_congruence(resolution_state)
    return CauseEffectStructure(
        sia=NullSystemIrreducibilityAnalysis(
            system_state=resolution_state,
            node_indices=system.node_indices,
            node_labels=system.node_labels,
        ),
        distinctions=resolved,
        relations=compute_relations(resolved),
    )


def _build_scope_report(
    manifest, scope, system, result, missing_groups, sia_mode
) -> "ScopeReport":
    from pyphi.formalism.iit4 import bounds

    n = len(system.node_indices)
    admitted = len(manifest["mechanism_workloads"])
    computed = len(result.distinctions)
    resolved = result.distinctions
    try:
        upper_r = float(bounds.sum_phi_relations_measured_bound(resolved).value)
        upper_phi = float(bounds.big_phi_measured_bound(resolved).value)
    except Exception:
        upper_r = None
        upper_phi = None
    return ScopeReport(
        mechanisms_computed=computed,
        mechanisms_admitted=admitted,
        mechanisms_possible=2**n - 1,
        missing_groups=tuple(sorted(missing_groups)),
        sum_phi_r_lower=float(result.relations.sum_phi()),
        sum_phi_r_upper=upper_r,
        big_phi_upper=upper_phi,
        sia_mode=sia_mode,
    )
```

Check the import paths before running: `CauseEffectStructure` (grep
`class CauseEffectStructure` — likely `pyphi/models/subsystem.py`),
`compute_relations` (how `iit4/__init__.py` imports it), the `UpperBound`
value attribute name (read the `UpperBound` class in
`pyphi/formalism/iit4/bounds.py` — if the attribute is `.bound` or the
object is float-like, adapt the two `float(...)` lines), and
`NullSystemIrreducibilityAnalysis`'s constructor (mirror `_null_sia` in
`iit4.sia`). Replace the broad `except Exception` with the specific
error the bounds raise on empty distinction sets (read their guard), or a
`if len(resolved) == 0` check — do not ship a silent catch-all. Add
`import dataclasses` to the module imports. Extend `__all__` with
`"ScopeReport"`, `"scope_report"`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/campaign/ -q > /tmp/c2t8b.log 2>&1`; read the log.
Expected: PASS — `test_sharded_equals_unsharded` under both presets is
the headline; a mismatch there means a merge or grouping bug, not a test
bug.

- [ ] **Step 5: Commit**

```bash
git add pyphi/campaign/__init__.py test/campaign/test_collect_ces.py
git commit -m "Assemble CES campaigns: exact merges, three SIA modes, scope report"
git log --oneline -1
```

---

### Task 9: MCP, docs, changelog, ROADMAP, full-suite gate

**Files:**
- Modify: `pyphi/mcp/server.py`, `pyphi/mcp/content/campaigns.md`,
  `docs/howto/campaigns.md`, `ROADMAP.md`
- Create: `changelog.d/ces-sharding.feature.md`
- Test: `test/mcp/test_server.py` (append)

- [ ] **Step 1: MCP — scope on `estimate_cost`, CES campaign tools**

In `pyphi/mcp/server.py`:

1. Add a scope-building helper and thread it through `estimate_cost`:

```python
def _scope_from_json(scope: dict[str, Any] | None) -> Any:
    if scope is None:
        return None
    from pyphi.campaign.scope import AxisScope
    from pyphi.campaign.scope import CESScope

    def axis(d: dict[str, Any] | None) -> AxisScope:
        if not d:
            return AxisScope()
        return AxisScope(
            explicit=None
            if d.get("explicit") is None
            else tuple(tuple(e) for e in d["explicit"]),
            min_order=d.get("min_order"),
            max_order=d.get("max_order"),
            containing=None
            if d.get("containing") is None
            else tuple(d["containing"]),
            within=None if d.get("within") is None else tuple(d["within"]),
        )

    return CESScope(
        mechanisms=axis(scope.get("mechanisms")),
        cause_purviews=axis(scope.get("cause_purviews")),
        effect_purviews=axis(scope.get("effect_purviews")),
    )
```

   `estimate_cost` gains `scope: dict[str, Any] | None = None`, passes
   `scope=_scope_from_json(scope)` into `estimate_analysis`, and its
   docstring documents the shape
   (`{"mechanisms": {"max_order": 2, "containing": [0]}, ...}`; units may
   be labels or indices — resolve with
   `resolve_scope(..., substrate.node_labels)` before estimating).

2. New tool `prepare_ces_campaign(handle, state, directory,
   units_per_job, subset=None, scope=None, formalism=None, sia_ref=None,
   ordering=None, seed=None)` → wraps `campaign.prepare_ces`
   (`sia_ref` resolves a precomputed SIA from the result registry via
   `_get_result`; write a `_get_result` accessor mirroring
   `_get_substrate` if only the raw dict exists). Returns
   `{"card", "status"}` like `prepare_campaign`.

3. `collect_campaign` already dispatches through `campaign.collect`;
   extend its return for CES campaigns: when the result is not a
   `SweepResult`, return
   `{"result_ref": ref, "type": type(result).__name__,
   "summary": _result_summary(result),
   "scope_report": dataclasses.asdict(campaign.scope_report(directory))}`.

4. Append to `test/mcp/test_server.py`'s `TestCampaignTools`:

```python
    def test_ces_campaign_roundtrip(self, tmp_path):
        handle = srv.load_example("basic")["handle"]
        directory = tmp_path / "ces-camp"
        prepared = srv.prepare_ces_campaign(
            handle=handle,
            state=[1, 0, 0],
            formalism="IIT_4_0_2026",
            directory=str(directory),
            units_per_job=50.0,
            scope={"mechanisms": {"max_order": 2}},
        )
        assert prepared["status"]["n_tasks"] >= 1

        from pyphi.campaign.runner import run_task

        for task_file in sorted((directory / "tasks").glob("task-*.json.gz")):
            assert (
                run_task(
                    task_file,
                    substrates_dir=directory / "substrates",
                    outputs_dir=directory / "outputs",
                )
                == 0
            )
        collected = srv.collect_campaign(directory=str(directory))
        assert collected["type"] == "CauseEffectStructure"
        assert "scope_report" in collected

    def test_estimate_cost_scope(self):
        handle = srv.load_example("basic")["handle"]
        full = srv.estimate_cost(handle, compute="ces")["estimate"]
        scoped = srv.estimate_cost(
            handle, compute="ces", scope={"mechanisms": {"max_order": 1}}
        )["estimate"]
        assert scoped["mechanisms"] < full["mechanisms"]
```

Run: `uv run pytest test/mcp/ -q > /tmp/c2t9a.log 2>&1`; read the log.
Expected: PASS.

- [ ] **Step 2: Docs**

`docs/howto/campaigns.md` — append two sections:

- `## Declare the feasible surface (scope)` — what a scope is (changes
  what is computed, with certificates — never a silent approximation),
  the `AxisScope` constraint forms with a code example
  (`CESScope(mechanisms=AxisScope(containing=("A",), max_order=3))`),
  intersection semantics, explicit-is-exclusive, labels or indices.
- `## Distribute one system's cause-effect structure` — `prepare_ces`
  worked example (scope + `units_per_job`), the planning ladder in one
  paragraph, the three SIA modes, `ordering="bottleneck_first"` for
  sparse substrates, collecting into a `CauseEffectStructure`, and
  reading the scope report (`scope_report()`, the Σφ_r lower bound and
  measured upper bounds, missing-vs-excluded).

`pyphi/mcp/content/campaigns.md` — append a `## Scoped CES campaigns`
section summarizing the same in ~15 lines, naming the
`prepare_ces_campaign` tool and the scope-dict shape.

Verify: `just docs > /tmp/c2docs.log 2>&1`; read the end for "build
succeeded".

- [ ] **Step 3: Changelog + ROADMAP**

```bash
printf '%s\n' 'Added scoped CES sharding to `pyphi.campaign`: declare the combinatorially feasible surface with `CESScope`/`AxisScope` (explicit lists, order bounds, unit containment), let `prepare_ces()` plan mechanism/purview-range/partition-stride shards to a per-job budget, and `collect()` reassembles the exact `CauseEffectStructure` — tie sets preserved, congruence and relations through the standard path — with a certified scope report (`Σφ_r` lower bound + measured upper bounds). `estimate_analysis` accepts `scope=`; `estimate_cost` and new `prepare_ces_campaign` MCP tools expose it.' > changelog.d/ces-sharding.feature.md
```

`ROADMAP.md` P11 row: flip to ✅ landed. Rewrite the row's tail: cycle 2
landed (scope objects, scope-aware estimation + per-mechanism workloads,
the three-rung planner, shard task kinds, exact tie-preserving merges
verified sharded ≡ unsharded under both IIT 4.0 presets, three SIA modes,
scope report with measured-bound certificates, MCP + docs). Keep the Dask
half's description; note the only remaining P11 item is external (CHTC
port-access confirmation for the Dask pilot pattern, a documentation
status, not code).

- [ ] **Step 4: Full-suite gate**

```bash
uv run pytest -q > /tmp/c2full.log 2>&1
```

Read the summary line (never the exit code of a pipeline). Expected: 0
failures. Then merge per the standing workflow (check main tip first;
merged-main gate before cleanup).

- [ ] **Step 5: Commit**

```bash
git add pyphi/mcp/server.py pyphi/mcp/content/campaigns.md docs/howto/campaigns.md changelog.d/ces-sharding.feature.md ROADMAP.md test/mcp/test_server.py
git commit -m "Expose scoped CES campaigns via MCP and docs; complete roadmap row"
git log --oneline -1
```
