# Scoped-CES Campaign Follow-ups Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the five 2026-07-21 ROADMAP items: `limit=` threading on `prepare_ces`, order-dependent purview caps on `CESScope`, memory-aware shard sizing with stratified packing, the scoped multi-system sweep, and the fat-node crossover doc note.

**Architecture:** Per the approved spec (`docs/superpowers/specs/2026-07-21-scoped-ces-campaign-followups-design.md`). `prepare_ces` stays a separate entry point from `prepare` but adopts its axis surface via the shared `pyphi.sweep` helpers; the shard planner gains per-item memory estimates and packs within memory classes; every purview-selection site routes through one new `CESScope.purview_axis(direction, mechanism)` method so planning and execution cannot disagree.

**Tech Stack:** Python 3.13, pytest, msgspec serialization, HTCondor submit files. Everything is unreleased 2.0 work — **no back-compat shims, no deprecation aliases**; rename and update callers.

## Global Constraints

- Run all Python through `uv run` (`uv run pytest`, `uv run python`).
- Tests that compute φ pin their formalism with complete presets (`presets.by_name["IIT_4_0_2026"]`), never partial `iit.*` fields.
- Docstrings: NumPy style, final-state impersonal voice, no process narrative, no planning artifacts (no "P11", "per ROADMAP", task numbers) in source/comments/docstrings/changelog.
- Never pipe test output through `tail`/`head`; redirect to a file and read the summary line.
- Do not destroy unrelated working-directory changes; stage only files this plan touches.
- Commit messages describe what changed and why, nothing conversational; end with the Co-Authored-By / Claude-Session trailer.
- Full `uv run pytest` (no path argument, so the doctest sweep runs) at least once before declaring done (Task 10).

**Key existing code (read before each task):**
- `pyphi/campaign/__init__.py` — `prepare` (:305), `_write_campaign_scaffold` (:458), `prepare_ces` (:483), `status` (:705), `collect` (:749), `_collect_ces` (:972), `_SUBMIT_TEMPLATE` (:279)
- `pyphi/campaign/shards.py` — `ShardSpec` (:42), `plan_ces_shards` (:139), `plan_sia_shards` (:225), `_pack_specs` (:122)
- `pyphi/campaign/scope.py` — `AxisScope`, `CESScope`, `resolve_scope`
- `pyphi/campaign/runner.py` — `_run_ces_shard` (:113), purview selection at :131–140
- `pyphi/cost.py` — `mechanism_workloads` (:464), `partition_sweep_count` (:457), `_Counter` (:45)
- `pyphi/sweep.py` — `_normalize_substrates`/`_normalize_states`/`_normalize_subsets`/`_normalize_formalisms`/`_enumerate_cells` (:68–133), `SweepResult` (:47), `_extract_row` (:208), `_build_df` (:287)
- `pyphi/serialize/schema.py` — `CESScopeSchema` (:554); `pyphi/serialize/convert.py` — scope encoders (:1351–1386)
- `pyphi/mcp/server.py` — `prepare_ces_campaign` (:521)
- Tests: `test/campaign/` (`test_shards.py`, `test_prepare_ces.py`, `test_runner_shards.py`, `test_collect_ces.py`, `test_scope.py`)

---

### Task 1: Thread `limit=` through `prepare_ces` and deduplicate the planning walk

**Files:**
- Modify: `pyphi/campaign/shards.py:139-153` (`plan_ces_shards`)
- Modify: `pyphi/campaign/__init__.py:483-590` (`prepare_ces`)
- Test: `test/campaign/test_prepare_ces.py`, `test/campaign/test_shards.py`

**Interfaces:**
- Consumes: `pyphi.cost.mechanism_workloads(substrate, subset, scope, limit)` (existing).
- Produces: `plan_ces_shards(system, scope, units_per_job, limit=10_000_000, workloads=None)` — `workloads` is a precomputed `mechanism_workloads` mapping; when given, the planner does not walk again. `prepare_ces(..., limit: int = 100_000_000)`.

- [ ] **Step 1: Write the failing tests**

Append to `test/campaign/test_prepare_ces.py`:

```python
def test_limit_threads_through_prepare_ces(tmp_path):
    with pytest.raises(ValueError, match="narrow the scope or raise the limit"):
        prepare_ces(
            examples.basic_substrate(),
            state=BASIC_STATE,
            formalism="IIT_4_0_2026",
            directory=tmp_path / "camp",
            units_per_job=50.0,
            limit=1,
        )
    # The failed call must not leave a campaign directory behind.
    assert not (tmp_path / "camp").exists()
```

Append to `test/campaign/test_shards.py`:

```python
def test_precomputed_workloads_match_internal_walk():
    from pyphi.cost import mechanism_workloads

    system = _system()
    with config.override(**presets.by_name["IIT_4_0_2026"], **PIN):
        workloads = mechanism_workloads(
            system.substrate, subset=system.node_indices, scope=CESScope()
        )
        a = plan_ces_shards(_system(), CESScope(), units_per_job=5.0)
        b = plan_ces_shards(
            _system(), CESScope(), units_per_job=5.0, workloads=workloads
        )
    assert a == b
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/campaign/test_prepare_ces.py::test_limit_threads_through_prepare_ces test/campaign/test_shards.py::test_precomputed_workloads_match_internal_walk -v`
Expected: FAIL — `TypeError: prepare_ces() got an unexpected keyword argument 'limit'` and `TypeError: plan_ces_shards() got an unexpected keyword argument 'workloads'`.

- [ ] **Step 3: Implement**

In `pyphi/campaign/shards.py`, change the `plan_ces_shards` signature and walk:

```python
def plan_ces_shards(
    system: Any,
    scope: Any,
    units_per_job: float,
    limit: int = 10_000_000,
    workloads: dict[tuple[int, ...], int] | None = None,
) -> list[ShardSpec]:
    """Plan the shards of a scoped cause-effect computation.

    Descends mechanism → purview-range → partition-stride only where the
    budget requires. Deterministic for fixed inputs; every spec carries its
    estimated work units.

    Parameters
    ----------
    system
        The system to analyze.
    scope
        The resolved feasibility surface.
    units_per_job : float
        Target work units per shard.
    limit : int, optional
        Work budget for the counting walk (ignored when ``workloads`` is
        given).
    workloads : dict, optional
        A precomputed :func:`pyphi.cost.mechanism_workloads` mapping for
        the same system and scope; when given, the walk is not repeated.
    """
    if workloads is None:
        workloads = mechanism_workloads(
            system.substrate, subset=system.node_indices, scope=scope, limit=limit
        )
```

(the rest of the function body is unchanged — it already iterates `workloads.items()`).

In `pyphi/campaign/__init__.py` `prepare_ces`:
1. Add `limit: int = 100_000_000,` to the signature (after `units_per_job`), and to the docstring Parameters:

```
    limit : int, optional
        Work budget for the planning walk. The walk raises
        :class:`ValueError` past the limit — the workload is then too
        large to plan; narrow the scope or raise the limit.
```

2. Replace the planning block (currently computes `ces_specs` first and `workloads` again at the end) so the walk runs exactly once, before `directory.mkdir`:

```python
    with config.override(**presets.by_name[formalism_], progress_bars=False):
        system = System.from_substrate(substrate, tuple(state), subset)
        resolved = resolve_scope(scope, system.node_labels)
        workloads = mechanism_workloads(
            substrate, subset=system.node_indices, scope=resolved, limit=limit
        )
        ces_specs = _shards.plan_ces_shards(
            system, resolved, units_per_job, workloads=workloads
        )
        ...
```

and delete the second `mechanism_workloads` call. `manifest["mechanism_workloads"]` keeps using `workloads.items()` as today.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/campaign/test_prepare_ces.py test/campaign/test_shards.py -v > /tmp/t1.log 2>&1; uv run python -c "print(open('/tmp/t1.log').read()[-2000:])"`
Expected: all PASS (existing tests too).

- [ ] **Step 5: Commit**

```bash
git add pyphi/campaign/shards.py pyphi/campaign/__init__.py test/campaign/test_prepare_ces.py test/campaign/test_shards.py
git commit -m "Expose limit= on prepare_ces and plan from a single workload walk"
```

---

### Task 2: Order-dependent purview caps on `CESScope`

**Files:**
- Modify: `pyphi/campaign/scope.py`
- Modify: `pyphi/serialize/schema.py:554-557`, `pyphi/serialize/convert.py:1373-1386`
- Test: `test/campaign/test_scope.py`, `test/test_serialization_roundtrip.py` if scope round-trips live there (check; otherwise test in `test_scope.py`)

**Interfaces:**
- Produces: `CESScope.max_purview_order_by_mechanism_order: tuple[tuple[int, int], ...] | None = None` and `CESScope.purview_axis(direction: Direction, mechanism: tuple[int, ...]) -> AxisScope`. Later tasks call `purview_axis` at every selection site.

- [ ] **Step 1: Write the failing tests**

Append to `test/campaign/test_scope.py`:

```python
from pyphi.direction import Direction


def test_purview_axis_applies_order_cap():
    scope = CESScope(max_purview_order_by_mechanism_order=((1, 1), (2, 3)))
    axis = scope.purview_axis(Direction.CAUSE, (0,))
    assert axis.admits((0,))
    assert not axis.admits((0, 1))
    axis2 = scope.purview_axis(Direction.EFFECT, (0, 1))
    assert axis2.admits((0, 1, 2))
    assert not axis2.admits((0, 1, 2, 3))


def test_purview_axis_falls_back_for_unlisted_orders():
    scope = CESScope(
        cause_purviews=AxisScope(max_order=2),
        max_purview_order_by_mechanism_order=((1, 1),),
    )
    # order-3 mechanism is not in the table: static cap alone applies
    axis = scope.purview_axis(Direction.CAUSE, (0, 1, 2))
    assert axis.max_order == 2


def test_purview_axis_intersects_with_static_cap():
    scope = CESScope(
        cause_purviews=AxisScope(max_order=2),
        max_purview_order_by_mechanism_order=((1, 5),),
    )
    # the static cap is tighter than the table's: intersection wins
    assert scope.purview_axis(Direction.CAUSE, (0,)).max_order == 2


def test_purview_axis_filters_explicit_lists():
    scope = CESScope(
        cause_purviews=AxisScope(explicit=((0,), (0, 1))),
        max_purview_order_by_mechanism_order=((1, 1),),
    )
    axis = scope.purview_axis(Direction.CAUSE, (0,))
    assert axis.explicit == ((0,),)


def test_order_cap_table_validation():
    with pytest.raises(ValueError, match="unique"):
        CESScope(max_purview_order_by_mechanism_order=((1, 1), (1, 2)))
    with pytest.raises(ValueError, match="positive"):
        CESScope(max_purview_order_by_mechanism_order=((0, 1),))
    with pytest.raises(ValueError, match="positive"):
        CESScope(max_purview_order_by_mechanism_order=((1, 0),))


def test_order_cap_survives_resolution_and_serialization(tmp_path):
    from pyphi import examples, serialize
    from pyphi.campaign.scope import resolve_scope

    substrate = examples.basic_substrate()
    scope = CESScope(max_purview_order_by_mechanism_order=((1, 2),))
    resolved = resolve_scope(scope, substrate.node_labels)
    assert resolved.max_purview_order_by_mechanism_order == ((1, 2),)
    path = tmp_path / "scope.json.gz"
    serialize.save(resolved, path)
    assert (
        serialize.load(path).max_purview_order_by_mechanism_order == ((1, 2),)
    )
```

(Add `import pytest` at the top of `test_scope.py` if absent. If `Substrate` lacks `.node_labels`, get labels the way `resolve_scope`'s existing tests in this file do — follow the local pattern.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/campaign/test_scope.py -v`
Expected: new tests FAIL with `TypeError: __init__() got an unexpected keyword argument`.

- [ ] **Step 3: Implement in `pyphi/campaign/scope.py`**

Add the field, validation, and method to `CESScope`:

```python
@dataclass(frozen=True)
class CESScope:
    """The feasibility surface of a cause-effect structure computation.

    ``max_purview_order_by_mechanism_order`` is an explicit table of
    ``(mechanism order, max purview order)`` pairs applying to both purview
    directions on top of the static axes; mechanism orders absent from the
    table fall back to the static constraints alone. This expresses
    order-tied purview bounds (e.g. purview order ≤ 2·order + 1) exactly
    while the scope remains callable-free named data.
    """

    mechanisms: AxisScope = field(default_factory=AxisScope)
    cause_purviews: AxisScope = field(default_factory=AxisScope)
    effect_purviews: AxisScope = field(default_factory=AxisScope)
    max_purview_order_by_mechanism_order: tuple[tuple[int, int], ...] | None = None

    def __post_init__(self) -> None:
        table = self.max_purview_order_by_mechanism_order
        if table is None:
            return
        orders = [mech_order for mech_order, _ in table]
        if len(set(orders)) != len(orders):
            raise ValueError("mechanism orders in the cap table must be unique")
        if any(m < 1 or p < 1 for m, p in table):
            raise ValueError("cap table orders must be positive")

    def purviews(self, direction: Direction) -> AxisScope:
        if direction == Direction.CAUSE:
            return self.cause_purviews
        return self.effect_purviews

    def purview_axis(
        self, direction: Direction, mechanism: tuple[int, ...]
    ) -> AxisScope:
        """The effective purview constraint for one mechanism.

        The static axis for ``direction``, intersected with the cap table's
        bound for ``len(mechanism)`` when one is listed. Every purview
        selection — planning, counting, execution, and collection — goes
        through this method, so they cannot disagree about the scope.
        """
        axis = self.purviews(direction)
        if self.max_purview_order_by_mechanism_order is None:
            return axis
        cap = dict(self.max_purview_order_by_mechanism_order).get(len(mechanism))
        if cap is None:
            return axis
        if axis.explicit is not None:
            return AxisScope(
                explicit=tuple(e for e in axis.explicit if len(e) <= cap)
            )
        return AxisScope(
            min_order=axis.min_order,
            max_order=cap if axis.max_order is None else min(axis.max_order, cap),
            containing=axis.containing,
            within=axis.within,
        )
```

Update `resolve_scope` to carry the table through:

```python
def resolve_scope(scope: CESScope, node_labels) -> CESScope:
    """Return the scope with every unit reference normalized to indices."""
    return CESScope(
        mechanisms=_resolve_axis(scope.mechanisms, node_labels),
        cause_purviews=_resolve_axis(scope.cause_purviews, node_labels),
        effect_purviews=_resolve_axis(scope.effect_purviews, node_labels),
        max_purview_order_by_mechanism_order=(
            scope.max_purview_order_by_mechanism_order
        ),
    )
```

In `pyphi/serialize/schema.py` add the field (with default, last):

```python
class CESScopeSchema(msgspec.Struct, frozen=True, tag="ces_scope"):
    mechanisms: AxisScopeSchema
    cause_purviews: AxisScopeSchema
    effect_purviews: AxisScopeSchema
    max_purview_order_by_mechanism_order: tuple[tuple[int, int], ...] | None = None
```

In `pyphi/serialize/convert.py` extend the encoder/decoder at :1373–1386 to pass `max_purview_order_by_mechanism_order=c.max_purview_order_by_mechanism_order` (encoder) and `max_purview_order_by_mechanism_order=s.max_purview_order_by_mechanism_order` (decoder), mirroring the existing field-by-field style.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/campaign/test_scope.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyphi/campaign/scope.py pyphi/serialize/schema.py pyphi/serialize/convert.py test/campaign/test_scope.py
git commit -m "Add order-dependent purview caps to CESScope"
```

---

### Task 3: Route every purview-selection site through `purview_axis`

**Files:**
- Modify: `pyphi/campaign/shards.py:169`, `pyphi/campaign/runner.py:131-140`, `pyphi/cost.py:510`, `pyphi/campaign/__init__.py:1082`
- Test: `test/campaign/test_runner_shards.py` (end-to-end), `test/campaign/test_shards.py`

**Interfaces:**
- Consumes: `CESScope.purview_axis(direction, mechanism)` from Task 2.
- Produces: all four sites agree on the purview set for any scope; no site calls `scope.purviews(direction).select(...)` with a mechanism in hand anymore.

- [ ] **Step 1: Write the failing tests**

Append to `test/campaign/test_shards.py`:

```python
def test_order_cap_restricts_planned_purviews():
    capped = CESScope(max_purview_order_by_mechanism_order=((1, 1),))
    with config.override(**presets.by_name["IIT_4_0_2026"], **PIN):
        base = plan_ces_shards(_system(), CESScope(), units_per_job=1e9)
        capped_specs = plan_ces_shards(_system(), capped, units_per_job=1e9)
    total = sum(s.units for s in base)
    capped_total = sum(s.units for s in capped_specs)
    assert capped_total < total
```

Append to `test/campaign/test_runner_shards.py`, following that file's existing prepare→run→collect pattern (read it first; reuse its helpers for running every task in a campaign directory):

```python
def test_order_cap_agrees_between_planning_and_execution(tmp_path):
    """Every collected distinction's purviews obey the per-order cap."""
    from pyphi import examples
    from pyphi.campaign import collect, prepare_ces
    from pyphi.campaign.runner import run_task
    from pyphi.campaign.scope import CESScope

    directory = tmp_path / "camp"
    prepare_ces(
        examples.basic_substrate(),
        state=(1, 0, 0),
        formalism="IIT_4_0_2026",
        scope=CESScope(max_purview_order_by_mechanism_order=((1, 1), (2, 2))),
        directory=directory,
        units_per_job=1e9,
    )
    for task_path in sorted((directory / "tasks").iterdir()):
        run_task(task_path, directory / "substrates", directory / "outputs")
    result = collect(directory)
    caps = {1: 1, 2: 2}
    for d in result.distinctions:
        cap = caps.get(len(d.mechanism))
        if cap is not None:
            assert len(d.cause.purview) <= cap
            assert len(d.effect.purview) <= cap
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/campaign/test_shards.py::test_order_cap_restricts_planned_purviews test/campaign/test_runner_shards.py::test_order_cap_agrees_between_planning_and_execution -v`
Expected: FAIL — the cap is ignored (equal totals; purviews exceed cap).

- [ ] **Step 3: Implement — switch the four sites**

`pyphi/campaign/shards.py:169` (rung 2, mechanism in hand):

```python
            purviews = list(scope.purview_axis(direction, mechanism).select(purviews))
```

`pyphi/campaign/runner.py:131-140`:

```python
                    cause_purviews = list(
                        task.scope.purview_axis(Direction.CAUSE, mechanism).select(
                            system.potential_purviews(Direction.CAUSE, mechanism)
                        )
                    )
                    effect_purviews = list(
                        task.scope.purview_axis(Direction.EFFECT, mechanism).select(
                            system.potential_purviews(Direction.EFFECT, mechanism)
                        )
                    )
```

`pyphi/cost.py:510` (inside `mechanism_workloads`):

```python
                if scope is not None:
                    purviews = list(
                        scope.purview_axis(direction, mechanism).select(purviews)
                    )
```

`pyphi/campaign/__init__.py:1082` (collect's canonical purview list):

```python
            canonical = list(
                scope.purview_axis(dir_, tuple(mechanism)).select(
                    system.potential_purviews(dir_, mechanism)
                )
            )
```

Then grep to confirm no selection site remains:
Run: `grep -rn "purviews(direction)\|purviews(dir_)\|purviews(Direction" pyphi/ | grep -v purview_axis`
Expected: no hits outside `scope.py` itself.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/campaign/ -v > /tmp/t3.log 2>&1; uv run python -c "print(open('/tmp/t3.log').read()[-3000:])"`
Expected: all PASS (whole campaign suite — the rewire touches shared paths).

- [ ] **Step 5: Commit**

```bash
git add pyphi/campaign/shards.py pyphi/campaign/runner.py pyphi/cost.py pyphi/campaign/__init__.py test/campaign/test_shards.py test/campaign/test_runner_shards.py
git commit -m "Route all scoped purview selection through CESScope.purview_axis"
```

---

### Task 4: Memory estimator and per-mechanism workload records in `cost.py`

**Files:**
- Modify: `pyphi/cost.py` (`mechanism_workloads` :464; new dataclass + functions near the top)
- Modify: `pyphi/campaign/shards.py` (`plan_ces_shards` reads `.units`), `pyphi/campaign/__init__.py` (manifest writes `.units`)
- Test: `test/test_cost.py` (check filename: `grep -rln mechanism_workloads test/` and use that file)

**Interfaces:**
- Produces:
  - `MechanismWorkload` frozen dataclass with `units: int`, `max_repertoire_cells: int` (exported from `pyphi.cost`).
  - `mechanism_workloads(...) -> dict[tuple[int, ...], MechanismWorkload]`.
  - `shard_memory_bytes(max_repertoire_cells: int) -> int` — `REPERTOIRE_FACTOR * 8 * cells + BASE_MEMORY_BYTES` with module constants `REPERTOIRE_FACTOR = 4`, `BASE_MEMORY_BYTES = 1 << 30`.
  - `round_memory_bytes(n: int) -> int` — round up to the next 512 MB.

- [ ] **Step 1: Write the failing tests**

In the test file that covers `pyphi.cost` (locate with `grep -rln mechanism_workloads test/`), append:

```python
def test_mechanism_workloads_records_max_repertoire_cells():
    from pyphi import examples
    from pyphi.cost import mechanism_workloads

    substrate = examples.basic_substrate()  # 3 binary units
    workloads = mechanism_workloads(substrate)
    wl = workloads[(0,)]
    assert wl.units > 0
    # binary units: the largest possible purview repertoire is 2**3 cells
    assert wl.max_repertoire_cells == 2**3
    total = sum(w.units for w in workloads.values())
    assert total > 0


def test_shard_memory_bytes_and_rounding():
    from pyphi.cost import (
        BASE_MEMORY_BYTES,
        REPERTOIRE_FACTOR,
        round_memory_bytes,
        shard_memory_bytes,
    )

    assert shard_memory_bytes(0) == BASE_MEMORY_BYTES
    assert shard_memory_bytes(100) == REPERTOIRE_FACTOR * 8 * 100 + BASE_MEMORY_BYTES
    half_gb = 512 * 1024**2
    assert round_memory_bytes(1) == half_gb
    assert round_memory_bytes(half_gb) == half_gb
    assert round_memory_bytes(half_gb + 1) == 2 * half_gb
```

(If the substrate in `examples.basic_substrate()` restricts purviews via connectivity so the max purview is smaller than the full set, compute the expected value from `substrate.factored_tpm.alphabet_sizes` and the actual `potential_purviews` — verify while implementing and adjust the assertion to the exact expected integer, not a weaker inequality.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest <cost test file> -v -k "max_repertoire or shard_memory"`
Expected: FAIL — `AttributeError`/`ImportError`.

- [ ] **Step 3: Implement in `pyphi/cost.py`**

Near the module's other public definitions:

```python
REPERTOIRE_FACTOR = 4
"""Repertoires concurrently alive during a mechanism-partition sweep."""

BASE_MEMORY_BYTES = 1 << 30
"""Per-task overhead: interpreter, imports, substrate TPM, task payload."""

_MEMORY_STEP_BYTES = 512 * 1024**2


@dataclass(frozen=True)
class MechanismWorkload:
    """One mechanism's scoped workload and peak-memory driver.

    ``units`` counts purview evaluations plus mechanism-partition sweeps;
    ``max_repertoire_cells`` is the state-space size of the largest scoped
    purview (the product of its units' state counts), which sets the
    mechanism's peak repertoire memory.
    """

    units: int
    max_repertoire_cells: int


def shard_memory_bytes(max_repertoire_cells: int) -> int:
    """Estimated peak memory of a shard from its largest repertoire.

    ``REPERTOIRE_FACTOR × 8 bytes × max_repertoire_cells + BASE_MEMORY_BYTES``.
    The factor and base are calibration constants validated against
    scheduler-reported memory usage; requests derived from this estimate
    are rounded with :func:`round_memory_bytes`.
    """
    return REPERTOIRE_FACTOR * 8 * max_repertoire_cells + BASE_MEMORY_BYTES


def round_memory_bytes(n: int) -> int:
    """Round a byte count up to the next 512 MB request boundary."""
    return max(1, math.ceil(n / _MEMORY_STEP_BYTES)) * _MEMORY_STEP_BYTES
```

(`math` and `dataclass` imports: add if absent.) Update `mechanism_workloads`: inside the walk, track the max purview cells and return records. Alphabet sizes come from the substrate:

```python
    alphabet = substrate.factored_tpm.alphabet_sizes
    ...
    try:
        for mechanism in mechanism_iter:
            units = 0
            max_cells = 0
            for direction in (Direction.CAUSE, Direction.EFFECT):
                purviews = cs.potential_purviews(direction, mechanism)
                if scope is not None:
                    purviews = list(
                        scope.purview_axis(direction, mechanism).select(purviews)
                    )
                for purview in purviews:
                    counter.charge(1)
                    units += 1 + _mechanism_partition_count(
                        len(mechanism), len(purview), counter
                    )
                    max_cells = max(
                        max_cells, math.prod(alphabet[u] for u in purview)
                    )
            workloads[tuple(mechanism)] = MechanismWorkload(
                units=units, max_repertoire_cells=max_cells
            )
```

Update the return annotation to `dict[tuple[int, ...], MechanismWorkload]` and the docstring Returns section. Update the two consumers:
- `pyphi/campaign/shards.py` `plan_ces_shards`: `for mechanism, wl in workloads.items():` with `units = wl.units` (memory use comes in Task 5; for now only `.units`). Update its `workloads` parameter annotation to `dict[tuple[int, ...], MechanismWorkload] | None`.
- `pyphi/campaign/__init__.py` manifest: `",".join(map(str, mechanism)): wl.units for mechanism, wl in workloads.items()` (manifest value stays a plain int; `scope_report` only uses `len`).

Grep for other callers and update any found:
Run: `grep -rn "mechanism_workloads" pyphi/ test/ --include="*.py" | grep -v "def mechanism_workloads"`

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest <cost test file> test/campaign/ -v > /tmp/t4.log 2>&1; uv run python -c "print(open('/tmp/t4.log').read()[-3000:])"`
Expected: PASS, including all existing campaign tests against the new return type.

- [ ] **Step 5: Commit**

```bash
git add pyphi/cost.py pyphi/campaign/shards.py pyphi/campaign/__init__.py <cost test file>
git commit -m "Record per-mechanism peak-repertoire size and add the shard memory estimator"
```

---

### Task 5: Memory on `ShardSpec` and stratified packing

**Files:**
- Modify: `pyphi/campaign/shards.py` (`ShardSpec`, `_pack_specs`, `plan_ces_shards`, `plan_sia_shards`)
- Modify: `pyphi/serialize/schema.py` (`ShardSpecSchema`), `pyphi/serialize/convert.py` (ShardSpec encoder/decoder)
- Test: `test/campaign/test_shards.py`

**Interfaces:**
- Consumes: `shard_memory_bytes`, `round_memory_bytes`, `MechanismWorkload` from Task 4.
- Produces: `ShardSpec.memory_bytes: int = 0` — the final rounded, floored request for the shard. `plan_ces_shards(..., memory_floor_bytes: int = 0)` and `plan_sia_shards(system, units_per_job, memory_floor_bytes: int = 0)`. Packing never mixes two `memory_bytes` classes in one shard.

- [ ] **Step 1: Write the failing tests**

Append to `test/campaign/test_shards.py`:

```python
def test_shards_carry_memory_and_respect_floor():
    floor = 4 * 1024**3
    with config.override(**presets.by_name["IIT_4_0_2026"], **PIN):
        specs = plan_ces_shards(
            _system(), CESScope(), units_per_job=1e9, memory_floor_bytes=floor
        )
        sia = plan_sia_shards(_system(), 1e9, memory_floor_bytes=floor)
    # tiny 3-unit system: every estimate is below the floor
    assert all(s.memory_bytes == floor for s in specs + sia)


def test_packing_never_mixes_memory_classes():
    from pyphi.campaign.shards import ShardSpec, _pack_specs

    small = [
        ShardSpec(payload_kind="mechanisms", mechanisms=((i,),), units=1.0,
                  memory_bytes=2 * 1024**3)
        for i in range(4)
    ]
    big = [
        ShardSpec(payload_kind="mechanisms", mechanisms=((10 + i,),), units=1.0,
                  memory_bytes=8 * 1024**3)
        for i in range(2)
    ]
    packed = _pack_specs(small + big, units_per_job=100.0)
    for spec in packed:
        members = set(spec.mechanisms)
        source_memories = {
            s.memory_bytes for s in small + big if set(s.mechanisms) & members
        }
        assert len(source_memories) == 1
        assert spec.memory_bytes == source_memories.pop()


def test_memory_classes_stratify_purview_ranges():
    """A scope admitting purviews of very different sizes yields
    purview-range shards whose memory matches their own purviews, not the
    global max."""
    with config.override(**presets.by_name["IIT_4_0_2026"], **PIN):
        # tiny units budget forces rung 2 (purview-range) splitting
        specs = plan_ces_shards(
            _system(), CESScope(), units_per_job=3.0, memory_floor_bytes=0
        )
    ranges = [s for s in specs if s.payload_kind == "purview_range"]
    if len({s.memory_bytes for s in ranges}) > 1:
        # at least two classes exist: no range shard may contain a purview
        # whose rounded request differs from the shard's class
        from pyphi.cost import round_memory_bytes, shard_memory_bytes

        alphabet = _system().substrate.factored_tpm.alphabet_sizes
        import math as _math

        for s in ranges:
            for p in s.purviews:
                cells = _math.prod(alphabet[u] for u in p)
                assert (
                    round_memory_bytes(shard_memory_bytes(cells)) == s.memory_bytes
                )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/campaign/test_shards.py -v -k "memory"`
Expected: FAIL — no `memory_bytes` field / no `memory_floor_bytes` kwarg.

- [ ] **Step 3: Implement in `pyphi/campaign/shards.py`**

Add to `ShardSpec`:

```python
    memory_bytes: int = 0
```

Add a module helper:

```python
def _memory_class(cells: int, floor: int) -> int:
    """The rounded, floored memory request for a shard holding ``cells``."""
    return max(floor, round_memory_bytes(shard_memory_bytes(cells)))
```

(import `round_memory_bytes` and `shard_memory_bytes` from `pyphi.cost`.)

Rewrite `_pack_specs` to stratify:

```python
def _pack_specs(items: list[ShardSpec], units_per_job: float) -> list[ShardSpec]:
    """Cost-balance whole-mechanism items into "mechanisms" shards.

    Packing runs within each memory class, so one large-purview mechanism
    never inflates the request of a shard of small ones.
    """
    packed: list[ShardSpec] = []
    for memory in sorted({s.memory_bytes for s in items}):
        group = [s for s in items if s.memory_bytes == memory]
        weights = [s.units for s in group]
        jobs = max(1, math.ceil(sum(weights) / units_per_job))
        bins = cost_balanced_partition(weights, jobs)
        packed.extend(
            ShardSpec(
                payload_kind="mechanisms",
                mechanisms=tuple(m for i in indices for m in group[i].mechanisms),
                units=float(sum(group[i].units for i in indices)),
                memory_bytes=memory,
            )
            for indices in (sorted(b) for b in bins)
        )
    return packed
```

In `plan_ces_shards`: add `memory_floor_bytes: int = 0` to the signature (documented as "Minimum per-shard memory request in bytes; every shard's `memory_bytes` is at least this."). Compute the substrate alphabet once: `alphabet = system.substrate.factored_tpm.alphabet_sizes`.

- Rung 1: `whole.append(ShardSpec(..., memory_bytes=_memory_class(wl.max_repertoire_cells, memory_floor_bytes)))`.
- Rung 2: compute per-purview classes alongside weights, then split the `fitting` packing by class:

```python
            weights = [
                1.0 + partition_sweep_count(len(mechanism), len(p)) for p in purviews
            ]
            memories = [
                _memory_class(
                    math.prod(alphabet[u] for u in p), memory_floor_bytes
                )
                for p in purviews
            ]
            triples = list(zip(purviews, weights, memories, strict=True))
            oversized = [(p, w, m) for p, w, m in triples if w > units_per_job]
            fitting = [(p, w, m) for p, w, m in triples if w <= units_per_job]
            for memory in sorted({m for _, _, m in fitting}):
                group = [(p, w) for p, w, m in fitting if m == memory]
                jobs = max(1, math.ceil(sum(w for _, w in group) / units_per_job))
                bins = cost_balanced_partition([w for _, w in group], jobs)
                specs.extend(
                    ShardSpec(
                        payload_kind="purview_range",
                        mechanism=mechanism,
                        direction=direction.name,
                        purviews=tuple(group[i][0] for i in bin_indices),
                        units=float(sum(group[i][1] for i in bin_indices)),
                        memory_bytes=memory,
                    )
                    for bin_indices in (sorted(b) for b in bins)
                )
```

- Rung 3: each stride spec gets `memory_bytes=m` from its `(p, w, m)` triple.

In `plan_sia_shards`: add `memory_floor_bytes: int = 0`; the whole-system repertoire drives it:

```python
    alphabet = system.substrate.factored_tpm.alphabet_sizes
    cells = math.prod(alphabet[u] for u in system.node_indices)
    memory = _memory_class(cells, memory_floor_bytes)
```

and set `memory_bytes=memory` on each spec.

Serialization: add `memory_bytes: int = 0` to `ShardSpecSchema` (last field, with default) and thread it through the ShardSpec encoder/decoder in `convert.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/campaign/ -v > /tmp/t5.log 2>&1; uv run python -c "print(open('/tmp/t5.log').read()[-3000:])"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyphi/campaign/shards.py pyphi/serialize/schema.py pyphi/serialize/convert.py test/campaign/test_shards.py
git commit -m "Estimate per-shard memory and pack shards within memory classes"
```

---

### Task 6: Per-task `request_memory` in the campaign scaffold

**Files:**
- Modify: `pyphi/campaign/__init__.py` (`_SUBMIT_TEMPLATE` :279, `_write_campaign_scaffold` :458, `prepare` :305, `prepare_ces` :483, `status` :705)
- Test: `test/campaign/test_prepare_ces.py`, `test/campaign/test_prepare.py`, `test/campaign/test_collect.py` (status round-trip lives wherever `status` is tested — check `grep -rln "def test.*status" test/campaign/`)

**Interfaces:**
- Produces:
  - `_SUBMIT_TEMPLATE` uses `request_memory = $(memory)` and `queue task_id, memory from remaining.txt`.
  - `remaining.txt` rows are `"<task_id>, <memory>"` (e.g. `0, 4608MB`).
  - `_parse_memory(s: str) -> int` (accepts `"4GB"`, `"512MB"`; case-insensitive; `ValueError` otherwise) and `_format_memory(n: int) -> str` (always MB: `f"{n // 1024**2}MB"`).
  - CES manifest task rows gain `"memory_bytes": int`; sweep manifests gain top-level `"request_memory": str`.
  - `_write_campaign_scaffold(directory, memory_by_task: list[str], container_image, request_disk)`.
  - `prepare_ces`'s `request_memory: str = "4GB"` becomes the **floor** (parsed and passed as `memory_floor_bytes` to both planners).
  - `status()` rewrites `remaining.txt` with the memory column intact.

- [ ] **Step 1: Write the failing tests**

Append to `test/campaign/test_prepare_ces.py`:

```python
def test_scaffold_requests_memory_per_task(tmp_path):
    directory = tmp_path / "camp"
    prepare_ces(
        examples.basic_substrate(),
        state=BASIC_STATE,
        formalism="IIT_4_0_2026",
        directory=directory,
        units_per_job=50.0,
    )
    sub = (directory / "pyphi.sub").read_text()
    assert "request_memory      = $(memory)" in sub
    assert "queue task_id, memory from remaining.txt" in sub
    manifest = json.loads((directory / "manifest.json").read_text())
    lines = (directory / "remaining.txt").read_text().splitlines()
    assert len(lines) == len(manifest["tasks"])
    for line, row in zip(lines, manifest["tasks"], strict=True):
        task_id, memory = (part.strip() for part in line.split(","))
        assert int(task_id) == row["task_id"]
        assert memory == f"{row['memory_bytes'] // 1024**2}MB"
    # default 4GB floor: no task requests less
    assert all(row["memory_bytes"] >= 4 * 1024**3 for row in manifest["tasks"])


def test_status_rewrite_preserves_memory_column(tmp_path):
    from pyphi.campaign import status

    directory = tmp_path / "camp"
    prepare_ces(
        examples.basic_substrate(),
        state=BASIC_STATE,
        formalism="IIT_4_0_2026",
        directory=directory,
        units_per_job=50.0,
    )
    before = (directory / "remaining.txt").read_text()
    (directory / "remaining.txt").write_text("")  # clobber
    status(directory)  # all tasks pending: full rewrite
    assert (directory / "remaining.txt").read_text() == before
```

Append to `test/campaign/test_prepare.py` (match its existing fixture style for calling `prepare` — read the file first and reuse its smallest passing invocation):

```python
def test_sweep_scaffold_writes_uniform_memory_column(tmp_path):
    # reuse this file's minimal prepare(...) invocation, adding:
    #   request_memory="2GB", directory=tmp_path / "camp"
    directory = tmp_path / "camp"
    prepare(
        examples.basic_substrate(),
        states=[(1, 0, 0)],
        formalisms=["IIT_4_0_2026"],
        directory=directory,
        request_memory="2GB",
    )
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["request_memory"] == "2GB"
    for line in (directory / "remaining.txt").read_text().splitlines():
        assert line.split(",")[1].strip() == "2GB"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/campaign/test_prepare_ces.py test/campaign/test_prepare.py -v -k "memory or scaffold"`
Expected: FAIL — old single-value template and bare task-id `remaining.txt`.

- [ ] **Step 3: Implement in `pyphi/campaign/__init__.py`**

Template:

```python
_SUBMIT_TEMPLATE = """\
universe            = container
container_image     = {container_image}
executable          = run_task.sh
arguments           = $(task_id)
transfer_input_files = tasks/task-$(task_id).json.gz, substrates/
transfer_output_remaps = "task-$(task_id).json.gz = outputs/task-$(task_id).json.gz"
should_transfer_files = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
request_cpus        = 1
request_memory      = $(memory)
request_disk        = {request_disk}
log                 = logs/task-$(task_id).log
output              = logs/task-$(task_id).out
error               = logs/task-$(task_id).err
queue task_id, memory from remaining.txt
"""
```

Helpers (module level, near `_pack`):

```python
_MEMORY_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(MB|GB)\s*$", re.IGNORECASE)


def _parse_memory(value: str) -> int:
    """Parse a scheduler memory string (``"4GB"``, ``"512MB"``) to bytes."""
    match = _MEMORY_RE.match(value)
    if match is None:
        raise ValueError(
            f"cannot parse memory value {value!r}; expected e.g. '4GB' or '512MB'"
        )
    number, unit = float(match.group(1)), match.group(2).upper()
    return int(number * (1024**3 if unit == "GB" else 1024**2))


def _format_memory(n: int) -> str:
    return f"{n // 1024**2}MB"


def _remaining_lines(memory_by_task: dict[int, str]) -> str:
    return "".join(
        f"{task_id}, {memory}\n" for task_id, memory in sorted(memory_by_task.items())
    )


def _task_memory_strings(manifest: dict) -> list[str]:
    """Per-task memory request strings for either campaign kind."""
    if manifest["kind"] == "sweep_cells":
        return [manifest["request_memory"]] * len(manifest["tasks"])
    return [_format_memory(row["memory_bytes"]) for row in manifest["tasks"]]
```

(`import re` if absent.) `_write_campaign_scaffold` becomes:

```python
def _write_campaign_scaffold(
    directory: Path,
    memory_by_task: list[str],
    container_image: str,
    request_disk: str,
) -> None:
    """Write the scheduler-facing campaign files common to every kind."""
    (directory / "remaining.txt").write_text(
        _remaining_lines(dict(enumerate(memory_by_task)))
    )
    run_task_sh = directory / "run_task.sh"
    run_task_sh.write_text(_RUN_TASK_SH)
    run_task_sh.chmod(
        run_task_sh.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    )
    (directory / "pyphi.sub").write_text(
        _SUBMIT_TEMPLATE.format(
            container_image=container_image,
            request_disk=request_disk,
        )
    )
```

`prepare`: add `"request_memory": request_memory,` to its manifest; call `_write_campaign_scaffold(directory, [request_memory] * len(tasks), container_image, request_disk)`. Docstring for `request_memory` unchanged in meaning (uniform per-task request).

`prepare_ces`:
- Parse the floor once: `memory_floor = _parse_memory(request_memory)`.
- Pass `memory_floor_bytes=memory_floor` to `plan_ces_shards` and `plan_sia_shards`.
- Task rows: `{"task_id": task_id, "kind": ..., "units": spec.units, "memory_bytes": spec.memory_bytes}`.
- Scaffold call: `_write_campaign_scaffold(directory, [_format_memory(row["memory_bytes"]) for row in task_rows], container_image, request_disk)`.
- Docstring for `request_memory`: "Minimum per-shard memory request (the floor). Every shard requests the greater of this and its estimated peak; a large floor disables stratification."

`status`: replace the `remaining.txt` write with:

```python
    memory = _task_memory_strings(manifest)
    (directory / "remaining.txt").write_text(
        _remaining_lines(
            {task_id: memory[task_id] for task_id in sorted(pending + failed)}
        )
    )
```

Also update `docs/howto/campaigns.md:60,79,95` mentions of `remaining.txt` ("task ids" → "task id, memory rows") — the fuller doc pass is Task 9, but these three lines describe the file format and must not lie; fix them here.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/campaign/ -v > /tmp/t6.log 2>&1; uv run python -c "print(open('/tmp/t6.log').read()[-3000:])"`
Expected: PASS (existing `status`/`collect` tests exercise the rewrite path).

- [ ] **Step 5: Commit**

```bash
git add pyphi/campaign/__init__.py test/campaign/ docs/howto/campaigns.md
git commit -m "Request memory per task in campaign scaffolds; request_memory becomes the shard floor"
```

---

### Task 7: `prepare_ces` adopts the sweep axes

**Files:**
- Modify: `pyphi/campaign/__init__.py` (`prepare_ces`), `pyphi/sweep.py` (`_normalize_formalisms` string guard)
- Modify: `pyphi/mcp/server.py:521-590` (`prepare_ces_campaign`)
- Test: `test/campaign/test_prepare_ces.py` (existing calls updated for renamed params + new tests)

**Interfaces:**
- Consumes: `_normalize_substrates`, `_enumerate_cells` from `pyphi.sweep`; `plan_ces_shards(..., workloads=..., memory_floor_bytes=...)`; `plan_sia_shards(..., memory_floor_bytes=...)`.
- Produces the new signature (parameters renamed, **no aliases kept**):

```python
def prepare_ces(
    substrates: Any,
    *,
    states: Any,
    subsets: Any = "full",
    formalisms: Any = None,
    scope: Any = None,
    directory: Any,
    units_per_job: float,
    limit: int = 100_000_000,
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
```

- New CES manifest shape (consumed by Task 8):

```json
{
  "kind": "ces",
  "cells": [[label, formalism, [subset...], [state...]], ...],
  "groups": [{"label": ..., "formalism": ..., "subset": [...],
              "partition_scheme": ..., "mechanism_partition_scheme": ...,
              "mechanism_workloads": {"0,1": units, ...}}, ...],
  "tasks": [{"task_id": 0, "kind": "ces_shard", "units": ..., "memory_bytes": ...,
             "cell": 0}, ...],
  "sia_mode": ..., "ordering": ..., "seed": ..., "units_per_job": ...,
  "infeasible_threshold": ..., "pyphi_version": ..., "created": ...
}
```

- Substrate files: `substrates/substrate-<label>.json.gz` per label (a bare substrate gets label `0`). Root `scope.json.gz` stores the **unresolved** user scope; each task carries its per-substrate resolved scope as before.
- A single-state scalar call — `prepare_ces(substrate, states=(1, 0, 0), ...)` — produces exactly one cell (via `_normalize_states`' scalar branch).

- [ ] **Step 1: Update existing tests for the rename and write the new failing tests**

In `test/campaign/test_prepare_ces.py` and `test/campaign/test_runner_shards.py` and `test/campaign/test_collect_ces.py`, mechanically rename call kwargs: `state=` → `states=`, `subset=` → `subsets=` (wrapping single subsets as `[subset]` where a tuple of ints was passed), `formalism=` → `formalisms="IIT_4_0_2026"` — note `formalisms` accepts a bare string after the `_normalize_formalisms` guard below. Substrate filename assertions change from `substrate-system.json.gz` to `substrate-0.json.gz`.

Then append to `test/campaign/test_prepare_ces.py`:

```python
def test_multi_state_campaign_replicates_shards_per_state(tmp_path):
    directory = tmp_path / "camp"
    states = [(1, 0, 0), (0, 1, 0)]
    prepare_ces(
        examples.basic_substrate(),
        states=states,
        formalisms="IIT_4_0_2026",
        directory=directory,
        units_per_job=1e9,
    )
    manifest = json.loads((directory / "manifest.json").read_text())
    assert len(manifest["cells"]) == 2
    assert len(manifest["groups"]) == 1  # one (label, formalism, subset) group
    by_cell: dict[int, int] = {}
    for row in manifest["tasks"]:
        by_cell[row["cell"]] = by_cell.get(row["cell"], 0) + 1
    # identical plan per state: same task count for each cell
    assert by_cell[0] == by_cell[1]
    task0 = load(directory / "tasks" / "task-0000.json.gz")
    assert tuple(task0.state) in set(states)


def test_multi_cell_rejects_precomputed_sia(tmp_path):
    import pyphi
    from pyphi.conf import presets as _presets

    substrate = examples.basic_substrate()
    with pyphi.config.override(
        **_presets.by_name["IIT_4_0_2026"], parallel=False, progress_bars=False
    ):
        sia = pyphi.System(substrate, BASIC_STATE).sia()
    with pytest.raises(ValueError, match="single-cell"):
        prepare_ces(
            substrate,
            states=[(1, 0, 0), (0, 1, 0)],
            formalisms="IIT_4_0_2026",
            directory=tmp_path / "camp",
            units_per_job=1e9,
            sia=sia,
        )


def test_formalisms_accepts_bare_string(tmp_path):
    prepare_ces(
        examples.basic_substrate(),
        states=BASIC_STATE,
        formalisms="IIT_4_0_2026",
        directory=tmp_path / "camp",
        units_per_job=1e9,
    )
    manifest = json.loads((tmp_path / "camp" / "manifest.json").read_text())
    assert manifest["cells"][0][1] == "IIT_4_0_2026"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/campaign/test_prepare_ces.py -v`
Expected: new tests FAIL (`unexpected keyword argument 'states'`); renamed existing tests FAIL the same way until Step 3.

- [ ] **Step 3: Implement**

In `pyphi/sweep.py`, guard `_normalize_formalisms` against the bare-string iteration trap (fixes `prepare` too):

```python
def _normalize_formalisms(formalisms: Any) -> list[str]:
    if formalisms is None:
        return [config.formalism.iit.version]
    if isinstance(formalisms, str):
        return [formalisms]
    return list(formalisms)
```

Rewrite `prepare_ces` (keeping the docstring's structure; new/changed parameter docs: `substrates`, `states`, `subsets`, `formalisms` say "As in :func:`pyphi.sweep.sweep`"; `scope` says "One :class:`CESScope` shared by every cell, resolved per substrate"). Body:

```python
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(
            f"campaign directory {directory} already exists; "
            "campaign directories are never overwritten"
        )
    if sia is not None and resolution_state is not None:
        raise ValueError("pass either sia or resolution_state, not both")
    formalisms_ = _normalize_formalisms(formalisms)
    for name in formalisms_:
        if name not in presets.by_name:
            raise ValueError(f"unknown formalism {name!r}")
    scope = scope if scope is not None else CESScope()
    labeled = _normalize_substrates(substrates)
    substrate_map = dict(labeled)
    cells = _enumerate_cells(labeled, states, subsets, formalisms_)
    if not cells:
        raise ValueError("the given axes enumerate no cells")
    if (sia is not None or resolution_state is not None) and len(cells) > 1:
        raise ValueError(
            "sia and resolution_state apply only to single-cell campaigns"
        )
    memory_floor = _parse_memory(request_memory)

    # Plan once per (label, formalism, subset) group; the shard plan is
    # state-independent, so states replicate tasks, not planning.
    group_keys: list[tuple[Any, str, tuple]] = []
    group_data: dict[tuple[Any, str, tuple], dict] = {}
    for label, formalism_, subset, state in cells:
        key = (label, formalism_, subset)
        if key in group_data:
            continue
        group_keys.append(key)
        with config.override(**presets.by_name[formalism_], progress_bars=False):
            # The plan is state-independent; the group's first cell state
            # stands in for construction (and is validated here).
            system = System.from_substrate(substrate_map[label], tuple(state), subset)
            resolved = resolve_scope(scope, system.node_labels)
            workloads = mechanism_workloads(
                substrate_map[label],
                subset=system.node_indices,
                scope=resolved,
                limit=limit,
            )
            ces_specs = _shards.plan_ces_shards(
                system,
                resolved,
                units_per_job,
                workloads=workloads,
                memory_floor_bytes=memory_floor,
            )
            if not any(s.mechanisms or s.mechanism for s in ces_specs):
                raise ValueError("the scope admits zero mechanisms")
            sia_specs = (
                _shards.plan_sia_shards(
                    system, units_per_job, memory_floor_bytes=memory_floor
                )
                if sia is None and resolution_state is None
                else []
            )
            group_data[key] = {
                "resolved": resolved,
                "workloads": workloads,
                "ces_specs": ces_specs,
                "sia_specs": sia_specs,
                "subset": tuple(system.node_indices),
                "partition_scheme": config.formalism.iit.system_partition_scheme,
                "mechanism_partition_scheme": (
                    config.formalism.iit.mechanism_partition_scheme
                ),
            }

    for data in group_data.values():
        for spec in data["ces_specs"] + data["sia_specs"]:
            if spec.units > infeasible_threshold:
                message = (
                    f"shard {spec!r} estimate {spec.units:.3g} exceeds "
                    f"infeasible_threshold {infeasible_threshold:.3g}"
                )
                if strict:
                    raise ValueError(message)
                warnings.warn(message, PyPhiWarning, stacklevel=2)

    directory.mkdir(parents=True)
    (directory / "outputs").mkdir()
    (directory / "logs").mkdir()
    substrates_dir = directory / "substrates"
    substrates_dir.mkdir()
    for label, substrate in labeled:
        serialize.save(substrate, substrates_dir / f"substrate-{label}.json.gz")
    serialize.save(scope, directory / "scope.json.gz")
    if sia is not None:
        serialize.save(sia, directory / "sia.json.gz")
    if resolution_state is not None:
        serialize.save(resolution_state, directory / "resolution_state.json.gz")

    tasks_dir = directory / "tasks"
    tasks_dir.mkdir()
    overrides = _wire_overrides()
    task_rows: list[dict] = []
    task_id = 0
    for cell_index, (label, formalism_, subset, state) in enumerate(cells):
        data = group_data[(label, formalism_, subset)]
        with config.override(**presets.by_name[formalism_], progress_bars=False):
            # Validate every cell's state at prepare time, not on the cluster.
            System(substrate_map[label], tuple(state), node_indices=data["subset"])
        for spec in data["ces_specs"]:
            shard_task = CESShardTask(
                task_id=task_id,
                kind="ces_shard",
                substrate_label=label,
                state=tuple(state),
                subset=data["subset"],
                scope=data["resolved"],
                config_overrides=overrides,
                formalism=formalism_,
                spec=spec,
                ordering=ordering,
            )
            serialize.save(shard_task, tasks_dir / f"task-{task_id:04d}.json.gz")
            task_rows.append(
                {
                    "task_id": task_id,
                    "kind": "ces_shard",
                    "units": spec.units,
                    "memory_bytes": spec.memory_bytes,
                    "cell": cell_index,
                }
            )
            task_id += 1
        for spec in data["sia_specs"]:
            assert spec.stride is not None, "SIA shards are always strides"
            sia_task = SIAShardTask(
                task_id=task_id,
                kind="sia_shard",
                substrate_label=label,
                state=tuple(state),
                subset=data["subset"],
                config_overrides=overrides,
                formalism=formalism_,
                stride=spec.stride,
            )
            serialize.save(sia_task, tasks_dir / f"task-{task_id:04d}.json.gz")
            task_rows.append(
                {
                    "task_id": task_id,
                    "kind": "sia_shard",
                    "units": spec.units,
                    "memory_bytes": spec.memory_bytes,
                    "cell": cell_index,
                }
            )
            task_id += 1

    sia_mode = (
        "precomputed"
        if sia is not None
        else "none"
        if resolution_state is not None
        else "shards"
    )
    manifest = {
        "kind": "ces",
        "pyphi_version": importlib.metadata.version("pyphi"),
        "created": datetime.now(UTC).isoformat(),
        "seed": seed,
        "sia_mode": sia_mode,
        "ordering": ordering,
        "cells": [
            [label, formalism_, list(subset), list(state)]
            for label, formalism_, subset, state in cells
        ],
        "groups": [
            {
                "label": label,
                "formalism": formalism_,
                "subset": list(group_data[key]["subset"]),
                "partition_scheme": group_data[key]["partition_scheme"],
                "mechanism_partition_scheme": group_data[key][
                    "mechanism_partition_scheme"
                ],
                "mechanism_workloads": {
                    ",".join(map(str, mechanism)): wl.units
                    for mechanism, wl in group_data[key]["workloads"].items()
                },
            }
            for key in group_keys
            for label, formalism_, _subset in [key]
        ],
        "tasks": task_rows,
        "units_per_job": units_per_job,
        "infeasible_threshold": infeasible_threshold,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    _write_campaign_scaffold(
        directory,
        [_format_memory(row["memory_bytes"]) for row in task_rows],
        container_image,
        request_disk,
    )
    return CampaignStatus(
        directory=str(directory),
        n_tasks=len(task_rows),
        n_cells=len(cells),
        done=(),
        failed=(),
        pending=tuple(range(len(task_rows))),
        total_units=float(sum(row["units"] for row in task_rows)),
    )
```

Note: cells in `_enumerate_cells` carry the *requested* subset (possibly labels); the group's `data["subset"]` holds resolved node indices. The `cells` manifest entry must store the resolved subset — use `group_data[(label, formalism_, subset)]["subset"]` when writing `manifest["cells"]`.

**Important:** Task 8 rewrites `_collect_ces` against this manifest; until then `test/campaign/test_collect_ces.py` will fail on the manifest keys (`state`/`subset`/`formalism` gone). Do Tasks 7 and 8 as one PR-sized unit — commit Task 7 only when its prepare-side tests pass, accepting the temporarily red collect tests, or implement 7+8 before running the collect suite. Prefer: implement Task 7, run only `test_prepare_ces.py`, commit; then Task 8 restores the rest.

In `pyphi/mcp/server.py` `prepare_ces_campaign`: rename pass-through kwargs (`substrates=substrate`, `states=tuple(state)`, `subsets="full" if subset is None else [tuple(subset)]`, `formalisms=formalism`) and add `limit: int | None = None` (pass through only when not None; otherwise omit so the library default applies). Keep the MCP tool single-state; the docstring notes multi-state campaigns are a library-level feature.

- [ ] **Step 4: Run the prepare-side tests**

Run: `uv run pytest test/campaign/test_prepare_ces.py test/campaign/test_shards.py test/campaign/test_scope.py -v > /tmp/t7.log 2>&1; uv run python -c "print(open('/tmp/t7.log').read()[-3000:])"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyphi/campaign/__init__.py pyphi/sweep.py pyphi/mcp/server.py test/campaign/
git commit -m "prepare_ces sweeps substrates/states/subsets/formalisms under one scope"
```

---

### Task 8: Multi-cell collection → `SweepResult`; per-cell scope reports

**Files:**
- Modify: `pyphi/campaign/__init__.py` (`_collect_ces` :972, `scope_report` :895, `_build_scope_report` :943)
- Test: `test/campaign/test_collect_ces.py`

**Interfaces:**
- Consumes: Task 7 manifest (`cells`, `groups`, per-row `cell`), `pyphi.sweep._extract_row` / `_build_df` / `SweepResult`.
- Produces:
  - `collect(directory)` on a single-cell CES campaign: the assembled structure (unchanged behavior).
  - On a multi-cell campaign: `SweepResult` — `df` built by `_build_df(keys, rows, cells)` with `_extract_row(structure, "ces")` rows; `results` aligned structures; `skipped=[]`.
  - `scope_report(directory)`: single-cell → `ScopeReport`; multi-cell → `dict[tuple, ScopeReport]` keyed by `(label, formalism, subset, state)` cell tuples.
  - `scope_report.json` on disk: single-cell → one report dict (unchanged); multi-cell → `{"cells": [{"cell": [...], "report": {...}}, ...]}`.

- [ ] **Step 1: Update existing collect tests for renamed params (Task 7) and write the new failing test**

Append to `test/campaign/test_collect_ces.py` (follow the file's existing run-all-tasks helper):

```python
def test_multi_state_collect_returns_sweep_result(tmp_path):
    import pyphi
    from pyphi.campaign import collect, prepare_ces, scope_report
    from pyphi.campaign.runner import run_task
    from pyphi.conf import presets as _presets
    from pyphi.sweep import SweepResult

    substrate = examples.basic_substrate()
    states = [(1, 0, 0), (0, 1, 0)]
    directory = tmp_path / "camp"
    prepare_ces(
        substrate,
        states=states,
        formalisms="IIT_4_0_2026",
        directory=directory,
        units_per_job=1e9,
    )
    for task_path in sorted((directory / "tasks").iterdir()):
        run_task(task_path, directory / "substrates", directory / "outputs")
    result = collect(directory)
    assert isinstance(result, SweepResult)
    assert len(result.results) == 2
    # each cell's structure equals the local computation
    for state, structure in zip(states, result.results, strict=True):
        with pyphi.config.override(
            **_presets.by_name["IIT_4_0_2026"], parallel=False, progress_bars=False
        ):
            local = pyphi.System(substrate, state).ces()
        assert sorted(
            (tuple(d.mechanism), round(float(d.phi), 10)) for d in structure.distinctions
        ) == sorted(
            (tuple(d.mechanism), round(float(d.phi), 10)) for d in local.distinctions
        )
    reports = scope_report(directory)
    assert set(reports) == {
        (0, "IIT_4_0_2026", (0, 1, 2), tuple(state)) for state in states
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest test/campaign/test_collect_ces.py -v`
Expected: new test FAILS (`KeyError` on old manifest keys / non-SweepResult return); pre-existing tests pass again only after Step 3 (they were red after Task 7).

- [ ] **Step 3: Implement**

Restructure `_collect_ces` into a per-cell loop. Shape (complete restructure of :972–1141; the inner merge logic is today's code, moved into the loop and parameterized by cell):

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

    cells = _manifest_cells(manifest)
    multi = len(cells) > 1
    if multi and (sia_override is not None or resolution_state_override is not None):
        raise ValueError(
            "sia and resolution_state apply only to single-cell campaigns"
        )
    groups = {
        (g["label"], g["formalism"], tuple(g["subset"])): g
        for g in manifest["groups"]
    }
    rows_by_cell: dict[int, list[dict]] = {}
    for row in manifest["tasks"]:
        rows_by_cell.setdefault(row["cell"], []).append(row)

    substrates = {
        label: serialize.load(path)
        for path in (directory / "substrates").glob("substrate-*.json.gz")
        for label in [path.name.removeprefix("substrate-").removesuffix(".json.gz")]
    }
    user_scope = serialize.load(directory / "scope.json.gz")

    structures: list[Any] = []
    reports: list[tuple[tuple, ScopeReport]] = []
    for cell_index, cell in enumerate(cells):
        label, formalism_, subset, state = cell
        # substrate labels are JSON keys in filenames; bare-substrate label 0
        # round-trips as the string "0"
        substrate = substrates[str(label)]
        group = groups[(label, formalism_, tuple(subset))]
        with config.override(
            **presets.by_name[formalism_], parallel=False, progress_bars=False
        ):
            system = System(substrate, tuple(state), node_indices=tuple(subset))
            scope = resolve_scope(user_scope, system.node_labels)
            structure, report = _merge_cell(
                directory,
                manifest,
                group,
                rows_by_cell.get(cell_index, []),
                incomplete,
                system,
                scope,
                sia_override,
                resolution_state_override,
            )
        structures.append(structure)
        reports.append((cell, report))
        with_provenance = getattr(structure, "with_provenance", None)
        if with_provenance is not None:
            note = json.dumps(
                {
                    "campaign": str(directory),
                    "cell": [label, formalism_, list(subset), list(state)],
                    "scope_report": dataclasses.asdict(report),
                }
            )
            with_provenance(note=note, seed=manifest["seed"])

    if not multi:
        (directory / "scope_report.json").write_text(
            json.dumps(dataclasses.asdict(reports[0][1]), indent=2)
        )
        return structures[0]
    (directory / "scope_report.json").write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell": [cell[0], cell[1], list(cell[2]), list(cell[3])],
                        "report": dataclasses.asdict(report),
                    }
                    for cell, report in reports
                ]
            },
            indent=2,
        )
    )
    from pyphi.sweep import _build_df
    from pyphi.sweep import _extract_row

    rows = [_extract_row(s, "ces") for s in structures]
    df = _build_df(cells, rows, cells)
    return SweepResult(df=df, results=structures, skipped=[])
```

`_merge_cell` is today's grouping/merge/assembly body (lines 1008–1128) with these substitutions: iterate `rows` (this cell's task rows) instead of `manifest["tasks"]`; `expected_schemes` from `group["partition_scheme"]` / `group["mechanism_partition_scheme"]`; canonical purviews via `scope.purview_axis(dir_, tuple(mechanism)).select(...)` (already done in Task 3); SIA-shard completeness counts only this cell's `sia_shard` rows; `_build_scope_report(group, system, result, missing_groups, sia_mode)` takes the group dict (its `mechanism_workloads` provides the admitted count) instead of the whole manifest. Return `(structure, report)`. Keep `_group_name`, `_assemble_without_sia` unchanged.

`_build_scope_report` first parameter becomes `group: dict` and reads `group["mechanism_workloads"]`.

`scope_report()`:

```python
def scope_report(directory: Any) -> Any:
    """Read the scope report(s) a CES campaign's collection wrote.

    Returns one :class:`ScopeReport` for a single-cell campaign, or a dict
    keyed by ``(label, formalism, subset, state)`` for a multi-cell one.
    """
    path = Path(directory) / "scope_report.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; collect the campaign first")
    data = json.loads(path.read_text())
    if "cells" not in data:
        data["missing_groups"] = tuple(data["missing_groups"])
        return ScopeReport(**data)
    reports = {}
    for entry in data["cells"]:
        label, formalism_, subset, state = entry["cell"]
        report = dict(entry["report"])
        report["missing_groups"] = tuple(report["missing_groups"])
        reports[(label, formalism_, tuple(subset), tuple(state))] = ScopeReport(
            **report
        )
    return reports
```

Add `from pyphi.campaign.scope import resolve_scope` where needed (top-level import in `_collect_ces` is fine, matching the existing deferred-import style).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/campaign/ -v > /tmp/t8.log 2>&1; uv run python -c "print(open('/tmp/t8.log').read()[-3000:])"`
Expected: ALL campaign tests PASS (including the ones red since Task 7).

- [ ] **Step 5: Commit**

```bash
git add pyphi/campaign/__init__.py test/campaign/test_collect_ces.py
git commit -m "Collect multi-cell scoped campaigns into a SweepResult with per-cell scope reports"
```

---

### Task 9: Documentation — fat-node crossover note, howto updates, MCP content

**Files:**
- Modify: `docs/howto/campaigns.md`, `docs/howto/chtc.md`, `pyphi/mcp/content/campaigns.md`, `pyphi/mcp/prompts.py` (walkthrough text if it names `state=`/`formalism=`)
- No test; verify with the docs build.

- [ ] **Step 1: Update `docs/howto/campaigns.md`**

In the `prepare_ces` section (:183): update the example call to the new axes (`states=`, `formalisms=`), show a multi-state example with nine states sharing one scope, document `limit=`, the `request_memory` floor semantics, the order-cap table field, and that `collect` returns a `SweepResult` for multi-cell campaigns. Then append a new subsection:

```markdown
## When one fat node beats a sharded campaign

For a sparse, scoped, mid-size system (tens of units, mechanism order
capped), the alternative to sharding is one job per state that runs the
whole scoped analysis with native parallelism: `request_cpus = 32`,
`request_memory` sized to the analysis, and `pyphi.config.parallel`
enabled. Prefer the fat-node pattern when:

- the shard count at your budget is large (thousands) while a single
  state's whole analysis fits comfortably in one slot's memory and a
  72-hour window — per-shard scheduling overhead then dominates; or
- most shards' memory requests approach the whole-analysis footprint
  anyway (peak memory is set by the largest purview repertoire, which
  sharding cannot reduce).

Prefer sharding when a single state cannot finish in one slot, when
big-memory slots are scarce (stratified shard requests keep small work in
small slots), or when you need per-shard retry granularity on a busy
pool. Per-shard memory requests are estimated automatically, so holds
from underestimated memory are no longer the deciding factor.
```

- [ ] **Step 2: Update `docs/howto/chtc.md`**

At the `prepare_ces` mention (:65), add a sentence pointing to the crossover subsection: "For sparse scoped systems of moderate size, one fat node per state can beat sharding; see the campaigns how-to for the crossover criteria."

- [ ] **Step 3: Update `pyphi/mcp/content/campaigns.md` (:75 region)**

Mirror the surface changes in brief: multi-cell `prepare_ces_campaign` semantics (the MCP tool remains single-state; note the library sweep), memory floor, `limit`, order caps. Check `pyphi/mcp/prompts.py:120` walkthrough wording still matches the tool.

- [ ] **Step 4: Build docs and read the output**

Run: `just docs > /tmp/docs.log 2>&1; uv run python -c "print(open('/tmp/docs.log').read()[-2000:])"`
Expected: build succeeds (the known pre-existing `whats-new-in-2.0.md` orphan warning is not a regression; anything new is).

- [ ] **Step 5: Commit**

```bash
git add docs/howto/campaigns.md docs/howto/chtc.md pyphi/mcp/content/campaigns.md pyphi/mcp/prompts.py
git commit -m "Document campaign memory requests, sweep axes, order caps, and the fat-node crossover"
```

---

### Task 10: Changelog, ROADMAP, full verification

**Files:**
- Create: `changelog.d/campaign-limit.feature.md`, `changelog.d/campaign-memory.feature.md`, `changelog.d/campaign-sweep.feature.md`, `changelog.d/scope-order-caps.feature.md`, `changelog.d/campaign-fat-node.doc.md`
- Modify: `ROADMAP.md`

- [ ] **Step 1: Write changelog fragments**

```bash
echo 'Added `limit=` to `pyphi.campaign.prepare_ces`, bounding the planning walk for large scoped systems.' > changelog.d/campaign-limit.feature.md
echo 'Campaign tasks now request memory individually: shard requests are estimated from their largest purview repertoire, packed within memory classes, and floored by `request_memory`.' > changelog.d/campaign-memory.feature.md
echo '`pyphi.campaign.prepare_ces` sweeps substrates × states × subsets × formalisms under one shared scope; multi-cell campaigns collect into a `SweepResult`.' > changelog.d/campaign-sweep.feature.md
echo '`CESScope` accepts `max_purview_order_by_mechanism_order`, an explicit table bounding purview order per mechanism order.' > changelog.d/scope-order-caps.feature.md
echo 'The campaigns how-to describes when one node with native parallelism beats a sharded campaign.' > changelog.d/campaign-fat-node.doc.md
```

- [ ] **Step 2: Update ROADMAP.md**

In the "2026-07-21 scoped-CES campaign follow-ups" section, mark each of the five bullets landed (e.g. prefix `**Landed 2026-MM-DD.**` with the merge reference once known), consistent with how other landed items in the file are annotated — read neighboring sections and match their convention. If a Status Dashboard row references these items, update it in the same change.

- [ ] **Step 3: Run the full test suite (no path argument — doctest sweep included)**

Run: `uv run pytest -q > /tmp/full.log 2>&1; uv run python -c "print(open('/tmp/full.log').read()[-3000:])"`
Expected: read the summary line; all pass, no new failures vs. main's baseline. Also run the slow lane in background per the project pattern if campaign-adjacent slow tests exist: `uv run pytest -m slow --slow -q > /tmp/slow.log 2>&1` (background; read the summary from the file when it finishes).

- [ ] **Step 4: Run linters/type checks**

Run: `uv run ruff check pyphi test && uv run pyright pyphi`
Expected: clean (pre-commit runs these too; fix anything flagged).

- [ ] **Step 5: Commit**

```bash
git add changelog.d/ ROADMAP.md
git commit -m "Add changelog fragments and roadmap status for the campaign follow-ups"
```

---

## Self-review checklist (run after writing, before handoff)

1. **Spec coverage:** limit (Task 1), order caps (Tasks 2–3), memory estimator + stratification + scaffold (Tasks 4–6), multi-system sweep (Tasks 7–8), fat-node note + docs (Task 9), changelog/ROADMAP/verification (Task 10). MCP wrapper: Task 7; MCP content: Task 9.
2. **Known coupling:** Tasks 7 and 8 leave the collect tests red in between — execute them back-to-back.
3. **Verification recipe:** full `uv run pytest` without a path argument in Task 10 (doctest sweep), summaries read from files, never exit codes through pipes.
