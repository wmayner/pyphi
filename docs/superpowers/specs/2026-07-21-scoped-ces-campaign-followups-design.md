# Scoped-CES campaign follow-ups — design

**Date:** 2026-07-21
**Status:** awaiting approval
**Source:** the five ROADMAP items under "2026-07-21 scoped-CES campaign follow-ups —
surfaced by a 21-unit port".

## Background

Porting a 21-unit scoped CES workload onto `pyphi.campaign.prepare_ces` surfaced five
gaps. Two block or degrade real campaigns at that scale (a fixed `request_memory` for
every shard; an unreachable planning `limit`), two are ergonomics (no multi-state sweep
sharing one scope; no order-dependent purview caps), one is documentation (when a fat
node beats sharding). This design covers all five.

Decisions settled during brainstorming:

- Memory: per-shard `request_memory` from an estimator, **plus** memory-stratified
  packing. Memory is not treated as a hard planning budget — a single
  (mechanism, direction, purview) evaluation is atomic, so a budget below the largest
  purview's need is unsatisfiable; the planner requests what each shard needs instead.
- Order-dependent caps: an explicit (mechanism order → max purview order) table on
  `CESScope`. `AxisScope` stays mechanism-blind.
- Multi-system sweep: `prepare_ces` adopts `prepare`'s axis surface (shared
  enumeration helpers, shared result type) but remains a separate entry point. Full
  unification into `prepare(scope=)` was rejected: the compute contracts differ
  (whole-cell arbitrary computes vs. exact intra-cell sharding), and one function would
  carry kwargs whose validity and meaning depend on mode (`units_per_job` as
  cell-packing target vs. shard budget).

## 1. `limit=` on `prepare_ces`

`prepare_ces` gains `limit: int = 100_000_000`, threaded to the planning walk.
The default is 10× the bare `mechanism_workloads` default: `prepare_ces` exists for
large scoped systems, where the walk is expected to be big; the limit still guards
against runaway full-powerset walks.

While threading it, remove the duplicated walk: `prepare_ces` currently runs
`mechanism_workloads` inside `plan_ces_shards` and again for the manifest. Compute the
workloads once in `prepare_ces` and pass them into `plan_ces_shards`.

## 2. Memory-aware shard sizing

### Estimator (`pyphi/cost.py`)

A shard's peak memory is set by the largest repertoire it holds; several are alive at
once during a mechanism-partition sweep. Estimate:

    peak_bytes ≈ REPERTOIRE_FACTOR × 8 × max_repertoire_cells + BASE_BYTES

`max_repertoire_cells` is the largest purview state-space size in the shard — the
product of per-unit state counts, so non-uniform (k-ary) alphabets are handled
exactly, not approximated as `alphabet^|purview|`. `REPERTOIRE_FACTOR` (repertoires
alive per sweep step; start at 4) and `BASE_BYTES` (interpreter + pyphi import +
substrate TPM + task payload; start at 1 GB) are module constants; they cannot be
derived exactly, so the first real campaign validates them against Condor's reported
`MemoryUsage`, and the user-facing knob is the `request_memory` floor (below).
Requests are rounded up to the next 512 MB.

The `mechanism_workloads` walk records `max_repertoire_cells` per mechanism alongside
its units. Its return value becomes a mapping to a small frozen record
(`units`, `max_repertoire_cells`); both consumers (the planner, the manifest) update.

### Planner (`pyphi/campaign/shards.py`)

`ShardSpec` gains a memory estimate. Each candidate item — a whole mechanism at rung 1,
a purview at rung 2, a (mechanism, purview) pair at rung 3 — gets its rounded memory
request computed first; cost-balanced packing then runs **within each memory class**
(items grouped by their rounded request). A 6 GB purview never shares a shard with
100 MB purviews, so small work never queues for big-memory slots.

### Scaffold and submit file

- Submit template: `request_memory = $(memory)`; queue line
  `queue task_id, memory from remaining.txt`.
- `remaining.txt` rows become `<task_id>, <memory>` (e.g. `0, 4GB`).
- Manifest task rows record each task's memory; `status()` regenerates the memory
  column from the manifest when it rewrites `remaining.txt`.
- `prepare` (whole-cell sweeps) writes the same format with its uniform
  `request_memory` value: one template, one `remaining.txt` contract, one `status()`
  path.
- `prepare_ces`'s `request_memory` parameter is redefined as the **floor**: every
  shard requests `max(floor, estimate)`. Default stays `"4GB"` — never less than
  today's behavior, more when the estimate demands it — and a large floor opts out of
  stratification entirely.

## 3. Scoped multi-system sweep

`prepare_ces` adopts the same axis surface as `prepare`:

```python
campaign.prepare_ces(
    substrates,            # one substrate, a sequence, or {label: substrate}
    states=...,            # one state or a sequence of states
    subsets=...,           # as in prepare; default the full substrate
    formalisms=...,        # preset names; default the active version
    scope=...,             # one CESScope shared by every cell
    directory=..., units_per_job=..., ...
)
```

- Cells are enumerated with the existing `pyphi.sweep` helpers
  (`_normalize_substrates`, `_normalize_states`, `_enumerate_cells`) — no bespoke
  axis code in the campaign module.
- The shard plan depends only on (substrate, subset, formalism), not state: cells are
  grouped by that key, planned once per group under the formalism's preset, and shard
  tasks replicated per state. Nine states of one substrate cost one planning pass.
- The one shared scope is resolved per substrate (unit labels normalize to indices
  through each substrate's `node_labels`); tasks carry their resolved scope, as today.
- `CESShardTask` / `SIAShardTask` already carry full cell identity; the substrate
  directory and manifest generalize from the hardcoded `"system"` label to one file
  per label and a cells table mapping tasks to cells.
- `sia=` / `resolution_state=` raise on campaigns with more than one cell — a
  precomputed SIA does not broadcast across cells.
- `collect()`: a single-cell campaign returns the assembled structure, as today. A
  multi-cell campaign returns a `SweepResult` with the same row shape as
  `prepare(compute="ces")` produces, so scoped-exact campaigns and whole-cell sweeps
  converge on one result type.
- `scope_report()` returns one report per cell.

## 4. Order-dependent purview caps

`CESScope` gains one field:

```python
max_purview_order_by_mechanism_order: tuple[tuple[int, int], ...] | None = None
```

An explicit (mechanism order → max purview order) table, applying to both purview
directions on top of the static `AxisScope`s; mechanism orders absent from the table
fall back to the static constraints alone. This expresses order-tied rules (e.g.
purview order ≤ 2·order + 1) exactly, stays callable-free named data, and keeps
`AxisScope` mechanism-blind.

One method is the sole point of truth:

```python
CESScope.purview_axis(direction, mechanism) -> AxisScope
```

returning the static axis intersected with the table's cap for `len(mechanism)`.
Every purview-selection site — the planner rungs, `mechanism_workloads`, and the shard
execution path — goes through it, so planning and execution cannot disagree about
what is in scope.

Validation in `__post_init__`: positive orders, unique keys. Serialization:
`CESScopeSchema` gains the field with a `None` default; `serialize/convert.py`
updated.

## 5. Fat-node crossover note (documentation)

A short section in `docs/howto/campaigns.md` and `docs/howto/chtc.md` (and the MCP
copy in `pyphi/mcp/content/campaigns.md`): for sparse, scoped, mid-n systems
(≈21 units, mechanism order ≤5), one fat node per state — native `parallel` with
large `request_cpus` / `request_memory` — beats sharding when shard count ×
scheduling overhead dominates and the per-shard memory floor is near the
whole-analysis footprint anyway. With per-shard requests now accurate, the crossover
criterion is scheduling overhead, not memory holds.

## Cross-cutting

- **MCP surface:** `prepare_ces_campaign` tool and the `campaign_walkthrough` prompt
  updated for the new signature (axes, `limit`, memory floor semantics).
- **Tests:** estimator units (k-ary alphabets, floor, rounding); stratified-packing
  invariants (no shard mixes memory classes; balance within class);
  `remaining.txt` format and `status()` rewrite round-trip (memory column survives);
  limit threading (a too-small limit raises from `prepare_ces`); order-cap agreement
  (planner, cost walk, and execution select identical purview sets); multi-cell
  end-to-end (prepare → run tasks locally → collect equals the local scoped result
  per cell, tie sets preserved).
- **Changelog:** one fragment per item in `changelog.d/`.
- **ROADMAP:** the five items marked landed in the same change.

## Implementation order

1. `limit=` threading + single planning walk (smallest, unblocks planning).
2. Order-dependent purview caps (touches scope + every selection site; later items
   build on the selection helper).
3. Memory estimator, stratified packing, scaffold format (planner + scaffold).
4. Multi-system sweep (builds on all of the above).
5. Documentation (fat-node note, howto updates for the new surface).
