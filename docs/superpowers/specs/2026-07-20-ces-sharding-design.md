# HTCondor campaign surface — cycle 2: scoped CES sharding + reconstruction

**Date:** 2026-07-20
**Status:** Draft for review
**Companion to:** `2026-07-20-htcondor-campaign-design.md` (cycle 1:
infrastructure + sweep cells). This spec assumes cycle 1's directory format,
runner, submit-file emitter, status/collect skeleton, and packing machinery.

## Context

The driving use case for campaigns is distributing **one system's** analysis:
unfolding its cause-effect structure (and its SIA) across many independent
condor jobs. The combinatorics are dominated by mechanism-partition sweeps
(~10⁹·⁷ evaluations for the all-units mechanism at n=8 under
`JOINT_PARTITION_ALL`; ~10⁴³ at n=21), so a usable design needs three things
working together:

1. **Declarative scope** — the user states the combinatorially feasible
   surface per axis; everything excluded is covered by certificates.
2. **Automatic sharding** — PyPhi plans the split of the scoped computation
   across jobs, descending to finer granularity only where estimates demand
   it.
3. **Exact reconstruction** — merges are tie-preserving and lossless; the
   collected structure is identical to what an (infeasible) single-machine
   run would produce over the same scope.

### Scope vs. shards — the central distinction

**Scope changes what is computed.** It selects mechanisms and purviews.
Exclusions are explicit, recorded in provenance, and certified by bounds.

**Sharding never changes the result.** It splits an exact computation and
merges back losslessly.

Mechanisms and purviews are both scopable and shardable. Partition sweeps
(mechanism partitions and system partitions) are **shard-only**: a partial
partition sweep would silently turn φ into an upper bound, so scope does not
reach them. (Deliberate partition subsetting as an approximation is a
possible later extension, not part of this design.)

## Goals

1. `AxisScope`/`CESScope`: serializable, named constraint forms combined by
   intersection; no lambdas.
2. `estimate_analysis(scope=…)`: the counting walk restricted to a scope —
   standalone pre-flight and the planner's cost model.
3. A shard planner in `prepare()` that descends mechanism → purview-range →
   partition-stride only where the per-job budget requires.
4. `ces_shard` and `sia_shard` task kinds executing through existing seams
   (`distinction(cause_purviews=…, effect_purviews=…)`,
   `find_mip(partitions=…)`, the formalism's system-partition evaluation).
5. Tie-preserving merges in `collect()`, reconstruction through the existing
   `ces(sia=…, distinctions=…)` assembly, and a scope report with certified
   bounds for what scope excluded.

## Non-goals

- Partition-subset approximation (scope on partition axes).
- Cross-job communication of any kind (shards cannot cancel each other on a
  reducibility verdict; bottleneck-first ordering is the mitigation).
- Work-unit ↔ wall-time calibration (unchanged from cycle 1).
- IIT 3.0 CES campaigns (the sweep-cell task type already covers IIT 3.0
  cells; sharding targets the IIT 4.0 path; extending the merge machinery to
  the IIT 3.0 concept path is future work).
- DAGMan or any submit-side orchestration beyond the flat job list.

## Verified mechanics (confirmed by running code on main, 2026-07-20)

- **Seams exist at every level.** `find_mice`/`distinction` accept explicit
  purview lists; `find_mip` accepts an explicit partition iterable
  (`pyphi/formalism/queries.py`). `all_distinctions` enumerates the
  mechanism powerset in one place. `ces()` accepts precomputed `sia=` and
  `distinctions=` and performs congruence resolution + relations + bounds
  itself (`pyphi/formalism/iit4/__init__.py`) — collect() assembles through
  this path, not a reimplementation.
- **Tie sets survive serialization.** MICE/RIA save→load preserves partition
  ties (including a genuine 2-element tie on the XOR system and
  self-only tie sets on the basic system), state ties, and tri-state purview
  ties (`pyphi/serialize/convert.py` encodes peers explicitly).
- **Bottleneck-first ordering works.** On a sparse 4-unit chain
  (0→1→2→3), 3 of 10 mechanism partitions for a far-apart
  (mechanism, purview) pair cut zero present connections; sorting the
  enumeration by present-connections-cut ascending puts a φ=0 partition at
  position 0 (default order: position 3), and zero-cut partitions evaluate
  to exactly φ = 0.
- **Partial-CES relations are exact.** Relations depend only on their
  relata: on fig4, a 3-of-4 distinction subset gives analytical == concrete
  == the full structure restricted to the subset. Partial structures are
  exact substructures with certified Σφ_r lower bounds.
- **Congruence needs only a resolution state at collect time.** Distinction
  computation is SIA-independent; `resolve_congruence` runs at assembly.
  When no SIA exists, `ces()` already falls back to the system's
  intrinsic-information state.

## Design

### 1. Scope declaration

```python
@dataclass(frozen=True)
class AxisScope:
    explicit: tuple[tuple[int, ...], ...] | None = None
    min_order: int | None = None
    max_order: int | None = None
    containing: tuple[int, ...] | None = None   # must include all these units
    within: tuple[int, ...] | None = None       # must be a subset of these units

@dataclass(frozen=True)
class CESScope:
    mechanisms: AxisScope = AxisScope()
    cause_purviews: AxisScope = AxisScope()
    effect_purviews: AxisScope = AxisScope()
```

- Constraint fields combine by **intersection**. `explicit` is exclusive:
  combining it with any other field is a `ValueError` (an explicit list *is*
  the axis).
- The default (all fields `None`) is unconstrained — full-scope campaigns
  are just `CESScope()`.
- Unit arguments accept labels or indices (normalized at construction, like
  the rest of the API).
- Scopes are serializable (`pyphi.serialize` schema, tag `axis_scope` /
  `ces_scope`), ship inside task files, and land in the manifest and in
  result provenance.
- Purview scopes **intersect** the connectivity-pruned potential purviews;
  they can only narrow, never add.

### 2. Scope-aware estimation

`estimate_analysis` gains `scope: CESScope | None = None`. The counting walk
filters mechanisms and purviews through the scope before counting, using the
same per-(m,p) partition-count memos. The planner and the standalone
pre-flight use the identical code path, so "what the estimate priced" and
"what the campaign runs" cannot drift.

A new per-mechanism breakdown (mechanism → summed scoped work units) is
computed for the planner and recorded in the campaign manifest — this is the
raw data behind packing decisions.

### 3. The shard-planning ladder

Entry point (extending cycle 1's `prepare`):

```python
prepare(substrate, *, kind="ces", state, subset=None, scope=CESScope(),
        directory, units_per_job, ordering=None, sia=None,
        resolution_state=None, …)   # cycle-1 condor/packing kwargs unchanged
```

Input to the planner: the scope, the per-job budget (`units_per_job`, as in
cycle 1), and the per-mechanism estimates. It descends only where needed:

1. **Mechanism tasks.** Mechanisms whose scoped estimate fits the budget are
   cost-balance-packed into tasks (`cost_balanced_partition`, as cycle 1).
   Each computes its full scoped MIC + MIE via
   `distinction(cs, mechanism, cause_purviews=…, effect_purviews=…)`.
2. **Purview-range tasks.** A mechanism over budget splits its scoped
   (direction, purview) list into cost-balanced ranges (weights = per-(m,p)
   partition counts). Each task runs complete partition sweeps for its
   purviews via `find_mip` and outputs per-purview RIAs with tie sets.
3. **Partition-stride tasks.** A single (mechanism, direction, purview) over
   budget splits its partition enumeration into k interleaved strides —
   shard i evaluates partitions i, i+k, i+2k, … via `islice(enumeration, i,
   None, k)`, never materializing the sequence. Interleaving balances any
   systematic cost trend along the enumeration. Stride tasks call
   `find_mip(partitions=<stride>)`.

`sia_shard` tasks are rung 3 applied to the system-partition enumeration
(same stride math, same merge). When a precomputed SIA or resolution state
is supplied to `prepare()`, no SIA tasks are planned.

The plan (which mechanisms per task, which ranges, which strides) is
deterministic for fixed inputs and recorded in the manifest. Stride
correctness depends on enumeration order, which is deterministic for a fixed
PyPhi version and partition scheme; the manifest already records both, and
`collect()` refuses to merge outputs whose recorded version/scheme disagree
with the manifest.

### 4. Task kinds and shard outputs

Cycle 1's task schema gains a `kind` discriminator; cycle 2 adds:

- **`ces_shard`**: the System reference (substrate label + state + node
  subset), the scope, and one of three payloads — a mechanism list, a
  (mechanism, direction, purview-range) triple, or a (mechanism, direction,
  purview, stride (i, k)) quadruple.
- **`sia_shard`**: the System reference and a stride (i, k) over system
  partitions.

Outputs (all `pyphi.serialize` documents, embedding existing registered
types):

- Mechanism tasks → serialized distinctions (which carry their MICE, RIAs,
  and complete tie sets).
- Purview-range tasks → per-purview RIA winners with partition/state tie
  sets, plus each purview's best and second-best values for margin
  reconstruction.
- Partition-stride tasks → the stride's winning RIA tie set **keyed by
  specified-state pin**, plus best and second-best normalized φ per pin.
- SIA-stride tasks → the stride's minimal system-partition evaluation with
  tie set, plus best and second-best for the margin.

Every shard output additionally records the runner's PyPhi version and the
active partition scheme — the data behind `collect()`'s stride-consistency
guard.

### 5. Merge semantics (exact, tie-preserving)

- **Partition merge (min).** Concatenate the shard tie sets per state pin
  and re-resolve through `pyphi.resolve_ties`. Exactness argument: the
  global minimum is ≤ every shard minimum, so any candidate within tolerance
  of the global minimum is within tolerance of its own shard's minimum and
  therefore present in that shard's tie set. State-pin resolution re-runs on
  the merged per-pin winners using the same machinery the single-machine
  path uses.
- **Purview merge (max).** Dual argument, same machinery
  (`resolve_ties.purviews` over the union of shard MICE candidates).
- **Margins.** Derivable from per-shard best/second-best when every shard
  swept exhaustively (no short-circuit); otherwise `None` — the existing
  truncated-sweep rule, extended to merges.
- **Short-circuiting.** Shards short-circuit internally on a zero-φ
  partition (existing `shortcircuit_sia` config). A shard that
  short-circuits reports it; the merged φ is 0 regardless of other shards.
- **Ordering.** `ordering="bottleneck_first"` (prepare option, default off)
  sorts each shard's partition slice by present-connections-cut ascending,
  so sparse-substrate reducibility surfaces in the first evaluations.
  Ordering never affects results (min is order-independent; tie resolution
  runs on the collected set) — only time-to-short-circuit.

Merging happens in `collect()`, bottom-up: partition strides → per-purview
RIAs → purview ranges → MICE → distinctions → `UnresolvedDistinctions`.
Every merge stage consumes and produces the same types the single-machine
code path uses, so merged objects are indistinguishable from locally
computed ones.

### 6. Collect and reconstruction

```python
collect(directory, partial=False, sia=None, resolution_state=None)
```

`collect()`'s return type follows the campaign's kind from the manifest:
`SweepResult` for `sweep_cells` campaigns (cycle 1), `CauseEffectStructure`
for CES campaigns. For CES campaigns:

1. Merge `sia_shard` outputs into the SIA (mode 1), or use the supplied
   `sia=` (mode 2), or neither (mode 3).
2. Merge `ces_shard` outputs into `UnresolvedDistinctions`.
3. Assemble via the existing `ces(sia=…, distinctions=…)` path: congruence
   resolution (mode 3 uses `resolution_state=` if given, else `ces()`'s
   intrinsic-information fallback), relations (analytical default),
   existing bounds validation. Mode 3 results carry no Φₛ.
4. Attach the **scope report**: per-axis exclusion summary (which constraint
   forms, how many mechanisms/purviews excluded), the computed Σφ_r (an
   exact lower bound for the full structure's Σφ_r, since partial structures
   are exact substructures), and the measured upper-bound certificates from
   `iit4/bounds.py` (`sum_phi_relations_measured_bound`,
   `big_phi_measured_bound`). The report is part of the result's provenance.

**Partial collect** (missing/failed tasks): a distinction is reconstructable
only when its entire shard group is present. `partial=True` returns the
structure over reconstructable distinctions (with the scope report counting
the missing groups separately from scope exclusions); default raises with a
per-group summary. Cycle 1's `status()`/`remaining.txt` resubmission flow is
unchanged.

## Error handling

Inherits cycle 1's task-level model (atomic outputs, per-cell/per-item error
entries, attempt renaming, resubmit via `remaining.txt`). Cycle-2 specifics:

| Failure | Where caught | Behavior |
|---|---|---|
| `explicit` combined with other constraint fields | `AxisScope` | `ValueError` at construction |
| Scope selects zero mechanisms | `prepare` | error (empty campaign is never intended) |
| Purview scope empties a mechanism's purviews | `prepare` | that mechanism yields a null MICE by the existing no-purviews path; counted in the scope report |
| Manifest version/scheme ≠ output's recorded version/scheme | `collect` | error naming the mismatch (stride semantics depend on enumeration order) |
| Missing shard in a group | `collect` | group unreconstructable: raise, or omit under `partial=True` |
| Budget below the cost of a single partition evaluation | `prepare` | the stride count is capped at the pair's partition count (one partition per shard is the floor); a warning notes the budget is unreachable for that pair |

## Testing

Headline invariant — **sharded ≡ unsharded**: a full-scope CES campaign on a
small system, forced through all three ladder rungs by a tiny
`units_per_job`, executed via subprocess, collects to exactly
`system.ces()`: φ values, distinction set, relation set, tie sets, margins.
Run under both `IIT_4_0_2026` and `IIT_4_0_2023` presets.

Around it:

- **Scope forms:** each constraint form and their intersections select the
  documented mechanism/purview sets; explicit-exclusivity `ValueError`;
  label normalization; scope serialization roundtrip.
- **Scoped equivalence:** a scoped campaign equals a locally computed
  reference (`distinction()` over the scoped mechanisms with scoped purview
  lists) — including the scope report's Σφ_r lower bound matching the
  computed relations, and its measured upper bounds matching direct
  `bounds.py` calls on fig4's known values.
- **Estimate/plan consistency:** `estimate_analysis(scope=…)` totals equal
  the sum of per-mechanism breakdowns; the planner's task weights equal the
  estimates; plan determinism for fixed inputs.
- **Stride enumeration:** strides are disjoint, their union is the full
  enumeration, order is deterministic; the version/scheme mismatch guard
  fires.
- **Merge units:** constructed candidate sets with ties exactly at the
  tolerance boundary — partition min-merge, purview max-merge, state-pin
  keying, margin derivation (and `None` when a shard short-circuited);
  merged objects pass the same invariants as locally computed ones.
- **Bottleneck-first:** on the sparse-chain specimen, the ordered sweep
  short-circuits within the zero-cut prefix; results identical with
  ordering on and off.
- **SIA modes:** mode 1 merged SIA equals a local `system.sia()`; mode 2
  with a supplied SIA plans no SIA tasks; mode 3 resolves against the
  intrinsic-information state and carries no Φₛ (matching `ces()`'s
  existing fallback semantics).
- **Partial collect:** delete one stride output → its distinction's group
  is reported missing; `partial=True` returns the remainder; the scope
  report distinguishes missing-vs-excluded.

## Deliverables

- `pyphi/campaign/scope.py`: `AxisScope`, `CESScope`.
- `pyphi/campaign/shards.py`: planner (ladder), stride enumeration,
  bottleneck-first ordering.
- `pyphi/campaign/merge.py`: tie-preserving merges, group bookkeeping.
- `pyphi/campaign/__init__.py`: `prepare(kind="ces", system=…, scope=…,
  sia=…)` entry, runner dispatch for the new task kinds, `collect()`
  assembly + scope report.
- `pyphi/cost.py`: `scope=` parameter, per-mechanism breakdown.
- `pyphi/serialize`: schemas for scopes, new task kinds, shard outputs.
- Tests: `test/campaign/` (mirrors the package), additions to
  `test/test_cost.py` and the `test/mcp` suite.
- Docs: `docs/howto/campaigns.md` (the dedicated how-to page from cycle 1)
  gains scope-declaration and CES-campaign sections, including reading the
  scope report and its certificates.
- MCP server: `estimate_cost` gains scope parameters (mirroring
  `estimate_analysis(scope=…)`); `prepare_campaign` gains
  `kind="ces"`, the system reference, scope, and SIA-mode arguments;
  `collect_campaign` registers the collected `CauseEffectStructure` as a
  result handle; `pyphi/mcp/content/campaigns.md` covers scope + shard
  concepts.
- Changelog fragments; ROADMAP P11 row update on merge.

## Accepted simplifications

- No cross-shard short-circuit propagation: a reducibility verdict in one
  shard does not stop sibling jobs. Mitigations: bottleneck-first ordering,
  and the user can `condor_rm` remaining jobs after an early `status()`.
- Margins are dropped (`None`) whenever any contributing shard
  short-circuited, even if the margin were partially derivable.
- The scope report certifies Σφ_r and Φ bounds; it does not attempt
  per-distinction φ bounds for excluded mechanisms.
- Purview-range and stride tasks recompute the repertoire for their
  (mechanism, purview) pair per task (no cross-task repertoire cache);
  repertoire cost is negligible against the partition sweeps that trigger
  those rungs.
