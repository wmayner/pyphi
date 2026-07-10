# Grain-search cost pre-flight — design

**Date:** 2026-07-10
**Status:** Approved design, pending implementation plan
**Depends on:** the bounded intrinsic-unit search (`pyphi/macro/search.py`)
**Relates to:** the grain-discovery exploration (`2026-07-07-grain-discovery.md` §2, the verified cost anatomy)

## Goal

Let users see the size and shape of a grain search before running it:
`SearchBounds.estimate(substrate)` returns a `SearchEstimate` with candidate
counts per axis plus the cost-relevant structure (candidates bucketed by
macro unit count, distinct construction keys), computed by pure counting —
no TPM construction, no φₛ evaluation.

## Background and constraints

- The search is **adaptive**: each level's unit pool depends on which
  candidate decompositions pass the intrinsic-unit judgment, so the exact
  total evaluation count is not computable a priori. What is analytic:
  level-1 judgment counts are **exact** (the micro pool is known), and
  everything deeper — plus the final assembly sweep — has an **exact worst
  case** under the all-pass assumption (every judged decomposition passes
  and spawns all its mapped and grained variants). The worst case is
  precisely the blow-up a pre-flight exists to warn about.
- Per-evaluation cost is driven by two knowables (exploration doc §2.2–2.3,
  measured): macro-TPM construction is Θ(τ·4ⁿ) per distinct
  (footprint, grain) key, and the SIA partition sweep grows with the macro
  unit count m (`DIRECTED_SET_PARTITION`: m = 3 → 22, m = 4 → 150,
  m = 5 → 1,061 partitions). The estimate reports counts and these
  structural weights; it does **not** predict wall time (machine- and
  config-dependent).
- **Fidelity ruling (Will, 2026-07-10): counts + weights.** Structured
  counts per axis plus m-bucket and construction-key structure; no seconds.
- The counting must not drift from the search implementation. The
  enumeration functions in `pyphi/macro/search.py` — `_decompositions`,
  `_apportionments`, `candidate_mappings`, `_assemble_systems`,
  `_variants`' multiplicity structure — embody many implementation details
  (canonicalization, complement dedup, `seen` keys, min-size rules,
  single-constituent grain floors) that closed-form formulas would get
  wrong.

## Design

### 1. Approach: a counting walk through the real enumerators

New module **`pyphi/macro/estimate.py`**. It mirrors the control flow of
`_derive_units` and `complexes()` but drives the real enumeration functions
with featherweight stand-in units instead of `MacroUnit`s, and counts
instead of evaluating.

- A stand-in unit carries exactly the attributes the enumerators touch:
  `micro_constituents`, `background_apportionment`, `micro_grain` — plus a
  `multiplicity` (how many mapped/grained variants it stands for). One
  stand-in per (footprint, grain, apportionment) shape represents all of
  that shape's variants; multiplicity is computed with the real
  `candidate_mappings` summed over update grains exactly as `_variants`
  does (including the single-constituent grain-2 floor).
- The walk replicates `_derive_units`' level/size-class loops: footprints ×
  `_decompositions` (over the stand-in pool) × `_apportionments`, with the
  same `seen`-key dedup. Each candidate contributes its judgment (1 own
  system + competitor assemblies via `_assemble_systems` over the
  strict-subset pool members, exactly as `_class_combos` does); each
  candidate assumed valid contributes its variant multiplicity to the next
  pool.
- Distinct-system counting matches the memo semantics: unit-combos are
  keyed by their stand-in composition and variant multiplicities multiply
  through, so `distinct_systems_upper_bound` counts what the real memo
  would hold in the all-pass world.
- The final assembly sweep is counted with `_assemble_systems` over the
  worst-case pool (stand-ins), multiplying member multiplicities.

Because degenerate mapping collisions can only shrink the real variant
count relative to the multiplicity model, and pruning can only shrink the
pool, every reported quantity beyond the exact prefix is a true upper
bound.

### 2. `SearchEstimate`

Frozen dataclass in `pyphi/macro/estimate.py`, `Displayable` +
`ToPandasMixin`:

- `judgments_by_level : tuple[int, ...]` — candidate decompositions judged
  per macroing level (index 0 = first level above micro). Level 1 exact;
  deeper levels worst-case.
- `worst_case_pool_by_level : tuple[int, ...]` — pool size (units,
  including mapped/grained variants) after each level, all-pass.
- `assemblies_upper_bound : int` — final sweep candidate systems (Eq. 18
  disjoint unit sets over the worst-case pool).
- `distinct_systems_upper_bound : int` — deduplicated unit-combos across
  judgment candidates, judgment competitors, and the final sweep; the
  headline number (compare `len(ComplexesResult.records)`).
- `systems_by_unit_count : dict[int, int]` — final-sweep candidates
  bucketed by macro unit count m (the SIA-cost axis).
- `partitions_by_unit_count : dict[int, int]` — partitions per SIA at each
  m present in the buckets, counted live under the current
  `system_partition_scheme` for m up to a small cap (6; the m = 6 count
  is 7,896 and enumerates in ~0.25 s, while m = 8 is ~510k and ~16 s) and
  memoized at module scope per (scheme, m); buckets above the cap omit
  their partition count and set `partitions_capped`.
- `partition_sweeps_upper_bound : int` — Σ over m of
  `systems_by_unit_count[m] × partitions_by_unit_count[m]` (buckets above
  the partition-count cap excluded and flagged via `partitions_capped`;
  the display renders that row with "≥" when capped).
- `partitions_capped : bool` — some bucket's m exceeds the
  partition-count cap. Independent of `truncated` and `is_exact`: an
  exact enumeration at `max_depth=0` stays exact even when its largest
  buckets have no partition count.
- `construction_keys_upper_bound : int` — distinct (footprint, grain)
  construction keys (the Θ(τ·4ⁿ) axis).
- `is_exact : bool` — `True` iff the candidate enumeration is exact
  rather than an upper bound: `max_depth == 0`, where no judgment happens
  and the sweep over micro units is fully determined. (Level-1 judgment
  counts are exact regardless; adaptivity begins with the level-1
  verdicts.) Even an exact enumeration can exceed
  `len(ComplexesResult.records)`: a candidate system whose state is
  unreachable under its own TPM is discarded at run time
  (`StateUnreachableError` → no record), and reachability cannot be
  predicted without constructing the TPM — which the pre-flight never
  does. Counts are therefore bounds on candidates *enumerated*, met with
  equality by records only when every candidate's state is reachable.
- `truncated : bool` — the counting walk hit its `limit`; all counts are
  then lower bounds of the upper bound ("at least"). The limit bounds the
  *reported counts*, checked at enumeration-phase granularity — one
  `_assemble_systems` call completes before the check, so wall time is
  not strictly bounded. A partial macroing level's judgment and pool
  tallies are flushed so the report stays coherent.

Display card (B21 conventions): headline rows (distinct systems ≤,
assemblies ≤, partition sweeps ≤, construction keys ≤, exact-through
/ truncated), a per-level table (judgments, pool), and the m-bucket table
(m, systems, partitions each). `_pandas_record` carries the scalar fields.

### 3. API

```python
class SearchBounds:
    def estimate(self, substrate, limit: int = 1_000_000) -> SearchEstimate: ...
```

- `substrate` is used only for its size (`substrate.size`); the counting
  never touches its TPM.
- `limit` caps the total items the counting walk may enumerate; on hitting
  it the walk stops and returns a `SearchEstimate` with `truncated=True`.
- `mappings="EXHAUSTIVE"` above `exhaustive_cap` raises a `ValueError`.
  The search itself raises only when such a decomposition passes
  judgment; the estimate assumes every decomposition passes, so it raises
  unconditionally — the conservative pre-flight reading.
- The estimate requires IIT 4.0, exactly as the search drivers do
  (`ValueError` under IIT 3.0) — a search that would refuse to run gets a
  pre-flight that refuses identically rather than weights from the wrong
  partition scheme.
- The method delegates to `pyphi.macro.estimate.estimate_search(bounds,
  n, limit=...)`, the testable core.

### 4. Non-goals

- No wall-time prediction and no calibration runs.
- No driver `describe()` — the `SearchEstimate` display card is the
  description.
- No retrospective statistics on completed `ComplexesResult`s.
- No estimate for the other drivers (`intrinsic_units`, `valid_systems`);
  `complexes` is the expensive front door. Extending later reuses the same
  walk.

## Testing

- **Fixture invariants** (`test/macro/`): on the min-substrate (defaults
  and `mappings="EXHAUSTIVE"`), bu-substrate, and decaying-chain
  fixtures, run the real `complexes()` and assert
  `estimate.distinct_systems_upper_bound >= len(result.records)`.
  **Equality pins** where verified live: min-substrate defaults
  (8 = 8 — the worst case is achieved because its only candidate
  decomposition passes), min EXHAUSTIVE (10 = 10), min at `max_depth=0`
  (3 = 3, `is_exact`). The bu-substrate at `max_depth=0` pins the
  unreachable-state gap: estimate 7, records 6.
- **Primitive agreement**: the stand-in walk's level-1 judgment count
  equals a direct enumeration of footprints × `_decompositions` ×
  `_apportionments` on hand-built micro pools; variant multiplicities
  equal `len(_variants(V, W, bounds))` for representative shapes.
- **Partition counts pinned**: `partitions_by_unit_count` reproduces the
  measured values under `DIRECTED_SET_PARTITION` (m = 3 → 22, m = 4 → 150,
  m = 5 → 1,061), verified live.
- **Truncation**: a tiny `limit` yields `truncated=True` and monotonically
  smaller-or-equal counts.
- **Error surfacing**: EXHAUSTIVE above `exhaustive_cap` raises
  `ValueError` from `estimate` without touching the substrate TPM.
- Full verification: `uv run pytest` with no path argument.
