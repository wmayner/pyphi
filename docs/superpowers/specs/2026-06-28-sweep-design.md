# `pyphi.sweep` — Cartesian batch driver — Design

**Status:** proposed
**Date:** 2026-06-28
**Roadmap item:** B12 (Wave 6, pulled into 2.0).

## Goal

Run the same IIT computation across a cartesian product of axes — states,
candidate subsystems (node subsets), and formalisms — and collect the results
into one tidy long-format pandas DataFrame, keeping the raw result objects
alongside and a per-row reproducibility record (the `ConfigSnapshot` each
result already carries). This replaces the hand-rolled nested-loop-plus-manual-
DataFrame ritual researchers write today with a single, reproducible call.

It adds no new IIT mathematics. It is an orchestration and tabulation layer
over existing entry points (`System.sia()` / `System.ces()`), enumerators
(`utils.all_states`, `utils.powerset`), the formalism presets, the
`ConfigSnapshot` reproducibility record, and the parallel `map_reduce`
infrastructure.

## Background and building blocks

All pieces this composes already exist:

- **Compute entry points** — `System(substrate, state, node_indices=…)`
  (`pyphi/system.py`); `System.sia(**kwargs)` returns a
  `SystemIrreducibilityAnalysis` (IIT 4.0: `pyphi/formalism/iit4/__init__.py:151`)
  carrying `.phi` and `.normalized_phi` but **not** distinctions/relations;
  `System.ces(**kwargs)` returns a `CauseEffectStructure`
  (`pyphi/models/ces.py:59`) carrying `.distinctions` and `.relations` but
  **not** the system Φ. Both SIA types conform to `SIAInterface`
  (`pyphi/models/protocols.py:19`, `.phi` / `.normalized_phi`).
- **Axis enumerators** — `utils.all_states(spec)` (states, little-endian),
  `utils.powerset(iterable, nonempty=True)` (node subsets). Formalisms switch
  via `config.override(**presets[name])` with presets `iit3` / `iit4_2023` /
  `iit4_2026` (`pyphi/conf/presets.py`).
- **Reproducibility** — every top-level result already carries a sibling
  `config` (`ConfigSnapshot`, `pyphi/conf/snapshot.py`) and `provenance`
  (`pyphi/provenance.py`, with an optional `seed`). IIT compute is **fully
  deterministic** (no internal RNG), so a sweep is reproducible by
  construction; `seed` is a caller-supplied bookkeeping label only.
- **Parallelism** — `pyphi.parallel` exposes `map_reduce` over a `Scheduler`
  Protocol (`pyphi/parallel/scheduler.py:55`) that **propagates a
  `config_snapshot` into workers**, with `ChunkingPolicy` / `ProgressPolicy`.
  The default backend under `config.infrastructure.parallel` is the process
  pool (`LocalProcessScheduler`).
- **Labeled export** — `ToPandasMixin` (`pyphi/models/pandas.py:120`) is
  per-distinction; there is **no** scalar `to_pandas` for a SIA or a CES today.
  The one genuinely new piece here is a small per-cell row extractor.

## Public API

```python
def sweep(
    substrate,
    *,
    states,                       # required: a state, an iterable of states, or "all"
    subsets="full",               # "full" | "all" | iterable of node-index tuples
    formalisms=None,              # None (current) | iterable of version names
    compute="sia",                # "sia" | "phi_structure" | callable
    parallel=None,                # None -> follow config; True/False to force
    progress=None,                # None -> follow config; True/False to force
    seed=None,                    # stamped into each result's provenance
) -> SweepResult: ...
```

Exposed as `pyphi.sweep`. The cartesian product of the three axes
(`formalisms` × `subsets` × `states`) defines the cells; each cell is one row.

**Axes.**
- `states` (required) — a single state tuple, an iterable of state tuples, or
  `"all"` (every state of the substrate via `utils.all_states`). Required
  because a `System` needs a state and a substrate has no "current" state.
- `subsets` — which node subset is the candidate system
  (`System(substrate, state, node_indices=subset)`; the complement is
  background). `"full"` (default — the whole system, one value), `"all"`
  (`utils.powerset(range(n), nonempty=True)`), or an explicit iterable of
  index tuples.
- `formalisms` — formalism version names (`"IIT_3_0"`, `"IIT_4_0_2023"`,
  `"IIT_4_0_2026"`). `None` (default) means the single currently-active
  formalism; otherwise each named formalism is a value of the axis, run under
  `config.override(**presets[name])`.

## `SweepResult` and the row schema

`SweepResult` is a small frozen value object:

- `.df` — a tidy long-format `DataFrame`, **one row per cell**. The index is a
  `MultiIndex` over exactly the axes that vary (more than one value), in order
  `(formalism, subset, state)`; if only one axis varies it is a single index;
  one cell is a `RangeIndex`. Axes that do **not** vary are recorded as scalar
  columns (so no information is lost). Metric columns follow from `compute`
  (below).
- `.results` — a list of the raw per-cell results, aligned 1:1 and in the same
  order as `.df` rows (`.results[i]` ↔ `.df.iloc[i]`), for drill-down. Nothing
  is discarded. Each result carries its own `.config` snapshot and
  `.provenance` (with the `seed` if supplied), so per-row reproducibility lives
  on the objects, not as DataFrame cells.
- `.to_pandas()` returns `.df` (so a `SweepResult` itself follows the
  to_pandas convention).

**Metric columns by `compute`:**

- `compute="sia"` (default) — `phi`, `normalized_phi`, `is_irreducible`
  (`utils.is_positive(phi)`). The raw result is the `SystemIrreducibilityAnalysis`.
- `compute="phi_structure"` — the cell runs **both** `sia()` and `ces()`:
  `phi` (from the SIA), `n_distinctions` (`len(ces.distinctions)`),
  `sum_phi_r` (summed relation φ; `0`/NaN under IIT 3.0, which has no
  relations). The raw result is a small `PhiStructureResult(sia, ces)` pairing.
- `compute=<callable>` — a function taking a `System` and returning a result;
  the row is the result's scalar `to_pandas()` record if it exposes one, else
  `{"phi": getattr(result, "phi", None)}`. The raw result is whatever the
  callable returns.

A small row-extractor module maps each built-in `compute` value to its column
set; it is the only genuinely new logic. Mixed-formalism sweeps where a column
applies to only some rows (e.g. `normalized_phi` absent under IIT 3.0) leave
`NaN` there — ordinary tidy behavior.

## Execution model

A sweep is an embarrassingly-parallel workload (independent cells), so the
efficiency is in the **outer** loop. The inner `sia()`/`ces()` already
parallelizes internally; doing both would oversubscribe cores. So the rule is
**one level of parallelism, at the cell grain**:

- **Parallel (default when `config.infrastructure.parallel`, or `parallel=True`).**
  Cells are **grouped by formalism**. For each formalism, one `map_reduce`
  runs its `(subset, state)` cells with `config_snapshot` set to that
  formalism's snapshot **with inner parallelism disabled** (so each worker's
  `sia()`/`ces()` runs sequentially — no oversubscription). Per-formalism
  batching means exactly one formalism is installed per batch, which also
  sidesteps any `config.override` race: cells within a batch differ only by
  their `(subset, state)` arguments, never by config. The worker returns the
  row record plus the raw result; the main process assembles the DataFrame.
- **Sequential (`parallel=False`, or `config.parallel` off).** A plain loop:
  `with config.override(**preset): result = compute(System(substrate, state,
  node_indices=subset))`. The inner compute uses config as-is (so a user who
  wants inner parallelism on a few-large-cell sweep sets `parallel=False` and
  leaves `config.parallel` on).
- **Progress** — an optional bar over cells (`ProgressPolicy`), `None` follows
  `config`.

The few-large-cells regime (a handful of n≥7 systems, where one cell can
saturate all cores) is served by `parallel=False`, which lets the inner
compute use the cores instead. The common many-small-cells sweep gets
near-linear outer speedup.

## Reproducibility, errors, scope

- **Seed** — optional; when given, each result is stamped via
  `result.with_provenance(seed=seed)`. Because compute is deterministic this is
  a bookkeeping label, meaningful only if a future version samples states
  rather than enumerating them.
- **Errors** — fail-loud. A cell that raises (e.g. an invalid state under a
  formalism's validation) aborts the whole sweep, so a partial table never
  passes for a complete one. An `on_error="skip"` mode is a noted follow-on,
  not in v1.
- **Out of scope (v1):** sampling axes (random subsets of states); the
  actual-causation `account` as a `compute` preset (a different entry point;
  reachable via a callable); nested/adaptive parallelism that splits cores
  between outer and inner; saving a sweep to disk (the result types already
  serialize individually via `pyphi.serialize`).

## Files

- `pyphi/sweep.py` — new: the `sweep()` function, `SweepResult`, the per-cell
  row extractors, the cell-enumeration + per-formalism batching, and the
  `map_reduce` wiring.
- `pyphi/__init__.py` — export `sweep` (and `SweepResult`); add to `__all__`.
- `test/test_sweep.py` — the cases below.
- `changelog.d/sweep.feature.md`.

## Testing

- **Shape** — a sweep over `states="all"`, `formalisms=["IIT_4_0_2023",
  "IIT_3_0"]` on a small substrate (`basic`/`xor`) yields a DataFrame whose row
  count equals the product of axis sizes and whose MultiIndex has the expected
  `(formalism, state)` levels (subset constant ⇒ a column, not a level).
- **Parity** — each row's `phi` equals a direct
  `System(substrate, state, node_indices=subset).sia().phi` recompute under the
  same formalism; `.results[i]` aligns with `.df.iloc[i]`.
- **`phi_structure`** — `n_distinctions` and `sum_phi_r` match a direct `ces()`
  on a known cell; IIT 3.0 rows report no relations.
- **Parallel ≡ sequential** — the same sweep under `parallel=True` and
  `parallel=False` produces an equal DataFrame (sorted) and equal per-cell φ
  (mirrors the N2 invariant for the outer loop).
- **Subsets / callable** — `subsets="all"` enumerates the non-empty powerset;
  a custom `compute=` callable tabulates.
- **Full suite** — `uv run --all-extras pytest` (no path argument) stays green.

## Roadmap bookkeeping

On landing, flip the B12 dashboard row to ✅ landed with a one-line summary,
in the same change.
