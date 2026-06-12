# Macro search parallelization — design

**Project:** Macro framework follow-up (item 10 addendum). Parallelize
the intrinsic-units search drivers (`pyphi/macro/search.py`) across
their many independent `phi_s` evaluations, using the existing
`MapReduce` engine and per-site config-dict pattern. SP2 explicitly
deferred this; the search currently evaluates every candidate system
sequentially.

**Sources of truth:** `pyphi/parallel/__init__.py` (`MapReduce`),
`pyphi/conf/_helpers.py::parallel_kwargs`, the consumption pattern in
`pyphi/substrate.py` (`parallel_complex_evaluation`), and the
`InfrastructureConfig` per-level option dicts.

## Where the time goes, and what is independent

Every driver run reduces to a set of `MacroSystem.sia()` evaluations:

1. **Within the recursion** (`_derive_units`): for each candidate
   decomposition, the constituent system plus every member of
   f(U^J, W^J).
2. **The P(u) sweep** (`valid_systems` / `complexes`): one evaluation
   per admissible system.

Two structural facts make batching safe:

- f(U^J, W^J) admits only units with footprints that are *proper*
  subsets of U^J, so candidates whose footprints have the **same
  size** never feed each other's competitor sets. All evaluations
  needed by one footprint-size class are therefore independent of one
  another, and the pool can be updated once per size class instead of
  once per footprint with no observable change.
- The P(u) sweep is embarrassingly parallel once the pool is fixed.

## Design

**Granularity: the individual system evaluation.** Each unit of work
is one `(MacroSystem) -> PyPhiFloat` call. This gives the best load
balance (evaluations range from milliseconds to minutes), keeps the
memo authoritative in the parent process, and makes parallel results
provably identical to sequential ones (same arithmetic per evaluation,
results merged in dispatch order).

**Execution shape.** Replace the two sequential evaluation sites with
a shared helper:

```python
def _evaluate_systems(systems, memo, parallel_kwargs) -> None:
    """Evaluate systems not yet in the memo, in order, possibly in
    parallel; merge (system, phi) results into the memo."""
```

- In `_derive_units`: per level, per footprint-size class — collect
  the unique systems needed by every candidate in the class
  (constituent systems and f members, deduplicated through the memo
  and canonical unit order), `_evaluate_systems` them, then run the
  judgments sequentially (pure float comparisons, microseconds).
  Verdicts, pool order, and emission order are unchanged.
- In `valid_systems` / `complexes`: construct all P(u) systems
  sequentially (construction is cheap numpy work and is where
  unreachable states are dropped), then `_evaluate_systems` the ones
  missing from the memo.

`_evaluate_systems` dispatches via `MapReduce(sia_fn, systems,
ordered=True, **pkwargs)` where `sia_fn` returns `(system, phi)`;
`MacroSystem` is picklable (plain frozen dataclasses over numpy
arrays, same as the `System` objects `pyphi.substrate` already ships
to workers).

**Configuration.** One new `InfrastructureConfig` option in the
existing family:

```python
parallel_macro_system_evaluation: Mapping[str, Any] = field(
    default_factory=lambda: _default_parallel_dict(2**4, 2**6, progress=True)
)
```

gated as everywhere else by the global `parallel` switch via
`conf.parallel_kwargs`. The drivers (`complexes`, `intrinsic_units`,
`valid_systems`, `is_intrinsic_unit`, `competing_systems`) accept an
optional `parallel_kwargs` mapping for per-call overrides, mirroring
the `pyphi.substrate` entry points. Defaults chosen to match the
family: `sequential_threshold=16`, `chunksize=64` — small searches
(min, the unit-test fixtures) stay sequential automatically.

**Nesting.** Each `sia()` already parallelizes internally
(partition/purview/concept levels) when `parallel=True`. Search-level
workers run their evaluations with the inherited config snapshot, so
inner parallelism inside worker processes falls back to sequential
(`ProcessPoolExecutor` workers do not nest pools). This is the desired
behavior — one level of process parallelism — and is documented in the
option's docstring: for searches of many small systems, search-level
parallelism wins; for a single huge evaluation, disable it (or stay
under the threshold) and let the partition-level parallelism work.

## Determinism contract

Parallel and sequential runs must produce **identical** results:
identical `ComplexesResult` (winners, ties, records and their order),
identical `IntrinsicUnitsResult`. Guaranteed by: per-evaluation
arithmetic unchanged; `ordered=True` merging; memo insertion in
dispatch order; judgments and assembly always sequential. The
size-class batching changes only the order in which the memo warms up,
never any verdict input (argued above; pinned by tests).

## Testing

1. **Equivalence battery:** the min EXHAUSTIVE driver and the tie-path
   driver under `config.override(parallel=True,
   parallel_macro_system_evaluation={"parallel": True,
   "sequential_threshold": 1, ...})` produce results equal field-by-
   field to the sequential SP2 goldens (same winners, same tie pair,
   same record systems in the same order, phi equal at precision; phi
   also bitwise-equal as a stricter assertion, relaxable to 1e-13 if a
   platform ever breaks it).
2. **Size-class batching invariance** (pure, no processes): the cg
   default-bounds driver sequential-before vs sequential-after — the
   refactor itself must not change `ComplexesResult`; pinned by the
   existing SP2/SP3 goldens staying green.
3. **Gating:** global `parallel=False` forces sequential regardless of
   the per-site dict (assert via the existing `parallel_kwargs`
   helper semantics — a unit test on the kwargs the drivers build).
4. **Threshold behavior:** below `sequential_threshold` no pool is
   spawned (observable via `MapReduce.tree.depth` or by monkeypatching
   the backend; keep it simple — test `parallel_kwargs` output).
5. **Slow lane:** the cg driver with parallelism on, asserting the
   SP2 golden outcome exactly (winner units, phi at 1e-13).

## Costs and expected gains

The cg default driver (~80 evaluations, 3.2 s sequential, dominated by
one 0.5 s evaluation plus many ~10 ms ones) is a wash-to-modest win on
4 cores; the realistic beneficiaries are larger substrates and
EXHAUSTIVE sweeps where hundreds-to-thousands of comparable-cost
evaluations dominate (e.g. the bbx-scale searches, hours sequential).
Worker startup (~100 ms) and per-task pickling of `MacroSystem` TPMs
are amortized by `chunksize`.

## Files

- `pyphi/conf/infrastructure.py` — add
  `parallel_macro_system_evaluation` (+ the options listing)
- `pyphi/macro/search.py` — `_evaluate_systems`, size-class batching,
  `parallel_kwargs` threading through the public drivers
- `test/test_macro_search.py` — equivalence/gating/threshold batteries
- `changelog.d/macro-search-parallel.feature.md`, `ROADMAP.md` —
  record as an item-10 addendum
- `docs/examples/macro.rst` — one short paragraph noting the option

Out of scope: resumability/checkpointing; dask/cluster backends
(inherited automatically if `MapReduce` grows them); parallelizing the
combinatorial assembly itself (pure Python, negligible).
