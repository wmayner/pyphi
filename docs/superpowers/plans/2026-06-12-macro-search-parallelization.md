# Macro search parallelization — implementation plan

**Goal:** Parallelize the intrinsic-units search drivers across their
independent `phi_s` evaluations, per the approved spec
(`docs/superpowers/specs/2026-06-12-macro-search-parallelization-design.md`),
preserving bit-identical results.

## Verified groundwork (probed during planning; do not re-derive)

- `MacroSystem` pickles round-trip with equal `==`/`hash`.
- `MapReduce(..., ordered=True)` returns results in input order,
  values bitwise-equal to sequential.
- Nested parallelism (a parallel `sia()` inside a search-level worker)
  is **safe** (no deadlock) but oversubscribes (12.5 s vs ~3 s). So the
  worker forces the inner `sia()` sequential — but only on the
  parallel-dispatch path, so the default (search-parallel off) path is
  byte-identical to SP2, including SP2's per-evaluation partition-level
  parallelism.
- `PARALLEL_KWARGS` includes `ordered`, `total`, `desc`, `parallel`,
  `sequential_threshold`, `chunksize`, `progress` — so per-call
  overrides flow through `conf.parallel_kwargs`; `ordered`/`total` are
  force-set after the merge to protect determinism.
- `_default_parallel_dict(seq_threshold, chunksize, *, progress)`
  yields `{"parallel": False, ...}`; the global `parallel` switch gates
  every per-site dict via `conf.parallel_kwargs`.

## Determinism contract (the spine of every task)

Parallel and sequential runs must produce field-identical
`ComplexesResult` / `IntrinsicUnitsResult`. Guaranteed by:

1. Per-evaluation arithmetic unchanged (each `MacroSystem.sia().phi`).
2. The memo is warmed in **sequential-first-need order**: a batch
   `pending` is built by walking candidates in the exact SP2 order
   (each candidate's constituent system, then its f-members in
   `_assemble_systems` order), deduped against the memo preserving
   first occurrence; results merged back in `pending` order.
3. Judgments and Eq.19 assembly stay sequential and read the warm memo.
4. Size-class batching is sound because `f(U^J,W^J)` admits only
   strict-subset footprints, so a size-`s` candidate never depends on
   any other size-`s` system — the whole class is mutually independent
   and the pool can grow once per class (proven equivalent to SP2's
   per-footprint growth, which `_f` already filters to strict subsets).

A per-run `system_cache: dict[canonical_units, MacroSystem|None]` is
threaded next to `memo` so the collect phase and the judge phase share
constructions (no double `MacroSystem.from_micro`), keeping the default
path at SP2 speed.

## Tasks

### Task 1 — config option (no behavior change)

`pyphi/conf/infrastructure.py`: add field
```python
parallel_macro_system_evaluation: Mapping[str, Any] = field(
    default_factory=lambda: _default_parallel_dict(2**4, 2**6, progress=True)
)
```
and add `"parallel_macro_system_evaluation"` to the validation tuple
(the `for parallel_field_name in (...)` listing). Test in
`test/test_macro_search.py`: the option exists, is a Mapping, defaults
to `parallel=False`, `sequential_threshold=16`, and is force-disabled
by `conf.parallel_kwargs` when global `parallel=False`. Commit.

### Task 2 — `system_cache` plumbing (no behavior change)

In `pyphi/macro/search.py` add
```python
def _system_of_cached(substrate, units, micro_history, system_cache):
    key = canonical_units(units)
    if key not in system_cache:
        system_cache[key] = _system_of(substrate, units, micro_history)
    return system_cache[key]
```
and thread a `system_cache` dict through `_phi`, `_f`, `_judge`,
`_derive_units`, `_f_for_unit`, and all five drivers (create `{}` next
to each `memo = {}`). `_phi` calls `_system_of_cached`. No other logic
changes. Run the full SP2/SP3 macro suites — all green (pure
refactor). Commit.

### Task 3 — evaluation helpers (no behavior change yet)

Add module-level:
```python
def _evaluate_one(system: MacroSystem) -> float:
    """Worker entry: phi_s of one system, inner sia forced sequential
    to avoid nested process pools (search-level parallelism is one
    level deep)."""
    from pyphi.conf import config as _config
    with _config.override(parallel=False):
        return float(system.sia().phi)


def _evaluate_systems(systems, memo, parallel_kwargs=None):
    """Evaluate `systems` (deduped against `memo`, in order) and merge
    phi_s into `memo` in dispatch order. Parallel only when the macro
    option is enabled; otherwise in-process under ambient config (so
    SP2's per-evaluation partition parallelism is preserved)."""
    from pyphi import conf as _conf
    from pyphi.conf import config as _config
    from pyphi.parallel import MapReduce

    pending = []
    seen = set()
    for system in systems:
        if system is None or system in memo or system in seen:
            continue
        seen.add(system)
        pending.append(system)
    if not pending:
        return
    pkwargs = _conf.parallel_kwargs(
        _config.infrastructure.parallel_macro_system_evaluation,
        **(parallel_kwargs or {}),
    )
    if pkwargs.get("parallel"):
        pkwargs["ordered"] = True
        pkwargs["total"] = len(pending)
        pkwargs.setdefault("desc", "Evaluating macro systems")
        phis = MapReduce(_evaluate_one, pending, **pkwargs).run()
    else:
        phis = [float(system.sia().phi) for system in pending]
    for system, phi in zip(pending, phis, strict=True):
        memo[system] = PyPhiFloat(phi)
```
Not yet wired into the drivers. Unit-test `_evaluate_systems` directly:
dedup against memo, empty-input no-op, in-process path equals direct
`sia`, parallel path (`config.override` macro option on,
`sequential_threshold=1`) equals sequential and inserts in order.
Commit.

### Task 4 — wire the P(u) sweep + driver signatures

`complexes` and `valid_systems`: build `sweep_systems =
[_system_of_cached(substrate, combo, history, system_cache) for combo
in _assemble_systems(list(units), bounds.max_background)]`, call
`_evaluate_systems(sweep_systems, memo, parallel_kwargs)` (complexes
only; valid_systems needs no phi), then build `evaluated`/records by
reading `memo[system]` over the non-None `sweep_systems` in order
(identical content/order to SP2's `_phi` loop). Add
`parallel_kwargs: dict | None = None` to `complexes`, `valid_systems`,
`intrinsic_units`, `is_intrinsic_unit`, `competing_systems`; thread to
`_derive_units` / `_f_for_unit`. `_f_for_unit` pre-warms its
competitor systems via `_evaluate_systems` before the final `_f`. Run
SP2/SP3 suites (default path) + a new parallel-equivalence test on the
min EXHAUSTIVE driver. Commit.

### Task 5 — batch the recursion

Restructure `_derive_units`'s per-level loop into per-size-class
collect → `_evaluate_systems` → judge, using a `pool_at_class_start`
snapshot for f-members and materializing the `(footprint, V, W)`
candidate list once (applying the cross-level `seen` dedup) for both
phases. Pre-warm the level-0 micro systems as a batch too. Verdict /
pool / records order unchanged. Run SP2/SP3 suites + the
parallel-equivalence and tie-path drivers under the macro option.
Commit.

### Task 6 — tests, docs, changelog, roadmap, verify, push

Add to `test/test_macro_search.py`: equivalence battery (min
EXHAUSTIVE, tie-path, bu — sequential vs `parallel=True` macro option,
field-identical `ComplexesResult` incl. record order and a bitwise phi
check), gating (global `parallel=False` forces in-process via the
kwargs the driver builds), and a slow cg-driver-under-parallel test
asserting the SP2 golden outcome. One paragraph in
`docs/examples/macro.rst` on the option. `changelog.d/
macro-search-parallel.feature.md`; ROADMAP item-10 addendum. Full
`uv run --no-sync pytest` (no path) + slow lane. Push. Report.

## Standing constraints

As prior sessions: signed commits (plain `git commit`; the container
signs); targeted `git add`; `uv run --no-sync`; no Unicode math in
Python; ruff/pyright clean; full verification = no-path pytest + slow
lane. Determinism is the acceptance bar — any goldens drift is a bug in
the refactor, not a value to update.
