# Unify the Parallel Engine — Design

**Status:** proposed
**Author:** brainstorming session, 2026-06-23
**Sub-project:** 1 of 2 (the follow-on is *Cost-balanced chunking*, which depends on this)

## Goal

Collapse PyPhi's two coexisting map-reduce implementations into a single flat
execution path exposed through one ergonomic function, `pyphi.parallel.map_reduce()`.
The `MapReduce` class is removed; the Scheduler Protocol becomes the backend
mechanism beneath the function. This finishes the migration the Scheduler
Protocol was built for, removes a vestigial tree abstraction, and deletes
duplicated plumbing — without changing any computed result.

This is a prerequisite refactor. It deliberately does **not** add cost-balanced
chunking; that is the second sub-project, which drops cleanly onto the unified
path once this lands.

## Background: the current state

There are two parallel implementations in `pyphi/parallel/`:

- **Path A — `MapReduce`** (`__init__.py`): the public API, used one-shot as
  `MapReduce(...).run()` at all ten production call-sites. It builds
  `TreeConstraints`/`TreeSpec` from its kwargs and dispatches to
  `LocalMapReduce`.
- **Path B — Scheduler Protocol** (`scheduler.py`, `backends/`): `Scheduler`,
  `ChunkingPolicy`/`ProgressPolicy`/`ShortcircuitPolicy`, `default_scheduler()`,
  and `LocalProcessScheduler`/`LocalThreadScheduler`/`DaskScheduler`. It has
  cost-sampling (`sampling.compute_chunksize`) and honors
  `config.infrastructure.parallel_backend`. It has **zero production
  call-sites** — only tests construct it.

Three facts discovered during design make the unification low-risk and motivate
removing the class outright:

1. **The tree is vestigial.** `tree.py` (~161 lines) plus the `TreeConstraints`
   hierarchy exist solely to produce two scalars: a `chunksize` and the boolean
   `tree.depth > 1`. `MapReduce.run()` reduces to
   `if self.parallel and self.tree.depth > 1: _run_parallel() else
   _run_sequential()`, and `_run_parallel` submits **one flat future per
   chunk** — there is no recursive/hierarchical execution. `branch_factor`,
   `max_depth`, `max_size`, and `max_leaves` are otherwise unused, and **no
   call-site passes them** (verified). `test/test_parallel.py` fuzzes them only
   to assert results are *invariant* to them.
2. **Both paths already converge on `LocalMapReduce`.**
   `LocalProcessScheduler.map_reduce` is itself a wrapper that translates
   policies into a `LocalMapReduce` call (`backends/local_process.py:365-430`).
   So `LocalMapReduce`'s flat executor is the real engine for both paths;
   unification removes the *duplicate parameter plumbing and the tree*, not the
   executor.
3. **`MapReduce` is a class doing a function's job.** Every one of the ten
   call-sites constructs it and immediately calls `.run()`; no instance is
   reused and nothing reads its attributes or repr. It is not lifted to the
   top-level `pyphi.*` namespace, and 2.0 is unreleased, so there are no
   external users to break. Replacing it with a function is a strict
   simplification and leaves the smallest correct surface to freeze in P15.

## Target architecture

```
call-sites ──> map_reduce(fn, items, *, reducer, chunksize, ...)   # the one public entry
                  │  builds ChunkingPolicy/ProgressPolicy/ShortcircuitPolicy
                  ▼
            default_scheduler()            # honors config.parallel_backend / backend=
                  │
                  ▼
        LocalProcessScheduler.map_reduce   # single engine entry (also Thread/Dask)
                  │  chunk → one future per chunk (flat)
                  ▼
              loky reusable executor
```

`map_reduce()` is an ordinary module-level function. It translates its flat
kwargs into the three policy objects, selects a scheduler via
`default_scheduler()` (or the explicit `backend=`), and calls the scheduler's
`map_reduce(...)`. The policy classes remain as the **internal backend
contract**; call-sites never assemble them. The tree is gone; chunking is
computed directly from `chunksize` (explicit, from config) or
`compute_chunksize` (when unspecified), and the sequential-vs-parallel decision
is `total < sequential_threshold` (or equivalently `n_chunks <= 1`).

## Components and changes

### 1. Replace the `MapReduce` class with `map_reduce()` (`pyphi/parallel/__init__.py`)

Delete the `MapReduce` class. Add:

```python
def map_reduce(
    fn,
    items,
    *more_items,
    reducer=_flatten,            # default preserves today's flatten behavior
    parallel=True,
    ordered=False,
    total=None,
    chunksize=None,
    sequential_threshold=1,
    shortcircuit_func=false,
    shortcircuit_callback=None,
    shortcircuit_callback_args=None,
    progress=None,               # None → config.infrastructure.progress_bars
    desc=None,
    map_kwargs=None,
    backend="auto",
):
    ...
```

- `parallel=False` keeps the existing sequential short-circuit (no scheduler;
  direct `_map_sequential` + `reducer`), so the trivial path stays trivial and
  cheap.
- Otherwise it builds the policies and delegates:
  ```python
  scheduler = default_scheduler() if backend == "auto" else scheduler_for(backend)
  return scheduler.map_reduce(
      fn, items, *more_items,
      reducer=reducer,
      chunking=ChunkingPolicy(chunksize=chunksize,
                              sequential_threshold=sequential_threshold),
      progress=ProgressPolicy(enabled=resolve_progress(progress), desc=desc or "",
                              total=total),
      shortcircuit=ShortcircuitPolicy(func=shortcircuit_func,
                                      callback=shortcircuit_callback),
      ordered=ordered,
      map_kwargs=map_kwargs,
  )
  ```
- `reducer` defaults to `_flatten` (not the Protocol's `list`) so call-sites
  that omit it keep today's behavior exactly. The name `reducer` matches the
  Scheduler Protocol.
- The tree-only parameters (`branch_factor`, `max_depth`, `max_size`,
  `max_leaves`) are **not** part of the new signature.
- `backend="auto"` resolves from `config.infrastructure.parallel_backend`;
  explicit `"thread"`/`"dask"`/`"process"` selects directly. This finally wires
  backend selection that `MapReduce` previously ignored (it always used the
  process pool). The process backend remains the default.

### 2. Migrate the ten call-sites

Rewrite each `MapReduce(...).run()` to `map_reduce(...)`. The kwargs map
one-to-one (`reduce_func=` → `reducer=`); none pass tree kwargs. Call-sites:

`substrate.py:729`, `relations.py:249`, `formalism/queries.py:118`,
`formalism/queries.py:280`, `formalism/iit3/__init__.py:145`,
`formalism/iit3/__init__.py:336`, `formalism/iit4/__init__.py:1092`,
`formalism/iit4/formalism.py:248`,
`formalism/actual_causation/compute.py:555`, `macro/search.py:291`.

`macro/search.py` passes `**parallel_kwargs(...)`; confirm that dict's keys are
all still valid `map_reduce` parameters after the tree kwargs are dropped from
the config allow-list (§4).

### 3. Reduce-semantics parity (the main correctness surface)

`reducer` must be invoked with the same flat iterable of per-item results and
return the same value as today. Each call-site's reducer:

| Call-site | Reducer today | Notes |
|---|---|---|
| `substrate.py:729` complexes | default `_flatten` | flat list of complexes |
| `relations.py:249` relations | default `_flatten` | |
| `formalism/queries.py:118` candidate MIPs | default `_flatten` | result asserted non-None |
| `formalism/queries.py:280` MICE | default `_flatten` | `mice_class` mapped *after* the call |
| `formalism/iit3/__init__.py:145` concepts | passed-in `reduce_func` | caller supplies reducer |
| `formalism/iit3/__init__.py:336` candidates | default `_flatten` | |
| `formalism/iit4/__init__.py:1092` SIAs | default `_flatten` | any selection is downstream |
| `formalism/iit4/formalism.py:248` MIPs | default `_flatten` | any selection is downstream |
| `formalism/actual_causation/compute.py:555` | `reduce_func=min` | min over alpha |
| `macro/search.py:291` | from `parallel_kwargs` | `ordered=True` set |

The plan includes one task per cluster to confirm parity. `min` (AC) and the
IIT3 passed-in reducer are the cases to watch; the rest are the default flatten.

### 4. Remove the tree

- **Delete** `pyphi/parallel/tree.py` (`TreeConstraints`, `TreeConstraintsSize`,
  `TreeConstraintsChunksize`, `TreeSpec`, `get_constraints`, `simulate`,
  `get_initial_chunksize`) and **`test/test_tree.py`**.
- Remove `tree`/`constraints` parameters and `tree.depth` gating from
  `LocalMapReduce` and from `LocalProcessScheduler.map_reduce`. The
  sequential-vs-parallel gate becomes: parallelize when `total >=
  sequential_threshold` and chunking yields more than one chunk.
- Remove the four tree-kwarg names (`max_depth`, `max_size`, `max_leaves`,
  `branch_factor`) from the `conf/_helpers.py` parallel-kwargs allow-list so
  config no longer advertises dead knobs.

### 5. Deduplicate shared helpers and collapse the executor entry

- `get_num_processes`, `false`, `_flatten`, `_map_sequential`, and `_reduce` are
  duplicated across `__init__.py` and `backends/local_process.py`. Consolidate
  to one home and delete the copies. No behavior change.
- Keep `LocalMapReduce` as the flat executor, but strip its tree parameters so
  it takes `chunksize` + `sequential_threshold` directly.
  `LocalProcessScheduler.map_reduce` remains the engine entry the function calls.

## Behavior changes (all result-preserving)

- **Chunksize when unspecified.** Today `MapReduce` with `chunksize=None` picks
  `total // branch_factor` (a coarse 2 chunks). The unified path runs
  `compute_chunksize` cost-sampling instead, which is strictly better
  granularity. All hot call-sites pass an explicit `chunksize` from config, so
  they are unaffected; only callers omitting `chunksize` see the (improved)
  sampled granularity. Correctness is unchanged — chunking never affects
  results, guarded by the N2 invariant.
- **Backend selection now honored.** `config.parallel_backend` and the
  `backend=` argument now actually route to thread/dask schedulers, where
  before `MapReduce` always used the process pool. The process backend remains
  the default, so default behavior is unchanged.

## Error handling

- Exceptions raised in workers propagate exactly as today (the executor
  re-raises; the function surfaces the first exception).
- Short-circuit + future cancellation is already implemented identically in the
  flat executor; it is retained unchanged.

## Testing strategy

- **N2 invariant** (`test/test_parallel_equals_sequential.py`): the primary
  guard — parallel results must equal sequential. Run after every task.
- **Golden regression suite** (`test/golden/`, `test/test_golden_regression.py`):
  full-computation results must be byte-identical.
- **Perf gate** (`test/test_perf_counters.py`): deterministic call-counts must
  not move (chunking is not on the counted frames, so this should be inert; it
  confirms no accidental recomputation).
- **`test/test_parallel.py`**: rewrite from constructing `MapReduce` to calling
  `map_reduce`; drop the fuzz over the removed tree kwargs; keep the
  result-invariance and short-circuit assertions.
- **`test/test_scheduler.py`**: now exercises the path the call-sites use;
  extend to cover the function's translation (policies built correctly, reducer
  threaded, backend resolution).
- **`test/test_tree.py`**: deleted with the module.
- Full `uv run pytest` (no path argument, so `pyphi/` doctests run) before
  declaring complete.

## Out of scope

- **Cost-balanced chunking / `size_func`** — sub-project 2. `ChunkingPolicy`
  already carries the dormant `size_func`/`target_seconds` fields; this refactor
  leaves them dormant and untouched.
- **Completing the Dask backend** — `DaskScheduler` stays a stub (tracked
  separately under cluster backends). Unification only needs the Protocol shape
  and the process/thread schedulers.
- **Changing any computed result, measure, or config default.**

## Verification checklist

- [ ] All ten call-sites migrated to `map_reduce()`; `MapReduce` class deleted;
      no remaining references to it in `pyphi/`.
- [ ] `tree.py` and `test_tree.py` deleted; no remaining imports of either.
- [ ] N2, goldens, and perf gate all green.
- [ ] `config.parallel_backend = "thread"` routes a `map_reduce` call through
      the thread scheduler (new coverage).
- [ ] Full `uv run pytest` green, including `pyphi/` doctests.
