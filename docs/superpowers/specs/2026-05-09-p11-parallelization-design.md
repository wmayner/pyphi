# P11 — Parallelization Redesign with `Scheduler` Protocol

**Status:** design (2026-05-09)
**Branch:** `feature/p11-parallelization-redesign` (cut from `2.0` at `b4d58aa2`)
**Predecessor projects:** P7 (subsystem layered rewrite), P8 (models split), P9 (unified cache), P10 (config split)
**Successor project:** P18 (post-2.0) — full Dask + HTCondor cluster backends

---

## Goal

Replace the current ad-hoc backend dispatch in `pyphi/parallel/` with a typed
`Scheduler` Protocol exposing a single `map_reduce(...)` entry point. Ship two
fully-implemented schedulers (`LocalProcessScheduler`, `LocalThreadScheduler`)
plus a `DaskScheduler` skeleton that exercises the Protocol against three call
shapes (loky pool, thread pool, dask client). Deliver explicit
`ConfigSnapshot` propagation to workers (replacing implicit-pickle-of-globals)
and replace speculative context-based chunking heuristics with cost-sampling.

## Why now

P10 enables this: workers can now receive an explicit `ConfigSnapshot` bundle
instead of pickling global state. P7–P9 stabilized the kernel and cache
abstractions, so the parallel layer can lift cleanly without rebuilding the
work units themselves.

The `LocalThreadScheduler` is the most consequential addition: under
free-threaded Python (3.13t), it allows shared-memory cache, generators that
cross worker boundaries without materialization, and cheap spawn for small
tasks. P6a's globals audit cleared the immediate safety concerns; P6b
(graphillion replacement) is deferred and gates the relations-workflow path
through `LocalThreadScheduler`.

## Non-goals

- Full Dask cluster implementation (skeleton only; P18 follow-up)
- HTCondor adapter (P18 follow-up)
- No-GIL CI matrix entry (gated on P6b per existing P6b deferral)
- Ray support (the `PROJECTS.md` Ray entry is stale; loky covers the local case)
- Distributed cache backend (P9 deferred-items registry; P11 leaves cache
  scope unchanged for process workers)

---

## Decisions

### 1. Scheduler scope for v2.0

Ship: **`LocalProcessScheduler` + `LocalThreadScheduler` fully implemented;
`DaskScheduler` skeleton.** The skeleton lazy-imports `dask.distributed` and
raises `NotImplementedError("DaskScheduler is a stub; fill in for cluster
deployments")` from `map_reduce`. Test asserts: import is lazy (no
`dask.distributed` import at `import pyphi`), the class exists in the
registry, and `map_reduce` raises with the documented message.

`HTCondorScheduler` and the full Dask implementation defer to **P18**
post-2.0.

### 2. `Scheduler` Protocol shape

Lives in `pyphi/parallel/scheduler.py`:

```python
@runtime_checkable
class Scheduler(Protocol):
    def map_reduce(
        self,
        fn: Callable[..., R],
        items: Iterable,
        *more_items: Iterable,
        reducer: Callable[[Iterable[R]], T] = list,
        config_snapshot: ConfigSnapshot | None = None,
        chunking: ChunkingPolicy | None = None,
        progress: ProgressPolicy | None = None,
        shortcircuit: ShortcircuitPolicy | None = None,
        ordered: bool = False,
    ) -> T: ...

    @property
    def supports_shared_state(self) -> bool: ...
```

`supports_shared_state` is `True` for `LocalThreadScheduler`, `False` for
`LocalProcessScheduler` and `DaskScheduler`. Call sites that depend on
parent-shared mutable state (e.g., the cache registry) check this property
before relying on it.

`ChunkingPolicy`, `ProgressPolicy`, `ShortcircuitPolicy` are small frozen
dataclasses bundling the parameters that today live as separate kwargs on
`MapReduce.__init__` (chunksize, sequential_threshold, max_depth, branch_factor,
etc.; progress / desc / total; shortcircuit_func / callback / args).

### 3. Worker config delivery

**Per-call snapshot via closure.** The scheduler captures
`snapshot = pyphi.config.snapshot()` at the start of each `map_reduce(...)`
call. It wraps the user's `fn`:

```python
def make_worker_fn(fn, snapshot):
    def worker_fn(*args, **kwargs):
        _apply_snapshot_if_changed(snapshot)
        return fn(*args, **kwargs)
    return worker_fn
```

Worker-side dedup uses `snapshot.id` (a content hash) cached in a process-local
module variable; first task pays the apply cost, subsequent tasks with the
same snapshot are no-ops.

For `LocalThreadScheduler`: workers share parent process memory.
`_apply_snapshot_if_changed` is a no-op when running on a thread scheduler
(detected via `threading.current_thread() is threading.main_thread()` or by
the wrapper checking `os.getpid()` against the captured parent PID).

**Why not init-once at executor spawn:** loky reuses executors across calls.
A `with config.override(precision=6): subsystem.find_mip(...)` block must
propagate to workers. Init-once would freeze workers at executor-creation
config and silently diverge from a parent inside an override block. Per-call
snapshot is the only way the override semantics work.

### 4. Cache state on workers

- **`LocalProcessScheduler`** — workers spawn fresh; per-process caches start
  empty. No special logic; preserves today's behavior. Each worker
  independently registers its caches via the P9 module-level `@cache(...)`
  decorators when its first task imports the relevant pyphi modules.
- **`LocalThreadScheduler`** — workers share parent's caches via the
  `DictCache` instances installed at module load. P9's threading documentation
  covers GIL-enabled safety. **The no-GIL safety audit of `DictCache` defers
  to a P6b/P11 follow-up commit, gated on P6b landing.** P11 ships with
  thread-shared cache safe under GIL-enabled runtimes.
- **`DaskScheduler`** — skeleton; defers cache discussion to P18.

### 5. Chunking — cost-sampling as the default; delete `chunking.py`

`pyphi/parallel/chunking.py` (254 lines) is **dead code**. None of
`adaptive_chunk` / `chunked_by_work` / `estimate_work_size` /
`estimate_total_work` / `calculate_target_work` is imported anywhere outside
the module itself. The speculative context-string heuristics (`'mechanism':
size·2^size`, `'partition': n²`, etc.) never ran in production. Today's
actual chunking path uses `more_itertools.chunked_even` in
`backends/local.py:184` with chunksize from `tree.py:TreeConstraints.get_initial_chunksize()`.

Delete `pyphi/parallel/chunking.py` entirely. Replace with cost-sampling
logic in `pyphi/parallel/sampling.py` (~60 lines) wired into the scheduler.

Default chunking algorithm (lives on the scheduler):

1. If `total < sequential_threshold`: skip sampling, run sequentially.
2. If user passes explicit `chunksize=N` to `MapReduce`: use that, skip
   sampling.
3. Otherwise sample 4 items spread across the iterable (positions `0`,
   `⌊N/4⌋`, `⌊N/2⌋`, `⌊3N/4⌋`) — handles the case where mechanism cost grows
   with size.
4. Time them inline; compute `mean_per_item`.
5. `target_chunksize = max(1, ⌊1.0s / mean_per_item⌋)`.
6. Chunk the remaining items at `target_chunksize` via `chunked_even`.

Generators preserved: only the 4 sampled items are materialized eagerly; the
rest stay lazy past sampling. For finite-known-length iterables (`__len__`
present), sampling positions are exact; for unknown-length iterables,
sampling reads the first 4 items consecutively (degraded but correct).

Keep `size_func=` parameter as an escape hatch on `MapReduce` for users who
provide their own per-item cost function — bypasses sampling entirely.

### 6. P10 Phase 4 follow-through (frozen-formalism conversion)

Fold into P11 as **Phase 1**, before scheduler work:

- `IIT3Formalism`, `IIT4_2023Formalism`, `IIT4_2026Formalism` become
  `@dataclass(frozen=True)` with `config: FormalismConfig` as a field (not a
  property delegating to the live global).
- Method bodies read `self.config.X` instead of `pyphi.config.X`.
- Pickle-roundtrip test for each — workers will receive these via
  cloudpickle.
- `SUBSYSTEM_PARALLEL_FORMALISM` field validation rule preserved.

This is a P10 deferred item that pairs naturally with worker-config-bundling.
Doing it as a P11 phase rather than a separate commit keeps the formalism +
worker boundary cohesive.

### 7. `MapReduce` user-facing API

**Keep `MapReduce` as the user-facing class; `Scheduler` is the dispatch
layer underneath.** `MapReduce.__init__` accepts `backend=...` as today (with
the additional `"thread"` / `"dask"` values), constructs the appropriate
`Scheduler` internally, and delegates `run()` to `scheduler.map_reduce(...)`.
This avoids migrating the 12 internal call sites (`subsystem.py:861,1189`,
`relations.py:227`, `actual.py:707`, `compute/network.py:115,162`,
`compute/subsystem.py:96,224`, `formalism/queries.py:115,277`,
`formalism/iit4/formalism.py:138`, `formalism/iit4/__init__.py:666`); the
surface change is internal.

`Scheduler` is exposed at `pyphi.parallel.scheduler.Scheduler` for users who
want the lower-level entry point (e.g., custom backends).

### 8. Backend selection

```python
def default_scheduler() -> Scheduler:
    backend = config.infrastructure.parallel_backend
    if backend == "auto":
        return LocalThreadScheduler() if not sys._is_gil_enabled() else LocalProcessScheduler()
    if backend in ("local", "process"): return LocalProcessScheduler()
    if backend == "thread": return LocalThreadScheduler()
    if backend == "dask": return DaskScheduler()
    raise ValueError(f"unknown parallel_backend: {backend}")
```

`config.infrastructure.parallel_backend: str = "local"` already exists from
P10. Keep `"local"` as an alias for `"process"` for migration ergonomics; the
nested-format YAML loader accepts both.

`"auto"` is the recommended setting for users who want runtime detection.
Default stays at `"local"` to preserve today's deterministic behavior — users
opt into `"auto"` once they're on a free-threaded runtime.

---

## Architecture

### File layout

```
pyphi/parallel/
  __init__.py                    # MapReduce facade, get_num_processes (legacy
                                 # backend resolver removed)
  scheduler.py                   # Scheduler Protocol, default_scheduler(),
                                 # ChunkingPolicy / ProgressPolicy /
                                 # ShortcircuitPolicy frozen dataclasses
  chunking.py                    # cost-sampling adaptive_chunk; heuristics
                                 # path deleted
  tree.py                        # unchanged (TreeConstraints, TreeSpec)
  backends/
    __init__.py
    local_process.py             # LocalProcessScheduler (renamed from local.py;
                                 # implements Scheduler, wraps loky executor)
    local_thread.py              # LocalThreadScheduler (new)
    dask.py                      # DaskScheduler skeleton (new)
    progress.py                  # unchanged (LocalProgressBar)
```

`backends/local.py` (today's `LocalMapReduce`) becomes
`backends/local_process.py:LocalProcessScheduler`; the class is renamed and
its public surface narrowed to the `Scheduler` Protocol. Internal helpers
(`_process_chunk`, `_run_parallel`, etc.) stay roughly as-is.

### Data flow (LocalProcessScheduler example)

```
caller              MapReduce             LocalProcessScheduler        loky worker
  |                   |                       |                          |
  |-- run() --------->|                       |                          |
  |                   |-- map_reduce(...) --->|                          |
  |                   |                       | snapshot = config.snapshot()
  |                   |                       | sample 4 items, time them
  |                   |                       | target_chunksize = 1s/mean
  |                   |                       | chunks = adaptive_chunk(...)
  |                   |                       | wrapped_fn = make_worker_fn(fn, snapshot)
  |                   |                       | for chunk in chunks:
  |                   |                       |   executor.submit(_process_chunk, chunk, wrapped_fn)
  |                   |                       |                              |
  |                   |                       |                              |-- _apply_snapshot_if_changed(snapshot)
  |                   |                       |                              |-- for arg in chunk: wrapped_fn(arg)
  |                   |                       |                              |-- return results
  |                   |                       |<-- chunk_results ------------|
  |                   |                       | reducer(all_results)
  |                   |                       |
  |                   |<-- T -----------------|
  |<-- T ------------|                       |
```

For `LocalThreadScheduler`: same flow, but the worker runs in the parent
process (no pickling), `_apply_snapshot_if_changed` is a no-op, and the
shared-state property is `True` (callers may rely on parent-shared cache).

### Error handling

- Worker exceptions: `LocalProcessScheduler` propagates via loky's future
  result. `LocalThreadScheduler` propagates via `concurrent.futures.Future`.
  `DaskScheduler` skeleton: `NotImplementedError`.
- Cancellation: `MapReduce`'s shortcircuit semantics preserved. On
  shortcircuit hit, scheduler cancels remaining futures, calls user-provided
  `shortcircuit_callback`, returns partial results to reducer.
- Config-apply failures: `_apply_snapshot_if_changed` wraps `config.apply`
  in a try/except and re-raises with a `WorkerConfigError` carrying the
  snapshot id so failures are diagnosable.
- Cost-sampling failures: if any of the 4 sampled items raise, propagate
  immediately (don't silently switch to fallback chunksize — sampling
  failures are real bugs).

### Testing strategy

**Per-phase acceptance:** golden 17/17 numerical match, hypothesis fast lane
21 green, fast unit lane green, pyright clean on touched files, ruff clean.

**New tests** (lives in `test/test_scheduler.py`):
- Protocol conformance: `isinstance(scheduler, Scheduler)` for each concrete.
- `default_scheduler()` returns correct type per `config.infrastructure.parallel_backend`.
- Snapshot delivery: spawn a worker, mutate parent config mid-run via
  `with config.override(precision=6)`, verify worker sees `precision=6`.
- Snapshot dedup: verify second task on the same worker doesn't re-apply
  identical snapshot (mock `_apply_snapshot_if_changed` and count calls).
- Cancellation: shortcircuit triggered, remaining futures cancelled, partial
  results returned.
- DaskScheduler: lazy-import test (no `dask.distributed` in `sys.modules`
  after `import pyphi`); `NotImplementedError` from `map_reduce`.
- `supports_shared_state` returns correct value per scheduler.

**Existing tests:**
- Re-enable `test/test_parallel.py` in CI (currently excluded). Mark thread-
  scheduler-specific tests `xfail(reason="requires P6b for relations workflows")`
  where applicable.
- `test/test_chunking.py` extended with cost-sampling tests; tests for the
  deleted heuristics path removed.

**Hypothesis fast lane:** parametrize over `parallel_backend in ("process",
"thread")` for properties that don't require P6b (i.e., everything except
relations-heavy paths).

**No new no-GIL CI matrix:** gated on P6b per ROADMAP.

---

## Phase plan

Each phase ends with a green-test commit (golden + hypothesis fast lane +
fast unit lane). No commit bypasses pre-commit hooks.

### Phase 0 — branch + audit (done)

Branch `feature/p11-parallelization-redesign` cut from `2.0` at `b4d58aa2`.
Context exploration done; design spec at this file.

### Phase 1 — frozen-formalism conversion (P10 follow-through)

Convert `IIT3Formalism`, `IIT4_2023Formalism`, `IIT4_2026Formalism` to
`@dataclass(frozen=True)` with `config: FormalismConfig` as a field. Method
bodies read `self.config.X`. Pickle-roundtrip test per formalism. Existing
tests pass; golden gate.

### Phase 2 — `Scheduler` Protocol + `pyphi/parallel/scheduler.py`

Define Protocol, `ChunkingPolicy`, `ProgressPolicy`, `ShortcircuitPolicy`.
Add `default_scheduler()` resolver. Feature-flagged behind
`config.infrastructure.parallel_backend = "auto"` so default behavior is
unchanged. No call sites migrated yet.

### Phase 3 — `LocalProcessScheduler`

Implement by wrapping today's `LocalMapReduce`. Wire snapshot-via-closure
config delivery. Snapshot dedup. All existing tests pass. Pickle-roundtrip
test for the worker-fn closure (cloudpickle).

### Phase 4 — `LocalThreadScheduler`

Implement using `concurrent.futures.ThreadPoolExecutor`. `supports_shared_state
= True`. Snapshot apply is a no-op when running on the thread scheduler. Add
hypothesis fast-lane parametrization over `parallel_backend in ("process",
"thread")` for properties that don't depend on P6b.

### Phase 5 — `DaskScheduler` skeleton

Class with lazy `dask.distributed` import; `map_reduce` raises
`NotImplementedError("DaskScheduler is a stub; fill in for cluster deployments")`.
Test for lazy import + the documented exception.

### Phase 6 — delete `chunking.py`; add `sampling.py`

Delete `pyphi/parallel/chunking.py` outright (dead code; nothing imports
its functions outside the module itself). Delete `test/test_chunking.py`
(tests for dead code). Create `pyphi/parallel/sampling.py` (~60 lines)
with the cost-sampling `compute_chunksize(items, target_seconds=1.0)`
function. Wire it into `LocalProcessScheduler.map_reduce` and
`LocalThreadScheduler.map_reduce` as the default chunk-size source when
the caller doesn't pass `chunksize=`. New `test/test_sampling.py` covers
the algorithm with mocked timing.

### Phase 7 — call-site propagation

The 12 `MapReduce(...)` call sites stay unchanged because `MapReduce` stays
as the facade. Fill `TODO(4.0) parallelize` markers in
`iit4/__init__.py:763,772,780` and `compute/subsystem.py:322` (and
`models/ces.py:194`, `actual.py:667`) using `MapReduce` with cost-sampling
chunking.

### Phase 8 — re-enable parallel tests + acceptance

Re-enable `test/test_parallel.py` in CI. Mark thread-scheduler-specific
tests `xfail(reason="requires P6b for relations workflows")`. Run full
acceptance gate: golden 17/17 + hypothesis full lane + fast unit + pyright.

### Phase 9 — cleanup + changelog

Delete `_resolve_backend` legacy string-dispatch from `MapReduce.__init__`
(replaced by `default_scheduler()`). Delete `LocalMapReduce` if its public
surface is fully absorbed by `LocalProcessScheduler`. Changelog fragment in
`changelog.d/`. ROADMAP P11 entry marked done with deferred items called
out (P18 cluster backends).

---

## Acceptance criteria for the project

- All 17 golden fixtures match numerically (1e-12) on `develop`-equivalent
  workloads run through both `LocalProcessScheduler` and (where P6b doesn't
  block) `LocalThreadScheduler`.
- Hypothesis fast lane (21 properties) green under both schedulers.
- Fast unit lane green.
- `test/test_parallel.py` re-enabled and green (modulo P6b xfails).
- `test/test_scheduler.py` covers Protocol conformance + snapshot delivery +
  dedup + cancellation + DaskScheduler skeleton.
- Pyright clean on `pyphi/parallel/`.
- Ruff clean.
- No `--no-verify` bypasses.
- Changelog fragment landed.

## Risks

| Risk | Mitigation |
|---|---|
| Snapshot-dedup logic crashes workers | `_apply_snapshot_if_changed` wraps `config.apply` in try/except + raises `WorkerConfigError` with snapshot id. Test covers identical-snapshot dedup + different-snapshot fresh apply |
| `LocalThreadScheduler` exposes shared-state races | P11 ships with thread scheduler safe under GIL-enabled runtimes only. No-GIL audit gated on P6b. CI runs hypothesis fast lane parametrized over both schedulers — race detection by random property failures |
| Cost-sampling miscalculates target_chunksize for workloads with long-tail outliers | Sampling 4 items spread across the iterable (not first 4) reduces this. If cost-sampling yields catastrophic chunk size, users override via `MapReduce(chunksize=...)` |
| Removing context-based heuristics changes runtime characteristics | Heuristics were never validated; removing them is a feature, not a regression. CI wall-time per fixture monitored manually during Phase 6 |
| `DaskScheduler` skeleton diverges from real cluster behavior | Documented as stub; raises with explicit message. P18 fills it in against the same Protocol — divergence caught at fill-in time |
| Frozen-formalism conversion (Phase 1) breaks IIT3/IIT4 dispatch | Pickle-roundtrip + golden 17/17 catch divergence. No semantic change beyond moving config from property to field |

## Files touched

```
pyphi/parallel/__init__.py                         # MapReduce facade adapted
pyphi/parallel/scheduler.py                        # NEW: Protocol + policies
pyphi/parallel/sampling.py                         # NEW: cost-sampling logic
pyphi/parallel/chunking.py                         # DELETED (dead code)
pyphi/parallel/tree.py                             # unchanged
pyphi/parallel/backends/__init__.py                # exports updated
pyphi/parallel/backends/local_process.py           # NEW (was local.py); LocalProcessScheduler
pyphi/parallel/backends/local_thread.py            # NEW: LocalThreadScheduler
pyphi/parallel/backends/dask.py                    # NEW: DaskScheduler skeleton
pyphi/parallel/backends/progress.py                # unchanged
pyphi/parallel/backends/local.py                   # DELETED (renamed to local_process.py)
pyphi/formalism/iit3/formalism.py                  # frozen dataclass conversion (Phase 1)
pyphi/formalism/iit4/formalism.py                  # frozen dataclass conversion (Phase 1)
pyphi/formalism/iit4/_2026.py                      # frozen dataclass conversion (Phase 1)
pyphi/conf/infrastructure.py                       # parallel_backend accepts "auto"/"thread"/"dask"
pyphi/formalism/iit4/__init__.py                   # TODO(4.0) parallelize markers filled
pyphi/compute/subsystem.py                         # TODO(4.0) parallelize marker filled
pyphi/models/ces.py                                # TODO(4.0) parallelize marker filled
pyphi/actual.py                                    # TODO(4.0) parallel-default marker resolved
test/test_scheduler.py                             # NEW
test/test_sampling.py                              # NEW
test/test_chunking.py                              # DELETED (heuristics tests for deleted code)
test/test_parallel.py                              # re-enabled in CI
ROADMAP.md                                         # P11 marked done; P18 follow-up entry (already added 2026-05-09)
changelog.d/p11-scheduler.feature.md               # NEW
```

## Deferred to follow-up

- Full `DaskScheduler` implementation + `dask-jobqueue` SLURM/PBS/LSF/SGE
  adapters → **P18** (post-2.0)
- `HTCondorScheduler` (`htcondor-dask` or `condor_submit` adapter) → **P18**
- No-GIL CI matrix entry running `PYTHON_GIL=0` → gated on **P6b**
- No-GIL safety audit of `DictCache` → gated on **P6b**
- The P9 loky/cloudpickle `BrokenProcessPool` curiosity (clear_all + per-instance
  Network purview-cache name registration interaction) — root cause never proven.
  P11's loky/cloudpickle boundary audit may resolve incidentally; if not, file as
  a curiosity per ROADMAP P9 deferred items
- Distributed cache backend (e.g. Redis for cross-process MICE caching) →
  Cross-cutting deferred item; surfaces during P11 if real workflow need
  emerges, otherwise post-2.0

## Open questions

None.
