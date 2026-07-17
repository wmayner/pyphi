# Wave 4: Parallel Determinism — Design

**Date:** 2026-07-16
**Source:** Whole-library review (REVIEW-2026-07-13.md), wave 4 — parallel
determinism. Four distinct defects (five review entries; two duplicate pairs).
All verified live at main tip `3959707e` by direct code read.

## Problem

Parallel evaluation must produce the same result as sequential evaluation.
Four defects break this:

1. **Thread backend collects in completion order under an active shortcircuit
   predicate** (`pyphi/parallel/backends/local_thread.py:79`). With
   `ordered=False` (the default on every SIA path), the collection loop
   iterates `as_completed(futures)`, so the truncated candidate prefix depends
   on OS thread scheduling. The MIP partition, tie set, and runner-up reported
   for a reducible system on the thread backend vary run to run. The process
   backend deliberately fixed exactly this (`local_process.py:274`, comment at
   266–273) by collecting in submission order whenever a shortcircuit
   predicate is active; the thread backend never received the fix.

2. **Thread backend's below-threshold sequential path ignores the shortcircuit
   predicate entirely** (`local_thread.py:67–71`). The fast path is a plain
   list comprehension: no truncation, no callback. A sweep below
   `sequential_threshold` returns every result where sequential semantics
   require stopping at the first triggering item, so a 63-item sweep behaves
   differently from a 65-item one.

3. **Tied-MICE selection is nondeterministic under parallel purview
   evaluation** (`pyphi/formalism/queries.py:314–329`). `find_mice` feeds
   `map_reduce` results directly into `resolve_ties.purviews` and takes
   `ties[0]`. `resolve()` preserves input order among survivors, and the input
   order is worker completion order — additionally permuted by the
   cost-balanced (LPT) chunking that `size_func` requests. When purviews tie
   at maximal φ (the case the ties machinery exists for), the chosen purview
   varies run to run and differs from sequential. The default
   `purview_tie_resolution` is the single strategy `"PHI"`, so genuine ties
   survive to the `ties[0]` selection.

4. **A worker exception leaves all remaining chunk futures running in the
   shared reusable executor** (`local_process.py:276` and `:293`).
   `future.result()` propagates the exception with no cancellation, so every
   outstanding chunk of the failed sweep keeps executing in loky's
   process-global pool — burning CPU and delaying the next `map_reduce` that
   reuses the pool. The thread backend has the same defect in worse form: the
   `with ThreadPoolExecutor(...)` block waits for all pending futures on exit,
   so the raising call *blocks* until every orphaned future finishes.

**Same-class site found during design** (not in the review):
`_find_mip_iit4` (`pyphi/formalism/iit4/formalism.py`, `map_reduce` over
`specified_states` ending in `resolve_ties.states(...)` and `ties[0]` at
line 267) collects with neither `ordered=True` nor a shortcircuit predicate,
so state-tie selection is completion-order-dependent in exactly the way
finding 3 describes. No `size_func` is involved, so `ordered=True` fixes it at
no cost.

## Approach

Surgical (chosen over deleting `as_completed` collection entirely): mirror the
process backend's deliberate fix in the thread backend, make the one
order-sensitive consumer that legitimately receives permuted results
(`find_mice`, because of `size_func`) restore canonical order itself, and add
cancellation on the exception path in both local backends. `as_completed`
collection is retained for plain unordered collection.

## Design

### 1. `ShortcircuitPolicy.active` (`pyphi/parallel/scheduler.py`)

There are two "no predicate" sentinels: `pyphi.parallel.false` (the public
`map_reduce` default, what backends actually receive) and
`_never_short_circuit` (`ShortcircuitPolicy`'s own default). Add a property so
"is a real predicate set?" is answered in one place:

```python
@property
def active(self) -> bool:
    """Whether a real short-circuit predicate is set."""
    from pyphi.parallel import false

    return self.func is not _never_short_circuit and self.func is not false
```

(Import inside the property mirrors the lazy-import pattern already used
between these modules; `local_process.py` imports `false` from
`pyphi.parallel` the same way.)

The process backend's own `is not false` check in `LocalMapReduce` is left
untouched: its callers pass the resolved `shortcircuit.func`, and the only
misclassification (`_never_short_circuit` treated as active) degrades to
submission-order collection, which is harmless.

### 2. Thread backend honors shortcircuit on both paths (`local_thread.py`)

**Below-threshold path** (fixes finding 2) — replace the list comprehension
with the collect-then-truncate loop, mirroring the process backend's
`_run_sequential` (callback receives the collected results):

```python
if len(materialized[0]) < chunking.sequential_threshold:
    results = []
    for args in zip(*materialized, strict=False):
        value = fn(*args, **map_kwargs)
        results.append(value)
        if shortcircuit.func(value):
            if shortcircuit.callback is not None:
                shortcircuit.callback(results)
            break
    return reducer(results)
```

**Parallel path** (fixes finding 1) — collect in submission order whenever
`ordered` is set or a shortcircuit predicate is active, with the same
rationale comment the process backend carries:

```python
# Collect in submission order when the caller asked for original order
# or a short-circuit predicate is active. When short-circuiting, the
# collected subset is truncated at the first triggering result, so
# completion order would make that subset — and any order-sensitive
# reduction over it (e.g. tie resolution among the surviving
# candidates) — depend on thread scheduling. Submission order yields
# the same prefix as sequential evaluation.
iterator: Iterable[Any] = (
    futures if ordered or shortcircuit.active else as_completed(futures)
)
```

### 3. Cancellation on the exception path (finding 4)

**Process backend** (`local_process.py`, `_run_parallel`) — wrap the entire
collection block (both the submission-order and `as_completed` loops) so a
worker exception cancels the outstanding futures before propagating:

```python
try:
    ...  # existing collection loops, unchanged
except BaseException:
    self._cancel_remaining(futures)
    raise
```

`BaseException` so `KeyboardInterrupt` also cancels. Already-running chunks
cannot be interrupted (loky `cancel()` returns False for running futures);
the win is that pending chunks never start, freeing the shared pool for the
next `map_reduce`.

**Thread backend** (`local_thread.py`) — the symmetric guard around its
collection loop, cancelling pending futures before the `with` block's
`shutdown(wait=True)` would otherwise run them all to completion:

```python
try:
    ...  # existing collection loop
except BaseException:
    for remaining in futures:
        if not remaining.done():
            remaining.cancel()
    raise
```

### 4. `find_mice` restores canonical purview order (finding 3)

In `pyphi/formalism/queries.py`, immediately after the `map_reduce` call,
re-sort the results into the enumeration order of `purviews_list` before tie
resolution:

```python
# Parallel evaluation returns results in completion / cost-bin order;
# restore the canonical purview enumeration order so tie resolution
# selects the same winner as sequential evaluation.
order = {purview: i for i, purview in enumerate(purviews_list)}
mip_results = sorted(mip_results, key=lambda ria: order[ria.purview])
```

Purviews are int-tuples (hashable) and each RIA's `purview` equals its input
purview, so the mapping is total. This makes the winner, tie set, and
`purview_margin` independent of backend, scheduling, and chunking, including
the LPT permutation from `size_func` — which is why the fix belongs at this
consumer rather than in the backends.

### 5. State-MIP site collects in input order (same-class)

In `_find_mip_iit4` (`pyphi/formalism/iit4/formalism.py`), pass
`ordered=True` to the `map_reduce` over `specified_states`, merged over the
threaded kwargs so a config-supplied value cannot conflict:

```python
mips = map_reduce(
    ...,
    **{**parallel_kwargs, "ordered": True},
)
```

No `size_func` is used at this site, so input-order collection is free.

## Out of scope (recorded findings)

- **Relations iteration order.** `all_relations` yields `map_reduce` results
  in completion / cost-bin order, but both consumers pour the yield directly
  into `ConcreteRelations`, a frozenset whose iteration order is driven by
  content-based hashes (`Relation.__hash__` → `Distinction.__hash__` over
  int-tuples; no strings, so no `PYTHONHASHSEED` sensitivity). Insertion
  order can leak into iteration order only through CPython hash-bucket
  collision probe order; the only numeric consumers of that order are
  `ConcreteRelations._sum_phi` / `_apportioned_sum_phi` (plain `sum()`,
  order-sensitive at the last ulp). Judged acceptable; a `math.fsum`
  hardening was considered and declined.
- **Backend-level always-submission-order collection** (deleting
  `as_completed`) was considered and declined in favor of the surgical
  mirror.
- The dask scheduler is a stub (`NotImplementedError`) — nothing to fix.

## Tests (TDD: each written first, failing, per finding)

New file `test/parallel/test_parallel_determinism.py` unless a test fits an
existing file's charter:

1. **Thread sub-threshold honors shortcircuit** (finding 2, deterministic —
   no timing): `map_reduce` over `[3, 0, 2]` with `backend="thread"`,
   `sequential_threshold=10`, falsy-shortcircuit → result `[3, 0]`; the map
   function is called exactly on `[3, 0]`; the callback fires once with the
   collected list.
2. **Thread parallel path collects submission-order prefix under
   shortcircuit** (finding 1): items with contrived delays chosen so the
   triggering item completes first but sits late in submission order
   (review-repro shape: delays `[0.5, 0.4, 0.3, 0.2, 0.1]`, values
   `[1, 1, 0, 1, 0]`, `sequential_threshold=1`); assert the collected prefix
   equals the sequential prefix `[1, 1, 0]`. Deterministic after the fix,
   because collection no longer depends on completion timing.
3. **`ShortcircuitPolicy.active`**: default policy and a policy carrying
   `pyphi.parallel.false` are inactive; a real predicate is active.
4. **`find_mice` winner is order-independent** (finding 3): monkeypatch
   `queries.map_reduce` to evaluate sequentially and return the results
   reversed; on a genuine φ-tie (`examples.iit4_2023_fig6a_system()`,
   CAUSE, mechanism `(0,)`, four purviews tied at φ ≈ 0.2448, pinned to the
   IIT 4.0 (2023) preset) the returned MICE's purview and tie set equal the
   unpatched sequential result.
5. **Worker exception cancels pending chunks** (finding 4, process):
   `LocalMapReduce` with `chunksize=1` over ~32 items where item 0 raises
   immediately and the rest sleep briefly; `run()` raises the worker's
   `ValueError` AND at least one future in `mr._futures` reports
   `cancelled()`.
6. **Worker exception cancels pending futures** (finding 4, thread): same
   shape through `map_reduce(backend="thread")` with a shared call-counter;
   the call raises and strictly fewer than all items were ever executed.
7. **State-MIP site passes `ordered=True`** (same-class site): monkeypatch
   the `map_reduce` name used by `pyphi/formalism/iit4/formalism.py` to
   capture kwargs while delegating to the real function; drive `find_mip`
   on a small system and assert the captured call used `ordered=True`.

Existing guards stay green: `test/parallel/test_parallel_equals_sequential.py`
(process), `test/parallel/test_thread_backend_equals_sequential.py` (thread),
and the Hypothesis map_reduce tests (which assert set-equality for
unordered collection, unaffected by the narrowed use of `as_completed`).

## Verification

- Fast lane in the worktree during development.
- Full pathless `uv run pytest` in the worktree before merge (log-file
  summary read, not exit codes).
- Full pathless `uv run pytest` in the main tree after merge.

## Changelog fragments

Three `fix` fragments: thread-backend shortcircuit determinism (findings
1–2), tie-selection determinism under parallel evaluation (finding 3 + the
state-MIP site), exception-path cancellation (finding 4).
