# Unify the Parallel Engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the one-shot `MapReduce` class with a single `pyphi.parallel.map_reduce()` function over the Scheduler Protocol, delete the vestigial tree abstraction, and dedupe plumbing — without changing any computed result.

**Architecture:** `map_reduce()` is a module-level function that builds `ChunkingPolicy`/`ProgressPolicy`/`ShortcircuitPolicy`, resolves a scheduler from `config.infrastructure.parallel_backend` (or the explicit `backend=`), and delegates to that scheduler's `map_reduce(...)`. The Scheduler Protocol and its process/thread backends remain the engine; the tree (`tree.py`) and the `MapReduce` class are removed.

**Tech Stack:** Python 3.12+, loky (process pool), `concurrent.futures` (thread pool), `more_itertools`, pytest + Hypothesis, uv.

## Global Constraints

- Python 3.12+ only; no backward-compat shims. `MapReduce` is removed, not aliased.
- **No computed result, measure, or config default may change.** Chunking never affects results.
- Run commands with `uv run`. Full verification is `uv run pytest` with **no path argument** (so `pyphi/` doctests run).
- Primary correctness guards, run after every task: N2 invariant (`test/test_parallel_equals_sequential.py`), golden suite (`test/test_golden_regression.py`), perf gate (`test/test_perf_counters.py`).
- The public function keeps the codebase's existing parameter names `reduce_func` / `reduce_kwargs` (not the Protocol's `reducer`); the function binds them into the scheduler's `reducer` internally.
- Commit trailer on every commit:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_012dtSzF2YgDjGpFC9mA47ve
  ```
- Pre-commit runs ruff + pyright and aborts the commit when ruff reformats; re-`git add` the named files and re-commit. Never `--no-verify`. `scripts/` is not T201-exempt; `test/` is pyright-excluded.
- Stage only files this plan touches (concurrent instances share branch `2.0`); never `git add -A`.

---

## File Structure

- `pyphi/parallel/__init__.py` — **add** `map_reduce()` (Task 1); **delete** `MapReduce` class (Task 7); helper home after dedup (Task 9).
- `pyphi/parallel/backends/local_process.py` — `LocalProcessScheduler` loses the tree (Task 8); `LocalMapReduce` loses tree params (Task 8); imports the deduped helpers (Task 9).
- `pyphi/parallel/backends/local_thread.py` — unchanged in behavior; already flat and policy-based.
- `pyphi/parallel/scheduler.py` — `default_scheduler` gains an explicit-backend resolver (Task 1).
- `pyphi/parallel/tree.py` — **deleted** (Task 8).
- `pyphi/conf/_helpers.py` — remove tree kwargs from `PARALLEL_KWARGS` (Task 8).
- Call-sites (Tasks 3–6): `substrate.py`, `relations.py`, `formalism/queries.py`, `formalism/iit3/__init__.py`, `formalism/iit4/__init__.py`, `formalism/iit4/formalism.py`, `formalism/actual_causation/compute.py`, `macro/search.py`.
- Tests: `test/test_parallel.py` rewritten (Task 10), `test/test_scheduler.py` extended (Task 1, 10), `test/test_tree.py` **deleted** (Task 8).

---

## Task 1: Add `map_reduce()` function (class untouched)

**Files:**
- Modify: `pyphi/parallel/__init__.py` (add function + reducer binder)
- Modify: `pyphi/parallel/scheduler.py` (`default_scheduler` accepts an optional backend override)
- Test: `test/test_scheduler.py`

**Interfaces:**
- Produces: `pyphi.parallel.map_reduce(fn, items, *more_items, reduce_func=_flatten, reduce_kwargs=None, parallel=True, ordered=False, total=None, chunksize=None, sequential_threshold=1, shortcircuit_func=false, shortcircuit_callback=None, shortcircuit_callback_args=None, progress=None, desc=None, map_kwargs=None, backend="auto") -> Any`
- Consumes: existing `default_scheduler()`, `ChunkingPolicy`, `ProgressPolicy`, `ShortcircuitPolicy`, and the module helpers `_flatten`, `false`, `_map_sequential`, `get`.

- [ ] **Step 1: Write the failing tests** in `test/test_scheduler.py`:

```python
def test_map_reduce_sequential_matches_builtin():
    from pyphi.parallel import map_reduce
    out = map_reduce(lambda x: x * 2, [1, 2, 3], parallel=False)
    assert sorted(out) == [2, 4, 6]


def test_map_reduce_reduce_func_min_with_kwargs():
    from pyphi.parallel import map_reduce
    # empty input + min(default=...) must return the default, mirroring AC usage
    out = map_reduce(lambda x: x, [], reduce_func=min,
                     reduce_kwargs={"default": 99}, parallel=False)
    assert out == 99


def test_map_reduce_parallel_equals_sequential():
    from pyphi.parallel import map_reduce
    import pyphi
    items = list(range(50))
    with pyphi.config.override(parallel=True):
        par = map_reduce(lambda x: x * x, items, chunksize=8)
    seq = map_reduce(lambda x: x * x, items, parallel=False)
    assert sorted(par) == sorted(seq)


def test_map_reduce_backend_thread_routes_to_thread_scheduler():
    from pyphi.parallel import map_reduce
    out = map_reduce(lambda x: x + 1, [1, 2, 3], backend="thread",
                     sequential_threshold=1, chunksize=1)
    assert sorted(out) == [2, 3, 4]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest test/test_scheduler.py -k map_reduce -x -q`
Expected: FAIL — `ImportError: cannot import name 'map_reduce'`.

- [ ] **Step 3: Add an explicit-backend resolver** to `pyphi/parallel/scheduler.py`. Change `default_scheduler()` to accept an optional override and keep the no-arg behavior:

```python
def default_scheduler(backend: str | None = None) -> Scheduler:
    """Return the scheduler for ``backend`` (or ``config.parallel_backend``).

    ``"auto"`` resolves to ``LocalThreadScheduler`` on free-threaded runtimes
    and ``LocalProcessScheduler`` otherwise.
    """
    import sys

    from pyphi.conf import config

    if backend is None:
        backend = config.infrastructure.parallel_backend
    if backend == "auto":
        gil_enabled = getattr(sys, "_is_gil_enabled", lambda: True)()
        if not gil_enabled:
            from pyphi.parallel.backends.local_thread import LocalThreadScheduler

            return LocalThreadScheduler()
        from pyphi.parallel.backends.local_process import LocalProcessScheduler

        return LocalProcessScheduler()
    if backend in ("local", "process"):
        from pyphi.parallel.backends.local_process import LocalProcessScheduler

        return LocalProcessScheduler()
    if backend == "thread":
        from pyphi.parallel.backends.local_thread import LocalThreadScheduler

        return LocalThreadScheduler()
    if backend == "dask":
        from pyphi.parallel.backends.dask import DaskScheduler

        return DaskScheduler()
    raise ValueError(f"unknown parallel_backend: {backend!r}")
```

- [ ] **Step 4: Add the function and reducer binder** to `pyphi/parallel/__init__.py` (after the existing helpers, before or in place of the `MapReduce` class — leave the class for now):

```python
def _bind_reducer(reduce_func: Callable, reduce_kwargs: dict | None) -> Callable:
    """Adapt MapReduce-style (reduce_func, reduce_kwargs) to a 1-arg reducer."""
    reduce_kwargs = reduce_kwargs or {}
    if reduce_func is _flatten:
        return lambda results: _flatten(results, branch=False)
    if reduce_kwargs:
        return lambda results: reduce_func(results, **reduce_kwargs)
    return reduce_func


def map_reduce(
    fn: Callable,
    items: Iterable,
    *more_items: Iterable,
    reduce_func: Callable = _flatten,
    reduce_kwargs: dict | None = None,
    parallel: bool = True,
    ordered: bool = False,
    total: int | None = None,
    chunksize: int | None = None,
    sequential_threshold: int = 1,
    shortcircuit_func: Callable = false,
    shortcircuit_callback: Callable | None = None,
    shortcircuit_callback_args: Any = None,
    progress: bool | None = None,
    desc: str | None = None,
    map_kwargs: dict | None = None,
    backend: str = "auto",
) -> Any:
    """Map ``fn`` over ``items`` (zipped with ``more_items``) and reduce.

    Runs in parallel through the scheduler selected by ``backend`` (or
    ``config.infrastructure.parallel_backend``). With ``parallel=False`` it
    runs serially in-process. ``reduce_func`` defaults to flattening the
    per-item results into a list.
    """
    iterables = (items, *more_items)
    show_progress = fallback(progress, config.infrastructure.progress_bars)
    resolved_total = fallback(try_len(*iterables), total)

    if not parallel:
        results = _map_sequential(fn, *iterables, **(map_kwargs or {}))
        if show_progress:
            results = tqdm(results, total=resolved_total, desc=desc)
        results = get(
            results,
            shortcircuit_func=shortcircuit_func,
            shortcircuit_callback=shortcircuit_callback,
            shortcircuit_callback_args=shortcircuit_callback_args,
        )
        return _reduce(list(results), reduce_func, reduce_kwargs or {}, branch=False)

    from .scheduler import ChunkingPolicy
    from .scheduler import ProgressPolicy
    from .scheduler import ShortcircuitPolicy
    from .scheduler import default_scheduler

    scheduler = default_scheduler(None if backend == "auto" else backend)
    return scheduler.map_reduce(
        fn,
        *iterables,
        reducer=_bind_reducer(reduce_func, reduce_kwargs),
        chunking=ChunkingPolicy(
            chunksize=chunksize, sequential_threshold=sequential_threshold
        ),
        progress=ProgressPolicy(
            enabled=show_progress, desc=desc or "", total=resolved_total
        ),
        shortcircuit=ShortcircuitPolicy(
            func=shortcircuit_func, callback=shortcircuit_callback
        ),
        ordered=ordered,
        map_kwargs=map_kwargs,
    )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest test/test_scheduler.py -k map_reduce -x -q`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add pyphi/parallel/__init__.py pyphi/parallel/scheduler.py test/test_scheduler.py
git commit -m "Add map_reduce() function over the Scheduler Protocol"
```

---

## Task 2: Migrate the default-flatten call-sites (relations, substrate, queries, iit3 cuts)

These pass no `reduce_func` (default flatten) or use `**pkwargs`/`**kwargs`. Migration is mechanical: `MapReduce(...).run()` → `map_reduce(...)`, drop `.run()`, keep all kwargs verbatim.

**Files:**
- Modify: `pyphi/relations.py:249`, `pyphi/substrate.py:729`, `pyphi/formalism/queries.py:118`, `pyphi/formalism/iit3/__init__.py:336`
- Modify imports: replace `MapReduce` import with `map_reduce` where these modules import it.

- [ ] **Step 1: Update the import in each file.** Find the existing `from pyphi.parallel import MapReduce` (or `from ..parallel import MapReduce`) and change `MapReduce` → `map_reduce`. Keep the import path.

- [ ] **Step 2: `relations.py:249`** — replace:

```python
    result = MapReduce(
        worker,
        combinations,
        desc="Evaluating relations",
        **pkwargs,  # type: ignore[arg-type]  # parallel_kwargs contains MapReduce params
    ).run()
```
with:
```python
    result = map_reduce(
        worker,
        combinations,
        desc="Evaluating relations",
        **pkwargs,  # type: ignore[arg-type]  # parallel_kwargs contains map_reduce params
    )
```

- [ ] **Step 3: `substrate.py:729`** — replace the `MapReduce(sia_fn, iterable, total=..., map_kwargs=map_kwargs, desc="Evaluating complexes", **pkwargs).run()` with the same call using `map_reduce(...)` and no `.run()`. Keep `assert result is not None` and `return result`.

- [ ] **Step 4: `formalism/queries.py:118`** — replace `candidate_mips = MapReduce(_eval, partitions, shortcircuit_func=_utils.is_falsy, desc="Evaluating mechanism partitions", **parallel_kwargs).run()` with `map_reduce(...)` (drop `.run()`). Update the adjacent assert message text `"MapReduce.run() should not return None"` → `"map_reduce() should not return None"`.

- [ ] **Step 5: `formalism/iit3/__init__.py:336`** — replace `candidates = MapReduce(evaluate_partition, cuts, map_kwargs={...}, shortcircuit_func=utils.is_falsy, desc="Evaluating cuts", **kwargs).run()` with `map_reduce(...)` (drop `.run()`).

- [ ] **Step 6: Verify guards green**

Run: `uv run pytest test/test_parallel_equals_sequential.py test/test_golden_regression.py test/test_perf_counters.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add pyphi/relations.py pyphi/substrate.py pyphi/formalism/queries.py pyphi/formalism/iit3/__init__.py
git commit -m "Migrate flatten-reducer call-sites to map_reduce()"
```

---

## Task 3: Migrate the reduce_func call-sites (iit3 concepts, actual causation)

These pass an explicit `reduce_func` (and AC also `reduce_kwargs`). Same mechanical change; verify the reducer parity explicitly.

**Files:**
- Modify: `pyphi/formalism/iit3/__init__.py:145`, `pyphi/formalism/actual_causation/compute.py:555`

- [ ] **Step 1: Update imports** (`MapReduce` → `map_reduce`) in both files.

- [ ] **Step 2: `iit3/__init__.py:145`** — replace `concepts = MapReduce(compute_concept, mechanisms, map_kwargs={...}, reduce_func=reduce_func, desc="Computing concepts", total=total, **parallel_kwargs).run()` with `map_reduce(...)` (drop `.run()`), keeping `reduce_func=reduce_func`.

- [ ] **Step 3: `actual_causation/compute.py:555`** — replace `result = MapReduce(_evaluate_partition, cuts, map_kwargs={...}, reduce_func=min, reduce_kwargs={"default": _null_ac_sia(...)}, ...).run()` with `map_reduce(...)` (drop `.run()`), keeping `reduce_func=min` and `reduce_kwargs={...}`.

- [ ] **Step 4: Verify guards + the two formalisms green**

Run: `uv run pytest test/test_parallel_equals_sequential.py test/test_golden_regression.py test/test_actual_causation.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyphi/formalism/iit3/__init__.py pyphi/formalism/actual_causation/compute.py
git commit -m "Migrate reduce_func call-sites (IIT 3.0 concepts, actual causation) to map_reduce()"
```

---

## Task 4: Migrate the IIT 4.0 + macro call-sites

**Files:**
- Modify: `pyphi/formalism/iit4/__init__.py:1092`, `pyphi/formalism/iit4/formalism.py:248`, `pyphi/macro/search.py:291`

- [ ] **Step 1: Update imports** (`MapReduce` → `map_reduce`) in all three files.

- [ ] **Step 2: `iit4/__init__.py:1092`** — replace `sias = MapReduce(evaluate_partition, partitions, map_kwargs={...}, shortcircuit_func=utils.is_falsy, desc="Evaluating partitions", **parallel_kwargs).run()` with `map_reduce(...)` (drop `.run()`).

- [ ] **Step 3: `iit4/formalism.py:248`** — replace `mips = MapReduce(partial(_find_mip_single_state, system), specified_states, map_kwargs={...}, desc="Finding MIP for maximum intrinsic information states", **parallel_kwargs).run()` with `map_reduce(...)` (drop `.run()`).

- [ ] **Step 4: `macro/search.py:291`** — replace `phis = MapReduce(_evaluate_one, pending, **pkwargs).run()` with `phis = map_reduce(_evaluate_one, pending, **pkwargs)`.

- [ ] **Step 5: Verify guards + iit4 + macro green**

Run: `uv run pytest test/test_parallel_equals_sequential.py test/test_golden_regression.py test/test_macro.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyphi/formalism/iit4/__init__.py pyphi/formalism/iit4/formalism.py pyphi/macro/search.py
git commit -m "Migrate IIT 4.0 and macro call-sites to map_reduce()"
```

---

## Task 5: Migrate `queries.py:280` (resolve the variable-name shadow)

This call-site assigns a local named `map_reduce`, which now shadows the function. Restructure to call the function directly.

**Files:**
- Modify: `pyphi/formalism/queries.py:280`

- [ ] **Step 1: Update the import** (`MapReduce` → `map_reduce`) if not already done by Task 2.

- [ ] **Step 2:** Replace:

```python
    map_reduce = MapReduce(
        _find_mip,
        purviews_list,
        total=len(purviews_list),
        desc="Evaluating purviews",
        **parallel_kwargs,
    )

    all_mice = map(mice_class, map_reduce.run())  # type: ignore[arg-type]
```
with:
```python
    mip_results = map_reduce(
        _find_mip,
        purviews_list,
        total=len(purviews_list),
        desc="Evaluating purviews",
        **parallel_kwargs,
    )

    all_mice = map(mice_class, mip_results)  # type: ignore[arg-type]
```

- [ ] **Step 3: Verify guards green**

Run: `uv run pytest test/test_parallel_equals_sequential.py test/test_golden_regression.py -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add pyphi/formalism/queries.py
git commit -m "Migrate purview-evaluation call-site to map_reduce()"
```

---

## Task 6: Confirm no remaining `MapReduce` references

**Files:** none (verification task)

- [ ] **Step 1: Search for remaining production references**

Run: `rg -n "MapReduce" pyphi/ | rg -v "LocalMapReduce|class MapReduce|map_reduce"`
Expected: only the `class MapReduce` definition and its docstring remain in `pyphi/parallel/__init__.py`. If any call-site remains, migrate it using the Task 2 pattern and re-run.

- [ ] **Step 2:** No commit (verification only). If a stray was fixed, commit it with `git add <file> && git commit -m "Migrate remaining map_reduce call-site"`.

---

## Task 7: Delete the `MapReduce` class

**Files:**
- Modify: `pyphi/parallel/__init__.py` (remove the class, its `_resolve_backend`, `_repr_attrs`, `__repr__`, `_run_parallel`, `_run_sequential`, `run`, and `cancel_all` if now unused)

- [ ] **Step 1: Check what the class still pulls in.** Before deleting, confirm `cancel_all`, `get`, `shortcircuit` usage:

Run: `rg -n "cancel_all|MapReduce|\.run\(\)" pyphi/parallel/__init__.py`
Keep `get`, `shortcircuit`, `_map_sequential`, `_reduce`, `false`, `_flatten`, `_bind_reducer`, `map_reduce`, `get_num_processes` (still used). Delete `cancel_all` only if no longer referenced anywhere (`rg -n cancel_all pyphi/`).

- [ ] **Step 2: Remove the `class MapReduce:` block** (from `class MapReduce:` through the end of its `run` method). Update the module docstring header (lines 1–22) to describe `map_reduce()` instead of "MapReduce interface".

- [ ] **Step 3: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS (collection no longer imports `MapReduce`; all call-sites use `map_reduce`).

- [ ] **Step 4: Commit**

```bash
git add pyphi/parallel/__init__.py
git commit -m "Remove the MapReduce class (superseded by map_reduce())"
```

---

## Task 8: Remove the tree

**Files:**
- Delete: `pyphi/parallel/tree.py`, `test/test_tree.py`
- Modify: `pyphi/parallel/backends/local_process.py` (drop tree from `LocalProcessScheduler.map_reduce` and `LocalMapReduce`)
- Modify: `pyphi/conf/_helpers.py` (drop tree kwargs from `PARALLEL_KWARGS`)

**Interfaces:**
- `LocalMapReduce.__init__` loses `constraints` and `tree`; it keeps `chunksize` and gains nothing. Its `run()` parallel-gate becomes: sequential when `total < sequential_threshold` or fewer than two chunks form.

- [ ] **Step 1: Simplify `LocalProcessScheduler.map_reduce`** (`backends/local_process.py:365-430`). Replace the `get_constraints(...)` + `constraints.simulate()` block and the `constraints=`/`tree=` arguments to `LocalMapReduce`. The chunksize already comes from `compute_chunksize(...)`; pass it directly. Concretely, delete:

```python
        from pyphi.parallel.tree import get_constraints
        ...
        constraints = get_constraints(
            total=total,
            chunksize=chunksize,
            sequential_threshold=chunking.sequential_threshold,
        )
        tree = constraints.simulate()
```
and remove `constraints=constraints, tree=tree,` from the `LocalMapReduce(...)` call, adding `sequential_threshold=chunking.sequential_threshold,` instead.

- [ ] **Step 2: Update `LocalMapReduce`** (`backends/local_process.py`): remove the `constraints` and `tree` parameters from `__init__`; add and store `sequential_threshold: int = 1`. Add a gate helper and use it in `run()` in place of `if self.tree.depth <= 1:`:

```python
    def _should_run_parallel(self) -> bool:
        """Parallelize only when there is more than one chunk of work."""
        if self.total is None:
            return True  # unknown length; let the executor chunk and dispatch
        if self.total < self.sequential_threshold:
            return False
        if self.chunksize and self.total <= self.chunksize:
            return False  # a single chunk → no parallel benefit
        return True
```
Then in `run()` replace `if self.tree.depth <= 1: return self._run_sequential()` with:
```python
        if not self._should_run_parallel():
            return self._run_sequential()
        return self._run_parallel()
```
Keep `_get_chunks`/`_run_parallel`/`_run_sequential` otherwise unchanged (they already use `self.chunksize`). Remove the `from pyphi.parallel.tree import TreeConstraints` and `TreeSpec` imports (lines 29–30).

- [ ] **Step 3: Remove tree kwargs from `PARALLEL_KWARGS`** in `pyphi/conf/_helpers.py` — delete the four lines `"max_depth"`, `"max_size"`, `"max_leaves"`, `"branch_factor"`.

- [ ] **Step 4: Delete the modules**

```bash
git rm pyphi/parallel/tree.py test/test_tree.py
```

- [ ] **Step 5: Confirm no remaining tree imports**

Run: `rg -n "from .*tree import|parallel.tree|TreeConstraints|TreeSpec|get_constraints" pyphi/ test/`
Expected: no matches.

- [ ] **Step 6: Run the guards then the full suite**

Run: `uv run pytest test/test_parallel_equals_sequential.py test/test_golden_regression.py test/test_perf_counters.py -q && uv run pytest -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add pyphi/parallel/backends/local_process.py pyphi/conf/_helpers.py
git commit -m "Remove the vestigial parallel execution tree"
```

---

## Task 9: Deduplicate shared helpers

`get_num_processes`, `false`, `_flatten`, `_map_sequential`, and `_reduce` are defined in both `pyphi/parallel/__init__.py` and `pyphi/parallel/backends/local_process.py`.

**Files:**
- Modify: `pyphi/parallel/backends/local_process.py` (import the helpers instead of redefining), `pyphi/parallel/__init__.py` (sole home)

- [ ] **Step 1: Delete the duplicate definitions** of `get_num_processes`, `false`, `_flatten`, `_map_sequential`, and `_reduce` from `backends/local_process.py`. Replace with imports from the package root:

```python
from pyphi.parallel import _flatten
from pyphi.parallel import _map_sequential
from pyphi.parallel import _reduce
from pyphi.parallel import false
from pyphi.parallel import get_num_processes
```
If this creates a circular import (the backend is imported lazily inside `default_scheduler`, so package-level import of the backend does not run at `pyphi.parallel` import time), keep the imports module-level; otherwise move them inside the functions that use them. Verify with Step 2.

- [ ] **Step 2: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS. If an ImportError appears, move the five imports inside the methods that use them (deferred import) and re-run.

- [ ] **Step 3: Commit**

```bash
git add pyphi/parallel/__init__.py pyphi/parallel/backends/local_process.py
git commit -m "Deduplicate parallel helper functions"
```

---

## Task 10: Rewrite `test/test_parallel.py`; extend `test/test_scheduler.py`

**Files:**
- Modify: `test/test_parallel.py` (construct `map_reduce` calls instead of `MapReduce`; drop tree-kwarg fuzz)
- Modify: `test/test_scheduler.py` (cover the function's policy translation + backend resolution)

- [ ] **Step 1: Inspect the current `test/test_parallel.py`** to find every `MapReduce(` construction and the Hypothesis strategy drawing `max_depth`/`branch_factor`/`max_leaves` (around lines 181–205).

Run: `rg -n "MapReduce\(|max_depth|branch_factor|max_leaves|\.run\(\)" test/test_parallel.py`

- [ ] **Step 2: Replace each `MapReduce(...).run()`** with `map_reduce(...)` and delete the drawn `max_depth`/`branch_factor`/`max_leaves`/`max_size` keys from the kwargs strategy (they are no longer parameters). Keep all result-invariance, short-circuit, and ordered/unordered assertions.

- [ ] **Step 3: Add a translation test** to `test/test_scheduler.py`:

```python
def test_map_reduce_passes_shortcircuit_through():
    from pyphi.parallel import map_reduce
    seen = []
    out = map_reduce(
        lambda x: x, [1, 2, 3, 4, 5],
        parallel=False,
        shortcircuit_func=lambda r: r == 3,
        shortcircuit_callback=lambda items: seen.append("stopped"),
    )
    assert 3 in out
    assert seen == ["stopped"]
```

- [ ] **Step 4: Run the parallel + scheduler tests**

Run: `uv run pytest test/test_parallel.py test/test_scheduler.py -q`
Expected: PASS.

- [ ] **Step 5: Full verification (no path argument — runs doctests)**

Run: `uv run pytest -q`
Expected: PASS, no failures, no collection errors.

- [ ] **Step 6: Commit**

```bash
git add test/test_parallel.py test/test_scheduler.py
git commit -m "Update parallel tests for the map_reduce() function"
```

---

## Final verification

- [ ] `rg -n "MapReduce|parallel/tree|TreeConstraints" pyphi/ test/` returns nothing (besides this plan/spec under `docs/`).
- [ ] `uv run pytest` (no path argument) is fully green, including `pyphi/` doctests.
- [ ] N2, goldens, and perf gate green.
- [ ] After all tasks, use superpowers:finishing-a-development-branch to complete.
