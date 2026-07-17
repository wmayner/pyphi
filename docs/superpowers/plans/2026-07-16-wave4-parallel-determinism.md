# Parallel Determinism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make parallel evaluation deterministic and equal to sequential evaluation: the thread backend honors the short-circuit predicate on both of its paths, tie selection no longer depends on worker completion order, and a worker exception cancels the remaining pending chunks.

**Architecture:** Surgical mirror of the process backend's deliberate submission-order fix into the thread backend, driven by a new `ShortcircuitPolicy.active` property; a canonical re-sort in `find_mice` (the one consumer that legitimately receives cost-bin-permuted results because it uses `size_func`); `ordered=True` at the IIT 4.0 state-MIP site; and `try/except BaseException` cancellation guards around both local backends' collection loops.

**Tech Stack:** Python 3.13+, pytest, `concurrent.futures`, loky (via joblib).

**Spec:** `docs/superpowers/specs/2026-07-16-wave4-parallel-determinism-design.md`

## Global Constraints

- Run everything with `uv run` (e.g. `uv run pytest`), from the worktree root `.claude/worktrees/wave4-parallel-determinism`.
- Never `git commit --no-verify`. If commit output shows only hook lines and `git status` shows `MM`, the formatter modified files: re-stage and re-commit. Check `git log --oneline -1` after every commit.
- Commit messages end with the two trailer lines shown in each commit step.
- No planning-artifact references (wave numbers, review file names, spec paths) in source code, docstrings, or changelog fragments.
- New/modified docstrings: NumPy style, final-state impersonal voice.
- Ruff traps: no unused lambda arguments (prefix `_`), no unused unpacks.

---

### Task 1: `ShortcircuitPolicy.active`

**Files:**
- Modify: `pyphi/parallel/scheduler.py:49-52` (the `ShortcircuitPolicy` dataclass)
- Test: `test/parallel/test_parallel_determinism.py` (new file)

**Interfaces:**
- Produces: `ShortcircuitPolicy.active -> bool` property — `True` iff `func` is neither of the two no-predicate sentinels (`pyphi.parallel.false`, `scheduler._never_short_circuit`). Task 2 branches on it.

- [ ] **Step 1: Write the failing test**

Create `test/parallel/test_parallel_determinism.py`:

```python
"""Determinism guards for the parallel layer.

Parallel evaluation must yield the same results as sequential evaluation:
short-circuit truncation must happen at the same submission-order prefix on
every backend, tie selection must not depend on worker completion order, and
a worker exception must not leave orphaned work running in the executors.
"""

from __future__ import annotations

from pyphi.parallel import false
from pyphi.parallel.scheduler import ShortcircuitPolicy


def test_shortcircuit_policy_active():
    assert not ShortcircuitPolicy().active
    assert not ShortcircuitPolicy(func=false).active
    assert ShortcircuitPolicy(func=lambda r: r == 0).active
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/parallel/test_parallel_determinism.py::test_shortcircuit_policy_active -v`
Expected: FAIL with `AttributeError: 'ShortcircuitPolicy' object has no attribute 'active'`

- [ ] **Step 3: Implement the property**

In `pyphi/parallel/scheduler.py`, replace:

```python
@dataclass(frozen=True)
class ShortcircuitPolicy:
    func: Callable[[Any], bool] = field(default=_never_short_circuit)
    callback: Callable[[Iterable[Any]], None] | None = None
```

with:

```python
@dataclass(frozen=True)
class ShortcircuitPolicy:
    func: Callable[[Any], bool] = field(default=_never_short_circuit)
    callback: Callable[[Iterable[Any]], None] | None = None

    @property
    def active(self) -> bool:
        """Whether a real short-circuit predicate is set.

        ``False`` when ``func`` is either no-predicate sentinel: this
        class's default, or :func:`pyphi.parallel.false` (the public
        ``map_reduce`` default that backends receive).
        """
        from pyphi.parallel import false

        return self.func is not _never_short_circuit and self.func is not false
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/parallel/test_parallel_determinism.py::test_shortcircuit_policy_active -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyphi/parallel/scheduler.py test/parallel/test_parallel_determinism.py
git commit -m "Expose whether a ShortcircuitPolicy carries a real predicate

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UB4jbV4bQ8Y7Eywt1TNgJe"
git log --oneline -1
```

---

### Task 2: Thread backend honors short-circuit on both paths

**Files:**
- Modify: `pyphi/parallel/backends/local_thread.py:67-91` (`LocalThreadScheduler.map_reduce`)
- Create: `changelog.d/thread-backend-shortcircuit.fix.md`
- Test: `test/parallel/test_parallel_determinism.py`

**Interfaces:**
- Consumes: `ShortcircuitPolicy.active` from Task 1.
- Produces: thread-backend `map_reduce` whose below-threshold path truncates at the first triggering result (callback receives the collected list) and whose parallel path collects in submission order whenever `ordered` is set or a predicate is active.

- [ ] **Step 1: Write the two failing tests**

In `test/parallel/test_parallel_determinism.py`, extend the imports (ruff's
isort ordering: stdlib first, then `pyphi`):

```python
from __future__ import annotations

import time

from pyphi.parallel import false
from pyphi.parallel import map_reduce
from pyphi.parallel.scheduler import ShortcircuitPolicy
```

and append:

```python
def test_thread_backend_sub_threshold_honors_shortcircuit():
    calls = []
    collected = []

    def record(x):
        calls.append(x)
        return x

    result = map_reduce(
        record,
        [3, 0, 2],
        parallel=True,
        backend="thread",
        sequential_threshold=10,
        shortcircuit_func=lambda r: r == 0,
        shortcircuit_callback=collected.append,
        progress=False,
    )
    assert result == [3, 0]
    assert calls == [3, 0]
    assert collected == [[3, 0]]


def test_thread_backend_shortcircuit_collects_submission_order_prefix():
    def slow_identity(delay, value):
        time.sleep(delay)
        return value

    delays = [0.5, 0.4, 0.3, 0.2, 0.1]
    values = [1, 1, 0, 1, 0]
    result = map_reduce(
        slow_identity,
        delays,
        values,
        parallel=True,
        backend="thread",
        sequential_threshold=1,
        shortcircuit_func=lambda r: r == 0,
        progress=False,
    )
    assert result == [1, 1, 0]
```

(The second test forces completion order to invert submission order: the
falsy item at index 4 finishes first. Before the fix, `as_completed`
collection returns `[0]`; after the fix the prefix equals the sequential
prefix `[1, 1, 0]` regardless of timing, because collection follows
submission order whenever a predicate is active.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/parallel/test_parallel_determinism.py -v -k thread_backend`
Expected: 2 FAILs — sub-threshold returns `[3, 0, 2]`; parallel path returns `[0]`.

- [ ] **Step 3: Implement both path fixes**

In `pyphi/parallel/backends/local_thread.py`, replace lines 67-91 (everything
from the sub-threshold `if` through the final `return reducer(results)`):

```python
        if len(materialized[0]) < chunking.sequential_threshold:
            results: list[Any] = []
            for args in zip(*materialized, strict=False):
                value = fn(*args, **map_kwargs)
                results.append(value)
                if shortcircuit.func(value):
                    if shortcircuit.callback is not None:
                        shortcircuit.callback(results)
                    break
            return reducer(results)

        results = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(fn, *args, **map_kwargs)
                for args in zip(*materialized, strict=False)
            ]
            # Collect in submission order when the caller asked for original
            # order or a short-circuit predicate is active. When
            # short-circuiting, the collected subset is truncated at the
            # first triggering result, so completion order would make that
            # subset — and any order-sensitive reduction over it (e.g. tie
            # resolution among the surviving candidates) — depend on thread
            # scheduling. Submission order yields the same prefix as
            # sequential evaluation.
            iterator: Iterable[Any] = (
                futures if ordered or shortcircuit.active else as_completed(futures)
            )
            for fut in iterator:
                value = fut.result()
                results.append(value)
                if shortcircuit.func(value):
                    for remaining in futures:
                        if not remaining.done():
                            remaining.cancel()
                    if shortcircuit.callback is not None:
                        shortcircuit.callback(futures)
                    break

        return reducer(results)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/parallel/test_parallel_determinism.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the thread-backend regression guards**

Run: `uv run pytest test/parallel/test_thread_backend_equals_sequential.py test/parallel/test_scheduler.py -v > /tmp/wave4-task2.log 2>&1; tail -5 /tmp/wave4-task2.log`
Expected: summary line reports all passed.

- [ ] **Step 6: Write the changelog fragment**

Create `changelog.d/thread-backend-shortcircuit.fix.md`:

```
The thread backend now honors the short-circuit predicate on its
below-threshold sequential path and collects short-circuited results in
submission order, so truncated sweeps match sequential evaluation (and the
process backend) instead of varying with thread scheduling.
```

- [ ] **Step 7: Commit**

```bash
git add pyphi/parallel/backends/local_thread.py test/parallel/test_parallel_determinism.py changelog.d/thread-backend-shortcircuit.fix.md
git commit -m "Honor the shortcircuit predicate on both thread-backend paths

The below-threshold path never consulted the predicate, returning every
result where sequential semantics require truncation. The parallel path
collected in completion order under an active predicate, making the
truncated candidate prefix depend on thread scheduling; collect in
submission order instead, matching the process backend.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UB4jbV4bQ8Y7Eywt1TNgJe"
git log --oneline -1
```

---

### Task 3: Worker exception cancels pending futures in both backends

**Files:**
- Modify: `pyphi/parallel/backends/local_process.py:265-307` (`LocalMapReduce._run_parallel` collection block)
- Modify: `pyphi/parallel/backends/local_thread.py` (the collection loop written in Task 2)
- Create: `changelog.d/parallel-exception-cancels-futures.fix.md`
- Test: `test/parallel/test_parallel_determinism.py`

**Interfaces:**
- Consumes: Task 2's thread-backend collection loop.
- Produces: both backends cancel pending futures before propagating a worker exception. `LocalMapReduce._futures` (existing attribute) is how the process-backend test observes cancellation.

- [ ] **Step 1: Write the two failing tests**

In `test/parallel/test_parallel_determinism.py`, add `import pytest` to the
imports (between the stdlib and `pyphi` groups). Then append —
`_boom_or_sleep` must be module-level so loky can pickle it cheaply:

```python
def _boom_or_sleep(x):
    if x == 0:
        raise ValueError("boom")
    time.sleep(0.5)
    return x


def test_worker_exception_cancels_pending_process_chunks():
    from pyphi.parallel.backends.local_process import LocalMapReduce

    mr = LocalMapReduce(
        map_func=_boom_or_sleep,
        iterables=(list(range(32)),),
        reduce_func=list,
        reduce_kwargs={},
        chunksize=1,
        progress=False,
        total=32,
    )
    with pytest.raises(ValueError, match="boom"):
        mr.run()
    assert any(future.cancelled() for future in mr._futures)


def test_worker_exception_cancels_pending_thread_futures():
    calls = []

    def boom_or_sleep(x):
        calls.append(x)
        if x == 0:
            raise ValueError("boom")
        time.sleep(0.3)
        return x

    with pytest.raises(ValueError, match="boom"):
        map_reduce(
            boom_or_sleep,
            list(range(32)),
            parallel=True,
            backend="thread",
            sequential_threshold=1,
            progress=False,
        )
    assert len(calls) < 32
```

(Both are robust to timing: the raising item is submitted first and fails in
under a millisecond while every worker slot is occupied by a sleeper, so far
more chunks are pending than workers when the exception surfaces. Before the
fix, the process test finds no cancelled future, and the thread pool's
context manager waits for all 32 calls to run, so `calls == 32`.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/parallel/test_parallel_determinism.py -v -k exception`
Expected: 2 FAILs — `any(... cancelled ...)` is False; `len(calls) < 32` is False.

- [ ] **Step 3: Implement the process-backend guard**

In `pyphi/parallel/backends/local_process.py`, `_run_parallel`: wrap the two
collection loops in `try/except`. Replace everything from the comment block
at line 266 through the end of the `else` loop (line 307) with:

```python
        # Collect results in order of completion, unless the caller asked for
        # original order or a short-circuit predicate is active. When
        # short-circuiting, the collected subset is truncated at the first
        # triggering result, so completion order would make that subset — and
        # any order-sensitive reduction over it (e.g. tie resolution among the
        # surviving candidates) — depend on worker scheduling. Collecting in
        # submission order instead yields the same prefix as sequential
        # evaluation, keeping the parallel result deterministic.
        # A worker exception cancels the pending chunks before propagating:
        # the executor is a process-global reusable pool, so orphaned chunks
        # would keep burning CPU and delay the next map-reduce.
        try:
            if self.ordered or self.shortcircuit_func is not false:
                for future in futures:
                    chunk_results = future.result()
                    results.extend(chunk_results)
                    # Update progress bar
                    if self.progress_bar is not None:
                        self.progress_bar.update(len(chunk_results))
                    # Check for short-circuit in any of the chunk results
                    for r in chunk_results:
                        if self.shortcircuit_func(r):
                            short_circuited = True
                            self._cancel_remaining(futures)
                            if self.shortcircuit_callback is not None:
                                self.shortcircuit_callback(futures)
                            break
                    if short_circuited:
                        break
            else:
                for future in as_completed(futures):
                    chunk_results = future.result()
                    results.extend(chunk_results)
                    # Update progress bar
                    if self.progress_bar is not None:
                        self.progress_bar.update(len(chunk_results))
                    # Check for short-circuit in any of the chunk results
                    for r in chunk_results:
                        if self.shortcircuit_func(r):
                            short_circuited = True
                            self._cancel_remaining(futures)
                            if self.shortcircuit_callback is not None:
                                self.shortcircuit_callback(futures)
                            break
                    if short_circuited:
                        break
        except BaseException:
            self._cancel_remaining(futures)
            raise
```

- [ ] **Step 4: Implement the thread-backend guard**

In `pyphi/parallel/backends/local_thread.py`, wrap the collection loop from
Task 2 (leave the `iterator` assignment and its comment unchanged):

```python
            # A worker exception cancels the pending futures before
            # propagating; otherwise the executor's shutdown would block
            # until every orphaned future had run to completion.
            try:
                for fut in iterator:
                    value = fut.result()
                    results.append(value)
                    if shortcircuit.func(value):
                        for remaining in futures:
                            if not remaining.done():
                                remaining.cancel()
                        if shortcircuit.callback is not None:
                            shortcircuit.callback(futures)
                        break
            except BaseException:
                for remaining in futures:
                    if not remaining.done():
                        remaining.cancel()
                raise
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest test/parallel/test_parallel_determinism.py -v`
Expected: PASS (5 tests)

- [ ] **Step 6: Write the changelog fragment**

Create `changelog.d/parallel-exception-cancels-futures.fix.md`:

```
A worker exception during parallel evaluation now cancels the remaining
pending chunks instead of leaving them running — orphaned chunks previously
kept burning CPU in the shared process pool (delaying the next parallel
computation) and forced the thread backend to block until every orphaned
task had finished.
```

- [ ] **Step 7: Commit**

```bash
git add pyphi/parallel/backends/local_process.py pyphi/parallel/backends/local_thread.py test/parallel/test_parallel_determinism.py changelog.d/parallel-exception-cancels-futures.fix.md
git commit -m "Cancel pending parallel chunks when a worker raises

Both local backends propagated worker exceptions without cancelling the
outstanding futures: the loky pool kept executing orphaned chunks (burning
CPU and delaying the next map-reduce on the shared executor), and the
thread pool's shutdown blocked until every orphaned future completed.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UB4jbV4bQ8Y7Eywt1TNgJe"
git log --oneline -1
```

---

### Task 4: `find_mice` restores canonical purview order

**Files:**
- Modify: `pyphi/formalism/queries.py:314-326` (in `find_mice`, between the `map_reduce` call and tie resolution)
- Test: `test/parallel/test_parallel_determinism.py`

**Interfaces:**
- Consumes: `queries.find_mice(cs, direction, mechanism, purviews=None, **kwargs)` (existing); `MICE.purview` and `MICE.purview_ties` (existing).
- Produces: `find_mice` whose winner and tie order are independent of the order `map_reduce` returns results in.

- [ ] **Step 1: Write the failing test**

In `test/parallel/test_parallel_determinism.py`, add to the `pyphi` import
group:

```python
from pyphi import Direction
from pyphi import examples
from pyphi.conf import config
from pyphi.conf import presets
```

Then append:

```python
def test_find_mice_tied_purview_winner_independent_of_result_order(monkeypatch):
    from pyphi.formalism import queries

    system = examples.iit4_2023_fig6a_system()
    with config.override(**presets.iit4_2026):
        baseline = queries.find_mice(system, Direction.CAUSE, (0,), parallel=False)

        real_map_reduce = queries.map_reduce

        def reversed_map_reduce(fn, items, **kwargs):
            results = real_map_reduce(
                fn, items, **{**kwargs, "parallel": False, "progress": False}
            )
            return list(reversed(list(results)))

        monkeypatch.setattr(queries, "map_reduce", reversed_map_reduce)
        adversarial = queries.find_mice(system, Direction.CAUSE, (0,))

    assert len(baseline.purview_ties) >= 2, "case must be a genuine phi tie"
    assert adversarial.purview == baseline.purview
    assert {m.purview for m in adversarial.purview_ties} == {
        m.purview for m in baseline.purview_ties
    }
```

(Mechanism `(0,)` on the IIT 4.0 (2023) Fig. 6a system has four cause
purviews tied at maximal φ under the pinned formalism, so `ties[0]` genuinely
depends on candidate order. The patched `map_reduce` simulates the worst
completion-order permutation by reversing the results; the returned MICE must
nonetheless match sequential evaluation. Reversal also affects the inner
partition sweep, which may select a different tied MIP partition — the
assertions therefore compare only purview-level results, which the fix
guarantees.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/parallel/test_parallel_determinism.py::test_find_mice_tied_purview_winner_independent_of_result_order -v`
Expected: FAIL — `adversarial.purview != baseline.purview` (a different member of the tie wins).

- [ ] **Step 3: Implement the re-sort**

In `pyphi/formalism/queries.py`, `find_mice`, directly after the
`mip_results = map_reduce(...)` call and before
`all_mice = [mice_class(result) for result in mip_results]`, insert:

```python
    # Parallel evaluation returns results in completion / cost-bin order;
    # restore the canonical purview enumeration order so tie resolution
    # selects the same winner as sequential evaluation.
    order = {purview: index for index, purview in enumerate(purviews_list)}
    mip_results = sorted(mip_results, key=lambda ria: order[ria.purview])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/parallel/test_parallel_determinism.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add pyphi/formalism/queries.py test/parallel/test_parallel_determinism.py
git commit -m "Make tied-MICE selection independent of parallel result order

find_mice fed map_reduce results straight into tie resolution and took the
first survivor, so with purviews tied at maximal phi the reported MICE
depended on worker completion order (and on the cost-balanced chunk
permutation requested by size_func). Restore the canonical purview
enumeration order before resolving ties.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UB4jbV4bQ8Y7Eywt1TNgJe"
git log --oneline -1
```

---

### Task 5: State-MIP site collects in input order

**Files:**
- Modify: `pyphi/formalism/iit4/formalism.py:248-262` (the `map_reduce` call in `_find_mip_iit4`)
- Create: `changelog.d/parallel-tie-selection-determinism.fix.md`
- Test: `test/parallel/test_parallel_determinism.py`

**Interfaces:**
- Consumes: `queries.find_mip(cs, direction, mechanism, purview)` (existing), which dispatches to `_find_mip_iit4` under an IIT 4.0 formalism; the module-level `map_reduce` name in `pyphi.formalism.iit4.formalism` (its only call site).
- Produces: the specified-state MIP sweep collects results in input order, so `resolve_ties.states(...)` and its `ties[0]` selection match sequential evaluation.

- [ ] **Step 1: Write the failing test**

Append to `test/parallel/test_parallel_determinism.py`:

```python
def test_state_mip_map_reduce_collects_in_input_order(monkeypatch):
    import pyphi.formalism.iit4.formalism as iit4_formalism
    from pyphi import System
    from pyphi.formalism import queries

    captured = []
    real_map_reduce = iit4_formalism.map_reduce

    def capturing_map_reduce(fn, items, *more_items, **kwargs):
        captured.append(kwargs)
        return real_map_reduce(fn, items, *more_items, **kwargs)

    monkeypatch.setattr(iit4_formalism, "map_reduce", capturing_map_reduce)
    with config.override(**presets.iit4_2026):
        system = System(examples.basic_substrate(), (1, 0, 0))
        queries.find_mip(system, Direction.CAUSE, (0,), (0, 1))

    assert captured, "the state-MIP path should invoke map_reduce"
    assert all(kwargs.get("ordered") is True for kwargs in captured)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/parallel/test_parallel_determinism.py::test_state_mip_map_reduce_collects_in_input_order -v`
Expected: FAIL — `kwargs.get("ordered")` is `None` (never passed).

- [ ] **Step 3: Implement the ordered collection**

In `pyphi/formalism/iit4/formalism.py`, `_find_mip_iit4`, change the
`map_reduce` call's final kwargs line from:

```python
        desc="Finding MIP for maximum intrinsic information states",
        **parallel_kwargs,
    )
```

to:

```python
        desc="Finding MIP for maximum intrinsic information states",
        # Specified-state ties are resolved positionally downstream, so
        # results must arrive in input order, not completion order.
        **{**parallel_kwargs, "ordered": True},
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/parallel/test_parallel_determinism.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Write the changelog fragment**

Create `changelog.d/parallel-tie-selection-determinism.fix.md`:

```
Tied-MICE purview selection and the IIT 4.0 specified-state MIP search no
longer depend on worker completion order: parallel purview evaluation is
restored to the canonical enumeration order before tie resolution, and the
specified-state sweep collects results in input order.
```

- [ ] **Step 6: Commit**

```bash
git add pyphi/formalism/iit4/formalism.py test/parallel/test_parallel_determinism.py changelog.d/parallel-tie-selection-determinism.fix.md
git commit -m "Collect specified-state MIPs in input order

The specified-state sweep fed completion-ordered results into state tie
resolution, whose first-survivor selection then depended on worker
scheduling.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UB4jbV4bQ8Y7Eywt1TNgJe"
git log --oneline -1
```

---

### Task 6: Full verification

**Files:** none (verification only).

- [ ] **Step 1: Fast lane**

Run: `uv run pytest test/parallel/ test/formalism/test_iit4_sia_components.py test/test_golden_regression.py -q > /tmp/wave4-fastlane.log 2>&1; tail -3 /tmp/wave4-fastlane.log`
Expected: summary line reports all passed (no failures, no errors).

- [ ] **Step 2: Full pathless suite (worktree)**

Run in background, then read the log's summary line — never judge by exit code:

```bash
uv run pytest -q > /tmp/wave4-full.log 2>&1
tail -3 /tmp/wave4-full.log
```

Expected: summary comparable to main's baseline (3758 passed, 284 skipped —
plus the 7 new tests), zero failures/errors.

- [ ] **Step 3: Confirm every commit landed**

Run: `git log --oneline main..HEAD`
Expected: 7 commits (spec + plan + 5 task commits).
