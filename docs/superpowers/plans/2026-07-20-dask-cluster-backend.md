# Dask Cluster Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fill the `DaskScheduler` stub so every parallel operation in PyPhi can distribute across a `dask.distributed` cluster, and ship a CHTC deployment guide.

**Architecture:** `DaskScheduler.map_reduce` mirrors the process backend, reusing its building blocks (`_make_worker_fn` snapshot propagation, `_process_chunk`, `compute_chunksize`, and a newly factored `iter_chunks`). The client is user-created and resolved with `distributed.get_client()`. Nested dispatch from inside a dask worker task runs sequentially in-task (submit-and-block from a task can deadlock a fully occupied pool). Chunks are submitted with `pure=False`; collection semantics (ordered / short-circuit prefix determinism / cancel-on-error) are identical to the process backend.

**Tech Stack:** `dask[distributed]>=2024.1.0` (verified against 2026.7.1), `dask-jobqueue>=0.8.0` (verified 0.9.0), existing `pyphi.parallel` machinery.

**Spec:** `docs/superpowers/specs/2026-07-20-dask-cluster-backend-design.md`

## Global Constraints

- Commit messages end with the two standing trailers (`Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and the `Claude-Session` URL). Never `--no-verify`. After EVERY commit run `git log --oneline -1` — the ruff-format hook aborts commits silently; on abort re-add and re-commit.
- No roadmap/planning references (P-numbers, "spec", "plan", phase names) in code, docstrings, comments, or the changelog fragment.
- Docstrings: NumPy style, final-state voice (no stub/deferral narrative, no migration history).
- Test verification: redirect pytest output to a log file and read the summary line; never pipe through `tail`/`grep` directly, never end a verification command with `; echo`.
- Formalism pinning in tests: complete presets only (`config.override(**presets.iit4_2023, ...)`).
- Dependency lines exactly: `cluster = ["dask[distributed]>=2024.1.0", "dask-jobqueue>=0.8.0"]` (optional extra) and `"dask[distributed]>=2024.1.0",` (dev group).
- Worktree venv setup (once, before Task 1): `uv venv`, then
  `WT_PY="$(uv run python -c 'import sys; print(sys.executable)')"; env -u VIRTUAL_ENV uv pip install --python "$WT_PY" -e ".[visualize,caching,emd,xarray,mcp]" 'pot==0.9.6.post1' 'dask[distributed]>=2024.1.0' 'dask-jobqueue>=0.8.0'`.
- Scripts that create a `LocalCluster` need a `if __name__ == "__main__":` guard (spawned workers re-import the main module). Test modules under pytest are safe.

---

## Verified reference facts (measured 2026-07-20, dask 2026.7.1)

- `LocalCluster(n_workers=2, threads_per_worker=1, processes=True, dashboard_address=None)` starts in ~0.4 s.
- `distributed.get_client()` with no client raises `ValueError("No global client found and no address provided")`.
- `distributed.get_worker()` raises `ValueError("No worker found")` outside a task and returns the `Worker` inside one.
- `_make_worker_fn`-wrapped functions submitted via `client.submit(_process_chunk, chunk, wrapped, {}, predicate)` propagate the config snapshot: workers read `precision == 11` under `config.override(precision=11)`.
- `future.cancel()` marks pending futures `'cancelled'`; `distributed.as_completed(futures)` collects completed futures.
- `len(client.scheduler_info()["workers"]) == 2` on the 2-worker cluster; `client.submit(..., pure=False)` works.
- A basic-substrate SIA computed inside a worker returns in ~0.1 s (workers import pyphi from the venv normally).

---

### Task 1: Factor chunk construction into `iter_chunks`; lock snapshot installation

**Files:**
- Modify: `pyphi/parallel/chunking.py`
- Modify: `pyphi/parallel/backends/local_process.py` (`_get_chunks`, `_apply_snapshot_if_changed`)
- Test: `test/parallel/test_chunking.py`

**Interfaces:**
- Produces: `pyphi.parallel.chunking.iter_chunks(materialized, chunksize, num_workers, size_func=None) -> Iterator[tuple]` — yields chunk tuples (one index-aligned list per input iterable). Task 2's backend consumes it.
- Produces: thread-safe `_apply_snapshot_if_changed` (unchanged signature).

- [ ] **Step 1: Write the failing tests**

Append to `test/parallel/test_chunking.py`:

```python
class TestIterChunks:
    def test_even_chunks_cover_all_items_once(self):
        from pyphi.parallel.chunking import iter_chunks

        chunks = list(iter_chunks([[10, 20, 30, 40, 50]], chunksize=2, num_workers=2))
        flat = [x for (chunk,) in chunks for x in chunk]
        assert flat == [10, 20, 30, 40, 50]
        assert all(len(chunk) >= 1 for (chunk,) in chunks)

    def test_multi_iterable_chunks_stay_aligned(self):
        from pyphi.parallel.chunking import iter_chunks

        chunks = list(
            iter_chunks([[1, 2, 3, 4], ["a", "b", "c", "d"]], chunksize=2, num_workers=1)
        )
        for xs, ys in chunks:
            for x, y in zip(xs, ys, strict=True):
                assert "abcd"[x - 1] == y

    def test_cost_balanced_chunks_cover_all_items(self):
        from pyphi.parallel.chunking import iter_chunks

        chunks = list(
            iter_chunks(
                [[3, 1, 2, 5, 4]],
                chunksize=2,
                num_workers=2,
                size_func=lambda x: float(x),
            )
        )
        flat = sorted(x for (chunk,) in chunks for x in chunk)
        assert flat == [1, 2, 3, 4, 5]

    def test_empty_input_yields_nothing(self):
        from pyphi.parallel.chunking import iter_chunks

        assert list(iter_chunks([[]], chunksize=2, num_workers=2)) == []
        assert list(iter_chunks([], chunksize=2, num_workers=2)) == []

    def test_worker_floor_spreads_chunks(self):
        from pyphi.parallel.chunking import iter_chunks

        # 4 items fit in one chunk of 100, but 4 workers force 4 chunks.
        chunks = list(iter_chunks([[1, 2, 3, 4]], chunksize=100, num_workers=4))
        assert len(chunks) == 4
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest test/parallel/test_chunking.py -k IterChunks -q > /tmp/t1a.log 2>&1; tail -3 /tmp/t1a.log`
Expected: 5 failed — `ImportError: cannot import name 'iter_chunks'`.

- [ ] **Step 3: Implement `iter_chunks` in `pyphi/parallel/chunking.py`**

Add to the imports:

```python
from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Sequence
from typing import Any
```

Append the function:

```python
def iter_chunks(
    materialized: Sequence[Sequence[Any]],
    chunksize: int,
    num_workers: int,
    size_func: Callable[[Any], float] | None = None,
) -> Iterator[tuple]:
    """Yield chunk tuples for parallel dispatch.

    Each yielded tuple holds one list per input sequence, index-aligned
    across sequences. Indices are grouped into ``max(ceil(n / chunksize),
    num_workers)`` bins — evenly, or cost-balanced when ``size_func``
    estimates per-item cost from the first sequence's items. ``n`` is the
    length of the shortest sequence, so ragged inputs truncate as
    ``zip(strict=False)`` would.

    Parameters
    ----------
    materialized : sequence of sequences
        The item sequences to chunk; the first is the primary axis.
    chunksize : int
        Target number of items per chunk.
    num_workers : int
        Lower bound on the number of chunks, so a small workload still
        spreads across available workers.
    size_func : callable, optional
        Estimated cost of one primary-axis item. If None, chunks are
        count-balanced.
    """
    if not materialized or not materialized[0]:
        return
    n = min(len(it) for it in materialized)
    k = max(math.ceil(n / chunksize), num_workers)
    if size_func is not None:
        weights = [size_func(materialized[0][i]) for i in range(n)]
        index_bins = cost_balanced_partition(weights, k)
    else:
        index_bins = even_partition(n, k)
    for indices in index_bins:
        if not indices:
            continue
        yield tuple([it[i] for i in indices] for it in materialized)
```

- [ ] **Step 4: Rewire `LocalMapReduce._get_chunks` onto the helper**

In `pyphi/parallel/backends/local_process.py`, replace the body of `_get_chunks` (keep the method and its docstring line):

```python
    def _get_chunks(self) -> Iterator[tuple]:
        """Chunk iterables for parallel processing."""
        # Materialize iterables if needed for chunking
        materialized = []
        for iterable in self.iterables:
            if hasattr(iterable, "__len__"):
                materialized.append(iterable)
            else:
                materialized.append(list(iterable))

        from pyphi.parallel.chunking import iter_chunks

        yield from iter_chunks(
            materialized,
            chunksize=self.chunksize,
            num_workers=get_num_processes(),
            size_func=self.size_func,
        )
```

(The `cost_balanced_partition` / `even_partition` imports inside the old body disappear with it; `math` stays used by `_should_run_parallel`.)

- [ ] **Step 5: Lock snapshot installation**

In `pyphi/parallel/backends/local_process.py`: add `import threading` to the top-level imports, add a module-level lock next to the existing globals, and make the check-and-install atomic. Dask workers may run more than one thread; loky workers are single-threaded, so this changes nothing for the process backend.

```python
_LAST_APPLIED_SNAPSHOT_HASH: int | None = None
_PARENT_PID: int | None = None
_SNAPSHOT_LOCK = threading.Lock()
```

```python
def _apply_snapshot_if_changed(snapshot: Any, snap_hash: int) -> None:
    """Apply ``snapshot`` to the worker's global config; idempotent.

    ``snap_hash`` identifies the snapshot; it is computed once on the
    parent side (hashing the snapshot repr is ~1 ms, far too slow to pay
    per item) and compared against the last-applied hash here. The
    check-and-install is atomic so that multithreaded workers cannot
    interleave installations.

    Skips application when running in the parent process (set by the thread
    scheduler before dispatch) — threads share the parent's globals and the
    parent's config is already authoritative.
    """
    global _LAST_APPLIED_SNAPSHOT_HASH  # noqa: PLW0603

    import os

    if _PARENT_PID is not None and os.getpid() == _PARENT_PID:
        return

    with _SNAPSHOT_LOCK:
        if snap_hash == _LAST_APPLIED_SNAPSHOT_HASH:
            return

        config.install_snapshot(snapshot)
        _LAST_APPLIED_SNAPSHOT_HASH = snap_hash
```

- [ ] **Step 6: Run the new tests and the parallel suite**

Run: `uv run pytest test/parallel/ -q > /tmp/t1b.log 2>&1; tail -3 /tmp/t1b.log`
Expected: all pass (103 = 98 existing + 5 new), 5 skipped.

- [ ] **Step 7: Commit**

```bash
git add pyphi/parallel/chunking.py pyphi/parallel/backends/local_process.py test/parallel/test_chunking.py
git commit -m "Factor chunk construction into iter_chunks; lock snapshot install"
git log --oneline -1
```

---

### Task 2: Implement `DaskScheduler`; activate the `cluster` extra

**Files:**
- Rewrite: `pyphi/parallel/backends/dask.py`
- Modify: `pyproject.toml` (optional-dependencies + dev group)
- Modify: `test/parallel/test_scheduler.py` (replace the not-implemented test; retitle the lazy-import test)
- Create: `test/parallel/conftest.py` (shared `dask_client` fixture)
- Test: `test/parallel/test_dask_backend.py`

**Interfaces:**
- Consumes: `iter_chunks` (Task 1), `_make_worker_fn` / `_process_chunk` (existing), `compute_chunksize` (existing).
- Produces: working `DaskScheduler.map_reduce` with the full Protocol surface; `dask_client` module-scoped fixture in `test/parallel/conftest.py` (Task 3 reuses it).

- [ ] **Step 1: Activate the packaging**

In `pyproject.toml`, replace:

```toml
[project.optional-dependencies]
# Note: Single-machine parallelization uses stdlib ProcessPoolExecutor - no extra deps needed.
# The 'cluster' extra will be available in v2.0 for Dask-based cluster support.
# cluster = ["dask[distributed]>=2024.1.0", "dask-jobqueue>=0.8.0"]
```

with:

```toml
[project.optional-dependencies]
# Note: Single-machine parallelization needs no extra dependencies.
# Distributed execution on a Dask cluster (dask-jobqueue covers HTCondor/Slurm
# deployments):
cluster = ["dask[distributed]>=2024.1.0", "dask-jobqueue>=0.8.0"]
```

In `[dependency-groups] dev`, insert `"dask[distributed]>=2024.1.0",` after `"coverage",` (alphabetical order).

Run: `uv sync --all-extras 2>&1 | tail -2` (or in the worktree, the `env -u VIRTUAL_ENV uv pip install` line from Global Constraints, which already includes dask) and verify: `uv run python -c "import distributed; print(distributed.__version__)"`.

- [ ] **Step 2: Create the shared client fixture**

Create `test/parallel/conftest.py`:

```python
"""Fixtures shared by the parallel-backend tests."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def dask_client():
    """A local two-worker Dask cluster, registered as the current client.

    Workers are single-threaded separate processes, matching the
    deployment recommendation for CPU-bound work.
    """
    distributed = pytest.importorskip("distributed")

    with (
        distributed.LocalCluster(
            n_workers=2,
            threads_per_worker=1,
            processes=True,
            dashboard_address=None,
        ) as cluster,
        distributed.Client(cluster) as client,
    ):
        yield client
```

- [ ] **Step 3: Write the failing backend tests**

Create `test/parallel/test_dask_backend.py`:

```python
"""Tests for the Dask distributed-cluster scheduler."""

from __future__ import annotations

import pytest

from pyphi.conf import config
from pyphi.parallel.backends.dask import DaskScheduler
from pyphi.parallel.scheduler import ChunkingPolicy
from pyphi.parallel.scheduler import ProgressPolicy
from pyphi.parallel.scheduler import Scheduler
from pyphi.parallel.scheduler import ShortcircuitPolicy

distributed = pytest.importorskip("distributed")


def _plus_one(x):
    """Top-level function for serialization."""
    return x + 1


def _identity(x):
    """Top-level function for serialization."""
    return x


def _read_precision(_x):
    from pyphi.conf import config as worker_config

    return worker_config.numerics.precision


def test_implements_protocol():
    s = DaskScheduler()
    assert isinstance(s, Scheduler)
    assert s.supports_shared_state is False


def test_requires_active_client():
    # Must be defined before any test that uses the module-scoped
    # ``dask_client`` fixture: once that fixture instantiates, its client
    # stays current for the rest of the module.
    with pytest.raises(RuntimeError, match=r"distributed\.Client"):
        DaskScheduler().map_reduce(_plus_one, [1, 2, 3])


def test_basic_map_reduce(dask_client):
    result = DaskScheduler().map_reduce(
        _plus_one,
        [1, 2, 3, 4],
        reducer=sum,
        chunking=ChunkingPolicy(chunksize=1, sequential_threshold=1),
    )
    assert result == 2 + 3 + 4 + 5


def test_ordered_returns_submission_order(dask_client):
    result = DaskScheduler().map_reduce(
        _identity,
        [3, 1, 2],
        reducer=list,
        chunking=ChunkingPolicy(chunksize=1, sequential_threshold=1),
        ordered=True,
    )
    assert result == [3, 1, 2]


def test_snapshot_propagation(dask_client):
    with config.override(precision=11):
        result = DaskScheduler().map_reduce(
            _read_precision,
            [1, 2, 3],
            reducer=list,
            chunking=ChunkingPolicy(chunksize=1, sequential_threshold=1),
        )
    assert result == [11, 11, 11]


def test_shortcircuit_collects_deterministic_prefix(dask_client):
    fired = []
    result = DaskScheduler().map_reduce(
        _identity,
        [1, 2, 3, 4, 5],
        reducer=list,
        chunking=ChunkingPolicy(chunksize=1, sequential_threshold=1),
        shortcircuit=ShortcircuitPolicy(
            func=lambda r: r == 2, callback=lambda *_: fired.append(True)
        ),
    )
    assert result == [1, 2]
    assert fired == [True]


def test_empty_items(dask_client):
    assert DaskScheduler().map_reduce(_plus_one, [], reducer=list) == []


def test_progress(dask_client, monkeypatch):
    bars = []

    class RecordingBar:
        def __init__(self, total=None, desc=""):
            self.total = total
            self.desc = desc
            self.updates = 0
            self.closed = False
            bars.append(self)

        def update(self, n=1):
            self.updates += n

        def close(self):
            self.closed = True

    monkeypatch.setattr(
        "pyphi.parallel.backends.dask.LocalProgressBar", RecordingBar
    )
    DaskScheduler().map_reduce(
        _plus_one,
        [1, 2, 3, 4],
        reducer=list,
        chunking=ChunkingPolicy(chunksize=1, sequential_threshold=1),
        progress=ProgressPolicy(enabled=True, desc="cells"),
    )
    (bar,) = bars
    assert bar.total == 4
    assert bar.desc == "cells"
    assert bar.updates == 4
    assert bar.closed


def _nested_dispatch(_x):
    from pyphi.parallel.backends.dask import DaskScheduler

    def double(y):
        return y * 2

    return DaskScheduler().map_reduce(double, [1, 2, 3], reducer=list)


def test_nested_dispatch_runs_in_task(dask_client):
    """map_reduce reached from inside a worker task runs in-task, not by
    submitting back to the cluster (which can deadlock an occupied pool)."""
    fut = dask_client.submit(_nested_dispatch, 0, pure=False)
    assert fut.result() == [2, 4, 6]
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `uv run pytest test/parallel/test_dask_backend.py -q > /tmp/t2a.log 2>&1; tail -3 /tmp/t2a.log`
Expected: failures/errors — `NotImplementedError` from the stub (and an `AttributeError` from the monkeypatch target in the progress test, since the stub module does not import `LocalProgressBar` yet).

- [ ] **Step 5: Implement the backend**

Replace the entire contents of `pyphi/parallel/backends/dask.py`:

```python
# parallel/backends/dask.py
"""Distributed-cluster scheduler backed by ``dask.distributed``.

Distributes map-reduce workloads across a Dask cluster. The cluster is
user-provided: create and connect a :class:`distributed.Client` — to a
``distributed.LocalCluster``, a ``dask_jobqueue`` cluster (HTCondor, Slurm,
PBS, ...), or a scheduler address — before computing, and select the backend
with ``config.parallel_backend = "dask"``. The backend resolves the active
client with :func:`distributed.get_client`.

Workers receive the caller's configuration snapshot with each chunk and
install it idempotently, so distributed results reflect the caller's
configuration regardless of worker process state. Items are grouped into
cost-balanced chunks (one future per chunk) to amortize network overhead;
collection preserves the same determinism guarantees as the local process
backend (submission-order prefixes under ``ordered`` or an active
short-circuit predicate).

A ``map_reduce`` call that executes *inside* a Dask worker task runs its
items sequentially in that task rather than submitting to the cluster:
blocking on subtasks from within a task can deadlock a fully occupied
worker pool, and worker slots are typically single-core. Consequently only
the outermost parallel level of a nested computation is distributed.

``dask.distributed`` is imported inside :meth:`DaskScheduler.map_reduce`;
importing this module is free and requires no optional dependencies.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from collections.abc import Iterable
from typing import Any

from pyphi.conf import config
from pyphi.parallel.scheduler import ChunkingPolicy
from pyphi.parallel.scheduler import ProgressPolicy
from pyphi.parallel.scheduler import ShortcircuitPolicy

from .progress import LocalProgressBar


def _run_in_place(
    fn: Callable[..., Any],
    materialized: list[list[Any]],
    reducer: Callable[[Iterable[Any]], Any],
    shortcircuit: ShortcircuitPolicy,
    map_kwargs: dict[str, Any],
    progress_bar: LocalProgressBar | None,
) -> Any:
    """Map ``fn`` over the items sequentially in the current process."""
    results: list[Any] = []
    for args in zip(*materialized, strict=False):
        value = fn(*args, **map_kwargs)
        results.append(value)
        if progress_bar is not None:
            progress_bar.update(1)
        if shortcircuit.func(value):
            shortcircuit.fire(results)
            break
    return reducer(results)


class DaskScheduler:
    """Scheduler backed by a user-provided ``dask.distributed`` cluster.

    Requires an active :class:`distributed.Client`; raises
    :exc:`RuntimeError` when none is connected and :exc:`ImportError` when
    ``distributed`` is not installed (install the ``cluster`` extra).
    """

    @property
    def supports_shared_state(self) -> bool:
        return False

    def map_reduce(
        self,
        fn: Callable[..., Any],
        items: Iterable[Any],
        *more_items: Iterable[Any],
        reducer: Callable[[Iterable[Any]], Any] = list,
        config_snapshot: Any | None = None,
        chunking: Any = None,
        progress: Any = None,
        shortcircuit: Any = None,
        ordered: bool = False,
        map_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        try:
            import distributed
        except ImportError as exc:
            raise ImportError(
                'The "dask" parallel backend requires the distributed '
                "package; install it with `pip install \"pyphi[cluster]\"`."
            ) from exc

        chunking = chunking or ChunkingPolicy()
        progress = progress or ProgressPolicy()
        shortcircuit = shortcircuit or ShortcircuitPolicy()
        map_kwargs = map_kwargs or {}

        # Inside a worker task, run in-task instead of submitting to the
        # cluster: blocking on subtasks can deadlock an occupied pool.
        try:
            distributed.get_worker()
        except ValueError:
            in_worker = False
        else:
            in_worker = True
        if in_worker:
            materialized = [list(it) for it in (items, *more_items)]
            if not materialized[0]:
                return reducer([])
            return _run_in_place(
                fn, materialized, reducer, shortcircuit, map_kwargs, None
            )

        try:
            client = distributed.get_client()
        except ValueError as exc:
            raise RuntimeError(
                'The "dask" parallel backend needs an active '
                "distributed.Client; create one before computing — e.g. "
                "`from distributed import Client; client = Client()` for a "
                "local cluster, or `Client(address)` for a running "
                "scheduler."
            ) from exc

        snapshot = (
            config_snapshot if config_snapshot is not None else config.snapshot()
        )

        from pyphi.parallel.backends.local_process import _make_worker_fn
        from pyphi.parallel.backends.local_process import _process_chunk
        from pyphi.parallel.chunking import iter_chunks
        from pyphi.parallel.sampling import compute_chunksize

        items_list = list(items)
        if not items_list:
            return reducer([])

        # The sampler times fn on bare items, so it must see the same call
        # shape as the real map: bind map_kwargs, and skip sampling entirely
        # for multi-iterable maps (a single item is not a valid call).
        if more_items:
            sampling_fn = None
        elif map_kwargs:
            sampling_fn = functools.partial(fn, **map_kwargs)
        else:
            sampling_fn = fn

        chunksize, sampled_iter = compute_chunksize(
            items_list,
            target_seconds=chunking.target_seconds,
            fn=sampling_fn,
            sequential_threshold=chunking.sequential_threshold,
            explicit_chunksize=chunking.chunksize,
        )
        items_list = list(sampled_iter)
        materialized = [items_list, *[list(it) for it in more_items]]
        total = len(items_list)

        # A sampled chunksize estimates the number of items per
        # ``target_seconds`` of work, so a workload that fits within one
        # such chunk is not worth dispatching; fold it into the threshold.
        # An explicitly configured chunksize governs granularity only.
        sequential_threshold = chunking.sequential_threshold
        if chunking.chunksize is None:
            sequential_threshold = max(sequential_threshold, chunksize + 1)

        num_workers = max(len(client.scheduler_info()["workers"]), 1)

        progress_bar = (
            LocalProgressBar(
                total=progress.total if progress.total is not None else total,
                desc=progress.desc,
            )
            if progress.enabled
            else None
        )
        try:
            if total < sequential_threshold or (
                min(max(math.ceil(total / chunksize), num_workers), total) <= 1
            ):
                return _run_in_place(
                    fn, materialized, reducer, shortcircuit, map_kwargs, progress_bar
                )

            wrapped_fn = _make_worker_fn(fn, snapshot)
            chunks = list(
                iter_chunks(
                    materialized,
                    chunksize=chunksize,
                    num_workers=num_workers,
                    size_func=chunking.size_func,
                )
            )

            # pure=False: every submission is a distinct task, so repeated
            # computations are re-executed rather than deduplicated by
            # dask's content-addressed task keys.
            futures = [
                client.submit(
                    _process_chunk,
                    chunk,
                    wrapped_fn,
                    map_kwargs,
                    shortcircuit.func,
                    pure=False,
                )
                for chunk in chunks
            ]

            results: list[Any] = []
            short_circuited = False
            # Collect in submission order when the caller asked for original
            # order or a short-circuit predicate is active: the collected
            # prefix then matches sequential evaluation regardless of
            # worker completion order.
            iterator: Iterable[Any] = (
                futures
                if ordered or shortcircuit.active
                else distributed.as_completed(futures)
            )
            try:
                for fut in iterator:
                    chunk_results = fut.result()
                    results.extend(chunk_results)
                    if progress_bar is not None:
                        progress_bar.update(len(chunk_results))
                    for value in chunk_results:
                        if shortcircuit.func(value):
                            short_circuited = True
                            for remaining in futures:
                                remaining.cancel()
                            shortcircuit.fire(futures)
                            break
                    if short_circuited:
                        break
            except BaseException:
                for remaining in futures:
                    remaining.cancel()
                raise

            return reducer(results)
        finally:
            if progress_bar is not None:
                progress_bar.close()
```

- [ ] **Step 6: Run the backend tests**

Run: `uv run pytest test/parallel/test_dask_backend.py -q > /tmp/t2b.log 2>&1; tail -3 /tmp/t2b.log`
Expected: 9 passed.

- [ ] **Step 7: Update the two stale scheduler tests**

In `test/parallel/test_scheduler.py`:

(a) The lazy-import test (~line 233): rename `test_dask_scheduler_skeleton_lazy_import` → `test_dask_backend_lazy_import` and keep its body — the invariant (importing the module must not load `dask.distributed`) still holds because `distributed` is imported inside `map_reduce`.

(b) Replace `test_map_reduce_dask_backend_is_not_implemented` (and update the section comment above `_double` that says the stub "raises ``NotImplementedError``" — reword to "the Dask backend (which requires an active client)"):

```python
def test_map_reduce_dask_backend_requires_client():
    """backend='dask' routes to the DaskScheduler, which needs a client."""
    pytest.importorskip("distributed")
    from pyphi.parallel import map_reduce

    with pytest.raises(RuntimeError, match=r"distributed\.Client"):
        map_reduce(
            _double, [1, 2, 3, 4, 5], backend="dask", sequential_threshold=1, chunksize=2
        )
```

- [ ] **Step 8: Run the parallel suite**

Run: `uv run pytest test/parallel/ -q > /tmp/t2c.log 2>&1; tail -3 /tmp/t2c.log`
Expected: all pass (112 = 103 + 9 new), 5 skipped.

- [ ] **Step 9: Commit**

```bash
git add pyphi/parallel/backends/dask.py pyproject.toml uv.lock test/parallel/conftest.py test/parallel/test_dask_backend.py test/parallel/test_scheduler.py
git commit -m "Implement the Dask distributed-cluster scheduler"
git log --oneline -1
```

(If `uv.lock` did not change — extras are not locked into the default group — drop it from the add.)

---

### Task 3: Dask-backend parallel≡sequential invariant

**Files:**
- Create: `test/parallel/test_dask_backend_equals_sequential.py`

**Interfaces:**
- Consumes: `dask_client` fixture from `test/parallel/conftest.py` (Task 2); `DaskScheduler` (Task 2).

- [ ] **Step 1: Write the test**

Create `test/parallel/test_dask_backend_equals_sequential.py` (mirrors the thread-backend companion file):

```python
"""The dask-backed parallel SIA must equal the sequential SIA.

Companion to test_parallel_equals_sequential.py (loky process scheduler) and
test_thread_backend_equals_sequential.py. The dask scheduler ships each chunk
to distributed worker processes with the caller's configuration snapshot;
this guards that distribution preserves results exactly. Runs against a
local two-worker cluster; nested parallel levels run in-task by design, so
the outermost level exercises distribution.
"""

from __future__ import annotations

import pytest

from pyphi import System
from pyphi import examples
from pyphi.conf import config
from pyphi.conf import presets

distributed = pytest.importorskip("distributed")

_SUBSTRATES = {
    "basic": (examples.basic_substrate, (1, 0, 0)),
    "xor": (examples.xor_substrate, (0, 0, 0)),
}


def _dask_override(threshold: int = 2) -> dict:
    """Force the dask scheduler on the outer SIA/CES evaluation levels at a
    low sequential threshold so dispatch actually parallelizes (map_reduce
    parallelizes only when a level produces more than one chunk)."""
    c = config.infrastructure
    forced = {"parallel": True, "sequential_threshold": threshold}
    keys = (
        "parallel_partition_evaluation",
        "parallel_distinction_evaluation",
        "parallel_purview_evaluation",
    )
    return {
        "parallel": True,
        "parallel_backend": "dask",
        **{k: {**getattr(c, k), **forced} for k in keys},
    }


@pytest.mark.parametrize("name", list(_SUBSTRATES))
def test_iit4_sia_dask_backend_equals_sequential(name: str, dask_client) -> None:
    """IIT 4.0 (2023, GID): the dask-backed SIA equals the sequential SIA."""
    factory, state = _SUBSTRATES[name]
    with config.override(**presets.iit4_2023, parallel=False):
        seq = System(factory(), state).sia()
    with config.override(**presets.iit4_2023, **_dask_override()):
        par = System(factory(), state).sia()

    assert seq == par, (
        f"{name}: IIT 4.0 SIA diverged under the dask backend — sequential "
        f"φ {seq.phi} vs dask φ {par.phi}"
    )
```

- [ ] **Step 2: Run it and check the timing**

Run: `uv run pytest test/parallel/test_dask_backend_equals_sequential.py -q --durations=5 > /tmp/t3a.log 2>&1; tail -8 /tmp/t3a.log`
Expected: 2 passed. Lane criterion: if either test exceeds ~15 s, add `marks=pytest.mark.slow` to that substrate via `pytest.param` (mirroring how `test_parallel_equals_sequential.py` marks `rule110`/`grid3`); below that, leave both in the fast lane.

- [ ] **Step 3: Commit**

```bash
git add test/parallel/test_dask_backend_equals_sequential.py
git commit -m "Guard dask-backend SIA against the sequential result"
git log --oneline -1
```

---

### Task 4: CHTC how-to, stale-reference fixes, changelog, ROADMAP

**Files:**
- Create: `docs/howto/chtc.md`
- Modify: `docs/howto/index.md` (toctree), `docs/howto/parallel.md` (cross-link)
- Modify: `pyphi/mcp/content/parallelization.md` (stub sentence)
- Create: `changelog.d/cluster-backend.feature.md`
- Modify: `ROADMAP.md` (dashboard row + Wave 4 item)

- [ ] **Step 1: Write the CHTC guide**

Create `docs/howto/chtc.md` with exactly this content:

````markdown
# Run PyPhi on a cluster (CHTC)

This guide covers running PyPhi on UW–Madison's Center for High-Throughput
Computing (CHTC), and applies with minor changes to any HTCondor pool. It
assumes a CHTC account and basic familiarity with submitting jobs; see the
[CHTC getting-started roadmap](https://chtc.cs.wisc.edu/uw-research-computing/htc-roadmap)
for that groundwork.

## Which CHTC system?

CHTC operates two systems. The **HTC system** (HTCondor) runs many
independent single-node jobs; the **HPC cluster** (Slurm) is for MPI-style
computations that internally span multiple nodes. PyPhi computations are
single-node by construction — parallelism spreads within one machine, or
across machines only via the Dask backend below — and CHTC directs
single-node work to the HTC system. Use HTC.

Three deployment patterns follow, ordered by how well CHTC supports them
today.

## Build the PyPhi container

All three patterns deliver PyPhi to execute nodes as an Apptainer
container. Build a wheel, then the image (on the access point or any Linux
machine with Apptainer):

```bash
# In your PyPhi checkout:
uv build                        # writes dist/pyphi-<version>-py3-none-any.whl
```

`pyphi.def`:

```
Bootstrap: docker
From: python:3.13-bookworm

%files
    dist/pyphi-*.whl /opt/

%post
    pip install --no-cache-dir /opt/pyphi-*.whl

%runscript
    exec python "$@"
```

```bash
apptainer build pyphi.sif pyphi.def
```

Once PyPhi 2.0 is on PyPI the `%files` section can be dropped in favor of
`pip install pyphi` in `%post`.

## Pattern A — many independent runs (fully supported)

The canonical HTC workload: each condor job runs one self-contained PyPhi
computation (one substrate/state/configuration cell), and you collect the
saved results afterwards. Write results with `pyphi.provenance.save_json`
(or `.save()` on result objects) so each output is self-describing.

`run_cell.py` — one cell per job, selected by the process number:

```python
import sys

import pyphi

cell = int(sys.argv[1])

# Define your substrates/states/configs however you like; index them by cell.
substrate = pyphi.examples.basic_substrate()
states = list(pyphi.utils.all_states(substrate.size))
state = states[cell % len(states)]

sia = pyphi.System(substrate, state).sia()
sia.save(f"sia_state{cell}.json.gz")
```

`sweep.sub`:

```
universe = container
container_image = pyphi.sif

executable = run_cell.py
arguments = $(Process)
transfer_executable = false

transfer_input_files = run_cell.py
should_transfer_files = YES
when_to_transfer_output = ON_EXIT

request_cpus = 1
request_memory = 4GB
request_disk = 4GB

log = sweep.log
error = sweep.$(Process).err
output = sweep.$(Process).out

queue 8
```

Submit with `condor_submit sweep.sub`. Jobs have a 72-hour default runtime
limit; keep per-job inputs/outputs under CHTC's file-transfer guidance
(~100 MB per file) or arrange staging with CHTC. For dependent stages
(compute → aggregate), see CHTC's DAGMan guides.

## Pattern B — one big analysis on a fat node (fully supported)

For a single analysis too large for a lab machine, request one many-core,
high-memory slot and let PyPhi's default process backend saturate it:

```
universe = container
container_image = pyphi.sif

executable = analyze.py
transfer_executable = false

transfer_input_files = analyze.py
should_transfer_files = YES
when_to_transfer_output = ON_EXIT

request_cpus = 32
request_memory = 200GB
request_disk = 20GB

log = analyze.log
error = analyze.err
output = analyze.out

queue
```

In `analyze.py`, enable parallelism (`pyphi.config.parallel = True`); the
process backend uses every requested core. See CHTC's high-memory-job
guide for current per-slot limits, and `pyphi.estimate_analysis` for
sizing the workload before submitting.

## Pattern C — distributing one analysis across machines (pilot)

The `dask` backend spreads a single computation's parallel levels
(distinctions, purviews, partitions) across a Dask worker pool. On an
HTCondor pool, [dask-jobqueue](https://jobqueue.dask.org) launches workers
as ordinary condor jobs that connect back to a scheduler in your session
on the access point:

```python
from dask_jobqueue import HTCondorCluster
from distributed import Client

cluster = HTCondorCluster(
    cores=1,
    processes=1,
    memory="4GB",
    disk="4GB",
    job_extra_directives={
        "universe": "container",
        "container_image": "pyphi.sif",
    },
)
cluster.scale(jobs=32)          # 32 single-core workers
client = Client(cluster)

import pyphi

pyphi.config.parallel = True
pyphi.config.parallel_backend = "dask"

substrate = pyphi.examples.basic_substrate()
sia = pyphi.System(substrate, (1, 0, 0)).sia()
```

Notes:

- **Single-threaded workers** (`cores=1, processes=1`): PyPhi's work is
  CPU-bound Python, so extra worker threads do not help.
- **Nesting**: only the outermost parallel level distributes; levels
  reached inside a worker task run within that task.
- **Preemption**: HTC slots can be preempted; Dask reschedules lost tasks
  automatically, at the cost of recomputing them.
- **Dashboard**: forward it over SSH
  (`ssh -L 8787:localhost:8787 <access point>`), then open
  `http://localhost:8787`.

**Support status — read before relying on this pattern.** CHTC does not
currently document or support Dask on the HTC system, and most ports on
CHTC submit and execute nodes are closed. A Dask cluster needs
bidirectional TCP between the scheduler (your access point session) and
the workers (execute nodes). Whether that traffic is permitted from your
access point is a site question. Before adopting this pattern, ask CHTC
facilitation (chtc@cs.wisc.edu):

1. Are inbound connections from execute nodes to a high port on my access
   point permitted (a `dask.distributed` scheduler listening in a user
   session)? If not, is there a designated machine where this is allowed?
2. Is there a policy on long-lived coordinator processes running on access
   points for the duration of a workload?
3. What wall-time and sizing guidance applies to held worker jobs (e.g.
   32 single-core workers held for a few hours)?

If the answers rule this pattern out, Patterns A and B cover sweeps and
single big analyses with fully supported mechanics.
````

- [ ] **Step 2: Wire the page into the docs**

In `docs/howto/index.md`, add `chtc` to the toctree after `parallel`:

```
configure
parallel
chtc
cache
```

In `docs/howto/parallel.md`, find the section discussing `parallel_backend`
(grep for `parallel_backend`; if no such section exists, add this at the end
of the page) and add:

```markdown
## Running on a cluster

The `dask` backend distributes the same parallel levels across a
`dask.distributed` cluster — a laptop `LocalCluster`, lab workstations, or
an HTCondor/Slurm pool via `dask-jobqueue`. Install the `cluster` extra
(`pip install "pyphi[cluster]"`), connect a `distributed.Client`, and set
`pyphi.config.parallel_backend = "dask"`. See {doc}`chtc` for cluster
deployment, including UW–Madison's CHTC.
```

- [ ] **Step 3: Fix the stale MCP content**

In `pyphi/mcp/content/parallelization.md` line 69, replace the sentence
fragment ``` `"dask"` is an unimplemented stub. ``` with:

```markdown
  `"dask"` distributes across a user-connected `dask.distributed` cluster
  (requires the `cluster` extra and an active `distributed.Client`).
```

- [ ] **Step 4: Changelog fragment**

```bash
cat > changelog.d/cluster-backend.feature.md <<'EOF'
Added the `dask` parallel backend: with the `cluster` extra installed
(`pip install "pyphi[cluster]"`) and a `distributed.Client` connected,
`config.parallel_backend = "dask"` distributes PyPhi's parallel levels
across the cluster. Added a how-to guide for running PyPhi on UW–Madison's
CHTC (independent condor jobs, fat-node jobs, and Dask worker pools via
dask-jobqueue).
EOF
```

- [ ] **Step 5: Update ROADMAP.md**

Dashboard row (`| P11 cluster backends | ⬜ open | 4 | ... |`) becomes:

```markdown
| P11 cluster backends | 🟡 partial | 4 | Dask backend landed 2026-07-20: `DaskScheduler` distributes map-reduce over a user-connected `distributed.Client` (snapshot propagation, cost-balanced chunks, deterministic short-circuit prefixes; nested dispatch runs in-task); `cluster` extra; `docs/howto/chtc.md` covers CHTC (plain condor sweeps + fat-node fully supported; Dask worker pools documented as a pilot pending CHTC port-access confirmation — most ports on CHTC nodes are closed, their first-party Dask/htmap glue is dead/archived, so the guide lists the facilitation questions). Remaining: HTCondor-native batch surface — materialize sweep cells (with a substrate axis) as independent condor jobs and collect into `SweepResult`; build when a pool-scale campaign (tens of thousands of independent runs) makes a held worker pool wasteful. |
```

Wave 4 item (`- **P11 — cluster backends.** Fill HTCondor / full Dask ...`) becomes:

```markdown
- **P11 — cluster backends — Dask half landed (2026-07-20).** `DaskScheduler` implements
  the full Scheduler Protocol against a user-connected `distributed.Client`
  (spec `docs/superpowers/specs/2026-07-20-dask-cluster-backend-design.md`).
  Remaining: the HTCondor-native batch-submission surface (sweep cells as
  independent condor jobs + `SweepResult` collection); sequence on demand.
```

- [ ] **Step 6: Verify the docs build**

Run: `rm -rf docs/reference/_autosummary && just docs > /tmp/t4a.log 2>&1; tail -5 /tmp/t4a.log`
Expected: build succeeded; then confirm `docs/_build/html/howto/chtc.html` exists.

- [ ] **Step 7: Commit**

```bash
git add docs/howto/chtc.md docs/howto/index.md docs/howto/parallel.md pyphi/mcp/content/parallelization.md changelog.d/cluster-backend.feature.md ROADMAP.md
git commit -m "Document cluster deployment on CHTC and the dask backend"
git log --oneline -1
```

---

## Final verification

- [ ] Full pathless suite: `uv run pytest -q > /tmp/final.log 2>&1; tail -3 /tmp/final.log` — expected: no failures/errors (baseline before this work: 3945 passed / 288 skipped).
- [ ] Draft the CHTC facilitation email (the three Pattern-C questions plus one line of context about PyPhi workloads) and deliver it in the final report to the user — not committed to the repo.
