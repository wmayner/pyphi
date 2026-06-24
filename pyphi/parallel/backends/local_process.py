# parallel/backends/local_process.py
"""Process-pool scheduler backed by loky.

Uses loky (via joblib) instead of ``ProcessPoolExecutor`` for cloudpickle
support, allowing functions defined in ``__main__`` (e.g., Jupyter notebooks)
to be serialized and sent to worker processes.

Also exports :class:`LocalProcessScheduler`, the Protocol-conforming
wrapper around :class:`LocalMapReduce` that delivers a ``ConfigSnapshot``
to workers via closure.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from concurrent.futures import as_completed
from typing import Any

from joblib.externals.loky import get_reusable_executor

from pyphi.conf import config
from pyphi.conf import fallback
from pyphi.parallel import _map_sequential
from pyphi.parallel import _reduce
from pyphi.parallel import false
from pyphi.parallel import get_num_processes

from .progress import LocalProgressBar

log = logging.getLogger(__name__)


def _process_chunk(
    chunk_iterables: tuple,
    map_func: Callable,
    map_kwargs: dict,
    shortcircuit_func: Callable,
) -> list:
    """Process a single chunk of work.

    This function runs in a worker process. It applies the map function
    to each element in the chunk and returns a list of results.
    Reduction is done at the end after all chunks are collected.
    """
    results = []
    for args in zip(*chunk_iterables, strict=False):
        result = map_func(*args, **map_kwargs)
        results.append(result)

        # Check for short-circuit condition
        if shortcircuit_func(result):
            break

    return results


class LocalMapReduce:
    """Single-machine parallelization using loky's reusable executor.

    Key features:
    - Low overhead (~1-5ms per task)
    - Cloudpickle support for functions defined in __main__ (Jupyter notebooks)
    - Tree-structured execution for hierarchical computations
    - Short-circuit support with future cancellation
    - Progress tracking compatible with Jupyter notebooks
    """

    def __init__(
        self,
        map_func: Callable,
        iterables: tuple[Iterable, ...],
        reduce_func: Callable,
        reduce_kwargs: dict,
        chunksize: int,
        sequential_threshold: int = 1,
        size_func: Callable[..., float] | None = None,
        shortcircuit_func: Callable = false,
        shortcircuit_callback: Callable | None = None,
        ordered: bool = False,
        map_kwargs: dict | None = None,
        progress: bool = True,
        desc: str = "",
        total: int | None = None,
    ):
        self.map_func = map_func
        self.iterables = iterables
        self.reduce_func = reduce_func
        self.reduce_kwargs = reduce_kwargs
        self.chunksize = chunksize
        self.sequential_threshold = sequential_threshold
        self.size_func = size_func
        self.shortcircuit_func = shortcircuit_func
        self.shortcircuit_callback = shortcircuit_callback
        self.ordered = ordered
        self.map_kwargs = fallback(map_kwargs, {})
        self.progress = progress
        self.desc = desc
        self.total = total

        # State
        self.progress_bar: LocalProgressBar | None = None
        self.result = None
        self.done = False
        self.error = None
        self._futures: list[Any] = []

    def _cancel_remaining(self, futures: list[Any]) -> None:
        """Cancel all remaining futures."""
        for future in futures:
            if not future.done():
                future.cancel()

    def _get_chunks(self) -> Iterator[tuple]:
        """Chunk iterables for parallel processing."""
        # Materialize iterables if needed for chunking
        materialized = []
        for iterable in self.iterables:
            if hasattr(iterable, "__len__"):
                materialized.append(iterable)
            else:
                materialized.append(list(iterable))

        # Chunk each iterable and zip them together
        if not materialized or not materialized[0]:
            return

        from pyphi.parallel.chunking import cost_balanced_partition
        from pyphi.parallel.chunking import even_partition

        n = len(materialized[0])
        k = max(math.ceil(n / self.chunksize), get_num_processes())
        if self.size_func is not None:
            weights = [self.size_func(x) for x in materialized[0]]
            index_bins = cost_balanced_partition(weights, k)
        else:
            index_bins = even_partition(n, k)

        for indices in index_bins:
            if not indices:
                continue
            yield tuple([it[i] for i in indices] for it in materialized)

    def _should_run_parallel(self) -> bool:
        """Parallelize only when there is more than one chunk of work."""
        if self.total is None:
            return True  # unknown length; let the executor chunk and dispatch
        if self.total < self.sequential_threshold:
            return False
        # a single chunk → no parallel benefit
        return not (self.chunksize and self.total <= self.chunksize)

    def run(self) -> Any:
        """Execute the parallel computation."""
        if self.done:
            return self.result

        try:
            # Set up progress bar if enabled
            if self.progress:
                self.progress_bar = LocalProgressBar(
                    total=self.total,
                    desc=self.desc or "",
                )

            if not self._should_run_parallel():
                return self._run_sequential()

            return self._run_parallel()

        except Exception as e:
            self.error = e
            raise e
        finally:
            if self.progress_bar is not None:
                self.progress_bar.close()

    def _run_sequential(self) -> Any:
        """Run computation sequentially."""
        results = _map_sequential(self.map_func, *self.iterables, **self.map_kwargs)

        # Apply short-circuiting
        collected = []
        for result in results:
            collected.append(result)
            if self.progress_bar is not None:
                self.progress_bar.update(1)
            if self.shortcircuit_func(result):
                if self.shortcircuit_callback is not None:
                    self.shortcircuit_callback(collected)
                break

        self.result = _reduce(
            collected, self.reduce_func, self.reduce_kwargs, branch=False
        )
        self.done = True
        return self.result

    def _run_parallel(self) -> Any:
        """Run computation in parallel using loky reusable executor.

        Uses loky instead of ProcessPoolExecutor for cloudpickle support,
        allowing functions defined in __main__ (e.g., Jupyter notebooks) to
        be serialized and sent to worker processes.
        """
        num_workers = get_num_processes()

        # Collect all chunks
        chunks = list(self._get_chunks())

        if not chunks:
            self.result = _reduce([], self.reduce_func, self.reduce_kwargs, branch=False)
            self.done = True
            return self.result

        results = []
        short_circuited = False

        # Use loky's reusable executor for cloudpickle support
        executor = get_reusable_executor(max_workers=num_workers)

        # Submit all chunks as futures
        futures = [
            executor.submit(
                _process_chunk,
                chunk_tuple,
                self.map_func,
                self.map_kwargs,
                self.shortcircuit_func,
            )
            for chunk_tuple in chunks
        ]
        self._futures = futures

        # Collect results in order of completion (or original order if ordered=True)
        if self.ordered:
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

        # Final reduction - apply user's reduce function
        self.result = _reduce(
            results, self.reduce_func, self.reduce_kwargs, branch=False
        )
        self.done = True
        return self.result


_LAST_APPLIED_SNAPSHOT_HASH: int | None = None
_PARENT_PID: int | None = None


def _apply_snapshot_if_changed(snapshot: Any) -> None:
    """Apply ``snapshot`` to the worker's global config; idempotent.

    Skips application when running in the parent process (set by the thread
    scheduler before dispatch) — threads share the parent's globals and the
    parent's config is already authoritative.
    """
    global _LAST_APPLIED_SNAPSHOT_HASH  # noqa: PLW0603

    import os

    if _PARENT_PID is not None and os.getpid() == _PARENT_PID:
        return

    snap_hash = hash(repr(snapshot))
    if snap_hash == _LAST_APPLIED_SNAPSHOT_HASH:
        return

    config.install_snapshot(snapshot)
    _LAST_APPLIED_SNAPSHOT_HASH = snap_hash


def _make_worker_fn(fn: Callable[..., Any], snapshot: Any) -> Callable[..., Any]:
    """Wrap ``fn`` so each worker call applies the parent's snapshot first."""

    def worker_fn(*args: Any, **kwargs: Any) -> Any:
        _apply_snapshot_if_changed(snapshot)
        return fn(*args, **kwargs)

    return worker_fn


class LocalProcessScheduler:
    """Scheduler backed by loky's reusable process executor.

    Workers receive a ``ConfigSnapshot`` via closure and apply it to their
    own global config at chunk start. Cache state is per-worker (fresh
    process, empty caches at start).
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
        from pyphi.parallel.scheduler import ChunkingPolicy
        from pyphi.parallel.scheduler import ProgressPolicy
        from pyphi.parallel.scheduler import ShortcircuitPolicy

        chunking = chunking or ChunkingPolicy()
        progress = progress or ProgressPolicy()
        shortcircuit = shortcircuit or ShortcircuitPolicy()
        snapshot = config_snapshot if config_snapshot is not None else config.snapshot()

        from pyphi.parallel.sampling import compute_chunksize

        items_list = list(items)
        total = len(items_list)

        chunksize, sampled_iter = compute_chunksize(
            items_list,
            target_seconds=chunking.target_seconds,
            fn=fn,
            sequential_threshold=chunking.sequential_threshold,
            explicit_chunksize=chunking.chunksize,
        )
        items_list = list(sampled_iter)
        iterables: tuple[Iterable[Any], ...] = (items_list, *more_items)

        wrapped_fn = _make_worker_fn(fn, snapshot)

        def _reduce_wrapper(results: Iterable[Any], **_: Any) -> Any:
            return reducer(results)

        local_mr = LocalMapReduce(
            map_func=wrapped_fn,
            iterables=iterables,
            reduce_func=_reduce_wrapper,
            reduce_kwargs={},
            chunksize=chunksize,
            sequential_threshold=chunking.sequential_threshold,
            size_func=chunking.size_func,
            shortcircuit_func=shortcircuit.func,
            shortcircuit_callback=shortcircuit.callback,
            ordered=ordered,
            map_kwargs=map_kwargs,
            progress=progress.enabled,
            desc=progress.desc,
            total=total,
        )
        return local_mr.run()
