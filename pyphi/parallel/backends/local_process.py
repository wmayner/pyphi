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

import functools
import logging
import math
import threading
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

    Items are grouped into chunks (evenly, or cost-balanced when a
    ``size_func`` is given), each chunk is submitted to a worker as one
    future, and the per-chunk result lists are concatenated and reduced.
    Loky's cloudpickle support lets functions defined in ``__main__`` (e.g.
    in a Jupyter notebook) be serialized to workers, and its reusable pool
    keeps per-task overhead low (roughly 1-5 ms). A short-circuit predicate
    stops collection early and cancels the remaining futures. Progress is
    reported through :class:`~pyphi.parallel.backends.progress.LocalProgressBar`,
    which renders in both terminals and notebooks.
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
        shortcircuit_callback_args: Any = None,
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
        self.shortcircuit_callback_args = shortcircuit_callback_args
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

    def _fire_shortcircuit_callback(self, default: Any) -> None:
        """Invoke the callback with the caller's args, or ``default``."""
        if self.shortcircuit_callback is not None:
            self.shortcircuit_callback(
                self.shortcircuit_callback_args
                if self.shortcircuit_callback_args is not None
                else default
            )

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

        from pyphi.parallel.chunking import iter_chunks

        yield from iter_chunks(
            materialized,
            chunksize=self.chunksize,
            num_workers=get_num_processes(),
            size_func=self.size_func,
        )

    def _should_run_parallel(self) -> bool:
        """Parallelize whenever the chunker would produce more than one chunk.

        ``sequential_threshold`` is the dispatch gate: below it, per-item
        cost is assumed too small to amortize process dispatch. At or above
        it, the chunker's ``num_workers`` chunk-count floor spreads the
        workload across cores even when it fits within a single
        ``chunksize`` — the chunksize governs chunk granularity, not
        dispatch. (When the chunksize was cost-sampled rather than
        explicitly configured, the scheduler folds it into
        ``sequential_threshold``, since a sampled chunksize estimates the
        number of items per ~1 s of work.)

        Measured basis (``benchmarks/b18_dispatch_gate.py``, 11 workers):
        warm-pool parallel dispatch beats sequential 3-4x for workloads of
        expensive items below one chunksize (~13 ms/item purview MIPs at
        64-230 of chunksize 256; ~1.3 ms/item system partitions at
        64-2048 of chunksize 4096), and loses only when total work is
        tens of ms (µs-scale relation construction, ~50 µs mechanism
        partitions) — which the per-level ``sequential_threshold``
        defaults now guard against.
        """
        if self.total is None:
            return True  # unknown length; let the executor chunk and dispatch
        if self.total < self.sequential_threshold:
            return False
        if not self.chunksize:
            return self.total > 1
        k = max(math.ceil(self.total / self.chunksize), get_num_processes())
        # a single chunk → no parallel benefit
        return min(k, self.total) > 1

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
                self._fire_shortcircuit_callback(collected)
                break

        self.result = _reduce(collected, self.reduce_func, self.reduce_kwargs)
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
            self.result = _reduce([], self.reduce_func, self.reduce_kwargs)
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
                            self._fire_shortcircuit_callback(futures)
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
                            self._fire_shortcircuit_callback(futures)
                            break
                    if short_circuited:
                        break
        except BaseException:
            self._cancel_remaining(futures)
            raise

        # Final reduction - apply user's reduce function
        self.result = _reduce(results, self.reduce_func, self.reduce_kwargs)
        self.done = True
        return self.result


_LAST_APPLIED_SNAPSHOT_HASH: int | None = None
_PARENT_PID: int | None = None
_SNAPSHOT_LOCK = threading.Lock()


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


def _make_worker_fn(fn: Callable[..., Any], snapshot: Any) -> Callable[..., Any]:
    """Wrap ``fn`` so each worker call applies the parent's snapshot first."""
    snap_hash = hash(repr(snapshot))

    def worker_fn(*args: Any, **kwargs: Any) -> Any:
        _apply_snapshot_if_changed(snapshot, snap_hash)
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

        if not hasattr(items, "__len__"):
            # Unknown-length input: decide sequential vs parallel without
            # draining the iterator. Items are consumed and mapped one at a
            # time up to ``sequential_threshold``, so a short workload — or a
            # short-circuit — finishes having pulled only the items it used.
            # At the threshold, the remainder is materialized and dispatched
            # in parallel, with the already-computed prefix prepended before
            # reduction (matching the sequential-evaluation prefix that
            # ordered / short-circuit collection guarantees).
            threshold = max(chunking.sequential_threshold, 1)
            kwargs = map_kwargs or {}
            prefix: list[Any] = []
            zipped = zip(items, *more_items, strict=False)
            for args in zipped:
                value = fn(*args, **kwargs)
                prefix.append(value)
                if shortcircuit.func(value):
                    shortcircuit.fire(prefix)
                    return reducer(prefix)
                if len(prefix) >= threshold:
                    break
            else:
                # Exhausted below the threshold: purely sequential.
                return reducer(prefix)
            rest = list(zipped)
            if not rest:
                return reducer(prefix)
            columns = tuple(list(col) for col in zip(*rest, strict=True))
            items, more_items = columns[0], columns[1:]
            base_reducer = reducer

            def _prefixed_reducer(results: Iterable[Any]) -> Any:
                return base_reducer([*prefix, *results])

            reducer = _prefixed_reducer

        items_list = list(items)
        total = len(items_list)

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
        iterables: tuple[Iterable[Any], ...] = (items_list, *more_items)

        # A sampled chunksize estimates the number of items per
        # ``target_seconds`` of work, so a workload that fits within one
        # such chunk is not worth dispatching; fold it into the threshold.
        # An explicitly configured chunksize governs granularity only.
        sequential_threshold = chunking.sequential_threshold
        if chunking.chunksize is None:
            sequential_threshold = max(sequential_threshold, chunksize + 1)

        wrapped_fn = _make_worker_fn(fn, snapshot)

        def _reduce_wrapper(results: Iterable[Any], **_: Any) -> Any:
            return reducer(results)

        local_mr = LocalMapReduce(
            map_func=wrapped_fn,
            iterables=iterables,
            reduce_func=_reduce_wrapper,
            reduce_kwargs={},
            chunksize=chunksize,
            sequential_threshold=sequential_threshold,
            size_func=chunking.size_func,
            shortcircuit_func=shortcircuit.func,
            shortcircuit_callback=shortcircuit.callback,
            shortcircuit_callback_args=shortcircuit.args,
            ordered=ordered,
            map_kwargs=map_kwargs,
            progress=progress.enabled,
            desc=progress.desc,
            total=total,
        )
        return local_mr.run()
