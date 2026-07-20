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
                'package; install it with `pip install "pyphi[cluster]"`.'
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

        snapshot = config_snapshot if config_snapshot is not None else config.snapshot()

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
