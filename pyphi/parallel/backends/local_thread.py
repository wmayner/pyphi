"""Thread-pool scheduler.

Workers run in the parent process, so they share the parent's global
config and caches. Snapshot apply is a no-op (the parent's live globals
already reflect the captured snapshot).

Best suited for free-threaded Python (3.13t+) where multiple OS threads can
execute Python concurrently. Under standard CPython the GIL limits the
throughput benefit but the scheduler still avoids pickle overhead and is
useful for IO-bound work.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from typing import Any

from pyphi.parallel.backends.progress import LocalProgressBar
from pyphi.parallel.scheduler import ChunkingPolicy
from pyphi.parallel.scheduler import ProgressPolicy
from pyphi.parallel.scheduler import ShortcircuitPolicy


class LocalThreadScheduler:
    """Scheduler backed by ``concurrent.futures.ThreadPoolExecutor``."""

    @property
    def supports_shared_state(self) -> bool:
        return True

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
        # Threads share the parent's globals, so there is no snapshot to apply.
        del config_snapshot

        chunking = chunking or ChunkingPolicy()
        progress = progress or ProgressPolicy()
        shortcircuit = shortcircuit or ShortcircuitPolicy()
        map_kwargs = map_kwargs or {}

        # Mark the parent PID so the snapshot-apply hook short-circuits when
        # called in-thread (threads share parent's globals). Restore the
        # previous value afterwards: a permanently latched PID would disable
        # snapshot installs for the rest of this process's life — silently
        # stale config when this process is itself a loky worker that later
        # receives a new parent snapshot.
        from pyphi.parallel.backends import local_process

        previous_parent_pid = local_process._PARENT_PID
        local_process._PARENT_PID = os.getpid()
        try:
            return self._map_reduce(
                fn,
                items,
                *more_items,
                reducer=reducer,
                chunking=chunking,
                progress=progress,
                shortcircuit=shortcircuit,
                ordered=ordered,
                map_kwargs=map_kwargs,
            )
        finally:
            local_process._PARENT_PID = previous_parent_pid

    def _map_reduce(
        self,
        fn: Callable[..., Any],
        items: Iterable[Any],
        *more_items: Iterable[Any],
        reducer: Callable[[Iterable[Any]], Any],
        chunking: Any,
        progress: Any,
        shortcircuit: Any,
        ordered: bool,
        map_kwargs: dict[str, Any],
    ) -> Any:
        from pyphi.parallel import get_num_processes

        num_workers = get_num_processes()

        materialized = [list(it) for it in (items, *more_items)]
        if not materialized or not materialized[0]:
            return reducer([])

        # Updates happen only in this (collecting) thread, never in workers,
        # so the bar needs no locking.
        progress_bar = (
            LocalProgressBar(
                total=progress.total
                if progress.total is not None
                else len(materialized[0]),
                desc=progress.desc,
            )
            if progress.enabled
            else None
        )
        try:
            if len(materialized[0]) < chunking.sequential_threshold:
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

            # Group items into chunks so each future carries many items:
            # per-item futures pay dispatch overhead per item and ignore the
            # caller's chunking policy. Without an explicit chunksize, items
            # are split evenly across the workers.
            from pyphi.parallel.backends.local_process import _process_chunk
            from pyphi.parallel.chunking import iter_chunks

            n = min(len(it) for it in materialized)
            chunksize = chunking.chunksize or math.ceil(n / num_workers)
            chunks = list(
                iter_chunks(
                    materialized,
                    chunksize=chunksize,
                    num_workers=num_workers,
                    size_func=chunking.size_func,
                )
            )

            results = []
            short_circuited = False
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(
                        _process_chunk, chunk, fn, map_kwargs, shortcircuit.func
                    )
                    for chunk in chunks
                ]
                # Collect in submission order when the caller asked for
                # original order or a short-circuit predicate is active. When
                # short-circuiting, the collected subset is truncated at the
                # first triggering result, so completion order would make that
                # subset — and any order-sensitive reduction over it (e.g. tie
                # resolution among the surviving candidates) — depend on
                # thread scheduling. Submission order yields the same prefix
                # as sequential evaluation.
                iterator: Iterable[Any] = (
                    futures if ordered or shortcircuit.active else as_completed(futures)
                )
                # A worker exception cancels the pending futures before
                # propagating; otherwise the executor's shutdown would block
                # until every orphaned future had run to completion.
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
                                    if not remaining.done():
                                        remaining.cancel()
                                shortcircuit.fire(results)
                                break
                        if short_circuited:
                            break
                except BaseException:
                    for remaining in futures:
                        if not remaining.done():
                            remaining.cancel()
                    raise

            return reducer(results)
        finally:
            if progress_bar is not None:
                progress_bar.close()
