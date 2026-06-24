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

import os
from collections.abc import Callable
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from typing import Any

from pyphi.parallel.scheduler import ChunkingPolicy
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
        # threads share parent globals; progress hookup deferred to a follow-up
        del config_snapshot, progress

        chunking = chunking or ChunkingPolicy()
        shortcircuit = shortcircuit or ShortcircuitPolicy()
        map_kwargs = map_kwargs or {}

        # Mark the parent PID so the snapshot-apply hook short-circuits when
        # called in-thread (threads share parent's globals).
        from pyphi.parallel.backends import local_process

        local_process._PARENT_PID = os.getpid()

        from pyphi.parallel.backends.local_process import get_num_processes

        num_workers = get_num_processes()

        materialized = [list(it) for it in (items, *more_items)]
        if not materialized or not materialized[0]:
            return reducer([])

        if len(materialized[0]) < chunking.sequential_threshold:
            results = [
                fn(*args, **map_kwargs) for args in zip(*materialized, strict=False)
            ]
            return reducer(results)

        results: list[Any] = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(fn, *args, **map_kwargs)
                for args in zip(*materialized, strict=False)
            ]
            iterator: Iterable[Any] = futures if ordered else as_completed(futures)
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
