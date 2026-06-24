# parallel/__init__.py
"""Provides an interface for parallel computation.

This module provides :func:`map_reduce`, a single parallel map-reduce entry
point over the Scheduler Protocol with pluggable backends. Currently supports:

- **local**: Fast single-machine parallelization using a process pool
  (default, ~1-5ms overhead per task)

Future backends planned:
- **dask**: Cluster support for large-scale computations

Backend selection:
- Use `pyphi.config.parallel_backend = "local"` (default)
- Or pass `backend="local"` to :func:`map_reduce`

Example:
    >>> import pyphi
    >>> with pyphi.config.override(parallel=True):
    ...     pyphi.config.infrastructure.parallel
    True
"""

from __future__ import annotations

import logging
import multiprocessing
from collections.abc import Callable
from collections.abc import Iterable
from typing import Any

from more_itertools import flatten
from tqdm.auto import tqdm

from pyphi.conf import config
from pyphi.conf import fallback
from pyphi.utils import try_len

log = logging.getLogger(__name__)


def get_num_processes() -> int:
    """Return the number of processes to use in parallel."""
    cpu_count = multiprocessing.cpu_count()

    if config.infrastructure.parallel_workers == 0:
        raise ValueError("Invalid parallel_workers; value may not be 0.")

    if cpu_count < config.infrastructure.parallel_workers:
        log.info(
            "Requesting %s workers; only %s CPUs available",
            config.infrastructure.parallel_workers,
            cpu_count,
        )
        return cpu_count

    if config.infrastructure.parallel_workers < 0:
        num = cpu_count + config.infrastructure.parallel_workers + 1
        if num <= 0:
            raise ValueError(
                "Invalid parallel_workers; negative value is too negative: "
                f"requesting {num} workers, {cpu_count} CPUs available."
            )
        return num

    return config.infrastructure.parallel_workers


def false(*args, **kwargs) -> bool:
    """Default short-circuit function that never short-circuits."""
    return False


def shortcircuit(
    items: Iterable,
    shortcircuit_func: Callable = false,
    shortcircuit_callback: Callable | None = None,
    shortcircuit_callback_args: Any = None,
):
    """Yield from an iterable, stopping early if a certain value is found."""
    for result in items:
        yield result
        if shortcircuit_func(result):
            if shortcircuit_callback is not None:
                shortcircuit_callback(fallback(shortcircuit_callback_args, items))
            return


def _flatten(items: Iterable, branch: bool = False) -> list:
    """Flatten results if branching occurred."""
    if branch:
        items = flatten(items)
    return list(items)


def _map_sequential(func: Callable, *arglists, **kwargs):
    """Map function over arguments sequentially."""
    for args in zip(*arglists, strict=False):
        yield func(*args, **kwargs)


def _reduce(
    results: Iterable, reduce_func: Callable, reduce_kwargs: dict, branch: bool
) -> Any:
    """Apply reduction function to results."""
    if reduce_func is _flatten:
        return reduce_func(results, branch=branch)
    return reduce_func(results, **reduce_kwargs)


def get(
    items: Iterable,
    shortcircuit_func: Callable = false,
    shortcircuit_callback: Callable | None = None,
    shortcircuit_callback_args: Any = None,
):
    """Iterate over results, optionally returning early if a value is found."""
    shortcircuit_callback_args = fallback(shortcircuit_callback_args, items)
    return shortcircuit(
        items,
        shortcircuit_func=shortcircuit_func,
        shortcircuit_callback=shortcircuit_callback,
        shortcircuit_callback_args=shortcircuit_callback_args,
    )


def cancel_all(futures: Iterable, *args, **kwargs) -> list:
    """Cancel all futures.

    For local backend, attempts to cancel concurrent.futures.Future objects.
    Returns the list of futures that were processed.
    """
    from concurrent.futures import Future

    result = []
    for future in futures:
        if isinstance(future, Future) and not future.done():
            future.cancel()
        result.append(future)
    return result


def _bind_reducer(
    reduce_func: Callable[..., Any], reduce_kwargs: dict[str, Any] | None
) -> Callable[..., Any]:
    """Adapt a ``(reduce_func, reduce_kwargs)`` pair to a 1-arg reducer."""
    reduce_kwargs = reduce_kwargs or {}
    if reduce_func is _flatten:
        return lambda results: _flatten(results, branch=False)
    if reduce_kwargs:
        return lambda results: reduce_func(results, **reduce_kwargs)
    return reduce_func


def map_reduce(
    fn: Callable[..., Any],
    items: Iterable[Any],
    *more_items: Iterable[Any],
    reduce_func: Callable[..., Any] = _flatten,
    reduce_kwargs: dict[str, Any] | None = None,
    parallel: bool = True,
    ordered: bool = False,
    total: int | None = None,
    chunksize: int | None = None,
    sequential_threshold: int = 1,
    shortcircuit_func: Callable[..., Any] = false,
    shortcircuit_callback: Callable[..., Any] | None = None,
    shortcircuit_callback_args: Any = None,
    progress: bool | None = None,
    desc: str | None = None,
    map_kwargs: dict[str, Any] | None = None,
    size_func: Callable[..., float] | None = None,
    backend: str = "auto",
) -> Any:
    """Map ``fn`` over ``items`` (zipped with ``more_items``) and reduce.

    Runs in parallel through the scheduler selected by ``backend`` (or
    ``config.infrastructure.parallel_backend``). With ``parallel=False`` it
    runs serially in-process. ``reduce_func`` defaults to flattening the
    per-item results into a list. ``size_func`` returns a relative per-item
    cost estimate used to pack cost-balanced chunks (parent-side, so it must
    be cheap); ``None`` packs equal-count chunks.
    """
    if size_func is not None and ordered:
        raise ValueError(
            "size_func cost-balancing reorders items across chunks and is "
            "incompatible with ordered=True"
        )
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
            chunksize=chunksize,
            sequential_threshold=sequential_threshold,
            size_func=size_func,
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
