# parallel/__init__.py
"""Provides an interface for parallel computation.

This module provides a unified MapReduce interface for parallel computation
with pluggable backends. Currently supports:

- **local**: Fast single-machine parallelization using ProcessPoolExecutor
  (default, ~1-5ms overhead per task)

Future backends planned:
- **dask**: Cluster support for large-scale computations

Backend selection:
- Use `pyphi.config.parallel_backend = "local"` (default)
- Or pass `backend="local"` to MapReduce

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
from textwrap import indent
from typing import Any

from more_itertools import flatten
from tqdm.auto import tqdm

from pyphi.conf import config
from pyphi.conf import fallback
from pyphi.utils import try_len

from .tree import get_constraints

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
    """Adapt MapReduce-style ``(reduce_func, reduce_kwargs)`` to a 1-arg reducer."""
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


class MapReduce:
    """Unified map-reduce engine with pluggable backends.

    Supports tree-structured parallel computation with configurable
    depth/branching, short-circuiting, and progress tracking.

    Backends:
    - "local" (default): ProcessPoolExecutor for single-machine parallelization
    - "auto": Auto-detect best available backend (currently always local)

    Example:
        >>> results = MapReduce(
        ...     lambda x: x ** 2,
        ...     range(100),
        ...     parallel=True,
        ... ).run()
    """

    def __init__(
        self,
        map_func: Callable,
        iterable: Iterable,
        *iterables: Iterable,
        reduce_func: Callable | None = None,
        reduce_kwargs: dict | None = None,
        parallel: bool = True,
        ordered: bool = False,
        total: int | None = None,
        chunksize: int | None = None,
        sequential_threshold: int = 1,
        max_depth: int | None = None,
        max_size: int | None = None,
        max_leaves: int | None = None,
        branch_factor: int = 2,
        shortcircuit_func: Callable = false,
        shortcircuit_callback: Callable | None = None,
        shortcircuit_callback_args: Any = None,
        progress: bool | None = None,
        desc: str | None = None,
        map_kwargs: dict | None = None,
        backend: str = "auto",
    ):
        """Initialize MapReduce computation.

        Args:
            map_func: Function to apply to each element.
            iterable: Primary iterable of elements.
            *iterables: Additional iterables (zipped with primary).
            reduce_func: Function to reduce results (default: flatten).
            reduce_kwargs: Keyword arguments for reduce_func.
            parallel: Whether to parallelize (True) or run sequentially.
            ordered: Whether to preserve input order in results.
            total: Total number of elements (for progress bar).
            chunksize: Size of chunks for parallel processing.
            sequential_threshold: Minimum elements to trigger parallelization.
            max_depth: Maximum tree depth for hierarchical execution.
            max_size: Maximum tree size.
            max_leaves: Maximum leaf nodes.
            branch_factor: Branching factor for tree.
            shortcircuit_func: Function to check for early termination.
            shortcircuit_callback: Callback when short-circuiting.
            shortcircuit_callback_args: Arguments for callback.
            progress: Whether to show progress bar.
            desc: Progress bar description.
            map_kwargs: Keyword arguments for map_func.
            backend: Backend to use ("auto", "local").
        """
        self.map_func = map_func
        self.iterables = (iterable, *iterables)
        self.reduce_func = fallback(reduce_func, _flatten)
        self.reduce_kwargs = fallback(reduce_kwargs, {})
        self.parallel = parallel
        self.ordered = ordered
        self.total = fallback(try_len(*self.iterables), total)
        self.shortcircuit_func = shortcircuit_func
        self.shortcircuit_callback = shortcircuit_callback
        self.shortcircuit_callback_args = shortcircuit_callback_args
        self.progress = fallback(progress, config.infrastructure.progress_bars)
        self.desc = desc
        self.map_kwargs = fallback(map_kwargs, {})
        self._shortcircuit_callback = shortcircuit_callback
        self.backend = self._resolve_backend(backend)

        if self.parallel:
            self.constraints = get_constraints(
                total=self.total,
                chunksize=chunksize,
                sequential_threshold=sequential_threshold,
                max_depth=max_depth,
                max_size=max_size,
                max_leaves=max_leaves,
                branch_factor=branch_factor,
            )
            # Get the tree specifications
            self.tree = self.constraints.simulate()
            # Get the chunksize
            self.chunksize = self.constraints.get_initial_chunksize()
            # Default to cancelling all futures
            if self.shortcircuit_callback is None:
                self.shortcircuit_callback = cancel_all

        # State
        self.progress_bar = None
        self.error = None
        self.done = False
        self.result = None

    def _resolve_backend(self, backend: str) -> str:
        """Validate ``backend`` and normalize ``"auto"`` to ``"local"``.

        ``MapReduce`` always dispatches through the local-process pool
        (``LocalMapReduce``); explicit ``"thread"`` / ``"dask"`` selection is
        done through :func:`pyphi.parallel.scheduler.default_scheduler`. This
        method exists so a misspelled backend name fails at construction
        rather than later inside the scheduler.
        """
        valid = {"auto", "local", "process", "thread", "dask"}
        if backend not in valid:
            raise ValueError(
                f"Unknown backend: {backend}. Available backends: {sorted(valid)}"
            )
        return "local" if backend in {"auto", "local", "process"} else backend

    def _repr_attrs(self) -> list[str]:
        attrs = [
            "map_func",
            "map_kwargs",
            "iterables",
            "reduce_func",
            "reduce_kwargs",
            "parallel",
            "ordered",
            "total",
            "shortcircuit_func",
            "shortcircuit_callback",
            "shortcircuit_callback_args",
            "progress",
            "desc",
            "backend",
        ]
        if self.parallel:
            attrs += ["constraints", "tree"]
        return attrs

    def __repr__(self) -> str:
        data = [f"{attr}={getattr(self, attr)}" for attr in self._repr_attrs()]
        return "\n".join(
            [f"{self.__class__.__name__}(", indent("\n".join(data), "  "), ")"]
        )

    def _run_parallel(self) -> Any:
        """Perform the computation in parallel using local backend.

        Captures the parent's ``ConfigSnapshot`` and wraps the map
        function so each worker installs the snapshot before applying
        ``map_func``. Without this, workers compute under their stale
        default config and the result records the wrong snapshot.
        """
        from .backends.local_process import LocalMapReduce
        from .backends.local_process import _make_worker_fn

        snapshot = config.snapshot()
        wrapped_map_func = _make_worker_fn(self.map_func, snapshot)

        local_mr = LocalMapReduce(
            map_func=wrapped_map_func,
            iterables=self.iterables,
            reduce_func=self.reduce_func,
            reduce_kwargs=self.reduce_kwargs,
            constraints=self.constraints,
            tree=self.tree,
            chunksize=self.chunksize,
            shortcircuit_func=self.shortcircuit_func,
            shortcircuit_callback=self.shortcircuit_callback,
            ordered=self.ordered,
            map_kwargs=self.map_kwargs,
            progress=self.progress,
            desc=self.desc or "",
            total=self.total,
        )

        try:
            self.result = local_mr.run()
            self.done = True
            return self.result
        except Exception as e:
            self.error = e
            raise e

    def _run_sequential(self) -> Any:
        """Perform the computation serially."""
        try:
            results = _map_sequential(self.map_func, *self.iterables, **self.map_kwargs)
            if self.progress:
                results = tqdm(results, total=self.total, desc=self.desc)
            results = get(
                results,
                shortcircuit_func=self.shortcircuit_func,
                shortcircuit_callback=self.shortcircuit_callback,
                shortcircuit_callback_args=self.shortcircuit_callback_args,
            )
            self.result = _reduce(
                results, self.reduce_func, self.reduce_kwargs, branch=False
            )
            self.done = True
            return self.result
        except Exception as e:
            self.error = e
            raise e

    def run(self) -> Any:
        """Perform the computation."""
        if self.done:
            return self.result
        if self.parallel and self.tree.depth > 1:
            return self._run_parallel()
        return self._run_sequential()
