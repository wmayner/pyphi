# parallel/backends/local.py
"""Local backend for parallel computation using loky.

Uses loky (via joblib) instead of ProcessPoolExecutor for cloudpickle support,
allowing functions defined in __main__ (e.g., Jupyter notebooks) to be
serialized and sent to worker processes.
"""

from __future__ import annotations

import logging
import multiprocessing
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from concurrent.futures import Future
from concurrent.futures import as_completed
from typing import Any

from joblib.externals.loky import get_reusable_executor
from more_itertools import chunked_even
from more_itertools import flatten

from pyphi.conf import config
from pyphi.conf import fallback
from pyphi.parallel.tree import TreeConstraints
from pyphi.parallel.tree import TreeSpec

from .progress import LocalProgressBar

log = logging.getLogger(__name__)


def get_num_processes() -> int:
    """Return the number of processes to use in parallel."""
    cpu_count = multiprocessing.cpu_count()

    if config.infrastructure.parallel_workers == 0:
        raise ValueError("Invalid PARALLEL_WORKERS; value may not be 0.")

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
                "Invalid PARALLEL_WORKERS; negative value is too negative: "
                f"requesting {num} workers, {cpu_count} CPUs available."
            )
        return num

    return config.infrastructure.parallel_workers


def false(*_args, **_kwargs) -> bool:
    """Default short-circuit function that never short-circuits."""
    return False


def _flatten(items: Iterable, branch: bool = False) -> list:
    """Flatten results if branching occurred."""
    if branch:
        items = flatten(items)
    return list(items)


def _map_sequential(func: Callable, *arglists, **kwargs) -> Iterator:
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
        constraints: TreeConstraints,
        tree: TreeSpec,
        chunksize: int,
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
        self.constraints = constraints
        self.tree = tree
        self.chunksize = chunksize
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
        self._futures: list[Future] = []

    def _cancel_remaining(self, futures: list[Future]) -> None:
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

        chunked_iterables = [
            list(chunked_even(it, self.chunksize)) for it in materialized
        ]

        # Yield tuples of corresponding chunks
        yield from zip(*chunked_iterables, strict=False)

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

            # If tree depth is 1 or less, run sequentially
            if self.tree.depth <= 1:
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
