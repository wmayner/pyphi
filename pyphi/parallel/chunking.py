# parallel/chunking.py
"""Adaptive chunking for heterogeneous workloads.

PyPhi's combinatorial computations involve elements that vary greatly in
computational cost. For example, mechanisms range from size 1 to 2^n-1,
and larger mechanisms have exponentially more purviews to evaluate.

Uniform chunking leads to load imbalance where some workers get all the
expensive elements. Adaptive chunking groups elements by estimated work
to create balanced chunks.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from typing import Any


def estimate_work_size(element: Any, context: str | None = None) -> float:
    """Estimate computational cost of processing an element.

    Context-specific heuristics based on PyPhi's computation patterns:
    - 'mechanism': Cost scales with mechanism size and number of purviews
    - 'partition': Cost scales with number of nodes squared
    - 'purview': Cost scales with purview size
    - 'cut': Cost scales with number of possible cuts

    Args:
        element: The element to estimate work for.
        context: The computational context ('mechanism', 'partition',
                 'purview', 'cut', or None for uniform weight).

    Returns:
        Estimated computational cost as a float.
    """
    if context is None:
        return 1.0

    if context == "mechanism":
        # Mechanism evaluation cost: roughly len(mechanism) * 2^len(mechanism)
        # since we evaluate all possible purviews
        try:
            size = len(element)
            return size * (2**size)
        except (TypeError, AttributeError):
            return 1.0

    elif context == "partition":
        # Partition evaluation cost: quadratic in number of nodes
        try:
            if hasattr(element, "nodes"):
                return len(element.nodes) ** 2
            return len(element) ** 2
        except (TypeError, AttributeError):
            return 1.0

    elif context == "purview":
        # Purview evaluation cost: linear in purview size
        try:
            return len(element)
        except (TypeError, AttributeError):
            return 1.0

    elif context == "cut":
        # Cut evaluation cost: roughly 2^n for n nodes
        try:
            if hasattr(element, "indices"):
                return 2 ** len(element.indices)
            return 2 ** len(element)
        except (TypeError, AttributeError):
            return 1.0

    elif context == "complex":
        # Complex evaluation cost: combinatorial in candidate systems
        try:
            size = len(element)
            return size**2
        except (TypeError, AttributeError):
            return 1.0

    else:
        return 1.0


def adaptive_chunk(
    iterable: Iterable,
    target_work_per_chunk: float,
    context: str | None = None,
    max_chunks: int | None = None,
    size_func: Callable[[Any], float] | None = None,  # pyright: ignore[reportRedeclaration]
) -> Iterator[list]:
    """Chunk iterable to balance work across chunks.

    Instead of equal-size chunks, creates equal-work chunks by grouping
    elements based on estimated computational cost.

    Args:
        iterable: The items to chunk.
        target_work_per_chunk: Target total work per chunk.
        context: Context for work estimation (see estimate_work_size).
        max_chunks: Maximum number of chunks to create.
        size_func: Optional custom function to estimate element work size.
                   If provided, overrides context-based estimation.

    Yields:
        Lists of elements with approximately equal total work.

    Example:
        >>> # Elements with varying sizes
        >>> items = [(1,), (1, 2), (1, 2, 3), (1,)]
        >>> chunks = list(adaptive_chunk(items, target_work_per_chunk=10,
        ...                              context='mechanism'))
        >>> # Chunks are balanced by work, not by count
    """
    if size_func is None:

        def size_func(x):  # pyright: ignore[reportRedeclaration]
            return estimate_work_size(x, context)

    current_chunk: list = []
    current_work: float = 0.0
    chunks_created = 0

    for element in iterable:
        work = size_func(element)

        # Start a new chunk if current is full (and not empty)
        # or if we've exceeded the target work
        if current_work + work > target_work_per_chunk and current_chunk:
            yield current_chunk
            chunks_created += 1

            # Check max_chunks limit
            if max_chunks is not None and chunks_created >= max_chunks - 1:
                # Last chunk gets everything remaining
                current_chunk = [element]
                for remaining in iterable:
                    current_chunk.append(remaining)
                if current_chunk:
                    yield current_chunk
                return

            current_chunk = [element]
            current_work = work
        else:
            current_chunk.append(element)
            current_work += work

    # Yield final chunk
    if current_chunk:
        yield current_chunk


def estimate_total_work(
    iterable: Iterable,
    context: str | None = None,
    size_func: Callable[[Any], float] | None = None,  # pyright: ignore[reportRedeclaration]
) -> tuple[float, list]:
    """Estimate total work and materialize iterable.

    Useful for determining optimal chunking parameters before chunking.

    Args:
        iterable: The items to estimate work for.
        context: Context for work estimation.
        size_func: Optional custom function for work estimation.

    Returns:
        Tuple of (total_work, materialized_list).
    """
    if size_func is None:

        def size_func(x):  # pyright: ignore[reportRedeclaration]
            return estimate_work_size(x, context)

    items = list(iterable)
    total_work = sum(size_func(item) for item in items)
    return total_work, items


def calculate_target_work(
    total_work: float,
    num_workers: int,
    min_chunks: int = 1,
    max_chunks_per_worker: int = 4,
) -> float:
    """Calculate optimal target work per chunk.

    Aims to create enough chunks for good load balancing while avoiding
    excessive overhead from too many small chunks.

    Args:
        total_work: Total estimated work.
        num_workers: Number of worker processes.
        min_chunks: Minimum number of chunks to create.
        max_chunks_per_worker: Maximum chunks per worker for overhead control.

    Returns:
        Target work per chunk.
    """
    min_target = total_work / (num_workers * max_chunks_per_worker)
    max_target = total_work / min_chunks

    # Aim for 2-4 chunks per worker for good balance
    target_chunks = num_workers * 2
    target = total_work / target_chunks

    return max(min_target, min(max_target, target))


def chunked_by_work(
    iterable: Iterable,
    num_workers: int,
    context: str | None = None,
    size_func: Callable[[Any], float] | None = None,  # pyright: ignore[reportRedeclaration]
) -> Iterator[list]:
    """High-level chunking API that auto-calculates optimal parameters.

    Convenience function that estimates total work, calculates optimal
    chunk targets, and returns balanced chunks.

    Args:
        iterable: The items to chunk.
        num_workers: Number of worker processes.
        context: Context for work estimation.
        size_func: Optional custom function for work estimation.

    Yields:
        Lists of elements with balanced total work.
    """
    if size_func is None:

        def size_func(x):  # pyright: ignore[reportRedeclaration]
            return estimate_work_size(x, context)

    total_work, items = estimate_total_work(
        iterable, context=context, size_func=size_func
    )

    if total_work == 0:
        if items:
            yield items
        return

    target_work = calculate_target_work(total_work, num_workers)

    yield from adaptive_chunk(
        items,
        target_work_per_chunk=target_work,
        context=context,
        size_func=size_func,
    )
