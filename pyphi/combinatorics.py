# combinatorics.py
"""Combinatorial utilities."""

from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Generator
from collections.abc import Iterable
from collections.abc import Sequence
from itertools import product
from typing import Any

import numpy as np

from .cache import cache

# TODO(4.0) move relevant functions from utils here


# TODO(docs) finish documenting
def pair_indices(n: int, m: int | None = None, k: int = 0) -> Generator[tuple[int, int]]:
    """Return indices of unordered pairs."""
    if m is None:
        m = n
    n, m = sorted([n, m])
    for i in range(n):
        for j in range(i + k, m):
            yield i, j


# TODO(docs) finish documenting
def pairs(seq: Sequence, k: int = 0) -> Generator[tuple[Any, Any]]:
    """Return unordered pairs of elements from a sequence.

    NOTE: This is *not* the Cartesian product.
    """
    for i, j in pair_indices(len(seq), k=k):
        yield seq[i], seq[j]


def combinations_with_nonempty_intersection_by_order(
    sets: Sequence[frozenset], min_size: int = 0, max_size: int | None = None
) -> dict[int, set[frozenset]]:
    """Return nonempty-intersection combinations grouped by size.

    The same combinations as :func:`combinations_with_nonempty_intersection`,
    bucketed into ``{size: {combination, ...}}``. Each combination is a set of
    the indices of the sets in that combination, not the sets themselves. Sizes
    with no combinations are omitted.

    Arguments:
        sets (Sequence[frozenset]): The sets to consider. Note that they must be
            ``frozensets``.

    Keyword Arguments:
        min_size (int): The minimum size of the combinations to return. Defaults
            to 0.
        max_size (int): The maximum size of the combinations to return. Defaults
            to ``None``, indicating all sizes.

    Returns:
        dict[int, set[frozenset]]: A mapping from combination size to
        combinations.
    """
    by_order: dict[int, set[frozenset]] = defaultdict(set)
    for combination in combinations_with_nonempty_intersection(
        sets, min_size=min_size, max_size=max_size
    ):
        by_order[len(combination)].add(combination)
    return dict(by_order)


def combinations_with_nonempty_intersection(
    sets: Sequence[frozenset], min_size: int = 0, max_size: int | None = None
) -> Generator[frozenset[int]]:
    """Yield index-combinations whose set-intersection is nonempty.

    Each yielded ``frozenset`` holds indices ``i`` into ``sets`` such that the
    intersection of the corresponding sets is nonempty. Combinations are
    enumerated by depth-first search over indices in increasing order, pruning a
    whole subtree as soon as the running intersection becomes empty (sound
    because intersection is monotone non-increasing under adding elements).
    Singletons are never yielded; the effective minimum size is
    ``max(2, min_size)``.

    Arguments:
        sets (Sequence[frozenset]): The sets to consider. Note that they must be
            ``frozensets``.

    Keyword Arguments:
        min_size (int): The minimum size of the combinations to yield. Defaults
            to 0.
        max_size (int): The maximum size of the combinations to yield. If
            ``None`` (the default), there is no upper bound.
    """
    n = len(sets)
    effective_min = max(2, min_size)
    upper = n if max_size is None else max_size
    if upper < effective_min:
        return

    def _extend(
        start: int, chosen: list[int], running: frozenset
    ) -> Generator[frozenset[int]]:
        size = len(chosen)
        if size >= effective_min:
            yield frozenset(chosen)
        if size >= upper:
            return
        for i in range(start, n):
            new_running = running & sets[i]
            if new_running:
                chosen.append(i)
                yield from _extend(i + 1, chosen, new_running)
                chosen.pop()

    for i in range(n):
        if sets[i]:
            yield from _extend(i + 1, [i], sets[i])


@cache(cache={}, maxmem=None)
def num_subsets_larger_than_one_element(n: int) -> int:
    """Return the number of subsets on N elements with size >1.

    |X| = |P(n)| - |{S ∈ P(n) | |S| = 1}| - |{S ∈ P(n) | |S| = 0}|
        = 2^n    - (n choose 1)             - |{ø}|
        = 2^n    - n                        - 1
    """
    return 2**n - n - 1  # type: ignore[no-any-return]


def sum_of_minimum_among_subsets(values: Sequence[float]) -> float:
    """Return the sum of the minimum of all subsets with size >1 of the values."""
    # This series counts, from i = 0 to (len(values) - 1), the number of subsets
    # of values of size >1 such that value i is included in all subsets.
    # Since each value is fixed to be in all subsets, this formula differs from
    # `num_subsets_larger_than_one_element`.
    counts = 2 ** (np.arange(len(values), 0, -1) - 1) - 1
    # Sorting ensures that we're taking the minimum of values for each subset
    return float(np.sum(np.sort(values) * counts))


def sum_of_minimum_over_size_among_subsets(values: Sequence[float]) -> float:
    """Return the sum of ``min(S) / |S|`` over all subsets ``S`` with size > 1.

    For values sorted ascending as ``v_0 <= ... <= v_{n-1}``, ``v_i`` is the
    minimum of exactly those subsets containing ``i`` whose other elements all
    come from the ``a = n - 1 - i`` larger positions. Summing ``1/|S|`` over
    those subsets gives the closed-form coefficient

        Σ_{k=2}^{a+1} C(a, k-1) / k  =  (2^{a+1} - 1 - (a+1)) / (a+1)

    via the hockey-stick identity, so the result is a sorted dot product.
    This is the apportioned (``φ_r / |r|``) analogue of
    :func:`sum_of_minimum_among_subsets`.
    """
    n = len(values)
    if n < 2:
        return 0.0
    sorted_values = np.sort(np.asarray(values, dtype=float))
    coefficients = np.zeros(n)
    for i in range(n):
        a = n - 1 - i
        if a > 0:
            coefficients[i] = (2 ** (a + 1) - 1 - (a + 1)) / (a + 1)
    return float(np.sum(sorted_values * coefficients))


def sum_of_ratio_of_minima_among_subsets(
    num_denom_pairs: list[tuple[float, float]],
) -> float:
    """Return the sum of the ratio of minima among numerators/denominators.

    Considers all subsets with size >1 of pairs of numerators and denominators
    (n_i, d_i) and implicitly computes the sum of the ratios of the minimum
    numerator / minimum denominator, where the minimum is taken within each
    subset.

    Arguments:
        num_denom_pairs (list[tuple[float]]): list of pairs of numerators and
        denominators.

    Returns:
        float: Sum of the ratios of minimum numerator to minimum denominator
        over all subsets of size >1.
    """
    numerators, denominators = zip(*num_denom_pairs, strict=False)
    # For each possible pair of values, we count the number of times the pair is
    # the minimal pair (sorting makes the counting easier)
    sorted_num_idx = np.argsort(numerators)
    sorted_denom_idx = np.argsort(denominators)
    sum_ratio = 0
    for i, j in product(range(len(num_denom_pairs)), range(len(num_denom_pairs))):
        # (numerator, denominator) pairs that contain the current candidate
        # values
        candiate_elements = {sorted_num_idx[i], sorted_denom_idx[j]}
        # The set of elements whose numerator >= candidate numerator
        num_superset = set(sorted_num_idx[i:])
        # The set of elements whose denominators >= candidate denominator
        denom_superset = set(sorted_denom_idx[j:])

        superset = num_superset.intersection(denom_superset)
        if not candiate_elements.issubset(superset):
            continue

        # Number of subsets of size >1 of the superset that contain the candiate
        # elements
        num_occurences = 2 ** len(superset - candiate_elements)
        if len(candiate_elements) == 1:
            num_occurences -= 1

        min_num = numerators[sorted_num_idx[i]]
        min_denom = denominators[sorted_denom_idx[j]]
        sum_ratio += num_occurences * min_num / min_denom
    return sum_ratio


def sum_of_min_times_avg_among_subsets(values: list[float]) -> float:
    """Return the sum of the product of the minimum and mean of each subset
    with size >1 of the values."""
    # This series counts, from i = 0 to (len(values) - 1), the number of subsets
    # of values of size >1 such that value i is included in all subsets.
    values.sort()
    _sum = 0
    for i, min_val in enumerate(values[:-1]):
        n = len(values[i:])
        # For each candidate min_val, we add its contibution to the sum of the
        # average \sum_k (1/k) * comb(n-1, k-1), where k is the size of the
        # subsets k = 2, ..., n-1
        _n = n - 1
        sum_avg_val = min_val * ((2 ** (_n + 1) - 1) / (_n + 1) - 1)
        # Contribution of the other elements to the sum of the average
        # \sum_k (1/k) * comb(n-2, k-2), k is the size of the subsets k = 2, ..., n-2
        _n = n - 2
        sum_avg_val += (
            sum(values[i + 1 :]) * (_n * 2 ** (_n + 1) + 1) / (_n**2 + 3 * _n + 2)
        )
        _sum += min_val * sum_avg_val
    return _sum


def only_nonsubsets(sets: Iterable[set]) -> list[set]:
    """Find sets that are not proper subsets of any other set."""
    sets = sorted(map(set, sets), key=len, reverse=True)
    keep: list[set] = []
    for a in sets:
        if all(not a.issubset(b) for b in keep):
            keep.append(a)
    return keep


# From stackoverflow.com/questions/19368375/set-partitions-in-python
def _set_partitions(collection: Sequence[Any]) -> Generator[list[list[Any]]]:
    collection = list(collection)

    # Special cases
    if not collection:
        return

    if len(collection) == 1:
        yield [collection]
        return

    first = collection[0]
    for smaller in set_partitions(collection[1:]):
        for n, subset in enumerate(smaller):
            yield [*smaller[:n], [first, *subset], *smaller[n + 1 :]]
        yield [[first], *smaller]


def set_partitions(
    collection: Sequence[Any], nontrivial: bool = False
) -> Generator[list[list[Any]]] | itertools.islice[list[list[Any]]]:
    """Generate all set partitions of a collection.

    Example:
        >>> list(set_partitions(range(3)))  # doctest: +NORMALIZE_WHITESPACE
        [[[0, 1, 2]],
         [[0], [1, 2]],
         [[0, 1], [2]],
         [[1], [0, 2]],
         [[0], [1], [2]]]
    """
    if nontrivial:
        return itertools.islice(_set_partitions(collection), 1, None)
    return _set_partitions(collection)
