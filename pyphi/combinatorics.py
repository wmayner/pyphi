# combinatorics.py
"""Combinatorial utilities."""

from __future__ import annotations

import itertools
import math
from collections.abc import Generator
from collections.abc import Iterable
from collections.abc import Sequence
from itertools import product
from typing import Any

import numpy as np

# TODO: move relevant functions from utils here


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

    Parameters
    ----------
    sets : Sequence[frozenset]
        The sets to consider. They must be ``frozenset`` instances.

    Other Parameters
    ----------------
    min_size : int
        The minimum size of the combinations to yield (default 0).
    max_size : int or None
        The maximum size of the combinations to yield. If ``None`` (the
        default), there is no upper bound.
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


def num_subsets_larger_than_one_element(n: int) -> int:
    """Return the number of subsets on N elements with size >1.

    ::

        |X| = |P(n)| - |{S ∈ P(n) | |S| = 1}| - |{S ∈ P(n) | |S| = 0}|
            = 2^n    - (n choose 1)             - |{ø}|
            = 2^n    - n                        - 1
    """
    return 2**n - n - 1  # type: ignore[no-any-return]


def saturating_pow2(exponents: Any) -> np.ndarray:
    """Return ``2.0 ** exponents`` in float64, saturating to ``inf``.

    The shared overflow policy for closed-form subset counts, whose ``2^k``
    growth outruns every fixed-width type: powers are exact for integer
    exponents through 1023 (every power of two in float64's range is exactly
    representable) and saturate to ``inf`` beyond — a valid, uninformative
    ceiling — without raising or warning. Callers must mask values that may
    not multiply an infinite count (a zero value contributes zero, not
    ``0 × inf = nan``), and may let an overflowing downstream *sum* saturate
    the same way, since a total beyond float64 range is exactly ``inf``.
    """
    with np.errstate(over="ignore"):
        return 2.0 ** np.asarray(exponents, dtype=float)


def sum_of_minimum_among_subsets(values: Sequence[float] | np.ndarray) -> float:
    """Return the sum of the minimum of all subsets with size >1 of the values.

    Totals beyond float64 range saturate to ``inf`` (see
    :func:`saturating_pow2`).
    """
    # This series counts, from i = 0 to (len(values) - 1), the number of subsets
    # of values of size >1 such that value i is included in all subsets.
    # Since each value is fixed to be in all subsets, this formula differs from
    # `num_subsets_larger_than_one_element`.
    # Sorting ensures that we're taking the minimum of values for each subset.
    exponents = np.arange(len(values), 0, -1) - 1
    sorted_values = np.sort(np.asarray(values, dtype=float))
    counts = saturating_pow2(exponents) - 1.0
    with np.errstate(over="ignore", invalid="ignore"):
        terms = sorted_values * counts
        terms[sorted_values == 0.0] = 0.0  # a zero value contributes 0, not 0*inf=nan
        return float(np.sum(terms))


def sum_of_minimum_over_size_among_subsets(
    values: Sequence[float] | np.ndarray,
) -> float:
    """Return the sum of ``min(S) / |S|`` over all subsets ``S`` with size > 1.

    For values sorted ascending as ``v_0 <= ... <= v_{n-1}``, ``v_i`` is the
    minimum of exactly those subsets containing ``i`` whose other elements all
    come from the ``a = n - 1 - i`` larger positions. Summing ``1/|S|`` over
    those subsets gives the closed-form coefficient

        Σ_{k=2}^{a+1} C(a, k-1) / k  =  (2^{a+1} - 1 - (a+1)) / (a+1)

    via the hockey-stick identity, so the result is a sorted dot product.
    This is the apportioned (``φ_r / |r|``) analogue of
    :func:`sum_of_minimum_among_subsets`. Totals beyond float64 range
    saturate to ``inf`` (see :func:`saturating_pow2`).
    """
    n = len(values)
    if n < 2:
        return 0.0
    sorted_values = np.sort(np.asarray(values, dtype=float))
    a_plus_one = np.arange(n, 0, -1, dtype=float)
    # At a = 0 the closed form is (2^1 - 1 - 1) / 1 = 0: a value with no
    # larger companions is the minimum of no size->1 subset.
    coefficients = (saturating_pow2(a_plus_one) - 1.0 - a_plus_one) / a_plus_one
    with np.errstate(over="ignore", invalid="ignore"):
        terms = sorted_values * coefficients
        terms[sorted_values == 0.0] = 0.0  # a zero value contributes 0, not 0*inf=nan
        return float(np.sum(terms))


def sum_of_minimum_of_size_among_subsets(values: Sequence[float], size: int) -> float:
    """Return the sum of ``min(S)`` over all subsets ``S`` with ``|S| == size``.

    For values sorted ascending, the ``i``-th smallest value is the minimum of
    exactly ``C(n − 1 − i, size − 1)`` subsets of size ``size`` (its
    companions must all come from the larger positions), so the result is a
    sorted dot product with binomial coefficients. This is the
    fixed-degree analogue of :func:`sum_of_minimum_among_subsets`. Counts
    beyond float64 range saturate the total to ``inf`` (the policy of
    :func:`saturating_pow2`).
    """
    if size < 1 or size > len(values):
        return 0.0
    ordered = sorted(values)
    n = len(ordered)

    def term(i: int, value: float) -> float:
        count = math.comb(n - 1 - i, size - 1)
        try:
            return value * count
        except OverflowError:  # count beyond float64 range
            return math.inf * value if value else 0.0

    terms = [term(i, value) for i, value in enumerate(ordered)]
    try:
        return math.fsum(terms)
    except OverflowError:  # finite terms whose exact total exceeds float64
        return sum(terms)  # IEEE addition saturates to the signed infinity


def intersection_closure(sets: Iterable[frozenset]) -> set[frozenset]:
    """Return every nonempty intersection of a nonempty subfamily of ``sets``.

    The closure is computed by repeatedly intersecting the frontier with the
    base family until no new element appears. Its size is bounded by ``2``
    raised to the size of the union of all sets, but is typically far smaller
    for structured families.
    """
    base = [frozenset(s) for s in sets if s]
    closure: set[frozenset] = set()
    frontier = set(base)
    while frontier:
        closure |= frontier
        frontier = {
            intersection
            for p in frontier
            for s in base
            if (intersection := p & s) and intersection not in closure
        }
    return closure


def exact_intersection_counts(sets: Sequence[frozenset]) -> dict[frozenset, int]:
    """Map each intersection-closure element to the number of subfamilies
    whose intersection is exactly that element.

    Subfamilies are index-subsets of ``sets`` of size ≥ 2 (duplicates in
    ``sets`` are distinct members). For a closure element ``P`` with ``m``
    supersets among ``sets``, ``2**m − m − 1`` subfamilies intersect to at
    least ``P``; Möbius inversion down the closure (subtracting the exact
    counts of every strict superset of ``P``) leaves the exact count. Closure
    elements that are never the exact intersection of a size-≥2 subfamily have
    an exact count of zero and are omitted from the result. All counts are
    Python ints.
    """
    closure = sorted(intersection_closure(sets), key=len, reverse=True)
    exact: dict[frozenset, int] = {}
    for p in closure:
        m = sum(1 for s in sets if p <= s)
        exact[p] = (2**m - m - 1) - sum(count for q, count in exact.items() if p < q)
    return {p: count for p, count in exact.items() if count}


def sum_of_ratio_of_minima_among_subsets(
    num_denom_pairs: list[tuple[float, float]],
) -> float:
    """Return the sum of the ratio of minima among numerators/denominators.

    Considers all subsets of size > 1 of pairs of numerators and denominators
    ``(n_i, d_i)`` and computes the sum, over those subsets, of the ratio of
    the minimum numerator to the minimum denominator, where each minimum is
    taken within the subset.

    Parameters
    ----------
    num_denom_pairs : list[tuple[float, float]]
        List of ``(numerator, denominator)`` pairs.

    Returns
    -------
    float
        Sum of the ratios of minimum numerator to minimum denominator over all
        subsets of size > 1.
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

    When ``nontrivial`` is ``True``, the single-block partition (the whole
    collection) is omitted.

    Examples
    --------
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
