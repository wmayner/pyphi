# combinatorics.py
"""Combinatorial utilities."""

import itertools
from collections import defaultdict
from itertools import chain, product

import numpy as np
from graphillion import setset

from .cache import cache

# TODO(4.0) move relevant functions from utils here


# TODO(docs) finish documenting
def pair_indices(n, m=None, k=0):
    """Return indices of unordered pairs."""
    if m is None:
        m = n
    n, m = sorted([n, m])
    for i in range(n):
        for j in range(i + k, m):
            yield i, j


# TODO(docs) finish documenting
def pairs(seq, k=0):
    """Return unordered pairs of elements from a sequence.

    NOTE: This is *not* the Cartesian product.
    """
    for i, j in pair_indices(len(seq), k=k):
        yield seq[i], seq[j]


def combinations_with_nonempty_intersection_by_order(sets, min_size=0, max_size=None):
    """Return combinations of sets that have nonempty intersection.

    The returned combinations are sets of the indices of the sets in that
    combination, not the sets themselves.

    Arguments:
        sets (Sequence[frozenset]): The sets to consider. Note that they must be
            ``frozensets``.

    Keyword Arguments:
        min_size (int): The minimum size of the combinations to return. Defaults
            to 0.
        max_size (int): The maximum size of the combinations to return. Defaults
            to ``None``, indicating all sizes.

    Returns:
        defaultdict(set): A mapping from combination size to combinations.
    """
    n = len(sets)
    if max_size is None:
        max_size = n
    min_size = max(2, min_size)

    # Begin by finding pairs with nonempty intersection
    pairs = list(map(frozenset, pair_indices(n, k=1)))
    # Store intersections so successive intersections can be computed faster
    intersections = {
        pair: frozenset.intersection(*[sets[i] for i in pair]) for pair in pairs
    }
    combinations = defaultdict(
        set, {2: set(pair for pair in pairs if intersections[pair])}
    )

    # Iteratively find larger combinations of sets with nonempty intersection
    for k in range(2, max_size):
        nonempty_intersection = combinations[k]
        if nonempty_intersection:
            for i in range(n):
                covered = set()
                for combination in nonempty_intersection:
                    if i in combination:
                        covered.add(combination)
                    else:
                        intersection = sets[i] & intersections[combination]
                        if intersection:
                            new_combination = frozenset([i]) | combination
                            intersections[new_combination] = intersection
                            combinations[k + 1].add(new_combination)
                nonempty_intersection = nonempty_intersection - covered
                if not nonempty_intersection:
                    break
        else:
            break

    return {
        size: combs
        for size, combs in combinations.items()
        if (size >= min_size) and combs
    }


def combinations_with_nonempty_intersection(sets, min_size=0, max_size=None):
    """Return combinations of sets that have nonempty intersection.

    Arguments:
        sets (Sequence[frozenset]): The sets to consider. Note that they must be
            ``frozensets``.

    Keyword Arguments:
        min_size (int): The minimum size of the combinations to return. Defaults
            to 0.
        max_size (int): The maximum size of the combinations to return. Defaults
            to ``None``, indicating all sizes.

    Returns:
        list[frozenset]: The combinations.
    """
    implicit = combinations_with_nonempty_intersection_by_order(
        sets, min_size=min_size, max_size=max_size
    )
    return chain.from_iterable(implicit.values())


def powerset_family(X, min_size=1, max_size=None, universe=None):
    """Return the power set of X as a set family.

    NOTE: The universe is assumed to have been set already.
    """
    if universe is None:
        universe = set(setset.universe())

    # This is necessary since `.set_size(0)` doesn't seem to work
    if min_size > 0:
        negation = [[]]
    else:
        negation = []
    P = ~setset(negation)

    for e in universe - set(X):
        P -= P.join(setset([[e]]))

    exclude = list(range(1, min_size))
    if max_size is not None:
        exclude += list(range(max_size + 1, len(X) + 1))
    for k in exclude:
        P -= P.set_size(k)

    return P


def union_powerset_family(sets, min_size=1, max_size=None):
    """Return union of the power set of each set in ``sets``.

    NOTE: The universe must already have been set to (at least) the union of the
    ``sets``.
    """
    U = set(setset.universe())
    S = setset([])
    for s in sets:
        S |= powerset_family(s, min_size=min_size, max_size=max_size, universe=U)
    return S


@cache(cache={}, maxmem=None)
def num_subsets_larger_than_one_element(n):
    """Return the number of subsets on N elements with size >1.

    |X| = |P(n)| - |{S ∈ P(n) | |S| = 1}| - |{S ∈ P(n) | |S| = 0}|
        = 2^n    - (n choose 1)             - |{ø}|
        = 2^n    - n                        - 1
    """
    return 2**n - n - 1


def sum_of_minimum_among_subsets(values):
    """Return the sum of the minimum of all subsets with size >1 of the values."""
    # This series counts, from i = 0 to (len(values) - 1), the number of subsets
    # of values of size >1 such that value i is included in all subsets.
    # Since each value is fixed to be in all subsets, this formula differs from
    # `num_subsets_larger_than_one_element`.
    counts = 2 ** (np.arange(len(values), 0, -1) - 1) - 1
    # Sorting ensures that we're taking the minimum of values for each subset
    return np.sum(np.sort(values) * counts)


def sum_of_ratio_of_minima_among_subsets(num_denom_pairs):
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
    numerators, denominators = zip(*num_denom_pairs)
    # For each possible pair of values, we count the number of times the pair is
    # the minimal pair (sorting makes the counting easier)
    sorted_num_idx = np.argsort(numerators)
    sorted_denom_idx = np.argsort(denominators)
    sum_ratio = 0
    for i, j in product(range(len(num_denom_pairs)), range(len(num_denom_pairs))):
        # (numerator, denominator) pairs that contain the current candidate
        # values
        candiate_elements = set([sorted_num_idx[i], sorted_denom_idx[j]])
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


def sum_of_min_times_avg_among_subsets(values):
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


def only_nonsubsets(sets):
    """Find sets that are not proper subsets of any other set."""
    sets = sorted(map(set, sets), key=len, reverse=True)
    keep = []
    for a in sets:
        if all(not a.issubset(b) for b in keep):
            keep.append(a)
    return keep


# From stackoverflow.com/questions/19368375/set-partitions-in-python
def _set_partitions(collection):
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
            yield smaller[:n] + [[first] + subset] + smaller[n + 1 :]
        yield [[first]] + smaller


def set_partitions(collection, nontrivial=False):
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
