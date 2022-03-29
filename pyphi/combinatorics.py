#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# combinatorics.py

"""Combinatorial functions."""

# TODO(4.0) move relevant functions from utils here

from collections import defaultdict
from itertools import chain
from graphillion import setset
import networkx as nx


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
        exclude += list(range(max_size + 1, 2 ** len(X) + 1))
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


def maximal_independent_sets(graph):
    """Yield the maximal independent sets of the graph.

    Time complexity is exponential in the worst case.
    """
    # Maximal independent sets are cliques in the graph's complement
    return nx.find_cliques(nx.complement(graph))


def sum_min_subset(l):
    """Calculates the sum of minimum of all the possible subsets of size larger than two
    of a set of elements"""
    l.sort()
    sum_min = 0
    for i, val in enumerate(l):
        sum_min += (2 ** (len(l) - i - 1) - 1) * val
    return sum_min
