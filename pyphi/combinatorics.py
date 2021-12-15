#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# combinatorics.py

"""Combinatorial functions."""

# TODO move relevant functions from utils here

from collections import defaultdict
from itertools import chain


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
    pairs = list(
        chain.from_iterable(
            [[frozenset([i, j]) for j in range(i + 1, n)] for i in range(n)]
        )
    )
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
    return list(chain.from_iterable(implicit.values()))