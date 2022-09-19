# upper_bounds.py


from itertools import combinations

import numpy as np
import scipy

from . import config
from .cache import cache
from .registry import Registry

##############################################################################
# Distinctions
##############################################################################


def number_of_possible_distinctions_of_order(n, k):
    """Return the number of possible distinctions of order k."""
    # Binomial coefficient
    return int(scipy.special.comb(n, k))


def number_of_possible_distinctions(n):
    """Return the number of possible distinctions."""
    return 2**n - 1


@cache(cache={}, maxmem=None)
def _f(n, k):
    return (2 ** (2 ** (n - k + 1))) - (1 + 2 ** (n - k + 1))


class DistinctionSumPhiUpperBoundRegistry(Registry):
    """Storage for functions for defining the upper bound of the sum of
    distinction phi when analyzing the system.

    NOTE: Functions should ideally return `int`s, if possible, to take advantage
    of the unbounded size of Python integers.
    """

    desc = "distinction sum phi bounds (system)"


distinction_sum_phi_upper_bounds = DistinctionSumPhiUpperBoundRegistry()


@distinction_sum_phi_upper_bounds.register("PURVIEW_SIZE")
def _(n):
    # This can be simplified to (n/2)*(2^n), but we don't use that identity so
    # we can keep things as `int`s
    return sum(
        k * number_of_possible_distinctions_of_order(n, k) for k in range(1, n + 1)
    )


_ = distinction_sum_phi_upper_bounds.register("2^N-1")(number_of_possible_distinctions)


@distinction_sum_phi_upper_bounds.register("(2^N-1)/(N-1)")
def _(n):
    try:
        return number_of_possible_distinctions(n) / (n - 1)
    except ZeroDivisionError:
        return 1


def _generate_all_purview_pairs(nodes, k1, k2):
    """Generates all the cause and effect purview pairs of size ``N - k1`` and
    ``N - k2``.

    Yields:
        tuple[tuple[int]]
    """
    N = len(nodes)
    effect_list = list(combinations(nodes, N - k1))
    cause_list = list(combinations(nodes, N - k2))
    # Iterating in a way to make purview inclusion more uniform
    for j in range(len(cause_list)):
        for i in range(len(effect_list)):
            z_e = effect_list[i]
            z_c = cause_list[(i + j) % len(cause_list)]
            yield (z_e, z_c)


def _add_distinction(purview_inclusion, min_z, z_e, z_c):
    for z in z_e:
        purview_inclusion[z] += 1
    for z in z_c:
        purview_inclusion[z] += 1
    min_z += [min(len(z_e), len(z_c))]
    return purview_inclusion, min_z


def optimal_purview_inclusion(N):
    """Calculate the optimal purview inclusion structure for system size N if
    duplicate purviews are allowed on one side (effect or cause), but two
    distinctions cannot have the same pair of cause and effect purviews.

    Returns:
        tuple[list[int], list[int]]: number of purviews that include each node
        min_z, minimum purview size for each distinction
    """
    nodes = [i for i in range(N)]
    purview_inclusion = [0 for _ in range(N)]
    n_distinctions = 0
    min_z = []
    for s in range(2 * N - 1):
        for k1 in range(s // 2, -1, -1):
            k2 = s - k1
            if k1 >= N or k2 >= N:
                break
            for z_e, z_c in _generate_all_purview_pairs(nodes, k1, k2):
                if n_distinctions >= 2**N - 1:
                    return purview_inclusion, min_z
                purview_inclusion, min_z = _add_distinction(
                    purview_inclusion, min_z, z_e, z_c
                )
                n_distinctions += 1
            if k1 == k2:
                continue
            for z_e, z_c in _generate_all_purview_pairs(nodes, k2, k1):
                if n_distinctions >= 2**N - 1:
                    return purview_inclusion, min_z
                purview_inclusion, min_z = _add_distinction(
                    purview_inclusion, min_z, z_e, z_c
                )
                n_distinctions += 1
    return purview_inclusion, min_z


@distinction_sum_phi_upper_bounds.register("DISTINCT_AND_CONGRUENT_PURVIEWS")
def _(N):
    """Calculates the maximum sum of phi_d if all the distinctions have a
    purview of size N for their effect purview.
    """
    sum_phi_d = 0
    for k in range(1, N + 1):
        t = (N - k + 1) // 2
        ratio = (
            2
            * sum([scipy.special.comb(N - k, b) for b in range(t + 1)])
            / sum([scipy.special.comb(N - k + 1, b) for b in range(t + 1)])
        )
        sum_phi_d += scipy.special.comb(N, k) * N * np.log2(ratio)
    return int(sum_phi_d)


##############################################################################
# Relations
##############################################################################


class RelationSumPhiUpperBoundRegistry(Registry):
    """Storage for functions for defining the upper bound of the sum of
    relation phi when analyzing the system.

    NOTE: Functions should ideally return `int`s, if possible, to take advantage
    of the unbounded size of Python integers.
    """

    desc = "distinction sum phi bounds (system)"


relation_sum_phi_upper_bounds = RelationSumPhiUpperBoundRegistry()


@cache(cache={}, maxmem=None)
def number_of_possible_relations_of_given_order_with_unique_purviews(n, k):
    """Return the number of possible relations with overlap of size k."""
    # Alireza's generalization of Will's theorem
    return int(scipy.special.comb(n, k)) * sum(
        ((-1) ** i * int(scipy.special.comb(n - k, i)) * _f(n, k + i))
        for i in range(n - k + 1)
    )


@cache(cache={}, maxmem=None)
def number_of_possible_relations_with_unique_purviews(n):
    """Return the number of possible relations of all orders."""
    return sum(
        number_of_possible_relations_of_given_order_with_unique_purviews(n, k)
        for k in range(1, n + 1)
    )


@relation_sum_phi_upper_bounds.register("UNIQUE_PURVIEWS")
def _(n):
    """Return the 'best possible' sum of small phi for relations, given
    the best possible sum of small phi for distinctions."""
    distinction_sum_phi = distinction_sum_phi_upper_bound(n)
    correction_factor = (distinction_sum_phi / (n * 2 ** (n - 1))) ** 2
    return correction_factor * sum(
        k * number_of_possible_relations_of_given_order_with_unique_purviews(n, k)
        for k in range(1, n + 1)
    )


@relation_sum_phi_upper_bounds.register("DISTINCT_AND_CONGRUENT_PURVIEWS")
def _(N):
    purview_inclusion, min_z = optimal_purview_inclusion(N)
    sum_phi_d = distinction_sum_phi_upper_bound(N)
    alpha = sum_phi_d / sum(min_z)
    return sum([2**p - p - 1 for p in purview_inclusion]) * alpha**2


##############################################################################
# API
##############################################################################


def distinction_sum_phi_upper_bound(n):
    """Return the 'best possible' sum of small phi for distinctions."""
    return distinction_sum_phi_upper_bounds[config.DISTINCTION_SUM_PHI_UPPER_BOUND](n)


def relation_sum_phi_upper_bound(n):
    return relation_sum_phi_upper_bounds[config.RELATION_SUM_PHI_UPPER_BOUND](n)


def sum_phi_upper_bound(n):
    """Return the 'best possible' sum of small phi for the system."""
    return distinction_sum_phi_upper_bound(n) + relation_sum_phi_upper_bound(n)
