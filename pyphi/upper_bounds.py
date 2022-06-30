# upper_bounds.py

import scipy

from . import config
from .cache import cache
from .registry import Registry


def number_of_possible_distinctions_of_order(n, k):
    """Return the number of possible distinctions of order k."""
    # Binomial coefficient
    return int(scipy.special.comb(n, k))


def number_of_possible_distinctions(n):
    """Return the number of possible distinctions."""
    return 2 ** n - 1


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


def distinction_sum_phi_upper_bound(n):
    """Return the 'best possible' sum of small phi for distinctions."""
    return distinction_sum_phi_upper_bounds[config.DISTINCTION_SUM_PHI_UPPER_BOUND](n)


@cache(cache={}, maxmem=None)
def number_of_possible_relations_of_order(n, k):
    """Return the number of possible relations with overlap of size k."""
    # Alireza's generalization of Will's theorem
    return int(scipy.special.comb(n, k)) * sum(
        ((-1) ** i * int(scipy.special.comb(n - k, i)) * _f(n, k + i))
        for i in range(n - k + 1)
    )


@cache(cache={}, maxmem=None)
def number_of_possible_relations(n):
    """Return the number of possible relations of all orders."""
    return sum(number_of_possible_relations_of_order(n, k) for k in range(1, n + 1))


def relation_sum_phi_upper_bound(n):
    """Return the 'best possible' sum of small phi for relations, given
    the best possible sum of small phi for distinctions."""
    distinction_sum_phi = distinction_sum_phi_upper_bound(n)
    correction_factor = (distinction_sum_phi / (n * 2 ** (n - 1))) ** 2
    return correction_factor * sum(
        k * number_of_possible_relations_of_order(n, k) for k in range(1, n + 1)
    )


def sum_phi_upper_bound(n):
    """Return the 'best possible' sum of small phi for the system."""
    return distinction_sum_phi_upper_bound(n) + relation_sum_phi_upper_bound(n)
