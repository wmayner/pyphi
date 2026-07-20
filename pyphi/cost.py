"""Analytic workload counting for single-system analyses.

Counts the work a :func:`pyphi.analyze` call would perform — system
partitions swept by the system irreducibility analysis, candidate
mechanisms, connectivity-pruned purview evaluations, and mechanism
partitions per (mechanism, purview) pair — without computing any φ.
Counts are produced by driving the same enumeration machinery the
analysis uses under the active configuration, so the partition schemes,
the connectivity, and the alphabet are all reflected exactly.

All quantities are counts and structural weights. Wall time depends on
the machine and configuration and is never predicted.
"""

from __future__ import annotations

_PARTITION_COUNT_CAP = 6


class _LimitReached(Exception):
    pass


class _Counter:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.spent = 0

    def charge(self, amount: int) -> None:
        self.spent += amount
        if self.spent > self.limit:
            raise _LimitReached


# Partition counts keyed by (system partition scheme name, m). Enumerating
# the partitions of m elements is the same regardless of substrate, so the
# count is memoized across calls at module scope. The default scheme's
# counts are seeded from direct enumeration of ``system_partitions``; the
# seed-verification tests re-enumerate them.
_PARTITION_COUNT_MEMO: dict[tuple[str, int], int] = {
    ("DIRECTED_SET_PARTITION", 1): 1,
    ("DIRECTED_SET_PARTITION", 2): 3,
    ("DIRECTED_SET_PARTITION", 3): 22,
    ("DIRECTED_SET_PARTITION", 4): 150,
    ("DIRECTED_SET_PARTITION", 5): 1_061,
    ("DIRECTED_SET_PARTITION", 6): 7_896,
    ("DIRECTED_SET_PARTITION", 7): 61_888,
    ("DIRECTED_SET_PARTITION", 8): 510_313,
    ("DIRECTED_SET_PARTITION", 9): 4_419_572,
}

# Mechanism-partition counts keyed by (mechanism partition scheme name,
# |mechanism|, |purview|); the count depends only on the two sizes. The
# default scheme's counts are seeded from direct enumeration of
# ``mechanism_partitions``; the seed-verification tests re-enumerate them.
_MECHANISM_PARTITION_COUNT_MEMO: dict[tuple[str, int, int], int] = {
    ("JOINT_PARTITION_ALL", 1, 1): 1,
    ("JOINT_PARTITION_ALL", 1, 2): 1,
    ("JOINT_PARTITION_ALL", 1, 3): 1,
    ("JOINT_PARTITION_ALL", 1, 4): 1,
    ("JOINT_PARTITION_ALL", 1, 5): 1,
    ("JOINT_PARTITION_ALL", 1, 6): 1,
    ("JOINT_PARTITION_ALL", 1, 7): 1,
    ("JOINT_PARTITION_ALL", 2, 1): 4,
    ("JOINT_PARTITION_ALL", 2, 2): 10,
    ("JOINT_PARTITION_ALL", 2, 3): 28,
    ("JOINT_PARTITION_ALL", 2, 4): 82,
    ("JOINT_PARTITION_ALL", 2, 5): 244,
    ("JOINT_PARTITION_ALL", 2, 6): 730,
    ("JOINT_PARTITION_ALL", 2, 7): 2_188,
    ("JOINT_PARTITION_ALL", 3, 1): 14,
    ("JOINT_PARTITION_ALL", 3, 2): 44,
    ("JOINT_PARTITION_ALL", 3, 3): 146,
    ("JOINT_PARTITION_ALL", 3, 4): 500,
    ("JOINT_PARTITION_ALL", 3, 5): 1_754,
    ("JOINT_PARTITION_ALL", 3, 6): 6_284,
    ("JOINT_PARTITION_ALL", 3, 7): 22_946,
    ("JOINT_PARTITION_ALL", 4, 1): 51,
    ("JOINT_PARTITION_ALL", 4, 2): 185,
    ("JOINT_PARTITION_ALL", 4, 3): 699,
    ("JOINT_PARTITION_ALL", 4, 4): 2_729,
    ("JOINT_PARTITION_ALL", 4, 5): 10_971,
    ("JOINT_PARTITION_ALL", 4, 6): 45_305,
    ("JOINT_PARTITION_ALL", 4, 7): 191_739,
    ("JOINT_PARTITION_ALL", 5, 1): 202,
    ("JOINT_PARTITION_ALL", 5, 2): 822,
    ("JOINT_PARTITION_ALL", 5, 3): 3_472,
    ("JOINT_PARTITION_ALL", 5, 4): 15_162,
    ("JOINT_PARTITION_ALL", 5, 5): 68_272,
    ("JOINT_PARTITION_ALL", 5, 6): 316_242,
    ("JOINT_PARTITION_ALL", 5, 7): 1_503_592,
    ("JOINT_PARTITION_ALL", 6, 1): 876,
    ("JOINT_PARTITION_ALL", 6, 2): 3_934,
    ("JOINT_PARTITION_ALL", 6, 3): 18_306,
    ("JOINT_PARTITION_ALL", 6, 4): 88_018,
    ("JOINT_PARTITION_ALL", 6, 5): 436_266,
    ("JOINT_PARTITION_ALL", 6, 6): 2_224_354,
    ("JOINT_PARTITION_ALL", 6, 7): 11_643_066,
    ("JOINT_PARTITION_ALL", 7, 1): 4_139,
    ("JOINT_PARTITION_ALL", 7, 2): 20_267,
    ("JOINT_PARTITION_ALL", 7, 3): 102_671,
    ("JOINT_PARTITION_ALL", 7, 4): 536_867,
    ("JOINT_PARTITION_ALL", 7, 5): 2_891_639,
    ("JOINT_PARTITION_ALL", 7, 6): 16_012_187,
    ("JOINT_PARTITION_ALL", 7, 7): 90_995_711,
}


def _partition_counts(ms) -> dict[int, int]:
    """System-partition counts per unit count, for m up to the cap."""
    from pyphi.conf import config
    from pyphi.partition import system_partitions

    scheme = config.formalism.iit.system_partition_scheme
    counts = {}
    for m in ms:
        if m > _PARTITION_COUNT_CAP:
            continue
        key = (scheme, m)
        count = _PARTITION_COUNT_MEMO.get(key)
        if count is None:
            count = sum(1 for _ in system_partitions(tuple(range(m))))
            _PARTITION_COUNT_MEMO[key] = count
        counts[m] = count
    return counts


def _system_partition_count(m: int, counter: _Counter) -> int:
    """Count the system partitions of ``m`` units under the active scheme.

    A memoized count is free; a fresh enumeration charges the counter one
    unit per partition, so an unmemoized (scheme, size) pair cannot exceed
    the walk's work budget.
    """
    from pyphi.conf import config
    from pyphi.partition import system_partitions

    scheme = config.formalism.iit.system_partition_scheme
    key = (scheme, m)
    count = _PARTITION_COUNT_MEMO.get(key)
    if count is None:
        count = 0
        for _ in system_partitions(tuple(range(m))):
            counter.charge(1)
            count += 1
        _PARTITION_COUNT_MEMO[key] = count
    return count


def _mechanism_partition_count(msize: int, psize: int, counter: _Counter) -> int:
    """Count the mechanism partitions of a (``msize``, ``psize``) pair
    under the active scheme, with the same budget behavior as
    :func:`_system_partition_count`.
    """
    from pyphi.conf import config
    from pyphi.partition import mechanism_partitions

    scheme = config.formalism.iit.mechanism_partition_scheme
    key = (scheme, msize, psize)
    count = _MECHANISM_PARTITION_COUNT_MEMO.get(key)
    if count is None:
        count = 0
        mechanism = tuple(range(msize))
        purview = tuple(range(msize, msize + psize))
        for _ in mechanism_partitions(mechanism, purview):
            counter.charge(1)
            count += 1
        _MECHANISM_PARTITION_COUNT_MEMO[key] = count
    return count
