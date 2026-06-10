"""Upper bounds for IIT 4.0 quantities.

Implements the published upper bounds from:

    Zaeemzadeh A, Tononi G. (2024). Upper bounds for integrated
    information. PLOS Computational Biology 20(8): e1012323.
    https://doi.org/10.1371/journal.pcbi.1012323

All bounds assume binary units and a conditionally independent TPM (the
system is realizable as a product of unit TPMs) and are derived for the
IIT 4.0 intrinsic-difference family of measures. Each bound function
returns an :class:`UpperBound` carrying the value together with its
certificate: ``certified=True`` means the bound is theorem-backed for
arbitrary systems satisfying its ``assumptions``; ``certified=False``
means it additionally relies on a scenario assumption or an open
conjecture, recorded in ``assumptions``.

The mechanism-level bounds are ceilings over states and hold at any
selected mechanism partition, so they are insensitive to the configured
partition scheme and MIP normalization. The system-level bound
additionally assumes a system partition scheme that does not sever
self-connections.

Functions taking only ``n`` are pure combinatorics of the binary
formalism; the binary-units assumption is the caller's responsibility
there. :func:`report` validates it when given a
:class:`~pyphi.substrate.Substrate`.

Relation-level sums grow like ``2**(2**n)``. They are computed as exact
Python ints where possible; values that are not integral are returned as
floats and overflow for ``n`` greater than about 10.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

CITATION = (
    "Zaeemzadeh A, Tononi G. (2024). Upper bounds for integrated "
    "information. PLOS Comput Biol 20(8): e1012323."
)

_CORE_ASSUMPTIONS = ("binary units", "conditionally independent TPM")


@dataclass(frozen=True)
class UpperBound:
    """An upper bound on an IIT quantity, with its certificate.

    Attributes:
        value: The bound. An exact ``int`` when the quantity is integral.
        certified: Whether the bound is theorem-backed for arbitrary
            systems satisfying ``assumptions``. Non-certified bounds rely
            on a scenario assumption or an open conjecture and are not
            valid pruning certificates.
        assumptions: The assumptions under which the bound holds.
        citation: Locus in Zaeemzadeh & Tononi (2024), e.g. ``"Eq 6"``.
    """

    value: float
    certified: bool
    assumptions: tuple[str, ...]
    citation: str

    def __float__(self) -> float:
        return float(self.value)


def _require_positive(n: int) -> None:
    if n < 1:
        raise ValueError(f"n must be a positive integer; got {n!r}")


##############################################################################
# Counting (pure combinatorics; no measure or version dependence)
##############################################################################


def number_of_possible_distinctions(n: int) -> int:
    """Number of candidate distinctions in a system of n units.

    One per nonempty mechanism: 2**n - 1.
    """
    _require_positive(n)
    return 2**n - 1


def number_of_possible_distinctions_of_order(n: int, k: int) -> int:
    """Number of candidate distinctions with mechanism size k."""
    _require_positive(n)
    if not 1 <= k <= n:
        raise ValueError(f"order must satisfy 1 <= k <= {n}; got {k}")
    return math.comb(n, k)


def number_of_possible_relations(n: int) -> int:
    """Number of candidate relations in a system of n units.

    One per nonempty subset of the candidate distinctions (Sec 2.2):
    2**(2**n - 1) - 1.
    """
    _require_positive(n)
    return 2 ** (2**n - 1) - 1


def _f(n: int, j: int) -> int:
    """Size->=2 subsets of the purview slots containing a fixed j-unit set.

    In the unique-purview scenario there are 2**(n - j) purviews
    containing the fixed set, each contributing a cause slot and an
    effect slot: 2**(n - j + 1) slots in total.
    """
    slots = 2 ** (n - j + 1)
    return 2**slots - 1 - slots


def number_of_possible_relation_faces_with_unique_purviews_of_order(
    n: int, k: int
) -> int:
    """Number of candidate relation faces whose overlap has exactly k units.

    Counts subsets of size at least 2 of the 2(2**n - 1) cause/effect
    purview slots whose purviews intersect in exactly k units, in the
    scenario where every nonempty subset of units appears as exactly one
    cause and one effect purview. Computed by inclusion-exclusion over
    the overlap set.
    """
    _require_positive(n)
    if not 1 <= k <= n:
        raise ValueError(f"order must satisfy 1 <= k <= {n}; got {k}")
    return math.comb(n, k) * sum(
        (-1) ** i * math.comb(n - k, i) * _f(n, k + i) for i in range(n - k + 1)
    )


def number_of_possible_relation_faces_with_unique_purviews(n: int) -> int:
    """Number of candidate relation faces in the unique-purview scenario."""
    _require_positive(n)
    return sum(
        number_of_possible_relation_faces_with_unique_purviews_of_order(n, k)
        for k in range(1, n + 1)
    )
