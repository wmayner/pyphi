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
from collections.abc import Callable
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING
from typing import Any

import numpy as np

from pyphi import utils
from pyphi.conf import config
from pyphi.models.partitions import JointPartition
from pyphi.models.partitions import Part

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from pyphi.substrate import Substrate

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


##############################################################################
# Domain guard
##############################################################################

# (version, measure) combinations for which the property-test battery in
# test/test_bounds.py confirms the bounds against the real pipeline.
MECHANISM_MEASURE_DOMAIN = frozenset(
    {
        ("IIT_4_0_2023", "GENERALIZED_INTRINSIC_DIFFERENCE"),
        ("IIT_4_0_2026", "GENERALIZED_INTRINSIC_DIFFERENCE"),
    }
)
SYSTEM_MEASURE_DOMAIN = frozenset(
    {
        ("IIT_4_0_2023", "GENERALIZED_INTRINSIC_DIFFERENCE"),
        ("IIT_4_0_2026", "GENERALIZED_INTRINSIC_DIFFERENCE"),
        ("IIT_4_0_2026", "INTRINSIC_INFORMATION"),
    }
)
# The n(n - 1) system bound counts connections severed by set partitions,
# which never cut self-connections; schemes that sever them break it.
SYSTEM_PARTITION_SCHEME_DOMAIN = frozenset({"DIRECTED_SET_PARTITION"})


def _require_valid_domain() -> None:
    """Raise unless the active config is in the confirmed mechanism-level domain."""
    version = config.formalism.iit.version
    measure = config.formalism.iit.mechanism_phi_measure
    if (version, measure) not in MECHANISM_MEASURE_DOMAIN:
        raise ValueError(
            f"the mechanism-level bounds are not confirmed for "
            f"(version={version!r}, mechanism_phi_measure={measure!r}); "
            f"confirmed combinations: {sorted(MECHANISM_MEASURE_DOMAIN)}. "
            f"See {CITATION}"
        )


def _require_valid_system_domain() -> None:
    """Raise unless the active config is in the confirmed system-level domain."""
    version = config.formalism.iit.version
    measure = config.formalism.iit.system_phi_measure
    if (version, measure) not in SYSTEM_MEASURE_DOMAIN:
        raise ValueError(
            f"the system-level bound is not confirmed for "
            f"(version={version!r}, system_phi_measure={measure!r}); "
            f"confirmed combinations: {sorted(SYSTEM_MEASURE_DOMAIN)}. "
            f"See {CITATION}"
        )
    scheme = config.formalism.iit.system_partition_scheme
    if scheme not in SYSTEM_PARTITION_SCHEME_DOMAIN:
        raise ValueError(
            f"the system-level bound assumes a partition scheme that does "
            f"not sever self-connections; got system partition scheme "
            f"{scheme!r}, confirmed: {sorted(SYSTEM_PARTITION_SCHEME_DOMAIN)}. "
            f"See {CITATION}"
        )


##############################################################################
# Per-object bounds
##############################################################################


def distinction_phi_upper_bound(
    mechanism: Iterable[int], purview: Iterable[int]
) -> UpperBound:
    """Upper bound on phi of a mechanism over a candidate purview.

    Theorem 1: phi(m, Z) <= |M| |Z|, the number of potential causal
    connections between the mechanism and the purview. Only the sizes
    matter.
    """
    _require_valid_domain()
    num_mechanism = len(tuple(mechanism))
    num_purview = len(tuple(purview))
    if num_mechanism < 1 or num_purview < 1:
        raise ValueError("mechanism and purview must be nonempty")
    return UpperBound(
        value=num_mechanism * num_purview,
        certified=True,
        assumptions=_CORE_ASSUMPTIONS,
        citation="Theorem 1",
    )


def partition_phi_upper_bound(partition: Any) -> UpperBound:
    """Upper bound on phi of a mechanism-purview pair under a given partition.

    Lemma 2: phi(m, Z given theta) <= N(theta), the number of connections
    severed by the partition. Holds for any partitioning, valid or not.

    Args:
        partition: Any partition exposing ``num_connections_cut()``
            (e.g. :class:`~pyphi.models.partitions.JointPartition`).
    """
    _require_valid_domain()
    return UpperBound(
        value=partition.num_connections_cut(),
        certified=True,
        assumptions=_CORE_ASSUMPTIONS,
        citation="Lemma 2",
    )


def relation_phi_upper_bound(relata_phis: Iterable[float]) -> UpperBound:
    """Upper bound on phi of a relation, given its relata's distinction phis.

    phi_r(d) <= min over relata of phi_d (Sec 2.2): the relation overlap
    is contained in every relatum's purview union.
    """
    _require_valid_domain()
    phis = tuple(float(phi) for phi in relata_phis)
    if not phis:
        raise ValueError("relata_phis must be nonempty")
    return UpperBound(
        value=min(phis),
        certified=True,
        assumptions=_CORE_ASSUMPTIONS,
        citation="Sec 2.2",
    )


def system_phi_upper_bound(n: int) -> UpperBound:
    """Upper bound on system integrated information for n units.

    phi_s <= n(n - 1) (Table 2, citing Marshall et al. 2023): system phi
    is bounded by the number of connections cut by the selected
    partition, and set partitions sever at most n(n - 1) connections
    (all between-part connections at the atomic partition;
    self-connections are never cut).
    """
    _require_valid_system_domain()
    _require_positive(n)
    return UpperBound(
        value=n * (n - 1),
        certified=True,
        assumptions=(
            *_CORE_ASSUMPTIONS,
            "system partitions do not sever self-connections",
        ),
        citation="Table 2",
    )


##############################################################################
# Sum of distinction phi
##############################################################################

_CONJECTURE_NOTE = (
    "conjectured: proven for reflexive selectivity-1 systems; "
    "generality is an open question in the paper"
)


def _log2_pi(n: int, k: int, a: int) -> float:
    """log2 of the partitioned single-unit effect probability pi(a).

    In the size-n, order-k high-selectivity construction, when the
    mechanism part connected to a purview unit has size a and contains
    that unit (S3 Appendix, Sec 3).
    """
    numerator = sum(math.comb(n - a, b) for b in range(k - a, n - a + 1))
    return math.log2(numerator) - (n - a)


def _log2_pi_bar(n: int, k: int, a: int) -> float:
    """log2 of pi-bar(a): as :func:`_log2_pi`, but the connected part does
    not contain the purview unit. Covers a = 0 (fully severed unit)."""
    numerator = sum(math.comb(n - a - 1, b - 1) for b in range(k - a, n - a + 1))
    return math.log2(numerator) - (n - a)


def _phi_e_star(n: int, k: int) -> float:
    """Integrated effect information of a size-k mechanism over itself in
    the high-selectivity construction (S3 Appendix, Sec 3).

    Selectivity is 1, so phi at a partition equals the informativeness
    lost. The MIP is among k // 2 + 1 candidates: the bipartitions that
    keep self-connections intact, and the cut severing one mechanism unit
    from all purview units (the only candidate for k = 1). Candidates are
    compared by value normalized by the number of connections severed;
    the returned phi is unnormalized. For k = n only the complete
    partition exists and phi equals n ** 2.
    """
    if k == n:
        return float(n * n)
    candidates: list[tuple[float, int]] = []
    for j in range(1, k // 2 + 1):
        value = -(j * _log2_pi(n, k, j) + (k - j) * _log2_pi(n, k, k - j))
        candidates.append((value, 2 * j * (k - j)))
    value = -(_log2_pi_bar(n, k, k - 1) + (k - 1) * _log2_pi(n, k, k - 1))
    candidates.append((value, k))
    return min(candidates, key=lambda c: (c[0] / c[1], c[0]))[0]


def sum_phi_distinctions_upper_bound(n: int, bound: str = "I") -> UpperBound:
    """Upper bound on the sum of distinction phi for a system of n units.

    Bounds:
        ``"I"`` (Eq 6, certified, not achievable): every mechanism at
            phi = |M| n; equals (n**2 / 2) 2**n.
        ``"II"`` (Eq 7, conditional): assumes each purview is assigned to
            exactly one mechanism with matching sizes; equals
            (n (n+1) / 4) 2**n.
        ``"III"`` (Sec 2.1.3, conjectured): the numerical bound from the
            high-selectivity reflexive construction,
            sum over K of C(n, K) phi*_e(K).
    """
    _require_valid_domain()
    _require_positive(n)
    if bound == "I":
        return UpperBound(
            value=sum(k * n * math.comb(n, k) for k in range(1, n + 1)),
            certified=True,
            assumptions=_CORE_ASSUMPTIONS,
            citation="Eq 6",
        )
    if bound == "II":
        return UpperBound(
            value=sum(k * k * math.comb(n, k) for k in range(1, n + 1)),
            certified=False,
            assumptions=(
                *_CORE_ASSUMPTIONS,
                "unique purviews: each purview assigned to exactly one mechanism",
            ),
            citation="Eq 7",
        )
    if bound == "III":
        return UpperBound(
            value=sum(math.comb(n, k) * _phi_e_star(n, k) for k in range(1, n + 1)),
            certified=False,
            assumptions=(*_CORE_ASSUMPTIONS, _CONJECTURE_NOTE),
            citation="Sec 2.1.3",
        )
    raise ValueError(f"unknown bound id {bound!r}; expected 'I', 'II', or 'III'")


##############################################################################
# Sum of relation phi
##############################################################################


def _grouped_subset_min_sum(groups: list[tuple[float, int]]) -> float:
    """Sum, over all subsets of size >= 2 of a multiset of ratios, of the
    subset's minimum (the inner sum of Eq 11).

    The i-th smallest of R elements (1-based) is the minimum of
    2**(R - i) - 1 subsets. Equal-ratio groups are summed as geometric
    series: a group of multiplicity m with ``after`` elements above it
    has total weight 2**after (2**m - 1) - m. The computation is exact
    (arbitrary-precision int) when the ratios are ints.

    Args:
        groups: ``(ratio, multiplicity)`` pairs; order irrelevant.
    """
    groups = sorted(groups)
    total_count = sum(multiplicity for _, multiplicity in groups)
    result = 0
    position = 0  # number of elements strictly below the current group
    for ratio, multiplicity in groups:
        after = total_count - position - multiplicity
        weight = 2**after * (2**multiplicity - 1) - multiplicity
        result += ratio * weight
        position += multiplicity
    return result


def _relation_profile(
    n: int, bound: str
) -> tuple[list[tuple[float, int]], float, tuple[str, ...]]:
    """Per-unit (ratio, multiplicity) groups, self-relation term, and
    extra assumptions for a sum-of-relation-phi scenario.

    The profiles realize the corresponding sum-of-distinction-phi
    scenarios (Table 3):

    - ``"I"``: every purview is the whole system in a congruent maximal
      state, so every distinction relates over every unit; ratio |M|.
    - ``"II"``: every purview is the mechanism itself; a unit relates the
      mechanisms containing it; ratio |M|.
    - ``"III"``: the high-selectivity construction profile as implemented
      in the paper's published experiment code (ratio phi*_K / K over all
      distinctions). The paper text instead assumes cause purviews span
      the system (ratio phi*_K / n); the implemented profile dominates
      both readings.
    """
    if bound == "I":
        groups: list[tuple[float, int]] = [(k, math.comb(n, k)) for k in range(1, n + 1)]
        self_term: float = sum(k * n * math.comb(n, k) for k in range(1, n + 1))
        extra = ("Bound I extremal purview profile (all purviews span the system)",)
    elif bound == "II":
        groups = [(k, math.comb(n - 1, k - 1)) for k in range(1, n + 1)]
        self_term = sum(k * k * math.comb(n, k) for k in range(1, n + 1))
        extra = (
            "unique purviews: each purview assigned to exactly one mechanism",
            "Bound II extremal purview profile (every purview is its mechanism)",
        )
    elif bound == "III":
        phi_star = {k: _phi_e_star(n, k) for k in range(1, n + 1)}
        groups = [(phi_star[k] / k, math.comb(n, k)) for k in range(1, n + 1)]
        self_term = sum(math.comb(n, k) * phi_star[k] for k in range(1, n + 1))
        extra = (
            _CONJECTURE_NOTE,
            "Bound III extremal purview profile (high-selectivity construction)",
        )
    else:
        raise ValueError(
            f"unknown bound id {bound!r}; expected 'I', 'II', 'III', or 'GENERAL'"
        )
    return groups, self_term, extra


def sum_phi_relations_upper_bound(n: int, bound: str = "I") -> UpperBound:
    """Upper bound on the sum of relation phi (self-relations included).

    For ``bound`` in ``"I"``, ``"II"``, ``"III"``: the exact Eq 11
    evaluation of the corresponding extremal purview profile (the Table 3
    closed forms plus the self-relation term). These are scenario
    bounds: they assume the system's distinction profile matches the
    scenario, so they are not certified for arbitrary systems.

    For ``bound="GENERAL"``: the certified growth bound of Eq 16, built
    from S(o) <= n 2**(n-1) (Theorem 1) and |Z(o)| <= 2**n - 1 via the
    Eq 14 linear-program maximum, summed over all 2n unit-states, plus
    the Eq 6 ceiling on self-relations.
    """
    _require_valid_domain()
    _require_positive(n)
    if bound == "GENERAL":
        budget = Fraction(n * 2**n, 2)  # S(o) <= n 2**(n-1)
        num_relata = 2**n - 1  # |Z(o)| <= number of distinctions
        per_unit_state = budget * (Fraction(2**num_relata - 1, num_relata) - 1)
        exact = (
            Fraction(sum(k * n * math.comb(n, k) for k in range(1, n + 1)))
            + 2 * n * per_unit_state
        )
        value = int(exact) if exact.denominator == 1 else float(exact)
        return UpperBound(
            value=value,
            certified=True,
            assumptions=_CORE_ASSUMPTIONS,
            citation="Eq 16",
        )
    groups, self_term, extra = _relation_profile(n, bound)
    value = self_term + n * _grouped_subset_min_sum(groups)
    return UpperBound(
        value=value,
        certified=False,
        assumptions=(*_CORE_ASSUMPTIONS, *extra),
        citation="Eqs 11-15, Table 3",
    )


def big_phi_upper_bound(n: int, bound: str = "I") -> UpperBound:
    """Upper bound on big phi: the sum of all distinction and relation phi.

    For ``bound`` in ``"I"``, ``"II"``, ``"III"``: the profile-consistent
    pair of sum bounds. For ``bound="GENERAL"``: the certified pair
    (Eq 6 + Eq 16).
    """
    distinctions = sum_phi_distinctions_upper_bound(
        n, bound="I" if bound == "GENERAL" else bound
    )
    relations = sum_phi_relations_upper_bound(n, bound=bound)
    assumptions = tuple(
        dict.fromkeys((*distinctions.assumptions, *relations.assumptions))
    )
    return UpperBound(
        value=distinctions.value + relations.value,
        certified=distinctions.certified and relations.certified,
        assumptions=assumptions,
        citation=f"{distinctions.citation} + {relations.citation}",
    )


##############################################################################
# High-selectivity construction (S3 Appendix, Sec 3)
##############################################################################


def _construction_tpm(n: int, k: int) -> NDArray[np.float64]:
    """State-by-node TPM of the size-n, order-k high-selectivity construction.

    Unit u turns OFF with probability 1 in exactly the states where u is
    OFF and at least k - 1 other units are OFF (S3 Appendix Eqs 18, 20);
    otherwise it turns ON with probability 1. In this TPM every size-k
    mechanism specifies itself (all-OFF) with probability 1. Rows are in
    little-endian state order.
    """
    _require_positive(n)
    if not 1 <= k <= n:
        raise ValueError(f"order must satisfy 1 <= k <= {n}; got {k}")
    rows = []
    for state in utils.all_states(n):
        row = []
        for unit in range(n):
            zeros_elsewhere = sum(
                1 for other in range(n) if other != unit and state[other] == 0
            )
            specifies_off = state[unit] == 0 and zeros_elsewhere >= k - 1
            row.append(0.0 if specifies_off else 1.0)
        rows.append(row)
    return np.array(rows)


def _candidate_partitions(n: int, k: int):
    """Yield the k // 2 + 1 candidate MIPs for a size-k mechanism over
    itself in the size-n high-selectivity construction (S3 Appendix, Sec 3).

    For k < n: the non-self-cutting bipartitions with part sizes
    (j, k - j), then the cut severing mechanism unit 0 from all purview
    units. For k = n: only the complete partition.
    """
    mechanism = tuple(range(k))
    if k == n:
        yield JointPartition(Part(mechanism, ()), Part((), mechanism))
        return
    for j in range(1, k // 2 + 1):
        first = tuple(range(j))
        rest = tuple(range(j, k))
        yield JointPartition(Part(first, first), Part(rest, rest))
    yield JointPartition(Part(tuple(range(1, k)), mechanism), Part((0,), ()))


##############################################################################
# Report
##############################################################################


def report(n: int | None = None, substrate: Substrate | None = None) -> dict[str, Any]:
    """All size-based bounds for a system of n binary units, in one call.

    Args:
        n: Number of binary units. Mutually exclusive with ``substrate``.
        substrate: A substrate whose size is used; its alphabet must be
            binary.

    Returns:
        Mapping from flat keys (e.g. ``"sum_phi_distinctions:I"``,
        ``"big_phi:GENERAL"``) to :class:`UpperBound` values, plus the
        ``int``-valued counting entries.
    """
    if (n is None) == (substrate is None):
        raise ValueError("provide exactly one of n or substrate")
    if substrate is not None:
        alphabet_sizes = substrate.factored_tpm.alphabet_sizes
        if not all(size == 2 for size in alphabet_sizes):
            raise ValueError(
                f"bounds assume binary units; alphabet sizes are {alphabet_sizes}"
            )
        n = substrate.size
    assert n is not None
    _require_positive(n)
    _require_valid_domain()
    _require_valid_system_domain()
    result: dict[str, Any] = {"system_phi": system_phi_upper_bound(n)}
    for bound_id in ("I", "II", "III"):
        result[f"sum_phi_distinctions:{bound_id}"] = sum_phi_distinctions_upper_bound(
            n, bound=bound_id
        )
        result[f"sum_phi_relations:{bound_id}"] = sum_phi_relations_upper_bound(
            n, bound=bound_id
        )
        result[f"big_phi:{bound_id}"] = big_phi_upper_bound(n, bound=bound_id)
    result["sum_phi_relations:GENERAL"] = sum_phi_relations_upper_bound(
        n, bound="GENERAL"
    )
    result["big_phi:GENERAL"] = big_phi_upper_bound(n, bound="GENERAL")
    result["number_of_possible_distinctions"] = number_of_possible_distinctions(n)
    result["number_of_possible_relations"] = number_of_possible_relations(n)
    result["number_of_possible_relation_faces_with_unique_purviews"] = (
        number_of_possible_relation_faces_with_unique_purviews(n)
    )
    return result


##############################################################################
# Runtime bound-certificate assertions (B1)
##############################################################################


class BoundViolationError(AssertionError):
    """A computed phi exceeded its theorem-certified upper bound.

    Within the certified domain the bound holds for every system, so an
    overshoot is a *proof* of a formalism bug, not a numerical artifact.
    Raised only when ``config.infrastructure.validate_phi_bounds`` is set and
    the system is in the certified domain (IIT 4.0 + GID/II, binary units,
    and — for the system bound — a set-partition scheme). Subclasses
    :class:`AssertionError` so it reads as a violated invariant.
    """

    def __init__(self, label: str, value: float, bound: UpperBound) -> None:
        super().__init__(
            f"{label}: phi={value!r} exceeds the certified upper bound "
            f"{bound.value!r} ({bound.citation}); this is a proof of a "
            f"formalism bug under assumptions {bound.assumptions}."
        )
        self.label = label
        self.value = value
        self.bound = bound


def _bounds_apply_to(system: Any) -> bool:
    """True iff ``system`` is in the bounds' certified *substrate* domain: a
    micro substrate of binary units.

    The bound functions verify the version/measure/scheme config but not the
    substrate, so this gate covers the two substrate assumptions they leave
    implicit:

    - **Binary units.** k-ary φ can legitimately exceed the binary ``|M||Z|``
      and ``n(n-1)`` ceilings.
    - **Micro (not macro).** A :class:`~pyphi.macro.system.MacroSystem` is a
      coarse-graining whose φ_s comes from its micro constituents; a single
      macro unit can have φ_s > 0 while the macro-unit count gives
      ``n(n-1) = 0``. The Zaeemzadeh bounds are stated over the micro
      substrate, so macro systems are out of domain.
    """
    try:
        from pyphi.macro.system import MacroSystem

        if isinstance(system, MacroSystem):
            return False
    except ImportError:
        pass
    try:
        sizes = system.substrate.factored_tpm.alphabet_sizes
    except AttributeError:
        return False
    return all(int(size) == 2 for size in sizes)


def check_phi_bound(
    value: float,
    bound: Callable[[], UpperBound],
    *,
    system: Any,
    label: str,
) -> None:
    """Assert a computed ``value`` does not exceed its certified upper bound.

    A no-op unless ``config.infrastructure.validate_phi_bounds`` is set. Then,
    only inside the certified domain — a binary ``system`` and a config in the
    bound's confirmed version/measure/scheme set — raise
    :class:`BoundViolationError` when ``value`` exceeds the certified ceiling
    by more than the active numerical precision. Outside the certified domain
    (IIT 3.0, non-GID, k-ary, degenerate partitions) it returns silently, so
    there are no false positives.

    ``bound`` is a thunk: the domain gate raises ``ValueError`` out of domain,
    so it is evaluated lazily and caught here.
    """
    if not config.infrastructure.validate_phi_bounds:
        return
    if not _bounds_apply_to(system):
        return
    try:
        certificate = bound()
    except (ValueError, AttributeError):
        # Out of the certified domain, or the bound's structural inputs (e.g.
        # a partition's severed-connection count) are unavailable — no
        # certified ceiling applies, so skip.
        return
    if not certificate.certified:
        return
    tolerance = 10.0 ** -config.numerics.precision
    if float(value) > float(certificate.value) + tolerance:
        raise BoundViolationError(label, float(value), certificate)
