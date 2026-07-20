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

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any

from pyphi.display import Description
from pyphi.display import Displayable
from pyphi.display import Row
from pyphi.display import Section
from pyphi.models.pandas import ToPandasMixin

if TYPE_CHECKING:
    from pyphi.substrate import Substrate

__all__ = ["AnalysisEstimate", "estimate_analysis"]

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


def _fmt(value: int) -> str:
    """Format a count for display; huge counts as a power of ten."""
    if value < 10**15:
        return f"{value:,}"
    return f"~10^{len(str(value)) - 1}"


@dataclass(frozen=True)
class AnalysisEstimate(Displayable, ToPandasMixin):
    """The workload of a single-system analysis, before running it.

    Work axes are counted by driving the analysis's own enumeration
    machinery under the active configuration. ``None`` marks an axis
    outside the estimate's scope: excluded by ``compute``, not applicable
    under the active formalism, or not reached before the work budget
    (``capped=True``).

    Attributes
    ----------
    n_units : int
        Number of units in the candidate system.
    state_space_size : int
        Product of the candidate units' alphabet sizes — the scale of one
        repertoire evaluation. Reported as a weight, never multiplied into
        the counts.
    compute : str
        ``"full"``, ``"sia"``, or ``"ces"``.
    system_partitions : int or None
        Partitions the system irreducibility analysis sweeps, under the
        active system partition scheme.
    mechanisms : int or None
        Candidate mechanisms: 2ⁿ − 1 for n units.
    purview_evaluations : int or None
        Connectivity-pruned (mechanism, direction, purview) triples — the
        repertoire-computation axis.
    mechanism_partition_sweeps : int or None
        Mechanism partitions summed over all counted triples, under the
        active mechanism partition scheme — the dominant cost of unfolding
        a cause-effect structure.
    relations_closed_form : bool or None
        Whether the active relation backend computes relations in closed
        form (``ANALYTICAL``) rather than by enumeration (``CONCRETE``).
        ``None`` when relations are outside the estimate's scope.
    possible_distinctions : int or None
        Candidate distinctions (2ⁿ − 1) — the size ceiling of the
        cause-effect structure. Present only under an IIT 4.0 formalism
        with binary units.
    possible_relations : int or None
        Candidate relations (2^(2ⁿ−1) − 1) — the size ceiling of the
        relation set, and the enumeration worst case when
        ``relations_closed_form`` is ``False``. Present only under an
        IIT 4.0 formalism with binary units.
    capped : bool
        The counting walk hit its work budget; walked counts are lower
        bounds (rendered with a ``≥`` qualifier) and axes never reached
        are ``None``.
    """

    n_units: int
    state_space_size: int
    compute: str
    system_partitions: int | None
    mechanisms: int | None
    purview_evaluations: int | None
    mechanism_partition_sweeps: int | None
    relations_closed_form: bool | None
    possible_distinctions: int | None
    possible_relations: int | None
    capped: bool

    def _qualifier(self) -> str:
        return "≥" if self.capped else "="

    def _pandas_record(self) -> dict:
        return {
            "n_units": self.n_units,
            "state_space_size": self.state_space_size,
            "compute": self.compute,
            "system_partitions": self.system_partitions,
            "mechanisms": self.mechanisms,
            "purview_evaluations": self.purview_evaluations,
            "mechanism_partition_sweeps": self.mechanism_partition_sweeps,
            "relations_closed_form": self.relations_closed_form,
            "possible_distinctions": self.possible_distinctions,
            "possible_relations": self.possible_relations,
            "capped": self.capped,
        }

    def _describe(self, verbosity: int) -> Description:  # noqa: ARG002
        q = self._qualifier()
        rows = [
            Row("Units", str(self.n_units)),
            Row("State space", _fmt(self.state_space_size)),
        ]
        if self.system_partitions is not None:
            rows.append(Row("System partitions", f"{q} {_fmt(self.system_partitions)}"))
        if self.mechanisms is not None:
            rows.append(Row("Mechanisms", _fmt(self.mechanisms)))
        if self.purview_evaluations is not None:
            rows.append(
                Row("Purview evaluations", f"{q} {_fmt(self.purview_evaluations)}")
            )
        if self.mechanism_partition_sweeps is not None:
            rows.append(
                Row(
                    "Mechanism partition sweeps",
                    f"{q} {_fmt(self.mechanism_partition_sweeps)}",
                )
            )
        if self.relations_closed_form is not None:
            rows.append(
                Row(
                    "Relations",
                    "closed form" if self.relations_closed_form else "enumerated",
                )
            )
        if self.possible_distinctions is not None:
            rows.append(Row("Possible distinctions", _fmt(self.possible_distinctions)))
        if self.possible_relations is not None:
            rows.append(Row("Possible relations", _fmt(self.possible_relations)))
        rows.append(Row("Capped", self.capped))
        return Description(
            title="AnalysisEstimate",
            subtitle=f"{self.n_units} units, {self.compute}",
            sections=(Section(rows=tuple(rows)),),
            compact=(
                f"AnalysisEstimate(n_units={self.n_units}, compute={self.compute!r})"
            ),
        )


def estimate_analysis(
    substrate: Substrate,
    subset: Any = None,
    compute: str | None = None,
    limit: int = 1_000_000,
) -> AnalysisEstimate:
    """Count the workload of a single-system analysis, without running it.

    Drives the same enumeration machinery :func:`pyphi.analyze` would use
    under the active configuration: the system partition scheme, the
    connectivity-pruned purview sets, and the mechanism partition scheme.
    No φ is computed and no state is needed — every counted quantity is
    state-independent.

    Parameters
    ----------
    substrate : Substrate
        The substrate to analyze.
    subset : optional
        Node indices (or labels) of the candidate system; ``None`` uses
        the whole substrate.
    compute : str or None, optional
        ``None`` estimates the full analysis; ``"sia"`` only the
        system-partition axis; ``"ces"`` only the distinction axis.
    limit : int, optional
        Work budget for the counting walk itself: purview evaluations and
        fresh partition enumerations each cost one unit, while memoized
        partition counts are free. A walk that exceeds the budget stops
        immediately and reports ``capped=True``.

    Returns
    -------
    AnalysisEstimate
        The counted workload.

    Raises
    ------
    ValueError
        If ``compute`` is not ``"sia"``, ``"ces"``, or ``None``.

    Examples
    --------
    >>> from pyphi import examples
    >>> est = estimate_analysis(examples.basic_substrate())
    >>> est.mechanisms
    7
    >>> est.system_partitions
    22
    """
    if compute not in (None, "sia", "ces"):
        raise ValueError(
            f"unknown compute: {compute!r}; expected 'sia', 'ces', or None "
            "for the full analysis"
        )
    from pyphi import utils
    from pyphi.conf import config
    from pyphi.direction import Direction
    from pyphi.system import System

    cs = System.from_substrate(substrate, (0,) * substrate.size, subset)
    indices = cs.node_indices
    m = len(indices)
    alphabet = substrate.factored_tpm.alphabet_sizes
    state_space_size = 1
    for i in indices:
        state_space_size *= int(alphabet[i])
    scope = "full" if compute is None else compute

    counter = _Counter(limit)
    capped = False
    system_partition_count = None
    mechanisms = None
    purview_evaluations = None
    sweeps = None
    try:
        if scope in ("full", "sia"):
            system_partition_count = _system_partition_count(m, counter)
        if scope in ("full", "ces"):
            mechanisms = 2**m - 1
            purview_evaluations = 0
            sweeps = 0
            for mechanism in utils.powerset(indices, nonempty=True):
                for direction in (Direction.CAUSE, Direction.EFFECT):
                    for purview in cs.potential_purviews(direction, mechanism):
                        counter.charge(1)
                        purview_evaluations += 1
                        sweeps += _mechanism_partition_count(
                            len(mechanism), len(purview), counter
                        )
    except _LimitReached:
        capped = True

    version = config.formalism.iit.version
    relations_closed_form = None
    possible_distinctions = None
    possible_relations = None
    if version.startswith("IIT_4_0") and scope in ("full", "ces"):
        relations_closed_form = config.formalism.iit.relation_computation == "ANALYTICAL"
        if all(int(alphabet[i]) == 2 for i in indices):
            from pyphi.formalism.iit4 import bounds

            possible_distinctions = bounds.number_of_possible_distinctions(m)
            possible_relations = bounds.number_of_possible_relations(m)

    return AnalysisEstimate(
        n_units=m,
        state_space_size=state_space_size,
        compute=scope,
        system_partitions=system_partition_count,
        mechanisms=mechanisms,
        purview_evaluations=purview_evaluations,
        mechanism_partition_sweeps=sweeps,
        relations_closed_form=relations_closed_form,
        possible_distinctions=possible_distinctions,
        possible_relations=possible_relations,
        capped=capped,
    )
