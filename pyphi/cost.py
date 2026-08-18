"""Analytic workload counting for single-system analyses.

Counts the work a :func:`pyphi.analyze` call would perform — system
partitions swept by the system irreducibility analysis, candidate
mechanisms, connectivity-pruned purview evaluations, and mechanism
partitions per (mechanism, purview) pair — without computing any φ.
Counts are produced by driving the same enumeration machinery the
analysis uses under the active configuration, so the partition schemes,
the connectivity, and the alphabet are all reflected exactly.

Counts are turned into work units by weighting each axis by its measured
relative cost, so a unit is the same amount of work whatever mix of
purview evaluations and partition sweeps produces it. One further
constant, :data:`SECONDS_PER_UNIT`, converts units to CPU seconds on
reference hardware; :func:`units_for_runtime` inverts it, which is how a
per-shard runtime target becomes a ``units_per_job`` budget.
"""

from __future__ import annotations

import math
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

__all__ = [
    "AnalysisEstimate",
    "MechanismWorkload",
    "estimate_analysis",
    "mechanism_workloads",
    "partition_sweep_count",
    "round_memory_bytes",
    "runtime_seconds",
    "shard_cache_budget_bytes",
    "shard_memory_bytes",
    "units_for_runtime",
]

_PARTITION_COUNT_CAP = 6

PURVIEW_EVALUATION_UNITS = 12
"""Work units one purview evaluation costs, taking one partition as the unit.

Before any partition is swept, a (mechanism, direction, purview) pair
computes its unpartitioned repertoire and searches the candidate specified
states. Regressing per-pair CPU time on the pair's partition count over
mechanism orders 1 to 6 and purview orders 1 to 3 puts that fixed cost at
about twelve partition evaluations (524 µs against 41.8 µs). Charging it as
one, as a plain operation count does, undercounts a scope whose purviews are
small enough that few partitions amortize it, by up to 2.5× on the smallest
pairs.

The measurement is in ``experiments/units_runtime_model``.
"""

SECONDS_PER_UNIT = 4.4e-5
"""CPU seconds one work unit costs on reference hardware.

Calibrated against per-shard CPU time over eleven shards spanning 16- and
21-unit Ising substrates, both payload kinds, 0.2 M to 20 M units, 1 to 66
distinct mechanisms packed, and cache ceilings from unlimited to fully
binding. Cost per unit held between 41 and 51 µs across all of them; the
widest departures are the largest shard (+15%, part of it plausibly
memory-bandwidth contention) and a starved cache (+12%).

Hardware differs, so treat this as the reference for
:func:`units_for_runtime` and re-derive it from a campaign's own recorded
metrics (``CampaignTaskOutput.metrics``) when planning against a hard
runtime deadline.
"""

REPERTOIRE_FACTOR = 4
"""Repertoires concurrently alive during a mechanism-partition sweep."""

BASE_MEMORY_BYTES = 1 << 30
"""Per-task overhead: interpreter, imports, substrate TPM, task payload."""

CACHE_HEADROOM_BYTES = 1 << 30
"""Memory a shard's request grants its repertoire caches.

A shard evaluates every mechanism it carries against one long-lived
``System``, whose cached repertoires are released only when that ``System``
is collected, so cache occupancy grows with the number of mechanisms packed
into a shard rather than with the size of any one repertoire. Granting the
allowance in the request and enforcing it during execution (see
:func:`shard_cache_budget_bytes`) bounds that growth whatever the shard
packs.
"""

_CACHE_RESERVE_BYTES = 256 * 1024**2

_MEMORY_STEP_BYTES = 512 * 1024**2


@dataclass(frozen=True)
class MechanismWorkload:
    """One mechanism's scoped workload and peak-memory driver.

    ``units`` counts purview evaluations plus mechanism-partition sweeps;
    ``max_repertoire_cells`` is the state-space size of the largest scoped
    purview (the product of its units' state counts), which sets the
    mechanism's peak repertoire memory.
    """

    units: int
    max_repertoire_cells: int


def shard_memory_bytes(max_repertoire_cells: int) -> int:
    """Estimated peak memory of a shard from its largest repertoire.

    ``REPERTOIRE_FACTOR × 8 bytes × max_repertoire_cells +
    BASE_MEMORY_BYTES + CACHE_HEADROOM_BYTES``. The factor and base are
    calibration constants validated against scheduler-reported memory
    usage; the headroom is the cache allowance that
    :func:`shard_cache_budget_bytes` enforces during execution. Requests
    derived from this estimate are rounded with
    :func:`round_memory_bytes`.
    """
    return (
        REPERTOIRE_FACTOR * 8 * max_repertoire_cells
        + BASE_MEMORY_BYTES
        + CACHE_HEADROOM_BYTES
    )


def shard_cache_budget_bytes(memory_bytes: int) -> int:
    """Cache ceiling for a shard whose memory request is ``memory_bytes``.

    The request less a reserve for the allocations in flight when the
    ceiling is reached, since the caches stop storing at the ceiling but
    the computation continues to allocate above it.
    """
    return max(0, memory_bytes - _CACHE_RESERVE_BYTES)


def round_memory_bytes(n: int) -> int:
    """Round a byte count up to the next 512 MB request boundary."""
    return max(1, math.ceil(n / _MEMORY_STEP_BYTES)) * _MEMORY_STEP_BYTES


def runtime_seconds(units: float) -> float:
    """Estimated CPU seconds to compute ``units`` work units.

    Examples
    --------
    >>> round(runtime_seconds(1e6))
    44
    """
    return units * SECONDS_PER_UNIT


def units_for_runtime(seconds: float) -> float:
    """The ``units_per_job`` budget targeting ``seconds`` of CPU per shard.

    Pass the result to :func:`pyphi.campaign.prepare_ces` to plan shards
    against a runtime deadline rather than an abstract work count. The
    estimate holds while a shard's caches fit its memory request; a starved
    cache costs roughly a further 20%.

    Examples
    --------
    >>> round(units_for_runtime(3600))  # a one-hour shard
    81818182
    """
    return seconds / SECONDS_PER_UNIT


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
    ("JOINT_PARTITION_ALL", 2, 1): 3,
    ("JOINT_PARTITION_ALL", 2, 2): 9,
    ("JOINT_PARTITION_ALL", 2, 3): 27,
    ("JOINT_PARTITION_ALL", 2, 4): 81,
    ("JOINT_PARTITION_ALL", 2, 5): 243,
    ("JOINT_PARTITION_ALL", 2, 6): 729,
    ("JOINT_PARTITION_ALL", 2, 7): 2_187,
    ("JOINT_PARTITION_ALL", 3, 1): 7,
    ("JOINT_PARTITION_ALL", 3, 2): 31,
    ("JOINT_PARTITION_ALL", 3, 3): 121,
    ("JOINT_PARTITION_ALL", 3, 4): 451,
    ("JOINT_PARTITION_ALL", 3, 5): 1_657,
    ("JOINT_PARTITION_ALL", 3, 6): 6_091,
    ("JOINT_PARTITION_ALL", 3, 7): 22_561,
    ("JOINT_PARTITION_ALL", 4, 1): 15,
    ("JOINT_PARTITION_ALL", 4, 2): 93,
    ("JOINT_PARTITION_ALL", 4, 3): 459,
    ("JOINT_PARTITION_ALL", 4, 4): 2_085,
    ("JOINT_PARTITION_ALL", 4, 5): 9_195,
    ("JOINT_PARTITION_ALL", 4, 6): 40_293,
    ("JOINT_PARTITION_ALL", 4, 7): 177_339,
    ("JOINT_PARTITION_ALL", 5, 1): 31,
    ("JOINT_PARTITION_ALL", 5, 2): 271,
    ("JOINT_PARTITION_ALL", 5, 3): 1_681,
    ("JOINT_PARTITION_ALL", 5, 4): 9_211,
    ("JOINT_PARTITION_ALL", 5, 5): 48_001,
    ("JOINT_PARTITION_ALL", 5, 6): 245_491,
    ("JOINT_PARTITION_ALL", 5, 7): 1_251_001,
    ("JOINT_PARTITION_ALL", 6, 1): 63,
    ("JOINT_PARTITION_ALL", 6, 2): 789,
    ("JOINT_PARTITION_ALL", 6, 3): 6_147,
    ("JOINT_PARTITION_ALL", 6, 4): 40_341,
    ("JOINT_PARTITION_ALL", 6, 5): 245_523,
    ("JOINT_PARTITION_ALL", 6, 6): 1_444_149,
    ("JOINT_PARTITION_ALL", 6, 7): 8_379_987,
    ("JOINT_PARTITION_ALL", 7, 1): 127,
    ("JOINT_PARTITION_ALL", 7, 2): 2_311,
    ("JOINT_PARTITION_ALL", 7, 3): 22_681,
    ("JOINT_PARTITION_ALL", 7, 4): 177_451,
    ("JOINT_PARTITION_ALL", 7, 5): 1_251_097,
    ("JOINT_PARTITION_ALL", 7, 6): 8_380_051,
    ("JOINT_PARTITION_ALL", 7, 7): 54_762_961,
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
        ``"full"``, ``"sia"``, ``"ces"``, or ``"distinctions"``.
    system_partitions : int or None
        Partitions the system irreducibility analysis sweeps, under the
        active system partition scheme. Counted for a ``"ces"`` analysis
        under IIT 4.0, whose cause-effect structure embeds a system
        irreducibility analysis, but not under IIT 3.0, whose structure is
        the bare distinctions.
    specified_state_evaluations : int or None
        Forward-repertoire evaluations the specified-state search performs
        (Albantakis et al. 2023, Eq. 53). The search maximizes intrinsic
        information over the whole system as both mechanism and purview, so
        it evaluates one repertoire per system state per direction: twice
        the state space, each evaluation over an array of that same size.
        Unlike the other axes this one grows with the size of the system
        rather than of any mechanism. Counted under IIT 4.0 for a ``"sia"``,
        ``"ces"``, or full analysis; ``None`` under IIT 3.0, which has no
        specified state. A ``"distinctions"`` analysis performs no search,
        though filtering those distinctions for congruence
        (:meth:`~pyphi.system.System.distinctions` with ``congruent=True``)
        performs one.
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
    specified_state_evaluations: int | None
    mechanisms: int | None
    purview_evaluations: int | None
    mechanism_partition_sweeps: int | None
    relations_closed_form: bool | None
    possible_distinctions: int | None
    possible_relations: int | None
    capped: bool

    @property
    def distinction_units(self) -> int | None:
        """Work units on the distinction axis, or ``None`` if not counted.

        The weighted sum :func:`mechanism_workloads` charges — purview
        evaluations at :data:`PURVIEW_EVALUATION_UNITS` each plus every
        mechanism partition — so :func:`runtime_seconds` applies to it. The
        system-partition axis is excluded: its cost per partition has not
        been calibrated against this unit.
        """
        if self.purview_evaluations is None or self.mechanism_partition_sweeps is None:
            return None
        return (
            PURVIEW_EVALUATION_UNITS * self.purview_evaluations
            + self.mechanism_partition_sweeps
        )

    def _qualifier(self) -> str:
        return "≥" if self.capped else "="

    def _pandas_record(self) -> dict:
        return {
            "n_units": self.n_units,
            "state_space_size": self.state_space_size,
            "compute": self.compute,
            "system_partitions": self.system_partitions,
            "specified_state_evaluations": self.specified_state_evaluations,
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
        if self.specified_state_evaluations is not None:
            rows.append(
                Row(
                    "Specified-state evaluations",
                    _fmt(self.specified_state_evaluations),
                )
            )
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
    scope: Any | None = None,
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
        system-partition axis; ``"distinctions"`` only the distinction
        axis; ``"ces"`` the distinction axis, plus the system-partition
        axis under IIT 4.0, where unfolding a cause-effect structure
        computes a system irreducibility analysis first.
    limit : int, optional
        Work budget for the counting walk itself: purview evaluations and
        fresh partition enumerations each cost one unit, while memoized
        partition counts are free. A walk that exceeds the budget stops
        immediately and reports ``capped=True``.
    scope : :class:`~pyphi.campaign.scope.CESScope`, optional
        Restrict the counted mechanisms and purviews to the scope's
        feasibility surface. Affects only the distinction axis; the
        system-partition count and the structural ceilings
        (``possible_distinctions``, ``possible_relations``) are properties
        of the full system.

    Returns
    -------
    AnalysisEstimate
        The counted workload.

    Raises
    ------
    ValueError
        If ``compute`` is not ``"sia"``, ``"ces"``, ``"distinctions"``, or
        ``None``.

    Examples
    --------
    >>> from pyphi import examples
    >>> est = estimate_analysis(examples.basic_substrate())
    >>> est.mechanisms
    7
    >>> est.system_partitions
    22
    """
    if compute not in (None, "sia", "ces", "distinctions"):
        raise ValueError(
            f"unknown compute: {compute!r}; expected 'sia', 'ces', "
            "'distinctions', or None for the full analysis"
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
    unit_scope = scope
    scope = "full" if compute is None else compute
    version = config.formalism.iit.version

    # Under IIT 4.0 a cause-effect structure embeds its own system
    # irreducibility analysis (Eq. 57), so it pays the system-partition axis
    # too; under IIT 3.0 the structure is the bare distinctions. Only
    # ``"distinctions"`` skips that axis under every formalism.
    counts_system_partitions = scope in ("full", "sia") or (
        scope == "ces" and version.startswith("IIT_4_0")
    )
    counts_distinctions = scope in ("full", "ces", "distinctions")

    # The specified-state search runs wherever a system irreducibility
    # analysis does, and only under IIT 4.0. Its cost follows from the state
    # space alone — one forward repertoire per system state per direction —
    # so no enumeration walk is needed and the work budget never caps it.
    specified_state_evaluations = (
        2 * state_space_size
        if counts_system_partitions and version.startswith("IIT_4_0")
        else None
    )

    counter = _Counter(limit)
    capped = False
    system_partition_count = None
    mechanisms = None
    purview_evaluations = None
    sweeps = None
    try:
        if counts_system_partitions:
            system_partition_count = _system_partition_count(m, counter)
        if counts_distinctions:
            mechanism_iter: Any = utils.powerset(indices, nonempty=True)
            if unit_scope is not None:
                mechanism_iter = unit_scope.mechanisms.select(mechanism_iter)
            mechanisms = 0
            purview_evaluations = 0
            sweeps = 0
            for mechanism in mechanism_iter:
                mechanisms += 1
                for direction in (Direction.CAUSE, Direction.EFFECT):
                    if unit_scope is not None:
                        axis = unit_scope.purview_axis(direction, mechanism)
                        purviews = list(
                            axis.select(
                                cs.potential_purviews(
                                    direction, mechanism, max_order=axis.order_bound()
                                )
                            )
                        )
                    else:
                        purviews = cs.potential_purviews(direction, mechanism)
                    for purview in purviews:
                        counter.charge(1)
                        purview_evaluations += 1
                        sweeps += _mechanism_partition_count(
                            len(mechanism), len(purview), counter
                        )
    except _LimitReached:
        capped = True

    relations_closed_form = None
    possible_distinctions = None
    possible_relations = None
    if version.startswith("IIT_4_0") and counts_distinctions:
        from pyphi.formalism.iit4 import bounds

        # A ``"distinctions"`` analysis stops before relations.
        unfolds_relations = scope != "distinctions"
        if unfolds_relations:
            relations_closed_form = (
                config.formalism.iit.relation_computation == "ANALYTICAL"
            )
        if all(int(alphabet[i]) == 2 for i in indices):
            possible_distinctions = bounds.number_of_possible_distinctions(m)
            if unfolds_relations:
                possible_relations = bounds.number_of_possible_relations(m)

    return AnalysisEstimate(
        n_units=m,
        state_space_size=state_space_size,
        compute=scope,
        system_partitions=system_partition_count,
        specified_state_evaluations=specified_state_evaluations,
        mechanisms=mechanisms,
        purview_evaluations=purview_evaluations,
        mechanism_partition_sweeps=sweeps,
        relations_closed_form=relations_closed_form,
        possible_distinctions=possible_distinctions,
        possible_relations=possible_relations,
        capped=capped,
    )


def partition_sweep_count(mechanism_size: int, purview_size: int) -> int:
    """Memoized mechanism-partition count for one (mechanism, purview) pair
    under the active mechanism partition scheme."""
    counter = _Counter(2**63)
    return _mechanism_partition_count(mechanism_size, purview_size, counter)


def mechanism_workloads(
    substrate: Substrate,
    subset: Any = None,
    scope: Any | None = None,
    limit: int = 10_000_000,
) -> dict[tuple[int, ...], MechanismWorkload]:
    """Per-mechanism workload under a scope, keyed by mechanism.

    Each mechanism's :class:`MechanismWorkload` weighs its scoped purview
    evaluations against its mechanism-partition sweeps by their measured
    relative cost, and records the state-space size of its largest scoped
    purview. Summed over all mechanisms, the units equal
    ``PURVIEW_EVALUATION_UNITS`` × the scoped
    :attr:`AnalysisEstimate.purview_evaluations` plus the scoped
    :attr:`AnalysisEstimate.mechanism_partition_sweeps`.

    Parameters
    ----------
    substrate : Substrate
        The substrate to analyze.
    subset : optional
        Node indices (or labels) of the candidate system; ``None`` uses
        the whole substrate.
    scope : :class:`~pyphi.campaign.scope.CESScope`, optional
        Restrict the counted mechanisms and purviews.
    limit : int, optional
        Work budget for the counting walk.

    Raises
    ------
    ValueError
        If the counting walk exceeds ``limit`` — the workload is then too
        large to plan; narrow the scope or raise the limit.
    """
    from pyphi import utils
    from pyphi.direction import Direction
    from pyphi.system import System

    cs = System.from_substrate(substrate, (0,) * substrate.size, subset)
    alphabet = substrate.factored_tpm.alphabet_sizes
    counter = _Counter(limit)
    workloads: dict[tuple[int, ...], MechanismWorkload] = {}
    mechanism_iter: Any = utils.powerset(cs.node_indices, nonempty=True)
    if scope is not None:
        mechanism_iter = scope.mechanisms.select(mechanism_iter)
    try:
        for mechanism in mechanism_iter:
            units = 0
            max_cells = 0
            for direction in (Direction.CAUSE, Direction.EFFECT):
                if scope is not None:
                    axis = scope.purview_axis(direction, mechanism)
                    purviews = list(
                        axis.select(
                            cs.potential_purviews(
                                direction, mechanism, max_order=axis.order_bound()
                            )
                        )
                    )
                else:
                    purviews = cs.potential_purviews(direction, mechanism)
                for purview in purviews:
                    counter.charge(1)
                    units += PURVIEW_EVALUATION_UNITS + _mechanism_partition_count(
                        len(mechanism), len(purview), counter
                    )
                    max_cells = max(max_cells, math.prod(alphabet[u] for u in purview))
            workloads[tuple(mechanism)] = MechanismWorkload(
                units=units, max_repertoire_cells=max_cells
            )
    except _LimitReached:
        raise ValueError(
            f"mechanism workload walk exceeded limit={limit}; narrow the "
            "scope or raise the limit"
        ) from None
    return workloads
