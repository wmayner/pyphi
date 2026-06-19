# formalism/iit3/__init__.py
"""IIT 3.0 (Oizumi et al. 2014) algorithms.

Distribution-distance-based phi computation. Partition scheme: bipartitions
(``BI``). Compatible measures: ``EMD``, ``L1``, ``KLD``, ``ENTROPY_DIFFERENCE``,
``PSQ2``, ``MP2Q``, ``AID`` (absolute intrinsic difference), ``ID``
(intrinsic difference).

The dispatch class :class:`IIT3Formalism` lives in :mod:`.formalism` and
calls back into this module's ``sia`` / ``ces`` / etc. for the actual
computation.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING
from typing import Any

from more_itertools import collapse

from pyphi import conf
from pyphi import connectivity
from pyphi import utils
from pyphi.conf import config
from pyphi.direction import Direction
from pyphi.measures.ces import ces_distance
from pyphi.models import Concept
from pyphi.models import DirectedBipartition
from pyphi.models import Distinctions
from pyphi.models import IIT3SystemIrreducibilityAnalysis
from pyphi.models import NullResultReason
from pyphi.models import ResolvedDistinctions
from pyphi.models import UnresolvedDistinctions
from pyphi.models import _null_sia
from pyphi.models.ces import CauseEffectStructure
from pyphi.models.explanation import runner_up_from_candidates
from pyphi.parallel import MapReduce
from pyphi.partition import system_partition_types
from pyphi.relations import NullRelations
from pyphi.types import Mechanism
from pyphi.types import Purview

if TYPE_CHECKING:
    from pyphi.labels import NodeLabels
    from pyphi.system import System

# Create a logger for this module.
log = logging.getLogger(__name__)


def concept(
    system: System,
    mechanism: Mechanism,
    purviews: Iterable[Purview] | None = None,
    cause_purviews: Iterable[Purview] | None = None,
    effect_purviews: Iterable[Purview] | None = None,
    **kwargs: Any,
) -> Concept:
    """Return the IIT 3.0 concept specified by a mechanism.

    Args:
        system (System): The system the mechanism belongs to.
        mechanism (tuple[int]): The mechanism for which to determine the
            concept.

    Keyword Args:
        purviews (tuple[tuple[int]]): A list of purviews to consider.
        cause_purviews (tuple[tuple[int]]): A list of cause purviews to
            consider, overriding ``purviews``.
        effect_purviews (tuple[tuple[int]]): A list of effect purviews to
            consider, overriding ``purviews``.

    Returns:
        Concept: The concept of the given mechanism.
    """
    from pyphi.core import repertoire_algebra as _ra
    from pyphi.formalism.queries import mic
    from pyphi.formalism.queries import mie

    if not mechanism:
        return _ra.null_concept(system)

    cause_purviews = cause_purviews if cause_purviews is not None else purviews
    cause = mic(system, mechanism, purviews=cause_purviews, **kwargs)

    effect_purviews = effect_purviews if effect_purviews is not None else purviews
    effect = mie(system, mechanism, purviews=effect_purviews, **kwargs)

    return Concept(mechanism=mechanism, cause=cause, effect=effect)


def _compute_distinctions(
    system: System,
    mechanisms: Iterable[Mechanism] | None = None,
    purviews: Iterable[Purview] | None = None,
    cause_purviews: Iterable[Purview] | None = None,
    effect_purviews: Iterable[Purview] | None = None,
    directions: Iterable[Direction] | None = None,
    only_positive_phi: bool = True,
    **kwargs: Any,
) -> Distinctions:
    """Compute the bag of distinctions for a system, restricted by the
    given mechanism / purview / direction filters.

    Args:
        system (System): The system for which to determine the
            |Distinctions|.

    Keyword Args:
        mechanisms (tuple[tuple[int]]): Restrict possible mechanisms to those
            in this list.
        purviews (tuple[tuple[int]]): Same as in :func:`pyphi.formalism.iit3.concept`.
        cause_purviews (tuple[tuple[int]]): Same as in
            :func:`pyphi.formalism.iit3.concept`.
        effect_purviews (tuple[tuple[int]]): Same as in
            :func:`pyphi.formalism.iit3.concept`.
        parallel (bool): Whether to compute concepts in parallel. If ``True``,
            overrides :data:`config.infrastructure.parallel_concept_evaluation`.
        directions (Iterable[Direction]): Restrict possible directions to these.
        only_positive_phi (bool): Whether to only return concepts with positive
            phi.

    Returns:
        Distinctions: A tuple of every |Concept| in the cause-effect
        structure.
    """
    total = None
    if mechanisms is None:
        mechanisms = utils.powerset(system.node_indices, nonempty=True)
        total = 2 ** len(system.node_indices) - 1
    else:
        with contextlib.suppress(TypeError):
            total = len(mechanisms)  # type: ignore[arg-type]  # mechanisms may be generator

    def compute_concept(*args, **kwargs):
        return concept(system, *args, **kwargs, progress=False)

    reduce_func = _only_positive_phi if only_positive_phi else _any_phi
    parallel_kwargs = conf.parallel_kwargs(
        dict(config.infrastructure.parallel_concept_evaluation), **kwargs
    )
    concepts = MapReduce(
        compute_concept,
        mechanisms,
        map_kwargs={  # type: ignore[arg-type]  # None values allowed in map_kwargs
            "purviews": purviews,
            "cause_purviews": cause_purviews,
            "effect_purviews": effect_purviews,
            "directions": directions,
        },
        reduce_func=reduce_func,
        desc="Computing concepts",
        total=total,
        **parallel_kwargs,  # type: ignore[arg-type]  # parallel_kwargs contains MapReduce params
    ).run()
    # ``find_mice`` may return tied specified states under the active
    # formalism (IIT 4.0 in particular), so the conservative answer is
    # ``UnresolvedDistinctions``. Callers in IIT 4.0 phi_structure
    # immediately call ``resolve_congruence``; IIT 3.0 callers route
    # through ``ces_distance`` which accepts the base ``Distinctions``.
    return UnresolvedDistinctions(concepts)


def ces(
    system: System,
    *,
    sia: IIT3SystemIrreducibilityAnalysis | None = None,
    distinctions: Distinctions | None = None,
    sia_kwargs: dict | None = None,
    distinctions_kwargs: dict | None = None,
) -> CauseEffectStructure:
    """Compute the cause-effect structure of a system under IIT 3.0.

    Returns a :class:`CauseEffectStructure` wrapping the SIA, the resolved
    distinctions, and an empty :class:`NullRelations` (IIT 3.0 does not
    define relations between distinctions).

    Pass ``sia=`` or ``distinctions=`` to reuse pre-computed values.
    """
    import time

    from pyphi.provenance import stamp_wall_time

    start = time.perf_counter()
    sia_kwargs = sia_kwargs or {}
    distinctions_kwargs = distinctions_kwargs or {}

    if sia is None:
        sia = _sia(system, **sia_kwargs)
    if distinctions is None:
        cached = sia.distinctions
        if cached is not None and not distinctions_kwargs:
            distinctions = cached
        else:
            distinctions = _compute_distinctions(system, **distinctions_kwargs)

    if not isinstance(distinctions, ResolvedDistinctions):
        distinctions = ResolvedDistinctions(distinctions)

    result = CauseEffectStructure(
        sia=sia,
        distinctions=distinctions,
        relations=NullRelations(),
    )
    return stamp_wall_time(result, time.perf_counter() - start)


def _only_positive_phi(concepts: Iterable[Any]) -> list[Concept]:
    return list(filter(None, collapse(concepts)))


def _any_phi(concepts: Iterable[Any]) -> list[Concept]:
    return list(collapse(concepts))


def conceptual_info(system: System, **kwargs: Any) -> float:
    """Return the conceptual information for a |System|.

    This is the distance from the system's |Distinctions| to the
    null concept.
    """
    ci = ces_distance(
        _compute_distinctions(system, **kwargs), ResolvedDistinctions(()), system=system
    )
    return round(ci, config.numerics.precision)  # type: ignore[arg-type]  # config.Option descriptor


def evaluate_partition(
    partition: DirectedBipartition,
    unpartitioned_system: System,
    unpartitioned_ces: Distinctions,
    **kwargs: Any,
) -> IIT3SystemIrreducibilityAnalysis:
    """Compute the system irreducibility for a given partition.

    Args:
        unpartitioned_system (System): The system without a partition applied.
        partition (DirectedBipartition): The partition to evaluate.
        unpartitioned_ces (Distinctions): The cause-effect structure of
            the unpartitioned system.

    Returns:
        IIT3SystemIrreducibilityAnalysis: The |big_phi| analysis for
        that partition.
    """
    log.debug("Evaluating %s...", partition)

    partitioned_system = unpartitioned_system.apply_cut(partition)

    if config.formalism.iit.assume_partitions_cannot_create_new_concepts:
        mechanisms = list(unpartitioned_ces.mechanisms)
    else:
        # Mechanisms can only produce concepts if they were concepts in the
        # original system, or the partition splits the mechanism.
        mechanisms = set(
            list(unpartitioned_ces.mechanisms)
            + list(partitioned_system.partitioned_mechanisms)
        )

    kwargs = {"progress": False, **kwargs}
    partitioned_ces = _compute_distinctions(partitioned_system, mechanisms, **kwargs)

    log.debug("Finished evaluating %s.", partition)

    phi_ = ces_distance(unpartitioned_ces, partitioned_ces, system=unpartitioned_system)

    return IIT3SystemIrreducibilityAnalysis(
        phi=phi_,
        partitioned_distinctions=partitioned_ces,
        partition=partitioned_system.partition,
        node_indices=unpartitioned_system.node_indices,
        node_labels=unpartitioned_system.substrate.node_labels,
        current_state=unpartitioned_system.state,
    )


def sia_partitions(
    nodes: tuple[int, ...], node_labels: NodeLabels | None = None
) -> list[DirectedBipartition]:
    """Return all |big_phi| cuts for the given nodes.

    Controlled by the :const:`config.formalism.iit.system_partition_scheme` option.

    Arguments:
        nodes (tuple[int]): The node indices to partition.

    Keyword Arguments:
        node_labels (NodeLabels): Enables printing the partition with labels.

    Returns:
        list[DirectedBipartition]: All unidirectional partitions.

    """
    # TODO(4.0 consolidate 3.0 and 4.0 cuts)
    scheme = config.formalism.iit.system_partition_scheme
    from pyphi.formalism.iit3.formalism import IIT3Formalism

    valid = IIT3Formalism.compatible_system_partition_schemes
    if valid is not None and scheme not in valid:
        raise ValueError(
            "IIT 3.0 calculations must use one of the following system "
            f"partition schemes: {sorted(valid)}; got {scheme}"
        )
    return system_partition_types[config.formalism.iit.system_partition_scheme](  # type: ignore[index]  # config.Option descriptor
        nodes, node_labels=node_labels
    )


def _ces(system: System, **kwargs: Any) -> Distinctions:
    """Compute the unpartitioned |Distinctions| using the partition-evaluation
    parallel settings, on the rationale that no cuts are being evaluated yet
    so the same worker budget is available.
    """
    kwargs = {**dict(config.infrastructure.parallel_partition_evaluation), **kwargs}
    return _compute_distinctions(system, **kwargs)


def _sia_map_reduce(
    cuts: Iterable[DirectedBipartition],
    system: System,
    unpartitioned_ces: Distinctions,
    **kwargs: Any,
) -> IIT3SystemIrreducibilityAnalysis:
    """Evaluate every partition and select the MIP via
    ``resolve_ties.sias``. The config knob
    ``config.formalism.iit.sia_tie_resolution`` controls the selection
    strategy.
    """
    from pyphi import resolve_ties

    kwargs = {**dict(config.infrastructure.parallel_partition_evaluation), **kwargs}
    null = _null_sia(system, reasons=[NullResultReason.NO_VALID_PARTITIONS])
    candidates = MapReduce(
        evaluate_partition,
        cuts,
        map_kwargs={
            "unpartitioned_system": system,
            "unpartitioned_ces": unpartitioned_ces,
        },
        shortcircuit_func=utils.is_falsy,
        desc="Evaluating cuts",
        **kwargs,
    ).run()
    if not candidates:
        return null
    ties = tuple(resolve_ties.sias(candidates, default=null))
    if not ties:
        return null
    winner = ties[0]
    winner.runner_up = runner_up_from_candidates(candidates, winner.phi)
    # Capture phi-tied peers (same minimum phi value) before any
    # lex-based tiebreaker collapses the set to a single winner. These
    # peers are equally valid MIPs for the system; preserving them on
    # the winner lets callers (serialization, diagnostic display, and
    # fixtures comparing across tied alternatives) see the full tie set.
    phi_ties = tuple(resolve_ties.sias(candidates, strategy=["PHI"], default=null))
    winner.set_ties(list(phi_ties) if len(phi_ties) > 1 else [winner])
    return winner


def _sia(system: System, **kwargs: Any) -> IIT3SystemIrreducibilityAnalysis:
    """Return the minimal information partition of a system.

    Args:
        system (System): The candidate set of nodes.

    Returns:
        IIT3SystemIrreducibilityAnalysis: A nested structure containing all the
        data from the intermediate calculations. The top level contains the
        basic irreducibility information for the given system.
    """
    # pylint: disable=unused-argument

    log.info("Calculating big-phi data for %s...", system)

    # Check for degenerate cases
    # =========================================================================
    # Phi is necessarily zero if the system is:
    #   - not strongly connected;
    #   - empty;
    #   - an elementary micro mechanism (i.e. no nontrivial bipartitions).
    # So in those cases we immediately return a null SIA.
    if not system:
        log.info("System %s is empty; returning null SIA immediately.", system)
        return _null_sia(system, reasons=[NullResultReason.NO_SYSTEM])

    if not connectivity.is_strong(system.cm, system.node_indices):
        log.info(
            "%s is not strongly connected; returning null SIA immediately.",
            system,
        )
        return _null_sia(system, reasons=[NullResultReason.NO_STRONG_CONNECTIVITY])

    # Handle elementary micro mechanism cases.
    # Single macro element systems have nontrivial bipartitions because their
    #   bipartitions are over their micro elements.
    if len(system.partition_indices) == 1:
        # If the node lacks a self-loop, phi is trivially zero.
        if not system.cm[system.node_indices][system.node_indices]:
            log.info(
                "Single micro nodes %s without selfloops cannot have "
                "phi; returning null SIA immediately.",
                system,
            )
            return _null_sia(system, reasons=[NullResultReason.MONAD_WITH_NO_SELFLOOP])
        # Even if the node has a self-loop, we may still define phi to be zero.
        if not config.formalism.iit.single_micro_nodes_with_selfloops_have_phi:
            log.info(
                "Single micro nodes %s with selfloops cannot have "
                "phi; returning null SIA immediately.",
                system,
            )
            return _null_sia(
                system,
                reasons=[NullResultReason.MONAD_WITH_SELFLOOP_DEFINED_TO_BE_ZERO_PHI],
            )
    # =========================================================================

    log.debug("Finding unpartitioned Distinctions...")
    unpartitioned_ces = _ces(system, progress=kwargs.get("progress"))

    if not unpartitioned_ces:
        log.info("Empty unpartitioned Distinctions; returning null SIA immediately.")
        # Short-circuit if there are no concepts in the unpartitioned CES.
        return _null_sia(system, reasons=[NullResultReason.EMPTY_CAUSE_EFFECT_STRUCTURE])

    log.debug("Found unpartitioned Distinctions.")

    # TODO: move this into sia_bipartitions?
    # Only True if SINGLE_MICRO_NODES...=True, no?
    if len(system.partition_indices) == 1:
        cuts = [
            DirectedBipartition(
                Direction.EFFECT,
                system.partition_indices,
                system.partition_indices,
                system.partition_node_labels,
            )
        ]
    else:
        cuts = sia_partitions(system.partition_indices, system.partition_node_labels)

    result = _sia_map_reduce(cuts, system, unpartitioned_ces, **kwargs)

    # Attach the unpartitioned distinctions already computed above so that
    # ``ces`` can reuse them rather than recomputing.
    result.distinctions = unpartitioned_ces

    if config.infrastructure.clear_system_caches_after_computing_sia:
        log.debug("Clearing system caches.")
        system.clear_caches()

    log.info("Finished calculating big-phi data for %s.", system)

    return result


sia = _sia


def phi(system: System) -> float:
    """Return the |big_phi| value of a system."""
    return sia(system).phi
