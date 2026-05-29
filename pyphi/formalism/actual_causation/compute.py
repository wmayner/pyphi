"""Actual-causation compute algorithms (Albantakis et al. 2019).

The algorithm layer of the AC formalism: account/SIA evaluation, mechanism
and causal-link MIP search, the partitioned-repertoire / background /
alpha-aggregation registries, and the ``probability_distance`` /
``account_distance`` utilities. Operates on :class:`pyphi.actual.Transition`
objects passed in; the data layer (``Transition`` / ``TransitionSystem``)
lives in :mod:`pyphi.actual`.
"""

from __future__ import annotations

import functools
import logging
from itertools import chain
from typing import TYPE_CHECKING
from typing import Any

import numpy as np

from pyphi import conf
from pyphi import connectivity
from pyphi import resolve_ties
from pyphi import utils
from pyphi import validate
from pyphi.conf import config
from pyphi.direction import Direction
from pyphi.measures.distribution import actual_causation_measures as measures
from pyphi.measures.protocols import DistributionMeasure
from pyphi.models import Account
from pyphi.models import AcRepertoireIrreducibilityAnalysis
from pyphi.models import AcSystemIrreducibilityAnalysis
from pyphi.models import CausalLink
from pyphi.models import DirectedAccount
from pyphi.models import DirectedJointPartition
from pyphi.models import _null_ac_ria
from pyphi.models import _null_ac_sia
from pyphi.parallel import MapReduce
from pyphi.partition import mechanism_partitions
from pyphi.registry import Registry

if TYPE_CHECKING:
    from pyphi.actual import Transition

log = logging.getLogger(__name__)


class PartitionedRepertoireSchemeRegistry(Registry):
    """Registry of partitioned-repertoire computation schemes for actual causation.

    Schemes consume ``(transition_system, direction, partition)`` and
    return the partitioned repertoire as a probability distribution
    consistent with the parent System's TPM shape.
    """

    desc = "partitioned-repertoire schemes"


class BackgroundStrategyRegistry(Registry):
    """Registry of background-conditioning strategies for actual causation.

    Strategies consume ``(substrate, before_state, external_indices)`` and
    return either ``None`` (signaling uniform causal marginalization) or
    a state-weight callable.
    """

    desc = "background-conditioning strategies"


class AlphaAggregationRegistry(Registry):
    """Registry of α-aggregation rules for actual causation.

    Aggregators consume ``(rho, rho_partitioned)`` and return α — the
    integrated information of an actual cause/effect link.
    """  # noqa: RUF002

    desc = "α-aggregation rules"  # noqa: RUF001


partitioned_repertoire_schemes = PartitionedRepertoireSchemeRegistry()
background_strategies = BackgroundStrategyRegistry()
alpha_aggregations = AlphaAggregationRegistry()


@partitioned_repertoire_schemes.register("PRODUCT")
def _partitioned_repertoire_product(
    transition_system: Any,
    direction: Direction,
    partition: Any,
) -> Any:
    from pyphi.core import repertoire_algebra as ra

    repertoires = [
        ra.repertoire(transition_system, direction, part.mechanism, part.purview)
        for part in partition
    ]
    return functools.reduce(np.multiply, repertoires)


@background_strategies.register("UNIFORM")
def _background_uniform(
    substrate: Any,  # noqa: ARG001
    before_state: Any,  # noqa: ARG001
    external_indices: Any,  # noqa: ARG001
) -> Any:
    return None


@alpha_aggregations.register("SUBTRACTIVE")
def _alpha_subtractive(rho: float, rho_partitioned: float) -> float:
    return rho - rho_partitioned


def probability_distance(
    p: float,
    q: float,
    measure: str | None = None,
    *,
    alpha_measure: DistributionMeasure | None = None,
) -> float:
    """Compute the distance between two probabilities in actual causation.

    Args:
        p (float): The first probability.
        q (float): The second probability.

    Keyword Args:
        measure (str): Optional measure name registered in
            :data:`pyphi.measures.distribution.actual_causation_measures`.
            Mutually exclusive with ``alpha_measure``.
        alpha_measure (DistributionMeasure): Optional resolved measure callable
            (e.g., from
            :func:`pyphi.measures.distribution.resolve_actual_causation_measure`).
            Internal callers thread the resolved object through to avoid
            repeated registry lookups; external callers may pass ``measure``.
            If both are ``None``, the active configuration's
            ``alpha_measure`` is resolved.

    Returns:
        float: The probability distance between ``p`` and ``q``.
    """
    if alpha_measure is not None and measure is not None:
        raise ValueError(
            "probability_distance accepts at most one of "
            "`measure` or `alpha_measure`; got both."
        )
    if alpha_measure is None:
        name = (
            config.formalism.actual_causation.alpha_measure
            if measure is None
            else measure
        )
        measure_func = measures[name]
    else:
        measure_func = alpha_measure
    dist = measure_func(p, q)
    return round(dist, config.numerics.precision)


def account_distance(A1, A2):
    """Return the distance between two accounts. Here that is just the
    difference in sum(alpha)

    Args:
        A1 (Account): The first account.
        A2 (Account): The second account

    Returns:
        float: The distance between the two accounts.
    """
    return sum([action.alpha for action in A1]) - sum([action.alpha for action in A2])


def _find_mip(
    transition: Transition,
    direction,
    mechanism,
    purview,
    allow_neg=False,
    *,
    alpha_measure: DistributionMeasure | None = None,
    partitioned_repertoire_scheme=None,
):
    """Find the ratio minimum information partition for a mechanism
    over a purview.

    Args:
        direction (str): |CAUSE| or |EFFECT|
        mechanism (tuple[int]): A mechanism.
        purview (tuple[int]): A purview.

    Keyword Args:
        allow_neg (boolean): If true, ``alpha`` is allowed to be negative.
            Otherwise, negative values of ``alpha`` will be treated as if
            they were 0.
        alpha_measure (DistributionMeasure): Resolved alpha measure callable.
            When ``None``, ``config.formalism.actual_causation.alpha_measure``
            is resolved at the call boundary.
        partitioned_repertoire_scheme: Resolved partitioned-repertoire
            scheme callable. When ``None``,
            ``config.formalism.actual_causation.partitioned_repertoire_scheme``
            is resolved at the call boundary.

    Returns:
        AcRepertoireIrreducibilityAnalysis: The irreducibility analysis for
        the mechanism.
    """
    if not purview:
        return _null_ac_ria(
            transition.mechanism_state(direction), direction, mechanism, purview
        )

    probability = transition.probability(direction, mechanism, purview)
    candidates: list[AcRepertoireIrreducibilityAnalysis] = []
    for partition in mechanism_partitions(mechanism, purview, transition.node_labels):
        partitioned_probability = transition.partitioned_probability(
            direction,
            partition,
            partitioned_repertoire_scheme=partitioned_repertoire_scheme,
        )
        alpha = probability_distance(
            probability,
            partitioned_probability,
            alpha_measure=alpha_measure,
        )
        # Reducibility short-circuit: |alpha|=0 (or negative when
        # disallowed) means the mechanism is reducible against this
        # partition; no need to keep searching since min |alpha| can't
        # go lower.
        if utils.eq(alpha, 0) or (alpha < 0 and not allow_neg):
            return _null_ac_ria(
                transition.mechanism_state(direction),
                direction,
                mechanism,
                purview,
                partition,
            )
        candidates.append(
            AcRepertoireIrreducibilityAnalysis(
                state=transition.mechanism_state(direction),
                direction=direction,
                mechanism=mechanism,
                purview=purview,
                partition=partition,
                probability=probability,
                partitioned_probability=partitioned_probability,
                node_labels=transition.node_labels,
                alpha=alpha,
            )
        )
    if not candidates:
        return None
    context = resolve_ties.ResolutionContext(max_escalation_level="Determinism")
    outcome = resolve_ties.resolve_ac_partition_tie(candidates, context=context)
    winner = outcome.resolved
    if winner is not None and len(outcome.tied_set) > 1:
        winner.set_partition_ties(outcome.tied_set)
    return winner


def _find_causal_link(
    transition: Transition,
    direction,
    mechanism,
    purviews=None,
    allow_neg=False,
    *,
    alpha_measure: DistributionMeasure | None = None,
    partitioned_repertoire_scheme=None,
):
    """Return the maximally irreducible cause or effect ratio for a
    mechanism.

    Args:
        direction (str): The temporal direction, specifying cause or
            effect.
        mechanism (tuple[int]): The mechanism to be tested for
            irreducibility.

    Keyword Args:
        purviews (tuple[int]): Optionally restrict the possible purviews
            to a subset of the system. This may be useful for _e.g._
            finding only concepts that are "about" a certain subset of
            nodes.
        alpha_measure (DistributionMeasure): Resolved alpha measure
            callable. When ``None``,
            ``config.formalism.actual_causation.alpha_measure`` is
            resolved at the call boundary.
        partitioned_repertoire_scheme: Resolved partitioned-repertoire
            scheme callable. When ``None``,
            ``config.formalism.actual_causation.partitioned_repertoire_scheme``
            is resolved at the call boundary.

    Returns:
        CausalLink: The maximally-irreducible actual cause or effect.
    """
    purviews = transition.potential_purviews(direction, mechanism, purviews)

    # Find the maximal RIA over the remaining purviews.
    if not purviews:
        max_ria = _null_ac_ria(
            transition.mechanism_state(direction), direction, mechanism, None
        )
        return CausalLink(max_ria)

    # Finds rias with maximum alpha
    all_ria = [
        _find_mip(
            transition,
            direction,
            mechanism,
            purview,
            allow_neg=allow_neg,
            alpha_measure=alpha_measure,
            partitioned_repertoire_scheme=partitioned_repertoire_scheme,
        )
        for purview in purviews
    ]
    # Filter out None values and bail if no candidates have alpha > 0.
    valid_ria = [ria for ria in all_ria if ria is not None and bool(ria)]
    if not valid_ria:
        return []
    context = resolve_ties.ResolutionContext(max_escalation_level="Determinism")
    outcome = resolve_ties.resolve_ac_causal_link_tie(valid_ria, context=context)
    winner = outcome.resolved
    assert winner is not None, "AC causal-link cascade returned no winner"
    extended_purview = tuple(r.purview for r in outcome.tied_set)
    purview_ties = tuple(outcome.tied_set) if len(outcome.tied_set) > 1 else None
    return CausalLink(winner, extended_purview, purview_ties=purview_ties)


def _directed_account(
    transition,
    direction,
    mechanisms=None,
    purviews=None,
    allow_neg=False,
    *,
    alpha_measure: DistributionMeasure | None = None,
    partitioned_repertoire_scheme=None,
):
    """Return the set of all |CausalLinks| of the specified direction.

    Keyword Args:
        alpha_measure (DistributionMeasure): Resolved alpha measure callable.
            When ``None``, ``config.formalism.actual_causation.alpha_measure``
            is resolved at the call boundary.
        partitioned_repertoire_scheme: Resolved partitioned-repertoire scheme
            callable. When ``None``, the active
            ``config.formalism.actual_causation.partitioned_repertoire_scheme``
            is resolved at the call boundary.
    """
    if mechanisms is None:
        mechanisms = utils.powerset(
            transition.mechanism_indices(direction), nonempty=True
        )
    links = [
        _find_causal_link(
            transition,
            direction,
            mechanism,
            purviews=purviews,
            allow_neg=allow_neg,
            alpha_measure=alpha_measure,
            partitioned_repertoire_scheme=partitioned_repertoire_scheme,
        )
        for mechanism in mechanisms
    ]

    # Filter out causal links with zero alpha
    return DirectedAccount(filter(None, links))


def _account(
    transition,
    direction=Direction.BIDIRECTIONAL,
    *,
    alpha_measure: DistributionMeasure | None = None,
    partitioned_repertoire_scheme=None,
):
    """Return the set of all causal links for a |Transition|.

    Args:
        transition (Transition): The transition of interest.

    Keyword Args:
        direction (Direction): By default the account contains actual causes
            and actual effects.
        alpha_measure (DistributionMeasure): Resolved alpha measure callable.
            When ``None``, ``config.formalism.actual_causation.alpha_measure``
            is resolved at the call boundary.
        partitioned_repertoire_scheme: Resolved partitioned-repertoire scheme
            callable. When ``None``, the active
            ``config.formalism.actual_causation.partitioned_repertoire_scheme``
            is resolved at the call boundary.
    """
    if direction != Direction.BIDIRECTIONAL:
        return _directed_account(
            transition,
            direction,
            alpha_measure=alpha_measure,
            partitioned_repertoire_scheme=partitioned_repertoire_scheme,
        )

    return Account(
        _directed_account(
            transition,
            Direction.CAUSE,
            alpha_measure=alpha_measure,
            partitioned_repertoire_scheme=partitioned_repertoire_scheme,
        )
        + _directed_account(
            transition,
            Direction.EFFECT,
            alpha_measure=alpha_measure,
            partitioned_repertoire_scheme=partitioned_repertoire_scheme,
        )
    )


def _evaluate_partition(
    partition,
    transition,
    unpartitioned_account,
    direction=Direction.BIDIRECTIONAL,
    *,
    alpha_measure: DistributionMeasure,
    partitioned_repertoire_scheme,
):
    """Find the |AcSystemIrreducibilityAnalysis| for a given partition."""
    partitioned_transition = transition.apply_cut(partition)
    partitioned_account = _account(
        partitioned_transition,
        direction,
        alpha_measure=alpha_measure,
        partitioned_repertoire_scheme=partitioned_repertoire_scheme,
    )

    log.debug("Finished evaluating %s.", partition)
    alpha = account_distance(unpartitioned_account, partitioned_account)

    return AcSystemIrreducibilityAnalysis(
        alpha=round(alpha, config.numerics.precision),
        direction=direction,
        account=unpartitioned_account,
        partitioned_account=partitioned_account,
        partition=partition,
        before_state=transition.before_state,
        after_state=transition.after_state,
        size=len(transition),
        node_indices=transition.node_indices,
        cause_indices=transition.cause_indices,
        effect_indices=transition.effect_indices,
        node_labels=transition.substrate.node_labels,
    )


def _get_partitions(transition, direction):
    """A list of possible partitions of a transition."""
    n = transition.substrate.size

    if direction is Direction.BIDIRECTIONAL:
        yielded = set()
        for partition in chain(
            _get_partitions(transition, Direction.CAUSE),
            _get_partitions(transition, Direction.EFFECT),
        ):
            cm = utils.np_hashable(partition.cut_matrix(n))
            if cm not in yielded:
                yielded.add(cm)
                yield partition

    else:
        mechanism = transition.mechanism_indices(direction)
        purview = transition.purview_indices(direction)
        for inner_partition in mechanism_partitions(
            mechanism, purview, transition.node_labels
        ):
            yield DirectedJointPartition(
                direction, inner_partition, transition.node_labels
            )


def _sia(
    transition,
    direction=Direction.BIDIRECTIONAL,
    *,
    alpha_measure=None,
    partitioned_repertoire_scheme=None,
    **kwargs,
):
    """Return the minimal information partition of a transition in a specific
    direction.

    Args:
        transition (Transition): The candidate system.

    Returns:
        AcSystemIrreducibilityAnalysis: A nested structure containing all the
        data from the intermediate calculations. The top level contains the
        basic irreducibility information for the given system.
    """
    validate.direction(direction, allow_bi=True)
    log.info("Calculating big-alpha for %s...", transition)

    if not transition:
        log.info("Transition %s is empty; returning null SIA immediately.", transition)
        return _null_ac_sia(transition, direction)

    if not connectivity.is_weak(transition.substrate.cm, transition.node_indices):
        log.info(
            "%s is not strongly/weakly connected; returning null SIA immediately.",
            transition,
        )
        return _null_ac_sia(transition, direction)

    log.debug("Finding unpartitioned account...")
    unpartitioned_account = _account(
        transition,
        direction,
        alpha_measure=alpha_measure,
        partitioned_repertoire_scheme=partitioned_repertoire_scheme,
    )
    log.debug("Found unpartitioned account.")

    if not unpartitioned_account:
        log.info("Empty unpartitioned account; returning null AC SIA immediately.")
        return _null_ac_sia(transition, direction)

    cuts = _get_partitions(transition, direction)

    parallel_kwargs = conf.parallel_kwargs(
        dict(config.infrastructure.parallel_partition_evaluation), **kwargs
    )
    result = MapReduce(
        _evaluate_partition,
        cuts,
        map_kwargs={
            "transition": transition,
            "direction": direction,
            "unpartitioned_account": unpartitioned_account,
            "alpha_measure": alpha_measure,
            "partitioned_repertoire_scheme": partitioned_repertoire_scheme,
        },
        reduce_func=min,
        reduce_kwargs={
            "default": _null_ac_sia(transition, direction, alpha=float("inf"))
        },
        shortcircuit_func=utils.is_falsy,
        **parallel_kwargs,
    ).run()
    log.info("Finished calculating big-ac-phi data for %s.", transition)
    log.debug("RESULT: \n%s", result)
    return result
