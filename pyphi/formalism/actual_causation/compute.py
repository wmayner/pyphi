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
from pyphi import numerics
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
from pyphi.models import NullResultReason
from pyphi.models import _null_ac_ria
from pyphi.models import _null_ac_sia
from pyphi.parallel import map_reduce
from pyphi.partition import partition_types
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
    """

    desc = "α-aggregation rules"


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

    # Key on the underlying System so these repertoires share the id()-keyed
    # kernel cache with the unpartitioned path (which delegates through
    # ``TransitionSystem.__getattr__`` to the same ``_underlying_system``).
    underlying_system = transition_system._underlying_system
    repertoires = [
        ra.repertoire(underlying_system, direction, part.mechanism, part.purview)
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

    The result is rounded to ``config.numerics.precision`` decimal places. It
    is signed rather than a metric distance: the default ``PMI`` measure is the
    pointwise mutual information ``log2(p / q)``, which is negative when ``p <
    q``.

    Parameters
    ----------
    p : float
        The first probability.
    q : float
        The second probability.
    measure : str, optional
        Measure name registered in
        :data:`pyphi.measures.distribution.actual_causation_measures`. Mutually
        exclusive with ``alpha_measure``.
    alpha_measure : DistributionMeasure, optional
        Resolved measure callable (e.g., from
        :func:`pyphi.measures.distribution.resolve_actual_causation_measure`).
        Internal callers thread the resolved object through to avoid repeated
        registry lookups; external callers may pass ``measure`` instead. If both
        are ``None``, the active configuration's ``alpha_measure`` is resolved.

    Returns
    -------
    float
        The probability distance between ``p`` and ``q``.

    Raises
    ------
    ValueError
        If both ``measure`` and ``alpha_measure`` are given.
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
    """Return the distance between two accounts.

    Defined as the difference in total α: ``sum(α over A1) - sum(α over A2)``.
    This is signed, and for the unpartitioned-minus-partitioned pairing used in
    the AC system analysis it gives the account's big-α (𝒜).

    Parameters
    ----------
    A1 : Account
        The first account.
    A2 : Account
        The second account.

    Returns
    -------
    float
        The distance between the two accounts.
    """
    return sum([action.alpha for action in A1]) - sum([action.alpha for action in A2])


def _ac_mechanism_partitions(mechanism, purview, node_labels=None):
    """Yield mechanism partitions under the actual-causation partition scheme.

    Resolves ``config.formalism.actual_causation.mechanism_partition_scheme``
    through the partition-scheme registry at call time, so actual-causation
    partitioning is governed by the AC formalism and never by the IIT
    ``mechanism_partition_scheme`` field. The default ``JOINT_PARTITION_ALL`` is
    the partition family of Albantakis et al. (2019), Eq. 7 and Fig. 3B: all
    partitions of the occurrence, excluding the m=1 non-full-cut cases the paper
    forbids for first-order occurrences.
    """
    scheme = config.formalism.actual_causation.mechanism_partition_scheme
    return partition_types[scheme](mechanism, purview, node_labels)


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

    Evaluates every mechanism partition, scores each by α (the
    ``alpha_measure`` distance between the unpartitioned and partitioned
    probabilities), and returns the partition of minimum α. The search
    short-circuits and returns a null analysis on the first partition against
    which the mechanism is reducible (α is zero, or negative when
    ``allow_neg`` is false), since the minimum α cannot then fall lower.

    Parameters
    ----------
    direction : Direction
        The temporal direction, ``Direction.CAUSE`` or ``Direction.EFFECT``.
    mechanism : tuple[int]
        A mechanism.
    purview : tuple[int]
        A purview.
    allow_neg : bool
        If true, α is allowed to be negative. Otherwise, negative values of α
        are treated as if they were zero.
    alpha_measure : DistributionMeasure, optional
        Resolved alpha measure callable. When ``None``,
        ``config.formalism.actual_causation.alpha_measure`` is resolved at the
        call boundary.
    partitioned_repertoire_scheme : optional
        Resolved partitioned-repertoire scheme callable. When ``None``,
        ``config.formalism.actual_causation.partitioned_repertoire_scheme`` is
        resolved at the call boundary.

    Returns
    -------
    AcRepertoireIrreducibilityAnalysis
        The irreducibility analysis for the mechanism.
    """
    if not purview:
        return _null_ac_ria(
            transition.mechanism_state(direction),
            direction,
            mechanism,
            purview,
            reasons=[NullResultReason.EMPTY_PURVIEW],
        )

    probability = transition.probability(direction, mechanism, purview)
    candidates: list[AcRepertoireIrreducibilityAnalysis] = []
    for partition in _ac_mechanism_partitions(
        mechanism, purview, transition.node_labels
    ):
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
        if numerics.eq(alpha, 0) or (alpha < 0 and not allow_neg):
            return _null_ac_ria(
                transition.mechanism_state(direction),
                direction,
                mechanism,
                purview,
                partition,
                reasons=[NullResultReason.REDUCIBLE_OVER_PARTITION],
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
    # Record only the |α|-cluster around the winning minimum; the cascade's
    # tied_set carries every candidate entering the resolving level.
    abs_alphas = [abs(r.alpha) for r in candidates]
    alpha_ties = resolve_ties._tied_with_extremum(
        candidates, abs_alphas, min(abs_alphas)
    )
    if winner is not None and len(alpha_ties) > 1:
        winner.set_partition_ties(alpha_ties)
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

    Runs :func:`_find_mip` over every candidate purview and returns the causal
    link of maximum α, resolving ties via :mod:`pyphi.resolve_ties`. The
    returned :class:`~pyphi.models.actual_causation.CausalLink` records the tied
    purviews as its extended purview.

    Parameters
    ----------
    direction : Direction
        The temporal direction, specifying cause or effect.
    mechanism : tuple[int]
        The mechanism to be tested for irreducibility.
    purviews : tuple[int], optional
        Restrict the possible purviews to a subset of the system. This is
        useful, for example, for finding only causal links that are "about" a
        certain subset of nodes.
    allow_neg : bool
        If true, α is allowed to be negative. Otherwise, negative values of α
        are treated as if they were zero.
    alpha_measure : DistributionMeasure, optional
        Resolved alpha measure callable. When ``None``,
        ``config.formalism.actual_causation.alpha_measure`` is resolved at the
        call boundary.
    partitioned_repertoire_scheme : optional
        Resolved partitioned-repertoire scheme callable. When ``None``,
        ``config.formalism.actual_causation.partitioned_repertoire_scheme`` is
        resolved at the call boundary.

    Returns
    -------
    CausalLink
        The maximally-irreducible actual cause or effect. An empty list is
        returned when no purview yields a positive α.
    """
    purviews = transition.potential_purviews(direction, mechanism, purviews)

    # Find the maximal RIA over the remaining purviews.
    if not purviews:
        max_ria = _null_ac_ria(
            transition.mechanism_state(direction),
            direction,
            mechanism,
            None,
            reasons=[NullResultReason.NO_PURVIEWS],
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
    # Filter out None values; if no candidate has alpha > 0, the mechanism
    # specifies no causal link, reported as a null link (matching the
    # no-purviews path above rather than a bare empty list).
    valid_ria = [ria for ria in all_ria if ria is not None and bool(ria)]
    if not valid_ria:
        return CausalLink(
            _null_ac_ria(
                transition.mechanism_state(direction),
                direction,
                mechanism,
                None,
                reasons=[NullResultReason.NO_POSITIVE_ALPHA],
            )
        )
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
    """Return the set of all
    :class:`~pyphi.models.actual_causation.CausalLink` of the specified
    direction.

    One causal link is found per candidate mechanism (the whole non-empty
    powerset of mechanism indices when ``mechanisms`` is ``None``); links with
    zero α are dropped from the returned
    :class:`~pyphi.models.DirectedAccount`.

    Parameters
    ----------
    alpha_measure : DistributionMeasure, optional
        Resolved alpha measure callable. When ``None``,
        ``config.formalism.actual_causation.alpha_measure`` is resolved at the
        call boundary.
    partitioned_repertoire_scheme : optional
        Resolved partitioned-repertoire scheme callable. When ``None``, the
        active ``config.formalism.actual_causation.partitioned_repertoire_scheme``
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
    """Return the set of all causal links for a
    :class:`~pyphi.actual.Transition`.

    A bidirectional account (the default) is the union of the cause-side and
    effect-side directed accounts, returned as an
    :class:`~pyphi.models.Account`; a directional call returns the single
    :class:`~pyphi.models.DirectedAccount`.

    Parameters
    ----------
    transition : Transition
        The transition of interest.
    direction : Direction
        By default (``Direction.BIDIRECTIONAL``) the account contains actual
        causes and actual effects.
    alpha_measure : DistributionMeasure, optional
        Resolved alpha measure callable. When ``None``,
        ``config.formalism.actual_causation.alpha_measure`` is resolved at the
        call boundary.
    partitioned_repertoire_scheme : optional
        Resolved partitioned-repertoire scheme callable. When ``None``, the
        active ``config.formalism.actual_causation.partitioned_repertoire_scheme``
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
    """Find the system irreducibility analysis for a given partition.

    Returns the :class:`~pyphi.models.AcSystemIrreducibilityAnalysis` for the
    transition under ``partition``.
    """
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
        alpha=numerics.round_to_precision(alpha),
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
        for inner_partition in _ac_mechanism_partitions(
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

    Computes the unpartitioned account, then evaluates every system partition
    and selects the one of minimum big-α (𝒜), the account distance between the
    unpartitioned and partitioned accounts. A null analysis is returned
    immediately when the transition is empty, is not at least weakly connected,
    or has an empty unpartitioned account.

    Parameters
    ----------
    transition : Transition
        The candidate system.

    Returns
    -------
    AcSystemIrreducibilityAnalysis
        A nested structure containing all the data from the intermediate
        calculations. The top level contains the basic irreducibility
        information for the given system.
    """
    validate.direction(direction, allow_bi=True)
    log.info("Calculating big-alpha for %s...", transition)

    if not transition:
        log.info("Transition %s is empty; returning null SIA immediately.", transition)
        return _null_ac_sia(transition, direction, reasons=[NullResultReason.NO_SYSTEM])

    if not connectivity.is_weak(transition.substrate.cm, transition.node_indices):
        log.info(
            "%s is not strongly/weakly connected; returning null SIA immediately.",
            transition,
        )
        return _null_ac_sia(
            transition, direction, reasons=[NullResultReason.NO_WEAK_CONNECTIVITY]
        )

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
        return _null_ac_sia(
            transition,
            direction,
            reasons=[NullResultReason.EMPTY_CAUSE_EFFECT_STRUCTURE],
        )

    cuts = _get_partitions(transition, direction)

    parallel_kwargs = conf.parallel_kwargs(
        dict(config.infrastructure.parallel_partition_evaluation), **kwargs
    )
    candidates = map_reduce(
        _evaluate_partition,
        cuts,
        map_kwargs={
            "transition": transition,
            "direction": direction,
            "unpartitioned_account": unpartitioned_account,
            "alpha_measure": alpha_measure,
            "partitioned_repertoire_scheme": partitioned_repertoire_scheme,
        },
        shortcircuit_func=utils.is_falsy,
        **parallel_kwargs,
    )
    if not candidates:
        log.info("No partitions to evaluate; returning null AC SIA.")
        return _null_ac_sia(
            transition, direction, reasons=[NullResultReason.NO_VALID_PARTITIONS]
        )
    context = resolve_ties.ResolutionContext(max_escalation_level="Determinism")
    outcome = resolve_ties.resolve_ac_sia_tie(candidates, context=context)
    result = outcome.resolved
    assert result is not None, "AC SIA cascade returned no winner"
    # Record only the α-cluster around the winning minimum; the cascade's
    # tied_set carries every candidate entering the resolving level.
    alphas = [c.alpha for c in candidates]
    alpha_ties = resolve_ties._tied_with_extremum(candidates, alphas, min(alphas))
    if len(alpha_ties) > 1:
        result.set_ties(alpha_ties)
    log.info("Finished calculating big-ac-phi data for %s.", transition)
    log.debug("RESULT: \n%s", result)
    return result
