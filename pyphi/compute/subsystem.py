# compute/subsystem.py
"""Functions for computing subsystem-level properties."""

import functools
import logging

from more_itertools import collapse

from .. import conf, connectivity, utils
from ..conf import config
from ..direction import Direction
from ..metrics.ces import ces_distance
from ..models import (
    CauseEffectStructure,
    Concept,
    Cut,
    KCut,
    SystemIrreducibilityAnalysis,
    _null_sia,
    cmp,
)
from ..partition import mip_partitions, system_partition_types
from ..parallel import MapReduce

# Create a logger for this module.
log = logging.getLogger(__name__)


def ces(
    subsystem,
    mechanisms=None,
    purviews=None,
    cause_purviews=None,
    effect_purviews=None,
    directions=None,
    only_positive_phi=True,
    **kwargs,
):
    """Return the conceptual structure of this subsystem, optionally restricted
    to concepts with the mechanisms and purviews given in keyword arguments.

    If you don't need the full |CauseEffectStructure|, restricting the possible
    mechanisms and purviews can make this function much faster.

    Args:
        subsystem (Subsystem): The subsystem for which to determine the
            |CauseEffectStructure|.

    Keyword Args:
        mechanisms (tuple[tuple[int]]): Restrict possible mechanisms to those
            in this list.
        purviews (tuple[tuple[int]]): Same as in |Subsystem.concept()|.
        cause_purviews (tuple[tuple[int]]): Same as in |Subsystem.concept()|.
        effect_purviews (tuple[tuple[int]]): Same as in |Subsystem.concept()|.
        parallel (bool): Whether to compute concepts in parallel. If ``True``,
            overrides :data:`config.PARALLEL_CONCEPT_EVALUATION`.
        directions (Iterable[Direction]): Restrict possible directions to these.
        only_positive_phi (bool): Whether to only return concepts with positive
            phi.

    Returns:
        CauseEffectStructure: A tuple of every |Concept| in the cause-effect
        structure.
    """
    total = None
    if mechanisms is None:
        mechanisms = utils.powerset(subsystem.node_indices, nonempty=True)
        total = 2 ** len(subsystem.node_indices) - 1
    else:
        try:
            total = len(mechanisms)
        except TypeError:
            pass

    def compute_concept(*args, **kwargs):
        # Don't serialize the subsystem; this is replaced after returning.
        # TODO(4.0) remove when subsystem reference is removed from Concept
        concept = subsystem.concept(*args, **kwargs, progress=False)
        concept.subsystem = None
        return concept

    reduce_func = _only_positive_phi if only_positive_phi else _any_phi
    parallel_kwargs = conf.parallel_kwargs(config.PARALLEL_CONCEPT_EVALUATION, **kwargs)
    concepts = MapReduce(
        compute_concept,
        mechanisms,
        map_kwargs=dict(
            purviews=purviews,
            cause_purviews=cause_purviews,
            effect_purviews=effect_purviews,
            directions=directions,
        ),
        reduce_func=reduce_func,
        desc="Computing concepts",
        total=total,
        **parallel_kwargs,
    ).run()
    # Replace subsystem references
    # TODO(4.0) remove when subsystem reference is removed from Concept
    for concept in concepts:
        concept.subsystem = subsystem
    return CauseEffectStructure(concepts, subsystem=subsystem)


def _only_positive_phi(concepts):
    return list(filter(None, collapse(concepts)))


def _any_phi(concepts):
    return list(collapse(concepts))


def conceptual_info(subsystem, **kwargs):
    """Return the conceptual information for a |Subsystem|.

    This is the distance from the subsystem's |CauseEffectStructure| to the
    null concept.
    """
    ci = ces_distance(
        ces(subsystem, **kwargs), CauseEffectStructure((), subsystem=subsystem)
    )
    return round(ci, config.PRECISION)


def evaluate_cut(cut, uncut_subsystem, unpartitioned_ces, **kwargs):
    """Compute the system irreducibility for a given cut.

    Args:
        uncut_subsystem (Subsystem): The subsystem without the cut applied.
        cut (Cut): The cut to evaluate.
        unpartitioned_ces (CauseEffectStructure): The cause-effect structure of
            the uncut subsystem.

    Returns:
        SystemIrreducibilityAnalysis: The |SystemIrreducibilityAnalysis| for
        that cut.
    """
    log.debug("Evaluating %s...", cut)

    cut_subsystem = uncut_subsystem.apply_cut(cut)

    if config.ASSUME_CUTS_CANNOT_CREATE_NEW_CONCEPTS:
        mechanisms = list(unpartitioned_ces.mechanisms)
    else:
        # Mechanisms can only produce concepts if they were concepts in the
        # original system, or the cut divides the mechanism.
        mechanisms = set(
            list(unpartitioned_ces.mechanisms) + list(cut_subsystem.cut_mechanisms)
        )

    kwargs = {"progress": False, **kwargs}
    partitioned_ces = ces(cut_subsystem, mechanisms, **kwargs)

    log.debug("Finished evaluating %s.", cut)

    phi_ = ces_distance(unpartitioned_ces, partitioned_ces)

    return SystemIrreducibilityAnalysis(
        phi=phi_,
        ces=unpartitioned_ces,
        partitioned_ces=partitioned_ces,
        subsystem=uncut_subsystem,
        cut_subsystem=cut_subsystem,
    )


def sia_partitions(nodes, node_labels=None):
    """Return all |big_phi| cuts for the given nodes.

    Controlled by the :const:`config.SYSTEM_PARTITION_TYPE` option.

    Arguments:
        nodes (tuple[int]): The node indices to partition.

    Keyword Arguments:
        node_labels (NodeLabels): Enables printing the partition with labels.

    Returns:
        list[Cut]: All unidirectional partitions.

    """
    # TODO(4.0 consolidate 3.0 and 4.0 cuts)
    scheme = config.SYSTEM_PARTITION_TYPE
    valid = ["DIRECTED_BI", "DIRECTED_BI_CUT_ONE"]
    if scheme not in valid:
        raise ValueError(
            "IIT 3.0 calculations must use one of the following system "
            f"partition schemes: {valid}; got {scheme}"
        )
    return system_partition_types[config.SYSTEM_PARTITION_TYPE](
        nodes, node_labels=node_labels
    )


def _ces(subsystem, **kwargs):
    """Parallelize the unpartitioned |CauseEffectStructure| if parallelizing
    cuts, since we have free processors because we're not computing any cuts
    yet.
    """
    kwargs = {"parallel": config.PARALLEL_CUT_EVALUATION, **kwargs}
    return ces(subsystem, **kwargs)


def _sia_map_reduce(cuts, subsystem, unpartitioned_ces, **kwargs):
    kwargs = {"parallel": config.PARALLEL_CUT_EVALUATION, **kwargs}
    return MapReduce(
        evaluate_cut,
        cuts,
        map_kwargs=dict(
            uncut_subsystem=subsystem,
            unpartitioned_ces=unpartitioned_ces,
        ),
        reduce_func=min,
        reduce_kwargs=dict(default=_null_sia(subsystem)),
        shortcircuit_func=utils.is_falsy,
        desc="Evaluating cuts",
        **kwargs,
    ).run()


def _sia(subsystem, **kwargs):
    """Return the minimal information partition of a subsystem.

    Args:
        subsystem (Subsystem): The candidate set of nodes.

    Returns:
        SystemIrreducibilityAnalysis: A nested structure containing all the
        data from the intermediate calculations. The top level contains the
        basic irreducibility information for the given subsystem.
    """
    # pylint: disable=unused-argument

    log.info("Calculating big-phi data for %s...", subsystem)

    # Check for degenerate cases
    # =========================================================================
    # Phi is necessarily zero if the subsystem is:
    #   - not strongly connected;
    #   - empty;
    #   - an elementary micro mechanism (i.e. no nontrivial bipartitions).
    # So in those cases we immediately return a null SIA.
    if not subsystem:
        log.info("Subsystem %s is empty; returning null SIA " "immediately.", subsystem)
        return _null_sia(subsystem)

    if not connectivity.is_strong(subsystem.cm, subsystem.node_indices):
        log.info(
            "%s is not strongly connected; returning null SIA " "immediately.",
            subsystem,
        )
        return _null_sia(subsystem)

    # Handle elementary micro mechanism cases.
    # Single macro element systems have nontrivial bipartitions because their
    #   bipartitions are over their micro elements.
    if len(subsystem.cut_indices) == 1:
        # If the node lacks a self-loop, phi is trivially zero.
        if not subsystem.cm[subsystem.node_indices][subsystem.node_indices]:
            log.info(
                "Single micro nodes %s without selfloops cannot have "
                "phi; returning null SIA immediately.",
                subsystem,
            )
            return _null_sia(subsystem)
        # Even if the node has a self-loop, we may still define phi to be zero.
        elif not config.SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI:
            log.info(
                "Single micro nodes %s with selfloops cannot have "
                "phi; returning null SIA immediately.",
                subsystem,
            )
            return _null_sia(subsystem)
    # =========================================================================

    log.debug("Finding unpartitioned CauseEffectStructure...")
    unpartitioned_ces = _ces(subsystem, progress=kwargs.get("progress"))

    if not unpartitioned_ces:
        log.info(
            "Empty unpartitioned CauseEffectStructure; returning null "
            "SIA immediately."
        )
        # Short-circuit if there are no concepts in the unpartitioned CES.
        return _null_sia(subsystem)

    log.debug("Found unpartitioned CauseEffectStructure.")

    # TODO: move this into sia_bipartitions?
    # Only True if SINGLE_MICRO_NODES...=True, no?
    if len(subsystem.cut_indices) == 1:
        cuts = [
            Cut(subsystem.cut_indices, subsystem.cut_indices, subsystem.cut_node_labels)
        ]
    else:
        cuts = sia_partitions(subsystem.cut_indices, subsystem.cut_node_labels)

    # TODO(4.0): parallel: expose options
    result = _sia_map_reduce(cuts, subsystem, unpartitioned_ces, **kwargs)

    if config.CLEAR_SUBSYSTEM_CACHES_AFTER_COMPUTING_SIA:
        log.debug("Clearing subsystem caches.")
        subsystem.clear_caches()

    log.info("Finished calculating big-phi data for %s.", subsystem)

    return result


@functools.wraps(_sia)
def sia(subsystem, **kwargs):
    if config.SYSTEM_CUTS == "CONCEPT_STYLE":
        return sia_concept_style(subsystem, **kwargs)
    return _sia(subsystem, **kwargs)


def phi(subsystem):
    """Return the |big_phi| value of a subsystem."""
    return sia(subsystem).phi


class ConceptStyleSystem:
    """A functional replacement for ``Subsystem`` implementing concept-style
    system cuts.
    """

    def __init__(self, subsystem, direction, cut=None):
        self.subsystem = subsystem
        self.direction = direction
        self.cut = cut
        self.cut_system = subsystem.apply_cut(cut)

    def apply_cut(self, cut):
        return ConceptStyleSystem(self.subsystem, self.direction, cut)

    def __getattr__(self, name):
        """Pass attribute access through to the basic subsystem."""
        # Unpickling calls `__getattr__` before the object's dict is populated;
        # check that `subsystem` exists to avoid a recursion error.
        # See https://bugs.python.org/issue5370.
        if "subsystem" in self.__dict__:
            return getattr(self.subsystem, name)
        raise AttributeError(name)

    def __len__(self):
        return len(self.subsystem)

    @property
    def cause_system(self):
        return {Direction.CAUSE: self.cut_system, Direction.EFFECT: self.subsystem}[
            self.direction
        ]

    @property
    def effect_system(self):
        return {Direction.CAUSE: self.subsystem, Direction.EFFECT: self.cut_system}[
            self.direction
        ]

    def concept(
        self, mechanism, purviews=False, cause_purviews=False, effect_purviews=False
    ):
        """Compute a concept, using the appropriate system for each side of the
        cut.
        """
        cause = self.cause_system.mic(mechanism, purviews=(cause_purviews or purviews))

        effect = self.effect_system.mie(
            mechanism, purviews=(effect_purviews or purviews)
        )

        return Concept(
            mechanism=mechanism,
            cause=cause,
            effect=effect,
        )

    def __str__(self):
        return "ConceptStyleSystem{}".format(self.node_indices)


def concept_cuts(direction, node_indices, node_labels=None):
    """Generator over all concept-syle cuts for these nodes."""
    for partition in mip_partitions(node_indices, node_indices):
        yield KCut(direction, partition, node_labels)


def directional_sia(subsystem, direction, unpartitioned_ces=None, **kwargs):
    """Calculate a concept-style SystemIrreducibilityAnalysisCause or
    SystemIrreducibilityAnalysisEffect.
    """
    if unpartitioned_ces is None:
        unpartitioned_ces = _ces(subsystem)

    c_system = ConceptStyleSystem(subsystem, direction)
    cuts = concept_cuts(direction, c_system.cut_indices, subsystem.node_labels)

    return _sia_map_reduce(cuts, c_system, unpartitioned_ces, **kwargs)


# TODO: only return the minimal SIA, instead of both
class SystemIrreducibilityAnalysisConceptStyle(cmp.Orderable):
    """Represents a |SIA| computed using concept-style system cuts."""

    def __init__(self, sia_cause, sia_effect):
        self.sia_cause = sia_cause
        self.sia_effect = sia_effect

    @property
    def min_sia(self):
        return min(self.sia_cause, self.sia_effect, key=lambda m: m.phi)

    def __getattr__(self, name):
        """Pass attribute access through to the minimal SIA."""
        if "sia_cause" in self.__dict__ and "sia_effect" in self.__dict__:
            return getattr(self.min_sia, name)
        raise AttributeError(name)

    def __eq__(self, other):
        return cmp.general_eq(self, other, ["phi"])

    unorderable_unless_eq = ["network"]

    def order_by(self):
        return [self.phi, len(self.subsystem)]

    def __repr__(self):
        return repr(self.min_sia)

    def __str__(self):
        return str(self.min_sia)


# TODO: cache
def sia_concept_style(subsystem):
    """Compute a concept-style SystemIrreducibilityAnalysis"""
    unpartitioned_ces = _ces(subsystem)

    sia_cause = directional_sia(subsystem, Direction.CAUSE, unpartitioned_ces)
    sia_effect = directional_sia(subsystem, Direction.EFFECT, unpartitioned_ces)

    return SystemIrreducibilityAnalysisConceptStyle(sia_cause, sia_effect)
