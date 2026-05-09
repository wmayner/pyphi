# formalism/iit3/__init__.py
"""IIT 3.0 (Oizumi et al. 2014) algorithms.

Distribution-distance-based phi computation. Partition scheme: bipartitions
(``BI``). Compatible metrics: ``EMD``, ``L1``, ``KLD``, ``ENTROPY_DIFFERENCE``,
``PSQ2``, ``MP2Q``, ``ABSOLUTE_INTRINSIC_DIFFERENCE``,
``INTRINSIC_DIFFERENCE``.

The dispatch class :class:`IIT3Formalism` lives in :mod:`.formalism` and
calls back into this module's ``sia`` / ``ces`` / etc. for the actual
computation.
"""

from __future__ import annotations

import contextlib
import functools
import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from more_itertools import collapse

from pyphi import conf
from pyphi import connectivity
from pyphi import utils
from pyphi.conf import config
from pyphi.direction import Direction
from pyphi.metrics.ces import ces_distance
from pyphi.models import CauseEffectStructure
from pyphi.models import Concept
from pyphi.models import KCut
from pyphi.models import SystemIrreducibilityAnalysis
from pyphi.models import SystemPartition
from pyphi.models import _null_sia
from pyphi.models import cmp
from pyphi.parallel import MapReduce
from pyphi.partition import mip_partitions
from pyphi.partition import system_partition_types
from pyphi.types import Mechanism
from pyphi.types import Purview

if TYPE_CHECKING:
    from pyphi.labels import NodeLabels
    from pyphi.system import System

# Create a logger for this module.
log = logging.getLogger(__name__)


def ces(
    system: System,
    mechanisms: Iterable[Mechanism] | None = None,
    purviews: Iterable[Purview] | None = None,
    cause_purviews: Iterable[Purview] | None = None,
    effect_purviews: Iterable[Purview] | None = None,
    directions: Iterable[Direction] | None = None,
    only_positive_phi: bool = True,
    **kwargs: Any,
) -> CauseEffectStructure:
    """Return the conceptual structure of this system, optionally restricted
    to concepts with the mechanisms and purviews given in keyword arguments.

    If you don't need the full |CauseEffectStructure|, restricting the possible
    mechanisms and purviews can make this function much faster.

    Args:
        system (System): The system for which to determine the
            |CauseEffectStructure|.

    Keyword Args:
        mechanisms (tuple[tuple[int]]): Restrict possible mechanisms to those
            in this list.
        purviews (tuple[tuple[int]]): Same as in |System.concept()|.
        cause_purviews (tuple[tuple[int]]): Same as in |System.concept()|.
        effect_purviews (tuple[tuple[int]]): Same as in |System.concept()|.
        parallel (bool): Whether to compute concepts in parallel. If ``True``,
            overrides :data:`config.infrastructure.parallel_concept_evaluation`.
        directions (Iterable[Direction]): Restrict possible directions to these.
        only_positive_phi (bool): Whether to only return concepts with positive
            phi.

    Returns:
        CauseEffectStructure: A tuple of every |Concept| in the cause-effect
        structure.
    """
    total = None
    if mechanisms is None:
        mechanisms = utils.powerset(system.node_indices, nonempty=True)
        total = 2 ** len(system.node_indices) - 1
    else:
        with contextlib.suppress(TypeError):
            total = len(mechanisms)  # type: ignore[arg-type]  # mechanisms may be generator

    from pyphi.formalism.queries import concept as _concept

    def compute_concept(*args, **kwargs):
        return _concept(system, *args, **kwargs, progress=False)

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
    return CauseEffectStructure(concepts)


def _only_positive_phi(concepts: Iterable[Any]) -> list[Concept]:
    return list(filter(None, collapse(concepts)))


def _any_phi(concepts: Iterable[Any]) -> list[Concept]:
    return list(collapse(concepts))


def conceptual_info(system: System, **kwargs: Any) -> float:
    """Return the conceptual information for a |System|.

    This is the distance from the system's |CauseEffectStructure| to the
    null concept.
    """
    ci = ces_distance(ces(system, **kwargs), CauseEffectStructure(()))
    return round(ci, config.numerics.precision)  # type: ignore[arg-type]  # config.Option descriptor


def evaluate_cut(
    cut: SystemPartition,
    uncut_system: System,
    unpartitioned_ces: CauseEffectStructure,
    **kwargs: Any,
) -> SystemIrreducibilityAnalysis:
    """Compute the system irreducibility for a given cut.

    Args:
        uncut_system (System): The system without the cut applied.
        cut (SystemPartition): The cut to evaluate.
        unpartitioned_ces (CauseEffectStructure): The cause-effect structure of
            the uncut system.

    Returns:
        SystemIrreducibilityAnalysis: The |SystemIrreducibilityAnalysis| for
        that cut.
    """
    log.debug("Evaluating %s...", cut)

    cut_system = uncut_system.apply_cut(cut)

    if config.formalism.assume_cuts_cannot_create_new_concepts:
        mechanisms = list(unpartitioned_ces.mechanisms)
    else:
        # Mechanisms can only produce concepts if they were concepts in the
        # original system, or the cut divides the mechanism.
        mechanisms = set(
            list(unpartitioned_ces.mechanisms) + list(cut_system.cut_mechanisms)
        )

    kwargs = {"progress": False, **kwargs}
    partitioned_ces = ces(cut_system, mechanisms, **kwargs)

    log.debug("Finished evaluating %s.", cut)

    phi_ = ces_distance(unpartitioned_ces, partitioned_ces)

    return SystemIrreducibilityAnalysis(
        phi=phi_,
        ces=unpartitioned_ces,
        partitioned_ces=partitioned_ces,
        system=uncut_system,
        cut_system=cut_system,
    )


def sia_partitions(
    nodes: tuple[int, ...], node_labels: NodeLabels | None = None
) -> list[SystemPartition]:
    """Return all |big_phi| cuts for the given nodes.

    Controlled by the :const:`config.formalism.system_partition_type` option.

    Arguments:
        nodes (tuple[int]): The node indices to partition.

    Keyword Arguments:
        node_labels (NodeLabels): Enables printing the partition with labels.

    Returns:
        list[SystemPartition]: All unidirectional partitions.

    """
    # TODO(4.0 consolidate 3.0 and 4.0 cuts)
    scheme = config.formalism.system_partition_type
    valid = ["DIRECTED_BI", "DIRECTED_BI_CUT_ONE"]
    if scheme not in valid:
        raise ValueError(
            "IIT 3.0 calculations must use one of the following system "
            f"partition schemes: {valid}; got {scheme}"
        )
    return system_partition_types[config.formalism.system_partition_type](  # type: ignore[index]  # config.Option descriptor
        nodes, node_labels=node_labels
    )


def _ces(system: System, **kwargs: Any) -> CauseEffectStructure:
    """Parallelize the unpartitioned |CauseEffectStructure| if parallelizing
    cuts, since we have free processors because we're not computing any cuts
    yet.
    """
    kwargs = {"parallel": config.infrastructure.parallel_cut_evaluation, **kwargs}
    return ces(system, **kwargs)


def _sia_map_reduce(
    cuts: Iterable[SystemPartition],
    system: System,
    unpartitioned_ces: CauseEffectStructure,
    **kwargs: Any,
) -> SystemIrreducibilityAnalysis:
    kwargs = {"parallel": config.infrastructure.parallel_cut_evaluation, **kwargs}
    result = MapReduce(
        evaluate_cut,
        cuts,
        map_kwargs={
            "uncut_system": system,
            "unpartitioned_ces": unpartitioned_ces,
        },
        reduce_func=min,
        reduce_kwargs={"default": _null_sia(system)},
        shortcircuit_func=utils.is_falsy,
        desc="Evaluating cuts",
        **kwargs,
    ).run()
    assert result is not None
    return result


def _sia(system: System, **kwargs: Any) -> SystemIrreducibilityAnalysis:
    """Return the minimal information partition of a system.

    Args:
        system (System): The candidate set of nodes.

    Returns:
        SystemIrreducibilityAnalysis: A nested structure containing all the
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
        return _null_sia(system)

    if not connectivity.is_strong(system.cm, system.node_indices):
        log.info(
            "%s is not strongly connected; returning null SIA immediately.",
            system,
        )
        return _null_sia(system)

    # Handle elementary micro mechanism cases.
    # Single macro element systems have nontrivial bipartitions because their
    #   bipartitions are over their micro elements.
    if len(system.cut_indices) == 1:
        # If the node lacks a self-loop, phi is trivially zero.
        if not system.cm[system.node_indices][system.node_indices]:
            log.info(
                "Single micro nodes %s without selfloops cannot have "
                "phi; returning null SIA immediately.",
                system,
            )
            return _null_sia(system)
        # Even if the node has a self-loop, we may still define phi to be zero.
        if not config.formalism.single_micro_nodes_with_selfloops_have_phi:
            log.info(
                "Single micro nodes %s with selfloops cannot have "
                "phi; returning null SIA immediately.",
                system,
            )
            return _null_sia(system)
    # =========================================================================

    log.debug("Finding unpartitioned CauseEffectStructure...")
    unpartitioned_ces = _ces(system, progress=kwargs.get("progress"))

    if not unpartitioned_ces:
        log.info(
            "Empty unpartitioned CauseEffectStructure; returning null SIA immediately."
        )
        # Short-circuit if there are no concepts in the unpartitioned CES.
        return _null_sia(system)

    log.debug("Found unpartitioned CauseEffectStructure.")

    # TODO: move this into sia_bipartitions?
    # Only True if SINGLE_MICRO_NODES...=True, no?
    if len(system.cut_indices) == 1:
        cuts = [
            SystemPartition(
                Direction.EFFECT,
                system.cut_indices,
                system.cut_indices,
                system.cut_node_labels,
            )
        ]
    else:
        cuts = sia_partitions(system.cut_indices, system.cut_node_labels)

    result = _sia_map_reduce(cuts, system, unpartitioned_ces, **kwargs)

    if config.infrastructure.clear_system_caches_after_computing_sia:
        log.debug("Clearing system caches.")
        system.clear_caches()

    log.info("Finished calculating big-phi data for %s.", system)

    return result


@functools.wraps(_sia)
def sia(
    system: System, **kwargs: Any
) -> SystemIrreducibilityAnalysis | SystemIrreducibilityAnalysisConceptStyle:
    if config.formalism.system_cuts == "CONCEPT_STYLE":
        return sia_concept_style(system, **kwargs)
    return _sia(system, **kwargs)


def phi(system: System) -> float:
    """Return the |big_phi| value of a system."""
    return sia(system).phi


class ConceptStyleSystem:
    """A functional replacement for ``System`` implementing concept-style
    system cuts.
    """

    def __init__(
        self,
        system: System,
        direction: Direction,
        cut: SystemPartition | None = None,
    ) -> None:
        self.system = system
        self.direction = direction
        self.cut = cut
        if cut is not None:
            self.cut_system = system.apply_cut(cut)
        else:
            self.cut_system = system

    def apply_cut(self, cut: SystemPartition) -> ConceptStyleSystem:
        return ConceptStyleSystem(self.system, self.direction, cut)

    def __getattr__(self, name: str) -> Any:
        """Pass attribute access through to the basic system."""
        # Unpickling calls `__getattr__` before the object's dict is populated;
        # check that `system` exists to avoid a recursion error.
        # See https://bugs.python.org/issue5370.
        if "system" in self.__dict__:
            return getattr(self.system, name)
        raise AttributeError(name)

    def __len__(self) -> int:
        return len(self.system)

    @property
    def cause_system(self) -> System:
        return {
            Direction.CAUSE: self.cut_system,
            Direction.EFFECT: self.system,
        }[self.direction]

    @property
    def effect_system(self) -> System:
        return {
            Direction.CAUSE: self.system,
            Direction.EFFECT: self.cut_system,
        }[self.direction]

    def concept(
        self,
        mechanism: Mechanism,
        purviews: Iterable[Purview] | bool = False,
        cause_purviews: Iterable[Purview] | bool = False,
        effect_purviews: Iterable[Purview] | bool = False,
    ) -> Concept:
        """Compute a concept, using the appropriate system for each side of the
        cut.
        """
        # Convert bool to None for purviews parameters
        cause_p: Iterable[Purview] | None = cause_purviews or purviews or None  # type: ignore[assignment]
        effect_p: Iterable[Purview] | None = effect_purviews or purviews or None  # type: ignore[assignment]

        from pyphi.formalism.queries import mic
        from pyphi.formalism.queries import mie

        cause = mic(self.cause_system, mechanism, purviews=cause_p)
        effect = mie(self.effect_system, mechanism, purviews=effect_p)

        return Concept(
            mechanism=mechanism,
            cause=cause,
            effect=effect,
        )

    def __str__(self) -> str:
        return f"ConceptStyleSystem{self.node_indices}"


def concept_cuts(
    direction: Direction,
    node_indices: tuple[int, ...],
    node_labels: NodeLabels | None = None,
) -> Iterable[KCut]:
    """Generator over all concept-syle cuts for these nodes."""
    for partition in mip_partitions(node_indices, node_indices):
        yield KCut(direction, partition, node_labels)


def directional_sia(
    system: System,
    direction: Direction,
    unpartitioned_ces: CauseEffectStructure | None = None,
    **kwargs: Any,
) -> SystemIrreducibilityAnalysis:
    """Calculate a concept-style SystemIrreducibilityAnalysisCause or
    SystemIrreducibilityAnalysisEffect.
    """
    if unpartitioned_ces is None:
        unpartitioned_ces = _ces(system)

    c_system = ConceptStyleSystem(system, direction)
    cuts = concept_cuts(direction, c_system.cut_indices, system.node_labels)

    # Type ignore: ConceptStyleSystem duck-types as System, KCut as SystemPartition
    return _sia_map_reduce(cuts, c_system, unpartitioned_ces, **kwargs)  # type: ignore[arg-type]


# TODO: only return the minimal SIA, instead of both
class SystemIrreducibilityAnalysisConceptStyle(cmp.Orderable):
    """Represents a |SIA| computed using concept-style system cuts."""

    def __init__(
        self,
        sia_cause: SystemIrreducibilityAnalysis,
        sia_effect: SystemIrreducibilityAnalysis,
    ) -> None:
        self.sia_cause = sia_cause
        self.sia_effect = sia_effect

    @property
    def min_sia(self) -> SystemIrreducibilityAnalysis:
        return min(self.sia_cause, self.sia_effect, key=lambda m: m.phi)

    def __getattr__(self, name: str) -> Any:
        """Pass attribute access through to the minimal SIA."""
        if "sia_cause" in self.__dict__ and "sia_effect" in self.__dict__:
            return getattr(self.min_sia, name)
        raise AttributeError(name)

    def __eq__(self, other: object) -> bool:
        return cmp.general_eq(self, other, ["phi"])

    unorderable_unless_eq: ClassVar[list[str]] = ["substrate"]

    def order_by(self) -> list[Any]:
        return [self.phi, len(self.system)]

    def __repr__(self) -> str:
        return repr(self.min_sia)

    def __str__(self) -> str:
        return str(self.min_sia)


# TODO: cache
def sia_concept_style(
    system: System,
) -> SystemIrreducibilityAnalysisConceptStyle:
    """Compute a concept-style SystemIrreducibilityAnalysis"""
    unpartitioned_ces = _ces(system)

    sia_cause = directional_sia(system, Direction.CAUSE, unpartitioned_ces)
    sia_effect = directional_sia(system, Direction.EFFECT, unpartitioned_ces)

    return SystemIrreducibilityAnalysisConceptStyle(sia_cause, sia_effect)


# ============================================================================
# Substrate-level complex evaluation (IIT 3.0 path)
# ============================================================================
#
# Mirrors the IIT 4.0 ``all_complexes`` / ``irreducible_complexes`` /
# ``maximal_complex`` in :mod:`pyphi.formalism.iit4`. Iterates over
# :func:`pyphi.substrate.possible_complexes` and dispatches to the IIT 3.0
# ``sia`` defined above.


def all_complexes(
    substrate: Any,
    state: tuple[int, ...],
    parallel_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> list[SystemIrreducibilityAnalysis]:
    """Return all complexes of the substrate under the IIT 3.0 SIA.

    Includes reducible, zero-|big_phi| complexes (which are not, strictly
    speaking, complexes at all).
    """
    from pyphi import conf as _conf
    from pyphi.substrate import possible_complexes

    pkwargs = _conf.parallel_kwargs(
        config.infrastructure.parallel_complex_evaluation,
        **(parallel_kwargs or {}),
    )
    result = MapReduce(
        sia,
        possible_complexes(substrate, state, **kwargs),
        total=2 ** len(substrate) - 1,
        map_kwargs={"progress": False},
        desc="Evaluating complexes",
        **pkwargs,  # type: ignore[arg-type]
    ).run()
    assert result is not None
    return result


def complexes(
    substrate: Any, state: tuple[int, ...], **kwargs: Any
) -> list[SystemIrreducibilityAnalysis]:
    """Return all irreducible complexes of the substrate."""
    return list(filter(None, all_complexes(substrate, state, **kwargs)))


def major_complex(
    substrate: Any, state: tuple[int, ...], **kwargs: Any
) -> SystemIrreducibilityAnalysis:
    """Return the major complex of the substrate."""
    from pyphi import conf as _conf
    from pyphi.substrate import possible_complexes
    from pyphi.system import System as _System

    log.info("Calculating major complex...")
    empty_system = _System.from_substrate(substrate, state, ())
    default = _null_sia(empty_system)
    pkwargs = _conf.parallel_kwargs(
        config.infrastructure.parallel_complex_evaluation, **kwargs
    )
    result = MapReduce(
        sia,
        possible_complexes(substrate, state),
        map_kwargs={"progress": False},
        reduce_func=max,
        reduce_kwargs={"default": default},
        total=2 ** len(substrate) - 1,
        desc="Evaluating complexes",
        **pkwargs,  # type: ignore[arg-type]
    ).run()
    log.info("Finished calculating major complex.")
    assert result is not None
    return result


def condensed(
    substrate: Any, state: tuple[int, ...], **kwargs: Any
) -> list[SystemIrreducibilityAnalysis]:
    """Return a list of maximal non-overlapping complexes."""
    result: list[SystemIrreducibilityAnalysis] = []
    covered_nodes: set[int] = set()

    for c in sorted(complexes(substrate, state, **kwargs), reverse=True):
        if c.system is not None and not any(
            n in covered_nodes for n in c.system.node_indices
        ):
            result.append(c)
            covered_nodes = covered_nodes | set(c.system.node_indices)

    return result
