# -*- coding: utf-8 -*-
# big_phi.py

from collections import UserDict, defaultdict
from dataclasses import dataclass
from itertools import product

import scipy

from . import config, models
from .combinatorics import pairs
from .compute.parallel import MapReduce
from .compute.subsystem import sia_bipartitions as directionless_sia_bipartitions
from .direction import Direction
from .models import fmt
from .models.subsystem import CauseEffectStructure, FlatCauseEffectStructure

# TODO
# - cache relations, compute as needed for each nonconflicting CES

DIRECTIONS = (Direction.CAUSE, Direction.EFFECT)

# TODO
def fmt_cut(cut):
    """Format a |Cut|."""
    return (
        f"Cut {fmt.fmt_mechanism(cut.from_nodes, cut.node_labels)} {fmt.CUT_SYMBOL} "
        f"{fmt.fmt_mechanism(cut.to_nodes, cut.node_labels)} ({str(cut.direction)[0]})"
    )


class Cut(models.cuts.Cut):
    """A system cut.

    Same as a IIT 3.0 unidirectional cut, but with a Direction.
    """

    def __init__(self, direction, *args, **kwargs):
        self.direction = direction
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return fmt_cut(self)


def is_affected_by_cut(distinction, cut):
    coming_from = set(cut.from_nodes) & set(distinction.mechanism)
    going_to = set(cut.to_nodes) & set(distinction.purview(cut.direction))
    return coming_from and going_to


def unaffected_distinctions(ces, cut):
    return CauseEffectStructure(
        [distinction for distinction in ces if not is_affected_by_cut(distinction, cut)]
    )


def unaffected_relations(ces, relations):
    """Return the relations that are not supported by the given CES."""
    # TODO use lattice data structure for efficiently finding the union of the
    # lower sets of lost distinctions
    ces = FlatCauseEffectStructure(ces)
    for relation in relations:
        if all(distinction in ces for distinction in relation.relata):
            yield relation


def sia_partitions(node_indices, node_labels):
    # TODO(4.0) configure
    for cut in directionless_sia_bipartitions(node_indices, node_labels):
        for direction in [Direction.CAUSE, Direction.EFFECT]:
            yield Cut(
                direction, cut.from_nodes, cut.to_nodes, node_labels=cut.node_labels
            )


class SystemIrreducibilityAnalysis(models.subsystem.SystemIrreducibilityAnalysis):
    def __init__(
        self,
        phi=None,
        selectivity=None,
        informativeness=None,
        ces=None,
        relations=None,
        # TODO rename to distinctions?
        partitioned_ces=None,
        partitioned_relations=None,
        subsystem=None,
        cut_subsystem=None,
    ):
        self.phi = phi
        self.selectivity = selectivity
        self.informativeness = informativeness
        self.ces = ces
        self.relations = relations

        # TODO use PhiStructure here
        self.partitioned_ces = partitioned_ces
        self.partitioned_relations = partitioned_relations

        self.subsystem = subsystem
        self.cut_subsystem = cut_subsystem


class PhiStructure:
    def __init__(self, distinctions, relations):
        self.distinctions = distinctions
        self.relations = relations


@dataclass(order=True)
class Informativeness:
    value: float
    partitioned_phi_structure: PhiStructure


def informativeness(cut, phi_structure):
    # TODO use a single pass through the phi structure?
    distinctions = unaffected_distinctions(phi_structure.distinctions, cut)
    distinction_term = sum(phi_structure.distinctions.phis) - sum(distinctions.phis)
    relations = list(unaffected_relations(distinctions, phi_structure.relations))
    relation_term = sum(relation.phi for relation in phi_structure.relations) - sum(
        relation.phi for relation in relations
    )
    return Informativeness(
        value=(distinction_term + relation_term),
        partitioned_phi_structure=PhiStructure(distinctions, relations),
    )


def number_of_possible_relations_with_overlap(n, k):
    """Return the number of possible relations with overlap of size k."""
    return (
        (-1) ** (k - 1)
        * scipy.special.comb(n, k)
        * (2 ** (2 ** (n - k + 1)) - 1 - 2 ** (n - k + 1))
    )


def optimum_sum_small_phi_relations(n):
    """Return the 'best possible' sum of small phi for relations."""
    # \sum_{k=1}^{n} (size of purview) * (number of relations with that purview size)
    return sum(
        k * number_of_possible_relations_with_overlap(n, k) for k in range(1, n + 1)
    )


def optimum_sum_small_phi_distinctions_one_direction(n):
    """Return the 'best possible' sum of small phi for distinctions in one direction"""
    # \sum_{k=1}^{n} k(n choose k)
    return (2 / n) * (2 ** n)


def optimum_sum_small_phi(n):
    """Return the 'best possible' sum of small phi for the system."""
    # Double distinction term for cause & effect sides
    distinction_term = 2 * optimum_sum_small_phi_distinctions_one_direction(n)
    relation_term = optimum_sum_small_phi_relations(n)
    return distinction_term + relation_term


def selectivity(subsystem, phi_structure):
    # TODO memoize and store sums on phi_structure
    # TODO make `Relations` object
    return (
        sum(phi_structure.distinctions.phis)
        + sum(relation.phi for relation in phi_structure.relations)
    ) / optimum_sum_small_phi(len(subsystem))


def phi(selectivity, informativeness):
    return selectivity * informativeness


def all_phi_structures(distinction_sets, all_relations):
    for distinctions in distinction_sets:
        yield PhiStructure(
            distinctions, list(unaffected_relations(distinctions, all_relations))
        )


def evaluate_cut(subsystem, phi_structure, selectivity, cut):
    cut_subsystem = subsystem.apply_cut(cut)
    _informativeness = informativeness(cut, phi_structure)
    _phi = phi(selectivity, _informativeness.value)
    return SystemIrreducibilityAnalysis(
        phi=_phi,
        selectivity=selectivity,
        informativeness=_informativeness.value,
        # TODO use actual phi structure; allow it to work with SIA printing
        ces=phi_structure.distinctions,
        partitioned_ces=_informativeness.partitioned_phi_structure.distinctions,
        relations=phi_structure.relations,
        partitioned_relations=_informativeness.partitioned_phi_structure.relations,
        subsystem=subsystem,
        cut_subsystem=cut_subsystem,
    )


def has_nonspecified_elements(subsystem, distinctions):
    """Return whether any elements are not specified by a purview in both
    directions."""
    elements = set(subsystem.node_indices)
    # TODO use something like `pyphi.Direction.both = [CAUSE, EFFECT]`
    directions = [Direction.CAUSE, Direction.EFFECT]
    specified = {direction: set() for direction in directions}
    for distinction in distinctions:
        for direction in directions:
            specified[direction].update(set(distinction.purview(direction)))
    return any(elements - _specified for _specified in specified.values())


def has_no_spanning_specification(subsystem, distinctions):
    """Return whether the system can be separated into disconnected components.

    Here disconnected means that there is no "spanning specification"; some
    subset of elements only specifies themselves and is not specified by any
    other subset.
    """
    # TODO
    return False


REDUCIBILITY_CHECKS = [
    has_nonspecified_elements,
    has_no_spanning_specification,
]


# TODO
class ComputeSystemIrreducibility(MapReduce):
    """Computation engine for system-level irreducibility."""

    description = "Evaluating {} cuts".format(fmt.BIG_PHI)

    def empty_result(self, subsystem, phi_structure, selectivity):
        """Begin with a |SIA| with infinite |big_phi|; all actual SIAs will have less."""
        return SystemIrreducibilityAnalysis(subsystem=subsystem, phi=float("inf"))

    @staticmethod
    def compute(cut, subsystem, phi_structure, selectivity):
        """Evaluate a cut."""
        return evaluate_cut(subsystem, phi_structure, selectivity, cut)

    def process_result(self, new_sia, min_sia):
        """Check if the new SIA has smaller |big_phi| than the standing result."""
        if new_sia.phi == 0:
            # Short circuit
            self.done = True
            return new_sia

        elif abs(new_sia.phi) < abs(min_sia.phi):
            return new_sia

        return min_sia


def evaluate_phi_structure(phi_structure, subsystem, check_trivial_reducibility=True):
    _selectivity = selectivity(subsystem, phi_structure)
    if check_trivial_reducibility and any(
        check(subsystem, phi_structure.distinctions) for check in REDUCIBILITY_CHECKS
    ):
        return SystemIrreducibilityAnalysis(
            phi=0.0,
            subsystem=subsystem,
            cut_subsystem=subsystem,
            selectivity=_selectivity,
            ces=phi_structure.distinctions,
            relations=phi_structure.relations,
        )
    cuts = sia_partitions(subsystem.cut_indices, subsystem.cut_node_labels)
    return ComputeSystemIrreducibility(
        cuts, subsystem, phi_structure, _selectivity
    ).run(parallel=False)


class ComputeMaximalCompositionalState(MapReduce):
    """Computation engine for resolving conflicts among compositional states."""

    description = "Evaluating compositional states"

    def empty_result(self, subsystem, check_trivial_reducibility):
        """Begin with a |SIA| with negative infinite |big_phi|; all actual SIAs will have more."""
        return SystemIrreducibilityAnalysis(subsystem=subsystem, phi=-float("inf"))

    compute = staticmethod(evaluate_phi_structure)

    def process_result(self, new_sia, max_sia):
        """Check if the new SIA has larger |big_phi| than the standing result."""
        if new_sia.phi > max_sia.phi:
            return new_sia
        return max_sia


class CompositionalState(UserDict):
    """A mapping from purviews to states."""


def is_congruent(distinction, state):
    """Return whether (any of) the (tied) specified state(s) is the given one."""
    return any(state == tuple(specified) for specified in distinction.specified_state)


def filter_ces(ces, direction, compositional_state):
    """Return only the distinctions consistent with the given compositional state."""
    for distinction in ces:
        try:
            if distinction.direction == direction and is_congruent(
                distinction,
                compositional_state[distinction.purview],
            ):
                yield distinction
        except KeyError:
            pass


def _nonconflicting_mice_set(purview_to_mice):
    """Return all combinations where each purview is mapped to a single mechanism."""
    return product(*purview_to_mice.values())


def all_nonconflicting_distinction_sets(distinctions):
    """Return all possible conflict-free distinction sets."""
    if isinstance(distinctions, FlatCauseEffectStructure):
        raise ValueError("Expected distinctions; got MICE (FlatCauseEffectStructure)")
    # Map mechanisms to their distinctions for later fast retrieval
    mechanism_to_distinction = {
        distinction.mechanism: distinction for distinction in distinctions
    }
    # Map purviews to mechanisms that specify them, on both cause and effect sides
    purview_to_mechanism = {direction: defaultdict(list) for direction in DIRECTIONS}
    for distinction in distinctions:
        for direction, mapping in purview_to_mechanism.items():
            # Cast mechanism to set so we can take intersections later
            mapping[distinction.purview(direction)].append(
                frozenset(distinction.mechanism)
            )
    # Generate nonconflicting sets of mechanisms on both cause and effect sides
    nonconflicting_causes, nonconflicting_effects = tuple(
        _nonconflicting_mice_set(purview_to_mechanism[direction])
        for direction in DIRECTIONS
    )
    yield from map(
        CauseEffectStructure,
        map(
            # Convert to actual MICE objects
            mechanism_to_distinction.get,
            # Ensure nonconflicting sets are unique
            set(
                # Take only distinctions that are nonconflicting on both sides
                cause_mechanisms & effect_mechanisms
                # Pair up nonconflicting sets from either side
                for cause_mechanisms, effect_mechanisms in pairs(
                    nonconflicting_causes, nonconflicting_effects
                )
            ),
        ),
    )


# TODO allow choosing whether you provide precomputed distinctions
# (sometimes faster to compute as you go if many distinctions are killed by conflicts)
# TODO document args
def sia(
    subsystem,
    all_distinctions,
    all_relations,
    parallel=False,
    check_trivial_reducibility=True,
):
    """Analyze the irreducibility of a system."""
    phi_structures = all_phi_structures(
        all_nonconflicting_distinction_sets(all_distinctions), all_relations
    )
    return ComputeMaximalCompositionalState(
        phi_structures, subsystem, check_trivial_reducibility
    ).run(parallel or config.PARALLEL_CUT_EVALUATION)
