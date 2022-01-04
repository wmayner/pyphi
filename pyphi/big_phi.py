# -*- coding: utf-8 -*-
# big_phi.py

import operator
from collections import UserDict, defaultdict
from dataclasses import dataclass
from itertools import product

import scipy
from dask.distributed import as_completed
from toolz.itertoolz import unique, partition_all

from . import models
from .compute.parallel import get_client
from .compute.subsystem import sia_bipartitions as directionless_sia_bipartitions
from .direction import Direction
from .models import fmt
from .models.subsystem import CauseEffectStructure, FlatCauseEffectStructure

# TODO
# - cache relations, compute as needed for each nonconflicting CES


class BigPhiCut(models.cuts.Cut):
    """A system cut.

    Same as a IIT 3.0 unidirectional cut, but with a Direction.
    """

    def __init__(self, direction, *args, **kwargs):
        self.direction = direction
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return fmt.fmt_cut(self) + f" ({str(self.direction)[0]})"

    def to_json(self):
        return {
            "direction": self.direction,
            **super().to_json(),
        }

    @classmethod
    def from_json(cls, data):
        """Return a Cut object from a JSON-serializable representation."""
        return cls(data["direction"], data["from_nodes"], data["to_nodes"])


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
        for direction in Direction.both():
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
    specified = {direction: set() for direction in Direction.both()}
    for distinction in distinctions:
        for direction in Direction.both():
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
    return map(frozenset, product(*purview_to_mice.values()))


# TODO(4.0) parallelize somehow?
def all_nonconflicting_distinction_sets(distinctions):
    """Return all possible conflict-free distinction sets."""
    if isinstance(distinctions, FlatCauseEffectStructure):
        raise ValueError("Expected distinctions; got MICE (FlatCauseEffectStructure)")
    # Map mechanisms to their distinctions for later fast retrieval
    mechanism_to_distinction = {
        frozenset(distinction.mechanism): distinction for distinction in distinctions
    }
    # Map purviews to mechanisms that specify them, on both cause and effect sides
    purview_to_mechanism = {
        direction: defaultdict(list) for direction in Direction.both()
    }
    for mechanism, distinction in mechanism_to_distinction.items():
        for direction, mapping in purview_to_mechanism.items():
            # Cast mechanism to set so we can take intersections later
            mapping[distinction.purview(direction)].append(mechanism)
    # Generate nonconflicting sets of mechanisms on both cause and effect sides
    nonconflicting_causes, nonconflicting_effects = tuple(
        _nonconflicting_mice_set(purview_to_mechanism[direction])
        for direction in Direction.both()
    )
    # Ensure nonconflicting sets are unique
    nonconflicting_mechanisms = unique(
        # Take only distinctions that are nonconflicting on both sides
        cause_mechanisms & effect_mechanisms
        # Pair up nonconflicting sets from either side
        for cause_mechanisms, effect_mechanisms in product(
            nonconflicting_causes, nonconflicting_effects
        )
    )
    for mechanisms in nonconflicting_mechanisms:
        # Convert to actual MICE objects
        yield CauseEffectStructure(map(mechanism_to_distinction.get, mechanisms))


# TODO put in utils
def extremum_with_short_circuit(
    seq,
    value_func=lambda item: item.phi,
    cmp=operator.lt,
    initial=float("inf"),
    shortcircuit_value=0,
    shortcircuit_callback=None,
):
    """Return the extreme value, optionally shortcircuiting."""
    extreme_item = None
    extreme_value = initial
    for item in seq:
        value = value_func(item)
        if value == shortcircuit_value:
            try:
                shortcircuit_callback()
            except TypeError:
                pass
            return item
        if cmp(value, extreme_value):
            extreme_value = value
            extreme_item = item
    return extreme_item


def _evaluate_cuts(subsystem, phi_structure, selectivity, cuts):
    return extremum_with_short_circuit(
        (evaluate_cut(subsystem, phi_structure, selectivity, cut) for cut in cuts),
        cmp=operator.lt,
        initial=float("inf"),
        shortcircuit_value=0,
    )


# TODO configure chunksize
def _compute_system_irreducibility(
    subsystem,
    phi_structure,
    selectivity,
    chunksize,
):
    """Analyze the irreducibility of a PhiStructure."""
    client = get_client()
    futures = [
        client.submit(
            _evaluate_cuts,
            subsystem,
            phi_structure,
            selectivity,
            cuts,
        )
        for cuts in partition_all(
            chunksize, sia_partitions(subsystem.cut_indices, subsystem.cut_node_labels)
        )
    ]
    return extremum_with_short_circuit(
        (result for _, result in as_completed(futures, with_results=True)),
        cmp=operator.lt,
        initial=float("inf"),
        shortcircuit_value=0,
        shortcircuit_callback=lambda: client.cancel(futures),
    )


def _null_sia(subsystem, phi_structure, selectivity):
    return SystemIrreducibilityAnalysis(
        phi=0.0,
        subsystem=subsystem,
        cut_subsystem=subsystem,
        selectivity=selectivity,
        ces=phi_structure.distinctions,
        relations=phi_structure.relations,
    )


def is_trivially_reducible(subsystem, phi_structure):
    return any(
        check(subsystem, phi_structure.distinctions) for check in REDUCIBILITY_CHECKS
    )


DEFAULT_CHUNKSIZE = 500


def evaluate_phi_structure(
    subsystem,
    phi_structure,
    check_trivial_reducibility=True,
    chunksize=DEFAULT_CHUNKSIZE,
):
    """Analyze the irreducibility of a PhiStructure."""
    _selectivity = selectivity(subsystem, phi_structure)
    if check_trivial_reducibility and is_trivially_reducible(subsystem, phi_structure):
        return _null_sia(subsystem, phi_structure, _selectivity)
    return _compute_system_irreducibility(
        subsystem, phi_structure, _selectivity, chunksize=chunksize
    )


# TODO allow choosing whether you provide precomputed distinctions
# (sometimes faster to compute as you go if many distinctions are killed by conflicts)
# TODO document args
def sia(
    subsystem,
    all_distinctions,
    all_relations,
    check_trivial_reducibility=True,
    phi_structures=None,
    chunksize=DEFAULT_CHUNKSIZE,
    wait=True,
):
    """Analyze the irreducibility of a system."""
    # First check that the entire set of distinctions/relations is not trivially reducible
    # (since then all subsets must be)
    phi_structure = PhiStructure(all_distinctions, all_relations)
    if check_trivial_reducibility and is_trivially_reducible(subsystem, phi_structure):
        return _null_sia(subsystem, phi_structure, None)

    client = get_client()
    if phi_structures is None:
        phi_structures = list(
            all_phi_structures(
                all_nonconflicting_distinction_sets(all_distinctions), all_relations
            )
        )
    # Broadcast subsystem object to workers
    [subsystem] = client.scatter([subsystem], broadcast=True)
    # Distribute PhiStructures to workers
    phi_structures = client.scatter(phi_structures)
    futures = [
        client.submit(
            evaluate_phi_structure,
            subsystem,
            phi_structure,
            check_trivial_reducibility=check_trivial_reducibility,
            chunksize=chunksize,
        )
        for phi_structure in phi_structures
    ]
    if wait:
        return max(client.gather(futures))
    return futures
