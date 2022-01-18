# -*- coding: utf-8 -*-
# big_phi.py

import operator
from collections import UserDict, defaultdict
from dataclasses import dataclass
from itertools import product

import ray
import scipy
from toolz.itertoolz import partition_all, unique
from tqdm.auto import tqdm

from pyphi import utils
from pyphi.cache import cache
from pyphi.models import cmp
from pyphi.models.cuts import Cut
from pyphi.subsystem import Subsystem

from . import models
from .compute.parallel import as_completed, init
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
            yield BigPhiCut(
                direction, cut.from_nodes, cut.to_nodes, node_labels=cut.node_labels
            )


@cache(cache={}, maxmem=None)
def number_of_possible_relations_with_overlap(n, k):
    """Return the number of possible relations with overlap of size k."""
    return (
        (-1) ** (k - 1)
        * scipy.special.comb(n, k)
        * (2 ** (2 ** (n - k + 1)) - 1 - 2 ** (n - k + 1))
    )


@cache(cache={}, maxmem=None)
def optimum_sum_small_phi_relations(n):
    """Return the 'best possible' sum of small phi for relations."""
    # \sum_{k=1}^{n} (size of purview) * (number of relations with that purview size)
    return sum(
        k * number_of_possible_relations_with_overlap(n, k) for k in range(1, n + 1)
    )


@cache(cache={}, maxmem=None)
def optimum_sum_small_phi_distinctions_one_direction(n):
    """Return the 'best possible' sum of small phi for distinctions in one direction"""
    # \sum_{k=1}^{n} k(n choose k)
    return (2 / n) * (2 ** n)


@cache(cache={}, maxmem=None)
def optimum_sum_small_phi(n):
    """Return the 'best possible' sum of small phi for the system."""
    # Double distinction term for cause & effect sides
    distinction_term = 2 * optimum_sum_small_phi_distinctions_one_direction(n)
    relation_term = optimum_sum_small_phi_relations(n)
    return distinction_term + relation_term


class PhiStructure:
    def __init__(self, distinctions, relations):
        self.distinctions = distinctions
        self.relations = relations
        self._sum_phi_distinctions = None
        self._sum_phi_relations = None
        self._selectivity = None
        if distinctions:
            # TODO improve this
            self._substrate_size = len(distinctions[0].subsystem)

    def sum_phi_distinctions(self):
        if self._sum_phi_distinctions is None:
            self._sum_phi_distinctions = sum(self.distinctions.phis)
        return self._sum_phi_distinctions

    def sum_phi_relations(self):
        if self._sum_phi_relations is None:
            # TODO make `Relations` object with .phis attr
            self._sum_phi_relations = sum(relation.phi for relation in self.relations)
        return self._sum_phi_relations

    def selectivity(self):
        if self._selectivity is None:
            self._selectivity = (
                self.sum_phi_distinctions() + self.sum_phi_relations()
            ) / optimum_sum_small_phi(self._substrate_size)
        return self._selectivity


class PartitionedPhiStructure(PhiStructure):
    def __init__(self, cut, phi_structure):
        self.unpartitioned_phi_structure = phi_structure
        super().__init__(
            self.unpartitioned_phi_structure.distinctions,
            self.unpartitioned_phi_structure.relations,
        )
        self.cut = cut
        # Lift values from unpartitioned PhiStructure
        for attr in [
            "_substrate_size",
            "_sum_phi_distinctions",
            "_sum_phi_relations",
            "_selectivity",
        ]:
            setattr(
                self,
                attr,
                getattr(self.unpartitioned_phi_structure, attr),
            )
        self._partitioned_distinctions = None
        self._partitioned_relations = None
        self._sum_phi_partitioned_distinctions = None
        self._sum_phi_partitioned_relations = None
        self._informativeness = None

    def partitioned_distinctions(self):
        if self._partitioned_distinctions is None:
            self._partitioned_distinctions = unaffected_distinctions(
                self.distinctions, self.cut
            )
        return self._partitioned_distinctions

    def partitioned_relations(self):
        if self._partitioned_relations is None:
            self._partitioned_relations = unaffected_relations(
                self.partitioned_distinctions(), self.relations
            )
        return self._partitioned_relations

    def sum_phi_partitioned_distinctions(self):
        if self._sum_phi_partitioned_distinctions is None:
            self._sum_phi_partitioned_distinctions = sum(
                self.partitioned_distinctions().phis
            )
        return self._sum_phi_partitioned_distinctions

    def sum_phi_partitioned_relations(self):
        if self._sum_phi_partitioned_relations is None:
            self._sum_phi_partitioned_relations = sum(
                relation.phi for relation in self.partitioned_relations()
            )
            # Remove reference to the (heavy and rather redundant) lists of
            # partitioned distinctions & relations under the assumption we won't
            # need them again, since most PartitionedPhiStructures will be used
            # only once, during SIA calculation
            self._partitioned_distinctions = None
            self._partitioned_relations = None
        return self._sum_phi_partitioned_relations

    # TODO use only a single pass through the distinctions / relations?
    def informativeness(self):
        if self._informativeness is None:
            distinction_term = (
                self.sum_phi_distinctions() - self.sum_phi_partitioned_distinctions()
            )
            relation_term = (
                self.sum_phi_relations() - self.sum_phi_partitioned_relations()
            )
            self._informativeness = distinction_term + relation_term
        return self._informativeness

    def phi(self):
        return self.selectivity() * self.informativeness()

    def compute(self):
        """Instantiate lazy properties."""
        self.phi()
        return self


def selectivity(phi_structure):
    """Return the selectivity of the PhiStructure."""
    return phi_structure.selectivity()


def informativeness(partitioned_phi_structure):
    """Return the informativeness of the PartitionedPhiStructure."""
    return partitioned_phi_structure.informativeness()


# TODO add rich methods, comparisons, etc.
@dataclass
class SystemIrreducibilityAnalysis(cmp.Orderable):
    subsystem: Subsystem
    phi_structure: PhiStructure
    partitioned_phi_structure: PartitionedPhiStructure
    cut: Cut
    selectivity: float
    informativeness: float
    phi: float

    _sia_attributes = ["phi", "phi_structure", "partitioned_phi_structure", "subsystem"]

    def order_by(self):
        return [self.phi, len(self.subsystem), self.subsystem.node_indices]

    def __eq__(self, other):
        return cmp.general_eq(self, other, self._sia_attributes)

    def __bool__(self):
        """A |SystemIrreducibilityAnalysis| is ``True`` if it has |big_phi > 0|."""
        return not utils.eq(self.phi, 0)

    def __hash__(self):
        return hash(
            (
                self.phi,
                self.ces,
                self.partitioned_ces,
                self.subsystem,
                self.cut_subsystem,
            )
        )


# TODO rename Cut -> Partition
def evaluate_cut(subsystem, phi_structure, cut):
    partitioned_phi_structure = PartitionedPhiStructure(cut, phi_structure)
    return SystemIrreducibilityAnalysis(
        subsystem=subsystem,
        phi_structure=phi_structure,
        partitioned_phi_structure=partitioned_phi_structure,
        cut=partitioned_phi_structure.cut,
        selectivity=partitioned_phi_structure.selectivity(),
        informativeness=partitioned_phi_structure.informativeness(),
        phi=partitioned_phi_structure.phi(),
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


@ray.remote
def _evaluate_cuts(subsystem, phi_structure, cuts):
    return extremum_with_short_circuit(
        (evaluate_cut(subsystem, phi_structure, cut) for cut in cuts),
        cmp=operator.lt,
        initial=float("inf"),
        shortcircuit_value=0,
    )


def _null_sia(subsystem, phi_structure):
    if not subsystem.cut.is_null():
        raise ValueError("subsystem must have no cut")
    return SystemIrreducibilityAnalysis(
        PartitionedPhiStructure(subsystem.cut, phi_structure)
    )


def is_trivially_reducible(subsystem, phi_structure):
    return any(
        check(subsystem, phi_structure.distinctions) for check in REDUCIBILITY_CHECKS
    )


# TODO configure
DEFAULT_CUT_CHUNKSIZE = 500
DEFAULT_PHI_STRUCTURE_CHUNKSIZE = 50


def _filter_relations(distinctions, relations):
    """Filters relations according to the distinctions in the phi structure."""
    return PhiStructure(
        distinctions=distinctions,
        relations=list(unaffected_relations(distinctions, relations)),
    )


# TODO document args
def evaluate_phi_structure(
    subsystem,
    phi_structure,
    check_trivial_reducibility=True,
    chunksize=DEFAULT_CUT_CHUNKSIZE,
    filter_relations=False,
):
    """Analyze the irreducibility of a PhiStructure."""
    if filter_relations:
        # Assumes `relations` is an ObjectRef from a prior `ray.put` call
        phi_structure = _filter_relations(
            phi_structure.distinctions, ray.get(phi_structure.relations)
        )

    if check_trivial_reducibility and is_trivially_reducible(subsystem, phi_structure):
        return _null_sia(subsystem, phi_structure)

    tasks = [
        _evaluate_cuts.remote(
            subsystem,
            phi_structure,
            cuts,
        )
        for cuts in partition_all(
            chunksize, sia_partitions(subsystem.cut_indices, subsystem.cut_node_labels)
        )
    ]
    return extremum_with_short_circuit(
        as_completed(tasks),
        cmp=operator.lt,
        initial=float("inf"),
        shortcircuit_value=0,
        shortcircuit_callback=lambda: [ray.cancel(task) for task in tasks],
    )


@ray.remote
def _evaluate_phi_structures(
    subsystem,
    phi_structures,
    filter_relations=False,
    **kwargs,
):
    return max(
        evaluate_phi_structure(
            subsystem, phi_structure, filter_relations=filter_relations, **kwargs
        )
        for phi_structure in phi_structures
    )


# TODO allow choosing whether you provide precomputed distinctions
# (sometimes faster to compute as you go if many distinctions are killed by conflicts)
# TODO document args
def sia(
    subsystem,
    all_distinctions,
    all_relations,
    phi_structures=None,
    check_trivial_reducibility=True,
    chunksize=DEFAULT_PHI_STRUCTURE_CHUNKSIZE,
    cut_chunksize=DEFAULT_CUT_CHUNKSIZE,
    filter_relations=False,
    wait=True,
    progress=True,
):
    """Analyze the irreducibility of a system."""
    # First check that the entire set of distinctions/relations is not trivially reducible
    # (since then all subsets must be)
    phi_structure = PhiStructure(all_distinctions, all_relations)
    if check_trivial_reducibility and is_trivially_reducible(subsystem, phi_structure):
        print("Returning trivially-reducible SIA")
        return _null_sia(subsystem, phi_structure)

    # Broadcast subsystem object to workers
    subsystem = ray.put(subsystem)
    print("Done putting subsystem")

    # Assume that phi structures passed by the user don't need to have their
    # relations filtered
    if phi_structures is None:
        filter_relations = True
        # Broadcast relations to workers
        all_relations = ray.put(all_relations)
        print("Done putting relations")
        phi_structures = (
            PhiStructure(distinctions, all_relations)
            for distinctions in all_nonconflicting_distinction_sets(all_distinctions)
        )

    tasks = [
        _evaluate_phi_structures.remote(
            subsystem,
            chunk,
            filter_relations=filter_relations,
            check_trivial_reducibility=check_trivial_reducibility,
            chunksize=cut_chunksize,
        )
        for chunk in tqdm(
            partition_all(chunksize, phi_structures), desc="Submitting tasks"
        )
    ]
    print("Done submitting tasks")
    if wait:
        results = as_completed(tasks)
        if progress:
            results = tqdm(results, total=len(tasks))
        return max(results)
    return tasks
