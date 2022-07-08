# -*- coding: utf-8 -*-
# big_phi.py

import functools
import logging
import pickle
import warnings
from collections import UserDict, defaultdict
from dataclasses import dataclass
from itertools import combinations, product

import networkx as nx
import numpy as np
import ray
from more_itertools import all_equal
from numpy import ma
from toolz.itertoolz import partition_all
from tqdm.auto import tqdm

from . import compute, config, upper_bounds, utils
from .combinatorics import largest_independent_sets, maximal_independent_sets
from .compute import parallel as _parallel
from .compute.network import reachable_subsystems
from .compute.parallel import as_completed
from .conf import fallback
from .direction import Direction
from .models import cmp, fmt
from .models.cuts import CompleteSystemPartition, NullCut, SystemPartition
from .models.subsystem import CauseEffectStructure, FlatCauseEffectStructure
from .partition import system_partition_types
from .registry import Registry
from .relations import ConcreteRelations, Relations
from .relations import relations as compute_relations
from .subsystem import Subsystem
from .utils import expsublog

# TODO
# - cache relations, compute as needed for each nonconflicting CES

# Create a logger for this module.
log = logging.getLogger(__name__)


def is_affected_by_partition(distinction, partition):
    """Return whether the distinctions is affected by the partition."""
    # TODO(4.0) standardize logic for complete partition vs other partition
    if isinstance(partition, CompleteSystemPartition):
        return True
    if isinstance(partition, NullCut):
        return False
    coming_from = set(partition.from_nodes) & set(distinction.mechanism)
    going_to = set(partition.to_nodes) & set(distinction.purview(partition.direction))
    return coming_from and going_to


def unaffected_distinctions(ces, partition):
    """Return the CES composed of distinctions that are not affected by the given partition."""
    # Special case for empty CES
    if isinstance(partition, CompleteSystemPartition):
        return CauseEffectStructure([], subsystem=ces.subsystem)
    return CauseEffectStructure(
        [
            distinction
            for distinction in ces
            if not is_affected_by_partition(distinction, partition)
        ],
        subsystem=ces.subsystem,
    )


def sia_partitions(node_indices, node_labels=None):
    """Yield all system partitions."""
    # TODO(4.0) consolidate 3.0 and 4.0 cuts
    scheme = config.SYSTEM_PARTITION_TYPE
    valid = ["TEMPORAL_DIRECTED_BI", "TEMPORAL_DIRECTED_BI_CUT_ONE"]
    if scheme not in valid:
        raise ValueError(
            "IIT 4.0 calculations must use one of the following system"
            f"partition schemes: {valid}; got {scheme}"
        )
    # Special case for single-element systems
    if len(node_indices) == 1:
        yield CompleteSystemPartition()
    else:
        yield from system_partition_types[config.SYSTEM_PARTITION_TYPE](
            node_indices, node_labels=node_labels
        )


def _requires_relations(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Get relations from Ray if they're remote
        if isinstance(self.relations, ray.ObjectRef):
            self.relations = ray.get(self.relations)
        # Filter relations if flag is set
        if self.requires_filter_relations:
            self.filter_relations()
        return func(self, *args, **kwargs)

    return wrapper


class PhiStructure(cmp.Orderable):
    def __init__(
        self,
        distinctions,
        relations=None,
        requires_filter_relations=False,
    ):
        if not isinstance(distinctions, CauseEffectStructure):
            raise ValueError(
                f"distinctions must be a CauseEffectStructure, got {type(distinctions)}"
            )
        if distinctions.subsystem is None:
            raise ValueError("CauseEffectStructure must have the `subsystem` attribute")
        if isinstance(distinctions, FlatCauseEffectStructure):
            distinctions = distinctions.unflatten()
        if not isinstance(relations, (Relations, ray.ObjectRef, type(None))):
            raise ValueError(
                f"relations must be a Relations object, ray.ObjectRef, or None; "
                f"got {type(relations)}"
            )
        self.requires_filter_relations = requires_filter_relations
        self.distinctions = distinctions
        self.relations = relations
        self._system_intrinsic_information = None
        self._sum_phi_distinctions = None
        self._selectivity = None
        # TODO improve this
        self._substrate_size = len(self.subsystem)

    @property
    def subsystem(self):
        return self.distinctions.subsystem

    def order_by(self):
        return self.system_intrinsic_information()

    @_requires_relations
    def __eq__(self, other):
        return cmp.general_eq(
            self,
            other,
            [
                "distinctions",
                "relations",
            ],
        )

    @_requires_relations
    def __hash__(self):
        return hash((self.distinctions, self.relations))

    def __bool__(self):
        return bool(self.distinctions)

    def __repr__(self):
        return fmt.fmt_phi_structure(self)

    def __getstate__(self):
        dct = self.__dict__
        if isinstance(self.relations, ConcreteRelations):
            distinctions = self.distinctions.flatten()
            dct["relations"] = self.relations.to_indirect_json(distinctions)
        return dct

    def __setstate__(self, state):
        try:
            distinctions = state["distinctions"]
            distinctions = distinctions.flatten()
            state["relations"] = ConcreteRelations.from_indirect_json(
                distinctions, state["relations"]
            )
        except:
            # Assume relations can be unpickled by default
            pass
        finally:
            self.__dict__ = state

    to_json = __getstate__

    def to_pickle(self, path):
        with open(path, mode="wb") as f:
            pickle.dump(self, f)
        return path

    @classmethod
    def read_pickle(cls, path):
        with open(path, mode="rb") as f:
            return pickle.load(f)

    @classmethod
    def from_json(cls, data):
        instance = cls(data["distinctions"], data["relations"])
        instance.__dict__.update(data)
        return instance

    def filter_relations(self):
        """Update relations so that only those supported by distinctions remain.

        Modifies the relations on this object in-place.
        """
        self.relations = self.relations.supported_by(self.distinctions)
        self.requires_filter_relations = False

    def sum_phi_distinctions(self):
        if self._sum_phi_distinctions is None:
            self._sum_phi_distinctions = sum(self.distinctions.phis)
        return self._sum_phi_distinctions

    @_requires_relations
    def sum_phi_relations(self):
        return self.relations.sum_phi()

    def sum_phi(self):
        return self.sum_phi_distinctions() + self.sum_phi_relations()

    def selectivity(self):
        if self._selectivity is None:
            # Use expsublog to deal with enormous denominator
            numerator = self.sum_phi()
            if numerator == 0:
                return 0
            denominator = upper_bounds.sum_phi_upper_bound(self._substrate_size)
            self._selectivity = expsublog(numerator, denominator)
        return self._selectivity

    @_requires_relations
    def realize(self):
        """Instantiate lazy properties."""
        # Currently this is just a hook to force _requires_relations to do its
        # work. Also very Zen.
        return self

    def apply_partition(self, partition):
        """Apply a partition to this PhiStructure."""
        return PartitionedPhiStructure(
            self.distinctions,
            self.relations,
            partition=partition,
            unpartitioned_phi_structure=self,
        )

    def system_intrinsic_information(self):
        """Return the system intrinsic information.

        This is the phi of the system with respect to the complete partition.
        """
        if self._system_intrinsic_information is None:
            self._system_intrinsic_information = self.apply_partition(
                CompleteSystemPartition()
            ).phi()
        return self._system_intrinsic_information


class PartitionedPhiStructure(PhiStructure):
    def __init__(
        self,
        distinctions,
        relations,
        partition,
        unpartitioned_phi_structure,
    ):
        # We need to realize the underlying PhiStructure in case
        # distinctions/relations are generators which may later become exhausted
        unpartitioned_phi_structure = unpartitioned_phi_structure.realize()

        distinctions = unaffected_distinctions(distinctions, partition)
        super().__init__(
            distinctions,
            relations=relations,
            requires_filter_relations=True,
        )

        self.partition = partition
        self.unpartitioned_phi_structure = unpartitioned_phi_structure
        self._informativeness = None

    def order_by(self):
        return self.phi()

    def __eq__(self, other):
        return super().__eq__(other) and cmp.general_eq(
            self,
            other,
            [
                "phi",
                "partition",
                "distinctions",
                "relations",
            ],
        )

    def __hash__(self):
        return hash((super().__hash__(), self.partition))

    def __bool__(self):
        """A |PartitionedPhiStructure| is ``True`` if it has |big_phi > 0|."""
        return utils.is_positive(self.phi())

    def __repr__(self):
        return fmt.fmt_partitioned_phi_structure(self)

    def informativeness(self):
        if self._informativeness is None:
            self._informativeness = (
                self.unpartitioned_phi_structure.sum_phi() - self.sum_phi()
            )
        return self._informativeness

    def phi(self):
        return self.unpartitioned_phi_structure.selectivity() * self.informativeness()

    def to_json(self):
        return {**super().to_json(), "partition": self.partition}


def selectivity(phi_structure):
    """Return the selectivity of the PhiStructure."""
    return phi_structure.selectivity()


def informativeness(partitioned_phi_structure):
    """Return the informativeness of the PartitionedPhiStructure."""
    return partitioned_phi_structure.informativeness()


def phi(partitioned_phi_structure):
    """Return the phi of the PartitionedPhiStructure."""
    return partitioned_phi_structure.phi()


# TODO add rich methods, comparisons, etc.
@dataclass
class SystemIrreducibilityAnalysis(cmp.Orderable):
    subsystem: Subsystem
    phi_structure: PhiStructure
    partitioned_phi_structure: PartitionedPhiStructure
    partition: SystemPartition
    selectivity: float
    informativeness: float
    phi: float
    reasons: list = None

    _sia_attributes = ["phi", "phi_structure", "partitioned_phi_structure", "subsystem"]

    def order_by(self):
        return [self.phi, len(self.subsystem), self.subsystem.node_indices]

    def __eq__(self, other):
        return cmp.general_eq(self, other, self._sia_attributes)

    def __bool__(self):
        """A |SystemIrreducibilityAnalysis| is ``True`` if it has |big_phi > 0|."""
        return utils.is_positive(self.phi)

    def __hash__(self):
        return hash(
            (
                self.phi,
                self.phi_structure,
                self.partitioned_phi_structure,
                self.subsystem,
            )
        )

    def __repr__(self):
        return fmt.fmt_sia_4(self)


def evaluate_partition(partition, subsystem, phi_structure):
    partitioned_phi_structure = phi_structure.apply_partition(partition)
    return SystemIrreducibilityAnalysis(
        subsystem=subsystem,
        phi_structure=phi_structure,
        selectivity=phi_structure.selectivity(),
        partitioned_phi_structure=partitioned_phi_structure,
        partition=partitioned_phi_structure.partition,
        informativeness=partitioned_phi_structure.informativeness(),
        phi=partitioned_phi_structure.phi(),
    )


@dataclass
class HAS_NONSPECIFIED_ELEMENTS:
    elements: list = None

    def __repr__(self):
        return f"Nonspecified elements: {self.elements}"


@dataclass
class HAS_NO_SPANNING_SPECIFICATION:
    elements: list = None

    def __repr__(self):
        return f"No specification spanning partition: {self.elements}"


def has_nonspecified_elements(distinctions):
    """Return whether any elements are not specified by a purview in both
    directions."""
    elements = set(distinctions.subsystem.node_indices)
    specified = {direction: set() for direction in Direction.both()}
    for distinction in distinctions:
        for direction in Direction.both():
            specified[direction].update(set(distinction.purview(direction)))
    nonspecified = set()
    for _specified in specified.values():
        if elements - _specified:
            nonspecified |= elements
    if nonspecified:
        return HAS_NONSPECIFIED_ELEMENTS(nonspecified)


def has_no_spanning_specification(distinctions):
    """Return whether the system can be separated into disconnected components.

    Here disconnected means that there is no "spanning specification"; some
    subset of elements only specifies themselves and is not specified by any
    other subset.
    """
    # TODO
    pass


REDUCIBILITY_CHECKS_FOR_DISTINCTIONS = [
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


def tied_distinction_sets(distinctions, purview=True, state=True, partition=True):
    """Yield all combinations of tied distinctions.

    NOTE: Only considers ties among purviews; ties among MIPs that share the
    same purview are resolved arbitrarily.
    """
    for tie in product(
        *[
            distinction.ties(purview=purview, state=state, partition=partition)
            for distinction in distinctions.flat
        ]
    ):
        yield FlatCauseEffectStructure(
            tie, subsystem=distinctions.subsystem
        ).unflatten()


def _purview_mapping(distinctions):
    """Return {purview: mechanism} and {mechanism: distinction} mappings."""
    # Map mechanisms to distinctions for fast retreival
    mechanism_to_distinction = dict()
    # Map purviews to mechanisms that specify them, on both cause and effect sides
    purview_to_mechanisms = {
        direction: defaultdict(list) for direction in Direction.both()
    }
    # Populate mappings
    for distinction in distinctions:
        mechanism_to_distinction[distinction.mechanism] = distinction
        for direction in Direction.both():
            purview_to_mechanisms[direction][distinction.purview(direction)].append(
                distinction.mechanism
            )
    return purview_to_mechanisms, mechanism_to_distinction


class CompositionalStateConflicts(Registry):
    """Storage for functions for defining when distinctions conflict."""

    desc = "conflict definitions"


compositional_state_conflict_definitions = CompositionalStateConflicts()


@compositional_state_conflict_definitions.register("SAME_PURVIEW")
def _(mice1, mice2):
    """Conflicts specify the same purview."""
    # Any two distinctions that share a purview conflict
    return True


@compositional_state_conflict_definitions.register("SAME_PURVIEW_AND_INCONGRUENT_STATE")
def _(mice1, mice2):
    """Conflicts specify the same purview and different states."""
    if any(len(mice.specified_state) > 1 for mice in [mice1, mice2]):
        raise ValueError(
            "Multiple specified states; expected only one. "
            "Nonconflicting sets should be computed with `state_ties=True` to "
            "consider each tied state separately."
        )
    # Conditions for conflict:
    # - Incongruent state
    # - They share the same purview in the other direction (they are not
    #   distinct, and they conflict even if they're congruent)
    # TODO(4.0) is_congruent() method on MICE?
    return (not np.array_equal(mice1.specified_state, mice2.specified_state)) or (
        mice1.flip().purview == mice2.flip().purview
    )


def are_conflicting(mice1, mice2):
    """Return whether two MICE conflict.

    The definition of 'conflict' is controlled by the
    COMPOSITIONAL_STATE_CONFLICTS option.
    """
    return compositional_state_conflict_definitions[
        config.COMPOSITIONAL_STATE_CONFLICTS
    ](mice1, mice2)


def _agree_on_global_state(distinctions, n):
    """Return whether the given MICE agree on a global state.

    Assumes the MICE all have the same direction.

    Arguments:
        distinctions (Iterable): The distinctions to consider.
        n (int): The maximum node index in the system.
    """
    # Initialize and mask the specification array. We use a mask to indicate
    # that the node is not specified (i.e., not in the purview)
    specification = np.empty([len(distinctions), n])
    specification.fill(np.nan)
    specification = ma.masked_invalid(specification)

    for i, distinction in enumerate(distinctions):
        specification[i, list(distinction.purview)] = distinction.specified_state[0]

    # Are all non-masked entries are equal?
    is_congruent = lambda column: all_equal(column[~column.mask])

    # Check congruence for all nodes (apply to columns, i.e. along rows)
    return np.all(ma.apply_along_axis(is_congruent, 0, specification))


# TODO refactor to combine with other conflict graph logic
# TODO optimize: if any node's state is not specified, we can shortcircuit since
#      there must be a cut
def _global_conflict_graph(distinctions):
    """Return a graph where conflicts are defined by global incongruence."""
    G = nx.Graph()
    G.add_nodes_from(distinctions.mechanisms)

    mechanism_to_distinction = dict()
    for distinction in distinctions:
        mechanism_to_distinction[distinction.mechanism] = distinction
        if any(
            len(distinction.mice(direction).specified_state) > 1
            for direction in Direction.both()
        ):
            raise ValueError(
                "found unexpected state tie for mechanism {distinction.mechanism}: "
                "when COMPOSITIONAL_STATE_CONFLICTS = 'GLOBAL', must generate "
                "nonconflicting phi structures with `state_ties=True`"
            )

    for d1, d2 in combinations(distinctions, 2):
        if all(
            # Do they the same cause & effect purviews?
            d1.purview(direction) == d2.purview(direction)
            for direction in Direction.both()
        ) or not all(
            # Are they incongruent globally?
            _agree_on_global_state(
                [d1.mice(direction), d2.mice(direction)],
                max(distinctions.subsystem.node_indices) + 1,
            )
            for direction in Direction.both()
        ):
            G.add_edge(d1.mechanism, d2.mechanism)

    return G, mechanism_to_distinction


def conflict_graph(distinctions):
    """Return a graph where nodes are distinctions and edges are conflicts.

    What defines a conflict is controlled by the COMPOSITIONAL_STATE_CONFLICTS
    option.
    """
    G = nx.Graph()
    G.add_nodes_from(distinctions.mechanisms)
    # TODO(4.0) possibly refactor to avoid special case
    if config.COMPOSITIONAL_STATE_CONFLICTS == "GLOBAL":
        return _global_conflict_graph(distinctions)
    purview_to_mechanisms, mechanism_to_distinction = _purview_mapping(distinctions)
    for direction, submapping in purview_to_mechanisms.items():
        for purview, mechanisms in submapping.items():
            # Pairs of mechanisms specifying the same purview in the same
            # direction
            for mechanism1, mechanism2 in combinations(mechanisms, 2):
                mice1 = mechanism_to_distinction[mechanism1].mice(direction)
                mice2 = mechanism_to_distinction[mechanism2].mice(direction)
                if are_conflicting(mice1, mice2):
                    G.add_edge(mechanism1, mechanism2)
    return G, mechanism_to_distinction


def _all_nonconflicting_distinction_sets(distinctions):
    # Nonconflicting sets are maximal independent sets of the conflict graph.
    # NOTE: The maximality criterion here depends on the property of big phi
    # that it is monotonic increasing with the number of distinctions. If this
    # changes, this function should be changed to yield all independent sets,
    # not just the maximal ones.
    graph, mechanism_to_distinction = conflict_graph(distinctions)
    for maximal_independent_set in maximal_independent_sets(graph):
        yield CauseEffectStructure(
            # Though distinctions are hashable, the hash function is relatively
            # expensive (since repertoires are hashed), so we work with
            # mechanisms instead
            map(mechanism_to_distinction.get, maximal_independent_set),
            subsystem=distinctions.subsystem,
        )


# TODO refactor
def largest_nonconflicting_distinction_sets(
    distinctions,
):
    graph, mechanism_to_distinction = conflict_graph(distinctions)
    for maximal_independent_set in largest_independent_sets(graph):
        yield CauseEffectStructure(
            # Though distinctions are hashable, the hash function is relatively
            # expensive (since repertoires are hashed), so we work with
            # mechanisms instead
            map(mechanism_to_distinction.get, maximal_independent_set),
            subsystem=distinctions.subsystem,
        )


def all_nonconflicting_distinction_sets(
    distinctions,
    purview_ties=True,
    state_ties=True,
    partition_ties=True,
    all_ties=False,
    only_largest=False,
):
    # TODO docstring
    """Return all maximal non-conflicting sets of distinctions.

    Arguments:
        distinctions (CauseEffectStructure): The set of distinctions to consider.

    Keyword Arguments:
        purview_ties (bool): Whether to also consider all resolutions of purview
            ties.
        state_ties (bool): Whether to also consider all resolutions of state
            ties.
        partition_ties (bool): Whether to also consider all resolutions of
            partition ties.
        all_ties (bool): Whether to consider all kinds of ties. Overrides the
            individual tie options.
        only_largest (bool): Whether to consider only the distinction sets with
            the most distinctions. This can greatly speed up the calculation,
            but the largest sets are not necessarily those that maximize system
            intrinsic information, and so this is an approximation.

    Yields:
        CauseEffectStructure: A CES without conflicts.
    """
    if all_ties:
        purview_ties = state_ties = partition_ties = True
    for tie in tied_distinction_sets(
        distinctions,
        purview=purview_ties,
        state=state_ties,
        partition=partition_ties,
    ):
        if only_largest:
            yield from largest_nonconflicting_distinction_sets(tie)
        else:
            yield from _all_nonconflicting_distinction_sets(tie)


def _null_sia(subsystem, phi_structure, reasons=None):
    if not subsystem.cut.is_null:
        raise ValueError("subsystem must have no partition")
    partitioned_phi_structure = phi_structure.apply_partition(subsystem.cut)
    return SystemIrreducibilityAnalysis(
        subsystem=subsystem,
        phi_structure=phi_structure,
        partitioned_phi_structure=partitioned_phi_structure,
        partition=partitioned_phi_structure.partition,
        selectivity=None,
        informativeness=None,
        phi=0.0,
        reasons=reasons,
    )


def _null_complex(network, state):
    return SystemIrreducibilityAnalysis(
        subsystem=Subsystem(network, state, ()),
        phi_structure=None,
        partitioned_phi_structure=None,
        partition=None,
        selectivity=None,
        informativeness=None,
        phi=0.0,
        reasons=None,
    )


def distinctions_are_trivially_reducible(distinctions):
    """Return a list of reasons that the distinctions are reducible."""
    reasons = []
    for check in REDUCIBILITY_CHECKS_FOR_DISTINCTIONS:
        reason = check(distinctions)
        if reason:
            reasons.append(reason)
    return reasons


def is_trivially_reducible(phi_structure):
    # TODO add relations check when available
    return distinctions_are_trivially_reducible(phi_structure.distinctions)


# TODO configure
# TODO optimize
DEFAULT_PARTITION_SEQUENTIAL_THRESHOLD = 256
DEFAULT_PARTITION_CHUNKSIZE = 4 * DEFAULT_PARTITION_SEQUENTIAL_THRESHOLD
DEFAULT_PHI_STRUCTURE_SEQUENTIAL_THRESHOLD = 8
DEFAULT_PHI_STRUCTURE_CHUNKSIZE = 4 * DEFAULT_PHI_STRUCTURE_SEQUENTIAL_THRESHOLD


# TODO document args
def evaluate_phi_structure(
    subsystem,
    phi_structure,
    check_trivial_reducibility=True,
    parallel=None,
    chunksize=DEFAULT_PARTITION_CHUNKSIZE,
    sequential_threshold=DEFAULT_PARTITION_SEQUENTIAL_THRESHOLD,
    progress=None,
):
    """Analyze the irreducibility of a PhiStructure."""
    parallel = fallback(parallel, config.PARALLEL_CUT_EVALUATION)

    # Realize the PhiStructure before distributing tasks
    phi_structure.realize()

    if check_trivial_reducibility and is_trivially_reducible(phi_structure):
        return _null_sia(subsystem, phi_structure)

    partitions = sia_partitions(subsystem.cut_indices, subsystem.cut_node_labels)
    if progress:
        partitions = tqdm(partitions, desc="Partitions")

    return _parallel.map_reduce(
        evaluate_partition,
        min,
        partitions,
        subsystem=subsystem,
        phi_structure=phi_structure,
        chunksize=chunksize,
        sequential_threshold=sequential_threshold,
        shortcircuit_value=0,
        parallel=parallel,
        progress=progress,
        desc="Evaluating partitions",
    )


def _system_intrinsic_information(phi_structure):
    return (phi_structure.system_intrinsic_information(), phi_structure)


# TODO refactor into a pattern
def find_maximal_compositional_state(
    phi_structures,
    parallel=None,
    chunksize=DEFAULT_PHI_STRUCTURE_CHUNKSIZE,
    sequential_threshold=DEFAULT_PHI_STRUCTURE_SEQUENTIAL_THRESHOLD,
    progress=None,
):
    parallel = fallback(parallel, config.PARALLEL_COMPOSITIONAL_STATE_EVALUATION)
    log.debug("Finding maximal compositional state...")
    _, phi_structure = _parallel.map_reduce(
        _system_intrinsic_information,
        max,
        phi_structures,
        shortcircuit_value=0,
        parallel=parallel,
        chunksize=chunksize,
        sequential_threshold=sequential_threshold,
        progress=progress,
        desc="Evaluating compositional states",
    )
    return phi_structure


_compute_relations = ray.remote(compute_relations)


def nonconflicting_phi_structures(
    all_distinctions,
    all_relations=None,
    purview_ties=True,
    state_ties=True,
    partition_ties=True,
    all_ties=False,
    only_largest=False,
    parallel=None,
    progress=None,
    desc=None,
):
    """Yield nonconflicting PhiStructures."""
    parallel = fallback(parallel, config.PARALLEL_COMPOSITIONAL_STATE_EVALUATION)
    progress = fallback(progress, config.PROGRESS_BARS)
    distinction_sets = all_nonconflicting_distinction_sets(
        all_distinctions,
        purview_ties=purview_ties,
        state_ties=state_ties,
        partition_ties=partition_ties,
        all_ties=all_ties,
        only_largest=only_largest,
    )
    if progress:
        distinction_sets = tqdm(distinction_sets, desc=desc)
    for distinctions in distinction_sets:
        if all_relations is None:
            # Compute relations on workers for each nonconflicting set
            if parallel:
                # Non-blocking task so we can yield immediately
                relations = _compute_relations.remote(
                    all_distinctions.subsystem,
                    distinctions,
                    parallel=True,
                    progress=progress,
                )
            else:
                relations = compute_relations(
                    all_distinctions.subsystem,
                    distinctions,
                    progress=progress,
                )
            requires_filter_relations = False
        else:
            relations = all_relations
            requires_filter_relations = True
        yield PhiStructure(
            distinctions,
            relations,
            requires_filter_relations=requires_filter_relations,
        )


# TODO allow choosing whether you provide precomputed distinctions
# (sometimes faster to compute as you go if many distinctions are killed by conflicts)
# TODO document args
def sia(
    subsystem,
    all_distinctions=None,
    phi_structures=None,
    check_trivial_reducibility=True,
    chunksize=DEFAULT_PHI_STRUCTURE_CHUNKSIZE,
    sequential_threshold=DEFAULT_PHI_STRUCTURE_SEQUENTIAL_THRESHOLD,
    partition_chunksize=DEFAULT_PARTITION_CHUNKSIZE,
    partition_sequential_threshold=DEFAULT_PARTITION_SEQUENTIAL_THRESHOLD,
    progress=None,
    purview_ties=True,
    state_ties=True,
    partition_ties=True,
    all_ties=False,
    only_largest=False,
    parallel=None,
):
    """Analyze the irreducibility of a system."""
    if not state_ties and config.RELATION_COMPUTATION == "ANALYTICAL":
        warnings.warn(
            "Using RELATION_COMPUTATION = 'ANALYTICAL' without setting state_ties=True "
            "may result in incorrect values for the sum of relation phis!"
        )

    progress = fallback(progress, config.PROGRESS_BARS)

    if all_distinctions is None:
        all_distinctions = compute.ces(subsystem)

    if not isinstance(all_distinctions, CauseEffectStructure):
        raise ValueError("all_distinctions must be a CauseEffectStructure")
    if isinstance(all_distinctions, FlatCauseEffectStructure):
        all_distinctions = all_distinctions.unflatten()

    # First check that the entire set of distinctions is not trivially reducible
    # (since then all subsets must be)
    full_phi_structure = PhiStructure(all_distinctions)

    # TODO(4.0) disable this check for now because it doesn't take ties into
    # account need to figure out a good way of checking this if possible, but
    # ties make it potentially expensive; maybe just check things that don't
    # have ties, or see if the number of tied combinations is small, etc.
    # if check_trivial_reducibility:
    #     reasons = is_trivially_reducible(full_phi_structure)
    #     if reasons:
    #         log.debug(
    #             "SIA is trivially-reducible; returning early.\nReasons: {%s}", reasons
    #         )
    #         return _null_sia(subsystem, full_phi_structure, reasons=reasons)

    if phi_structures is None:
        phi_structures = nonconflicting_phi_structures(
            all_distinctions,
            purview_ties=purview_ties,
            state_ties=state_ties,
            partition_ties=partition_ties,
            all_ties=all_ties,
            only_largest=only_largest,
            parallel=parallel,
            progress=progress,
            desc="Generating nonconflicting phi-structures",
        )

    if config.IIT_VERSION == "maximal-state-first":
        maximal_compositional_state = find_maximal_compositional_state(
            phi_structures,
            chunksize=chunksize,
            sequential_threshold=sequential_threshold,
            progress=progress,
            parallel=parallel,
        )
        log.debug("Evaluating maximal compositional state...")
        analysis = evaluate_phi_structure(
            subsystem,
            maximal_compositional_state,
            check_trivial_reducibility=check_trivial_reducibility,
            chunksize=partition_chunksize,
            sequential_threshold=partition_sequential_threshold,
            parallel=parallel,
            progress=progress,
        )
        log.debug("Done evaluating maximal compositional state; returning SIA.")
        return analysis
    else:
        # TODO(4.0) remove this block?
        # Broadcast subsystem object to workers
        log.debug("Putting subsystem into all workers...")
        subsystem = ray.put(subsystem)
        log.debug("Done putting subsystem into all workers.")

        log.debug("Evaluating all compositional states...")
        tasks = [
            _evaluate_phi_structures.remote(
                subsystem,
                chunk,
                check_trivial_reducibility=check_trivial_reducibility,
                chunksize=partition_chunksize,
            )
            for chunk in tqdm(
                partition_all(chunksize, phi_structures),
                desc="Submitting compositional states for evaluation",
            )
        ]
        log.debug("Done submitting tasks.")
        results = as_completed(tasks)
        if progress:
            results = tqdm(
                results, total=len(tasks), desc="Evaluating compositional states"
            )
        maximum = max(results)
        log.debug("Done evaluating all compositional states; returning SIA.")
        return maximum


def complexes(network, state, **kwargs):
    # TODO(4.0) parallelize
    for subsystem in reachable_subsystems(network, network.node_indices, state):
        ces = compute.ces(subsystem)
        _sia = sia(subsystem, ces, **kwargs)
        if _sia:
            yield _sia


def major_complex(network, state, **kwargs):
    result = complexes(network, state, **kwargs)
    if result:
        return max(result)
    return _null_complex(network, state)
