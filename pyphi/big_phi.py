# -*- coding: utf-8 -*-
# big_phi.py

import functools
import logging
import operator
import pickle
from collections import UserDict, defaultdict
from dataclasses import dataclass
from itertools import product

import networkx as nx
import ray
import scipy
from toolz.itertoolz import partition_all
from tqdm.auto import tqdm

from . import combinatorics, compute, config, utils
from .cache import cache
from .combinatorics import maximal_independent_sets
from .compute.network import reachable_subsystems
from .compute.parallel import as_completed, init
from .direction import Direction
from .models import cmp, fmt
from .models.cuts import CompleteSystemPartition, NullCut, SystemPartition
from .models.subsystem import CauseEffectStructure, FlatCauseEffectStructure
from .partition import system_partition_types
from .registry import Registry
from .relations import ConcreteRelations, Relations
from .relations import relations as compute_relations
from .subsystem import Subsystem
from .utils import expsublog, extremum_with_short_circuit

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


def number_of_possible_distinctions_of_order(n, k):
    """Return the number of possible distinctions of order k."""
    # Binomial coefficient
    return int(scipy.special.comb(n, k))


def number_of_possible_distinctions(n):
    """Return the number of possible distinctions."""
    return 2 ** n - 1


@cache(cache={}, maxmem=None)
def _f(n, k):
    return (2 ** (2 ** (n - k + 1))) - (1 + 2 ** (n - k + 1))


class DistinctionSumPhiUpperBoundRegistry(Registry):
    """Storage for functions for defining the upper bound of the sum of
    distinction phi when analyzing the system.

    NOTE: Functions should ideally return `int`s, if possible, to take advantage
    of the unbounded size of Python integers.
    """

    desc = "distinction sum phi bounds (system)"


distinction_sum_phi_upper_bounds = DistinctionSumPhiUpperBoundRegistry()


@distinction_sum_phi_upper_bounds.register("PURVIEW_SIZE")
def _(n):
    # This can be simplified to (n/2)*(2^n), but we don't use that identity so
    # we can keep things as `int`s
    return sum(
        k * number_of_possible_distinctions_of_order(n, k) for k in range(1, n + 1)
    )


_ = distinction_sum_phi_upper_bounds.register("2^N-1")(number_of_possible_distinctions)


@distinction_sum_phi_upper_bounds.register("(2^N-1)/(N-1)")
def _(n):
    try:
        return number_of_possible_distinctions(n) / (n - 1)
    except ZeroDivisionError:
        return 1


def distinction_sum_phi_upper_bound(n):
    """Return the 'best possible' sum of small phi for distinctions."""
    return distinction_sum_phi_upper_bounds[config.DISTINCTION_SUM_PHI_UPPER_BOUND](n)


@cache(cache={}, maxmem=None)
def number_of_possible_relations_of_order(n, k):
    """Return the number of possible relations with overlap of size k."""
    # Alireza's generalization of Will's theorem
    return int(scipy.special.comb(n, k)) * sum(
        ((-1) ** i * int(scipy.special.comb(n - k, i)) * _f(n, k + i))
        for i in range(n - k + 1)
    )


@cache(cache={}, maxmem=None)
def number_of_possible_relations(n):
    """Return the number of possible relations of all orders."""
    return sum(number_of_possible_relations_of_order(n, k) for k in range(1, n + 1))


def _relation_sum_phi_distinction_phi_is_purview_size(n):
    return sum(k * number_of_possible_relations_of_order(n, k) for k in range(1, n + 1))


def _relation_sum_phi_distinction_phi_is_one(n):
    # Distinction phi <= 1 implies relation phi is bounded by 1/|z| where z is
    # the largest purview in the relation
    subsets = [
        1 / (len(z) + 1) for z in utils.powerset(range(n - 1), nonempty=False)
    ] * 2
    return n * combinatorics.sum_of_minimum_among_subsets(subsets)


def _relation_sum_phi_distinction_phi_is_one_over_n_minus_one(n):
    try:
        return (2 / ((n - 1) ** 2)) * combinatorics.sum_of_minimum_among_subsets(
            [1 / (len(z) + 1) for z in utils.powerset(range(n - 1), nonempty=False)] * 2
        )
    except ZeroDivisionError:
        return 1


RELATION_SUM_PHI_UPPER_BOUNDS = {
    "PURVIEW_SIZE": _relation_sum_phi_distinction_phi_is_purview_size,
    "2^N-1": _relation_sum_phi_distinction_phi_is_one,
    "(2^N-1)/(N-1)": _relation_sum_phi_distinction_phi_is_one_over_n_minus_one,
}


def relation_sum_phi_upper_bound(n):
    """Return the 'best possible' sum of small phi for relations."""
    return RELATION_SUM_PHI_UPPER_BOUNDS[config.DISTINCTION_SUM_PHI_UPPER_BOUND](n)


def sum_phi_upper_bound(n):
    """Return the 'best possible' sum of small phi for the system."""
    return distinction_sum_phi_upper_bound(n) + relation_sum_phi_upper_bound(n)


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
            denominator = sum_phi_upper_bound(self._substrate_size)
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
        return not utils.eq(self.phi(), 0)

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
        return not utils.eq(self.phi, 0)

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


def evaluate_partition(subsystem, phi_structure, partition):
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


def tied_distinction_sets(distinctions):
    """Yield all combinations of tied distinctions.

    NOTE: Only considers ties among purviews; ties among MIPs that share the
    same purview are resolved arbitrarily.
    """
    for tie in product(
        *[distinction.purview_ties for distinction in distinctions.flatten()]
    ):
        yield FlatCauseEffectStructure(
            tie, subsystem=distinctions.subsystem
        ).unflatten()


def conflict_graph(distinctions):
    """Return a graph where nodes are distinctions and edges are conflicts."""
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
    # Construct graph where nodes are distinctions and edges are conflicts
    G = nx.Graph()
    for direction, mapping in purview_to_mechanisms.items():
        for mechanisms in mapping.values():
            # Conflicting distinctions on one side form a clique
            G.update(nx.complete_graph(mechanisms))
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


def all_nonconflicting_distinction_sets(distinctions, ties=False):
    """Return all maximal non-conflicting sets of distinctions.

    Arguments:
        distinctions (CauseEffectStructure): The set of distinctions to consider.

    Keyword Arguments:
        ties (bool): Whether to also consider all combinations of tied distinctions.

    Yields:
        CauseEffectStructure: A CES without conflicts.
    """
    if ties:
        for tie in tied_distinction_sets(distinctions):
            yield from _all_nonconflicting_distinction_sets(tie)
    else:
        yield from _all_nonconflicting_distinction_sets(distinctions)


def evaluate_partitions(subsystem, phi_structure, partitions):
    return extremum_with_short_circuit(
        (
            evaluate_partition(subsystem, phi_structure, partition)
            for partition in partitions
        ),
        cmp=operator.lt,
        initial=float("inf"),
        shortcircuit_value=0,
    )


_evaluate_partitions = ray.remote(evaluate_partitions)


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
DEFAULT_PARTITION_CHUNKSIZE = 500
DEFAULT_PHI_STRUCTURE_CHUNKSIZE = 50


# TODO document args
def evaluate_phi_structure(
    subsystem,
    phi_structure,
    check_trivial_reducibility=True,
    chunksize=DEFAULT_PARTITION_CHUNKSIZE,
    remote=True,
    progress=False,
):
    """Analyze the irreducibility of a PhiStructure."""
    # Realize the PhiStructure before distributing tasks
    phi_structure.realize()

    if check_trivial_reducibility and is_trivially_reducible(phi_structure):
        return _null_sia(subsystem, phi_structure)

    if remote:
        tasks = [
            _evaluate_partitions.remote(
                subsystem,
                phi_structure,
                partitions,
            )
            for partitions in partition_all(
                chunksize,
                sia_partitions(subsystem.cut_indices, subsystem.cut_node_labels),
            )
        ]
        return extremum_with_short_circuit(
            as_completed(tasks),
            cmp=operator.lt,
            initial=float("inf"),
            shortcircuit_value=0,
            shortcircuit_callback=lambda: [ray.cancel(task) for task in tasks],
        )
    else:
        partitions = sia_partitions(subsystem.cut_indices, subsystem.cut_node_labels)
        if progress:
            partitions = tqdm(partitions, desc="Partitions")
        return extremum_with_short_circuit(
            (
                evaluate_partition(subsystem, phi_structure, partition)
                for partition in partitions
            ),
            cmp=operator.lt,
            initial=float("inf"),
            shortcircuit_value=0,
        )


def evaluate_phi_structures(
    subsystem,
    phi_structures,
    **kwargs,
):
    return max(
        evaluate_phi_structure(subsystem, phi_structure, **kwargs)
        for phi_structure in phi_structures
    )


_evaluate_phi_structures = ray.remote(evaluate_phi_structures)


_compute_relations = ray.remote(compute_relations)


def max_system_intrinsic_information(phi_structures):
    return max(
        phi_structures,
        key=lambda phi_structure: phi_structure.system_intrinsic_information(),
    )


_max_system_intrinsic_information = ray.remote(max_system_intrinsic_information)


# TODO refactor into a pattern
def find_maximal_compositional_state(
    phi_structures,
    chunksize=DEFAULT_PHI_STRUCTURE_CHUNKSIZE,
    remote=True,
    progress=False,
):
    progress = config.PROGRESS_BARS or progress
    log.debug("Finding maximal compositional state...")
    if remote:
        tasks = [
            _max_system_intrinsic_information.remote(chunk)
            for chunk in tqdm(
                partition_all(chunksize, phi_structures),
                desc="Submitting compositional states for evaluation",
            )
        ]
        log.debug("Done submitting tasks.")
        results = as_completed(tasks)
        if progress:
            results = tqdm(
                results, total=len(tasks), desc="Finding maximal compositional state"
            )
        log.debug("Done finding maximal compositional state.")
        return max_system_intrinsic_information(results)
    else:
        if progress:
            phi_structures = tqdm(phi_structures, desc="Nonconflicting sets")
        return max_system_intrinsic_information(phi_structures)


def nonconflicting_phi_structures(
    all_distinctions,
    ties=False,
    all_relations=None,
    remote=True,
):
    """Yield nonconflicting PhiStructures."""
    for distinctions in all_nonconflicting_distinction_sets(
        all_distinctions, ties=ties
    ):
        if all_relations is None:
            # Compute relations on workers for each nonconflicting set
            if remote:
                relations = _compute_relations.remote(
                    all_distinctions.subsystem, distinctions
                )
            else:
                relations = compute_relations(all_distinctions.subsystem, distinctions)
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
    partition_chunksize=DEFAULT_PARTITION_CHUNKSIZE,
    progress=None,
    ties=False,
    remote=True,
):
    """Analyze the irreducibility of a system."""
    progress = config.PROGRESS_BARS or progress

    if all_distinctions is None:
        all_distinctions = compute.ces(subsystem)

    if not isinstance(all_distinctions, CauseEffectStructure):
        raise ValueError("all_distinctions must be a CauseEffectStructure")
    if isinstance(all_distinctions, FlatCauseEffectStructure):
        all_distinctions = all_distinctions.unflatten()

    # First check that the entire set of distinctions is not trivially reducible
    # (since then all subsets must be)
    full_phi_structure = PhiStructure(all_distinctions)
    if check_trivial_reducibility:
        reasons = is_trivially_reducible(full_phi_structure)
        if reasons:
            log.debug(
                "SIA is trivially-reducible; returning early.\nReasons: {%s}", reasons
            )
            return _null_sia(subsystem, full_phi_structure, reasons=reasons)

    if phi_structures is None:
        phi_structures = nonconflicting_phi_structures(
            all_distinctions,
            ties=ties,
            remote=remote,
        )

    if config.IIT_VERSION == "maximal-state-first":
        maximal_compositional_state = find_maximal_compositional_state(
            phi_structures,
            chunksize=chunksize,
            progress=progress,
            remote=remote,
        )
        log.debug("Evaluating maximal compositional state...")
        analysis = evaluate_phi_structure(
            subsystem,
            maximal_compositional_state,
            check_trivial_reducibility=check_trivial_reducibility,
            chunksize=partition_chunksize,
            remote=remote,
            progress=progress,
        )
        log.debug("Done evaluating maximal compositional state; returning SIA.")
        return analysis
    else:
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
    for subsystem in reachable_subsystems(network, network.node_indices, state):
        ces = compute.ces(subsystem)
        _sia = sia(subsystem, ces, **kwargs)
        if _sia:
            yield _sia


def major_complex(network, state, **kwargs):
    result = complexes(network, state, **kwargs)
    if result:
        return max(result)
    else:
        empty_subsystem = Subsystem(network, state, ())
        return _null_sia(empty_subsystem)
