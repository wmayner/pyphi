#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# relations.py

"""Functions for computing relations between concepts."""

import operator
import random
from enum import Enum, auto, unique
from itertools import product
from time import time
from graphillion import setset

import numpy as np
from joblib import Parallel, delayed
from toolz import concat, curry
from tqdm.auto import tqdm

from pyphi.compute.parallel import get_num_processes
from pyphi.partition import (
    relation_partition_aggregations,
    relation_partition_one_distinction,
    relation_partition_types,
)

from . import config, validate
from .combinatorics import combinations_with_nonempty_intersection
from .data_structures import HashableOrderedSet
from .metrics.distribution import absolute_information_density
from .models import cmp, fmt
from .models.cuts import RelationPartition
from .models.subsystem import FlatCauseEffectStructure
from .registry import Registry
from .utils import eq, powerset


@unique
class ShortCircuitConditions(Enum):
    NO_OVERLAP = auto()
    NO_POSSIBLE_PURVIEWS = auto()
    RELATA_IS_SINGLETON = auto()
    RELATA_CONTAINS_DUPLICATE_PURVIEWS = auto()


class PotentialPurviewRegistry(Registry):
    """Storage for potential purview schemes registered with PyPhi.

    Users can define custom functions to determine the set of potential purviews
    for a relation:

    Examples:
        >>> @relation_potential_purviews.register('NONE')  # doctest: +SKIP
        ... def no_purviews(congruent_overlap):
        ...    return []

    And use them by setting ``config.RELATION_POTENTIAL_PURVIEWS = 'NONE'``
    """

    desc = "potential relation purviews"


relation_potential_purviews = PotentialPurviewRegistry()


@relation_potential_purviews.register("ALL")
def all_subsets(congruent_overlap):
    """Return all subsets of the congruent overlap.

    If there are multiple congruent overlaps because of ties, it is the union of
    the powerset of each.
    """
    return set(
        concat(powerset(overlap, nonempty=True) for overlap in congruent_overlap)
    )


@relation_potential_purviews.register("WHOLE")
def whole_overlap(congruent_overlap):
    """Return only the congruent overlap.

    If there are multiple congruent overlaps because of ties, it is the union of
    all of them.
    """
    return set(map(tuple, congruent_overlap))


class RelationPhiSchemeRegistry(Registry):
    """Storage for functions for evaluating a relation.

    Users can define custom functions to determine how relations are evaluated:

    Examples:
        >>> @relation_phi_schemes.register('ALWAYS_ZERO')  # doctest: +SKIP
        ... def zero(relata):
        ...    return 0

    And use them by setting ``config.RELATION_PHI_SCHEME = 'ALWAYS_ZERO'``
    """

    desc = "relation phi schemes"


relation_phi_schemes = RelationPhiSchemeRegistry()


@relation_phi_schemes.register("OVERLAP_RATIO_TIMES_RELATION_INFORMATIVENESS")
def overlap_ratio_times_relation_informativeness(relata, candidate_joint_purview):
    """Return the relation phi according to the following formula::

        min_{partitions} [ (overlap ratio) * (relation informativeness)_{partition} ]

    where:

        - ``(overlap ratio)`` is the ratio of the overlap size to the size of the
          smallest purview in the relation (i.e., the maximum-conceivable
          overlap), and
        - ``(relation informativeness)_{partition} == (sum of small phi of the
          relata) - (sum of small phi of the relata under the partition)``

    Note that this scheme implies that phi is a monotonic increasing function of
    the size of the overlap, so in practice there is no need to search over
    subsets of the congruent overlap. Thus, when this scheme is used, the most
    efficient setting for RELATION_POSSIBLE_PURVIEWS is "WHOLE".
    """
    # This implementation relies on several assumptions:
    # - Overlap ratio does not depend on the partition
    # - The minimum of the informativeness term over the set of partitions is
    #   just the minimal distinction phi, so there's no need to actually search
    #   over partitions
    overlap_ratio = len(candidate_joint_purview) / relata.minimal_purview_size
    minimal_distinction_phi = min(relatum.parent.phi for relatum in relata)
    phi = overlap_ratio * minimal_distinction_phi
    tied_partitions = [
        relation_partition_one_distinction(
            relata, candidate_joint_purview, i, node_labels=relata.subsystem.node_labels
        )
        for i in range(len(relata))
        if relata[i].parent.phi == minimal_distinction_phi
    ]
    return [
        Relation(
            relata=relata, purview=candidate_joint_purview, phi=phi, partition=partition
        )
        for partition in tied_partitions
    ]


@relation_phi_schemes.register("AGGREGATE_DISTINCTION_RELATIVE_DIFFERENCES")
def aggregate_distinction_relative_repertoire_differences(
    relata, candidate_joint_purview
):
    _partitions = list(
        partitions(
            relata, candidate_joint_purview, node_labels=relata.subsystem.node_labels
        )
    )
    return all_minima(
        Relation(
            relata=relata,
            purview=candidate_joint_purview,
            phi=relata.evaluate_partition(partition),
            partition=partition,
        )
        for partition in _partitions
    )


# TODO there should be an option to resolve ties at different levels

# TODO Requests from Matteo
# - have relations report their type (supertext, etc.; ask andrew/matteo)
# - consider refactoring subsystem reference off the concepts for serialization
# - node labelzzzzz!

# TODO notes from Matteo 2019-02-20
# - object encapsulating interface to pickled concepts from CHTC for matteo and andrew

# TODO overhaul:
# - memoize purview
# - think about whether we can just do the specification calculation once?
# - and then the for_relatum information calculation?
# - speed up hash and eq for MICE
# - make examples renaming consistent in rest of code
# - check TODOs
# - move stuff to combinatorics
# - jsonify cases
# - NodeLabels on RIA?
# - make all cut.mechanism and .purviews sets, throughout
# - fix __str__ of RelationPartition


@curry
def _all_same(comparison, seq):
    sentinel = object()
    first = next(seq, sentinel)
    if first is sentinel:
        # Vacuously
        return True
    return all(comparison(first, other) for other in seq)


# Compare equality up to precision
all_are_equal = _all_same(eq)
all_are_identical = _all_same(operator.is_)


# TODO test
@curry
def _all_extrema(comparison, seq):
    """Return the extrema of ``seq``.

    Use ``<`` as the comparison to obtain the minima; use ``>`` as the
    comparison to obtain the maxima.

    Uses only one pass through ``seq``.

    Args:
        comparison (callable): A comparison operator.
        seq (iterator): An iterator over a sequence.

    Returns:
        list: The maxima/minima in ``seq``.
    """
    extrema = []
    sentinel = object()
    current_extremum = next(seq, sentinel)
    if current_extremum is sentinel:
        # Return an empty list if the sequence is empty
        return extrema
    extrema.append(current_extremum)
    for element in seq:
        if comparison(element, current_extremum):
            extrema = [element]
            current_extremum = element
        elif element == current_extremum:
            extrema.append(element)
    return extrema


all_minima = _all_extrema(operator.lt)
all_maxima = _all_extrema(operator.gt)


# TODO(4.0) move to combinatorics
def only_nonsubsets(sets):
    """Find sets that are not proper subsets of any other set."""
    sets = sorted(map(set, sets), key=len, reverse=True)
    keep = []
    for a in sets:
        if all(not a.issubset(b) for b in keep):
            keep.append(a)
    return keep


def overlap_states(specified_states, purviews, overlap):
    """Return the specified states of only the elements in the overlap."""
    overlap = np.array(list(overlap), dtype=int)

    purviews = list(purviews)
    minimum = min(min(purview) for purview in purviews)
    maximum = max(max(purview) for purview in purviews)
    n = 1 + maximum - minimum
    # Global-state-relative index of overlap nodes
    idx = overlap - minimum

    states = []
    for state, purview in zip(specified_states, purviews):
        # Construct the specified state in a common reference frame
        global_state = np.empty([state.shape[0], n])
        relative_idx = [p - minimum for p in purview]
        global_state[:, relative_idx] = state
        # Retrieve only the overlap
        states.append(global_state[:, idx])

    return states


def congruent_overlap(specified_states, overlap):
    if not overlap:
        return []
    # Generate combinations of indices of tied states
    combination_indices = np.array(
        list(product(*(range(state.shape[0]) for state in specified_states)))
    )
    # Form a single array containing all combinations of tied states
    state_combinations = np.stack(
        [state[combination_indices[:, i]] for i, state in enumerate(specified_states)]
    )
    # Compute congruence (vectorized over combinations of ties)
    congruence = (state_combinations[0, ...] == state_combinations).all(axis=0)
    # Find the combinations where some elements are congruent
    congruent_indices, congruent_elements = congruence.nonzero()
    # Find the elements that are congruent
    congruent_subsets = set(
        tuple(elements.nonzero()[0]) for elements in congruence[congruent_indices]
    )
    # Remove any congruent overlaps that are subsets of another congruent overlap
    congruent_subsets = only_nonsubsets(map(set, congruent_subsets))
    # Convert overlap indices to purview element indices
    overlap = np.array(list(overlap))
    return [overlap[list(subset)] for subset in congruent_subsets]


def partitions(relata, candidate_joint_purview, node_labels=None):
    """Return the set of relation partitions.

    Controlled by the RELATION_PARTITION_TYPE configuration option.
    """
    return relation_partition_types[config.RELATION_PARTITION_TYPE](
        relata, candidate_joint_purview, node_labels=node_labels
    )


class Relation(cmp.Orderable):
    """A relation among causes/effects."""

    def __init__(self, relata, purview, phi, partition, ties=None):
        self._relata = relata
        self._purview = frozenset(purview)
        self._phi = phi
        self._partition = partition
        self._ties = ties

    @property
    def relata(self):
        return self._relata

    @property
    def purview(self):
        return self._purview

    @property
    def phi(self):
        return self._phi

    @property
    def partition(self):
        return self._partition

    @property
    def ties(self):
        return self._ties

    @property
    def subsystem(self):
        return self.relata.subsystem

    @property
    def degree(self):
        """The number of relata bound by this relation."""
        return len(self)

    @property
    def order(self):
        """The size of this relation's purview."""
        return len(self.purview)

    @property
    def mechanisms(self):
        return [relatum.mechanism for relatum in self.relata]

    def __repr__(self):
        return f"{type(self).__name__}(relata={self.relata}, purview={self.purview}, phi={self.phi})"

    def __str__(self):
        return repr(self)

    def __bool__(self):
        return bool(round(self.phi, config.PRECISION) > 0.0)

    def __len__(self):
        return len(self.relata)

    def __eq__(self, other):
        return cmp.general_eq(self, other, ["phi", "relata"])

    def __hash__(self):
        return hash((self.relata, self.purview, round(self.phi, config.PRECISION)))

    def order_by(self):
        # NOTE: This implementation determines the definition of ties
        return round(self.phi, config.PRECISION)

    @staticmethod
    def union(tied_relations):
        """Return the 'union' of tied relations.

        This is a new Relation object that contains the purviews of all tied
        relations in the ``ties`` attribute.
        """
        if not tied_relations:
            raise ValueError("tied relations cannot be empty")
        if not all_are_equal(r.phi for r in tied_relations):
            raise ValueError("tied relations must have the same phi")
        if not all_are_identical(r.relata for r in tied_relations):
            raise ValueError("tied relations must be among the same relata.")
        first = tied_relations[0]
        tied_purviews = set(r.purview for r in tied_relations)
        return Relation(
            first.relata, first.purview, first.phi, first.partition, ties=tied_purviews
        )

    def to_json(self):
        return {
            "relata": self.relata,
            "purview": sorted(self.purview),
            "partition": self.partition,
            "phi": self.phi,
            "ties": self.ties,
        }

    def to_indirect_json(self, ces):
        """Return an indirect representation of the Relation.

        This uses the integer indices of distinctions in the given CES rather
        than the objects themselves, which is more efficient for storage on
        disk.
        """
        return [
            self.relata.to_indirect_json(ces),
            sorted(self.purview),
            self.partition.to_indirect_json(),
            self.phi,
            [list(purview) for purview in self.ties],
        ]

    @classmethod
    def from_indirect_json(cls, ces, data):
        relata, purview, partition, phi, ties = data
        relata = Relata.from_indirect_json(ces, relata)
        return cls(
            relata,
            purview,
            phi,
            RelationPartition.from_indirect_json(
                relata, partition, node_labels=ces.subsystem.node_labels
            ),
            ties=set(map(frozenset, ties)),
        )


class NullRelation(Relation):
    """A zero-phi relation that was returned early because of a short-circuit
    condition.

    The condition is listed in the ``reason`` attribute.
    """

    def __init__(self, reason, *args, **kwargs):
        self.reason = reason
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return super().__repr__()[:-1] + f", reason={self.reason.name})"


class Relata(HashableOrderedSet):
    """A set of potentially-related causes/effects."""

    def __init__(self, subsystem, relata):
        self._subsystem = subsystem
        self._overlap = None
        self._congruent_overlap = None
        self._minimal_purview_size = None
        self._contains_duplicate_purviews_cached = None
        super().__init__(relata)
        validate.relata(self)

    @property
    def subsystem(self):
        return self._subsystem

    @property
    def mechanisms(self):
        # TODO store
        return (relatum.mechanism for relatum in self)

    @property
    def purviews(self):
        # TODO store
        return (relatum.purview for relatum in self)

    @property
    def directions(self):
        # TODO store
        return (relatum.direction for relatum in self)

    @property
    def specified_indices(self):
        return (relatum.specified_index for relatum in self)

    @property
    def specified_states(self):
        return (relatum.specified_state for relatum in self)

    def __repr__(self):
        return "Relata({" + ", ".join(map(fmt.fmt_relatum, self)) + "})"

    # TODO(4.0) pickle relations indirectly
    def __getstate__(self):
        return {
            "relata": list(self),
            "subsystem": self.subsystem,
        }

    # TODO(4.0) pickle relations indirectly
    def __setstate__(self, state):
        self.__init__(state["subsystem"], state["relata"])

    to_json = __getstate__

    @classmethod
    def from_json(cls, dct):
        return cls(dct["subsystem"], dct["relata"])

    def to_indirect_json(self, flat_ces):
        """Return an indirect representation of the Relata.

        This uses the integer indices of distinctions in the given CES rather
        than the objects themselves, which is more efficient for storage on
        disk.
        """
        if not isinstance(flat_ces, FlatCauseEffectStructure):
            raise ValueError("CES must be a FlatCauseEffectStructure")
        return [flat_ces.index(relatum) for relatum in self]

    @classmethod
    def from_indirect_json(cls, flat_ces, data):
        return cls(flat_ces.subsystem, [flat_ces[i] for i in data])

    @property
    def overlap(self):
        """The set of elements that are in the purview of every relatum."""
        if self._overlap is None:
            self._overlap = set.intersection(*map(set, self.purviews))
        return self._overlap

    @property
    def minimal_purview_size(self):
        """The size of the smallest purview in the relation."""
        if self._minimal_purview_size is None:
            self._minimal_purview_size = min(map(len, self.purviews))
        return self._minimal_purview_size

    def null_relation(self, reason, purview=None, phi=0.0, partition=None):
        # TODO(4.0): set default for partition to be the null partition
        if purview is None:
            purview = frozenset()
        return NullRelation(reason, self, purview, phi, partition)

    # TODO(4.0) allow indexing directly into relation?
    # TODO(4.0) make a property for the maximal state of the purview only
    @property
    def congruent_overlap(self):
        """Return the congruent overlap(s) among the relata.

        These are the common purview elements among the relata whose specified
        states are consistent; that is, the largest subset of the union of the
        purviews such that each relatum specifies the same state for each
        element.

        Note that there can be multiple congruent overlaps.
        """
        if self._congruent_overlap is None:
            self._congruent_overlap = congruent_overlap(
                overlap_states(
                    self.specified_states,
                    self.purviews,
                    self.overlap,
                ),
                self.overlap,
            )
        return self._congruent_overlap

    def possible_purviews(self):
        """Return all possible purviews.

        Controlled by the RELATION_POTENTIAL_PURVIEWS configuration option.
        """
        # TODO note: ties are included here
        return relation_potential_purviews[config.RELATION_POTENTIAL_PURVIEWS](
            self.congruent_overlap
        )

    def evaluate_partition_for_relatum(self, i, partition):
        """Evaluate the relation partition with respect to a particular relatum.

        If the relatum specifies multiple (tied) states, we take the maximum
        over the unpartitioned-partitioned distances for those states.

        Args:
            purview (set): The set of node indices in the purview.
            i (int): The index of relatum to consider.
        """
        relatum = self[i]
        partitioned_repertoire = self.subsystem.partitioned_repertoire(
            relatum.direction, partition.for_relatum(i)
        )
        # TODO(4.0) make this configurable?
        information = absolute_information_density(
            relatum.repertoire, partitioned_repertoire
        )
        # Take the information for only the specified states, leaving -Inf elsewhere
        specified = np.empty_like(information, dtype=float)
        specified[:] = -np.inf
        specified[relatum.specified_index] = information[relatum.specified_index]
        non_joint_purview_elements = tuple(
            element for element in relatum.purview if element not in partition.purview
        )
        # Propagate specification through to a (potentially) smaller repertoire
        # over just the joint purview
        # TODO(4.0) configuration for handling ties?
        # TODO tie breaking happens here! double-check with andrew
        specified = np.max(specified, axis=non_joint_purview_elements, keepdims=True)
        return specified

    def combine_distinction_relative_differences(self, differences):
        """Return the phi value given a difference for each distinction.

        A RelationPartition implies a distinction-relative partition for each
        distinction in the relata; this function combines the
        partitioned-unpartitioned differences across all distinctions into a phi
        value for the relation.

        Controlled by the RELATION_PARTITION_AGGREGATION configuration option.
        """
        return relation_partition_aggregations[config.RELATION_PARTITION_AGGREGATION](
            differences, axis=0
        )

    def evaluate_partition(self, partition):
        specified = np.stack(
            [
                self.evaluate_partition_for_relatum(i, partition)
                for i in range(len(self))
            ]
        )
        # Aggregate across relata; any non-specified states will propagate
        # -np.inf through the aggregation, leaving only tied congruent states
        # Then we take the max across tied congruent states
        return np.max(self.combine_distinction_relative_differences(specified))

    # TODO: do we care about ties here?
    # 2019-05-30: no, according to andrew
    def minimum_information_relation(self, candidate_joint_purview):
        """Return the minimal information relation for this purview.

        Behavior is controlled by the RELATION_PHI_SCHEME configuration option.

        Arguments:
            candidate_joint_purview (set): The purview to consider (subset of
                the overlap).
        """
        if not self.overlap:
            return self.null_relation(
                reason=ShortCircuitConditions.NO_OVERLAP,
                purview=candidate_joint_purview,
                phi=0,
            )
        return relation_phi_schemes[config.RELATION_PHI_SCHEME](
            self, candidate_joint_purview
        )

    def _contains_duplicate_purviews(self):
        seen = set()
        for relatum in self:
            purview = (relatum.direction, relatum.purview)
            if purview in seen:
                return True
            seen.add(purview)
        return False

    @property
    def contains_duplicate_purviews(self):
        """Return whether there are duplicate purviews (in the same direction)."""
        if self._contains_duplicate_purviews_cached is None:
            self._contains_duplicate_purviews_cached = (
                self._contains_duplicate_purviews()
            )
        return self._contains_duplicate_purviews_cached

    def maximally_irreducible_relation(self):
        """Return the maximally-irreducible relation among these relata.

        If there are ties, the tied relations will be recorded in the 'ties'
        attribute of the returned relation.

        Returns:
            Relation: the maximally irreducible relation among these relata,
            with any tied purviews recorded.
        """
        # Short-circuit conditions
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Singletons cannot have relations
        if len(self) == 1:
            return self.null_relation(
                reason=ShortCircuitConditions.RELATA_IS_SINGLETON,
            )
        # Because of the constraint that there are no duplicate purviews in a
        # compositional state, relations among relata with duplicate purviews
        # never occur. This is the default setting, but the user can change this
        # to investigate relations in that case if desired.
        if (
            not config.RELATION_ALLOW_DUPLICATE_PURVIEWS
            and self.contains_duplicate_purviews
        ):
            return self.null_relation(
                reason=ShortCircuitConditions.RELATA_CONTAINS_DUPLICATE_PURVIEWS,
            )
        # There must be at least one possible purview
        possible_purviews = self.possible_purviews()
        if not possible_purviews:
            return self.null_relation(
                reason=ShortCircuitConditions.NO_POSSIBLE_PURVIEWS,
            )
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Find maximal relations
        tied_relations = all_maxima(
            concat(
                self.minimum_information_relation(candidate_joint_purview)
                for candidate_joint_purview in possible_purviews
            )
        )
        # Keep track of ties
        return Relation.union(tied_relations)


def relation(relata):
    """Return the maximally irreducible relation among the given relata.

    Alias for the ``Relata.maximally_irreducible_relation()`` method.
    """
    return relata.maximally_irreducible_relation()


def all_relata(subsystem, ces, min_degree=2, max_degree=None):
    """Return all relata in the CES, even if they have no ovelap."""
    if min_degree < 2:
        # Relations are necessarily degree 2 or higher
        min_degree = 2
    for subset in powerset(ces, min_size=min_degree, max_size=max_degree):
        yield Relata(subsystem, subset)


def potential_relata(subsystem, ces, min_degree=2, max_degree=None):
    """Return Relata with nonempty overlap."""
    purviews = list(map(frozenset, ces.purviews))
    for combination in combinations_with_nonempty_intersection(
        purviews, min_size=min_degree, max_size=max_degree
    ):
        yield Relata(subsystem, tuple(ces[i] for i in combination))


# TODO: change to candidate_relations?
def all_relations(
    subsystem, ces, parallel=False, parallel_kwargs=None, progress=True, **kwargs
):
    """Return all relations, even those with zero phi."""
    # Relations can be over any combination of causes/effects in the CES, so we
    # get a flat list of all causes and effects
    ces = FlatCauseEffectStructure(ces)
    relata = list(potential_relata(subsystem, ces, **kwargs))
    if progress:
        relata = tqdm(relata)
    # Compute all relations
    n_jobs = get_num_processes()
    parallel_kwargs = {
        "n_jobs": n_jobs,
        "batch_size": max(len(relata) // (n_jobs - 1), 1),
        **(parallel_kwargs if parallel_kwargs else dict()),
    }
    if parallel:
        return Parallel(**parallel_kwargs)(map(delayed(relation), relata))
    return map(relation, relata)


class Relations:
    """A collection of relations."""

    def _sum_phi(self):
        raise NotImplementedError

    def sum_phi(self):
        """The sum of small phi of these relations."""
        if not getattr(self, "_sum_phi_cached", False):
            self._sum_phi_cached = self._sum_phi()
        return self._sum_phi_cached

    def supported_by(self):
        """Return only relations that are supported by the given CES."""
        raise NotImplementedError


class ConcreteRelations(HashableOrderedSet, Relations):
    """A concrete set of relations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def mechanisms(self):
        for relation in self:
            yield relation.mechanisms

    @property
    def purviews(self):
        for relation in self:
            yield relation.purviews

    @property
    def phis(self):
        for relation in self:
            yield relation.phi

    def _sum_phi(self):
        return sum(self.phis)

    def supported_by(self, distinctions):
        # Special case for empty distinctions, for speed
        if not distinctions:
            relations = []
        else:
            # TODO use lattice data structure for efficiently finding the union of
            # the lower sets of lost distinctions
            relations = [
                relation
                for relation in self
                if all(
                    distinction in FlatCauseEffectStructure(distinctions)
                    for distinction in relation.relata
                )
            ]
        return type(self)(relations)

    def to_json(self):
        return list(self)

    def to_indirect_json(self, ces):
        return [relation.to_indirect_json(ces) for relation in self]

    @classmethod
    def from_indirect_json(cls, ces, data):
        return cls(
            [Relation.from_indirect_json(ces, relation_data) for relation_data in data]
        )


class RelationApproximationRegistry(Registry):
    """Storage for approximations of the sum of relation phis.

    Users can define custom schemes:

    Examples:
        >>> @relation_approximations.register('NONE')  # doctest: +SKIP
        ... class NoRelations(ApproximateRelations):
        ...     def _sum_phi(self):
        ...         return 0

    And use them by setting ``config.RELATION_SUM_PHI_APPROXIMATION = 'NONE'``
    """

    desc = "approximations of relations"


relation_approximations = RelationApproximationRegistry()


# TODO(4.0) to_json method
class ApproximateRelations(Relations):
    def __init__(self, distinctions):
        self.distinctions = distinctions

    def supported_by(self, distinctions):
        return type(self)(distinctions)

    def to_json(self):
        return self.__dict__


@relation_approximations.register("ANALYTICAL_DEGREE_ALL")
class AnalyticalRelations(ApproximateRelations):
    DEGREE = None

    @property
    def purview_inclusion(self):
        return self.distinctions.purview_inclusion(degree=self.DEGREE)

    def mean_phi(self):
        """This approximation assumes all relation |small_phi| = 1."""
        return 1.0

    def _sum_phi(self):
        # Keiko's approximate formula
        # Assumes:
        #   - small phi = 1 for all relations
        #   - all overlaps are congruent
        return self.mean_phi() * sum(
            (-1) ** (len(subset) - 1) * (2 ** num_purviews)
            for subset, num_purviews in self.purview_inclusion.items()
        )


@relation_approximations.register("ANALYTICAL_DEGREE_ONE")
class AnalyticalRelationsDegreeOne(AnalyticalRelations):
    DEGREE = 1


@relation_approximations.register("SAMPLED_DEGREE_ALL")
class SampledRelations(AnalyticalRelations):
    """Use the analytical approximations, but weight by the average phi of a set
    of sampled relations which are computed exactly.
    """

    DEGREE = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sample = None
        self._max_degree = None

    @classmethod
    def from_json(cls, data):
        instance = cls(data["distinctions"])
        instance.__dict__.update(data)
        return instance

    @property
    def max_degree(self):
        if self._max_degree is None:
            self._max_degree = max(self.purview_inclusion.values(), default=0)
        return self._max_degree

    @staticmethod
    def nonempty_intersection(relata):
        return set.intersection(*(set(distinction.purview) for distinction in relata))

    def draw_sample(self, R_iter):
        while True:
            relata = next(R_iter, None)
            if relata is None:
                return
            if self.nonempty_intersection(relata):
                return Relata(
                    self.distinctions.subsystem, relata
                ).maximally_irreducible_relation()

    def draw_samples(self, potential_relata, sample_size, target_degrees):
        setset.set_universe(potential_relata)

        # Power set of potential relata
        R = ~setset([[]])
        # Cannot have singleton relata
        R -= R.set_size(1)

        if target_degrees is None:
            R_target = R
        else:
            R_target = setset([])
            for degree in target_degrees:
                R_target |= R.set_size(degree)

        R_iter = R_target.rand_iter()
        sample = []
        start = time()
        while (len(sample) < sample_size) and (
            time() < start + config.RELATION_APPROXIMATION_SAMPLE_TIMEOUT
        ):
            draw = self.draw_sample(R_iter)
            if draw is None:
                break
            sample.append(draw)
        return sample

    def sample(self, target_degrees=None):
        if self._sample is None:
            if target_degrees is None:
                # Most numerous order is ([max_degree] choose [max_degree // 2])
                target_degrees = [self.max_degree // 2]
            if any(degree > self.max_degree for degree in target_degrees):
                raise ValueError(
                    f"all degrees must be < max degree = {self.max_degree}; got {target_degrees}"
                )
            # if any(degree < 2 for degree in target_degrees):
            #     raise ValueError(f"all degrees must be > 2; got {target_degrees}")
            # TODO(4.0) generalize to list of target degrees; need another class var?
            potential_relata = FlatCauseEffectStructure(self.distinctions)
            self._sample = self.draw_samples(
                potential_relata,
                config.RELATION_APPROXIMATION_SAMPLE_SIZE,
                target_degrees,
            )
        return self._sample

    def mean_phi(self):
        if not self.sample():
            # Either empty distinctions, or sampling failed
            return 0.0
        return np.mean([relation.phi for relation in self.sample()])


@relation_approximations.register("SAMPLED_DEGREE_ONE")
class SampledRelationsDegreeOne(SampledRelations):
    DEGREE = 1


class RelationComputationsRegistry(Registry):
    """Storage for functions for computing relations.

    Users can define custom schemes:

    Examples:
        >>> @relation_computations.register('NONE')  # doctest: +SKIP
        ... def no_relations(subsystem, ces):
        ...    return Relations([])

    And use them by setting ``config.RELATION_COMPUTATIONS = 'NONE'``
    """

    desc = "approximations of sum of relation phi"


relation_computations = RelationComputationsRegistry()


@relation_computations.register("APPROXIMATE")
def approximate_relations(subsystem, distinctions, **kwargs):
    return relation_approximations[config.RELATION_APPROXIMATION](
        distinctions.unflatten()
    )


@relation_computations.register("EXACT")
def concrete_relations(subsystem, distinctions, **kwargs):
    return ConcreteRelations(
        filter(None, all_relations(subsystem, distinctions, **kwargs))
    )


def relations(subsystem, distinctions, **kwargs):
    """Return the irreducible relations among the causes/effects in the CES."""
    return relation_computations[config.RELATION_COMPUTATION](
        subsystem, FlatCauseEffectStructure(distinctions), **kwargs
    )
