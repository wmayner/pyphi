#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# relations.py

"""Functions for computing relations between concepts."""

import operator
from itertools import starmap

import numpy as np
from toolz import concat, curry

from pyphi import config
from pyphi.models import cmp
from pyphi.models.subsystem import CauseEffectStructure
from pyphi.utils import all_eq, powerset

# TODO: deal with Andrew's custom partition module
from .partition import Part, Tripartition, mip_partitions


# TODO test
def all_equal(seq):
    seq = list(seq)
    if not seq:
        # Vacuously
        return True
    return all(seq[0] == other for other in seq[1:])


@curry
# TODO test
def _all_maxima_or_minima(comparison, seq):
    """Return the maxima or minima of ``seq``.

    Use ``<`` as the comparison to obtain the maxima; use ``>`` as the
    comparison to obtain the minima.

    Only uses one pass through ``seq``.
    """
    seq = list(seq)
    if not seq:
        return []
    current_max = seq[0]
    maxima = [current_max]
    for element in seq[1:]:
        if comparison(current_max, element):
            maxima = [element]
            current_max = element
        elif current_max == element:
            maxima.append(element)
    return maxima


all_maxima = _all_maxima_or_minima(operator.lt)
all_minima = _all_maxima_or_minima(operator.gt)


def all_argmax(array):
    """Return the indices of the array's maxima."""
    return zip(*np.where(array == array.max()))


# TODO: rethink this
def invert_part(part):
    return Part(part.purview, part.mechanism)


# # TODO: Refactor into method on KPartition
def invert_partition(partition):
    return type(partition)(
        *map(invert_part, partition), node_labels=partition.node_labels
    )


# TODO: rename
def divergence(p, q):
    return np.abs(p * np.nan_to_num(np.log2(p / q)))


# TODO: subsystem isn't accessible from MIC or MIE; fix that
# TODO: does this account for the possibility that the divergences are
# different? is that actually possible?
def maximal_states(subsystem, ce):
    # Compute the divergence associated with each of the partitions that tied
    # for minimal small-phi
    # TODO: ENSURE THAT THE SAME DIVERGENCE WAS USED FOR SMALL PHI
    # TODO: this involves asserting that SMALL_PHI_MEASURE == divergence
    # Otherwise, the maximal value in each of the divergences for each
    # partition might not be the same
    divergences = [
        divergence(
            ce.repertoire, subsystem.partitioned_repertoire(ce.direction, partition)
        )
        for partition in ce.ria._partlist  # TODO: check this attribute
    ]
    # Find the set of states where the divergence is maximal; the maximum is
    # taken over all tied partitions
    return set(concat(map(all_argmax, divergences)))


def possible_joint_purviews(individual_purviews):
    """Return the power set of the intersection of the purviews.

    Any candidate purview with elements that are not in one of the cause/effect
    purviews will be reducible (|small phi = 0|), so we only consider subsets
    of the purview intersection.
    """
    return powerset(set.intersection(*map(set, individual_purviews)), nonempty=True)


class Relation(cmp.Orderable):
    def __init__(self, relata, phi, purview):
        self.relata = relata
        self.phi = phi
        self.purview = purview

    @property
    def mechanisms(self):
        return [relatum.mechanism for relatum in self.relata]

    def __repr__(self):
        return f"Relation({self.mechanisms}, {self.phi}, {self.purview})"

    def __str__(self):
        return repr(self)

    def order_by(self):
        return [round(self.phi, config.PRECISION), len(self.relata)]

    def __eq__(self, other):
        attrs = ["phi", "relata"]
        return cmp.general_eq(self, other, attrs)

    @staticmethod
    def union(*relations):
        """Return the "union" of tied relations."""
        # TODO document better
        if not all_eq(r.phi for r in relations):
            raise ValueError("cannot combine relations that do not have the same phi.")
        if not all_equal(r.relata for r in relations):
            raise ValueError(
                "cannot combine relations that are not among the same relata."
            )
        tied_purviews = set(r.purview for r in relations)
        return Relation(relations[0].relata, relations[0].phi, tied_purviews)


class NullRelation(Relation):
    def __init__(self):
        # TODO: Andrew returns -1 phi for singletons; why?
        super().__init__(tuple(), -1.0, tuple())


# TODO: make class RelationPartition (equivalent to metapartition) that can
# apply itself to a MIC/MIE by generating the appropriate partition


def metapart_to_part(mechanism_index, mechanism, joint_purview, metapart):
    """Lines 148-156 in relFunc.py"""
    # TODO docstring
    purview_intersection = tuple(set(metapart.purview) & set(joint_purview))
    if mechanism_index in metapart.mechanism:
        return Part(mechanism, purview_intersection)
    return Part((), purview_intersection)


def metapartition_to_partition(
    mechanism_index, mechanism, joint_purview, metapartition
):
    "Lines 140–160 in relFunc.py"
    return Tripartition(
        *[
            metapart_to_part(mechanism_index, mechanism, joint_purview, metapart)
            for metapart in metapartition
        ]
    )


def metapartitioned_repertoire(
    subsystem, mechanism_index, ce, joint_purview, metapartition
):
    """Lines 135–165 in refFunc.py"""
    return subsystem.partitioned_repertoire(
        ce.direction,
        metapartition_to_partition(
            mechanism_index, ce.mechanism, joint_purview, metapartition
        ),
    )


# TODO: figure out if this should be max ent always, or if it should really be
# the unconstrained repertoire (which can be different than max ent for the
# effect side), as it is in andrew's code
# TODO: change to the max entropy with the right size, should NOT be
# unconstrained
def expand_joint_divergence(subsystem, joint_purview, joint_divergence, ce):
    non_joint_purview_nodes = set(ce.purview) - set(joint_purview)
    return joint_divergence * subsystem.unconstrained_repertoire(
        ce.direction, tuple(non_joint_purview_nodes)
    )


def evaluate_metapartition(
    subsystem,
    unpartitioned_repertoires,
    relata,
    metamechanism,
    joint_purview,
    metapartition,
):
    "Lines 133-220 in relFunc.py."
    partitioned_repertoires = [
        metapartitioned_repertoire(
            subsystem, mechanism_index, ce, joint_purview, metapartition
        )
        for mechanism_index, ce in zip(metamechanism, relata)
    ]
    divergences = list(
        starmap(divergence, zip(unpartitioned_repertoires, partitioned_repertoires))
    )
    joint_divergence = np.prod(divergences, axis=0)
    expanded_divergences = {
        ce: expand_joint_divergence(subsystem, joint_purview, joint_divergence, ce)
        for ce in relata
    }
    # Get the divergence at each maximal state (since there can be “ties”;
    # states that are equally maximal).
    #   For each CE, the divergence from the “perspective” of that CE is the
    #   maximal divegence among the tied maximal states; thus a state tie may
    #   be (but is not necessarily) resolved as that state which is maximally
    #   irreducible.
    # TODO: record maximal states on the object somewhere
    maximal_divergences = {
        ce: max(
            abs(expanded_divergence[state]) for state in maximal_states(subsystem, ce)
        )
        for ce, expanded_divergence in expanded_divergences.items()
    }
    relations = [
        Relation(relata, phi, joint_purview) for phi in maximal_divergences.values()
    ]
    # TODO: we can shortcut earlier; iterate over CEs with for loop?
    # The irreducibility over the 'metapartition' is given by the
    # cause/effect whose state is least divergent
    # TODO: do we want to save the min_ce on the relation object?
    # TODO: andrew's code on line 221 doesn't return all minima; why? should
    #       we? can there be ties here?
    # VV HERE this has to be all_min, not just min
    return all_min(relations)


@curry
def minimum_information_relation(subsystem, relata, joint_purview):
    # This is a tuple of mechanism meta-labels, e.g. (0,1,2) for the 1st, 2nd,
    # and 3rd involved mechanisms
    metamechanism = list(range(len(relata)))
    unpartitioned_repertoires = [
        subsystem.repertoire(ce.direction, ce.mechanism, joint_purview) for ce in relata
    ]
    relations = []
    for metapartition in mip_partitions(
        metamechanism,
        joint_purview,
        # TODO: deal with this
        relationsV=True,
    ):
        relation = evaluate_metapartition(
            subsystem,
            unpartitioned_repertoires,
            relata,
            metamechanism,
            joint_purview,
            metapartition,
        )
        relations.append(relation)
        if relation.phi == 0:
            # This relation is reducible; don't compute further
            break
    # Return the relation(s) that minimize(s) information
    return all_minima(relations)


def relation(subsystem, relata):
    # Singletons cannot have a relation
    if len(relata) == 1:
        # phi_max, relation_purview
        return NullRelation()

    candidate_joint_purviews = list(
        possible_joint_purviews(mice.purview for mice in relata)
    )

    if not candidate_joint_purviews:
        return NullRelation()

    maximal_relations = all_maxima(
        concat(
            map(
                minimum_information_relation(subsystem, relata),
                candidate_joint_purviews,
            )
        )
    )
    # Combine tied purviews into a single relation
    return Relation.union(*maximal_relations)


def separate_ces(ces):
    """Return the individual causes and effects, unpaired, from a CES."""
    return CauseEffectStructure(
        concat((concept.cause, concept.effect) for concept in ces)
    )


# TODO: add order parameter
def all_relations(subsystem, ces, order=0):
    # Relations can be over any combination of causes/effects present in the
    # CES, so we get a flat list of all causes and effects
    ces = separate_ces(ces)
    # TODO use map when/if this is part of subsystem
    # Compute all relations
    return (relation(subsystem, relata) for relata in powerset(ces, nonempty=True))


def relations(subsystem, ces):
    # Return nonzero phi relations
    return filter(lambda r: r.phi > 0, all_relations(subsystem, ces))


# TODO notes from Matteo 2019-02-20
# - object encapsulating interface to pickled concepts from CHTC for matteo and andrew
