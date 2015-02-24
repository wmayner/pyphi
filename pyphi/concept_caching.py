#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# concept_caching.py
"""
Objects and functions for managing the normalization, caching, and retrieval of
concepts.
"""

from collections import namedtuple
import numpy as np
from marbl import MarblSet

from . import utils, models, db, constants, config, convert
from .constants import DIRECTIONS, PAST, FUTURE


class NormalizedMechanism:

    """A mechanism rendered into a normal form, suitable for use as a cache key
    in concept memoization.

    The broad outline for the normalization procedure is as follows:

    - Get the set of all nodes that input to (output from) at least one
      mechanism node.
    - Sort the Marbls in a stable way (this is done by marbl-python, when the
      MarblSet is initialized).
    - Iterate over the sorted Marbls; for each one, iterate over its
      corresponding mechanism node's inputs (outputs).
    - Label each input (output) with a unique integer. These are the
      "normalized indices" of the inputs (outputs).
    - Record the inverse mapping, which sends a normalized index to a real
      index.
    - Record the state of the mechanism and all input/ouptut nodes.

    Then two normalized mechanisms are the same if they have the same
    MarblSets, inputs, outputs, state, and input/output state.

    Attributes:
        marblset (MarblSet): A dictionary where keys are directions, and values
            are MarblSets containing Marbls generated from the TPMs of the
            mechanism nodes' corresponding to the direction.
        normalized_indices (dict): A dictionary where keys are directions, and
            values are dictionaries mapping mechanism node indices to their
            normalized indices for that direction.
        unnormalized_indices (dict): The inverse of ``normalized_indices``.
        inputs (tuple(tuple(int))): A tuple where the |ith| element contains a
            tuple of the normalized indices of the |ith| node, where |i| is a
            normalized index.
        outputs (tuple(tuple(int))): The same as ``inputs``, but the elements
            contain normalized indices of the outputs.
        permutation (dict()): A dictionary where the keys are directions and
            the values are the permutations that maps mechanism nodes to the
            position of their marbl in the marblset for that direction.
    """
    # NOTE: We use lists and indices throughout, instead of dictionaries (which
    # would perhaps be more elegant), to avoid repeatedly computing the hash of
    # the marbls.
    def __init__(self, mechanism, subsystem, normalize_tpms=True):
        # Ensure the mechanism is in sorted order for consistency.
        mechanism = sorted(mechanism)
        self.indices = convert.nodes2indices(mechanism)
        # Record the state of the mechanism.
        self.state = tuple(n.state for n in mechanism)
        # Grab the marbls from the mechanism nodes.
        marbls = {
            DIRECTIONS[PAST]: [(n.past_marbl if normalize_tpms else
                                n.raw_past_marbl) for n in mechanism],
            DIRECTIONS[FUTURE]: [(n.current_marbl if normalize_tpms else
                                  n.raw_current_marbl) for n in mechanism]
        }
        self.marblset = {
            direction: MarblSet(marbls[direction]) for direction in DIRECTIONS
        }
        self.permutation = {
            direction: self.marblset[direction].permutation
            for direction in DIRECTIONS
        }
        M = range(len(mechanism))
        # Associate marbls in the marblset to the mechanism nodes they were
        # generated from.
        marbl_preimage = {
            direction: [
                # The ith marbl corresponds to the jth node in the mechanism,
                # where j is the image of i under the marblset's permutation.
                mechanism[self.permutation[direction][i]]
                for i, marbl in enumerate(self.marblset[direction])
            ] for direction in DIRECTIONS
        }
        # Associate each marbl to the inputs of its preimage node.
        io = {
            # Inputs
            DIRECTIONS[PAST]: [
                marbl_preimage[DIRECTIONS[PAST]][m].inputs for m in M],
            # Outputs
            DIRECTIONS[FUTURE]: [
                marbl_preimage[DIRECTIONS[FUTURE]][m].outputs for m in M]
        }
        # Now, we generate the normalized index of each node that inputs
        # (outputs) to at least one mechanism node. Also record the reverse
        # mapping, that sends normalized indices to the original indices.
        #
        # This is done by iterating through the marbls' inputs (outputs) and
        # assigning each new node we encounter a unique integer label.
        #
        # So, for example, if the preimage nodes of the first and second marbl
        # share an input node i, node i will be assigned a label only once,
        # when the first marbl's inputs are encountered. When that node i is
        # encountered again while iterating through the inputs of the second
        # marbl's preimage, it will not be assigned a new label.
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # This will hold a mapping from the indices of the nodes of the inputs
        # to the mechanism that the NormalizedMechanism was initialized with to
        # their normalized indices.
        self.normalized_indices = {
            DIRECTIONS[PAST]: {},
            DIRECTIONS[FUTURE]: {}
        }
        # This will be used to label newly-encountered inputs/outpus.
        counter = {DIRECTIONS[PAST]: 0, DIRECTIONS[FUTURE]: 0}
        for d in DIRECTIONS:
            for m in M:
                # Assign each of the marbl's inputs (outputs) a label if it
                # hasn't been labeled already.
                for node in io[d][m]:
                    if node.index not in self.normalized_indices[d]:
                        normal_index = counter[d]
                        # Assign the next unused integer as the label.
                        self.normalized_indices[d][node.index] = normal_index
                        # Increment the counter so the next label is different.
                        counter[d] += 1
        # Get the inverse mappings.
        self.unnormalized_indices = {
            DIRECTIONS[PAST]: {
                v: k for k, v in
                self.normalized_indices[DIRECTIONS[PAST]].items()},
            DIRECTIONS[FUTURE]: {
                v: k for k, v in
                self.normalized_indices[DIRECTIONS[FUTURE]].items()} }
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Get the states of the input and output nodes.
        self.io_state = tuple(
            tuple(subsystem.network.current_state[i] for i in
                  self.normalized_indices[d].keys())
            for d in DIRECTIONS
        )
        # Associate each marbl with its normally-labeled inputs.
        # This captures the interrelationships between the mechanism nodes in a
        # stable way.
        self.inputs = tuple(
            tuple(self.normalized_indices[DIRECTIONS[PAST]][n.index]
                  for n in io[DIRECTIONS[PAST]][m])
            for m in M)
        self.outputs = tuple(
            tuple(self.normalized_indices[DIRECTIONS[FUTURE]][n.index]
                  for n in io[DIRECTIONS[FUTURE]][m])
            for m in M)

    # TODO!!!: make hash independent of python
    def __hash__(self):
        return hash((self.marblset[DIRECTIONS[PAST]],
                     self.marblset[DIRECTIONS[FUTURE]],
                     self.inputs, self.outputs,
                     self.state, self.io_state))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __str__(self):
        return str(self.indices)

    def __repr__(self):
        return str(self)


# A simple container for Mice data without the nested Mip structure.
_NormalizedMice = namedtuple('NormalizedMice', ['phi', 'direction',
                                                'mechanism', 'purview',
                                                'repertoire'])


class NormalizedMice(_NormalizedMice):

    """A lightweight container for MICE data.

    See :class:`pyphi.models.Mice` for its unnormalized counterpart.

    Attributes:
        phi (float):
            The difference between the mechanism's unpartitioned and
            partitioned repertoires.
        direction (str):
            Either 'past' or 'future'. If 'past' ('future'), this represents a
            maximally irreducible cause (effect).
        mechanism (tuple(int)):
            The normalized indices of the MICE's mechanism.
        purview (tuple(int)):
            A normalized purview. This is a tuple of the normalized indices of
            its nodes.
        repertoire (np.ndarray):
            The normalized unpartitioned repertoire of the mechanism over the
            purview. A repertoire is normalized by squeezing and then
            reordering its dimensions so they correspond to the normalized
            purview.
    """

    pass


def _normalize_purview_and_repertoire(purview, repertoire, normalized_indices):
    """Return a normalized purview and repertoire.

    A normalized purview is a tuple of the normalized indices of its nodes, and
    a normalized repertoire is obtained by squeezing and then reordering its
    dimensions so they correspond to the normalized purview.
    """
    # Ensure that the purview nodes are in the same order as their dimensions
    # in the repertoire.
    purview = sorted(purview)
    # Get the normalized indices of the purview nodes. Sort it to ensure that
    # the indices are in the same order as their dimensions in the normalized
    # repertoire.
    normalized_purview = tuple(sorted(normalized_indices[n.index] for n in
                                      purview))
    # If the repertoire is None, from a null MIP, return immediately.
    if repertoire is None:
        return normalized_purview, repertoire
    # Get the permutation of the purview nodes' indices that sends them to the
    # sorted list of their normalized indices.
    L = [(normalized_indices[n.index], i) for i, n in enumerate(purview)]
    L.sort()
    _, permutation = zip(*L)
    # Permute and squeeze the dimensions of the repertoire so they are ordered
    # by their corresponding purview nodes' normalized indices.
    normalized_repertoire = repertoire.squeeze().transpose(permutation)
    # Return the normalized purview and repertoire.
    return normalized_purview, normalized_repertoire


def _normalize_mice(direction, mice, normalized_mechanism):
    normalized_indices = normalized_mechanism.normalized_indices[direction]
    # Get the normalized purview and repertoire
    purview, repertoire = _normalize_purview_and_repertoire(
        mice.purview, mice.repertoire, normalized_indices)
    return NormalizedMice(
        phi=mice.phi,
        direction=mice.direction,
        mechanism=mice.mechanism,
        purview=purview,
        repertoire=repertoire)


class NormalizedConcept:

    """A precomputed concept in a form suitable for memoization.

    Upon initialization, the normal form of the concept to be cached is
    computed, and data relating its cause and effect purviews are stored,
    which the concept to be reconstituted in a different subsystem.

    Attributes:
        mechanism (NormalizedMechanism): The mechanism the concept consists of.
        phi (float): The |phi| value of the concept.
        cause (NormalizedMice): The concept's normalized core cause.
        effect (NormalizedMice): The concept's normalized core effect.
    """

    # TODO put in phi values

    def __init__(self, normalized_mechanism, concept):
        self.mechanism = normalized_mechanism
        self.phi = concept.phi
        self.cause = _normalize_mice(DIRECTIONS[PAST], concept.cause,
                                     self.mechanism)
        self.effect = _normalize_mice(DIRECTIONS[FUTURE], concept.effect,
                                      self.mechanism)

    def __hash__(self):
        return hash(self.mechanism)

    def __str__(self):
        return str(self.mechanism.indices)

    def __repr__(self):
        return str(self)


def _unnormalize_purview_and_repertoire(normalized_purview,
                                        normalized_repertoire,
                                        unnormalized_indices,
                                        subsystem):
    network = subsystem.network
    # Get the unnormalized purview indices.
    purview_indices = tuple(unnormalized_indices[normal_index]
                            for normal_index in normalized_purview)
    # Get the actual purview nodes.
    purview = subsystem.indices2nodes(purview_indices)
    # If the normalized repertoire is None, from a null MIP, return
    # immediately.
    if normalized_repertoire is None:
        return purview, normalized_repertoire
    # Expand the repertoire's dimensions to fit the network dimensionality.
    new_shape = (normalized_repertoire.shape +
                 tuple([1] * (network.size - normalized_repertoire.ndim)))
    repertoire = normalized_repertoire.reshape(new_shape)
    # Get the permutation that sends the normalized repertoire's non-singleton
    # dimensions to the dimensions corresponding to the indices of the
    # unnormalized purview nodes.
    permutation = (purview_indices +
                   tuple(set(range(network.size)) - set(purview_indices)))
    # np.transpose actually takes the inverse permutation, so invert it.
    permutation = np.arange(network.size)[np.argsort(permutation)]
    # Permute the repertoires dimensions so they correspond to the unnormalized
    # purview.
    repertoire = repertoire.transpose(permutation)
    return purview, repertoire


def _unnormalize_mice(normalized_mice, normalized_mechanism, mechanism,
                      subsystem):
    """Convert a normalized MICE to its proper representation in the context of
    a subsystem.

    .. warning::
        Information about the underlying MIP's partition is lost during
        normalization since it is dependent on the specific structure of the
        subsystem in which the MIP was computed. Thus, the underlying MIPs of
        unnormalized MICE have no parition or partitioned repertoire.
    """
    # Get the unnormalized purview and repertoire.
    purview, repertoire = _unnormalize_purview_and_repertoire(
        normalized_mice.purview,
        normalized_mice.repertoire,
        normalized_mechanism.unnormalized_indices[normalized_mice.direction],
        subsystem)
    return models.Mice(models.Mip(
        phi=normalized_mice.phi,
        direction=normalized_mice.direction,
        mechanism=mechanism,
        purview=purview,
        unpartitioned_repertoire=repertoire,
        # Information about the partition is lost during normalization.
        partitioned_repertoire=None,
        partition=None
    ))


def _unnormalize(normalized_concept, normalized_mechanism, mechanism,
                 subsystem):
    """Convert a normalized concept to its proper representation in the context
    of the given subsystem."""
    cause = _unnormalize_mice(normalized_concept.cause,
                              normalized_mechanism,
                              mechanism,
                              subsystem)
    effect = _unnormalize_mice(normalized_concept.effect,
                               normalized_mechanism,
                               mechanism,
                               subsystem)
    concept = models.Concept(
        phi=normalized_concept.phi,
        mechanism=mechanism,
        cause=cause,
        effect=effect,
        subsystem=subsystem,
        normalized=normalized_concept)
    # Record that this concept was retrieved and unnormalized.
    concept.cached = True
    return concept


def _get(raw, normalized_mechanism, mechanism, subsystem):
    """Get a normalized concept from the database and unnormalize it before
    returning it."""
    key = db.generate_key(normalized_mechanism)
    normalized_concept = db.find(key)
    if normalized_concept is None:
        return None
    concept = _unnormalize(normalized_concept, normalized_mechanism, mechanism,
                           subsystem)
    return concept


def _set(normalized_mechanism, concept):
    """Normalize and store a concept with a normalized mechanism as the key."""
    key = db.generate_key(normalized_mechanism)
    value = NormalizedConcept(normalized_mechanism, concept)
    return db.insert(key, value)


def concept(subsystem, mechanism):
    """Find the concept specified by a mechanism, returning a cached value if
    one is found and computing and caching it otherwise."""
    # First we try to retrieve the concept without normalizing TPMs, which is
    # expensive.
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    raw_normalized_mechanism = NormalizedMechanism(mechanism, subsystem,
                                                   normalize_tpms=False)
    # See if we have a precomputed value without normalization.
    cached_concept = _get(True, raw_normalized_mechanism, mechanism, subsystem)
    if cached_concept is not None:
        return cached_concept
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if config.NORMALIZE_TPMS:
        # We didn't find a precomputed concept with the raw normalized TPM, so
        # now we normalize TPMs as well.
        normalized_mechanism = NormalizedMechanism(mechanism, subsystem)
        # Try to retrieve the concept with the fully-normalized mechanism.
        cached_concept = _get(False, normalized_mechanism, mechanism,
                              subsystem)
        if cached_concept is not None:
            return cached_concept
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # We didn't find any precomputed concept at all, so compute it, and store
    # the result with the raw normalized mechanism and the fully-normalized
    # mechanism as keys.
    concept = subsystem.concept(mechanism)
    _set(raw_normalized_mechanism, concept)
    if config.NORMALIZE_TPMS:
        _set(normalized_mechanism, concept)
    return concept
