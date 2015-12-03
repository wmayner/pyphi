#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# subsystem.py

"""
Represents a candidate set for |small_phi| calculation.
"""

import itertools
import os
from collections import namedtuple

import numpy as np
import psutil

from . import config, constants, convert, utils, validate
from .config import PRECISION
from .constants import DIRECTIONS, FUTURE, PAST
from .jsonify import jsonify
from .models import Concept, Cut, Mice, Mip, Part
from .network import list_future_purview, list_past_purview
from .node import Node

_CacheInfo = namedtuple("CacheInfo", ["hits", "misses", "size"])
HITS, MISSES = 0, 1


# TODO! go through docs and make sure to say when things can be None
# TODO: validate that purview and mechanism args are explicitly *tuples*?
class Subsystem:

    """A set of nodes in a network.

    Args:
        network (Network): The network the subsystem belongs to.
        state (tuple(int)): The state of the network.
        node_indices (tuple(int)): A sequence of indices of the nodes in this
            subsystem.

    Attributes:
        nodes (list(Node)): A list of nodes in the subsystem.
        node_indices (tuple(int)): The indices of the nodes in the subsystem.
        size (int): The number of nodes in the subsystem.
        network (Network): The network the subsystem belongs to.
        state (tuple): The current state of the subsystem. ``state[i]`` gives
            the state of node |i|.
        cut (Cut): The cut that has been applied to this subsystem.
        connectivity_matrix (np.array): The connectivity matrix after applying
            the cut.
        cut_matrix (np.array): A matrix of connections which have been severed
            by the cut.
        perturb_vector (np.array): The vector of perturbation probabilities for
            each node.
        null_cut (Cut): The cut object representing no cut.
        tpm (np.array): The TPM conditioned on the state of the external nodes.
    """

    def __init__(self, network, state, node_indices, cut=None, mice_cache=None,
                 repertoire_cache=None, cache_info=None):
        # The network this subsystem belongs to.
        self.network = network
        # The state the network is in.
        self._state = tuple(state)
        # TODO don't need map to ints anymore?
        # Remove duplicates, sort, and ensure indices are native Python `int`s
        # (for JSON serialization).
        self.node_indices = tuple(sorted(list(set(map(int, node_indices)))))
        # Get the size of this subsystem.
        self.size = len(self.node_indices)
        # Get the external nodes.
        self.external_indices = tuple(
            set(range(network.size)) - set(self.node_indices))
        # The TPM conditioned on the state of the external nodes.
        self.tpm = utils.condition_tpm(
            self.network.tpm, self.external_indices,
            self.state)
        # Validate.
        validate.subsystem(self)
        # The null cut (that leaves the system intact).
        self.null_cut = Cut((), self.node_indices)
        # The unidirectional cut applied for phi evaluation within the
        self.cut = cut if cut is not None else self.null_cut
        # Only compute hash once.
        self._hash = hash((self.network, self.node_indices, self.state,
                           self.cut))
        # Get the subsystem's connectivity matrix. This is the network's
        # connectivity matrix, but with the cut applied, and with all
        # connections to/from external nodes severed.
        self.connectivity_matrix = utils.apply_cut(
            cut, network.connectivity_matrix)
        # Get the perturbation probabilities for each node in the network
        self.perturb_vector = network.perturb_vector
        # Generate the nodes.
        self.nodes = tuple(Node(self, i) for i in self.node_indices)
        # The matrix of connections which are severed due to the cut
        self.null_cut_matrix = np.zeros((len(self), len(self)))
        self.cut_matrix = (self._find_cut_matrix(cut) if cut is not None
                           else self.null_cut_matrix)
        # A cache for keeping core causes and effects that can be reused later
        # in the event that a cut doesn't effect them.
        if mice_cache is None:
            mice_cache = dict()
        self._mice_cache = mice_cache
        # Set up cause/effect repertoire cache.
        self._repertoire_cache = (dict() if repertoire_cache is None else
                                  repertoire_cache)
        self._repertoire_cache_info = ([0, 0] if cache_info is None else
                                       cache_info)

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        # Cast state to a tuple so it can be hashed and properly used as
        # np.array indices.
        state = tuple(state)
        self._state = state
        # Validate.
        validate.subsystem(self)

    def repertoire_cache_info(self):
        """Report repertoire cache statistics."""
        return _CacheInfo(self._repertoire_cache_info[HITS],
                          self._repertoire_cache_info[MISSES],
                          len(self._repertoire_cache))

    # TODO write docstring
    def _find_cut_matrix(self, cut):
        cut_matrix = np.zeros((self.network.size, self.network.size))
        list_of_cuts = np.array(list(itertools.product(cut[0], cut[1])))
        cut_matrix[list_of_cuts[:, 0], list_of_cuts[:, 1]] = 1
        return cut_matrix[np.ix_(self.node_indices, self.node_indices)]

    def __repr__(self):
        return "Subsystem(" + repr(self.nodes) + ")"

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        """Return whether this subsystem is equal to the other object.

        Two subsystems are equal if their sets of nodes, networks, and cuts are
        equal."""
        return (set(self.node_indices) == set(other.node_indices)
                and self.state == other.state
                and self.network == other.network
                and self.cut == other.cut)

    def __bool__(self):
        """Return false if the subsystem has no nodes, true otherwise."""
        return bool(self.nodes)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __ge__(self, other):
        return len(self.nodes) >= len(other.nodes)

    def __le__(self, other):
        return len(self.nodes) <= len(other.nodes)

    def __gt__(self, other):
        return len(self.nodes) > len(other.nodes)

    def __lt__(self, other):
        return len(self.nodes) < len(other.nodes)

    def __len__(self):
        return len(self.nodes)

    def __hash__(self):
        return self._hash

    def to_json(self):
        return {
            'node_indices': jsonify(self.node_indices),
            'cut': jsonify(self.cut),
        }

    def indices2nodes(self, indices):
        """Return a tuple of Nodes for these indices.

        Raises a ValueError if the requested indices are not in the subsystem.
        """
        if not indices:
            return ()
        non_subsys_indices = set(indices) - set(self.node_indices)
        if non_subsys_indices:
            raise ValueError("Invalid indices {}. Indices must be a subset "
                             "of subsystem indices.".format(non_subsys_indices))
        return tuple(n for n in self.nodes if n.index in indices)

    def _make_repertoire_key(self, direction, mechanism, purview):
        """Make a key for looking up repertoires in the cache."""
        return (direction, mechanism, purview)

    def _get_cached_repertoire(self, direction, mechanism, purview):
        """Return a cached repertoire if there is one, ``False`` otherwise."""
        key = self._make_repertoire_key(direction, mechanism, purview)
        if key in self._repertoire_cache:
            cached = self._repertoire_cache[key]
            self._repertoire_cache_info[HITS] += 1
            return cached
        self._repertoire_cache_info[MISSES] += 1
        return None

    def _set_cached_repertoire(self, direction, mechanism, purview,
                               repertoire):
        """Store a repertoire in the cache."""
        key = self._make_repertoire_key(direction, mechanism, purview)
        if key not in self._repertoire_cache:
            self._repertoire_cache[key] = repertoire

    def cause_repertoire(self, mechanism, purview):
        """Return the cause repertoire of a mechanism over a purview.

        Args:
            mechanism (tuple(int)): The mechanism for which to calculate the
                cause repertoire.
            purview (tuple(int)): The purview over which to calculate the
                cause repertoire.
        Returns:
            cause_repertoire (``np.ndarray``): The cause repertoire of the
                mechanism over the purview.
        """
        # Return a cached repertoire if there is one.
        cached_repertoire = self._get_cached_repertoire(
            DIRECTIONS[PAST], mechanism, purview)
        if cached_repertoire is not None:
            return cached_repertoire
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # NOTE: In the Matlab version's terminology,
        #
        # "Cause repertoire" is "backward repertoire"
        # "Mechanism" is "numerator"
        # "Purview" is "denominator"
        # ``conditioned_tpm`` is ``next_num_node_distribution``
        # ``cjd`` is ``numerator_conditional_joint``
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        purview_nodes = self.indices2nodes(purview)
        mechanism_nodes = self.indices2nodes(mechanism)
        # If the purview is empty, the distribution is empty, so return the
        # multiplicative identity.
        if not purview:
            cjd = np.array([1.0])
            self._set_cached_repertoire(DIRECTIONS[PAST], mechanism, purview,
                                        cjd)
            return cjd
        # Calculate the maximum entropy distribution.
        # If the mechanism is empty, nothing is specified about the past state
        # of the purview, so just return the purview's maximum entropy
        # distribution.
        max_entropy_dist = utils.max_entropy_distribution(
            purview,
            self.network.size,
            tuple(self.perturb_vector[i] for i in purview))
        if not mechanism:
            self._set_cached_repertoire(DIRECTIONS[PAST], mechanism, purview,
                                        max_entropy_dist)
            return max_entropy_dist
        # Preallocate the mechanism's conditional joint distribution.
        # TODO extend to nonbinary nodes
        cjd = np.ones(tuple(2 if i in purview else
                            1 for i in self.network.node_indices))
        # Loop over all nodes in this mechanism, successively taking the
        # product (with expansion/broadcasting of singleton dimensions) of each
        # individual node's TPM (conditioned on that node's state) in order to
        # get the conditional joint distribution for the whole mechanism
        # (conditioned on the whole mechanism's state). After normalization,
        # this is the cause repertoire. Normalization happens after this loop.
        for mechanism_node in mechanism_nodes:
            # TODO extend to nonbinary nodes
            # We're conditioning on this node's state, so take the probability
            # table for the node being in that state.
            node_state = self.state[mechanism_node.index]
            conditioned_tpm = mechanism_node.tpm[node_state]
            # Collect the nodes that are not in the purview and have
            # connections to this node.
            # TODO: use straight indices for this.
            non_purview_inputs = (set(mechanism_node.inputs) -  # inputs
                                  set(purview_nodes))
            # Marginalize-out the non-purview inputs.
            for node in non_purview_inputs:
                conditioned_tpm = utils.marginalize_out(
                    node.index,
                    conditioned_tpm,
                    self.perturb_vector[node.index])
            # Incorporate this node's CPT into the mechanism's conditional
            # joint distribution by taking the product (with singleton
            # broadcasting, which spreads the singleton probabilities in the
            # collapsed dimensions out along the whole distribution in the
            # appropriate way.
            cjd *= conditioned_tpm
        # If the perturbation vector is not maximum entropy, then weight the
        # probabilities before normalization.
        if not np.all(self.perturb_vector == 0.5):
            cjd *= max_entropy_dist
        # Finally, normalize to get the mechanism's actual conditional joint
        # distribution.
        cjd_sum = np.sum(cjd)
        # Don't divide by zero
        if cjd_sum != 0:
            cjd /= cjd_sum
        # NOTE: we're not returning a distribution over all the nodes in the
        # network, only a distribution over the nodes in the purview. This is
        # because we never actually need to compare proper cause/effect
        # repertoires, which are distributions over the whole network; we need
        # only compare the purview-repertoires with each other, since cut vs.
        # whole comparisons are only ever done over the same purview.
        self._set_cached_repertoire(DIRECTIONS[PAST], mechanism, purview, cjd)
        return cjd

    def effect_repertoire(self, mechanism, purview):
        """Return the effect repertoire of a mechanism over a purview.

        Args:
            mechanism (tuple(int)): The mechanism for which to calculate the
                effect repertoire.

            purview (tuple(int)): The purview over which to calculate the
                effect repertoire.

        Returns:
            effect_repertoire (``np.ndarray``): The effect repertoire of the
                mechanism over the purview.
        """
        # Return a repertoire if there's a hit.
        cached_repertoire = self._get_cached_repertoire(DIRECTIONS[FUTURE],
                                                        mechanism, purview)
        if cached_repertoire is not None:
            return cached_repertoire
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # NOTE: In the Matlab version's terminology,
        #
        # "Effect repertoire" is "forward repertoire"
        # "Mechanism" is "numerator"
        # "Purview" is "denominator"
        # ``conditioned_tpm`` is ``next_denom_node_distribution``
        # ``accumulated_cjd`` is ``denom_conditional_joint``
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        purview_nodes = self.indices2nodes(purview)
        mechanism_nodes = self.indices2nodes(mechanism)
        # If the purview is empty, the distribution is empty, so return the
        # multiplicative identity.
        if not purview:
            accumulated_cjd = np.array([1.0])
            self._set_cached_repertoire(
                DIRECTIONS[FUTURE], mechanism, purview, accumulated_cjd)
            return accumulated_cjd
        # Preallocate the purview's joint distribution
        # TODO extend to nonbinary nodes
        accumulated_cjd = np.ones(
            [1] * self.network.size + [2 if i in purview else
                                       1 for i in self.network.node_indices])
        # Loop over all nodes in the purview, successively taking the product
        # (with 'expansion'/'broadcasting' of singleton dimensions) of each
        # individual node's TPM in order to get the joint distribution for the
        # whole purview. After conditioning on the mechanism's state and that
        # of external nodes, this will be the effect repertoire as a
        # distribution over the purview.
        for purview_node in purview_nodes:
            # Unlike in calculating the cause repertoire, here the TPM is not
            # conditioned yet. `tpm` is an array with twice as many dimensions
            # as the network has nodes. For example, in a network with three
            # nodes {n0, n1, n2}, the CPT for node n1 would have shape
            # (2,2,2,1,2,1). The CPT for the node being off would be given by
            # `tpm[:,:,:,0,0,0]`, and the CPT for the node being on would be
            # given by `tpm[:,:,:,0,1,0]`. The second half of the shape is for
            # indexing based on the current node's state, and the first half of
            # the shape is the CPT indexed by network state, so that the
            # overall CPT can be broadcast over the `accumulated_cjd` and then
            # later conditioned by indexing.
            # TODO extend to nonbinary nodes
            # Rotate the dimensions so the first dimension is the last (the
            # first dimension corresponds to the state of the node)
            tpm = purview_node.tpm
            tpm = tpm.transpose(list(range(tpm.ndim))[1:] + [0])
            # Expand the dimensions so the TPM can be indexed as described
            first_half_shape = list(tpm.shape[:-1])
            second_half_shape = [1] * self.network.size
            second_half_shape[purview_node.index] = 2
            tpm = tpm.reshape(first_half_shape + second_half_shape)
            # Marginalize-out non-mechanism purview inputs.
            non_mechanism_inputs = set(purview_node.inputs) - set(mechanism_nodes)
            for node in non_mechanism_inputs:
                tpm = utils.marginalize_out(node.index, tpm,
                                            self.perturb_vector[node.index])
            # Incorporate this node's CPT into the future_nodes' conditional
            # joint distribution by taking the product (with singleton
            # broadcasting).
            accumulated_cjd = accumulated_cjd * tpm
        # Now we condition on the state of mechanism nodes (by collapsing the
        # CJD onto those states):
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Collect all nodes with inputs to any purview node.
        inputs_to_purview = set.union(*[set(node.inputs) for node in purview_nodes])
        # Collect mechanism nodes with inputs to any purview node.
        fixed_inputs = convert.nodes2indices(inputs_to_purview &
                                             set(mechanism_nodes))
        # Initialize the conditioning indices, taking the slices as singleton
        # lists-of-lists for later flattening with `chain`.
        accumulated_cjd = utils.condition_tpm(
            accumulated_cjd, fixed_inputs, self.state)
        # The distribution still has twice as many dimensions as the network
        # has nodes, with the first half of the shape now all singleton
        # dimensions, so we reshape to eliminate those singleton dimensions
        # (the second half of the shape may also contain singleton dimensions,
        # depending on how many nodes are in the purview).
        accumulated_cjd = accumulated_cjd.reshape(
            accumulated_cjd.shape[self.network.size:])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Note that we're not returning a distribution over all the nodes in
        # the network, only a distribution over the nodes in the purview. This
        # is because we never actually need to compare proper cause/effect
        # repertoires, which are distributions over the whole network; we need
        # only compare the purview-repertoires with each other, since cut vs.
        # whole comparisons are only ever done over the same purview.
        self._set_cached_repertoire(
            DIRECTIONS[FUTURE], mechanism, purview, accumulated_cjd)
        return accumulated_cjd

    # TODO check if the cache is faster
    def _get_repertoire(self, direction):
        """Returns the cause or effect repertoire function based on a
        direction.

        Args:
            direction (str): The temporal direction (|past| or |future|)
                specifiying the cause or effect repertoire.

        Returns:
            repertoire_function (``function``): The cause or effect repertoire
                function.
        """
        if direction == DIRECTIONS[PAST]:
            return self.cause_repertoire
        elif direction == DIRECTIONS[FUTURE]:
            return self.effect_repertoire

    def _unconstrained_repertoire(self, direction, purview):
        """Return the unconstrained cause or effect repertoire over a
        purview."""
        return self._get_repertoire(direction)((), purview)

    def unconstrained_cause_repertoire(self, purview):
        """Return the unconstrained cause repertoire for a purview.

        This is just the cause repertoire in the absence of any mechanism.
        """
        return self._unconstrained_repertoire(DIRECTIONS[PAST], purview)

    def unconstrained_effect_repertoire(self, purview):
        """Return the unconstrained effect repertoire for a purview.

        This is just the effect repertoire in the absence of any mechanism.
        """
        return self._unconstrained_repertoire(DIRECTIONS[FUTURE], purview)

    def expand_repertoire(self, direction, purview, repertoire,
                          new_purview=None):
        """Expand a partial repertoire over a purview to a distribution
        over a new state space.

        TODO: can the purview be extrapolated from the repertoire?

        Args:
            direction (str): Either |past| or |future|
            purview (tuple(int) or None): The purview over which the repertoire 
                was calculated
            repertoire (``np.ndarray``): A repertoire computed over ``purview``

        Keyword Args:
            new_purview (tuple(int)): The purview to expand the repertoire over.
                Defaults to the entire subsystem.

        Returns:
            ``np.ndarray``: The expanded repertoire
        """
        if purview is None:
            purview = ()
        if new_purview is None:
            new_purview = self.node_indices  # full subsystem
        if not set(purview).issubset(new_purview):
            raise ValueError("Expanded purview must contain original purview.")

        # Get the unconstrained repertoire over the other nodes in the network.
        non_purview_indices = tuple(set(new_purview) - set(purview))
        uc = self._unconstrained_repertoire(direction, non_purview_indices)
        # Multiply the given repertoire by the unconstrained one to get a
        # distribution over all the nodes in the network.
        expanded_repertoire = repertoire * uc

        # Renormalize
        if (np.sum(expanded_repertoire > 0)):
            return expanded_repertoire / np.sum(expanded_repertoire)
        else:
            return expanded_repertoire

    def expand_cause_repertoire(self, purview, repertoire, new_purview=None):
        """Expand a partial cause repertoire over a purview to a distribution
        over the entire subsystem's state space."""
        return self.expand_repertoire(DIRECTIONS[PAST], purview, repertoire,
                                      new_purview)

    def expand_effect_repertoire(self, purview, repertoire, new_purview=None):
        """Expand a partial effect repertoire over a purview to a distribution
        over the entire subsystem's state space."""
        return self.expand_repertoire(DIRECTIONS[FUTURE], purview, repertoire,
                                      new_purview)

    def cause_info(self, mechanism, purview):
        """Return the cause information for a mechanism over a purview."""
        return round(utils.hamming_emd(
            self.cause_repertoire(mechanism, purview),
            self.unconstrained_cause_repertoire(purview)),
            PRECISION)

    def effect_info(self, mechanism, purview):
        """Return the effect information for a mechanism over a purview."""
        return round(utils.hamming_emd(
            self.effect_repertoire(mechanism, purview),
            self.unconstrained_effect_repertoire(purview)),
            PRECISION)

    def cause_effect_info(self, mechanism, purview):
        """Return the cause-effect information for a mechanism over a
        purview.

        This is the minimum of the cause and effect information."""
        return min(self.cause_info(mechanism, purview),
                   self.effect_info(mechanism, purview))

    # MIP methods
    # =========================================================================

    @staticmethod
    def _mip_bipartition(mechanism, purview):
        """Return all bipartitions of a mechanism over a purview.

        Returns:
            list((Part, Part))

        TODO: use ``itertools.product``??
        """
        purview_bipartitions = utils.bipartition(purview)
        # Also consider reverse or each partition, eg:
        #   [((A), (BC)), ...] -> [((BC), (A)), ...]
        reverse_bipartitions = [x[::-1] for x in purview_bipartitions]
        result = []
        for denominators in purview_bipartitions + reverse_bipartitions:
            for numerators in utils.bipartition(mechanism):
                # Exclude partitions whose numerator and
                # denominator are both empty.
                valid_partition = (
                    len(numerators[0]) + len(denominators[0]) > 0 and
                    len(numerators[1]) + len(denominators[1]) > 0)
                if valid_partition:
                    part0 = Part(mechanism=numerators[0],
                                 purview=denominators[0])
                    part1 = Part(mechanism=numerators[1],
                                 purview=denominators[1])
                    result.append((part0, part1))
        return result

    @staticmethod
    def _null_mip(direction, mechanism, purview):
        # TODO Use properties here to infer mechanism and purview from
        # partition yet access them with .mechanism and .partition
        return Mip(direction=direction,
                   mechanism=mechanism,
                   purview=purview,
                   partition=None,
                   unpartitioned_repertoire=None,
                   partitioned_repertoire=None,
                   phi=0.0)

    def find_mip(self, direction, mechanism, purview):
        """Return the minimum information partition for a mechanism over a
        purview.

        Args:
            direction (str): Either |past| or |future|.
            mechanism (tuple(int)): The nodes in the mechanism.
            purview (tuple(int)): The nodes in the purview.

        Returns:
            mip (|Mip|): The mininum-information partition in one temporal
                direction.
        """
        repertoire = self._get_repertoire(direction)

        # We default to the null MIP (the MIP of a reducible mechanism)
        null_mip = self._null_mip(direction, mechanism, purview)
        mip = null_mip

        if not purview:
            return mip

        phi_min = float('inf')
        # Calculate the unpartitioned repertoire to compare against the
        # partitioned ones
        unpartitioned_repertoire = repertoire(mechanism, purview)

        # Loop over possible MIP bipartitions
        for part0, part1 in self._mip_bipartition(mechanism, purview):
            # Find the distance between the unpartitioned repertoire and
            # the product of the repertoires of the two parts, e.g.
            #   D( p(ABC/ABC) || p(AC/C) * p(B/AB) )
            part1rep = repertoire(part0.mechanism, part0.purview)
            part2rep = repertoire(part1.mechanism, part1.purview)
            partitioned_repertoire = part1rep * part2rep

            phi = utils.hamming_emd(unpartitioned_repertoire,
                                    partitioned_repertoire)

            # Return immediately if mechanism is reducible.
            if utils.phi_eq(phi, 0):
                return Mip(direction=direction,
                           mechanism=mechanism,
                           purview=purview,
                           partition=(part0, part1),
                           unpartitioned_repertoire=unpartitioned_repertoire,
                           partitioned_repertoire=partitioned_repertoire,
                           phi=0.0)

            # Update MIP if it's more minimal. We take the bigger purview if
            # the the phi values are indistinguishable.
            if ((phi_min - phi) > constants.EPSILON or (
                    utils.phi_eq(phi_min, phi) and
                    len(purview) > len(mip.purview))):
                phi_min = phi
                # TODO Use properties here to infer mechanism and purview from
                # partition yet access them with .mechanism and .purview
                mip = Mip(direction=direction,
                          mechanism=mechanism,
                          purview=purview,
                          partition=(part0, part1),
                          unpartitioned_repertoire=unpartitioned_repertoire,
                          partitioned_repertoire=partitioned_repertoire,
                          phi=round(phi, PRECISION))
        return mip

    # TODO Don't use these internally
    def mip_past(self, mechanism, purview):
        """Return the past minimum information partition.

        Alias for |find_mip| with ``direction`` set to |past|.
        """
        return self.find_mip(DIRECTIONS[PAST], mechanism, purview)

    def mip_future(self, mechanism, purview):
        """Return the future minimum information partition.

        Alias for |find_mip| with ``direction`` set to |future|.
        """
        return self.find_mip(DIRECTIONS[FUTURE], mechanism, purview)

    def phi_mip_past(self, mechanism, purview):
        """Return the |small_phi| value of the past minimum information
        partition.

        This is the distance between the unpartitioned cause repertoire and the
        MIP cause repertoire.
        """
        mip = self.mip_past(mechanism, purview)
        return mip.phi if mip else 0

    def phi_mip_future(self, mechanism, purview):
        """Return the |small_phi| value of the future minimum information
        partition.

        This is the distance between the unpartitioned effect repertoire and
        the MIP cause repertoire.
        """
        mip = self.mip_future(mechanism, purview)
        return mip.phi if mip else 0

    def phi(self, mechanism, purview):
        """Return the |small_phi| value of a mechanism over a purview."""
        return min(self.phi_mip_past(mechanism, purview),
                   self.phi_mip_future(mechanism, purview))

    # Phi_max methods
    # =========================================================================

    def _fully_connected(self, nodes1, nodes2):
        """Tests connectivity of one set of nodes to another.

        Args:
            nodes1 (tuple(int)): The nodes whose outputs to ``nodes2`` will be
                tested.
            nodes2 (tuple(int)): The nodes whose inputs from ``nodes1`` will
                be tested.

        Returns:
            bool: Returns True if all elements in ``nodes1`` output to
                some element in ``nodes2`` AND all elements in ``nodes2``
                have an input from some element in ``nodes1``. Otherwise
                return False. Return True if either set of nodes is empty.
        """
        # If either set of nodes is empty, return (vacuously) True.
        if not nodes1 or not nodes2:
            return True
        # Apply the cut to the network's connectivity matrix.
        cm = utils.apply_cut(self.cut, self.network.connectivity_matrix)
        # Get the connectivity matrix representing the connections from the
        # first node list to the second.
        submatrix_indices = np.ix_(nodes1, nodes2)
        cm = cm[submatrix_indices]
        # Check that all nodes have at least one connection by summing over
        # rows of connectivity submatrix.
        if len(nodes1) == 1:
            return cm.sum(0).all()
        else:
            return cm.sum(0).all() and cm.sum(1).all()

    def _connections_relevant_for_mice(self, mip):
        """Return a matrix that identifies connections that “matter” to this
        concept."""
        # Get an empty square matrix the size of the network.
        cm = np.zeros((self.network.size, self.network.size))
        direction = mip.direction
        if direction == DIRECTIONS[FUTURE]:
            # Set `i, j` to 1 if `i` is a mechanism node and `j` is an effect
            # purview node.
            connections = np.array(
                list(itertools.product(mip.mechanism, mip.purview)))
            cm[connections[:, 0], connections[:, 1]] = 1
        elif direction == DIRECTIONS[PAST]:
            # Set `i, j` to 1 if `i` is a cause purview node and `j` is a
            # mechanism node.
            connections = np.array(
                list(itertools.product(mip.purview, mip.mechanism)))
            cm[connections[:, 0], connections[:, 1]] = 1
        # Return only the submatrix that corresponds to this subsystem's nodes.
        return cm[np.ix_(self.node_indices, self.node_indices)]

    def _get_cached_mice(self, direction, mechanism):
        """Return a cached MICE if there is one and the cut doesn't affect it.
        Return False otherwise."""
        if (direction, mechanism) in self._mice_cache:
            cached = self._mice_cache[(direction, mechanism)]
            if (not self.cut.splits_mechanism(mechanism)
                    and not utils.cut_mice(cached, self.cut_matrix)):
                return cached
        return False

    def find_mice(self, direction, mechanism, purviews=False):
        """Return the maximally irreducible cause or effect for a mechanism.

        Args:
            direction (str): The temporal direction (|past| or |future|)
                specifying cause or effect.
            mechanism (tuple(int)): The mechanism to be tested for
                irreducibility.

        Keyword Args:
            purviews (tuple(int)): Optionally restrict the possible purviews
                to a subset of the subsystem. This may be useful for _e.g._
                finding only concepts that are "about" a certain subset of
                nodes.
        Returns:
            mice (|Mice|): The maximally-irreducible cause or effect.

        .. note::
            Strictly speaking, the MICE is a pair of repertoires: the core
            cause repertoire and core effect repertoire of a mechanism, which
            are maximally different than the unconstrained cause/effect
            repertoires (*i.e.*, those that maximize |small_phi|). Here, we
            return only information corresponding to one direction, |past| or
            |future|, i.e., we return a core cause or core effect, not the pair
            of them.
        """
        # Return a cached MICE if there's a hit.
        cached_mice = self._get_cached_mice(direction, mechanism)
        if cached_mice:
            return cached_mice

        if purviews is False:
            # Get cached purviews if available.
            if config.CACHE_POTENTIAL_PURVIEWS:
                purviews = self.network.purview_cache[(direction, mechanism)]
            else:
                if direction == DIRECTIONS[PAST]:
                    purviews = list_past_purview(self.network, mechanism)
                elif direction == DIRECTIONS[FUTURE]:
                    purviews = list_future_purview(self.network, mechanism)
                else:
                    validate.direction(direction)
            # Filter out purviews that aren't in the subsystem and convert to
            # nodes.
            purviews = [purview for purview in purviews if
                        set(purview).issubset(self.node_indices)]

        # Filter out trivially reducible purviews.
        def not_trivially_reducible(purview):
            if direction == DIRECTIONS[PAST]:
                return self._fully_connected(purview, mechanism)
            elif direction == DIRECTIONS[FUTURE]:
                return self._fully_connected(mechanism, purview)
        purviews = tuple(filter(not_trivially_reducible, purviews))

        # Find the maximal MIP over the remaining purviews.
        if not purviews:
            maximal_mip = self._null_mip(direction, mechanism, None)
        else:
            maximal_mip = max(self.find_mip(direction, mechanism, purview) for
                              purview in purviews)

        # Identify the relevant connections for the MICE.
        if not utils.phi_eq(maximal_mip.phi, 0):
            relevant_connections = \
                self._connections_relevant_for_mice(maximal_mip)
        else:
            relevant_connections = None
        # Construct the corresponding MICE.
        mice = Mice(maximal_mip, relevant_connections)
        # Store the MICE if there was no cut, since some future cuts won't
        # effect it and it can be reused.
        key = (direction, mechanism)
        current_process = psutil.Process(os.getpid())
        not_full = (current_process.memory_percent() <
                    config.MAXIMUM_CACHE_MEMORY_PERCENTAGE)
        if (self.cut == self.null_cut
                and key not in self._mice_cache
                and not_full
                and not utils.phi_eq(mice.phi, 0)):
            self._mice_cache[key] = mice
        return mice

    def core_cause(self, mechanism, purviews=False):
        """Returns the core cause repertoire of a mechanism.

        Alias for |find_mice| with ``direction`` set to |past|."""
        return self.find_mice('past', mechanism, purviews=purviews)

    def core_effect(self, mechanism, purviews=False):
        """Returns the core effect repertoire of a mechanism.

        Alias for |find_mice| with ``direction`` set to |past|."""
        return self.find_mice('future', mechanism, purviews=purviews)

    def phi_max(self, mechanism):
        """Return the |small_phi_max| of a mechanism.

        This is the maximum of |small_phi| taken over all possible purviews."""
        return min(self.core_cause(mechanism).phi,
                   self.core_effect(mechanism).phi)

    # Big Phi methods
    # =========================================================================

    # TODO add `concept-space` section to the docs:
        # The first dimension corresponds to the direction, past or future; the
        # correspond to the subsystem's state space."""
    @property
    def null_concept(self):
        """Return the null concept of this subsystem.

        The null concept is a point in concept space identified with
        the unconstrained cause and effect repertoire of this subsystem.
        """
        # Unconstrained cause repertoire.
        cause_repertoire = self.cause_repertoire((), ())
        # Unconstrained effect repertoire.
        effect_repertoire = self.effect_repertoire((), ())
        # Null cause.
        cause = Mice(
            Mip(unpartitioned_repertoire=cause_repertoire,
                phi=0, direction=DIRECTIONS[PAST], mechanism=(),
                purview=(),
                partition=None, partitioned_repertoire=None))
        # Null mip.
        effect = Mice(
            Mip(unpartitioned_repertoire=effect_repertoire,
                phi=0, direction=DIRECTIONS[FUTURE], mechanism=(),
                purview=(),
                partition=None, partitioned_repertoire=None))
        # All together now...
        return Concept(mechanism=(), phi=0, cause=cause, effect=effect,
                       subsystem=self)

    def concept(self, mechanism, purviews=False, past_purviews=False,
                future_purviews=False):
        """Calculate a concept.
        
        See :func:`pyphi.compute.concept` for more information.
        """
        # Calculate the maximally irreducible cause repertoire.
        cause = self.core_cause(mechanism,
                                purviews=(past_purviews or purviews))
        # Calculate the maximally irreducible effect repertoire.
        effect = self.core_effect(mechanism,
                                  purviews=(future_purviews or purviews))
        # Get the minimal phi between them.
        phi = min(cause.phi, effect.phi)
        # NOTE: Make sure to expand the repertoires to the size of the
        # subsystem when calculating concept distance. For now, they must
        # remain un-expanded so the concept doesn't depend on the subsystem.
        return Concept(mechanism=mechanism, phi=phi, cause=cause,
                       effect=effect, subsystem=self)
