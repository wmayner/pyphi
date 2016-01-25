#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# subsystem.py

"""Represents a candidate system for |small_phi| and |big_phi| evaluation."""

import functools
import itertools

import numpy as np

from . import cache, utils, validate
from .config import PRECISION
from .constants import DIRECTIONS, FUTURE, PAST
from .jsonify import jsonify
from .models import Concept, Cut, Mice, Mip, Part
from .node import Node

# Cache decorator for Subsystem repertoire methods
# TODO: if repertoire caches are never reused, there's no reason to
# have an accesible object-level cache. Just use a simple memoizer
cache_repertoire = functools.partial(cache.method_cache, '_repertoire_cache')

# Cache decorator for `Subsytem.find_mice`
cache_mice = cache.method_cache('_mice_cache')


class Subsystem:
    # TODO! go through docs and make sure to say when things can be None
    # TODO: Subsystem.cut() method, to return a cut version of Subsystem?
    """A set of nodes in a network.

    Args:
        network (Network): The network the subsystem belongs to.
        state (tuple(int)): The state of the network.
        node_indices (tuple(int)): A sequence of indices of the nodes in this
            subsystem.

    Keyword Args:
        cut (Cut): The unidirectional |Cut| to apply to this subsystem.

    Attributes:
        nodes (list(Node)): A list of nodes in the subsystem.
        node_indices (tuple(int)): The indices of the nodes in the subsystem.
        size (int): The number of nodes in the subsystem.
        network (Network): The network the subsystem belongs to.
        state (tuple(int)): The state of the subsystem's network. ``state[i]``
            gives the state of node |i|.
        proper_state (tuple(int)): The state of the subsystem.
            ``proper_state[i]`` gives the |ith| node in the subsystem. Note
            that this is **not** the state of node |i|.
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

    def __init__(self, network, state, node_indices, cut=None,
                 mice_cache=None, repertoire_cache=None):
        """Construct a Subsystem."""
        # The network this subsystem belongs to.
        self.network = network

        # Remove duplicates, sort, and ensure native Python `int`s
        # (for JSON serialization).
        self.node_indices = tuple(sorted(set(map(int, node_indices))))

        validate.state_length(state, self.network.size)

        # The state of the network.
        self._state = tuple(state)
        # The state of the subsystem.
        self._proper_state = utils.state_of(self.node_indices, self.state)

        # Get the external node indices.
        # TODO: don't expose this as an attribute?
        self.external_indices = tuple(
            set(network.node_indices) - set(self.node_indices))

        # The TPM conditioned on the state of the external nodes.
        self.tpm = utils.condition_tpm(
            self.network.tpm, self.external_indices, self.state)

        # The null cut (that leaves the system intact)
        self.null_cut = Cut((), self.node_indices)

        # The unidirectional cut applied for phi evaluation
        self.cut = cut if cut is not None else self.null_cut

        # The matrix of connections which are severed due to the cut
        # Note: this matrix is N x N, where N is the number of elements in
        # the subsystem, *not* the number of elements in the network.
        # TODO: save/memoize on the cut so we just say self.cut.matrix()?
        self.cut_matrix = self.cut.cut_matrix()

        # The network's connectivity matrix with cut applied
        self.connectivity_matrix = utils.apply_cut(
            cut, network.connectivity_matrix)

        # The perturbation probabilities for each node in the network
        self.perturb_vector = network.perturb_vector

        # Only compute hash once.
        self._hash = hash((self.network, self.node_indices, self.state,
                           self.cut))

        # Reusable cache for core causes & effects
        self._mice_cache = cache.MiceCache(self, mice_cache)

        # Cause & effect repertoire cache
        self._repertoire_cache = repertoire_cache or cache.DictCache()

        validate.subsystem(self)

        # The nodes of the subsystem
        self.nodes = tuple(Node(self, i) for i in self.node_indices)

    @property
    def state(self):
        """The state the Network this Subsystem belongs to."""
        return self._state

    @state.setter
    def state(self, state):
        # Cast state to a tuple so it can be hashed and properly used as
        # np.array indices.
        self._state = tuple(state)
        # Validate.
        validate.subsystem(self)

    @property
    def proper_state(self):
        """The state the Network this Subsystem belongs to."""
        return self._state

    @proper_state.setter
    def proper_state(self, proper_state):
        # Cast state to a tuple so it can be hashed and properly used as
        # np.array indices.
        self._proper_state = tuple(proper_state)
        # Update the network's state.
        self.state = tuple(proper_state[n] if n in self.node_indices else
                           self.state[n] for n in self.network.node_indices)
        # Validate.
        validate.subsystem(self)

    @property
    def size(self):
        """The size of this Subsystem."""
        return len(self.node_indices)

    def is_cut(self):
        """Return whether this Subsystem has a cut applied to it."""
        return self.cut != self.null_cut

    def repertoire_cache_info(self):
        """Report repertoire cache statistics."""
        return self._repertoire_cache.info()

    def __repr__(self):
        """Return a representation of this Subsystem."""
        return "Subsystem(" + repr(self.nodes) + ")"

    def __str__(self):
        """Return this Subsystem as a string."""
        return repr(self)

    def __eq__(self, other):
        """Return whether this Subsystem is equal to the other object.

        Two Subsystems are equal if their sets of nodes, networks, and cuts are
        equal.
        """
        return (set(self.node_indices) == set(other.node_indices)
                and self.state == other.state
                and self.network == other.network
                and self.cut == other.cut)

    def __bool__(self):
        """Return false if the Subsystem has no nodes, true otherwise."""
        return bool(self.nodes)

    def __ne__(self, other):
        """Return whether this Subsystem is not equal to the other object."""
        return not self.__eq__(other)

    def __ge__(self, other):
        """Return whether this Subsystem >= the other object."""
        return len(self.nodes) >= len(other.nodes)

    def __le__(self, other):
        """Return whether this Subsystem <= the other object."""
        return len(self.nodes) <= len(other.nodes)

    def __gt__(self, other):
        """Return whether this Subsystem > the other object."""
        return len(self.nodes) > len(other.nodes)

    def __lt__(self, other):
        """Return whether this Subsystem < the other object."""
        return len(self.nodes) < len(other.nodes)

    def __len__(self):
        """Return the number of nodes in this Subsystem."""
        return len(self.node_indices)

    def __hash__(self):
        """Return the hash value of this Subsystem."""
        return self._hash

    def to_json(self):
        """Return this Subsystem as a JSON object."""
        return {
            'node_indices': jsonify(self.node_indices),
            'cut': jsonify(self.cut),
        }

    def indices2nodes(self, indices):
        """Return nodes for these indices.

        Args:
            indices (iterable(int)):

        Returns:
            nodes (tuple(Node)): The |Node| objects corresponding to
                these indices.

        Raises:
            ValueError: If requested indices are not in the subsystem.
        """
        if not indices:
            return ()

        if set(indices) - set(self.node_indices):
            raise ValueError(
                "`indices` must be a subset of the Subsystem's indices.")

        return tuple(n for n in self.nodes if n.index in indices)

    @cache_repertoire(DIRECTIONS[PAST])
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

        .. note::
            The returned repertoire is a distribution over the nodes in the
            purview, not the whole network. This is because we never actually
            need to compare proper cause/effect repertoires, which are
            distributions over the whole network; we need only compare the
            purview-repertoires with each other, since cut vs. whole
            comparisons are only ever done over the same purview.
        """
        # If the purview is empty, the distribution is empty; return the
        # multiplicative identity.
        if not purview:
            return np.array([1.0])

        # If the mechanism is empty, nothing is specified about the past state
        # of the purview -- return the purview's maximum entropy distribution.
        max_entropy_dist = utils.max_entropy_distribution(
            purview, self.network.size,
            tuple(self.perturb_vector[i] for i in purview))
        if not mechanism:
            return max_entropy_dist

        # Preallocate the mechanism's conditional joint distribution.
        # TODO extend to nonbinary nodes
        cjd = np.ones(tuple(2 if i in purview else
                            1 for i in self.network.node_indices))

        # Loop over all nodes in this mechanism, successively taking the
        # product (with expansion/broadcasting of singleton dimensions) of each
        # individual node's TPM (conditioned on that node's state) in order to
        # get the conditional joint distribution for the whole mechanism
        # (conditioned on the whole mechanism's state).
        for mechanism_node in self.indices2nodes(mechanism):
            # TODO extend to nonbinary nodes
            # We're conditioning on this node's state, so take the probability
            # table for the node being in that state.
            conditioned_tpm = mechanism_node.tpm[mechanism_node.state]

            # Marginalize-out all nodes which connect to this node but which
            # are not in the purview:
            non_purview_inputs = (set(mechanism_node.input_indices) -
                                  set(purview))
            for index in non_purview_inputs:
                conditioned_tpm = utils.marginalize_out(
                    index, conditioned_tpm, self.perturb_vector[index])

            # Incorporate this node's CPT into the mechanism's conditional
            # joint distribution by taking the product (with singleton
            # broadcasting, which spreads the singleton probabilities in the
            # collapsed dimensions out along the whole distribution in the
            # appropriate way.
            cjd *= conditioned_tpm

        # If the perturbation vector is not maximum entropy, weight the
        # probabilities before normalization.
        if not np.all(self.perturb_vector == 0.5):
            cjd *= max_entropy_dist

        # Normalize conditional joint distribution.
        cjd_sum = np.sum(cjd)
        if cjd_sum != 0:  # Don't divide by zero
            cjd /= cjd_sum

        return cjd

    @cache_repertoire(DIRECTIONS[FUTURE])
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

        .. note::
            The returned repertoire is a distribution over the nodes in the
            purview, not the whole network. This is because we never actually
            need to compare proper cause/effect repertoires, which are
            distributions over the whole network; we need only compare the
            purview-repertoires with each other, since cut vs. whole
            comparisons are only ever done over the same purview.
        """
        purview_nodes = self.indices2nodes(purview)
        mechanism_nodes = self.indices2nodes(mechanism)

        # If the purview is empty, the distribution is empty, so return the
        # multiplicative identity.
        if not purview:
            return np.array([1.0])

        # Preallocate the purview's joint distribution
        # TODO extend to nonbinary nodes
        accumulated_cjd = np.ones(
            [1] * self.network.size + [2 if i in purview else
                                       1 for i in self.network.node_indices])

        # Loop over all nodes in the purview, successively taking the product
        # (with 'expansion'/'broadcasting' of singleton dimensions) of each
        # individual node's TPM in order to get the joint distribution for the
        # whole purview.
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
            non_mechanism_inputs = (set(purview_node.input_indices) -
                                    set(mechanism))
            for index in non_mechanism_inputs:
                tpm = utils.marginalize_out(index, tpm,
                                            self.perturb_vector[index])

            # Incorporate this node's CPT into the future_nodes' conditional
            # joint distribution (with singleton broadcasting).
            accumulated_cjd = accumulated_cjd * tpm

        # Collect all mechanism nodes which input to purview nodes; condition
        # on the state of these nodes by collapsing the CJD onto those states.
        mechanism_inputs = [node.index for node in mechanism_nodes
                            if set(node.output_indices) & set(purview)]
        accumulated_cjd = utils.condition_tpm(
            accumulated_cjd, mechanism_inputs, self.state)

        # The distribution still has twice as many dimensions as the network
        # has nodes, with the first half of the shape now all singleton
        # dimensions, so we reshape to eliminate those singleton dimensions
        # (the second half of the shape may also contain singleton dimensions,
        # depending on how many nodes are in the purview).
        accumulated_cjd = accumulated_cjd.reshape(
            accumulated_cjd.shape[self.network.size:])

        return accumulated_cjd

    def _get_repertoire(self, direction):
        """Return the cause or effect repertoire function based on a direction.

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
        """Return the unconstrained cause/effect repertoire over a purview."""
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

    # TODO: can the purview be extrapolated from the repertoire?
    def expand_repertoire(self, direction, purview, repertoire,
                          new_purview=None):
        """Expand a partial repertoire over a purview to a distribution over a
        new state space.

        Args:
            direction (str): Either |past| or |future|.
            purview (tuple(int) or None): The purview over which the repertoire
                was calculated.
            repertoire (``np.ndarray``): A repertoire computed over
                ``purview``.

        Keyword Args:
            new_purview (tuple(int)): The purview to expand the repertoire
                over. Defaults to the entire subsystem.

        Returns:
            ``np.ndarray``: The expanded repertoire.
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
        if expanded_repertoire.sum() > 0:
            return expanded_repertoire / expanded_repertoire.sum()
        else:
            return expanded_repertoire

    def expand_cause_repertoire(self, purview, repertoire, new_purview=None):
        """Expand a partial cause repertoire over a purview to a distribution
        over the entire subsystem's state space.
        """
        return self.expand_repertoire(DIRECTIONS[PAST], purview, repertoire,
                                      new_purview)

    def expand_effect_repertoire(self, purview, repertoire, new_purview=None):
        """Expand a partial effect repertoire over a purview to a distribution
        over the entire subsystem's state space.
        """
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
        """Return the cause-effect information for a mechanism over a purview.

        This is the minimum of the cause and effect information.
        """
        return min(self.cause_info(mechanism, purview),
                   self.effect_info(mechanism, purview))

    # MIP methods
    # =========================================================================

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
        mip = Mip._null_mip(direction, mechanism, purview)

        if not purview:
            return mip

        phi_min = float('inf')
        # Calculate the unpartitioned repertoire to compare against the
        # partitioned ones
        unpartitioned_repertoire = repertoire(mechanism, purview)

        # Loop over possible MIP bipartitions
        for part0, part1 in mip_bipartitions(mechanism, purview):
            # Find the distance between the unpartitioned repertoire and
            # the product of the repertoires of the two parts, e.g.
            #   D( p(ABC/ABC) || p(AC/C) * p(B/AB) )
            part1rep = repertoire(part0.mechanism, part0.purview)
            part2rep = repertoire(part1.mechanism, part1.purview)
            partitioned_repertoire = part1rep * part2rep

            phi = utils.hamming_emd(unpartitioned_repertoire,
                                    partitioned_repertoire)
            phi = round(phi, PRECISION)

            # Return immediately if mechanism is reducible.
            if phi == 0:
                return Mip(direction=direction,
                           mechanism=mechanism,
                           purview=purview,
                           partition=(part0, part1),
                           unpartitioned_repertoire=unpartitioned_repertoire,
                           partitioned_repertoire=partitioned_repertoire,
                           phi=0.0)

            # Update MIP if it's more minimal.
            if phi < phi_min:
                phi_min = phi
                # TODO: Use properties here to infer mechanism and purview from
                # partition yet access them with `.mechanism` and `.purview`.
                mip = Mip(direction=direction,
                          mechanism=mechanism,
                          purview=purview,
                          partition=(part0, part1),
                          unpartitioned_repertoire=unpartitioned_repertoire,
                          partitioned_repertoire=partitioned_repertoire,
                          phi=phi)
        return mip

    # TODO Don't use these internally to avoid function call overhead
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
        """Return the |small_phi| of the past minimum information partition.

        This is the distance between the unpartitioned cause repertoire and the
        MIP cause repertoire.
        """
        mip = self.mip_past(mechanism, purview)
        return mip.phi if mip else 0

    def phi_mip_future(self, mechanism, purview):
        """Return the |small_phi| of the future minimum information partition.

        This is the distance between the unpartitioned effect repertoire and
        the MIP cause repertoire.
        """
        mip = self.mip_future(mechanism, purview)
        return mip.phi if mip else 0

    def phi(self, mechanism, purview):
        """Return the |small_phi| of a mechanism over a purview."""
        return min(self.phi_mip_past(mechanism, purview),
                   self.phi_mip_future(mechanism, purview))

    # Phi_max methods
    # =========================================================================

    def _potential_purviews(self, direction, mechanism, purviews=False):
        """Return all purviews that could belong to the core cause/effect.

        Filters out trivially-reducible purviews.

        Args:
            direction ('str'): Either |past| or |future|.
            mechanism (tuple(int)): The mechanism of interest.

        Kwargs:
            purviews (tuple(int)): Optional subset of purviews of interest.
        """
        if purviews is False:
            purviews = self.network._potential_purviews(direction, mechanism)
            # Filter out purviews that aren't in the subsystem
            purviews = [purview for purview in purviews
                        if set(purview).issubset(self.node_indices)]

        def reducible(purview):
            # Returns True if purview is trivially reducible.
            # (Purviews are already filtered in network._potential_purviews
            # over the full network connectivity matrix. However, since the cm
            # is cut/smaller we check again here.)
            if direction == DIRECTIONS[PAST]:
                _from, to = purview, mechanism
            elif direction == DIRECTIONS[FUTURE]:
                _from, to = mechanism, purview
            return utils.block_reducible(self.connectivity_matrix, _from, to)

        return [purview for purview in purviews if not reducible(purview)]

    @cache_mice
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
        purviews = self._potential_purviews(direction, mechanism, purviews)

        if not purviews:
            max_mip = Mip._null_mip(direction, mechanism, None)
        else:
            max_mip = max(self.find_mip(direction, mechanism, purview)
                          for purview in purviews)

        return Mice(max_mip)

    def core_cause(self, mechanism, purviews=False):
        """Return the core cause repertoire of a mechanism.

        Alias for |find_mice| with ``direction`` set to |past|.
        """
        return self.find_mice('past', mechanism, purviews=purviews)

    def core_effect(self, mechanism, purviews=False):
        """Return the core effect repertoire of a mechanism.

        Alias for |find_mice| with ``direction`` set to |past|.
        """
        return self.find_mice('future', mechanism, purviews=purviews)

    def phi_max(self, mechanism):
        """Return the |small_phi_max| of a mechanism.

        This is the maximum of |small_phi| taken over all possible purviews.
        """
        return min(self.core_cause(mechanism).phi,
                   self.core_effect(mechanism).phi)

    # Big Phi methods
    # =========================================================================

    # TODO add `concept-space` section to the docs:
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


def mip_bipartitions(mechanism, purview):
    """Return all |small_phi| bipartitions of a mechanism over a purview.

    Excludes all bipartitions where one half is entirely empty, e.g:

         A    []                     A    []
        --- X -- is not valid,  but --- X --- is.
         B    []                    []     B

    Args:
        mechanism (tuple(int)): The mechanism to partition
        purview (tuple(int)): The purview to partition

    Returns:
        bipartitions (list(tuple((Part, Part)))): Where each partition is

            bipart[0].mechanism   bipart[1].mechanism
            ------------------- X -------------------
             bipart[0].purview     bipart[1].purview

    Example:
        >>> from pyphi.subsystem import mip_bipartitions
        >>> mechanism = (0,)
        >>> purview = (2, 3)
        >>> mip_bipartitions(mechanism, purview)
        [(Part(mechanism=(), purview=(2,)), Part(mechanism=(0,), purview=(3,))), (Part(mechanism=(), purview=(3,)), Part(mechanism=(0,), purview=(2,))), (Part(mechanism=(), purview=(2, 3)), Part(mechanism=(0,), purview=()))]
    """
    numerators = utils.bipartition(mechanism)
    denominators = utils.directed_bipartition(purview)

    return [(Part(n[0], d[0]), Part(n[1], d[1]))
            for (n, d) in itertools.product(numerators, denominators)
            if len(n[0]) + len(d[0]) > 0 and len(n[1]) + len(d[1]) > 0]
