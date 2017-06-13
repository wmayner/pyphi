#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# subsystem.py

"""Represents a candidate system for |small_phi| and |big_phi| evaluation."""

import functools
import itertools

import numpy as np

from . import cache, config, distance, distribution, utils, validate
from .constants import EMD, ENTROPY_DIFFERENCE, KLD, L1, Direction
from .distance import entropy_difference, kld, l1
from .models import (Bipartition, Concept, Cut, KPartition, Mice, Mip, Part,
                     Tripartition, _null_mip)
from .network import irreducible_purviews
from .node import generate_nodes
from .partition import (bipartition, directed_bipartition,
                        directed_bipartition_of_one, directed_tripartition,
                        k_partitions, partitions)
from .tpm import condition_tpm, marginalize_out


class Subsystem:
    # TODO! go through docs and make sure to say when things can be None
    # TODO: make subsystem attributes private and wrap them in getters?
    """A set of nodes in a network.

    Args:
        network (Network): The network the subsystem belongs to.
        state (tuple[int]): The state of the network.
        nodes (tuple[int] or tuple[str]): The nodes of the network which are in
            this subsystem. Nodes can be specified either as indices or as
            labels if the |Network| was passed ``node_labels``.

    Keyword Args:
        cut (Cut): The unidirectional |Cut| to apply to this subsystem.

    Attributes:
        network (Network): The network the subsystem belongs to.
        tpm (np.ndarray): The TPM conditioned on the state of the external
            nodes.
        cm (np.ndarray): The connectivity matrix after applying the cut.
        state (tuple[int]): The state of the network.
        nodes (tuple[Node]): The nodes of the subsystem.
        node_indices (tuple[int]): The indices of the nodes in the subsystem.
        cut (Cut): The cut that has been applied to this subsystem.
        cut_matrix (np.ndarray): A matrix of connections which have been
            severed by the cut.
        null_cut (Cut): The cut object representing no cut.
    """

    def __init__(self, network, state, nodes, cut=None,
                 mice_cache=None, repertoire_cache=None):
        # The network this subsystem belongs to.
        validate.is_network(network)
        self.network = network

        # Remove duplicates, sort, and ensure native Python `int`s
        # (for JSON serialization).
        self.node_indices = network.parse_node_indices(nodes)

        validate.state_length(state, self.network.size)

        # The state of the network.
        self.state = tuple(state)

        # Get the external node indices.
        # TODO: don't expose this as an attribute?
        self.external_indices = tuple(
            set(network.node_indices) - set(self.node_indices))

        # The TPM conditioned on the state of the external nodes.
        self.tpm = condition_tpm(
            self.network.tpm, self.external_indices, self.state)

        # The null cut (that leaves the system intact)
        self.null_cut = Cut((), self.cut_indices)

        # The unidirectional cut applied for phi evaluation
        self.cut = cut if cut is not None else self.null_cut

        # The matrix of connections which are severed due to the cut
        # Note: this matrix is N x N, where N is the number of elements in
        # the subsystem, *not* the number of elements in the network.
        # TODO: save/memoize on the cut so we just say self.cut.matrix()?
        self.cut_matrix = self.cut.cut_matrix()

        # The network's connectivity matrix with cut applied
        self.cm = self.cut.apply_cut(network.cm)

        # Only compute hash once.
        self._hash = hash((self.network, self.node_indices, self.state,
                           self.cut))

        # Reusable cache for core causes & effects
        self._mice_cache = cache.MiceCache(self, mice_cache)

        # Cause & effect repertoire cache
        # TODO: if repertoire caches are never reused, there's no reason to
        # have an accesible object-level cache. Just use a simple memoizer
        self._repertoire_cache = repertoire_cache or cache.DictCache()

        self.nodes = generate_nodes(self.tpm, self.cm, self.state,
                                    network.indices2labels(self.node_indices))

        validate.subsystem(self)

    @property
    def proper_state(self):
        """tuple[int]): The state of the subsystem.

        ``proper_state[i]`` gives the state of the |ith| node **in the
        subsystem**. Note that this is **not** the state of ``nodes[i]``.
        """
        return utils.state_of(self.node_indices, self.state)

    @property
    def connectivity_matrix(self):
        """np.ndarray: Alias for ``Subsystem.cm``."""
        return self.cm

    @property
    def size(self):
        """int: The number of nodes in the subsystem."""
        return len(self.node_indices)

    @property
    def is_cut(self):
        """bool: ``True`` if this Subsystem has a cut applied to it."""
        return self.cut != self.null_cut

    @property
    def cut_indices(self):
        """tuple[int]: The nodes of this subsystem cut for |big_phi|
        computations.

        This was added to support ``MacroSubsystem``, which cuts indices other
        than ``node_indices``.
        """
        return self.node_indices

    @property
    def tpm_size(self):
        """int: The number of nodes in the TPM."""
        return self.tpm.shape[-1]

    def repertoire_cache_info(self):
        """Report repertoire cache statistics."""
        return self._repertoire_cache.info()

    def clear_caches(self):
        """Clear the mice and repertoire caches."""
        self._repertoire_cache.clear()
        self._mice_cache.clear()

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
            'network': self.network,
            'state': self.state,
            'nodes': self.node_indices,
            'cut': self.cut,
        }

    def apply_cut(self, cut):
        """Return a cut version of this |Subsystem|.

        Args:
            cut (|Cut|): The cut to apply to this |Subsystem|.

        Returns:
            |Subsystem|
        """
        return Subsystem(self.network, self.state, self.node_indices,
                         cut=cut, mice_cache=self._mice_cache)

    def indices2nodes(self, indices):
        """Return nodes for these indices.

        Args:
            indices (tuple[int]): The indices in question.

        Returns:
            tuple[Node]: The |Node| objects corresponding to these indices.

        Raises:
            ValueError: If requested indices are not in the subsystem.
        """
        if not indices:
            return ()

        if set(indices) - set(self.node_indices):
            raise ValueError(
                "`indices` must be a subset of the Subsystem's indices.")

        return tuple(n for n in self.nodes if n.index in indices)

    def indices2labels(self, indices):
        """Returns the node labels for these indices."""
        return tuple(n.label for n in self.indices2nodes(indices))

    @cache.method('_repertoire_cache', Direction.PAST)
    def cause_repertoire(self, mechanism, purview):
        """Return the cause repertoire of a mechanism over a purview.

        Args:
            mechanism (tuple[int]): The mechanism for which to calculate the
                cause repertoire.
            purview (tuple[int]): The purview over which to calculate the
                cause repertoire.

        Returns:
            np.ndarray: The cause repertoire of the mechanism over the purview.

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
        if not mechanism:
            return distribution.max_entropy_distribution(purview,
                                                         self.tpm_size)

        # Preallocate the mechanism's conditional joint distribution.
        # TODO extend to nonbinary nodes
        cjd = np.ones(distribution.repertoire_shape(purview, self.tpm_size))

        # Loop over all nodes in this mechanism, successively taking the
        # product (with expansion/broadcasting of singleton dimensions) of each
        # individual node's TPM (conditioned on that node's state) in order to
        # get the conditional joint distribution for the whole mechanism
        # (conditioned on the whole mechanism's state).
        for mechanism_node in self.indices2nodes(mechanism):
            # TODO extend to nonbinary nodes
            # We're conditioning on this node's state, so take the probability
            # table for the node being in that state.
            tpm = mechanism_node.tpm[mechanism_node.state]

            # Marginalize-out all nodes which connect to this node but which
            # are not in the purview:
            other_inputs = set(mechanism_node.input_indices) - set(purview)
            tpm = marginalize_out(other_inputs, tpm)

            # Incorporate this node's CPT into the mechanism's conditional
            # joint distribution by taking the product (with singleton
            # broadcasting, which spreads the singleton probabilities in the
            # collapsed dimensions out along the whole distribution in the
            # appropriate way.
            cjd *= tpm

        return distribution.normalize(cjd)

    @cache.method('_repertoire_cache', Direction.FUTURE)
    def effect_repertoire(self, mechanism, purview):
        """Return the effect repertoire of a mechanism over a purview.

        Args:
            mechanism (tuple[int]): The mechanism for which to calculate the
                effect repertoire.
            purview (tuple[int]): The purview over which to calculate the
                effect repertoire.

        Returns:
            np.ndarray: The effect repertoire of the mechanism over the
            purview.

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
            [1] * self.tpm_size +
            distribution.repertoire_shape(purview, self.tpm_size))

        # Loop over all nodes in the purview, successively taking the product
        # (with 'expansion'/'broadcasting' of singleton dimensions) of each
        # individual node's TPM in order to get the joint distribution for the
        # whole purview.
        for purview_node in purview_nodes:
            # Unlike in calculating the cause repertoire, here the TPM is not
            # conditioned yet. `tpm` is an array with twice as many dimensions
            # as the network has nodes. For example, in a network with three
            # nodes {A, B, C}, the CPT for node B would have shape
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
            second_half_shape = [1] * self.tpm_size
            second_half_shape[purview_node.index] = 2
            tpm = tpm.reshape(first_half_shape + second_half_shape)

            # Marginalize-out non-mechanism inputs.
            other_inputs = set(purview_node.input_indices) - set(mechanism)
            tpm = marginalize_out(other_inputs, tpm)

            # Incorporate this node's CPT into the future_nodes' conditional
            # joint distribution (with singleton broadcasting).
            accumulated_cjd = accumulated_cjd * tpm

        # Collect all mechanism nodes which input to purview nodes; condition
        # on the state of these nodes by collapsing the CJD onto those states.
        mechanism_inputs = [node.index for node in mechanism_nodes
                            if set(node.output_indices) & set(purview)]
        accumulated_cjd = condition_tpm(
            accumulated_cjd, mechanism_inputs, self.state)

        # The distribution still has twice as many dimensions as the network
        # has nodes, with the first half of the shape now all singleton
        # dimensions, so we reshape to eliminate those singleton dimensions
        # (the second half of the shape may also contain singleton dimensions,
        # depending on how many nodes are in the purview).
        accumulated_cjd = accumulated_cjd.reshape(
            accumulated_cjd.shape[self.tpm_size:])

        return accumulated_cjd

    def _repertoire(self, direction, mechanism, purview):
        """Return the cause or effect repertoire based on a direction.

        Args:
            direction (Direction): :const:`~pyphi.constants.Direction.PAST` or
                :const:`~pyphi.constants.Direction.FUTURE`.
            mechanism (tuple[int]): The mechanism for which to calculate the
                repertoire.
            purview (tuple[int]): The purview over which to calculate the
                repertoire.

        Returns:
            np.ndarray: The cause or effect repertoire of the mechanism over
            the purview.

        Raises:
            ValueError: If ``direction`` is invalid.
        """
        if direction == Direction.PAST:
            return self.cause_repertoire(mechanism, purview)
        elif direction == Direction.FUTURE:
            return self.effect_repertoire(mechanism, purview)
        else:
            # TODO: test that ValueError is raised
            validate.direction(direction)

    def _unconstrained_repertoire(self, direction, purview):
        """Return the unconstrained cause/effect repertoire over a purview."""
        return self._repertoire(direction, (), purview)

    def unconstrained_cause_repertoire(self, purview):
        """Return the unconstrained cause repertoire for a purview.

        This is just the cause repertoire in the absence of any mechanism.
        """
        return self._unconstrained_repertoire(Direction.PAST, purview)

    def unconstrained_effect_repertoire(self, purview):
        """Return the unconstrained effect repertoire for a purview.

        This is just the effect repertoire in the absence of any mechanism.
        """
        return self._unconstrained_repertoire(Direction.FUTURE, purview)

    def partitioned_repertoire(self, direction, partition):
        """Compute the repertoire of a partitioned mechanism and purview."""
        repertoires = [
            self._repertoire(direction, part.mechanism, part.purview)
            for part in partition]

        return functools.reduce(np.multiply, repertoires)

    def expand_repertoire(self, direction, repertoire, new_purview=None):
        """Expand a partial repertoire over a purview to a distribution over a
        new state space.

        Args:
            direction (Direction): :const:`~pyphi.constants.Direction.PAST` or
                :const:`~pyphi.constants.Direction.FUTURE`.
            repertoire (np.ndarray): A repertoire.

        Keyword Args:
            new_purview (tuple[int]): The purview to expand the repertoire
                over. Defaults to the entire subsystem.

        Returns:
            np.ndarray: The expanded repertoire.

        Raises:
            ValueError: If the expanded purview doesn't contain the original
                purview.
        """
        if repertoire is None:
            return None

        purview = distribution.purview(repertoire)

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

        return distribution.normalize(expanded_repertoire)

    def expand_cause_repertoire(self, repertoire, new_purview=None):
        """Expand a partial cause repertoire over a purview to a distribution
        over the entire subsystem's state space.
        """
        return self.expand_repertoire(Direction.PAST, repertoire,
                                      new_purview)

    def expand_effect_repertoire(self, repertoire, new_purview=None):
        """Expand a partial effect repertoire over a purview to a distribution
        over the entire subsystem's state space.
        """
        return self.expand_repertoire(Direction.FUTURE, repertoire,
                                      new_purview)

    def cause_info(self, mechanism, purview):
        """Return the cause information for a mechanism over a purview."""
        return measure(Direction.PAST,
                       self.cause_repertoire(mechanism, purview),
                       self.unconstrained_cause_repertoire(purview))

    def effect_info(self, mechanism, purview):
        """Return the effect information for a mechanism over a purview."""
        return measure(Direction.FUTURE,
                       self.effect_repertoire(mechanism, purview),
                       self.unconstrained_effect_repertoire(purview))

    def cause_effect_info(self, mechanism, purview):
        """Return the cause-effect information for a mechanism over a purview.

        This is the minimum of the cause and effect information.
        """
        return min(self.cause_info(mechanism, purview),
                   self.effect_info(mechanism, purview))

    # MIP methods
    # =========================================================================

    def evaluate_partition(self, direction, mechanism, purview, partition,
                           unpartitioned_repertoire=None):
        """Return the |small_phi| of a mechanism over a purview for the given
        partition.

        Args:
            direction (Direction): :const:`~pyphi.constants.Direction.PAST` or
                :const:`~pyphi.constants.Direction.FUTURE`.
            mechanism (tuple[int]): The nodes in the mechanism.
            purview (tuple[int]): The nodes in the purview.
            partition (Bipartition): The partition to evaluate.

        Keyword Args:
            unpartitioned_repertoire (np.array): The unpartitioned repertoire.
                If not supplied, it will be computed.

        Returns:
            tuple[phi, partitioned_repertoire]: The distance between the
            unpartitioned and partitioned repertoires, and the partitioned
            repertoire.
        """
        if unpartitioned_repertoire is None:
            unpartitioned_repertoire = self._repertoire(direction, mechanism,
                                                        purview)

        partitioned_repertoire = self.partitioned_repertoire(direction,
                                                             partition)

        phi = measure(direction, unpartitioned_repertoire,
                      partitioned_repertoire)

        return (phi, partitioned_repertoire)

    def find_mip(self, direction, mechanism, purview):
        """Return the minimum information partition for a mechanism over a
        purview.

        Args:
            direction (Direction): :const:`~pyphi.constants.Direction.PAST` or
                :const:`~pyphi.constants.Direction.FUTURE`.
            mechanism (tuple[int]): The nodes in the mechanism.
            purview (tuple[int]): The nodes in the purview.

        Returns:
            |Mip|: The mininum-information partition in one temporal direction.
        """
        # We default to the null MIP (the MIP of a reducible mechanism)
        mip = _null_mip(direction, mechanism, purview)

        if not purview:
            return mip

        phi_min = float('inf')
        # Calculate the unpartitioned repertoire to compare against the
        # partitioned ones.
        unpartitioned_repertoire = self._repertoire(direction, mechanism,
                                                    purview)

        def _mip(phi, partition, partitioned_repertoire):
            # Prototype of MIP with already known data
            # TODO: Use properties here to infer mechanism and purview from
            # partition yet access them with `.mechanism` and `.purview`.
            return Mip(phi=phi,
                       direction=direction,
                       mechanism=mechanism,
                       purview=purview,
                       partition=partition,
                       unpartitioned_repertoire=unpartitioned_repertoire,
                       partitioned_repertoire=partitioned_repertoire,
                       subsystem=self)

        # State is unreachable - return 0 instead of giving nonsense results
        if (direction == Direction.PAST and
                np.all(unpartitioned_repertoire == 0)):
            return _mip(0, None, None)

        # Loop over possible MIP partitions
        for partition in mip_partitions(mechanism, purview):
            # Find the distance between the unpartitioned and partitioned
            # repertoire.
            phi, partitioned_repertoire = self.evaluate_partition(
                direction, mechanism, purview, partition,
                unpartitioned_repertoire=unpartitioned_repertoire)

            # Return immediately if mechanism is reducible.
            if phi == 0:
                return _mip(0.0, partition, partitioned_repertoire)

            # Update MIP if it's more minimal.
            if phi < phi_min:
                phi_min = phi
                mip = _mip(phi, partition, partitioned_repertoire)

        return mip

    def mip_past(self, mechanism, purview):
        """Return the past minimum information partition.

        Alias for |find_mip| with ``direction`` set to ``Direction.FUTURE``.
        """
        return self.find_mip(Direction.PAST, mechanism, purview)

    def mip_future(self, mechanism, purview):
        """Return the future minimum information partition.

        Alias for |find_mip| with ``direction`` set to |future|.
        """
        return self.find_mip(Direction.FUTURE, mechanism, purview)

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
            direction (Direction): :const:`~pyphi.constants.Direction.PAST` or
                :const:`~pyphi.constants.Direction.FUTURE`.
            mechanism (tuple[int]): The mechanism of interest.

        Keyword Args:
            purviews (tuple[int]): Optional subset of purviews of interest.
        """
        if purviews is False:
            purviews = self.network._potential_purviews(direction, mechanism)
            # Filter out purviews that aren't in the subsystem
            purviews = [purview for purview in purviews
                        if set(purview).issubset(self.node_indices)]

        # Purviews are already filtered in network._potential_purviews
        # over the full network connectivity matrix. However, since the cm
        # is cut/smaller we check again here.
        return irreducible_purviews(self.cm, direction, mechanism, purviews)

    @cache.method('_mice_cache')
    def find_mice(self, direction, mechanism, purviews=False):
        """Return the maximally irreducible cause or effect for a mechanism.

        Args:
            direction (Direction): :const:`~pyphi.constants.Direction.PAST` or
                :const:`~pyphi.constants.Direction.FUTURE`.
            mechanism (tuple[int]): The mechanism to be tested for
                irreducibility.

        Keyword Args:
            purviews (tuple[int]): Optionally restrict the possible purviews
                to a subset of the subsystem. This may be useful for _e.g._
                finding only concepts that are "about" a certain subset of
                nodes.

        Returns:
            |Mice|: The maximally-irreducible cause or effect in one temporal
            direction.

        .. note::
            Strictly speaking, the MICE is a pair of repertoires: the core
            cause repertoire and core effect repertoire of a mechanism, which
            are maximally different than the unconstrained cause/effect
            repertoires (*i.e.*, those that maximize |small_phi|). Here, we
            return only information corresponding to one direction,
            ``Direction.PAST`` or ``Direction.FUTURE``, i.e., we return a
            core cause or core effect, not the pair of them.
        """
        purviews = self._potential_purviews(direction, mechanism, purviews)

        if not purviews:
            max_mip = _null_mip(direction, mechanism, ())
        else:
            mips = [self.find_mip(direction, mechanism, purview)
                    for purview in purviews]
            max_mip = maximal_mip(mips)

        return Mice(max_mip)

    def core_cause(self, mechanism, purviews=False):
        """Return the core cause repertoire of a mechanism.

        Alias for |find_mice| with ``direction`` set to
        :const:`~pyphi.constants.Direction.PAST`.
        """
        return self.find_mice(Direction.PAST, mechanism, purviews=purviews)

    def core_effect(self, mechanism, purviews=False):
        """Return the core effect repertoire of a mechanism.

        Alias for |find_mice| with ``direction`` set to
        :const:`~pyphi.constants.Direction.FUTURE`.
        """
        return self.find_mice(Direction.FUTURE, mechanism, purviews=purviews)

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
        cause = Mice(_null_mip(Direction.PAST, (), (), cause_repertoire))
        # Null effect.
        effect = Mice(_null_mip(Direction.FUTURE, (), (), effect_repertoire))

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


def maximal_mip(mips):
    """Pick the maximal mip out of a collection."""
    if config.PICK_SMALLEST_PURVIEW:
        max_mip = max(mips, key=lambda m: (m.phi, -len(m.purview)))
    else:
        max_mip = max(mips)

    return max_mip


def mip_partitions(mechanism, purview):
    """Return a generator over all MIP partitions, based on the current
    configuration."""
    validate.partition_type(config.PARTITION_TYPE)

    func = {
        'BI': mip_bipartitions,
        'TRI': wedge_partitions,
        'ALL': all_partitions
    }[config.PARTITION_TYPE]

    return func(mechanism, purview)


def mip_bipartitions(mechanism, purview):
    """Return an generator of all |small_phi| bipartitions of a mechanism over
    a purview.

    Excludes all bipartitions where one half is entirely empty, e.g::

         A    []
        --- X --
         B    []

    is not valid, but ::

        A    []
        -- X --
        []   B

    is.

    Args:
        mechanism (tuple[int]): The mechanism to partition
        purview (tuple[int]): The purview to partition

    Yields:
        Bipartition: Where each bipartition is

        ::

            bipart[0].mechanism   bipart[1].mechanism
            ------------------- X -------------------
            bipart[0].purview     bipart[1].purview

    Example:
        >>> mechanism = (0,)
        >>> purview = (2, 3)
        >>> for partition in mip_bipartitions(mechanism, purview):
        ...     print(partition, "\\n")  # doctest: +NORMALIZE_WHITESPACE
        []   0
        -- X -
        2    3
        <BLANKLINE>
        []   0
        -- X -
        3    2
        <BLANKLINE>
        []    0
        --- X --
        2,3   []
    """
    numerators = bipartition(mechanism)
    denominators = directed_bipartition(purview)

    for n, d in itertools.product(numerators, denominators):
        if (n[0] or d[0]) and (n[1] or d[1]):
            yield Bipartition(Part(n[0], d[0]), Part(n[1], d[1]))


def wedge_partitions(mechanism, purview):
    """Return an iterator over all wedge partitions.

    These are partitions which strictly split the mechanism and allow a subset
    of the purview to be split into a third partition, eg::

        A    B   []
        -- X - X --
        B    C   D

    See ``pyphi.config.PARTITION_TYPE`` for more information.

    Args:
        mechanism (tuple[int]): A mechanism.
        purview (tuple[int]): A purview.

    Yields:
        Tripartition: all unique tripartitions of this mechanism and purview.
    """

    numerators = bipartition(mechanism)
    denominators = directed_tripartition(purview)

    yielded = set()

    for n, d in itertools.product(numerators, denominators):
        if ((n[0] or d[0]) and (n[1] or d[1]) and
           ((n[0] and n[1]) or not d[0] or not d[1])):

            # Normalize order of parts to remove duplicates.
            tripart = Tripartition(*sorted((
                Part(n[0], d[0]),
                Part(n[1], d[1]),
                Part((),   d[2]))))

            def nonempty(part):
                return part.mechanism or part.purview

            # Check if the tripartition can be transformed into a causally
            # equivalent partition by combing two of its parts; eg.
            # A/[] x B/[] x []/CD is equivalent to AB/[] x []/CD so we don't
            # include it.
            def compressible(tripart):
                pairs = [
                    (tripart[0], tripart[1]),
                    (tripart[0], tripart[2]),
                    (tripart[1], tripart[2])]

                for x, y in pairs:
                    if (nonempty(x) and nonempty(y) and
                        (x.mechanism + y.mechanism == () or
                         x.purview + y.purview == ())):
                        return True

            if not compressible(tripart) and tripart not in yielded:
                yielded.add(tripart)
                yield tripart


def all_partitions(m, p):
    m = list(m)
    p = list(p)
    mechanism_partitions = partitions(m)
    for mechanism_partition in mechanism_partitions:
        mechanism_partition.append([])
        n_mechanism_parts = len(mechanism_partition)
        max_purview_partition = min(len(p), n_mechanism_parts)
        for n_purview_parts in range(1, max_purview_partition + 1):
            purview_partitions = k_partitions(p, n_purview_parts)
            n_empty = n_mechanism_parts - n_purview_parts
            for purview_partition in purview_partitions:
                purview_partition = [tuple(_list)
                                     for _list in purview_partition]
                # Extend with empty tuples so purview partition has same size
                # as mechanism purview
                purview_partition.extend([() for j in range(n_empty)])
                # Unique permutations to avoid duplicates empties
                for permutation in set(itertools.permutations(purview_partition)):
                    yield KPartition(
                        *(Part(tuple(mechanism_partition[i]), tuple(permutation[i]))
                          for i in range(n_mechanism_parts)))


def effect_emd(d1, d2):
    """Compute the EMD between two effect repertoires.

    Billy's synopsis: Because the nodes are independent, the EMD between effect
    repertoires is equal to the sum of the EMDs between the marginal
    distributions of each node, and the EMD between marginal distribution for a
    node is the absolute difference in the probabilities that the node is off.

    Args:
        d1 (np.ndarray): The first repertoire.
        d2 (np.ndarray): The second repertoire.

    Returns:
        float: The EMD between ``d1`` and ``d2``.
    """
    return sum(abs(distribution.marginal_zero(d1, i) -
                   distribution.marginal_zero(d2, i))
               for i in range(d1.ndim))


def emd(direction, d1, d2):
    """Compute the EMD between two repertoires for a given direction.

    The full EMD computation is used for cause repertoires. A fast analytic
    solution is used for effect repertoires.

    Args:
        direction (Direction): :const:`~pyphi.constants.Direction.PAST` or
            :const:`~pyphi.constants.Direction.FUTURE`.
        d1 (np.ndarray): The first repertoire.
        d2 (np.ndarray): The second repertoire.

    Returns:
        float: The EMD between ``d1`` and ``d2``, rounded to |PRECISION|.

    Raises:
        ValueError: If ``direction`` is invalid.
    """
    if direction == Direction.PAST:
        func = distance.hamming_emd
    elif direction == Direction.FUTURE:
        func = effect_emd
    else:
        # TODO: test that ValueError is raised
        validate.direction(direction)

    return round(func(d1, d2), config.PRECISION)


def measure(direction, d1, d2):
    """Compute the distance between two repertoires for the given direction.

    Args:
        direction (Direction): :const:`~pyphi.constants.Direction.PAST` or
            :const:`~pyphi.constants.Direction.FUTURE`.
        d1 (np.ndarray): The first repertoire.
        d2 (np.ndarray): The second repertoire.

    Returns:
        float: The distance between ``d1`` and ``d2``, rounded to |PRECISION|.
    """

    measure_name = config.SMALL_PHI_MEASURE

    if measure_name == EMD:
        dist = emd(direction, d1, d2)

    elif measure_name == KLD:
        dist = kld(d1, d2)

    elif measure_name == L1:
        dist = l1(d1, d2)

    elif measure_name == ENTROPY_DIFFERENCE:
        dist = entropy_difference(d1, d2)

    else:
        validate.measure(measure_name, 'config.SMALL_PHI_MEASURE')

    # TODO do we actually need to round here?
    return round(dist, config.PRECISION)
