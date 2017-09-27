#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# subsystem.py

'''Represents a candidate system for |small_phi| and |big_phi| evaluation.'''

# pylint: disable=too-many-instance-attributes,too-many-public-methods,
# pylint: disable=too-many-public-methods,too-many-arguments

import functools
import itertools

import numpy as np

from . import cache, config, distance, distribution, utils, validate
from .constants import Direction
from .distance import small_phi_measure as measure
from .distribution import max_entropy_distribution, repertoire_shape
from .models import (Bipartition, Concept, NullCut, KPartition, Mice, Mip, Part,
                     Tripartition, _null_mip)
from .network import irreducible_purviews
from .node import generate_nodes
from .partition import (bipartition, directed_bipartition,
                        directed_tripartition, k_partitions, partitions)
from .tpm import condition_tpm, marginalize_out
from .utils import combs


class Subsystem:
    '''A set of nodes in a network.

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
        node_indices (tuple[int]): The indices of the nodes in the subsystem.
        cut (Cut): The cut that has been applied to this subsystem.
        cut_matrix (np.ndarray): A matrix of connections which have been
            severed by the cut.
        null_cut (Cut): The cut object representing no cut.
    '''

    def __init__(self, network, state, nodes, cut=None, mice_cache=None,
                 repertoire_cache=None, single_node_repertoire_cache=None):
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

        # The unidirectional cut applied for phi evaluation
        self.cut = cut if cut is not None else NullCut(self.node_indices)

        # The matrix of connections which are severed due to the cut
        # TODO: save/memoize on the cut so we just say self.cut.matrix()?
        self.cut_matrix = self.cut.cut_matrix(self.network.size)

        # The network's connectivity matrix with cut applied
        self.cm = self.cut.apply_cut(network.cm)

        # Reusable cache for core causes & effects
        self._mice_cache = cache.MiceCache(self, mice_cache)

        # Cause & effect repertoire caches
        # TODO: if repertoire caches are never reused, there's no reason to
        # have an accesible object-level cache. Just use a simple memoizer
        self._single_node_repertoire_cache = \
            single_node_repertoire_cache or cache.DictCache()
        self._repertoire_cache = repertoire_cache or cache.DictCache()

        self.nodes = generate_nodes(self.tpm, self.cm, self.state,
                                    network.indices2labels(self.node_indices))

        validate.subsystem(self)

    @property
    def nodes(self):
        '''tuple[Node]: The nodes in this |Subsystem|.'''
        return self._nodes

    # pylint: disable=attribute-defined-outside-init
    @nodes.setter
    def nodes(self, value):
        # Remap indices to nodes whenever nodes are changed, e.g. in the
        # `macro` module
        self._nodes = value
        self._index2node = {node.index: node for node in self._nodes}
    # pylint: enable=attribute-defined-outside-init

    @property
    def proper_state(self):
        '''tuple[int]: The state of the subsystem.

        ``proper_state[i]`` gives the state of the |ith| node **in the
        subsystem**. Note that this is **not** the state of ``nodes[i]``.
        '''
        return utils.state_of(self.node_indices, self.state)

    @property
    def connectivity_matrix(self):
        '''np.ndarray: Alias for ``Subsystem.cm``.'''
        return self.cm

    @property
    def size(self):
        '''int: The number of nodes in the subsystem.'''
        return len(self.node_indices)

    @property
    def is_cut(self):
        '''bool: ``True`` if this Subsystem has a cut applied to it.'''
        return not self.cut.is_null

    @property
    def cut_indices(self):
        '''tuple[int]: The nodes of this subsystem to cut for |big_phi|
        computations.

        This was added to support ``MacroSubsystem``, which cuts indices other
        than ``node_indices``.
        '''
        return self.node_indices

    @property
    def cut_mechanisms(self):
        '''list[tuple[int]]: The mechanisms that are cut in this system.'''
        return self.cut.all_cut_mechanisms()

    @property
    def tpm_size(self):
        '''int: The number of nodes in the TPM.'''
        return self.tpm.shape[-1]

    def cache_info(self):
        '''Report repertoire cache statistics.'''
        return {
            'single_node_repertoire': \
                self._single_node_repertoire_cache.info(),
            'repertoire': self._repertoire_cache.info(),
            'mice': self._mice_cache.info()
        }

    def clear_caches(self):
        '''Clear the mice and repertoire caches.'''
        self._single_node_repertoire_cache.clear()
        self._repertoire_cache.clear()
        self._mice_cache.clear()

    def __repr__(self):
        return "Subsystem(" + ', '.join(map(repr, self.nodes)) + ")"

    def __str__(self):
        return repr(self)

    def __bool__(self):
        '''Return ``False`` if the Subsystem has no nodes, ``True``
        otherwise.'''
        return bool(self.nodes)

    def __eq__(self, other):
        '''Return whether this Subsystem is equal to the other object.

        Two Subsystems are equal if their sets of nodes, networks, and cuts are
        equal.
        '''
        return (set(self.node_indices) == set(other.node_indices)
                and self.state == other.state
                and self.network == other.network
                and self.cut == other.cut)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        '''Return whether this subsystem has fewer nodes than the other.'''
        return len(self.nodes) < len(other.nodes)

    def __gt__(self, other):
        '''Return whether this subsystem has more nodes than the other.'''
        return len(self.nodes) > len(other.nodes)

    def __le__(self, other):
        return len(self.nodes) <= len(other.nodes)

    def __ge__(self, other):
        return len(self.nodes) >= len(other.nodes)

    def __len__(self):
        '''Return the number of nodes in this Subsystem.'''
        return len(self.node_indices)

    def __hash__(self):
        return hash((self.network, self.node_indices, self.state, self.cut))

    def to_json(self):
        '''Return a JSON-serializable representation.'''
        return {
            'network': self.network,
            'state': self.state,
            'nodes': self.node_indices,
            'cut': self.cut,
        }

    def apply_cut(self, cut):
        '''Return a cut version of this |Subsystem|.

        Args:
            cut (Cut): The cut to apply to this |Subsystem|.

        Returns:
            Subsystem: The cut subsystem.
        '''
        return Subsystem(self.network, self.state, self.node_indices,
                         cut=cut, mice_cache=self._mice_cache)

    def indices2nodes(self, indices):
        '''Return |Nodes| for these indices.

        Args:
            indices (tuple[int]): The indices in question.

        Returns:
            tuple[Node]: The |Node| objects corresponding to these indices.

        Raises:
            ValueError: If requested indices are not in the subsystem.
        '''
        if set(indices) - set(self.node_indices):
            raise ValueError(
                "`indices` must be a subset of the Subsystem's indices.")
        return tuple(self._index2node[n] for n in indices)

    def indices2labels(self, indices):
        '''Returns the node labels for these indices.'''
        return tuple(n.label for n in self.indices2nodes(indices))

    # TODO extend to nonbinary nodes
    @cache.method('_single_node_repertoire_cache', Direction.PAST)
    def _single_node_cause_repertoire(self, mechanism_node_index, purview):
        mechanism_node = self._index2node[mechanism_node_index]
        # We're conditioning on this node's state, so take the TPM for the node
        # being in that state.
        tpm = mechanism_node.tpm[..., mechanism_node.state]
        # Marginalize-out all parents of this mechanism node that aren't in the
        # purview.
        return marginalize_out((mechanism_node.inputs - purview), tpm)

    # TODO extend to nonbinary nodes
    @cache.method('_repertoire_cache', Direction.PAST)
    def cause_repertoire(self, mechanism, purview):
        '''Return the cause repertoire of a mechanism over a purview.

        Args:
            mechanism (tuple[int]): The mechanism for which to calculate the
                cause repertoire.
            purview (tuple[int]): The purview over which to calculate the
                cause repertoire.

        Returns:
            np.ndarray: The cause repertoire of the mechanism over the purview.

        .. note::
            The returned repertoire is a distribution over purview node states,
            not the states of the whole network.
        '''
        # If the purview is empty, the distribution is empty; return the
        # multiplicative identity.
        if not purview:
            return np.array([1.0])
        # If the mechanism is empty, nothing is specified about the past state
        # of the purview; return the purview's maximum entropy distribution.
        if not mechanism:
            return max_entropy_distribution(purview, self.tpm_size)
        # Use a frozenset so the arguments to `_single_node_cause_repertoire`
        # can be hashed and cached.
        purview = frozenset(purview)
        # Preallocate the repertoire with the proper shape, so that
        # probabilities are broadcasted appropriately.
        joint = np.ones(repertoire_shape(purview, self.tpm_size))
        # The cause repertoire is the product of the cause repertoires of the
        # individual nodes.
        joint *= functools.reduce(
            np.multiply, [self._single_node_cause_repertoire(m, purview)
                          for m in mechanism]
        )
        # The resulting joint distribution is over past states, which are rows
        # in the TPM, so the distribution is a column. In a state-by-node TPM
        # the columns don't sum to 1, so we normalize.
        return distribution.normalize(joint)

    # TODO extend to nonbinary nodes
    @cache.method('_single_node_repertoire_cache', Direction.FUTURE)
    def _single_node_effect_repertoire(self, mechanism, purview_node_index):
        purview_node = self._index2node[purview_node_index]
        # Condition on the state of the inputs that are in the mechanism.
        mechanism_inputs = (purview_node.inputs & mechanism)
        tpm = condition_tpm(purview_node.tpm, mechanism_inputs, self.state)
        # Marginalize-out the inputs that aren't in the mechanism.
        nonmechanism_inputs = (purview_node.inputs - mechanism)
        tpm = marginalize_out(nonmechanism_inputs, tpm)
        # Reshape so that the distribution is over future states.
        return tpm.reshape(repertoire_shape([purview_node.index],
                                            self.tpm_size))

    @cache.method('_repertoire_cache', Direction.FUTURE)
    def effect_repertoire(self, mechanism, purview):
        '''Return the effect repertoire of a mechanism over a purview.

        Args:
            mechanism (tuple[int]): The mechanism for which to calculate the
                effect repertoire.
            purview (tuple[int]): The purview over which to calculate the
                effect repertoire.

        Returns:
            np.ndarray: The effect repertoire of the mechanism over the
            purview.

        .. note::
            The returned repertoire is a distribution over purview node states,
            not the states of the whole network.
        '''
        # If the purview is empty, the distribution is empty, so return the
        # multiplicative identity.
        if not purview:
            return np.array([1.0])
        # Use a frozenset so the arguments to `_single_node_effect_repertoire`
        # can be hashed and cached.
        mechanism = frozenset(mechanism)
        # Preallocate the repertoire with the proper shape, so that
        # probabilities are broadcasted appropriately.
        joint = np.ones(repertoire_shape(purview, self.tpm_size))
        # The effect repertoire is the product of the effect repertoires of the
        # individual nodes.
        return joint * functools.reduce(
            np.multiply, [self._single_node_effect_repertoire(mechanism, p)
                          for p in purview]
        )

    def repertoire(self, direction, mechanism, purview):
        '''Return the cause or effect repertoire based on a direction.

        Args:
            direction (Direction): |PAST| or |FUTURE|.
            mechanism (tuple[int]): The mechanism for which to calculate the
                repertoire.
            purview (tuple[int]): The purview over which to calculate the
                repertoire.

        Returns:
            np.ndarray: The cause or effect repertoire of the mechanism over
            the purview.

        Raises:
            ValueError: If ``direction`` is invalid.
        '''
        if direction == Direction.PAST:
            return self.cause_repertoire(mechanism, purview)
        elif direction == Direction.FUTURE:
            return self.effect_repertoire(mechanism, purview)
        else:
            # TODO: test that ValueError is raised
            validate.direction(direction)

    def unconstrained_repertoire(self, direction, purview):
        '''Return the unconstrained cause/effect repertoire over a purview.'''
        return self.repertoire(direction, (), purview)

    def unconstrained_cause_repertoire(self, purview):
        '''Return the unconstrained cause repertoire for a purview.

        This is just the cause repertoire in the absence of any mechanism.
        '''
        return self.unconstrained_repertoire(Direction.PAST, purview)

    def unconstrained_effect_repertoire(self, purview):
        '''Return the unconstrained effect repertoire for a purview.

        This is just the effect repertoire in the absence of any mechanism.
        '''
        return self.unconstrained_repertoire(Direction.FUTURE, purview)

    def partitioned_repertoire(self, direction, partition):
        '''Compute the repertoire of a partitioned mechanism and purview.'''
        repertoires = [
            self.repertoire(direction, part.mechanism, part.purview)
            for part in partition
        ]
        return functools.reduce(np.multiply, repertoires)

    def expand_repertoire(self, direction, repertoire, new_purview=None):
        '''Distribute an effect repertoire over a larger purview.

        Args:
            direction (Direction): |PAST| or |FUTURE|.
            repertoire (np.ndarray): The repertoire to expand.

        Keyword Args:
            new_purview (tuple[int]): The new purview to expand the repertoire
                over. If ``None`` (the default), the new purview is the entire
                network.

        Returns:
            np.ndarray: A distribution over the new purview, where probability
            is spread out over the new nodes.

        Raises:
            ValueError: If the expanded purview doesn't contain the original
                purview.
        '''
        if repertoire is None:
            return None

        purview = distribution.purview(repertoire)

        if new_purview is None:
            new_purview = self.node_indices  # full subsystem

        if not set(purview).issubset(new_purview):
            raise ValueError("Expanded purview must contain original purview.")

        # Get the unconstrained repertoire over the other nodes in the network.
        non_purview_indices = tuple(set(new_purview) - set(purview))
        uc = self.unconstrained_repertoire(direction, non_purview_indices)
        # Multiply the given repertoire by the unconstrained one to get a
        # distribution over all the nodes in the network.
        expanded_repertoire = repertoire * uc

        return distribution.normalize(expanded_repertoire)

    def expand_cause_repertoire(self, repertoire, new_purview=None):
        '''Same as |expand_repertoire| with ``direction`` set to |PAST|.'''
        return self.expand_repertoire(Direction.PAST, repertoire,
                                      new_purview)

    def expand_effect_repertoire(self, repertoire, new_purview=None):
        '''Same as |expand_repertoire| with ``direction`` set to |FUTURE|.'''
        return self.expand_repertoire(Direction.FUTURE, repertoire,
                                      new_purview)

    def cause_info(self, mechanism, purview):
        '''Return the cause information for a mechanism over a purview.'''
        return measure(Direction.PAST,
                       self.cause_repertoire(mechanism, purview),
                       self.unconstrained_cause_repertoire(purview))

    def effect_info(self, mechanism, purview):
        '''Return the effect information for a mechanism over a purview.'''
        return measure(Direction.FUTURE,
                       self.effect_repertoire(mechanism, purview),
                       self.unconstrained_effect_repertoire(purview))

    def cause_effect_info(self, mechanism, purview):
        '''Return the cause-effect information for a mechanism over a purview.

        This is the minimum of the cause and effect information.
        '''
        return min(self.cause_info(mechanism, purview),
                   self.effect_info(mechanism, purview))

    # MIP methods
    # =========================================================================

    def evaluate_partition(self, direction, mechanism, purview, partition,
                           unpartitioned_repertoire=None):
        '''Return the |small_phi| of a mechanism over a purview for the given
        partition.

        Args:
            direction (Direction): |PAST| or |FUTURE|.
            mechanism (tuple[int]): The nodes in the mechanism.
            purview (tuple[int]): The nodes in the purview.
            partition (Bipartition): The partition to evaluate.

        Keyword Args:
            unpartitioned_repertoire (np.array): The unpartitioned repertoire.
                If not supplied, it will be computed.

        Returns:
            tuple[int, np.ndarray]: The distance between the unpartitioned and
            partitioned repertoires, and the partitioned repertoire.
        '''
        if unpartitioned_repertoire is None:
            unpartitioned_repertoire = self.repertoire(direction, mechanism,
                                                       purview)

        partitioned_repertoire = self.partitioned_repertoire(direction,
                                                             partition)

        phi = measure(direction, unpartitioned_repertoire,
                      partitioned_repertoire)

        return (phi, partitioned_repertoire)

    def find_mip(self, direction, mechanism, purview):
        '''Return the minimum information partition for a mechanism over a
        purview.

        Args:
            direction (Direction): |PAST| or |FUTURE|.
            mechanism (tuple[int]): The nodes in the mechanism.
            purview (tuple[int]): The nodes in the purview.

        Returns:
            Mip: The mininum-information partition in one temporal direction.
        '''
        # We default to the null MIP (the MIP of a reducible mechanism)
        mip = _null_mip(direction, mechanism, purview)

        if not purview:
            return mip

        phi_min = float('inf')
        # Calculate the unpartitioned repertoire to compare against the
        # partitioned ones.
        unpartitioned_repertoire = self.repertoire(direction, mechanism,
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
        for partition in mip_partitions(mechanism, purview, direction):
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
        '''Return the past minimum information partition.

        Alias for |find_mip| with ``direction`` set to |PAST|.
        '''
        return self.find_mip(Direction.PAST, mechanism, purview)

    def mip_future(self, mechanism, purview):
        '''Return the future minimum information partition.

        Alias for |find_mip| with ``direction`` set to |FUTURE|.
        '''
        return self.find_mip(Direction.FUTURE, mechanism, purview)

    def phi_mip_past(self, mechanism, purview):
        '''Return the |small_phi| of the past minimum information partition.

        This is the distance between the unpartitioned cause repertoire and the
        MIP cause repertoire.
        '''
        mip = self.mip_past(mechanism, purview)
        return mip.phi if mip else 0

    def phi_mip_future(self, mechanism, purview):
        '''Return the |small_phi| of the future minimum information partition.

        This is the distance between the unpartitioned effect repertoire and
        the MIP cause repertoire.
        '''
        mip = self.mip_future(mechanism, purview)
        return mip.phi if mip else 0

    def phi(self, mechanism, purview):
        '''Return the |small_phi| of a mechanism over a purview.'''
        return min(self.phi_mip_past(mechanism, purview),
                   self.phi_mip_future(mechanism, purview))

    # Phi_max methods
    # =========================================================================

    def potential_purviews(self, direction, mechanism, purviews=False):
        '''Return all purviews that could belong to the core cause/effect.

        Filters out trivially-reducible purviews.

        Args:
            direction (Direction): |PAST| or |FUTURE|.
            mechanism (tuple[int]): The mechanism of interest.

        Keyword Args:
            purviews (tuple[int]): Optional subset of purviews of interest.
        '''
        if purviews is False:
            purviews = self.network.potential_purviews(direction, mechanism)
            # Filter out purviews that aren't in the subsystem
            purviews = [purview for purview in purviews
                        if set(purview).issubset(self.node_indices)]

        # Purviews are already filtered in network.potential_purviews
        # over the full network connectivity matrix. However, since the cm
        # is cut/smaller we check again here.
        return irreducible_purviews(self.cm, direction, mechanism, purviews)

    @cache.method('_mice_cache')
    def find_mice(self, direction, mechanism, purviews=False):
        '''Return the maximally irreducible cause or effect for a mechanism.

        Args:
            direction (Direction): :|PAST| or |FUTURE|.
            mechanism (tuple[int]): The mechanism to be tested for
                irreducibility.

        Keyword Args:
            purviews (tuple[int]): Optionally restrict the possible purviews
                to a subset of the subsystem. This may be useful for _e.g._
                finding only concepts that are "about" a certain subset of
                nodes.

        Returns:
            Mice: The maximally-irreducible cause or effect in one temporal
            direction.

        .. note::
            Strictly speaking, the MICE is a pair of repertoires: the core
            cause repertoire and core effect repertoire of a mechanism, which
            are maximally different than the unconstrained cause/effect
            repertoires (*i.e.*, those that maximize |small_phi|). Here, we
            return only information corresponding to one direction, |PAST| or
            |FUTURE|, i.e., we return a core cause or core effect, not the pair
            of them.
        '''
        purviews = self.potential_purviews(direction, mechanism, purviews)

        if not purviews:
            max_mip = _null_mip(direction, mechanism, ())
        else:
            max_mip = max(self.find_mip(direction, mechanism, purview)
                          for purview in purviews)

        return Mice(max_mip)

    def core_cause(self, mechanism, purviews=False):
        '''Return the core cause repertoire of a mechanism.

        Alias for |find_mice| with ``direction`` set to |PAST|.
        '''
        return self.find_mice(Direction.PAST, mechanism, purviews=purviews)

    def core_effect(self, mechanism, purviews=False):
        '''Return the core effect repertoire of a mechanism.

        Alias for |find_mice| with ``direction`` set to |PAST|.
        '''
        return self.find_mice(Direction.FUTURE, mechanism, purviews=purviews)

    def phi_max(self, mechanism):
        '''Return the |small_phi_max| of a mechanism.

        This is the maximum of |small_phi| taken over all possible purviews.
        '''
        return min(self.core_cause(mechanism).phi,
                   self.core_effect(mechanism).phi)

    # Big Phi methods
    # =========================================================================

    # TODO add `concept-space` section to the docs:
    @property
    def null_concept(self):
        '''Return the null concept of this subsystem.

        The null concept is a point in concept space identified with
        the unconstrained cause and effect repertoire of this subsystem.
        '''
        # Unconstrained cause repertoire.
        cause_repertoire = self.cause_repertoire((), ())
        # Unconstrained effect repertoire.
        effect_repertoire = self.effect_repertoire((), ())

        # Null cause.
        cause = Mice(_null_mip(Direction.PAST, (), (), cause_repertoire))
        # Null effect.
        effect = Mice(_null_mip(Direction.FUTURE, (), (), effect_repertoire))

        # All together now...
        return Concept(mechanism=(), cause=cause, effect=effect, subsystem=self)

    def concept(self, mechanism, purviews=False, past_purviews=False,
                future_purviews=False):
        '''Calculate a concept.

        See :func:`pyphi.compute.concept` for more information.
        '''
        # Calculate the maximally irreducible cause repertoire.
        cause = self.core_cause(mechanism,
                                purviews=(past_purviews or purviews))
        # Calculate the maximally irreducible effect repertoire.
        effect = self.core_effect(mechanism,
                                  purviews=(future_purviews or purviews))
        # NOTE: Make sure to expand the repertoires to the size of the
        # subsystem when calculating concept distance. For now, they must
        # remain un-expanded so the concept doesn't depend on the subsystem.
        return Concept(mechanism=mechanism, cause=cause,
                       effect=effect, subsystem=self)


def mip_partitions(mechanism, purview, direction=None):
    '''Return a generator over all MIP partitions, based on the current
    configuration.'''
    validate.partition_type(config.PARTITION_TYPE)

    if config.PARTITION_TYPE == 'PD':
        return purview_disconnection_partitions(mechanism, purview, direction)
    else:
        func = {
            'BI': mip_bipartitions,
            'TRI': wedge_partitions,
            'ALL': all_partitions,
            'PD': purview_disconnection_partitions
        }[config.PARTITION_TYPE]
        return func(mechanism, purview)


def mip_bipartitions(mechanism, purview):
    '''Return an generator of all |small_phi| bipartitions of a mechanism over
    a purview.

    Excludes all bipartitions where one half is entirely empty, *e.g*::

         A     ∅
        ─── ✕ ───
         B     ∅

    is not valid, but ::

         A     ∅
        ─── ✕ ───
         ∅     B

    is.

    Args:
        mechanism (tuple[int]): The mechanism to partition
        purview (tuple[int]): The purview to partition

    Yields:
        Bipartition: Where each bipartition is::

            bipart[0].mechanism   bipart[1].mechanism
            ─────────────────── ✕ ───────────────────
            bipart[0].purview     bipart[1].purview

    Example:
        >>> mechanism = (0,)
        >>> purview = (2, 3)
        >>> for partition in mip_bipartitions(mechanism, purview):
        ...     print(partition, '\\n')  # doctest: +NORMALIZE_WHITESPACE
         ∅     0
        ─── ✕ ───
         2     3
        <BLANKLINE>
         ∅     0
        ─── ✕ ───
         3     2
        <BLANKLINE>
         ∅     0
        ─── ✕ ───
        2,3    ∅
    '''
    numerators = bipartition(mechanism)
    denominators = directed_bipartition(purview)

    for n, d in itertools.product(numerators, denominators):
        if (n[0] or d[0]) and (n[1] or d[1]):
            yield Bipartition(Part(n[0], d[0]), Part(n[1], d[1]))


def wedge_partitions(mechanism, purview):
    '''Return an iterator over all wedge partitions.

    These are partitions which strictly split the mechanism and allow a subset
    of the purview to be split into a third partition, e.g.::

         A     B     ∅
        ─── ✕ ─── ✕ ───
         B     C     D

    See |PARTITION_TYPE| in |config| for more information.

    Args:
        mechanism (tuple[int]): A mechanism.
        purview (tuple[int]): A purview.

    Yields:
        Tripartition: all unique tripartitions of this mechanism and purview.
    '''
    numerators = bipartition(mechanism)
    denominators = directed_tripartition(purview)

    yielded = set()

    # pylint: disable=too-many-boolean-expressions
    def valid(factoring):
        '''Return whether the factoring should be considered.'''
        numerator, denominator = factoring
        return (
            (numerator[0] or denominator[0]) and
            (numerator[1] or denominator[1]) and
            ((numerator[0] and numerator[1]) or
             not denominator[0] or
             not denominator[1])
        )
    # pylint: enable=too-many-boolean-expressions

    for n, d in filter(valid, itertools.product(numerators, denominators)):
        # Normalize order of parts to remove duplicates.
        tripart = Tripartition(
            Part(n[0], d[0]),
            Part(n[1], d[1]),
            Part((),   d[2])).normalize()  # pylint: disable=bad-whitespace

        def nonempty(part):
            '''Check that the part is not empty.'''
            return part.mechanism or part.purview

        def compressible(tripart):
            '''Check if the tripartition can be transformed into a causally
            equivalent partition by combing two of its parts; eg. A/∅ x B/∅ x
            ∅/CD is equivalent to AB/∅ x ∅/CD so we don't include it. '''
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


def all_partitions(mechanism, purview):
    '''Returns all possible partitions of a mechanism and purview.

    Partitions can consist of any number of parts.

    Args:
        mechanism (tuple[int]): A mechanism.
        purview (tuple[int]): A purview.

    Yields:
        KPartition: A partition of this mechanism and purview into ``k`` parts.
    '''
    for mechanism_partition in partitions(mechanism):
        mechanism_partition.append([])
        n_mechanism_parts = len(mechanism_partition)
        max_purview_partition = min(len(purview), n_mechanism_parts)
        for n_purview_parts in range(1, max_purview_partition + 1):
            n_empty = n_mechanism_parts - n_purview_parts
            for purview_partition in k_partitions(purview, n_purview_parts):
                purview_partition = [tuple(_list)
                                     for _list in purview_partition]
                # Extend with empty tuples so purview partition has same size
                # as mechanism purview
                purview_partition.extend([()] * n_empty)

                # Unique permutations to avoid duplicates empties
                for purview_permutation in set(itertools.permutations(purview_partition)):

                    parts = [
                        Part(tuple(m), tuple(p))
                        for m, p in zip(mechanism_partition, purview_permutation)]

                    # Must partition the mechanism, unless the purview is fully
                    # cut away from the mechanism.
                    if parts[0].mechanism == mechanism and parts[0].purview:
                        continue

                    yield KPartition(*parts)


def purview_disconnection_partitions(mechanism, purview, direction):
    '''A partition is a purview disconnection partition (PDP) if every element
    of the purview has been cut from at least one mechanism element.

    Optimization: The MIP always belongs to a restricted class of PDPs where
    every purview element is cut from EXACTLY one mechanism element. Therefore,
    we only return these restricted PDPs.

    Implementation note: Although a given PDP does not change from |PAST| to
    |FUTURE| (the mechanism element that each purview element is cut from does
    not change), the repertoires that must be multiplied together in order to
    achieve this effect do. This is a consequence of the fact that PDPs can cut
    individual connections. In the |FUTURE|, the purview can always be
    parititoned into disjoint subsets, but the mechanism cannot. In the |PAST|,
    the opposite is true.

    Args:
        mechanism (tuple[int]): A mechanism.
        purview (tuple[int]): A purview.
        direction (Direction): |PAST| or |FUTURE|.

    Yields:
        KPartition: A partition of this mechanism and purview into ``K`` parts.
        If ``direction`` is |PAST|, ``K`` is the number of mechanism elements.
        If ``direction`` is |FUTURE|, ``K`` is the number of purview elements.
    '''
    # Get all subsets of the mechanism with exactly one element missing. These
    # represent the possible "remainders": mechanism elements left uncut from
    # each purview element. First order mechanisms have no valid remainders.
    remainders = [()] if len(mechanism) == 1 else \
            [tuple(x) for x in combs(mechanism, len(mechanism) - 1)]

    # Get all ways assign the remainders to the purview elements.
    all_assignments = itertools.product(remainders, repeat=len(purview))

    # Make a partition for each possible assignment.
    for assignment in all_assignments:
        parts = [Part(m, (p,)) for m, p in zip(assignment, purview)]
        # Create a part with a null denominator, if necessary.
        leftover_mechanism_elements = set(mechanism) - set(
                itertools.chain.from_iterable(part.mechanism for part in parts))
        if leftover_mechanism_elements:
            parts.append(Part(tuple(leftover_mechanism_elements), ()))

        # How the partition should be factorized depends on direction.
        # Partitions are factorized by purview (suitable for |FUTURE|) already.
        if direction == Direction.PAST:
            yield KPartition(*parts).refactor_by_mechanism()
        elif direction == Direction.FUTURE:
            yield KPartition(*parts)
        else:
            validate.direction(direction)
