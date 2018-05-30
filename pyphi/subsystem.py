#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# subsystem.py

"""Represents a candidate system for |small_phi| and |big_phi| evaluation."""

import functools
import logging

import numpy as np

from . import Direction, cache, distribution, utils, validate
from .distance import repertoire_distance
from .distribution import max_entropy_distribution, repertoire_shape
from .models import (Concept, MaximallyIrreducibleCause,
                     MaximallyIrreducibleEffect, NullCut,
                     RepertoireIrreducibilityAnalysis, _null_ria)
from .network import irreducible_purviews
from .node import generate_nodes
from .partition import mip_partitions
from .tpm import condition_tpm, marginalize_out
from .utils import time_annotated

log = logging.getLogger(__name__)


class Subsystem:
    """A set of nodes in a network.

    Args:
        network (Network): The network the subsystem belongs to.
        state (tuple[int]): The state of the network.

    Keyword Args:
        nodes (tuple[int] or tuple[str]): The nodes of the network which are in
            this subsystem. Nodes can be specified either as indices or as
            labels if the |Network| was passed ``node_labels``. If this is
            ``None`` then the full network will be used.
        cut (Cut): The unidirectional |Cut| to apply to this subsystem.

    Attributes:
        network (Network): The network the subsystem belongs to.
        tpm (np.ndarray): The TPM conditioned on the state of the external
            nodes.
        cm (np.ndarray): The connectivity matrix after applying the cut.
        state (tuple[int]): The state of the network.
        node_indices (tuple[int]): The indices of the nodes in the subsystem.
        cut (Cut): The cut that has been applied to this subsystem.
        null_cut (Cut): The cut object representing no cut.
    """

    def __init__(self, network, state, nodes=None, cut=None, mice_cache=None,
                 repertoire_cache=None, single_node_repertoire_cache=None,
                 _external_indices=None):
        # The network this subsystem belongs to.
        validate.is_network(network)
        self.network = network

        self.node_labels = network.node_labels
        # Remove duplicates, sort, and ensure native Python `int`s
        # (for JSON serialization).
        self.node_indices = self.node_labels.coerce_to_indices(nodes)

        validate.state_length(state, self.network.size)

        # The state of the network.
        self.state = tuple(state)

        # Get the external node indices.
        # TODO: don't expose this as an attribute?
        if _external_indices is None:
            self.external_indices = tuple(
                set(network.node_indices) - set(self.node_indices))
        else:
            self.external_indices = _external_indices

        # The TPM conditioned on the state of the external nodes.
        self.tpm = condition_tpm(
            self.network.tpm, self.external_indices, self.state)

        # The unidirectional cut applied for phi evaluation
        self.cut = (cut if cut is not None
                    else NullCut(self.node_indices, self.node_labels))

        # The network's connectivity matrix with cut applied
        self.cm = self.cut.apply_cut(network.cm)

        # Reusable cache for maximally-irreducible causes and effects
        self._mice_cache = cache.MICECache(self, mice_cache)

        # Cause & effect repertoire caches
        # TODO: if repertoire caches are never reused, there's no reason to
        # have an accesible object-level cache. Just use a simple memoizer
        self._single_node_repertoire_cache = \
            single_node_repertoire_cache or cache.DictCache()
        self._repertoire_cache = repertoire_cache or cache.DictCache()

        self.nodes = generate_nodes(
            self.tpm, self.cm, self.state, self.node_indices, self.node_labels)

        validate.subsystem(self)

    @property
    def nodes(self):
        """tuple[Node]: The nodes in this |Subsystem|."""
        return self._nodes

    @nodes.setter
    def nodes(self, value):
        """Remap indices to nodes whenever nodes are changed, e.g. in the
        `macro` module.
        """
        # pylint: disable=attribute-defined-outside-init
        self._nodes = value
        self._index2node = {node.index: node for node in self._nodes}

    @property
    def proper_state(self):
        """tuple[int]: The state of the subsystem.

        ``proper_state[i]`` gives the state of the |ith| node **in the
        subsystem**. Note that this is **not** the state of ``nodes[i]``.
        """
        return utils.state_of(self.node_indices, self.state)

    @property
    def connectivity_matrix(self):
        """np.ndarray: Alias for |Subsystem.cm|."""
        return self.cm

    @property
    def size(self):
        """int: The number of nodes in the subsystem."""
        return len(self.node_indices)

    @property
    def is_cut(self):
        """bool: ``True`` if this Subsystem has a cut applied to it."""
        return not self.cut.is_null

    @property
    def cut_indices(self):
        """tuple[int]: The nodes of this subsystem to cut for |big_phi|
        computations.

        This was added to support ``MacroSubsystem``, which cuts indices other
        than ``node_indices``.

        Yields:
            tuple[int]
        """
        return self.node_indices

    @property
    def cut_mechanisms(self):
        """list[tuple[int]]: The mechanisms that are cut in this system."""
        return self.cut.all_cut_mechanisms()

    @property
    def cut_node_labels(self):
        """``NodeLabels``: Labels for the nodes of this system that will be
        cut.
        """
        return self.node_labels

    @property
    def tpm_size(self):
        """int: The number of nodes in the TPM."""
        return self.tpm.shape[-1]

    def cache_info(self):
        """Report repertoire cache statistics."""
        return {
            'single_node_repertoire':
                self._single_node_repertoire_cache.info(),
            'repertoire': self._repertoire_cache.info(),
            'mice': self._mice_cache.info()
        }

    def clear_caches(self):
        """Clear the mice and repertoire caches."""
        self._single_node_repertoire_cache.clear()
        self._repertoire_cache.clear()
        self._mice_cache.clear()

    def __repr__(self):
        return "Subsystem(" + ', '.join(map(repr, self.nodes)) + ")"

    def __str__(self):
        return repr(self)

    def __bool__(self):
        """Return ``False`` if the Subsystem has no nodes, ``True``
        otherwise.
        """
        return bool(self.nodes)

    def __eq__(self, other):
        """Return whether this Subsystem is equal to the other object.

        Two Subsystems are equal if their sets of nodes, networks, and cuts are
        equal.
        """
        if not isinstance(other, Subsystem):
            return False

        return (
            set(self.node_indices) == set(other.node_indices) and
            self.state == other.state and
            self.network == other.network and
            self.cut == other.cut
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        """Return whether this subsystem has fewer nodes than the other."""
        return len(self.nodes) < len(other.nodes)

    def __gt__(self, other):
        """Return whether this subsystem has more nodes than the other."""
        return len(self.nodes) > len(other.nodes)

    def __le__(self, other):
        return len(self.nodes) <= len(other.nodes)

    def __ge__(self, other):
        return len(self.nodes) >= len(other.nodes)

    def __len__(self):
        """Return the number of nodes in this Subsystem."""
        return len(self.node_indices)

    def __hash__(self):
        return hash((self.network, self.node_indices, self.state, self.cut))

    def to_json(self):
        """Return a JSON-serializable representation."""
        return {
            'network': self.network,
            'state': self.state,
            'nodes': self.node_indices,
            'cut': self.cut,
        }

    def apply_cut(self, cut):
        """Return a cut version of this |Subsystem|.

        Args:
            cut (Cut): The cut to apply to this |Subsystem|.

        Returns:
            Subsystem: The cut subsystem.
        """
        return Subsystem(self.network, self.state, self.node_indices,
                         cut=cut, mice_cache=self._mice_cache)

    def indices2nodes(self, indices):
        """Return |Nodes| for these indices.

        Args:
            indices (tuple[int]): The indices in question.

        Returns:
            tuple[Node]: The |Node| objects corresponding to these indices.

        Raises:
            ValueError: If requested indices are not in the subsystem.
        """
        if set(indices) - set(self.node_indices):
            raise ValueError(
                "`indices` must be a subset of the Subsystem's indices.")
        return tuple(self._index2node[n] for n in indices)

    # TODO extend to nonbinary nodes
    @cache.method('_single_node_repertoire_cache', Direction.CAUSE)
    def _single_node_cause_repertoire(self, mechanism_node_index, purview):
        # pylint: disable=missing-docstring
        mechanism_node = self._index2node[mechanism_node_index]
        # We're conditioning on this node's state, so take the TPM for the node
        # being in that state.
        tpm = mechanism_node.tpm[..., mechanism_node.state]
        # Marginalize-out all parents of this mechanism node that aren't in the
        # purview.
        return marginalize_out((mechanism_node.inputs - purview), tpm)

    # TODO extend to nonbinary nodes
    @cache.method('_repertoire_cache', Direction.CAUSE)
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
            The returned repertoire is a distribution over purview node states,
            not the states of the whole network.
        """
        # If the purview is empty, the distribution is empty; return the
        # multiplicative identity.
        if not purview:
            return np.array([1.0])
        # If the mechanism is empty, nothing is specified about the previous
        # state of the purview; return the purview's maximum entropy
        # distribution.
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
        # The resulting joint distribution is over previous states, which are
        # rows in the TPM, so the distribution is a column. The columns of a
        # TPM don't necessarily sum to 1, so we normalize.
        return distribution.normalize(joint)

    # TODO extend to nonbinary nodes
    @cache.method('_single_node_repertoire_cache', Direction.EFFECT)
    def _single_node_effect_repertoire(self, mechanism, purview_node_index):
        # pylint: disable=missing-docstring
        purview_node = self._index2node[purview_node_index]
        # Condition on the state of the inputs that are in the mechanism.
        mechanism_inputs = (purview_node.inputs & mechanism)
        tpm = condition_tpm(purview_node.tpm, mechanism_inputs, self.state)
        # Marginalize-out the inputs that aren't in the mechanism.
        nonmechanism_inputs = (purview_node.inputs - mechanism)
        tpm = marginalize_out(nonmechanism_inputs, tpm)
        # Reshape so that the distribution is over next states.
        return tpm.reshape(repertoire_shape([purview_node.index],
                                            self.tpm_size))

    @cache.method('_repertoire_cache', Direction.EFFECT)
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
            The returned repertoire is a distribution over purview node states,
            not the states of the whole network.
        """
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
        """Return the cause or effect repertoire based on a direction.

        Args:
            direction (Direction): |CAUSE| or |EFFECT|.
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
        if direction == Direction.CAUSE:
            return self.cause_repertoire(mechanism, purview)
        elif direction == Direction.EFFECT:
            return self.effect_repertoire(mechanism, purview)

        return validate.direction(direction)

    def unconstrained_repertoire(self, direction, purview):
        """Return the unconstrained cause/effect repertoire over a purview."""
        return self.repertoire(direction, (), purview)

    def unconstrained_cause_repertoire(self, purview):
        """Return the unconstrained cause repertoire for a purview.

        This is just the cause repertoire in the absence of any mechanism.
        """
        return self.unconstrained_repertoire(Direction.CAUSE, purview)

    def unconstrained_effect_repertoire(self, purview):
        """Return the unconstrained effect repertoire for a purview.

        This is just the effect repertoire in the absence of any mechanism.
        """
        return self.unconstrained_repertoire(Direction.EFFECT, purview)

    def partitioned_repertoire(self, direction, partition):
        """Compute the repertoire of a partitioned mechanism and purview."""
        repertoires = [
            self.repertoire(direction, part.mechanism, part.purview)
            for part in partition
        ]
        return functools.reduce(np.multiply, repertoires)

    def expand_repertoire(self, direction, repertoire, new_purview=None):
        """Distribute an effect repertoire over a larger purview.

        Args:
            direction (Direction): |CAUSE| or |EFFECT|.
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
        uc = self.unconstrained_repertoire(direction, non_purview_indices)
        # Multiply the given repertoire by the unconstrained one to get a
        # distribution over all the nodes in the network.
        expanded_repertoire = repertoire * uc

        return distribution.normalize(expanded_repertoire)

    def expand_cause_repertoire(self, repertoire, new_purview=None):
        """Alias for |expand_repertoire()| with ``direction`` set to |CAUSE|.
        """
        return self.expand_repertoire(Direction.CAUSE, repertoire,
                                      new_purview)

    def expand_effect_repertoire(self, repertoire, new_purview=None):
        """Alias for |expand_repertoire()| with ``direction`` set to |EFFECT|.
        """
        return self.expand_repertoire(Direction.EFFECT, repertoire,
                                      new_purview)

    def cause_info(self, mechanism, purview):
        """Return the cause information for a mechanism over a purview."""
        return repertoire_distance(
            Direction.CAUSE,
            self.cause_repertoire(mechanism, purview),
            self.unconstrained_cause_repertoire(purview)
        )

    def effect_info(self, mechanism, purview):
        """Return the effect information for a mechanism over a purview."""
        return repertoire_distance(
            Direction.EFFECT,
            self.effect_repertoire(mechanism, purview),
            self.unconstrained_effect_repertoire(purview)
        )

    def cause_effect_info(self, mechanism, purview):
        """Return the cause-effect information for a mechanism over a purview.

        This is the minimum of the cause and effect information.
        """
        return min(self.cause_info(mechanism, purview),
                   self.effect_info(mechanism, purview))

    # MIP methods
    # =========================================================================

    def evaluate_partition(self, direction, mechanism, purview, partition,
                           repertoire=None):
        """Return the |small_phi| of a mechanism over a purview for the given
        partition.

        Args:
            direction (Direction): |CAUSE| or |EFFECT|.
            mechanism (tuple[int]): The nodes in the mechanism.
            purview (tuple[int]): The nodes in the purview.
            partition (Bipartition): The partition to evaluate.

        Keyword Args:
            repertoire (np.array): The unpartitioned repertoire.
                If not supplied, it will be computed.

        Returns:
            tuple[int, np.ndarray]: The distance between the unpartitioned and
            partitioned repertoires, and the partitioned repertoire.
        """
        if repertoire is None:
            repertoire = self.repertoire(direction, mechanism, purview)

        partitioned_repertoire = self.partitioned_repertoire(direction,
                                                             partition)

        phi = repertoire_distance(
            direction, repertoire, partitioned_repertoire)

        return (phi, partitioned_repertoire)

    def find_mip(self, direction, mechanism, purview):
        """Return the minimum information partition for a mechanism over a
        purview.

        Args:
            direction (Direction): |CAUSE| or |EFFECT|.
            mechanism (tuple[int]): The nodes in the mechanism.
            purview (tuple[int]): The nodes in the purview.

        Returns:
            RepertoireIrreducibilityAnalysis: The irreducibility analysis for
            the mininum-information partition in one temporal direction.
        """
        if not purview:
            return _null_ria(direction, mechanism, purview)

        # Calculate the unpartitioned repertoire to compare against the
        # partitioned ones.
        repertoire = self.repertoire(direction, mechanism, purview)

        def _mip(phi, partition, partitioned_repertoire):
            # Prototype of MIP with already known data
            # TODO: Use properties here to infer mechanism and purview from
            # partition yet access them with `.mechanism` and `.purview`.
            return RepertoireIrreducibilityAnalysis(
                phi=phi,
                direction=direction,
                mechanism=mechanism,
                purview=purview,
                partition=partition,
                repertoire=repertoire,
                partitioned_repertoire=partitioned_repertoire,
                node_labels=self.node_labels
            )

        # State is unreachable - return 0 instead of giving nonsense results
        if (direction == Direction.CAUSE and
                np.all(repertoire == 0)):
            return _mip(0, None, None)

        mip = _null_ria(direction, mechanism, purview, phi=float('inf'))

        for partition in mip_partitions(mechanism, purview, self.node_labels):
            # Find the distance between the unpartitioned and partitioned
            # repertoire.
            phi, partitioned_repertoire = self.evaluate_partition(
                direction, mechanism, purview, partition,
                repertoire=repertoire)

            # Return immediately if mechanism is reducible.
            if phi == 0:
                return _mip(0.0, partition, partitioned_repertoire)

            # Update MIP if it's more minimal.
            if phi < mip.phi:
                mip = _mip(phi, partition, partitioned_repertoire)

        return mip

    def cause_mip(self, mechanism, purview):
        """Return the irreducibility analysis for the cause MIP.

        Alias for |find_mip()| with ``direction`` set to |CAUSE|.
        """
        return self.find_mip(Direction.CAUSE, mechanism, purview)

    def effect_mip(self, mechanism, purview):
        """Return the irreducibility analysis for the effect MIP.

        Alias for |find_mip()| with ``direction`` set to |EFFECT|.
        """
        return self.find_mip(Direction.EFFECT, mechanism, purview)

    def phi_cause_mip(self, mechanism, purview):
        """Return the |small_phi| of the cause MIP.

        This is the distance between the unpartitioned cause repertoire and the
        MIP cause repertoire.
        """
        mip = self.cause_mip(mechanism, purview)
        return mip.phi if mip else 0

    def phi_effect_mip(self, mechanism, purview):
        """Return the |small_phi| of the effect MIP.

        This is the distance between the unpartitioned effect repertoire and
        the MIP cause repertoire.
        """
        mip = self.effect_mip(mechanism, purview)
        return mip.phi if mip else 0

    def phi(self, mechanism, purview):
        """Return the |small_phi| of a mechanism over a purview."""
        return min(self.phi_cause_mip(mechanism, purview),
                   self.phi_effect_mip(mechanism, purview))

    # Phi_max methods
    # =========================================================================

    def potential_purviews(self, direction, mechanism, purviews=False):
        """Return all purviews that could belong to the |MIC|/|MIE|.

        Filters out trivially-reducible purviews.

        Args:
            direction (Direction): |CAUSE| or |EFFECT|.
            mechanism (tuple[int]): The mechanism of interest.

        Keyword Args:
            purviews (tuple[int]): Optional subset of purviews of interest.
        """
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
        """Return the |MIC| or |MIE| for a mechanism.

        Args:
            direction (Direction): :|CAUSE| or |EFFECT|.
            mechanism (tuple[int]): The mechanism to be tested for
                irreducibility.

        Keyword Args:
            purviews (tuple[int]): Optionally restrict the possible purviews
                to a subset of the subsystem. This may be useful for _e.g._
                finding only concepts that are "about" a certain subset of
                nodes.

        Returns:
            MaximallyIrreducibleCauseOrEffect: The |MIC| or |MIE|.
        """
        purviews = self.potential_purviews(direction, mechanism, purviews)

        if not purviews:
            max_mip = _null_ria(direction, mechanism, ())
        else:
            max_mip = max(self.find_mip(direction, mechanism, purview)
                          for purview in purviews)

        if direction == Direction.CAUSE:
            return MaximallyIrreducibleCause(max_mip)
        elif direction == Direction.EFFECT:
            return MaximallyIrreducibleEffect(max_mip)
        return validate.direction(direction)

    def mic(self, mechanism, purviews=False):
        """Return the mechanism's maximally-irreducible cause (|MIC|).

        Alias for |find_mice()| with ``direction`` set to |CAUSE|.
        """
        return self.find_mice(Direction.CAUSE, mechanism, purviews=purviews)

    def mie(self, mechanism, purviews=False):
        """Return the mechanism's maximally-irreducible effect (|MIE|).

        Alias for |find_mice()| with ``direction`` set to |EFFECT|.
        """
        return self.find_mice(Direction.EFFECT, mechanism, purviews=purviews)

    def phi_max(self, mechanism):
        """Return the |small_phi_max| of a mechanism.

        This is the maximum of |small_phi| taken over all possible purviews.
        """
        return min(self.mic(mechanism).phi, self.mie(mechanism).phi)

    # Big Phi methods
    # =========================================================================

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
        cause = MaximallyIrreducibleCause(
            _null_ria(Direction.CAUSE, (), (), cause_repertoire))
        # Null effect.
        effect = MaximallyIrreducibleEffect(
            _null_ria(Direction.EFFECT, (), (), effect_repertoire))

        # All together now...
        return Concept(mechanism=(),
                       cause=cause,
                       effect=effect,
                       subsystem=self)

    @time_annotated
    def concept(self, mechanism, purviews=False, cause_purviews=False,
                effect_purviews=False):
        """Return the concept specified by a mechanism within this subsytem.

        Args:
            mechanism (tuple[int]): The candidate set of nodes.

        Keyword Args:
            purviews (tuple[tuple[int]]): Restrict the possible purviews to
                those in this list.
            cause_purviews (tuple[tuple[int]]): Restrict the possible cause
                purviews to those in this list. Takes precedence over
                ``purviews``.
            effect_purviews (tuple[tuple[int]]): Restrict the possible effect
                purviews to those in this list. Takes precedence over
                ``purviews``.

        Returns:
            Concept: The pair of maximally irreducible cause/effect repertoires
            that constitute the concept specified by the given mechanism.
        """
        log.debug('Computing concept %s...', mechanism)

        # If the mechanism is empty, there is no concept.
        if not mechanism:
            log.debug('Empty concept; returning null concept')
            return self.null_concept

        # Calculate the maximally irreducible cause repertoire.
        cause = self.mic(mechanism, purviews=(cause_purviews or purviews))

        # Calculate the maximally irreducible effect repertoire.
        effect = self.mie(mechanism, purviews=(effect_purviews or purviews))

        log.debug('Found concept %s', mechanism)

        # NOTE: Make sure to expand the repertoires to the size of the
        # subsystem when calculating concept distance. For now, they must
        # remain un-expanded so the concept doesn't depend on the subsystem.
        return Concept(mechanism=mechanism, cause=cause, effect=effect,
                       subsystem=self)
