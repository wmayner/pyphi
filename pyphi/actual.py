#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# actual.py

"""
Methods for computing actual causation of subsystems and mechanisms.

Bidirectional analysis of a transition:
    Note: during the init the subsystem should always come with the past state.
    This is because the subsystem should take the past state as background
    states in both directions. Then in the "past" case, the subsystem state
    should be swapped, to condition on the current state. All functions that
    take subsystems assume that the subsystem has already been prepared in this
    way.

    "past": evaluate cause-repertoires given current state
        background: past, condition on: current, actual state: past

    "future": evaluate effect-repertoires given past state
        background: past, condition on: past, actual state: current

To do this with the minimal effort, the subsystem state and actual state have
to be swapped in the "past" case, after the subsystem is conditioned on the
background condition.
"""

# Todo: check that transition between past and current state is possible for
# every function


import logging
import numpy as np

from . import cache, validate, utils, compute
from .network import irreducible_purviews, Network
from .utils import powerset, bipartition, directed_bipartition, phi_eq
from .constants import DIRECTIONS, FUTURE, PAST, EPSILON
from .models import Part, ActualCut
from .node import Node
from .config import PRECISION
from .models import AcMip, AcMice, AcBigMip, _null_ac_mip
from .subsystem import mip_bipartitions

import itertools
from pprint import pprint
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# Create a logger for this module.
log = logging.getLogger(__name__)


# TODO! go through docs and make sure to say when things can be None
# TODO: validate that purview and mechanism args are explicitly *tuples*?
class Context:

    """A set of nodes in a network, with state transition

    Args:
        network (Network): The network the subsystem belongs to.
        state (tuple(int)):
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

    def __init__(self, network, before_state, after_state, cause_indices,
                 effect_indices, cut=None, mice_cache=None,
                 repertoire_cache=None):
        # The network this subsystem belongs to.
        self.network = network
        # The state the network is in.
        self._before_state = tuple(before_state)
        self._after_state = tuple(after_state)
        # The state of the system of the actual probably calculation. Default
        # to before state but will swtich depending on contexti
        self.nodes = tuple()
        self._position = 'before'
        self._state = before_state
        self._actual_state = after_state
        # TODO don't need map to ints anymore?
        self.cause_indices = network.parse_node_indices(cause_indices)
        self.effect_indices = network.parse_node_indices(effect_indices)
        # Remove duplicates, sort, and ensure indices are native Python `int`s
        # (for JSON serialization).
        self.node_indices = network.parse_node_indices(
            tuple(set(cause_indices+effect_indices)))
        # Get the external nodes.
        self.external_indices = tuple(
            set(network.node_indices) - set(self.node_indices))
        # The TPM conditioned on the state of the external nodes.
        self.tpm = utils.condition_tpm(
            self.network.tpm, self.external_indices,
            self._before_state)
        # TODO Validate.
        # validate.context(self)
        # The null cut (that leaves the system intact).
        self.null_cut = ActualCut((), self.cause_indices,
                                   (), self.effect_indices)
        # The unidirectional cut applied for phi evaluation within the
        self.cut = cut if cut is not None else self.null_cut
        # Only compute hash once.
        self._hash = hash((self.network, self.node_indices, self.before_state,
                           self.after_state, self.cut))
        # Get the subsystem's connectivity matrix. This is the network's
        # connectivity matrix, but with the cut applied, and with all
        # connections to/from external nodes severed.
        # TODO: validate that this is an ActualCut
        self.connectivity_matrix = self.cut.apply_cut(network.cm)
        # Get the perturbation probabilities for each node in the network
        self.perturb_vector = network.perturb_vector
        # Generate the nodes.
        self.nodes = tuple(Node(self, i) for i in self.node_indices)
        # TODO: Reimplement the matrix of connections which are severed due to
        # the cut
        # self.cut_matrix = self.cut.cut_matrix()
        # A cache for keeping core causes and effects that can be reused later
        # in the event that a cut doesn't effect them.
        self._mice_cache = cache.MiceCache(self, mice_cache)
        # Set up cause/effect repertoire cache.
        self._repertoire_cache = repertoire_cache or cache.DictCache()
        # The nodes represented in computed repertoire distributions. This
        # supports `MacroSubsystem`'s alternate TPM representation.
        self._dist_indices = self.network.node_indices

    @property
    def before_state(self):
        return self._before_state

    @property
    def after_state(self):
        return self._after_state

    @property
    def position(self):
        return self._position

    @property
    def state(self):
        return self._state

    @property
    def actual_state(self):
        return self._actual_state

    @position.setter
    def position(self, position):
        if position == 'before':
            self.state = self.before_state
            self.actual_state = self.after_state
            self._position = position
        elif position == 'after':
            self.state = self.after_state
            self.actual_state = self.before_state
            self._position = position

    @before_state.setter
    def before_state(self, state):
        # Cast state to a tuple so it can be hashed and properly used as
        # np.array indices.
        before_state = tuple(state)
        self._before_state = before_state
        # TODO Validate.
        # validate.context(self)
        for node in self.nodes:
            node.state = state[node.index]

    @after_state.setter
    def after_state(self, state):
        # Cast state to a tuple so it can be hashed and properly used as
        # np.array indices.
        after_state = tuple(state)
        self._after_state = after_state
        # TODO Validate.
        # validate.context(self)

    @state.setter
    def state(self, state):
        # Cast state to a tuple so it can be hashed and properly used as
        # np.array indices.
        state = tuple(state)
        self._state = state
        # update the state of all subsystem nodes
        for node in self.nodes:
            node.state = state[node.index]

    @actual_state.setter
    def actual_state(self, state):
        actual_state = tuple(state)
        self._actual_state = actual_state

    @property
    def cm(self):
        """Alias for ``connectivity_matrix`` attribute."""
        return self.connectivity_matrix

    @cm.setter
    def cm(self, cm):
        self.connectivity_matrix = cm

    @property
    def size(self):
        """The size of this Subsystem."""
        return len(self.node_indices)

    @property
    def is_cut(self):
        """Return whether this Subsystem has a cut applied to it."""
        return self.cut != self.null_cut

    @property
    def cut_indices(self):
        """The indices of this system to be cut for |big_phi| computations.

        This was added to support ``MacroSubsystem``, which cuts indices other
        than ``self.node_indices``.
        """
        return self.node_indices

    def repertoire_cache_info(self):
        """Report repertoire cache statistics."""
        return self._repertoire_cachce.info()

    def __repr__(self):
        return "Context(cause:" + repr(self.indices2nodes(self.cause_indices)) + \
            ", effect:" + repr(self.indices2nodes(self.effect_indices)) + ")"

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        """Return whether this subsystem is equal to the other object.

        Two subsystems are equal if their sets of nodes, networks, and cuts are
        equal."""
        return (set(self.node_indices) == set(other.node_indices)
                and self.before_state == other.before_state
                and self.after_state == other.after_state
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

    # TODO jsonify
    # def to_json(self):
    #    return {
    #         'node_indices': jsonify(self.node_indices),
    #         'cut': jsonify(self.cut),
    #    }

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

    @cache.method('_repertoire_cache', DIRECTIONS[PAST])
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

        # Check that the current position is "before", otherwise adjust
        if not self.position == 'after':
            self.position = 'after'
        # If the purview is empty, the distribution is empty; return the
        # multiplicative identity.
        if not purview:
            return np.array([1.0])

        # If the mechanism is empty, nothing is specified about the past state
        # of the purview -- return the purview's maximum entropy distribution.
        max_entropy_dist = utils.max_entropy_distribution(
            purview, len(self._dist_indices),
            tuple(self.perturb_vector[i] for i in purview))
        if not mechanism:
            return max_entropy_dist

        # Preallocate the mechanism's conditional joint distribution.
        # TODO extend to nonbinary nodes
        cjd = np.ones(tuple(2 if i in purview else
                            1 for i in self._dist_indices))

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

        return utils.normalize(cjd)

    @cache.method('_repertoire_cache', DIRECTIONS[FUTURE])
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

        # Check that the position is "before", else adjust
        if not self.position == 'before':
            self.position = 'before'
        purview_nodes = self.indices2nodes(purview)
        mechanism_nodes = self.indices2nodes(mechanism)

        # If the purview is empty, the distribution is empty, so return the
        # multiplicative identity.
        if not purview:
            return np.array([1.0])

        # Preallocate the purview's joint distribution
        # TODO extend to nonbinary nodes
        accumulated_cjd = np.ones(
            [1] * len(self._dist_indices) + [2 if i in purview else
                                             1 for i in self._dist_indices])

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
            second_half_shape = [1] * len(self._dist_indices)
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
            accumulated_cjd.shape[len(self._dist_indices):])

        return accumulated_cjd

    # TODO check if the cache is faster
    def _get_repertoire(self, direction):
        """Returns the cause or effect repertoire function based on a
        direction.

        Args:
            direction (str): The temporal direction, specifiying the cause or
                effect repertoire.

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

    def state_probability(self, repertoire, purview):
        """ The dimensions of the repertoire that correspond to the fixed nodes
        are collapsed onto their state. All other dimension should be singular
        already (repertoire size and fixed_nodes need to match), and thus
        should receive 0 as the conditioning index. A single probability is
        returned.
        """
        index = tuple(self.actual_state[i] if i in purview else 0
                      for i in range(len(self.state)))
        return repertoire[index]

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

    def cause_coefficient(self, mechanism, purview, norm=True):
        """ Return the cause coefficient for a mechanism in a state over a
        purview in the actual past state """
        if not self.position == 'after':
            self.position = 'after'
        if norm:
            normalization = self.state_probability(self.cause_repertoire((), purview), purview)
        else:
            normalization = 1
        return self.state_probability(self.cause_repertoire(mechanism, purview),
                                      purview) / normalization

    def effect_info(self, mechanism, purview):
        """Return the effect information for a mechanism over a purview."""
        return round(utils.hamming_emd(
            self.effect_repertoire(mechanism, purview),
            self.unconstrained_effect_repertoire(purview)),
            PRECISION)

    def effect_coefficient(self, mechanism, purview, norm=True):
        """ Return the effect coefficient for a mechanism in a state over a
        purview in the actual future state """
        if not self.position == 'before':
            self.position = 'before'
        if norm:
            normalization = self.state_probability(self.effect_repertoire((), purview), purview)
        else:
            normalization = 1
        return self.state_probability(self.effect_repertoire(mechanism, purview),
                                      purview) / normalization

    def cause_effect_info(self, mechanism, purview):
        """Return the cause-effect information for a mechanism over a
        purview.

        This is the minimum of the cause and effect information."""
        return min(self.cause_info(mechanism, purview),
                   self.effect_info(mechanism, purview))

    # MIP methods
    # =========================================================================

    def find_mip(self, direction, mechanism, purview,
                 norm=True, allow_neg=False):
        """ Return the cause coef mip minimum information partition for a mechanism
            over a cause purview.
            Returns:
                ap_phi_min: The min. difference of the actual probabilities of
                            the unpartitioned cause and its MIP
            Todo: also return cut etc. ?
        """

        repertoire = self._get_repertoire(direction)
        alpha_min = float('inf')
        # Calculate the unpartitioned probability to compare against the
        # partitioned probabilities
        unpartitioned_repertoire = repertoire(mechanism, purview)
        probability = self.state_probability(unpartitioned_repertoire, purview)
        if norm:
            unconstrained_repertoire = repertoire((), purview)
            normalization = self.state_probability(
                unconstrained_repertoire, purview)
        else:
            normalization = 1
        # Loop over possible MIP bipartitions
        for part0, part1 in mip_bipartitions(mechanism, purview):
            # Find the distance between the unpartitioned repertoire and
            # the product of the repertoires of the two parts, e.g.
            #   D( p(ABC/ABC) || p(AC/C) * p(B/AB) )
            part1rep = repertoire(part0.mechanism, part0.purview)
            part2rep = repertoire(part1.mechanism, part1.purview)
            partitioned_repertoire = part1rep * part2rep
            partitioned_probability = self.state_probability(
                partitioned_repertoire, purview)
            alpha = (probability - partitioned_probability) / normalization
            # First check for 0
            # Default: don't count contrary causes and effects
            if phi_eq(alpha, 0) or (alpha < 0 and not allow_neg):
                return AcMip(state=self.state,
                             direction=direction,
                             mechanism=mechanism,
                             purview=purview,
                             partition=(part0, part1),
                             probability=probability,
                             partitioned_probability=partitioned_probability,
                             unconstrained_probability=normalization,
                             alpha=0.0)
            # Then take closest to 0
            if (abs(alpha_min) - abs(alpha)) > EPSILON:
                alpha_min = alpha
                acmip = AcMip(state=self.state,
                              direction=direction,
                              mechanism=mechanism,
                              purview=purview,
                              partition=(part0, part1),
                              probability=probability,
                              partitioned_probability=partitioned_probability,
                              unconstrained_probability=normalization,
                              alpha=alpha_min)
        return acmip

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

        # Purviews are already filtered in network._potential_purviews
        # over the full network connectivity matrix. However, since the cm
        # is cut/smaller we check again here.
        return irreducible_purviews(self.connectivity_matrix,
                                    direction, mechanism, purviews)

    # TODO: Implement mice cache
    # @cache.method('_mice_cache')
    def find_mice(self, direction, mechanism, purviews=False,
                  norm=True, allow_neg=False):
        """Return the maximally irreducible cause or effect coefficient for a mechanism.

        Args:
            direction (str): The temporal direction, specifying cause or
                effect.
            mechanism (tuple(int)): The mechanism to be tested for
                irreducibility.

        Keyword Args:
            purviews (tuple(int)): Optionally restrict the possible purviews
                to a subset of the subsystem. This may be useful for _e.g._
                finding only concepts that are "about" a certain subset of
                nodes.
        Returns:
            ac_mice: The maximally-irreducible actual cause or effect.

        .. note::
            Strictly speaking, the AC_MICE is a pair of coefficients: the
            actual cause and actual effect of a mechanism. Here, we return only
            information corresponding to one direction, |past| or |future|,
            i.e., we return an actual cause or actual effect coefficient, not
            the pair of them.
        """

        purviews = self._potential_purviews(direction, mechanism, purviews)

        # Find the maximal MIP over the remaining purviews.
        if not purviews:
            maximal_mip = _null_ac_mip(self.state, direction, mechanism, None)
        else:
            # This max should be most positive
            maximal_mip = max(self.find_mip(direction, mechanism,
                                            purview, norm, allow_neg)
                              for purview in purviews)

        # Construct the corresponding AcMICE.
        return AcMice(maximal_mip)

    def core_cause(self, mechanism, purviews=False):
        """Returns the core cause repertoire of a mechanism.

        Alias for |find_mice| with ``direction`` set to |past|."""
        return self.find_mice('past', mechanism, purviews=purviews)

    def core_effect(self, mechanism, purviews=False):
        """Returns the core effect repertoire of a mechanism.

        Alias for |find_mice| with ``direction`` set to |past|."""
        return self.find_mice('future', mechanism, purviews=purviews)

# ===========================================================================
# Printed Results
# ============================================================================


def nice_ac_composition(account):
    if account:
        if account[0].direction == DIRECTIONS[PAST]:
            dir_arrow = '<--'
        elif account[0].direction == DIRECTIONS[FUTURE]:
            dir_arrow = '-->'
        else:
            validate.direction(account.direction)
        actions = [["{0:.4f}".format(round(action.alpha, 4)),
                    action.mechanism, dir_arrow, action.purview]
                   for action in account]
        return actions
    else:
        return None


def multiple_states_nice_ac_composition(network, transitions, cause_indices,
                                        effect_indices, mechanisms=False,
                                        purviews=False, norm=True,
                                        allow_neg=False):
    """nice composition for multiple pairs of states
    Args: As above
        transitions (list(2 state tuples)):
            The first is past the second current.
            For 'past' current belongs to subsystem and past is the second
            state. Vice versa for "future"
    """
    for transition in transitions:
        context = Context(network, transition[0], transition[1], cause_indices,
                          effect_indices)
        cause_account = directed_account(context, 'past', mechanisms, purviews,
                                         norm, allow_neg)
        effect_account = directed_account(context, 'future', mechanisms,
                                          purviews, norm, allow_neg)
        print('#####################################')
        print(transition)
        print('- cause coefs ----------------------')
        pprint(nice_ac_composition(cause_account, 'past'))
        print('- effect coefs ----------------------')
        pprint(nice_ac_composition(effect_account, 'future'))
        print('---------------------------')

# ============================================================================
# Average over mechanisms - constellations
# ============================================================================


def directed_account(context, direction, mechanisms=False, purviews=False,
                     norm=True, allow_neg=False):
    """Set of all AcMice of the specified direction"""
    if mechanisms is False:
        if direction == 'past':
            mechanisms = powerset(context.effect_indices)
        elif direction == 'future':
            mechanisms = powerset(context.cause_indices)
    actions = [context.find_mice(direction, mechanism, purviews=purviews,
                                 norm=norm, allow_neg=allow_neg)
               for mechanism in mechanisms]
    # Filter out falsy acmices, i.e. those with effectively zero ac_diff.
    return tuple(filter(None, actions))

# ============================================================================
# AcBigMips and System cuts
# ============================================================================


def account_distance(A1, A2):
    """Return the distance between two accounts. Here that is just the
    difference in sum(alpha)

    Args:
        A1 (tuple(Concept)): The first constellation.
        A2 (tuple(Concept)): The second constellation.

    Returns:
        distance (``float``): The distance between the two constellations in
            concept-space.
    """
    return (sum([action.alpha for action in A1])
            - sum([action.alpha for action in A2]))


def _null_ac_bigmip(context, direction):
    """Returns an ac |BigMip| with zero |big_ap_phi| and empty constellations.
    For direction = bidirectional, the subsystem is subsystem_past and
    subsystem2_or_actual_state is subsystem_future. """
    if direction == DIRECTIONS[FUTURE]:
        return AcBigMip(context=context, direction=direction, alpha=0.0,
                        unpartitioned_account=(), partitioned_account=())
    else:
        return AcBigMip(context=context, direction=direction, alpha=0.0,
                        unpartitioned_account=(), partitioned_account=())


# TODO: single node BigMip
def _evaluate_cut_directed(context, cut, account, direction):
    """ Returns partitioned constellation for one direction past/future of the
    transition. For direction = bidirectional, the uncut subsystem is
    subsystem_past and uncut_subsystem2_or_actual_state is subsystem_future. To
    make cut subsystem: To have the right background, the init state for the
    subsystem should always be the past_state. In past direction after
    subsystem is created the actual state and the system state need to be
    swapped."""

    # Important to use make_ac_subsystem because otherwise the past
    # cut_subsystem has the wrong conditioning.
    cut_context = Context(context.network,
                          context.before_state,
                          context.after_state,
                          context.cause_indices,
                          context.effect_indices,
                          cut)
    # TODO: Implement shortcuts to avoid recomputing actions?
    # if config.ASSUME_CUTS_CANNOT_CREATE_NEW_CONCEPTS:
    #     mechanisms = set([c.mechanism for c in unpartitioned_constellation])
    # else:
    # mechanisms = set([c.mechanism for c in unpartitioned_constellation] +
    #                      list(cut_mechanism_indices(uncut_subsystem, cut)))
    partitioned_account = directed_account(cut_context, direction)
    return partitioned_account


def _evaluate_cut(context, cut, unpartitioned_account, direction=None):
    """Find the |AcBigMip| for a given cut. For direction = bidirectional, the
    uncut subsystem is subsystem_past and uncut_subsystem2_or_actual_state is
    subsystem_future. """
    cut_context = Context(context.network,
                          context.before_state,
                          context.after_state,
                          context.cause_indices,
                          context.effect_indices,
                          cut)
    if not direction:
        direction = 'bidirectional'
    if direction == 'bidirectional':
        past_partitioned_account = directed_account(cut_context, 'past')
        future_partitioned_account = directed_account(cut_context, 'future')
        partitioned_account = tuple(past_partitioned_account
                                    + future_partitioned_account)
    else:
        partitioned_account = directed_account(cut_context, direction)

    log.debug("Finished evaluating cut {}.".format(cut))
    alpha = account_distance(unpartitioned_account, partitioned_account)

    return AcBigMip(
        alpha=alpha,
        direction=direction,
        unpartitioned_account=unpartitioned_account,
        partitioned_account=partitioned_account,
        context=context,
        cut=cut)


def _get_cuts(context):
    """ A list of possible cuts to a context.
    Returns:
        cuts: A list of cuts to evaluate. """

    # TODO: Add one-cut approximation as an option.
    # if config.CUT_ONE_APPROXIMATION:
    #     bipartitions = directed_bipartition_of_one(subsystem.node_indices)
    # else:
    cause_bipartitions = bipartition(context.cause_indices)
    effect_bipartitions = directed_bipartition(context.effect_indices)
    # The first element of the list is the null cut.
    partitions = list(itertools.product(cause_bipartitions,
                                        effect_bipartitions))[1:]
    cuts = [ActualCut(part[0][0], part[0][1], part[1][0], part[1][1])
            for part in partitions]
    return cuts


def big_acmip(context, direction=None):
    """Return the minimal information partition of a context in a specific
    direction.

    Args:
        subsystem (Subsystem): The candidate set of nodes.
    Returns:
        big_mip (|BigMip|): A nested structure containing all the data from the
            intermediate calculations. The top level contains the basic MIP
            information for the given subsystem.
    """
    if not direction:
        direction = 'bidirectional'
    validate.direction(direction)
    log.info("Calculating big-alpha for {}...".format(context))

    if not context:
        log.info('Context {} is empty; returning null MIP '
                 'immediately.'.format(context))
        return _null_ac_bigmip(context, direction)

    submatrix_indices = np.ix_(context.node_indices, context.node_indices)
    cm = context.network.connectivity_matrix[submatrix_indices]
    # Get the number of weakly or strongly connected components.
    num_components, _ = connected_components(csr_matrix(cm),
                                             connection='weak')
    if num_components > 1:
        log.info('{} is not strongly/weakly connected; returning null MIP '
                 'immediately.'.format(context))
        return _null_ac_bigmip(context, direction)
    cuts = _get_cuts(context)

    log.debug("Finding unpartitioned account...")
    if direction == 'bidirectional':
        unpartitioned_account = tuple(directed_account(context, 'past')
                                      + directed_account(context, 'future'))
    else:
        unpartitioned_account = directed_account(context, direction)
    log.debug("Found unpartitioned account.")
    if not unpartitioned_account:
        # Short-circuit if there are no actions in the unpartitioned
        # account.
        result = _null_ac_bigmip(context, direction)
    else:
        ac_mip = _null_ac_bigmip(context, direction)
        ac_mip.alpha = float('inf')
        for i, cut in enumerate(cuts):
            new_ac_mip = _evaluate_cut(context, cut, unpartitioned_account,
                                       direction)
            log.debug("Finished {} of {} cuts.".format(
                i + 1, len(cuts)))
            if new_ac_mip < ac_mip:
                ac_mip = new_ac_mip
            # Short-circuit as soon as we find a MIP with effectively 0 phi.
            if not ac_mip:
                break
        result = ac_mip
    log.info("Finished calculating big-ac-phi data for {}.".format(context))
    log.debug("RESULT: \n" + str(result))
    return result

# ============================================================================
# Complexes
# ============================================================================


# TODO: Fix this to test whether the transition is possible
def contexts(network, before_state, after_state):
    """Return a generator of all **possible** subsystems of a network.
    """
    # TODO: Does not return subsystems that are in an impossible transitions.
    context_list = []
    # Elements without inputs are reducibe effects,
    # elements without outputs are reducible causes.
    possible_causes = np.where(np.sum(network.connectivity_matrix, 1) > 0)[0]
    possible_effects = np.where(np.sum(network.connectivity_matrix, 0) > 0)[0]
    for cause_subset in powerset(possible_causes):
        for effect_subset in powerset(possible_effects):
            context_list.append(Context(network, before_state, after_state,
                                        cause_subset, effect_subset))

    def not_empty(context):
        """
        Ensures both cause and effect indices are not empty
        """
        return bool(context.cause_indices and context.effect_indices)

    return list(filter(not_empty, context_list))


def nexus(network, before_state, after_state, direction=None):
    """Return a generator for all irreducible nexus of the network.
       Direction options are past, future, bidirectional. """
    if not isinstance(network, Network):
        raise ValueError(
            """Input must be a Network (perhaps you passed a Subsystem
            instead?)""")

    if not direction:
        direction = 'bidirectional'

    return tuple(filter(None, (big_acmip(context, direction) for context in
                               contexts(network, before_state, after_state))))


def causal_nexus(network, before_state, after_state, direction=None):
    """Return the causal nexus of the network."""
    if not isinstance(network, Network):
        raise ValueError(
            """Input must be a Network (perhaps you passed a Subsystem
            instead?)""")
    if not direction:
        direction = 'bidirectional'
    log.info("Calculating causal nexus...")
    result = nexus(network, before_state, after_state, direction)
    if result:
        result = max(result)
    else:
        empty_context = Context(network, before_state, after_state, (), ())
        result = _null_ac_bigmip(empty_context, direction)
    log.info("Finished calculating causal nexus.")
    log.debug("RESULT: \n" + str(result))
    return result

# ============================================================================
# True Causes
# ============================================================================


def nice_true_constellation(true_constellation):
    # TODO: Make sure the past and future purviews are ordered in the same way
    past_list = []
    future_list = []
    cause = '<--'
    effect = '-->'
    for event in true_constellation:
        if event.direction == DIRECTIONS[PAST]:
            past_list.append(["{0:.4f}".format(round(event.alpha, 4)),
                              event.mechanism, cause, event.purview])
        elif event.direction == DIRECTIONS[FUTURE]:
            future_list.append(["{0:.4f}".format(round(event.alpha, 4)),
                                event.mechanism, effect, event.purview])
        else:
            validate.direction(event.direction)

    true_list = [(past_list[event], future_list[event])
                 for event in range(len(past_list))]
    return true_list


def true_constellation(subsystem, past_state, future_state):
    """Set of all sets of elements that have true causes and true effects.
       Note: Since the true constellation is always about the full system,
       the background conditions don't matter and the subsystem should be
       conditioned on the current state."""
    log.info("Calculating true causes ...")
    network = subsystem.network
    nodes = subsystem.node_indices
    state = subsystem.state
    past_context = Context(network, past_state, state, nodes, nodes)
    true_causes = directed_account(past_context, direction=DIRECTIONS[PAST])
    log.info("Calculating true effects ...")
    future_context = Context(network, state, future_state, nodes, nodes)
    true_effects = directed_account(future_context,
                                    direction=DIRECTIONS[FUTURE])
    true_mechanisms = set([c.mechanism for c in true_causes]).\
        intersection(c.mechanism for c in true_effects)
    if true_mechanisms:
        true_events = true_causes + true_effects
        result = tuple(filter(lambda t: t.mechanism in true_mechanisms,
                              true_events))
        log.info("Finished calculating true events.")
        log.debug("RESULT: \n" + str(result))
        return result
    else:
        log.info("Finished calculating, no true events.")
        return None


def true_events(network, past_state, current_state, future_state, indices=None,
                main_complex=None):
    """Set of all mechanisms that have true causes and true effects within the
    complex.

    Args:
        network (Network):

        past_state (tuple(int)): The state of the network at t-1
        current_state (tuple(int)): The state of the network at t
        future_state (tuple(int)): The state of the network at t+1

    Optional Args:
        indices (tuple(int)): The indices of the main complex
        main_complex (big_mip): The main complex

        Note: If main_complex is given, then indices is ignored.

    Returns:
        events (tuple(actions)): List of true events in the main complex
    """
    # TODO: validate triplet of states
    if not indices and not main_complex:
        main_complex = compute.main_complex(network, current_state)
    elif not main_complex:
        main_complex = compute.big_mip(network, current_state, indices)
    # Calculate true causes
    nodes = main_complex.subsystem.node_indices
    past_context = Context(network, past_state, current_state, nodes, nodes)
    true_causes = directed_account(past_context, direction=DIRECTIONS[PAST])
    # Calculate true_effects
    future_context = Context(network, current_state, future_state, nodes, nodes)
    true_effects = directed_account(future_context,
                                    direction=DIRECTIONS[FUTURE])
    true_mechanisms = set([c.mechanism for c in true_causes]).\
        intersection(c.mechanism for c in true_effects)
    # TODO: Make sort function that sorts events by mechanism so that
    # causes and effects match up.
    if true_mechanisms:
        true_causes = tuple(filter(lambda t: t.mechanism in true_mechanisms,
                                   true_causes))
        true_effects = tuple(filter(lambda t: t.mechanism in true_mechanisms,
                                    true_effects))
        true_events = tuple([true_causes[i], true_effects[i]] for i in
                            range(len(true_mechanisms)))
    else:
        true_events = ()
    return true_events


def extrinsic_events(network, past_state, current_state, future_state,
                     indices=None, main_complex=None):
    """Set of all mechanisms that have true causes and true effects within the
    complex.

    Args:
        network (Network):
        past_state (tuple(int)): The state of the network at t-1
        current_state (tuple(int)): The state of the network at t
        future_state (tuple(int)): The state of the network at t+1

    Optional Args:
        indices (tuple(int)): The indices of the main complex
        main_complex (big_mip): The main complex

        Note: If main_complex is given, then indices is ignored.

    Returns:
        events (tuple(actions)): List of true events in the main complex
    """
    # TODO: validate triplet of states
    if not indices and not main_complex:
        main_complex = compute.main_complex(network, current_state)
    elif not main_complex:
        main_complex = compute.big_mip(network, current_state, indices)
    # Identify the potential mechanisms for extrinsic events within the main
    # complex
    all_nodes = network.node_indices
    mechanisms = list(utils.powerset(main_complex.subsystem.node_indices))[1:]
    # Calculate true causes
    past_context = Context(network, past_state, current_state, all_nodes,
                           all_nodes)
    true_causes = directed_account(past_context, direction=DIRECTIONS[PAST],
                                   mechanisms=mechanisms)
    # Calculate true_effects
    future_context = Context(network, current_state, future_state, all_nodes,
                             all_nodes)
    true_effects = directed_account(future_context,
                                    direction=DIRECTIONS[FUTURE],
                                    mechanisms=mechanisms)
    true_mechanisms = set([c.mechanism for c in true_causes]).\
        intersection(c.mechanism for c in true_effects)
    # TODO: Make sort function that sorts events by mechanism so that
    # causes and effects match up.
    if true_mechanisms:
        true_causes = tuple(filter(lambda t: t.mechanism in true_mechanisms,
                                   true_causes))
        true_effects = tuple(filter(lambda t: t.mechanism in true_mechanisms,
                                    true_effects))
        true_events = tuple([true_causes[i], true_effects[i]] for i in
                            range(len(true_mechanisms)))
    else:
        true_events = ()
    return true_events
