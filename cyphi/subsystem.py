#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from itertools import chain
from collections import namedtuple, Iterable
from joblib import Parallel, delayed
from . import constants
from . import validate
# TODO use namespaces more (honking great idea, etc.)
from .utils import (marginalize_out, emd, hamming_emd, max_entropy_distribution, powerset,
                    bipartition, phi_eq)
from .models import Cut, Mip, Part, Mice, Concept, BigMip


# TODO remove
DEBUG = False
def dprint(*args):
  if DEBUG:
    print(*args)


# TODO? refactor the computational methods out of the class so they explicitly
# take a subsystem as a parameter
class Subsystem:

    """A set of nodes in a network.

    Represents a candidate set for |phi| calculation.
    """

    def __init__(self, nodes, current_state, past_state, network):
        """
        :param nodes: A list of nodes in this subsystem
        :type nodes: ``[Node]``
        :param current_state: The current state of this subsystem
        :type current_state: ``np.ndarray``
        :param past_state: The past state of this subsystem
        :type past_state: ``np.ndarray``
        :param network: The network the subsystem is part of
        :type network: ``Network``
        """
        # This nodes in this subsystem.
        self.nodes = tuple(nodes)

        self.current_state = current_state
        self.past_state = past_state
        # Make the state and past state immutable (for hashing).
        self.current_state.flags.writeable = False
        self.past_state.flags.writeable = False

        # The network this subsystem belongs to.
        self.network = network

        # TODO use properties?
        # (https://docs.python.org/2/library/functions.html#property)

    def __repr__(self):
        return "Subsystem(" + ", ".join([repr(self.nodes),
                                         repr(self.current_state),
                                         repr(self.past_state)]) + ")"

    def __str__(self):
        return "Subsystem([" + ", ".join([str(list(map(str, self.nodes))),
                                          str(self.current_state),
                                          str(self.past_state),
                                          str(self.network)]) + ")"

    def __eq__(self, other):
        """Two subsystems are equal if their sets of nodes, current and past
        states, and networks are equal."""
        return (set(self.nodes) == set(other.nodes) and
                np.array_equal(self.current_state, other.current_state) and
                np.array_equal(self.past_state, other.past_state) and
                self.network == other.network)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((frozenset(self.nodes), self.current_state.tostring(),
                     self.past_state.tostring(), self.network))

    def cause_repertoire(self, mechanism, purview, cut=None):
        """Return the cause repertoire of a mechanism over a purview.

        :param mechanism: The mechanism for which to calculate the cause
            repertoire
        :type mechanism: ``[Node]``
        :param purview: The purview over which to calculate the cause
            repertoire
        :type purview: ``[Node]``

        :returns: The cause repertoire of the mechanism over a purview
        :rtype: ``np.ndarray``
        """
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # NOTE: In the Matlab version's terminology,
        #
        # "Cause repertoire" is "backward repertoire"
        # "Mechanism" is "numerator"
        # "Purview" is "denominator"
        # ``conditioned_tpm`` is ``next_num_node_distribution``
        # ``cjd`` is ``numerator_conditional_joint``
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        validate.nodelist(mechanism, 'Mechanism')
        validate.nodelist(purview, 'Purview')
        # Default cut is the null cut that leaves the system intact
        if not cut:
            cut = Cut((), self.nodes)
        # If a cut was provided, validate it
        else:
            cut = validate.cut(self, cut)
        # If the mechanism is empty, nothing is specified about the past state
        # of the purview, so just return the purview's maximum entropy
        # distribution.
        if (len(mechanism) == 0):
            return max_entropy_distribution(purview, self.network)
        # If the purview is empty, the distribution is empty, so return the
        # multiplicative identity.
        if (len(purview) == 0):
            return 1
        # Preallocate the mechanism's conditional joint distribution.
        # TODO extend to nonbinary nodes
        cjd = np.ones(tuple(2 if node in purview else
                            1 for node in self.network.nodes))
        # Loop over all nodes in this mechanism, successively taking the
        # product (with expansion/broadcasting of singleton dimensions) of each
        # individual node's CPT (conditioned on that node's state) in order to
        # get the conditional joint distribution for the whole mechanism
        # (conditioned on the whole mechanism's state). After normalization,
        # this is the cause repertoire. Normalization happens after this loop.
        for mechanism_node in mechanism:
            # TODO extend to nonbinary nodes

            # We're conditioning on this node's state, so take the
            # probabilities that correspond to that state (The TPM subtracted
            # from 1 gives the probability that the node is off).
            conditioned_tpm = (mechanism_node.tpm if
                               self.current_state[mechanism_node.index] == 1
                               else 1 - mechanism_node.tpm)

            # TODO explicit inputs to nodes (right now each node is implicitly
            # connected to all other nodes, since initializing a Network with a
            # connectivity matrix isn't implemented yet)
            # TODO add this when inputs are implemented:
            # ... and node in self.input_nodes):
            non_purview_inputs = set(self.network.nodes) - set(purview)
            # Collect the nodes in the network who had inputs to this mechanism
            # that were severed by this subsystem's cut.
            severed_inputs = set([n for n in self.network.nodes if
                                  (n in cut.severed and
                                   mechanism_node in cut.intact)])
            # Fixed boundary condition nodes are those that are outside this
            # subsystem, and are not in the purview or have been severed by a
            # cut.
            boundary_inputs = ((non_purview_inputs | severed_inputs)
                               - set(self.nodes))
            # We will marginalize-out nodes that are within the subsystem, but
            # are either not in the purview or severed by a cut.
            marginal_inputs = ((non_purview_inputs | severed_inputs)
                               - boundary_inputs)
            # Condition the CPT on the past states of the external input nodes.
            # These nodes are treated as fixed boundary conditions. We collapse
            # the dimensions corresponding to the fixed nodes so they contain
            # only the probabilities that correspond to their past states.
            for node in boundary_inputs:
                conditioning_indices = [slice(None)] * self.network.size
                conditioning_indices[node.index] = [self.past_state[node.index]]
                conditioned_tpm = conditioned_tpm[conditioning_indices]
            # Marginalize-out the nodes in this subsystem with inputs to this
            # mechanism that are not in the purview and whose connections to
            # this mechanism have not been severed by a subsystem cut.
            for node in marginal_inputs:
                conditioned_tpm = marginalize_out(node, conditioned_tpm)
            # Incorporate this node's CPT into the mechanism's conditional
            # joint distribution by taking the product (with singleton
            # broadcasting)
            cjd = np.multiply(cjd, conditioned_tpm)
        # Finally, normalize by the marginal probability of the past state to
        # get the mechanism's CJD
        cjd_sum = np.sum(cjd)
        # Don't divide by zero
        if cjd_sum != 0:
            cjd = np.divide(cjd, cjd_sum)
        # Note that we're not returning a distribution over all the nodes in
        # the network, only a distribution over the nodes in the purview. This
        # is because we never actually need to compare proper cause/effect
        # repertoires, which are distributions over the whole network; we need
        # only compare the purview-repertoires with each other, since cut vs.
        # whole comparisons are only ever done over the same purview.
        return cjd

    def effect_repertoire(self, mechanism, purview, cut=None):
        """Return the effect repertoire of a mechanism over a purview.

        :param mechanism: The mechanism for which to calculate the effect
            repertoire
        :type mechanism: ``[Node]``
        :param purview: The purview over which to calculate the effect
            repertoire
        :type purview: ``[Node]``

        :returns: The effect repertoire of the mechanism over a purview
        :rtype: ``np.ndarray``
        """
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # NOTE: In the Matlab version's terminology,
        #
        # "Effect repertoire" is "forward repertoire"
        # "Mechanism" is "numerator"
        # "Purview" is "denominator"
        # ``conditioned_tpm`` is ``next_denom_node_distribution``
        # ``accumulated_cjd`` is ``denom_conditional_joint``
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        validate.nodelist(mechanism, 'Mechanism')
        validate.nodelist(purview, 'Purview')
        # Default cut is the null cut that leaves the system intact
        if not cut:
            cut = Cut((), self.nodes)
        # If a cut was provided, validate it
        else:
            cut = validate.cut(self, cut)
        # If the purview is empty, the distribution is empty, so return the
        # multiplicative identity.
        if (len(purview) == 0):
            return 1
        # Preallocate the purview's joint distribution
        # TODO extend to nonbinary nodes
        accumulated_cjd = np.ones(tuple([1] * self.network.size +
                                        [2 if node in purview else 1
                                         for node in self.network.nodes]))

        # TODO explicit inputs to nodes (right now each node is implicitly
        # connected to all other nodes, since initializing a Network with a
        # connectivity matrix isn't implemented yet)
        mechanism_inputs = set(mechanism)
        # Nodes outside this subsystem.
        external_inputs = set(self.network.nodes) - set(self.nodes)

        # Loop over all nodes in the purview, successively taking the product
        # (with 'expansion'/'broadcasting' of singleton dimensions) of each
        # individual node's TPM in order to get the joint distribution for the
        # whole purview. After conditioning on the mechanism's state and that
        # of external nodes, this will be the effect repertoire as a
        # distribution over the purview.
        for purview_node in purview:
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
            # Allocate the TPM.
            tpm = np.zeros([2] * self.network.size +
                           [2 if i == purview_node.index else 1 for i in
                            range(self.network.size)])
            tpm_off_indices = [slice(None)] * self.network.size + \
                [0] * self.network.size
            # Insert the TPM for the node being off.
            tpm[tpm_off_indices] = 1 - purview_node.tpm
            # Insert the TPM for the node being on.
            tpm_on_indices = [slice(None)] * self.network.size + \
                [1 if i == purview_node.index else 0 for i in
                 range(self.network.size)]
            tpm[tpm_on_indices] = purview_node.tpm

            # Collect the nodes in the network who had inputs to this mechanism
            # that were severed by this subsystem's cut.
            severed_inputs = set([n for n in self.network.nodes if
                                    (n in cut.severed and
                                    purview_node in cut.intact)])
            dprint('Severed inputs:', severed_inputs)
            severed_mechanism_inputs = severed_inputs & mechanism_inputs
            # We will marginalize-out nodes that are within the subsystem, but
            # are either not in the purview or severed by a cut.
            marginal_inputs = ((set(self.nodes) - mechanism_inputs) |
                                severed_inputs)

            # Marginalize-out nodes in this subsystem with inputs to the
            # purview that aren't in the mechanism, and the nodes whose
            # connections to the purview have been severed.
            for node in marginal_inputs:
                tpm = marginalize_out(node, tpm)
            # TODO do I need this?
                # for node in severed_mechanism_inputs:
                #     # TODO expand dimensions here
                #     print('todo')
            # Incorporate this node's CPT into the future_nodes' conditional
            # joint distribution by taking the product (with singleton
            # broadcasting).
            accumulated_cjd = np.multiply(accumulated_cjd, tpm)

        # Now we condition on the state of the boundary nodes, whose states we
        # fix (by collapsing the CJD onto those states):
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Fixed boundary condition nodes are those that are outside this
        # subsystem, and are not in the purview or have been severed by a
        # cut.
        boundary_inputs = mechanism_inputs | external_inputs
        # Initialize the conditioning indices, taking the slices as singleton
        # lists-of-lists for later flattening with `chain`.
        # TODO! are the external nodes really the ones outside this
        # subsystem?
        conditioning_indices = [[slice(None)]] * self.network.size
        for node in boundary_inputs:
            # Preserve singleton dimensions with `np.newaxis`
            conditioning_indices[node.index] = [self.current_state[node.index],
                                                np.newaxis]
        # Flatten the indices
        conditioning_indices = list(chain.from_iterable(conditioning_indices))
        # Obtain the actual conditioned distribution by indexing with the
        # conditioning indices
        accumulated_cjd = accumulated_cjd[tuple(conditioning_indices)]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # The distribution still has twice as many dimensions as the network
        # has nodes, with the first half of the shape now all singleton
        # dimensions, so we reshape to eliminate those singleton dimensions
        # (the second half of the shape may also contain singleton dimensions,
        # depending on how many nodes are in the purview).
        accumulated_cjd = accumulated_cjd.reshape(
            accumulated_cjd.shape[self.network.size:2 * self.network.size])

        # Note that we're not returning a distribution over all the nodes in
        # the network, only a distribution over the nodes in the purview. This
        # is because we never actually need to compare proper cause/effect
        # repertoires, which are distributions over the whole network; we need
        # only compare the purview-repertoires with each other, since cut vs.
        # whole comparisons are only ever done over the same purview.
        return accumulated_cjd

    def _get_repertoire(self, direction):
        """Returns the cause or effect repertoire function based on a
        direction."""
        if direction == 'past':
            return self.cause_repertoire
        elif direction == 'future':
            return self.effect_repertoire

    def _unconstrained_repertoire(self, direction, purview, cut=None):
        """Return the unconstrained cause or effect repertoire over a
        purview."""
        return self._get_repertoire(direction)([], purview, cut)


    def unconstrained_cause_repertoire(self, purview, cut=None):
        """Return the unconstrained cause repertoire for a purview.

        This is just the cause repertoire in the absence of any mechanism.
        """
        return self._unconstrained_repertoire('past', purview)

    def unconstrained_effect_repertoire(self, purview, cut=None):
        """Return the unconstrained effect repertoire for a purview.

        This is just the effect repertoire in the absence of any mechanism.
        """
        return self._unconstrained_repertoire('future', purview, cut)

    def _expand_repertoire(self, direction, mechanism, purview, repertoire,
                           cut):
        """Return the unconstrained cause or effect repertoire based on a
        direction."""
        validate.direction(direction)
        non_purview_nodes = set(self.nodes) - set(purview)
        return (repertoire * self._unconstrained_repertoire(direction,
                                                            non_purview_nodes,
                                                            cut))

    def expand_cause_repertoire(self, mechanism, purview, repertoire, cut=None):
        """Expand a partial cause repertoire over a purview to a distribution
        over the entire subsystem's state space."""
        return self._expand_repertoire('past', mechanism, purview, repertoire,
                                       cut)

    def expand_effect_repertoire(self, mechanism, purview, repertoire, cut=None):
        """Expand a partial effect repertoire over a purview to a distribution
        over the entire subsystem's state space."""
        return self._expand_repertoire('future', mechanism, purview,
                                       repertoire, cut)

    def cause_info(self, mechanism, purview, cut=None):
        """Return the cause information for a mechanism over a purview."""
        return hamming_emd(self.cause_repertoire(mechanism, purview),
                           self.unconstrained_cause_repertoire(purview, cut))

    def effect_info(self, mechanism, purview, cut=None):
        """Return the effect information for a mechanism over a purview."""
        return hamming_emd(self.effect_repertoire(mechanism, purview),
                           self.unconstrained_effect_repertoire(purview, cut))

    def cause_effect_info(self, mechanism, purview, cut=None):
        """Return the cause-effect information for a mechanism over a
        purview."""
        return min(self.cause_info(mechanism, purview, cut),
                   self.effect_info(mechanism, purview, cut))

    # MIP methods
    # =========================================================================
    # TODO test everything below here

    # TODO? something clever here so we don't do the full iteration
    @staticmethod
    def _mip_bipartition(mechanism, purview):
        # TODO? better not to build this whole list in memory
        purview_bipartitions = list(bipartition(purview))
        for denominators in (purview_bipartitions +
                             list(map(lambda x: x[::-1],
                                      purview_bipartitions))):
            for numerators in bipartition(mechanism):
                # For the MIP, we only consider the bipartitions in which each
                # node appears exactly once, e.g. for AB/ABC, (A/B) * (C/[]) is
                # valid but (AB/BC) * ([]/A) is not (since B appears in both
                # numerator and denominator), and exclude partitions whose
                # numerator and denominator are both empty.
                valid_partition = (
                    len(numerators[0]) + len(denominators[0]) > 0 and
                    len(numerators[1]) + len(denominators[1]) > 0)
                if valid_partition:
                    part0 = Part(mechanism=numerators[0],
                                 purview=denominators[0])
                    part1 = Part(mechanism=numerators[1],
                                 purview=denominators[1])
                    yield (part0, part1)
        return

    def find_mip(self, direction, mechanism, purview, cut=None):
        """Return the minimum information partition for the past or future.
        Where the ``partition`` attribute is a pair of objects, each with the
        following attributes:

        * ``mechanism``: list of nodes in the numerator of this part of the
            bipartition
        * ``purview``: list of nodes in the denominator of this part of the
            bipartition

        :param direction: Either ``'past'`` or ``'future'``.
        :type direction: ``str``
        :param mechanism: A list of nodes in the mechanism
        :type mechanism: ``[Node]``
        :param purview: A list of nodes in the purview
        :type mechanism: ``[Node]``
        :returns: The minimum information partition.
        """
        validate.direction(direction)
        repertoire = self._get_repertoire(direction)

        mip = None
        phi_min = float('inf')
        # Calculate the unpartitioned repertoire to compare against the
        # partitioned ones
        unpartitioned_repertoire = repertoire(mechanism, purview)

        # Loop over possible MIP bipartitions
        for part0, part1 in self._mip_bipartition(mechanism, purview):
            # Find the distance between the unpartitioned repertoire and
            # the product of the repertoires of the two parts, e.g.
            #   D( p(ABC/ABC) || p(AC/C) * p(B/AB) )
            partitioned_repertoire = (repertoire(part0.mechanism,
                                                 part0.purview) *
                                      repertoire(part1.mechanism,
                                                 part1.purview))
            phi = hamming_emd(unpartitioned_repertoire, partitioned_repertoire)
            # Return immediately if mechanism is reducible
            if phi < constants.EPSILON:
                return None
            # Update MIP if it's more minimal
            if (phi_min - phi) > constants.EPSILON:
                phi_min = phi
                mip = Mip(direction=direction,
                          partition=(part0, part1),
                          unpartitioned_repertoire=unpartitioned_repertoire,
                          partitioned_repertoire=partitioned_repertoire,
                          phi=phi)
        return mip

    def mip_past(self, mechanism, purview, cut=None):
        """Return the past minimum information partition.

        For a description of the MIP object that is returned, see
        :func:`find_mip`.
        """
        return self.find_mip('past', mechanism, purview, cut)

    def mip_future(self, mechanism, purview, cut=None):
        """Return the future minimum information partition.

        For a description of the MIP object that is returned, see
        :func:`find_mip`.
        """
        return self.find_mip('future', mechanism, purview, cut)

    def phi_mip_past(self, mechanism, purview, cut=None):
        """Return the |phi| value of the past minimum information partition.

        This is the distance between the unpartitioned cause repertoire and the
        MIP cause repertoire.
        """
        mip = self.mip_past(mechanism, purview, cut)
        if mip:
            return mip.phi
        else:
            return 0

    def phi_mip_future(self, mechanism, purview, cut=None):
        """Return the |phi| value of the future minimum information partition.

        This is the distance between the unpartitioned effect repertoire and
        the MIP cause repertoire.
        """
        mip = self.mip_future(mechanism, purview, cut)
        if mip:
            return mip.phi
        else:
            return 0

    def phi(self, mechanism, purview, cut=None):
        """Return the integrated information, "small |phi|"."""
        return min(self.phi_mip_past(mechanism, purview, cut),
                   self.phi_mip_future(mechanism, purview, cut))

    # Phi_max methods
    # =========================================================================

    # TODO update docs
    def _find_mice(self, direction, mechanism):
        """Return the maximally irreducible cause or effect for a mechanism.

        .. note:: Strictly speaking, the MICE is a pair of repertoires: the
            core cause repertoire and core effect repertoire of a mechanism,
            which are maximally different than the unconstrained cause/effect
            repertoires (*i.e.*, those that maximize |phi|). Here, we return
            only information corresponding to one direction, ``'past'`` or
            ``'future'``.

        :returns: An object with attributes ``purview`` and ``phi``, containing
            the core cause or effect purview and the |phi| value, respectively.
        """
        mip_max = None
        phi_max = float('-inf')
        maximal_purview = None
        maximal_repertoire = None
        # Loop over all possible purviews in this candidate set and find the
        # purview over which phi is maximal.
        for purview in powerset(self.nodes):
            mip = self.find_mip(direction, mechanism, purview)
            if mip:
                # Take the purview with higher phi, or if phi is equal, take
                # the larger one (exclusion principle)
                if mip.phi > phi_max or (phi_eq(mip.phi, phi_max) and
                                         len(purview) > len(maximal_purview)):
                    mip_max = mip
                    phi_max = mip.phi
                    maximal_purview = purview
                    maximal_repertoire = mip.unpartitioned_repertoire
        if phi_max == float('-inf'):
            phi_max = 0
        return Mice(direction=direction,
                    mechanism=mechanism,
                    purview=maximal_purview,
                    repertoire=maximal_repertoire,
                    mip=mip_max,
                    phi=phi_max)

    def core_cause(self, mechanism):
        """Returns the core cause repertoire of a mechanism."""
        return self._find_mice('past', mechanism)

    def core_effect(self, mechanism):
        """Returns the core effect repertoire of a mechanism."""
        return self._find_mice('future', mechanism)

    def phi_max(self, mechanism):
        """Return the |phi_max| of a mechanism."""
        return min(self.core_cause(mechanism).phi,
                   self.core_effect(mechanism).phi)

    # TODO! factor these out to compute.py
    # Big Phi methods
    # =========================================================================

    # TODO add `concept-space` section to the docs:
        # The first dimension corresponds to the direction, past or future; the
        # correspond to the subsystem's state space."""
    def null_concept(self, cut=None):
        """Return the null concept of this subsystem, a point in concept space
        identified with the unconstrained cause and effect repertoire of this
        subsystem.

        For information on the indices used in the returned array, see
        :ref:concept-space."""
        return Concept(
            mechanism=(),
            location=np.array([self.unconstrained_cause_repertoire(self.nodes, cut),
                               self.unconstrained_effect_repertoire(self.nodes, cut)]),
            phi=0,
            cause=None,
            effect=None)

    def concept(self, mechanism, cut=None):
        """Returns the concept specified by a mechanism"""
        past_mice = self.core_cause(mechanism)
        future_mice = self.core_cause(mechanism)
        phi = min(past_mice.phi, future_mice.phi)
        if phi <= 0:
            return None
        return Concept(
            mechanism=mechanism,
            location=np.array([
                self.expand_cause_repertoire(past_mice.mechanism,
                                             past_mice.purview,
                                             past_mice.repertoire,
                                             cut),
                self.expand_effect_repertoire(future_mice.mechanism,
                                              future_mice.purview,
                                              future_mice.repertoire,
                                              cut)]),
            phi=phi,
            cause=past_mice,
            effect=future_mice)

    def constellation(self, cut=None):
        """Return the conceptual structure of this subsystem."""
        return [concept for concept in [self.concept(mechanism) for mechanism
                                        in powerset(self.nodes)] if concept]

    @staticmethod
    def concept_distance(c1, c2):
        """Return the distance between two concepts in concept-space."""
        return sum([hamming_emd(c1.location[constants.PAST],
                                c2.location[constants.PAST]),
                    hamming_emd(c1.location[constants.FUTURE],
                                c2.location[constants.FUTURE])])

    def _constellation_distance_simple(self, C1, C2, null_concept):
        """Return the distance between two constellations in concept-space,
        assuming the only difference between them is that some concepts have
        disappeared."""
        # Make C1 refer to the bigger constellation
        if len(C2) > len(C1):
            C1, C2 = C2, C1
        destroyed = [c for c in C1 if c not in C2]
        return sum(c.phi * self.concept_distance(c, null_concept)
                   for c in destroyed)

    def _constellation_distance_emd(self, C1, C2, unique_C1,
                                    unique_C2, null_concept):
        """Return the distance between two constellations in concept-space,
        using the generalized EMD."""
        shared_concepts = [c for c in C1 if c in C2]
        # Construct null concept and list of all unique concepts.
        all_concepts = shared_concepts + unique_C1 + unique_C2 + [null_concept]
        # Construct the two phi distributions.
        d1, d2 = [[c.phi if c in constellation else 0 for c in all_concepts]
                  for constellation in (C1, C2)]
        # Calculate how much phi disappeared and assign it to the null concept
        # (the null concept is the last element in the distribution).
        residual = sum(d1) - sum(d2)
        if residual > 0:
            d2[-1] = residual
        if residual < 0:
            d1[-1] = residual
        # Generate the ground distance matrix.
        distance_matrix = np.array([
            [self.concept_distance(i, j) for i in all_concepts] for j in
            all_concepts])

        return emd(np.array(d1), np.array(d2), distance_matrix)

    def constellation_distance(self, C1, C2):
        """Return the distance between two constellations in concept-space."""
        null_concept = self.null_concept()
        concepts_only_in_C1 = [c for c in C1 if c not in C2]
        concepts_only_in_C2 = [c for c in C2 if c not in C1]
        # If the only difference in the constellations is that some concepts
        # disappeared, then we don't need to use the EMD
        if not concepts_only_in_C1 or not concepts_only_in_C2:
            return self._constellation_distance_simple(C1, C2, null_concept)
        else:
            return self._constellation_distance_emd(C1, C2,
                                                    concepts_only_in_C1,
                                                    concepts_only_in_C2,
                                                    null_concept)

    def conceptual_information(self):
        return constellation_distance(self.constellation, [])

    def big_mip(self):
        """Return the MIP for this subsystem."""
        # Calculate the unpartitioned constellation
        unpartitioned_constellation = self.constellation()
        # The first bipartition is the null cut, so skip it
        bipartitions = list(bipartition(self.nodes))[1:]
        # Loop over all partitions
        mip_candidates = Parallel(n_jobs=8)(
            delayed(_evaluate_cut)(self, partition,
                                   unpartitioned_constellation)
            for partition in bipartitions)
        # Return minimal MIP candidate
        return min(mip_candidates) if any(mip_candidates) else None



# TODO refactor this to somewhere sensible; it can't be a class method because
# those can't be pickled.
def _evaluate_cut(subsystem, cut, unpartitioned_constellation):
    # Cut connections from part 1 to part 2
    forward_constellation = subsystem.constellation(cut)
    forward_phi = subsystem.constellation_distance(
        unpartitioned_constellation, forward_constellation)
    # Cut connections from part 2 to part 1
    backward_constellation = subsystem.constellation(cut[::-1])
    backward_phi = subsystem.constellation_distance(
        unpartitioned_constellation, backward_constellation)

    print('cut:', cut, 'forward_phi', forward_phi,
            'backward_phi:', backward_phi, sep='\n')

    # Choose minimal unidirectional cut
    if forward_phi <= backward_phi:
        min_cut = cut
        min_constellation = forward_constellation
        phi = forward_phi
    else:
        min_cut = cut[::-1]
        min_constellation = backward_constellation
        phi = backward_phi

    if phi < constants.EPSILON:
        return None

    return(partition, forward_constellation, forward_phi)
