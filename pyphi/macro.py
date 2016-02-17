#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# macro.py

"""
Methods for coarse-graining systems to different levels of spatial analysis.
"""

import itertools
import logging
import os

import numpy as np

from . import compute, config, constants, convert, utils, validate
from .constants import DIRECTIONS, PAST, FUTURE
from .network import Network
from .node import Node
from .subsystem import Subsystem

# Create a logger for this module.
log = logging.getLogger(__name__)

# Load precomputed partition lists.
_NUM_PRECOMPUTED_PARTITION_LISTS = 10
_partition_lists = utils.load_data('partition_lists',
                                   _NUM_PRECOMPUTED_PARTITION_LISTS)


class MacroSubsystem(Subsystem):
    """A subclass of |Subsystem| implementing macro computations."""

    def __init__(self, network, state, node_indices, cut=None,
                 mice_cache=None, time_scale=1, hidden_indices=None,
                 output_grouping=None, state_grouping=None):
        super().__init__(network, state, node_indices, cut, mice_cache)

        # TODO: move back to property
        self._size = len(self.node_indices)

        # HACk: Remember original values to use in `apply_cut`
        self._network_state = state
        self._node_indices = node_indices
        self._hidden_indices = hidden_indices
        self._output_grouping = output_grouping
        self._state_grouping = state_grouping

        self.independent = True

        # Indices internal to the micro subsystem
        self.internal_indices = internal_indices = node_indices

        # Don't squeeze out the final dim ension (which contains the
        # probability) for networks of size one
        if self.network.size > 1:
            self.tpm = np.squeeze(self.tpm)[..., self.internal_indices]

        # Re-index the subsystem nodes with the external nodes removed
        self.micro_size = len(self.internal_indices)
        self.micro_indices = tuple(range(self.micro_size))

        # A variable to tell if a system is a pure micro without blackboxing or
        # coarse-grain.
        self.micro = (output_grouping is None and hidden_indices is None)

        # Get the subsystem's connectivity matrix. This is the network's
        # connectivity matrix, but with the cut applied, and with all
        # connections to/from external nodes severed.
        if self.internal_indices:
            self.micro_connectivity_matrix = utils.apply_cut(
                cut, network.connectivity_matrix)[np.ix_(self.internal_indices,
                                                         self.internal_indices)]
            self.connectivity_matrix = self.micro_connectivity_matrix
        #else:
        #    self.micro_connectivity_matrix = np.array([[]])
        #    self.connectivity_matrix = self.micro_connectivity_matrix

        # Calculate the nodes for all internal indices
        # ============================================
        self.nodes = tuple(Node(self, i, indices=self.micro_indices)
                           for i in self.micro_indices)
        # Re-calcuate the tpm based on the results of the cut
        self.tpm = np.rollaxis(
            np.array([
                node.expand_tpm(self.micro_indices) for node in self.nodes
            ]), 0, self.micro_size + 1)

        # Create the TPM and CM for the defined time scale
        # ================================================
        validate.time_scale(time_scale)
        self.time_scale = time_scale

        # TODO(billy) This is a blackboxed time. Coarse grain time not yet implemented.
        if internal_indices and time_scale > 1:
            self.tpm = utils.run_tpm(self.tpm, time_scale)
            self.connectivity_matrix = utils.run_cm(
                self.micro_connectivity_matrix, time_scale)

        # Generate the TPM and CM after blackboxing
        # =========================================
        # Set the elements for blackboxing
        if hidden_indices is None:
            hidden_indices = ()

        # Using network-based indexing.
        self.micro_hidden_indices = hidden_indices
        # Using indexing of subsystem internal elements.
        self.hidden_indices = tuple(
            i for i in self.micro_indices
            if self.internal_indices[i] in hidden_indices)
        # Blackbox output indices using the subsystem's internal indexing.
        self.output_indices = tuple(
            i for i in self.micro_indices
            if self.internal_indices[i] not in hidden_indices)
        # Koan of the Black Box:
        #   "Blackbox indices are the blackbox indices using the blackbox
        #    indexing."
        #        - The Blackbox Master
        self.blackbox_indices = tuple(range(len(self.output_indices)))

        # The TPM conditioned on the current value of the hidden nodes.
        if self.hidden_indices:
            self.tpm = utils.condition_tpm(self.tpm,
                                           self.hidden_indices,
                                           self.proper_state)
            self.tpm = np.squeeze(self.tpm)
            self.tpm = self.tpm[..., self.output_indices]
            self.connectivity_matrix = np.array([
                [1 if np.sum(self.connectivity_matrix[
                    np.ix_([self.output_indices[cause_index]],
                           [self.output_indices[effect_index]])])
                    > 0 else 0
                    for effect_index in range(len(self.output_indices))]
                for cause_index in range(len(self.output_indices))])
            self._state = tuple(self.proper_state[index]
                               for index in self.output_indices)

        # Generate the TPM and CM after coarse-graining
        # =============================================
        # Set the elements for coarse-graining
        if output_grouping is not None:
            # TODO(billy) validate.macro(output_grouping, state_grouping)
            self.micro_output_grouping = output_grouping
            self.output_grouping = tuple(
                tuple(i for i in self.blackbox_indices if
                      self.internal_indices[self.output_indices[i]] in group)
                for group in output_grouping)
            self.state_grouping = state_grouping
            self.mapping = make_mapping(self.output_grouping,
                                              self.state_grouping)
            self._size = len(self.output_grouping)
            self.subsystem_indices = tuple(range(self._size))
            state = np.array(self.state)
            self._state = tuple(0 if sum(state[list(self.output_grouping[0])])
                               in state_grouping[i][0] else 1 for i in self.subsystem_indices)
        else:
            self.micro_output_grouping = None
            self.output_grouping = ()
            self.state_grouping = None
            self.mapping = None
            self._size = len(self.output_indices)
            self.subsystem_indices = tuple(range(self._size))

        # Coarse-grain the remaining nodes into the appropriate groups
        if output_grouping:
            self.tpm = make_macro_tpm(self.tpm, self.mapping)
            if cut is None:
                self.independent = validate.conditionally_independent(self.tpm)
            self.tpm = convert.state_by_state2state_by_node(self.tpm)
            self.connectivity_matrix = np.array([
                [np.max(self.connectivity_matrix[
                    np.ix_(self.output_grouping[row],
                           self.output_grouping[col])])
                 for col in range(self._size)]
                for row in range(self._size)])

        if self.independent:
            self.nodes = tuple(Node(self, i, indices=self.subsystem_indices)
                               for i in self.subsystem_indices)
        else:
            self.nodes = ()

        # Hash the final subsystem and nodes
        # Only compute hash once.
        self._hash = hash((self.internal_indices,
                           self.hidden_indices,
                           self.output_grouping,
                           self.state_grouping,
                           self.cut,
                           self.network))
        for node in self.nodes:
            node._hash = hash((node.index, node.subsystem))

        # TODO: combine subsystem_indices and node_indices
        self.node_indices = self.subsystem_indices

        # The nodes represented in computed repertoires.
        self._dist_indices = self.subsystem_indices

        # Nodes to cut for big-phi computations. For macro computations the cut
        # is applied to the underlying micro network.
        self._cut_indices = self._node_indices

        validate.subsystem(self)

    @property
    def size(self):
        """Override `Subsystem.size`."""
        return self._size

    def apply_cut(self, cut):
        """Return a cut version of this `MacroSubsystem`

        Args:
            cut (Cut): The cut to apply to this `MacroSubsystem`.

        Returns:
            subsystem (MacroSubsystem)
        """
        return MacroSubsystem(self.network, self._network_state,
                              self._node_indices, cut=cut,
                              time_scale=self.time_scale,
                              hidden_indices=self._hidden_indices,
                              output_grouping=self._output_grouping,
                              state_grouping=self._state_grouping)
                              # TODO: is the MICE cache reusable?
                              # mice_cache=self._mice_cache)

    def _potential_purviews(self, direction, mechanism, purviews=False):
        """Override Subsystem implementation which depends on Network-level
        indices."""
        all_purviews = utils.powerset(self.node_indices)

        def reducible(purview):
            # Returns True if purview is trivially reducible.
            if direction == DIRECTIONS[PAST]:
                _from, to = purview, mechanism
            elif direction == DIRECTIONS[FUTURE]:
                _from, to = mechanism, purview
            return utils.block_reducible(self.connectivity_matrix, _from, to)

        return [purview for purview in all_purviews if not reducible(purview)]

    def __repr__(self):
        return "MacroSubsystem(" + repr(self.nodes) + ")"

    def __str__(self):
        return repr(self)


class MacroNetwork:
    """A coarse-grained network of nodes.

    See the :ref:`macro-micro` example in the documentation for more
    information.

    Attributes:
        network (Network): The network object of the macro-system.
        phi (float): The |big_phi| of the network's main complex.
        micro_network (Network): The network object of the corresponding micro
            system.
        micro_phi (float): The |big_phi| of the main complex of the
            corresponding micro-system.
        partition (list): The partition which defines macro-elements in terms
            of micro-elements.
        grouping (list(list)): The correspondence between micro-states and
            macro-states.
        emergence (float): The difference between the |big_phi| of the macro-
            and the micro-system.
    """
    def __init__(self, network, system, macro_phi, micro_phi,
                 partition, grouping):
        self.network = network
        self.system = system
        self.phi = macro_phi
        self.micro_phi = micro_phi
        self.partition = partition
        self.grouping = grouping
        self.emergence = round(self.phi - self.micro_phi, config.PRECISION)


def _partitions_list(N):
    """Return a list of partitions of the |N| binary nodes.

    Args:
        N (int): The number of nodes under consideration.

    Returns:
        partition_list (``list``): A list of lists, where each inner list is
        the set of micro-elements corresponding to a macro-element.

    Example:
        >>> from pyphi.macro import _partitions_list
        >>> _partitions_list(3)
        [[[0, 1], [2]], [[0, 2], [1]], [[0], [1, 2]], [[0], [1], [2]]]
    """
    if N < (_NUM_PRECOMPUTED_PARTITION_LISTS):
        return list(_partition_lists[N])
    else:
        raise ValueError(
            'Partition lists not yet available for system with {} '
            'nodes or more'.format(_NUM_PRECOMPUTED_PARTITION_LISTS))


def list_all_partitions(size):
    """Return a list of all possible coarse grains of a network.

    Args:
        network (Network): The physical system to act as the 'micro' level.

    Returns:
        partitions (``list(list))``): A list of possible partitions. Each
            element of the list is a list of micro-elements which correspong to
            macro-elements.
    """
    partitions = _partitions_list(size)
    if size > 0:
        partitions[-1] = [list(range(size))]
    return tuple(tuple(tuple(part)
                       for part in partition)
                 for partition in partitions)


def list_all_groupings(partition):
    """Return all possible groupings of states for a particular coarse graining
    (partition) of a network.

    Args:
        network (Network): The physical system on the micro level.
        partitions (list(list)): The partition of micro-elements into macro
            elements.

    Returns:
        groupings (``list(list(list(list)))``): A list of all possible
            correspondences between micro-states and macro-states for the
            partition.
    """
    if not all(len(part) > 0 for part in partition):
        raise ValueError('Each part of the partition must have at least one '
                         'element.')
    micro_state_groupings = [_partitions_list(len(part) + 1) if len(part) > 1
                             else [[[0], [1]]] for part in partition]
    groupings = [list(grouping) for grouping in
                 itertools.product(*micro_state_groupings) if
                 np.all(np.array([len(element) < 3 for element in grouping]))]
    return tuple(tuple(tuple(tuple(state) for state in states)
                       for states in group)
                 for group in groupings)


def make_macro_tpm(micro_tpm, mapping):
    """Create the macro TPM for a given mapping from micro to macro-states.

    Args:
        micro_tpm (nd.array): The TPM of the micro-system.

        mapping (nd.array): A mapping from micro-states to macro-states.

    Returns:
        macro_tpm (``nd.array``): The TPM of the macro-system.
    """
    # Validate the TPM
    validate.tpm(micro_tpm)
    if (micro_tpm.ndim > 2) or (not micro_tpm.shape[0] == micro_tpm.shape[1]):
        micro_tpm = convert.state_by_node2state_by_state(micro_tpm)
    num_macro_states = max(mapping) + 1
    num_micro_states = len(micro_tpm)
    macro_tpm = np.zeros((num_macro_states, num_macro_states))
    # For every possible micro-state transition, get the corresponding past and
    # current macro-state using the mapping and add that probability to the
    # state-by-state macro TPM.
    micro_state_transitions = itertools.product(range(num_micro_states),
                                                range(num_micro_states))
    for past_state_index, current_state_index in micro_state_transitions:
        macro_tpm[mapping[past_state_index],
                  mapping[current_state_index]] += \
            micro_tpm[past_state_index, current_state_index]
    # Because we're going from a bigger TPM to a smaller TPM, we have to
    # re-normalize each row.
    return np.array([list(row) if sum(row) == 0 else list(row / sum(row))
                     for row in macro_tpm])


def make_mapping(partition, grouping):
    """Return a mapping from micro-state to the macro-states based on the
    partition of elements and grouping of states.

    Args:
        partition (list(list)): A partition of micro-elements into macro
            elements.
        grouping (list(list(list))): For each macro-element, a list of micro
            states which set it to ON or OFF.

    Returns:
        mapping (``nd.array``): A mapping from micro-states to macro-states.
    """
    num_macro_nodes = len(grouping)
    num_micro_nodes = sum([len(part) for part in partition])
    num_micro_states = 2**num_micro_nodes
    micro_states = [convert.loli_index2state(micro_state_index,
                                             num_micro_nodes)
                    for micro_state_index in range(num_micro_states)]
    mapping = np.zeros(num_micro_states)
    # For every micro-state, find the corresponding macro-state and add it to
    # the mapping.
    for micro_state_index, micro_state in enumerate(micro_states):
        # Sum the number of micro-elements that are ON for each macro-element.
        micro_sum = [sum([micro_state[node] for node in partition[i]])
                     for i in range(num_macro_nodes)]
        # Check if the number of micro-elements that are ON corresponds to the
        # macro-element being ON or OFF.
        macro_state = [0 if micro_sum[i] in grouping[i][0] else 1
                       for i in range(num_macro_nodes)]
        # Record the mapping.
        mapping[micro_state_index] = convert.state2loli_index(macro_state)
    return mapping


def coarse_grain(network, state, internal_indices):
    index_partitions = list_all_partitions(len(internal_indices))
    partitions = tuple(tuple(tuple(internal_indices[i] for i in part)
                             for part in partition)
                       for partition in index_partitions)
    max_phi = float('-inf')
    max_partition = ()
    max_grouping = ()
    for partition in partitions:
        groupings = list_all_groupings(partition)
        for grouping in groupings:
            subsystem = MacroSubsystem(network, state, internal_indices,
                                       output_grouping=partition,
                                       state_grouping=grouping)
            phi = compute.big_phi(subsystem)
            if (phi - max_phi) > constants.EPSILON:
                max_phi = phi
                max_partition = partition
                max_grouping = grouping
    return (max_phi, max_partition, max_grouping)


def emergence(network, state):
    """Check for emergence of a macro-system into a macro-system.

    Checks all possible partitions and groupings of the micro-system to find
    the spatial scale with maximum integrated information.

    Args:
        network (Network): The network of the micro-system under investigation.

    Returns:
        macro_network (``MacroNetwork``): The maximal coarse-graining of the
            micro-system.
    """
    micro_phi = compute.main_complex(network, state).phi
    systems = utils.powerset(network.node_indices)
    max_phi = float('-inf')
    for system in systems:
        (phi, partition, grouping) = coarse_grain(network, state, system)
        if (phi - max_phi) > constants.EPSILON:
            max_phi = phi
            max_partition = partition
            max_grouping = grouping
            max_system = system
    return MacroNetwork(network=network,
                        system=max_system,
                        macro_phi=max_phi,
                        micro_phi=micro_phi,
                        partition=max_partition,
                        grouping=max_grouping)


def phi_by_grain(network, state):
    list_of_phi = []
    systems = utils.powerset(network.node_indices)
    for system in systems:
        micro_subsystem = Subsystem(network, state, system)
        mip = compute.big_mip(micro_subsystem)
        list_of_phi.append([len(micro_subsystem), mip.phi])
        index_partitions = list_all_partitions(len(system))
        partitions = tuple(tuple(tuple(system[i] for i in part)
                                 for part in partition)
                           for partition in index_partitions)
        for partition in partitions:
            groupings = list_all_groupings(partition)
            for grouping in groupings:
                subsystem = MacroSubsystem(network, state, system,
                                           output_grouping=partition,
                                           state_grouping=grouping)
                phi = compute.big_phi(subsystem)
                list_of_phi.append([len(subsystem), phi, system,
                                    partition, grouping])
    return list_of_phi


# TODO write tests
# TODO? give example of doing it for a bunch of coarse-grains in docstring
# (make all groupings and partitions, make_network for each of them, etc.)
def effective_info(network):
    """Return the effective information of the given network.

    This is equivalent to the average of the
    :func:`~pyphi.subsystem.Subsystem.effect_info` (with the entire network as
    the mechanism and purview) over all posisble states of the network. It can
    be interpreted as the “noise in the network's TPM,” weighted by the size of
    its state space.

    .. warning::

        If ``config.VALIDATE_SUBSYSTEM_STATES`` is enabled, then unreachable
        states are omitted from the average.

    .. note::

        For details, see:

        Hoel, Erik P., Larissa Albantakis, and Giulio Tononi.
        “Quantifying causal emergence shows that macro can beat micro.”
        Proceedings of the
        National Academy of Sciences 110.49 (2013): 19790-19795.

        Available online: `doi: 10.1073/pnas.1314922110
        <http://www.pnas.org/content/110/49/19790.abstract>`_.
    """
    # TODO? move to utils
    states = itertools.product(*((0, 1),)*network.size)
    subsystems = []
    for state in states:
        try:
            subsystems.append(Subsystem(network, state, network.node_indices))
        except validate.StateUnreachableError:
            continue
    return np.array([
        subsystem.effect_info(network.node_indices, network.node_indices)
        for subsystem in subsystems
    ]).mean()
