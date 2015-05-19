#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# macro.py
"""
Methods for coarse-graining systems to different levels of spatial analysis.
"""

import numpy as np
import itertools
import logging
from . import constants, validate, compute, convert
from .network import Network

# Create a logger for this module.
log = logging.getLogger(__name__)

# Load precomputed partition lists.
import os
_ROOT = os.path.abspath(os.path.dirname(__file__))
_NUM_PRECOMPUTED_PARTITION_LISTS = 10
_partition_lists = [
    np.load(os.path.join(_ROOT, 'data', 'partition_lists', str(i) + '.npy'))
    for i in range(_NUM_PRECOMPUTED_PARTITION_LISTS)
]


class MacroNetwork:
    """A coarse-grained network of nodes.

    See the 'macro' example in the documentation for more information.

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
    def __init__(self, macro_network, macro_phi, micro_network, micro_phi,
                 partition, grouping):
        self.network = macro_network
        self.phi = macro_phi
        self.micro_network = micro_network
        self.micro_phi = micro_phi
        self.partition = partition
        self.grouping = grouping
        self.emergence = self.phi - self.micro_phi


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
        raise ValueError('Partition lists not yet available for system with %d \
                         nodes or more' % (_NUM_PRECOMPUTED_PARTITION_LISTS))


def list_all_partitions(network):
    """Return a list of all possible coarse grains of a network.

    Args:
        network (Network): The physical system to act as the 'micro' level.

    Returns:
        partitions (``list(list))``): A list of possible partitions. Each
            element of the list is a list of micro-elements which correspong to
            macro-elements.
    """
    partitions = _partitions_list(network.size)
    if network.size > 0:
        partitions[-1] = [list(range(network.size))]
    return partitions


def list_all_groupings(partition):
    """Return a list of all possible groupings of states, for a particular
    coarse graining (partition) of a network.

    Args:
        network (Network): The physical system on the micro level.
        partitions (list(list)): The partition of micro-elements into macro
            elements.

    Returns:
        groupings (``list(list(list(list)))``): A list of all possible
            correspondences between micro-states and macro-states for the
            partition.
    """
    if not all([len(part) > 0 for part in partition]):
        raise ValueError('Each part of the partition must have at least one '
                         'element.')
    micro_state_groupings = [_partitions_list(len(part) + 1) if len(part) > 1
                             else [[[0], [1]]] for part in partition]
    return [list(grouping) for grouping in
            itertools.product(*micro_state_groupings) if
            np.all(np.array([len(element) < 3 for element in grouping]))]


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


def make_macro_network(network, mapping):
    """Create the macro-network for a given mapping from micro to macro-states.

    Returns ``None`` if the macro TPM does not satisfy the conditional
    independence assumption.

    Args:
        micro_tpm (nd.array): TPM of the micro-system.
        mapping (nd.array): Mapping from micro-states to macro-states.

    Returns:
        macro_network (``Network``): Network of the macro-system, or ``None``.
    """
    num_macro_nodes = int(np.log2(max(mapping) + 1))
    macro_tpm = make_macro_tpm(network.tpm, mapping)
    macro_current_state = convert.loli_index2state(
        mapping[convert.state2loli_index(network.current_state)].astype(int),
        num_macro_nodes)
    if validate.conditionally_independent(macro_tpm):
        return Network(macro_tpm, macro_current_state)
    else:
        return None


def emergence(network):
    """Check for emergence of a macro-system into a macro-system.

    Checks all possible partitions and groupings of the micro-system to find
    the spatial scale with maximum integrated information.

    Args:
        network (Network): The network of the micro-system under investigation.

    Returns:
        macro_network (``MacroNetwork``): The maximal coarse-graining of the
            micro-system.
    """
    micro_phi = compute.main_complex(network).phi
    partitions = list_all_partitions(network)
    max_phi = float('-inf')
    for partition in partitions:
        groupings = list_all_groupings(partition)
        for grouping in groupings:
            mapping = make_mapping(partition, grouping)
            macro_network = make_macro_network(network, mapping)
            if macro_network:
                main_complex = compute.main_complex(macro_network)
                if (main_complex.phi - max_phi) > constants.EPSILON:
                    max_phi = main_complex.phi
                    max_partition = partition
                    max_grouping = grouping
                    max_network = macro_network
    return MacroNetwork(macro_network=max_network,
                        macro_phi=max_phi,
                        micro_network=network,
                        micro_phi=micro_phi,
                        partition=max_partition,
                        grouping=max_grouping)
