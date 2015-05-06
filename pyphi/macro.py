#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# macro.py

"""
Methods for coarse graining systems to different levels of
spatial analysis.
"""

import pyphi
import numpy as np
from pyphi.convert import loli_index2state
from pyphi.convert import state2loli_index
import itertools
import logging
from functools import namedtuple

# Create a logger for this  module
log = logging.getLogger(__name__)

# Load precomputed partition lists
import os
_ROOT = os.path.abspath(os.path.dirname(__file__))
_NUM_PRECOMPUTED_PARTITION_LISTS = 10
_partition_lists = [
   np.load(os.path.join(_ROOT, 'data', 'partition_lists', str(i) + '.npy'))
   for i in range(_NUM_PRECOMPUTED_PARTITION_LISTS)
]

_MacroInfo = namedtuple("MacroInfo", ["micro_network", "micro_phi", "partition", "grouping", "macro_network", "macro_phi", "emergence"])

def _partitions_list(N):
    """ Return a list of partitions of the |N| binary nodes.

    Args:
        N (int): the number of nodes under consideration

    Returns:
        ``list`` -- A list of lists, where each inner list is the set of micro
        elements corresponding to a macro element.

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
    """ Return a list of all possible coarse grains of a network

    Args:
        network (Network): The physical system to act as the 'micro'
            level

    Returns: partitions (list(list)): A list of possible partitions.
            Each element of the list is a list of micro elements which
            correspong to macro elements

    """
    partitions = _partitions_list(network.size)
    partitions[-1] = [[index for index in range(network.size)]]
    return partitions


def list_all_groupings(partition):
    """ Return a list of all possible groupings of states, for a
        particular coarse graining (partition) of a network

    Args:
        network (Network): The physical system to act as the 'micro'
            level

        partitions (list(list)): The partition of micro elements into
            macro elements

    Returns:
        groupings (list(list(list(list)))): A list of all possible correspondences
            between micro states and macro states for the partition.

    """
    micro_state_groupings = [_partitions_list(len(part)+1) if len(part) > 1
                             else [[[0], [1]]] for part in partition]
    def is_binary(part):
        return np.all(np.array([len(element)  < 3 for element in part]))
    macro_states = [list(tuple) for tuple in itertools.product(*micro_state_groupings)] # Used to be full_system_partition
    return list(filter(is_binary, macro_states))

def make_mapping(micro_tpm, partition, grouping):
    """ Return a mapping from micro state to the
    macro states based on the partition of elements
    and grouping of states

    Args:
        micro_tpm (nd.array): The state-by-node tpm of the micro
            level

        partition (list(list)): A partition of micro elements into
            macro elements

        grouping (list(list(list))): For each macro element, a list of
            micro states which set it to ON or OFF.

    Returns:
        mapping (nd.array): A mapping from micro states to macro states.

    """
    number_of_macro_nodes = len(grouping)
    number_of_micro_nodes = sum([len(part) for part in partition])
    number_of_micro_states = 2**number_of_micro_nodes
    micro_states = np.array([tuple(map(int, bin(state_index)[2:].zfill(number_of_micro_nodes)[::-1]))
                             for state_index in range(number_of_micro_states)]).astype(float)
    mapping = np.zeros(number_of_micro_states)
    macro_states = list(itertools.product(*grouping[::-1]))
    macro_states = [list(tuple) for tuple in macro_states]
    for macro_index, macro_state in enumerate(macro_states):
        for micro_index, micro_state in enumerate(micro_states):
            micro_sum = [sum([micro_state[node] for node in partition[x]]) for x in range(number_of_macro_nodes)]
            if all([micro_sum[x] in macro_state[::-1][x] for x in range(number_of_macro_nodes)]):
                mapping[micro_index] = macro_index
    return mapping

def make_macro_tpm(micro_tpm, mapping):
    number_of_macro_states = int(max(mapping)+1)
    tpm = pyphi.convert.state_by_node2state_by_state(micro_tpm)
    number_of_micro_states = len(tpm)
    macro_tpm = np.zeros((number_of_macro_states, number_of_macro_states))
    for past_state_index in range(number_of_micro_states):
        for current_state_index in range(number_of_micro_states):
            macro_tpm[mapping[past_state_index],
                      mapping[current_state_index]] = (macro_tpm[mapping[past_state_index],
                                                                 mapping[current_state_index]]
                                                       + tpm[past_state_index, current_state_index])
    return np.array([list(row) if sum(row) == 0 else list(row/sum(row)) for row in macro_tpm])

def make_macro_network(network, mapping):
    number_of_macro_nodes = int(np.log2(max(mapping)+1))
    macro_tpm = make_macro_tpm(network.tpm, mapping)
    macro_current_state = loli_index2state(mapping[state2loli_index(network.current_state)].astype(int), number_of_macro_nodes)
    macro_past_state = loli_index2state(mapping[state2loli_index(network.past_state)].astype(int), number_of_macro_nodes)
    if pyphi.validate.conditionally_independent(macro_tpm):
        return pyphi.Network(macro_tpm, macro_current_state, macro_past_state)
    else:
        return None

def emergence(network):
    micro_phi = pyphi.compute.main_complex(network).phi
    partitions = list_all_partitions(network)
    max_phi = 0
    for partition in partitions:
        groupings = list_all_groupings(partition)
        for grouping in groupings:
            mapping = make_mapping(network.tpm, partition, grouping)
            macro_network = make_macro_network(network, mapping)
            if macro_network:
                main_complex = pyphi.compute.main_complex(macro_network)
                phi = main_complex.phi
                if phi > max_phi:
                    max_phi = phi
                    max_partition = partition
                    max_grouping = grouping
                    max_network = macro_network
    return _MacroInfo(micro_network = network,
                      micro_phi = micro_phi,
                      partition = max_partition,
                      grouping = max_grouping,
                      macro_network = max_network,
                      macro_phi = max_phi,
                      emergence = max_phi - micro_phi)















