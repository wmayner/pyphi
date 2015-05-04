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



def list_possible_grains(network):
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


def list_possible_groupings(network, partition):
    """ Return a list of all possible groupings of states, for a
        particular coarse graining (partition) of a network

    Args:
        network (Network): The physical system to act as the 'micro'
            level

        partitions (list(list)): The partition of micro elements into
            macro elements

    Returns:
        groupings (list(list)): A list of all possible correspondences
            between micro states and macro states.

    """
    num_macro_elements = len(partition)
    micro_groups = [[list(loli_index2state(index, len(part)))[::-1]
                             for index in range(2**len(part))]
                            for part in partition] # Used to be possible_elemental_states
    micro_state_groupings = [_partition_lists(len(part)+1) if len(part) > 1
                             else [[0], [1]] for part in partition]
    def is_binary(part):
        return np.all(np.array([len(element) for element in part]))

    micro_state_groupings = [groupings(state) for state in micro_element_states] # Used to be possible_partitions_element
    macro_states = [list(tuple) for tuple in itertools.product(*micro_state_groups)] # Used to be full_system_partition
    return list(filter(is_binary, macro_states))







