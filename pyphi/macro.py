#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# macro.py

"""
Methods for coarse graining systems to different levels of
spatial analysis.
"""

import pyphi
import numpy as np
from pyphi.convert import loli_index2state as i2s
from pyphi.convert import state2loli_index as s2i
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
        >>> _partition_list(3)
        [[[0, 1], [2]], [[0, 2], [1]], [[0], [1, 2]], [[0], [1], [2]]]

    """
    if N < (_NUM_PRECOMPUTED_PARTITION_LISTS):
        return _partition_lists[N]
    else:
        raise ValueError('Partition lists not yet available for system with %d \
                         nodes or more' % (_NUM_PRECOMPUTED_PARTITION_LISTS))



def list_all_possible_grains(network):
    """ Return a list of all possible coarse grains of a network

    Args:
        network (Network): The physical system to act as the 'micro'
            level

    Returns: partitions (list(list)): A list of possible partitions.
            Each element of the list is a list of micro elements which
            correspong to macro elements

    """
    partitions = partitions_list(network.size)
    partitions[-1] = [[index for index in range(network.size)]]
    return partitions






