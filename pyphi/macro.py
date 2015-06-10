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

from . import compute, constants, convert, utils, validate
from .network import Network
from .subsystem import Subsystem

# Create a logger for this module.
log = logging.getLogger(__name__)

# Load precomputed partition lists.
_NUM_PRECOMPUTED_PARTITION_LISTS = 10
_partition_lists = utils.load_data('partition_lists',
                                   _NUM_PRECOMPUTED_PARTITION_LISTS)


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
            subsystem = Subsystem(network, state, internal_indices,
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
                subsystem = Subsystem(network, state, system,
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
