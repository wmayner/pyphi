#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# macro.py

"""
Methods for coarse-graining systems to different levels of spatial analysis.
"""

from collections import namedtuple
import itertools
import logging
import os

import numpy as np

from . import compute, config, constants, convert, utils, validate
from .network import irreducible_purviews
from .node import Node
from .subsystem import Subsystem

# Create a logger for this module.
log = logging.getLogger(__name__)

# Load precomputed partition lists.
_NUM_PRECOMPUTED_PARTITION_LISTS = 10
_partition_lists = utils.load_data('partition_lists',
                                   _NUM_PRECOMPUTED_PARTITION_LISTS)


class ConditionallyDependentError(ValueError):
    pass


def reindex(indices):
    """Generate a new set of node indices, the size of indices."""
    return tuple(range(len(indices)))


class MacroSubsystem(Subsystem):
    """A subclass of |Subsystem| implementing macro computations.

    We maintain the following invariant after each macro update:
    `tpm.shape == [2] * len(state) + [len(state)]`.
     """

    def __init__(self, network, state, node_indices, cut=None,
                 mice_cache=None, time_scale=1, hidden_indices=None,
                 coarse_grain=None):
        super().__init__(network, state, node_indices, cut, mice_cache)

        # Store original arguments to use in `apply_cut`
        self._network_state = state
        self._node_indices = node_indices
        self._time_scale = time_scale
        self._hidden_indices = hidden_indices
        self._coarse_grain = coarse_grain

        # Indices internal to the micro subsystem
        internal_indices = node_indices

        # Compute the TPM and Nodes for the internal indices
        # ==================================================
        # Don't squeeze out the final dimension (which contains the
        # probability) for networks of size one
        if self.network.size > 1:
            self.tpm = np.squeeze(self.tpm)[..., internal_indices]

        # Re-index the subsystem nodes with the external nodes removed
        self.node_indices = reindex(internal_indices)

        self.nodes = tuple(Node(self, i, indices=self.node_indices)
                           for i in self.node_indices)
        # Re-calcuate the tpm based on the results of the cut
        self.tpm = np.rollaxis(
            np.array([
                node.expand_tpm(self.node_indices) for node in self.nodes
            ]), 0, len(self.node_indices) + 1)

        # The connectivity matrix is the network's connectivity matrix, with
        # cut applied, with all connections to/from external nodes severed,
        # shrunk to the size of the internal nodes.
        self.connectivity_matrix = self.connectivity_matrix[np.ix_(
            internal_indices, internal_indices)]

        self._state = tuple(self.state[i] for i in internal_indices)

        # Blackbox over time
        # ==================
        if time_scale != 1:
            validate.time_scale(time_scale)
            self.tpm, self.cm = self._blackbox_time(time_scale)

        # Blackbox in space
        # =================
        if hidden_indices is not None:
            # Compute hidden and output indices from node_indices, which are
            # now reindexed from 0..n
            hidden_indices = tuple(
                i for i in self.node_indices
                if internal_indices[i] in hidden_indices)
            output_indices = tuple(set(self.node_indices) -
                                   set(hidden_indices))

            self.tpm, self.cm, self.node_indices, self._state = (
                self._blackbox_space(hidden_indices, output_indices))

        # Coarse-grain in space
        # =====================
        if coarse_grain is not None:
            validate.coarse_grain(coarse_grain)
            # Reindex the coarse graining
            coarse_grain = coarse_grain.reindex()
            self.tpm, self.cm, self.node_indices, self._state = (
                self._coarsegrain_space(coarse_grain))

        self.nodes = tuple(Node(self, i, indices=self.node_indices)
                           for i in self.node_indices)

        # Hash the final subsystem - only compute hash once.
        self._hash = hash((self.network,
                           self.cut,
                           self._network_state,
                           self._node_indices,
                           self._hidden_indices,
                           self._coarse_grain))

        # The nodes represented in computed repertoires.
        self._dist_indices = self.node_indices

        # Nodes to cut for big-phi computations. For macro computations the cut
        # is applied to the underlying micro network.
        self._cut_indices = self._node_indices

        validate.subsystem(self)

    def _blackbox_time(self, time_scale):
        """Black box the CM and TPM over the given time_scale.

        TODO(billy): This is a blackboxed time. Coarse grain time is not yet
        implemented.
        """
        tpm = utils.run_tpm(self.tpm, time_scale)
        cm = utils.run_cm(self.connectivity_matrix, time_scale)

        return (tpm, cm)

    def _blackbox_space(self, hidden_indices, output_indices):
        """Blackbox the TPM and CM in space.

        Conditions the TPM on the current value of the hidden nodes. The CM is
        set to universal connectivity.
        TODO: ^^ change this.

        This shrinks the size of the TPM by the number of hidden indices; now
        there is only `len(output_indices)` dimensions in the TPM and in the
        state of the subsystem.
        """
        # TODO: validate conditional independence?
        tpm = utils.condition_tpm(self.tpm, hidden_indices, self.state)
        tpm = np.squeeze(tpm)[..., output_indices]

        # Universal connectivity, for now.
        n = len(output_indices)
        cm = np.ones((n, n))

        state = tuple(self.state[index] for index in output_indices)
        node_indices = reindex(output_indices)

        return (tpm, cm, node_indices, state)

    def _coarsegrain_space(self, coarse_grain):
        """Spatially coarse-grain the TPM and CM."""

        # Coarse-grain the remaining nodes into the appropriate groups
        tpm = coarse_grain.make_macro_tpm(self.tpm)
        if not self.is_cut():
            if not validate.conditionally_independent(tpm):
                raise ConditionallyDependentError
        tpm = convert.state_by_state2state_by_node(tpm)

        node_indices = coarse_grain.macro_indices
        state = coarse_grain.macro_state(self.state)

        # Universal connectivity, for now.
        n = len(node_indices)
        cm = np.ones((n, n))

        return (tpm, cm, node_indices, state)

    @property
    def is_micro(self):
        """True if the system is pure micro without blackboxing of coarse-
        graining."""
        # TODO: do we need this?
        return self._coarse_grain is None and self._hidden_indices is None

    def apply_cut(self, cut):
        """Return a cut version of this `MacroSubsystem`

        Args:
            cut (Cut): The cut to apply to this `MacroSubsystem`.

        Returns:
            subsystem (MacroSubsystem)
        """
        return MacroSubsystem(self.network, self._network_state,
                              self._node_indices, cut=cut,
                              time_scale=self._time_scale,
                              hidden_indices=self._hidden_indices,
                              coarse_grain=self._coarse_grain)
                              # TODO: is the MICE cache reusable?
                              # mice_cache=self._mice_cache)

    def _potential_purviews(self, direction, mechanism, purviews=False):
        """Override Subsystem implementation using Network-level indices."""
        all_purviews = utils.powerset(self.node_indices)
        return irreducible_purviews(self.connectivity_matrix,
                                    direction, mechanism, all_purviews)

    def __repr__(self):
        return "MacroSubsystem(" + repr(self.nodes) + ")"

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        """Two macro systems are equal if the underlying |Subsystems| are equal
        and all macro attributes are equal.

        TODO: handle cases where a MacroSubsystem is identical to a micro
        Subsystem, e.g. the macro has no timescale, hidden indices, etc.
        """
        if type(self) != type(other):
            return False

        return (super().__eq__(other) and
                self._time_scale == other._time_scale and
                self._hidden_indices == other._hidden_indices and
                self._coarse_grain == other.coarse_grain)

    def __hash__(self):
        return self._hash


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


class CoarseGrain(namedtuple('CoarseGrain',
                             ['output_grouping', 'state_grouping'])):
    """Represents a coarse graining of a collection of nodes."""

    # TODO: validate grouping size

    def make_mapping(self):
        # TODO: move `make_mapping` function to here entirely
        return make_mapping(self.output_grouping, self.state_grouping)

    def make_macro_tpm(self, tpm):
        return make_macro_tpm(tpm, self.make_mapping())

    @property
    def micro_indices(self):
        """Indices of micro elements represented in this coarse-graining."""
        return tuple({index for group in self.output_grouping
                      for index in group})

    @property
    def macro_indices(self):
        """Indices of macro elements of this coarse-graining."""
        return tuple(range(len(self.output_grouping)))

    def reindex(self):
        """Re-index this coarse graining to use squeezed indices.

        The output grouping is translated to use indices `0..n`, where `n` is
        the number of micro indices in the coarse-graining. Re-indexing does
        not effect the state grouping, which is already index-independent.

        Returns:
            CoarseGrain: A new CoarseGrain object, indexed from `0..n`

        Example:
            >>> output_grouping = ((1, 2),)
            >>> state_grouping = (((0,), (1, 2)),)
            >>> coarse_grain = CoarseGrain(output_grouping, state_grouping)
            >>> coarse_grain.reindex()
            CoarseGrain(output_grouping=((0, 1),), state_grouping=(((0,), (1, 2)),))
        """
        _map = dict(zip(self.micro_indices, reindex(self.micro_indices)))
        output_grouping = tuple(
            tuple(_map[index] for index in group)
            for group in self.output_grouping
        )
        return CoarseGrain(output_grouping, self.state_grouping)

    def macro_state(self, micro_state):
        """Translate a micro state to a macro state

        .. warning::

            This will return incorrect results if this CoarseGrain has been
            re-indexed unless the `micro_state` has also been re-indexed
            (shrunk to `len(self.micro_indices)`, containing only the state of
            `self.micro_indices`.)

        Args:
            micro_state (tuple(int)): The state of the micro system.

        Returns:
            tuple(int): The state of the macro system, translated as specified
                by this coarse-graining.
        """
        micro_state = np.array(micro_state)
        return tuple(0 if sum(micro_state[list(self.output_grouping[i])])
                     in self.state_grouping[i][0] else 1
                     for i in self.macro_indices)


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


def list_all_partitions(indices):
    """Return a list of all possible coarse grains of a network.

    Args:
        indices (tuple(int)): The micro indices to partition.

    Returns:
        tuple(tuple): A tuple of possible partitions. Each element of the tuple
            is a tuple of micro-elements which correspond to macro-elements.
    """
    n = len(indices)
    partitions = _partitions_list(n)
    if n > 0:
        partitions[-1] = [list(range(n))]
    return tuple(tuple(tuple(indices[i] for i in part)
                       for part in partition)
                 for partition in partitions)


def list_all_groupings(partition):
    """Return all possible groupings of states for a particular coarse graining
    (partition) of a network.

    Args:
        partition (tuple(tuple))): A partition of micro-elements into macro
            elements.

    Returns:
        tuple(tuple(tuple(tuple))): A tuple of all possible correspondences
            between micro-states and macro-states for the partition.
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


def list_all_coarse_grainings(indices):
    """Returns all possible ``CoarseGrains`` over these indices. """
    for partition in list_all_partitions(indices):
        for grouping in list_all_groupings(partition):
            yield CoarseGrain(partition, grouping)


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
        partition (tuple(tuple)): A partition of micro-elements into macro
            elements.
        grouping (tuple(tuple(tuple))): For each macro-element, a list of micro
            states which set it to ON or OFF.

    Returns:
        (nd.ndarray): A mapping from micro-states to macro-states.
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
    """Find the maximal coarse-graining of a micro-system.

    Args:
        network (Network): The network in question.
        state (tuple(int)): The state of the network.
        internal_indices (tuple(indices)): Nodes in the micro-system.

    Returns:
        tuple(int, CoarseGrain): The phi-value of the maximal CoarseGrain.
    """
    max_phi = float('-inf')
    max_coarse_grain = CoarseGrain((), ())

    for coarse_grain in list_all_coarse_grainings(internal_indices):
        try:
            subsystem = MacroSubsystem(network, state, internal_indices,
                                       coarse_grain=coarse_grain)
        except ConditionallyDependentError:
            continue

        phi = compute.big_phi(subsystem)
        if (phi - max_phi) > constants.EPSILON:
            max_phi = phi
            max_coarse_grain = coarse_grain

    return (max_phi, max_coarse_grain)


def emergence(network, state):
    """Check for emergence of a micro-system into a macro-system.

    Checks all possible partitions and groupings of the micro-system to find
    the spatial scale with maximum integrated information.

    Args:
        network (Network): The network of the micro-system under investigation.
        state (tuple(int)): The state of the network.

    Returns:
        MacroNetwork: The maximal coarse-graining of the micro-system.
    """
    micro_phi = compute.main_complex(network, state).phi
    max_phi = float('-inf')

    systems = utils.powerset(network.node_indices)
    for system in systems:
        (phi, _coarse_grain) = coarse_grain(network, state, system)
        if (phi - max_phi) > constants.EPSILON:
            max_phi = phi
            max_coarse_grain = _coarse_grain
            max_system = system

    return MacroNetwork(network=network,
                        system=max_system,
                        macro_phi=max_phi,
                        micro_phi=micro_phi,
                        partition=max_coarse_grain.output_grouping,
                        grouping=max_coarse_grain.state_grouping)


def phi_by_grain(network, state):
    list_of_phi = []

    systems = utils.powerset(network.node_indices)
    for system in systems:
        micro_subsystem = Subsystem(network, state, system)
        phi = compute.big_phi(micro_subsystem)
        list_of_phi.append([len(micro_subsystem), phi, system, None])

        for coarse_grain in list_all_coarse_grainings(system):
            try:
                subsystem = MacroSubsystem(network, state, system,
                                           coarse_grain=coarse_grain)
            except ConditionallyDependentError:
                continue

            phi = compute.big_phi(subsystem)
            list_of_phi.append([len(subsystem), phi, system, coarse_grain])
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
