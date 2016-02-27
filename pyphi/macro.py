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

    This subsystem performs blackboxing and coarse-graining of elements.

    Unlike |Subsystem|, whose TPM has dimensionality equal to that of the
    subsystem's network and represents nodes external to the system using
    singleton dimensions, ``MacroSubsystem`` squeezes the TPM to remove these
    singletons. As a result, the node indices of the system are also squeezed
    to ``0..n`` so they properly index the TPM, and the state-tuple is
    reduced to the size of the system.

    After each macro update (temporal blackboxing, spatial blackboxing, and
    spatial coarse-graining) the TPM, CM, nodes, and state are updated so that
    they correctly represent the updated system.
    """
    # TODO refactor the _blackbox_space, _coarsegrain_space methods to methods
    # on their respective Blackbox and CoarseGrain objects? This would nicely
    # abstract the logic into a discrete, disconnected transformation.

    def __init__(self, network, state, node_indices, cut=None,
                 mice_cache=None, time_scale=1, blackbox=None,
                 coarse_grain=None):

        # Store original arguments to use in `apply_cut`
        self._network_state = state
        self._node_indices = node_indices  # Internal nodes
        self._time_scale = time_scale
        self._blackbox = blackbox
        self._coarse_grain = coarse_grain

        super().__init__(network, state, node_indices, cut, mice_cache)

        # Shrink TPM to size of internal indices
        # ======================================
        self.tpm, self.cm, self.node_indices, self._state = (
            self._squeeze(node_indices))

        # Blackbox over time
        # ==================
        if time_scale != 1:
            validate.time_scale(time_scale)
            self.tpm, self.cm = self._blackbox_time(time_scale)

        # Blackbox in space
        # =================
        if blackbox is not None:
            validate.blackbox(blackbox)
            blackbox = blackbox.reindex()
            self.tpm, self.cm, self.node_indices, self._state = (
                self._blackbox_space(blackbox))

        # Coarse-grain in space
        # =====================
        if coarse_grain is not None:
            validate.coarse_grain(coarse_grain)
            coarse_grain = coarse_grain.reindex()
            self.tpm, self.cm, self.node_indices, self._state = (
                self._coarsegrain_space(coarse_grain))

        # Regenerate nodes
        # ================
        self.nodes = tuple(Node(self, i, indices=self.node_indices)
                           for i in self.node_indices)

        # Hash the final subsystem - only compute hash once.
        self._hash = hash((self.network,
                           self.cut,
                           self._network_state,
                           self._node_indices,
                           self._blackbox,
                           self._coarse_grain))

        # The nodes represented in computed repertoires.
        self._dist_indices = self.node_indices

        validate.subsystem(self)

    def _squeeze(self, internal_indices):
        """Squeeze out all singleton dimensions in the Subsystem.

        Reindexes the subsystem so that the nodes are ``0..n`` where ``n`` is
        the number of internal indices in the system.
        """
        # Don't squeeze out the final dimension (which contains the
        # probability) for networks of size one
        if self.network.size > 1:
            self.tpm = np.squeeze(self.tpm)[..., internal_indices]

        # Re-index the subsystem nodes with the external nodes removed
        node_indices = reindex(internal_indices)
        nodes = tuple(Node(self, i, indices=node_indices)
                      for i in node_indices)

        # Re-calcuate the tpm based on the results of the cut
        tpm = np.rollaxis(
            np.array([
                node.expand_tpm(node_indices) for node in nodes
            ]), 0, len(node_indices) + 1)

        # The connectivity matrix is the network's connectivity matrix, with
        # cut applied, with all connections to/from external nodes severed,
        # shrunk to the size of the internal nodes.
        cm = self.cm[np.ix_(internal_indices, internal_indices)]

        state = utils.state_of(internal_indices, self.state)

        return (tpm, cm, node_indices, state)

    def _blackbox_time(self, time_scale):
        """Black box the CM and TPM over the given time_scale.

        TODO(billy): This is a blackboxed time. Coarse grain time is not yet
        implemented.
        """
        tpm = utils.run_tpm(self.tpm, time_scale)
        cm = utils.run_cm(self.cm, time_scale)

        return (tpm, cm)

    def _blackbox_space(self, blackbox):
        """Blackbox the TPM and CM in space.

        Conditions the TPM on the current value of the hidden nodes. The CM is
        set to universal connectivity.
        TODO: ^^ change this.

        This shrinks the size of the TPM by the number of hidden indices; now
        there is only `len(output_indices)` dimensions in the TPM and in the
        state of the subsystem.
        """
        # TODO: validate conditional independence?
        tpm = utils.condition_tpm(self.tpm, blackbox.hidden_indices,
                                  self.state)

        if len(self.node_indices) > 1:
            tpm = np.squeeze(tpm)[..., blackbox.output_indices]

        # Universal connectivity, for now.
        n = len(blackbox.output_indices)
        cm = np.ones((n, n))

        state = utils.state_of(blackbox.output_indices, self.state)
        node_indices = blackbox.macro_indices

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
    def cut_indices(self):
        """The indices of this system to be cut for |big_phi| computations.

        For macro computations the cut is applied to the underlying
        micro-system.
        """
        return self._node_indices

    @property
    def is_micro(self):
        """True if the system is pure micro without blackboxing of coarse-
        graining."""
        # TODO: do we need this?
        return self._coarse_grain is None and self._blackbox is None

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
                              blackbox=self._blackbox,
                              coarse_grain=self._coarse_grain)
                              # TODO: is the MICE cache reusable?
                              # mice_cache=self._mice_cache)

    def _potential_purviews(self, direction, mechanism, purviews=False):
        """Override Subsystem implementation using Network-level indices."""
        all_purviews = utils.powerset(self.node_indices)
        return irreducible_purviews(self.cm, direction,
                                    mechanism, all_purviews)

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
                self._blackbox == other._blackbox and
                self._coarse_grain == other._coarse_grain)

    def __hash__(self):
        return self._hash


class CoarseGrain(namedtuple('CoarseGrain', ['partition', 'grouping'])):
    """Represents a coarse graining of a collection of nodes.

    Attributes:
        partition (tuple(tuple)): The partition of micro-elements into
            macro-elements.
        grouping (tuple(tuple(tuple))): The grouping of micro-states into
            macro-states.
    """
    # TODO: validate? Currently implemented in validate.coarse_grain, but
    # should be moved here if this ever has an __init__ method

    def make_mapping(self):
        # TODO: move `make_mapping` function to here entirely
        return make_mapping(self.partition, self.grouping)

    def make_macro_tpm(self, tpm):
        return make_macro_tpm(tpm, self.make_mapping())

    @property
    def micro_indices(self):
        """Indices of micro elements represented in this coarse-graining."""
        return tuple(sorted(index for group in self.partition
                            for index in group))

    @property
    def macro_indices(self):
        """Indices of macro elements of this coarse-graining."""
        return tuple(range(len(self.partition)))

    def reindex(self):
        """Re-index this coarse graining to use squeezed indices.

        The output grouping is translated to use indices `0..n`, where `n` is
        the number of micro indices in the coarse-graining. Re-indexing does
        not effect the state grouping, which is already index-independent.

        Returns:
            CoarseGrain: A new CoarseGrain object, indexed from `0..n`

        Example:
            >>> partition = ((1, 2),)
            >>> grouping = (((0,), (1, 2)),)
            >>> coarse_grain = CoarseGrain(partition, grouping)
            >>> coarse_grain.reindex()
            CoarseGrain(partition=((0, 1),), grouping=(((0,), (1, 2)),))
        """
        _map = dict(zip(self.micro_indices, reindex(self.micro_indices)))
        partition = tuple(
            tuple(_map[index] for index in group)
            for group in self.partition
        )
        return CoarseGrain(partition, self.grouping)

    def macro_state(self, micro_state):
        """Translate a micro state to a macro state

        Args:
            micro_state (tuple(int)): The state of the micro nodes in this
                coarse-graining.

        Returns:
            tuple(int): The state of the macro system, translated as specified
                by this coarse-graining.

        Example:
            >>> coarse_grain = CoarseGrain(((1, 2),), (((0,), (1, 2)),))
            >>> coarse_grain.macro_state((0, 0))
            (0,)
            >>> coarse_grain.macro_state((1, 0))
            (1,)
            >>> coarse_grain.macro_state((1, 1))
            (1,)
        """
        assert len(micro_state) == len(self.micro_indices)

        reindexed = self.reindex()

        micro_state = np.array(micro_state)
        return tuple(0 if sum(micro_state[list(reindexed.partition[i])])
                     in self.grouping[i][0] else 1
                     for i in self.macro_indices)


class Blackbox(namedtuple('Blackbox', ['hidden_indices', 'output_indices'])):
    """Class representing a blackboxing of a system.

    Attributes:
        hidden_indices (tuple(int)): Nodes which are hidden inside blackboxes.
        output_indices (tuple(int)): Outputs of the blackboxes.
    """
    # TODO: validate!

    @property
    def micro_indices(self):
        """Indices of micro-elements in this blackboxing."""
        return tuple(sorted(self.hidden_indices + self.output_indices))

    @property
    def macro_indices(self):
        """Fresh indices of macro-elements of the blackboxing."""
        return reindex(self.output_indices)

    def reindex(self):
        """Squeeze the indices of this blackboxing to ``0..n``.

        Returns:
            Blackbox: a new, reindexed ``Blackbox``.
        """
        _map = dict(zip(self.micro_indices, reindex(self.micro_indices)))
        hidden_indices = tuple(_map[i] for i in self.hidden_indices)
        output_indices = tuple(_map[i] for i in self.output_indices)

        return Blackbox(hidden_indices, output_indices)


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


def all_partitions(indices):
    """Return a list of all possible coarse grains of a network.

    Args:
        indices (tuple(int)): The micro indices to partition.

    Yields:
        tuple(tuple): A possible partition. Each element of the tuple
            is a tuple of micro-elements which correspond to macro-elements.
    """
    n = len(indices)
    partitions = _partitions_list(n)
    if n > 0:
        partitions[-1] = [list(range(n))]

    for partition in partitions:
        yield tuple(tuple(indices[i] for i in part)
                    for part in partition)


def list_all_partitions(indices):
    """Cast ``all_partitions`` to a list.

    TODO: remove this alias.
    """
    list(all_groupings(indices))


def all_groupings(partition):
    """Return all possible groupings of states for a particular coarse graining
    (partition) of a network.

    Args:
        partition (tuple(tuple))): A partition of micro-elements into macro
            elements.

    Yields:
        tuple(tuple(tuple)): A grouping of micro-states into macro states of
            system.

    TODO: document exactly how to interpret the grouping.
    """
    if not all(len(part) > 0 for part in partition):
        raise ValueError('Each part of the partition must have at least one '
                         'element.')

    micro_groupings = [_partitions_list(len(part) + 1) if len(part) > 1
                       else [[[0], [1]]] for part in partition]

    for grouping in itertools.product(*micro_groupings):
        if all(len(element) < 3 for element in grouping):
            yield tuple(tuple(tuple(tuple(state) for state in states)
                        for states in grouping))


def list_all_groupings(partition):
    """Cast ``all_groupings`` to a list.

    TODO: remove this alias.
    """
    return list(all_groupings(partition))


def all_coarse_grains(indices):
    """Generator over all possible ``CoarseGrains`` of these indices.

    Args:
        indices (tuple(int)): Node indices to coarse grain.

    Yields:
        CoarseGrain: The next coarse-grain for ``indices``.
    """
    for partition in all_partitions(indices):
        for grouping in all_groupings(partition):
            yield CoarseGrain(partition, grouping)


def all_blackboxes(indices):
    """Generator over all possible blackboxings of these indices.

    Args:
        indices (tuple(int)): Nodes to blackbox.

    Yields:
        Blackbox: The next blackbox of ``indices``.
    """
    for hidden_indices, output_indices in utils.directed_bipartition(indices):
        yield Blackbox(hidden_indices, output_indices)


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
        coarse_grain (CoarseGrain): The coarse-graining of micro-elements into
            macro-elements.
        time_scale (int): The time scale the macro-network run over.
        blackbox (Blackbox): The blackboxing of micro elements in the network.
        emergence (float): The difference between the |big_phi| of the macro-
            and the micro-system.
    """
    def __init__(self, network, system, macro_phi, micro_phi, coarse_grain,
                 time_scale=1, blackbox=None):

        self.network = network
        self.system = system
        self.phi = macro_phi
        self.micro_phi = micro_phi
        self.time_scale = time_scale
        self.coarse_grain = coarse_grain
        self.blackbox = blackbox

    def __str__(self):
        return "MacroNetwork(phi={0}, emergence={1})".format(
            self.phi, self.emergence)

    @property
    def emergence(self):
        """Difference between the |big_phi| of the macro and micro systems"""
        return round(self.phi - self.micro_phi, config.PRECISION)


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

    for coarse_grain in all_coarse_grains(internal_indices):
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
                        coarse_grain=max_coarse_grain)


def blackbox_emergence(network, state, time_scales=None):
    """Check for the emergence of a micro-system into a macro-system, using
    blackboxing and coarse-graining.

    Args:
        network (Network): The network of the micro-system under investigation.
        state (tuple(int)): The state of the network.
        time_scales (list(int)): List of all time steps to check for emergence.

    Returns:
        MacroNetwork: The maximal coarse-graining of the micro-system.

    TODO: refactor this to ``emergence``; parameterize so that you can choose
    blackboxing, coarse-graining, or both.
    """
    if time_scales is None:
        time_scales = [1]

    micro_phi = compute.main_complex(network, state).phi

    max_phi = float('-inf')
    max_network = None

    for system in utils.powerset(network.node_indices):
        for time_scale in time_scales:
            for blackbox in all_blackboxes(system):
                for coarse_grain in all_coarse_grains(blackbox.output_indices):
                    try:
                        subsystem = MacroSubsystem(
                            network, state, system,
                            time_scale=time_scale,
                            blackbox=blackbox,
                            coarse_grain=coarse_grain)
                    except (validate.StateUnreachableError,
                            ConditionallyDependentError):
                        continue

                    phi = compute.big_phi(subsystem)

                    if (phi - max_phi) > constants.EPSILON:
                        max_phi = phi
                        max_network = MacroNetwork(
                            network=network,
                            macro_phi=phi,
                            micro_phi=micro_phi,
                            system=system,
                            time_scale=time_scale,
                            blackbox=blackbox,
                            coarse_grain=coarse_grain)

    return max_network


def phi_by_grain(network, state):
    list_of_phi = []

    systems = utils.powerset(network.node_indices)
    for system in systems:
        micro_subsystem = Subsystem(network, state, system)
        phi = compute.big_phi(micro_subsystem)
        list_of_phi.append([len(micro_subsystem), phi, system, None])

        for coarse_grain in all_coarse_grains(system):
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
