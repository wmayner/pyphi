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
from .node import Node, expand_node_tpm
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


def generate_nodes(subsystem, indices):
    """Generate the |Node| objects for these indices."""
    # TODO: refactor this to node.py?
    return tuple(Node(subsystem, i, indices=indices) for i in indices)


class MacroSubsystem(Subsystem):
    """A subclass of |Subsystem| implementing macro computations.

    This subsystem performs blackboxing and coarse-graining of elements.

    Unlike |Subsystem|, whose TPM has dimensionality equal to that of the
    subsystem's network and represents nodes external to the system using
    singleton dimensions, |MacroSubsystem| squeezes the TPM to remove these
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

        # Ensure indices are not a `range`
        node_indices = tuple(node_indices)

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

        # TODO: refactor all blackboxing into one method?

        # Blackbox partial freeze
        # =======================
        if blackbox is not None:
            validate.blackbox(blackbox)
            blackbox = blackbox.reindex()
            self.tpm = self._blackbox_partial_freeze(blackbox)

        # Blackbox over time
        # ==================
        if time_scale != 1:
            validate.time_scale(time_scale)
            self.tpm, self.cm = self._blackbox_time(time_scale)

        # Blackbox in space
        # =================
        if blackbox is not None:
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
        self.nodes = generate_nodes(self, self.node_indices)

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
        # TODO: somehow don't assign to self.tpm, but still generate the nodes,
        # perhaps by passing the tpm to the node constructor?

        # Don't squeeze out the final dimension (which contains the
        # probability) for networks of size one
        if self.network.size > 1:
            self.tpm = np.squeeze(self.tpm)[..., internal_indices]

        # Re-index the subsystem nodes with the external nodes removed
        node_indices = reindex(internal_indices)
        nodes = generate_nodes(self, node_indices)

        # Re-calcuate the tpm based on the results of the cut
        tpm = np.rollaxis(
            np.array([
                expand_node_tpm(node.tpm[1]) for node in nodes
            ]), 0, len(node_indices) + 1)

        # The connectivity matrix is the network's connectivity matrix, with
        # cut applied, with all connections to/from external nodes severed,
        # shrunk to the size of the internal nodes.
        cm = self.cm[np.ix_(internal_indices, internal_indices)]

        state = utils.state_of(internal_indices, self.state)

        return (tpm, cm, node_indices, state)

    def _blackbox_partial_freeze(self, blackbox):
        """Freeze connections from hidden elements to elements in other boxes.

        Effectively this makes it so that only the output elements of each
        blackbox output to the rest of the system.
        """
        nodes = generate_nodes(self, self.node_indices)

        def hidden_from(a, b):
            # Returns True if a is a hidden in a different blackbox than b
            return (a in blackbox.hidden_indices and
                    not blackbox.in_same_box(a, b))

        # Condition each node on the state of input nodes in other boxes
        node_tpms = []
        for node in nodes:
            hidden_inputs = [input for input in node.input_indices
                             if hidden_from(input, node.index)]
            node_tpms.append(utils.condition_tpm(node.tpm[1],
                                                 hidden_inputs,
                                                 self.state))

        # Recalculate the system TPM
        expanded_tpms = [expand_node_tpm(tpm) for tpm in node_tpms]
        return np.rollaxis(
            np.array(expanded_tpms), 0, len(self.node_indices) + 1)

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

        state = blackbox.macro_state(self.state)
        node_indices = blackbox.macro_indices

        return (tpm, cm, node_indices, state)

    def _coarsegrain_space(self, coarse_grain):
        """Spatially coarse-grain the TPM and CM."""

        tpm = coarse_grain.macro_tpm(
            self.tpm, check_independence=(not self.is_cut))

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

    @property
    def micro_indices(self):
        """Indices of micro elements represented in this coarse-graining."""
        return tuple(sorted(idx for part in self.partition for idx in part))

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

        # TODO: only reindex if this coarse grain is not already from 0..n?
        # make_mapping calls this in a tight loop so it might be more efficient
        # to reindex conditionally.
        reindexed = self.reindex()

        micro_state = np.array(micro_state)
        return tuple(0 if sum(micro_state[list(reindexed.partition[i])])
                     in self.grouping[i][0] else 1
                     for i in self.macro_indices)

    def make_mapping(self):
        """Return a mapping from micro-state to the macro-states based on the
        partition and state grouping of this coarse-grain.

        Return:
            (nd.ndarray): A mapping from micro-states to macro-states. The
            |ith| entry in the mapping is the macro-state corresponding to the
            |ith| micro-state.
        """
        micro_states = utils.all_states(len(self.micro_indices))

        # Find the corresponding macro-state for each micro-state.
        # The i-th entry in the mapping is the macro-state corresponding to the
        # i-th micro-state.
        mapping = [convert.state2loli_index(self.macro_state(micro_state))
                   for micro_state in micro_states]
        return np.array(mapping)

    def macro_tpm(self, micro_tpm, check_independence=True):
        """Create a coarse-grained macro TPM.

        Args:
            micro_tpm (nd.array): The TPM of the micro-system.
            check_independence (boolean): If True, the method will raise a
                ``ConditionallyDependentError`` if the macro tpm is not
                conditionally independent.

        Returns:
            (np.ndarray): The state-by-node TPM of the macro-system.
        """
        validate.tpm(micro_tpm)

        if not utils.state_by_state(micro_tpm):
            micro_tpm = convert.state_by_node2state_by_state(micro_tpm)

        mapping = self.make_mapping()

        num_macro_states = 2 ** len(self.macro_indices)
        macro_tpm = np.zeros((num_macro_states, num_macro_states))

        micro_states = range(2 ** len(self.micro_indices))
        micro_state_transitions = itertools.product(micro_states, micro_states)

        # For every possible micro-state transition, get the corresponding past
        # and current macro-state using the mapping and add that probability to
        # the state-by-state macro TPM.
        for past_state, current_state in micro_state_transitions:
            macro_tpm[mapping[past_state], mapping[current_state]] += (
                micro_tpm[past_state, current_state])

        # Re-normalize each row because we're going from larger to smaller TPM
        # TODO: use utils.normalize when rebased onto develop
        macro_tpm = np.array([list(row) if sum(row) == 0
                              else list(row / sum(row))
                              for row in macro_tpm])

        if (check_independence and
                not validate.conditionally_independent(macro_tpm)):
            raise ConditionallyDependentError

        return convert.state_by_state2state_by_node(macro_tpm)


class Blackbox(namedtuple('Blackbox', ['partition', 'output_indices'])):
    """Class representing a blackboxing of a system.

    Attributes:
        partition (tuple(tuple(int)): The partition of nodes into boxes.
        output_indices (tuple(int)): Outputs of the blackboxes.
    """
    # TODO: validate!
    # TODO: validate that output indices are ordered?

    @property
    def hidden_indices(self):
        """All elements hidden inside the blackboxes."""
        return tuple(sorted(set(self.micro_indices) -
                            set(self.output_indices)))

    @property
    def micro_indices(self):
        """Indices of micro-elements in this blackboxing."""
        return tuple(sorted(idx for part in self.partition for idx in part))

    @property
    def macro_indices(self):
        """Fresh indices of macro-elements of the blackboxing."""
        return reindex(self.output_indices)

    def reindex(self):
        """Squeeze the indices of this blackboxing to ``0..n``.

        Returns:
            Blackbox: a new, reindexed ``Blackbox``.

        Example:
            >>> partition = ((3,), (2, 4))
            >>> output_indices = (2, 3)
            >>> blackbox = Blackbox(partition, output_indices)
            >>> blackbox.reindex()
            Blackbox(partition=((1,), (0, 2)), output_indices=(0, 1))
        """
        _map = dict(zip(self.micro_indices, reindex(self.micro_indices)))
        partition = tuple(
            tuple(_map[index] for index in group)
            for group in self.partition
        )
        output_indices = tuple(_map[i] for i in self.output_indices)

        return Blackbox(partition, output_indices)

    def macro_state(self, micro_state):
        """Compute the macro-state of this blackbox.

        This is just the state of the blackbox's output indices.

        Args:
            micro_state (tuple(int)): The state of the micro-elements in the
                blackbox.

        Returns:
            tuple(int): The state of the output indices.
        """
        assert len(micro_state) == len(self.micro_indices)

        reindexed = self.reindex()
        return utils.state_of(reindexed.output_indices, micro_state)

    def in_same_box(self, a, b):
        """Returns True if nodes ``a`` and ``b``` are in the same box."""
        assert a in self.micro_indices and b in self.micro_indices

        for part in self.partition:
            if a in part and b in part:
                return True

        return False


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
    # TODO: this only gives one output per box. Should there be more?
    for partition in all_partitions(indices):
        # Pick one output from each box
        for output_indices in itertools.product(*partition):
            yield Blackbox(partition, tuple(sorted(output_indices)))


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


def emergence(network, state, blackbox=False, coarse_grain=True,
              time_scales=None):
    """Check for the emergence of a micro-system into a macro-system.

    Checks all possible blackboxings and coarse-grainings of a system to find
    the spatial scale with maximum integrated information.

    Use the ``blackbox`` and ``coarse_grain`` args to specifiy whether to use
    blackboxing, coarse-graining, or both. The default is to just coarse-grain
    the system.

    Args:
        network (Network): The network of the micro-system under investigation.
        state (tuple(int)): The state of the network.
        blackbox (boolean): Set to True to enable blackboxing. Defaults to
            False.
        coarse_grain (boolean): Set to True to enable coarse-graining.
            Defaults to True.
        time_scales (list(int)): List of all time steps over which to check
            for emergence.

    Returns:
        MacroNetwork: The maximal macro-system generated from the micro-system.
    """
    if time_scales is None:
        time_scales = [1]

    def blackboxes(system):
        # Returns all blackboxes to evaluate
        if not blackbox:
            return [None]
        return all_blackboxes(system)

    def coarse_grains(blackbox, system):
        # Returns all coarse-grains to test
        if not coarse_grain:
            return [None]
        if blackbox is None:
            return all_coarse_grains(system)
        return all_coarse_grains(blackbox.output_indices)

    micro_phi = compute.main_complex(network, state).phi

    max_phi = float('-inf')
    max_network = None

    for system in utils.powerset(network.node_indices):
        for time_scale in time_scales:
            for blackbox in blackboxes(system):
                for coarse_grain in coarse_grains(blackbox, system):
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
    subsystems = []
    for state in utils.all_states(network.size):
        try:
            subsystems.append(Subsystem(network, state, network.node_indices))
        except validate.StateUnreachableError:
            continue
    return np.array([
        subsystem.effect_info(network.node_indices, network.node_indices)
        for subsystem in subsystems
    ]).mean()
