# macro.py
"""Methods for coarse-graining systems to different levels of spatial analysis."""

import itertools
import logging
from collections import namedtuple

import numpy as np
from scipy.stats import entropy

from pyphi.formalism import iit3 as _iit3

from . import convert
from . import distribution
from . import utils
from . import validate
from .conf import config
from .direction import Direction
from .exceptions import ConditionallyDependentError
from .exceptions import StateUnreachableError
from .labels import NodeLabels
from .node import expand_node_tpm
from .node import generate_nodes
from .substrate import irreducible_purviews
from .system import System
from .tpm import ExplicitTPM

# Create a logger for this module.
log = logging.getLogger(__name__)

# Load precomputed partition lists.
_NUM_PRECOMPUTED_PARTITION_LISTS = 10
_partition_lists = utils.load_data("partition_lists", _NUM_PRECOMPUTED_PARTITION_LISTS)


def reindex(indices):
    """Generate a new set of node indices, the size of indices."""
    return tuple(range(len(indices)))


def rebuild_system_tpm(node_tpms):
    """Reconstruct the substrate TPM from a collection of node TPMs.

    Args:
        node_tpms (Iterable[ExplicitTPM]): The collection of node TPMs.

    Returns:
        ExplicitTPM: The system TPM which comprises the input node TPMs.
    """
    tpm = np.stack([expand_node_tpm(tpm).tpm for tpm in node_tpms], axis=-1)
    return ExplicitTPM(tpm, validate=True)


# TODO This should be a method of the TPM class in tpm.py
def remove_singleton_dimensions(tpm):
    """Remove singleton dimensions from the TPM.

    Singleton dimensions are created by conditioning on a set of elements.
    This removes those elements from the TPM, leaving a TPM that only
    describes the non-conditioned elements.

    Note that indices used in the original TPM must be reindexed for the
    smaller TPM.
    """
    # Don't squeeze out the final dimension (which contains the probability)
    # for substrates with one element.
    if tpm.ndim <= 2:
        return tpm

    return tpm.squeeze()[..., tpm.tpm_indices()]


def run_tpm(system, direction, steps, blackbox):
    """Iterate the TPM for the given number of timesteps.

    Returns:
        ExplicitTPM: tpm * (noise_tpm^(t-1))
    """
    # Generate noised TPM
    # Noise the connections from every output element to elements in other
    # boxes.
    node_tpms = []
    for node in system.nodes:
        if direction == Direction.CAUSE:
            node_tpm = node.cause_tpm_on
        elif direction == Direction.EFFECT:
            node_tpm = node.effect_tpm_on
        else:
            return validate.direction(direction)
        for input_node in node.inputs:
            if (
                not blackbox.in_same_box(node.index, input_node)
                and input_node in blackbox.output_indices
            ):
                node_tpm = node_tpm.marginalize_out([input_node])

        node_tpms.append(node_tpm)

    noised_tpm = rebuild_system_tpm(node_tpms)
    noised_tpm = convert.state_by_node2state_by_state(noised_tpm.tpm)

    if direction == Direction.CAUSE:
        tpm = convert.state_by_node2state_by_state(system.cause_tpm.tpm)
    elif direction == Direction.EFFECT:
        tpm = convert.state_by_node2state_by_state(system.effect_tpm.tpm)
    else:
        return validate.direction(direction)

    # Muliply by noise
    tpm = np.dot(tpm, np.linalg.matrix_power(noised_tpm, steps - 1))

    return ExplicitTPM(convert.state_by_state2state_by_node(tpm), validate=True)


class SystemAttrs(
    namedtuple("SystemAttrs", ["cause_tpm", "effect_tpm", "cm", "node_indices", "state"])
):
    """An immutable container that holds all the attributes of a system.

    Versions of this object are passed down the steps of the micro-to-macro
    pipeline.
    """

    @property
    def node_labels(self):
        """Return the labels for macro nodes."""
        assert next(iter(self.node_indices)) == 0
        labels = [f"m{i}" for i in self.node_indices]
        return NodeLabels(labels, self.node_indices)

    @property
    def nodes(self):
        return generate_nodes(
            self.cause_tpm,
            self.effect_tpm,
            self.cm,
            self.state,
            self.node_indices,
            self.node_labels,
        )

    @staticmethod
    def pack(system):
        return SystemAttrs(
            system.cause_tpm,
            system.effect_tpm,
            system.cm,
            system.node_indices,
            system.state,
        )

    def apply(self, system):
        system.cause_tpm = self.cause_tpm
        system.effect_tpm = self.effect_tpm
        system.cm = self.cm
        system.node_indices = self.node_indices
        system.node_labels = self.node_labels
        system.nodes = self.nodes
        system.state = self.state


class MacroSystem(System):
    """A subclass of |System| implementing macro computations.

    This system performs blackboxing and coarse-graining of elements.

    Unlike |System|, whose TPM has dimensionality equal to that of the
    system's substrate and represents nodes external to the system using
    singleton dimensions, |MacroSystem| squeezes the TPM to remove these
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

    def __init__(
        self,
        substrate,
        state,
        nodes=None,
        partition=None,
        time_scale=1,
        blackbox=None,
        *_args,
        **_kwargs,
    ):
        raise NotImplementedError(
            "MacroSystem is undergoing rewrite in P7b; "
            "restored after the kernel rewrite lands."
        )

    def __init_disabled__(
        self,
        substrate,
        state,
        nodes=None,
        partition=None,
        time_scale=1,
        blackbox=None,
        coarse_grain=None,
    ):
        # Ensure indices are not a `range`
        micro_node_indices = substrate.node_labels.coerce_to_indices(nodes)

        # Store original arguments to use in `apply_cut`
        self.substrate_state = state
        self.micro_node_indices = micro_node_indices  # Internal nodes
        self.time_scale = time_scale
        self.blackbox = blackbox
        self.coarse_grain = coarse_grain

        super().__init__(substrate, state, micro_node_indices, partition)  # pyright: ignore[reportArgumentType]

        validate.blackbox_and_coarse_grain(blackbox, coarse_grain)

        system = SystemAttrs.pack(self)

        # Shrink TPM to size of internal indices
        # ======================================
        system = self._squeeze(system)

        # Blackbox partial freeze
        # =======================
        if blackbox is not None:
            validate.blackbox(blackbox)
            blackbox = blackbox.reindex()
            system = self._blackbox_partial_noise(blackbox, system)

        # Blackbox over time
        # ==================
        if time_scale != 1:
            assert blackbox is not None
            validate.time_scale(time_scale)
            system = self._blackbox_time(time_scale, blackbox, system)

        # Blackbox in space
        # =================
        if blackbox is not None:
            system = self._blackbox_space(blackbox, system)

        # Coarse-grain in space
        # =====================
        if coarse_grain is not None:
            validate.coarse_grain(coarse_grain)
            coarse_grain = coarse_grain.reindex()
            system = self._coarsegrain_space(coarse_grain, self.is_partitioned, system)

        system.apply(self)

        validate.system(self)

    @staticmethod
    def _squeeze(system):
        """Squeeze out all singleton dimensions in the System.

        Reindexes the system so that the nodes are ``0..n`` where ``n`` is
        the number of internal indices in the system.
        """
        assert system.node_indices == system.cause_tpm.tpm_indices()
        assert system.node_indices == system.effect_tpm.tpm_indices()

        internal_indices = system.effect_tpm.tpm_indices()

        cause_tpm = remove_singleton_dimensions(system.cause_tpm)
        effect_tpm = remove_singleton_dimensions(system.effect_tpm)

        # The connectivity matrix is the substrate's connectivity matrix, with
        # partition applied, with all connections to/from external nodes
        # severed, shrunk to the size of the internal nodes.
        cm = system.cm[np.ix_(internal_indices, internal_indices)]

        state = utils.state_of(internal_indices, system.state)

        # Re-index the system nodes with the external nodes removed
        node_indices = reindex(internal_indices)
        nodes = generate_nodes(cause_tpm, effect_tpm, cm, state, node_indices)

        # Re-calculate the tpm based on the results of the partition
        cause_tpm = rebuild_system_tpm(node.cause_tpm_on for node in nodes)
        effect_tpm = rebuild_system_tpm(node.effect_tpm_on for node in nodes)

        return SystemAttrs(cause_tpm, effect_tpm, cm, node_indices, state)

    @staticmethod
    def _blackbox_partial_noise(blackbox, system):
        """Noise connections from hidden elements to other boxes."""
        # Noise inputs from non-output elements hidden in other boxes
        node_cause_tpms = []
        node_effect_tpms = []
        for node in system.nodes:
            node_cause_tpm = node.cause_tpm_on
            node_effect_tpm = node.effect_tpm_on
            for input_node in node.inputs:
                if blackbox.hidden_from(input_node, node.index):
                    node_cause_tpm = node_cause_tpm.marginalize_out([input_node])
                    node_effect_tpm = node_effect_tpm.marginalize_out([input_node])

            node_cause_tpms.append(node_cause_tpm)
            node_effect_tpms.append(node_effect_tpm)

        cause_tpm = rebuild_system_tpm(node_cause_tpms)
        effect_tpm = rebuild_system_tpm(node_effect_tpms)

        system = system._replace(cause_tpm=cause_tpm)
        system = system._replace(effect_tpm=effect_tpm)
        return system

    @staticmethod
    def _blackbox_time(time_scale, blackbox, system):
        """Black box the CM and TPM over the given time_scale."""
        blackbox = blackbox.reindex()

        cause_tpm = run_tpm(system, Direction.CAUSE, time_scale, blackbox)
        effect_tpm = run_tpm(system, Direction.EFFECT, time_scale, blackbox)

        # Universal connectivity, for now.
        n = len(system.node_indices)
        cm = np.ones((n, n))

        return SystemAttrs(cause_tpm, effect_tpm, cm, system.node_indices, system.state)

    def _blackbox_space(self, blackbox, system):
        """Blackbox the TPM and CM in space.

        Conditions the TPM on the current value of the hidden nodes. The CM is
        set to universal connectivity.

        .. TODO: change this ^

        This shrinks the size of the TPM by the number of hidden indices; now
        there is only `len(output_indices)` dimensions in the TPM and in the
        state of the system.
        """
        cause_tpm = system.cause_tpm.marginalize_out(blackbox.hidden_indices)
        effect_tpm = system.effect_tpm.marginalize_out(blackbox.hidden_indices)

        assert blackbox.output_indices == cause_tpm.tpm_indices()
        assert blackbox.output_indices == effect_tpm.tpm_indices()

        cause_tpm = remove_singleton_dimensions(cause_tpm)
        effect_tpm = remove_singleton_dimensions(effect_tpm)
        n = len(blackbox)
        cm = np.zeros((n, n))
        for i, j in itertools.product(range(n), repeat=2):
            # TODO: don't pull cm from self
            # self.blackbox is guaranteed to exist here since we're in _blackbox_space
            assert self.blackbox is not None, (
                "_blackbox_space called with self.blackbox=None"
            )
            outputs = self.blackbox.outputs_of(i)
            to = self.blackbox.partition[j]
            if self.cm[np.ix_(outputs, to)].sum() > 0:
                cm[i, j] = 1

        state = blackbox.macro_state(system.state)
        node_indices = blackbox.macro_indices

        return SystemAttrs(cause_tpm, effect_tpm, cm, node_indices, state)

    @staticmethod
    def _coarsegrain_space(coarse_grain, is_cut, system):
        """Spatially coarse-grain the TPM and CM."""
        cause_tpm = coarse_grain.macro_tpm(
            system.cause_tpm.tpm, check_independence=(not is_cut)
        )
        effect_tpm = coarse_grain.macro_tpm(
            system.effect_tpm.tpm, check_independence=(not is_cut)
        )

        node_indices = coarse_grain.macro_indices
        state = coarse_grain.macro_state(system.state)

        # Universal connectivity, for now.
        n = len(node_indices)
        cm = np.ones((n, n))

        cause_tpm = ExplicitTPM(cause_tpm, validate=True)
        effect_tpm = ExplicitTPM(effect_tpm, validate=True)
        return SystemAttrs(cause_tpm, effect_tpm, cm, node_indices, state)

    @property
    def partition_indices(self):  # pyright: ignore[reportIncompatibleVariableOverride]
        """The indices of this system's partition for |big_phi| computations.

        For macro computations the partition is applied to the underlying
        micro-system.
        """
        return self.micro_node_indices

    @property
    def partitioned_mechanisms(self):  # pyright: ignore[reportIncompatibleVariableOverride]
        """The mechanisms of this system that are currently partitioned.

        Note that although ``partition_indices`` returns micro indices, this
        returns macro mechanisms.

        Returns:
            list[tuple[int, ...]]: The list of partitioned mechanisms.
        """
        return [
            mechanism
            for mechanism in utils.powerset(self.node_indices, nonempty=True)
            if self.partition.splits_mechanism(self.macro2micro(mechanism))
        ]

    @property
    def partition_node_labels(self):  # pyright: ignore[reportIncompatibleVariableOverride]
        """Labels for the nodes that can be partitioned.

        These are the labels of the micro elements.
        """
        return self.substrate.node_labels

    def apply_cut(self, partition):
        """Return a partitioned version of this |MacroSystem|.

        Args:
            partition (DirectedBipartition): The partition to apply to this
                |MacroSystem|.

        Returns:
            MacroSystem: The partitioned version of this |MacroSystem|.
        """
        # TODO: is the MICE cache reusable?
        return MacroSystem(
            self.substrate,
            self.substrate_state,
            self.micro_node_indices,
            partition=partition,
            time_scale=self.time_scale,
            blackbox=self.blackbox,
            coarse_grain=self.coarse_grain,
        )

    def potential_purviews(self, direction, mechanism, purviews=False):  # pyright: ignore[reportIncompatibleMethodOverride]
        """Override System implementation using Substrate-level indices."""
        all_purviews = utils.powerset(self.node_indices)
        return irreducible_purviews(self.cm, direction, mechanism, all_purviews)

    def macro2micro(self, macro_indices):
        """Return all micro indices which compose the elements specified by
        ``macro_indices``.
        """

        def from_partition(partition, macro_indices):
            micro_indices = itertools.chain.from_iterable(
                partition[i] for i in macro_indices
            )
            return tuple(sorted(micro_indices))

        if self.blackbox and self.coarse_grain:
            cg_micro_indices = from_partition(self.coarse_grain.partition, macro_indices)
            return from_partition(self.blackbox.partition, reindex(cg_micro_indices))
        if self.blackbox:
            return from_partition(self.blackbox.partition, macro_indices)
        if self.coarse_grain:
            return from_partition(self.coarse_grain.partition, macro_indices)
        return macro_indices

    def macro2blackbox_outputs(self, macro_indices):
        """Given a set of macro elements, return the blackbox output elements
        which compose these elements.
        """
        if not self.blackbox:
            raise ValueError("System is not blackboxed")

        return tuple(
            sorted(
                set(self.macro2micro(macro_indices)).intersection(
                    self.blackbox.output_indices
                )
            )
        )

    def __repr__(self):
        return "MacroSystem(" + repr(self.nodes) + ")"

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        """Two macro systems are equal if each underlying |System| is equal
        and all macro attributes are equal.
        """
        if type(self) is not type(other):
            return False

        # Type narrowing: we know other is MacroSystem now
        assert isinstance(other, MacroSystem)
        return (
            super().__eq__(other)
            and self.time_scale == other.time_scale
            and self.blackbox == other.blackbox
            and self.coarse_grain == other.coarse_grain
        )

    def __hash__(self):
        return hash(
            (super().__hash__(), self.time_scale, self.blackbox, self.coarse_grain)
        )


class CoarseGrain(namedtuple("CoarseGrain", ["partition", "grouping"])):
    """Represents a coarse graining of a collection of nodes.

    Attributes:
        partition (tuple[tuple]): The partition of micro-elements into
            macro-elements.
        grouping (tuple[tuple[tuple]]): The grouping of micro-states into
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

    def __len__(self):
        return len(self.partition)

    def reindex(self):
        """Re-index this coarse graining to use squeezed indices.

        The output grouping is translated to use indices ``0..n``, where ``n``
        is the number of micro indices in the coarse-graining. Re-indexing does
        not effect the state grouping, which is already index-independent.

        Returns:
            CoarseGrain: A new |CoarseGrain| object, indexed from ``0..n``.

        Example:
            >>> partition = ((1, 2),)
            >>> grouping = (((0,), (1, 2)),)
            >>> coarse_grain = CoarseGrain(partition, grouping)
            >>> coarse_grain.reindex()
            CoarseGrain(partition=((0, 1),), grouping=(((0,), (1, 2)),))
        """
        _map = dict(zip(self.micro_indices, reindex(self.micro_indices), strict=False))
        partition = tuple(
            tuple(_map[index] for index in group) for group in self.partition
        )
        return CoarseGrain(partition, self.grouping)

    def macro_state(self, micro_state):
        """Translate a micro state to a macro state

        Args:
            micro_state (tuple[int]): The state of the micro nodes in this
                coarse-graining.

        Returns:
            tuple[int]: The state of the macro system, translated as specified
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
        return tuple(
            (
                0
                if sum(micro_state[list(reindexed.partition[i])]) in self.grouping[i][0]
                else 1
            )
            for i in self.macro_indices
        )

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
        mapping = [
            convert.state2le_index(self.macro_state(micro_state))
            for micro_state in micro_states
        ]
        return np.array(mapping)

    def macro_tpm_sbs(self, state_by_state_micro_tpm):
        """Create a state-by-state coarse-grained macro TPM.

        Args:
            micro_tpm (nd.array): The state-by-state TPM of the micro-system.

        Returns:
            np.ndarray: The state-by-state TPM of the macro-system.
        """
        tpm = ExplicitTPM(state_by_state_micro_tpm)
        tpm.validate(check_independence=False)

        mapping = self.make_mapping()

        num_macro_states = 2 ** len(self.macro_indices)
        macro_tpm = np.zeros((num_macro_states, num_macro_states))

        micro_states = range(2 ** len(self.micro_indices))
        micro_state_transitions = itertools.product(micro_states, repeat=2)

        # For every possible micro-state transition, get the corresponding
        # previous and next macro-state using the mapping and add that
        # probability to the state-by-state macro TPM.
        for previous_state, current_state in micro_state_transitions:
            macro_tpm[mapping[previous_state], mapping[current_state]] += (
                state_by_state_micro_tpm[previous_state, current_state]
            )

        # Re-normalize each row because we're going from larger to smaller TPM
        return np.array([distribution.normalize(row) for row in macro_tpm])

    def macro_tpm(self, micro_tpm, check_independence=True):
        """Create a coarse-grained macro TPM.

        Args:
            micro_tpm (nd.array): The TPM of the micro-system.
            check_independence (bool): Whether to check that the macro TPM is
                conditionally independent.

        Raises:
            ConditionallyDependentError: If ``check_independence`` is ``True``
                and the macro TPM is not conditionally independent.

        Returns:
            np.ndarray: The state-by-node TPM of the macro-system.
        """
        if not ExplicitTPM(micro_tpm).is_state_by_state():
            micro_tpm = convert.state_by_node2state_by_state(micro_tpm)

        macro_tpm = self.macro_tpm_sbs(micro_tpm)

        if check_independence:
            tpm = ExplicitTPM(macro_tpm)
            tpm.conditionally_independent()

        return convert.state_by_state2state_by_node(macro_tpm)


class Blackbox(namedtuple("Blackbox", ["partition", "output_indices"])):
    """Class representing a blackboxing of a system.

    Attributes:
        partition (tuple[tuple[int]]): The partition of nodes into boxes.
        output_indices (tuple[int]): Outputs of the blackboxes.
    """

    # TODO: validate!
    # TODO: validate that output indices are ordered?

    @property
    def hidden_indices(self):
        """All elements hidden inside the blackboxes."""
        return tuple(sorted(set(self.micro_indices) - set(self.output_indices)))

    @property
    def micro_indices(self):
        """Indices of micro-elements in this blackboxing."""
        return tuple(sorted(idx for part in self.partition for idx in part))

    @property
    def macro_indices(self):
        """Fresh indices of macro-elements of the blackboxing."""
        return reindex(self.output_indices)

    def __len__(self):
        return len(self.partition)

    def outputs_of(self, partition_index):
        """The outputs of the partition at ``partition_index``.

        Note that this returns a tuple of element indices, since coarse-
        grained blackboxes may have multiple outputs.
        """
        partition = self.partition[partition_index]
        outputs = set(partition).intersection(self.output_indices)
        return tuple(sorted(outputs))

    def reindex(self):
        """Squeeze the indices of this blackboxing to ``0..n``.

        Returns:
            Blackbox: a new, reindexed |Blackbox|.

        Example:
            >>> partition = ((3,), (2, 4))
            >>> output_indices = (2, 3)
            >>> blackbox = Blackbox(partition, output_indices)
            >>> blackbox.reindex()
            Blackbox(partition=((1,), (0, 2)), output_indices=(0, 1))
        """
        _map = dict(zip(self.micro_indices, reindex(self.micro_indices), strict=False))
        partition = tuple(
            tuple(_map[index] for index in group) for group in self.partition
        )
        output_indices = tuple(_map[i] for i in self.output_indices)

        return Blackbox(partition, output_indices)

    def macro_state(self, micro_state):
        """Compute the macro-state of this blackbox.

        This is just the state of the blackbox's output indices.

        Args:
            micro_state (tuple[int]): The state of the micro-elements in the
                blackbox.

        Returns:
            tuple[int]: The state of the output indices.
        """
        assert len(micro_state) == len(self.micro_indices)

        reindexed = self.reindex()
        return utils.state_of(reindexed.output_indices, micro_state)

    def in_same_box(self, a, b):
        """Return ``True`` if nodes ``a`` and ``b``` are in the same box."""
        assert a in self.micro_indices
        assert b in self.micro_indices

        return any(a in part and b in part for part in self.partition)

    def hidden_from(self, a, b):
        """Return True if ``a`` is hidden in a different box than ``b``."""
        return a in self.hidden_indices and not self.in_same_box(a, b)


def _partitions_list(N):
    """Return a list of partitions of the |N| binary nodes.

    Args:
        N (int): The number of nodes under consideration.

    Returns:
        list[list]: A list of lists, where each inner list is the set of
        micro-elements corresponding to a macro-element.

    Example:
        >>> _partitions_list(3)
        [[[0, 1], [2]], [[0, 2], [1]], [[0], [1, 2]], [[0], [1], [2]]]
    """
    if N < (_NUM_PRECOMPUTED_PARTITION_LISTS):
        return list(_partition_lists[N])
    raise ValueError(
        f"Partition lists not yet available for system with "
        f"{_NUM_PRECOMPUTED_PARTITION_LISTS} nodes or more"
    )


def all_partitions(indices):
    """Return a list of all possible coarse grains of a substrate.

    Args:
        indices (tuple[int]): The micro indices to partition.

    Yields:
        tuple[tuple]: A possible partition. Each element of the tuple
        is a tuple of micro-elements which correspond to macro-elements.
    """
    n = len(indices)
    partitions = _partitions_list(n)
    if n > 0:
        partitions[-1] = [list(range(n))]

    for partition in partitions:
        yield tuple(tuple(indices[i] for i in part) for part in partition)


def all_groupings(partition):
    """Return all possible groupings of states for a particular coarse graining
    (partition) of a substrate.

    Args:
        partition (tuple[tuple]): A partition of micro-elements into macro
            elements.

    Yields:
        tuple[tuple[tuple]]: A grouping of micro-states into macro states of
        system.

    TODO: document exactly how to interpret the grouping.
    """
    if not all(partition):
        raise ValueError("Each part of the partition must have at least one element.")

    micro_groupings = [
        _partitions_list(len(part) + 1) if len(part) > 1 else [[[0], [1]]]
        for part in partition
    ]

    for grouping in itertools.product(*micro_groupings):
        if all(len(element) < 3 for element in grouping):
            yield tuple(tuple(tuple(state) for state in states) for states in grouping)


def all_coarse_grains(indices):
    """Generator over all possible |CoarseGrains| of these indices.

    Args:
        indices (tuple[int]): Node indices to coarse grain.

    Yields:
        CoarseGrain: The next |CoarseGrain| for ``indices``.
    """
    for partition in all_partitions(indices):
        for grouping in all_groupings(partition):
            yield CoarseGrain(partition, grouping)


def all_coarse_grains_for_blackbox(blackbox):
    """Generator over all |CoarseGrains| for the given blackbox.

    If a box has multiple outputs, those outputs are partitioned into the same
    coarse-grain macro-element.
    """
    for partition in all_partitions(blackbox.output_indices):
        for grouping in all_groupings(partition):
            coarse_grain = CoarseGrain(partition, grouping)
            try:
                validate.blackbox_and_coarse_grain(blackbox, coarse_grain)
            except ValueError:
                continue
            yield coarse_grain


def all_blackboxes(indices):
    """Generator over all possible blackboxings of these indices.

    Args:
        indices (tuple[int]): Nodes to blackbox.

    Yields:
        Blackbox: The next |Blackbox| of ``indices``.
    """
    for partition in all_partitions(indices):
        # TODO? don't consider the empty set here
        # (pass `nonempty=True` to `powerset`)
        for output_indices in utils.powerset(indices):
            blackbox = Blackbox(partition, output_indices)
            try:  # Ensure every box has at least one output
                validate.blackbox(blackbox)
            except ValueError:
                continue
            yield blackbox


class MacroNetwork:
    """A coarse-grained substrate of nodes.

    See the :ref:`macro-micro` example in the documentation for more
    information.

    Attributes:
        substrate (Substrate): The substrate object of the macro-system.
        phi (float): The |big_phi| of the substrate's major complex.
        micro_substrate (Substrate): The substrate object of the corresponding micro
            system.
        micro_phi (float): The |big_phi| of the major complex of the
            corresponding micro-system.
        coarse_grain (CoarseGrain): The coarse-graining of micro-elements
            into macro-elements.
        time_scale (int): The time scale the macro-substrate run over.
        blackbox (Blackbox): The blackboxing of micro elements in the substrate.
        emergence (float): The difference between the |big_phi| of the macro-
            and the micro-system.
    """

    def __init__(
        self,
        substrate,
        system,
        macro_phi,
        micro_phi,
        coarse_grain,
        time_scale=1,
        blackbox=None,
    ):
        # Preserve DistanceResult type if possible, otherwise convert to PyPhiFloat
        from pyphi.data_structures.pyphi_float import PyPhiFloat
        from pyphi.measures.distribution import DistanceResult

        self.substrate = substrate
        self.system = system
        if isinstance(macro_phi, DistanceResult):
            self.phi = macro_phi
        else:
            self.phi = PyPhiFloat(macro_phi)

        if isinstance(micro_phi, DistanceResult):
            self.micro_phi = micro_phi
        else:
            self.micro_phi = PyPhiFloat(micro_phi)

        self.time_scale = time_scale
        self.coarse_grain = coarse_grain
        self.blackbox = blackbox

    def __str__(self):
        return f"MacroNetwork(phi={self.phi}, emergence={self.emergence})"

    @property
    def emergence(self):
        """Difference between the |big_phi| of the macro and micro systems"""
        return round(self.phi - self.micro_phi, config.numerics.precision)


def coarse_graining(substrate, state, internal_indices):
    """Find the maximal coarse-graining of a micro-system.

    Args:
        substrate (Substrate): The substrate in question.
        state (tuple[int]): The state of the substrate.
        internal_indices (tuple[int]): Nodes in the micro-system.

    Returns:
        tuple[int, CoarseGrain]: The phi-value of the maximal |CoarseGrain|.
    """
    max_phi = float("-inf")
    max_coarse_grain = CoarseGrain((), ())

    for coarse_grain in all_coarse_grains(internal_indices):
        try:
            system = MacroSystem(
                substrate, state, internal_indices, coarse_grain=coarse_grain
            )
        except ConditionallyDependentError:
            continue

        phi = _iit3.phi(system)  # type: ignore[arg-type]  # P7b
        if (phi - max_phi) > 10 ** (-config.numerics.precision):
            max_phi = phi
            max_coarse_grain = coarse_grain

    return (max_phi, max_coarse_grain)


# TODO: refactor this
def all_macro_systems(
    substrate, state, do_blackbox=False, do_coarse_grain=False, time_scales=None
):
    """Generator over all possible macro-systems for the substrate."""
    if time_scales is None:
        time_scales = [1]

    def blackboxes(system):
        # Returns all blackboxes to evaluate
        if not do_blackbox:
            return [None]
        return all_blackboxes(system)

    def coarse_grains(blackbox, system):
        # Returns all coarse-grains to test
        if not do_coarse_grain:
            return [None]
        if blackbox is None:
            return all_coarse_grains(system)
        return all_coarse_grains_for_blackbox(blackbox)

    # TODO? don't consider the empty set here
    # (pass `nonempty=True` to `powerset`)
    for system in utils.powerset(substrate.node_indices):
        for time_scale in time_scales:
            for blackbox in blackboxes(system):
                for coarse_grain in coarse_grains(blackbox, system):
                    try:
                        yield MacroSystem(
                            substrate,
                            state,
                            system,
                            time_scale=time_scale,
                            blackbox=blackbox,
                            coarse_grain=coarse_grain,
                        )
                    except (StateUnreachableError, ConditionallyDependentError):
                        continue


def emergence(
    substrate, state, do_blackbox=False, do_coarse_grain=True, time_scales=None
):
    """Check for the emergence of a micro-system into a macro-system.

    Checks all possible blackboxings and coarse-grainings of a system to find
    the spatial scale with maximum integrated information.

    Use the ``do_blackbox`` and ``do_coarse_grain`` args to specifiy whether to
    use blackboxing, coarse-graining, or both. The default is to just
    coarse-grain the system.

    Args:
        substrate (Substrate): The substrate of the micro-system under investigation.
        state (tuple[int]): The state of the substrate.
        do_blackbox (bool): Set to ``True`` to enable blackboxing. Defaults to
            ``False``.
        do_coarse_grain (bool): Set to ``True`` to enable coarse-graining.
            Defaults to ``True``.
        time_scales (list[int]): List of all time steps over which to check
            for emergence.

    Returns:
        MacroNetwork: The maximal macro-system generated from the
        micro-system.
    """
    micro_phi = substrate.maximal_complex(state).phi

    max_phi = float("-inf")
    max_substrate = None

    for system in all_macro_systems(
        substrate,
        state,
        do_blackbox=do_blackbox,
        do_coarse_grain=do_coarse_grain,
        time_scales=time_scales,
    ):
        phi = _iit3.phi(system)  # type: ignore[arg-type]  # P7b

        if (phi - max_phi) > 10 ** (-config.numerics.precision):
            max_phi = phi
            max_substrate = MacroNetwork(
                substrate=substrate,
                macro_phi=phi,
                micro_phi=micro_phi,
                system=system.micro_node_indices,
                time_scale=system.time_scale,
                blackbox=system.blackbox,
                coarse_grain=system.coarse_grain,
            )

    return max_substrate


# TODO refactor; return a proper model; remove?
def phi_by_grain(substrate, state):
    # pylint: disable=missing-docstring
    list_of_phi = []

    systems = utils.powerset(substrate.node_indices, nonempty=True)
    for system in systems:
        micro_system = System(substrate, state, system)
        phi = _iit3.phi(micro_system)  # type: ignore[arg-type]  # P7b
        list_of_phi.append([len(micro_system), phi, system, None])

        for coarse_grain in all_coarse_grains(system):
            try:
                system = MacroSystem(substrate, state, system, coarse_grain=coarse_grain)
            except ConditionallyDependentError:
                continue

            phi = _iit3.phi(system)  # type: ignore[arg-type]  # P7b
            list_of_phi.append([len(system), phi, system, coarse_grain])
    return list_of_phi


# TODO write tests
# TODO? give example of doing it for a bunch of coarse-grains in docstring
# (make all groupings and partitions, make_substrate for each of them, etc.)
def effective_info(substrate):
    """Return the effective information of the given substrate.

    .. note::
        For details, see:

        Hoel, Erik P., Larissa Albantakis, and Giulio Tononi.
        “Quantifying causal emergence shows that macro can beat micro.”
        Proceedings of the
        National Academy of Sciences 110.49 (2013): 19790-19795.

        Available online: `doi: 10.1073/pnas.1314922110
        <http://www.pnas.org/content/110/49/19790.abstract>`_.
    """
    validate.is_substrate(substrate)

    sbs_tpm = convert.state_by_node2state_by_state(substrate.tpm.tpm)
    avg_repertoire = np.mean(sbs_tpm, 0)

    return np.mean([entropy(repertoire, avg_repertoire, 2.0) for repertoire in sbs_tpm])
