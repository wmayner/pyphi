#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# subsystem.py

"""Represents a candidate system for |small_phi| and |big_phi| evaluation."""

import functools
import logging
from typing import Iterable, Tuple

import numpy as np
from numpy.typing import ArrayLike

from . import cache, conf, connectivity, distribution, metrics
from . import repertoire as _repertoire
from . import resolve_ties, utils, validate
from .compute.parallel import MapReduce
from .conf import config, fallback
from .data_structures import FrozenMap
from .direction import Direction
from .distribution import max_entropy_distribution, repertoire_shape
from .metrics.distribution import repertoire_distance as _repertoire_distance
from .models import (
    Concept,
    MaximallyIrreducibleCause,
    MaximallyIrreducibleEffect,
    NullCut,
    RepertoireIrreducibilityAnalysis,
    _null_ria,
)
from .models.mechanism import ShortCircuitConditions, StateSpecification
from .network import irreducible_purviews
from .node import generate_nodes
from .partition import mip_partitions
from .repertoire import forward_repertoire, unconstrained_forward_repertoire
from .tpm import backward_tpm as _backward_tpm
from .utils import state_of

log = logging.getLogger(__name__)


class Subsystem:
    """A set of nodes in a network.

    Args:
        network (Network): The network the subsystem belongs to.
        state (tuple[int]): The state of the network.

    Keyword Args:
        nodes (tuple[int] or tuple[str]): The nodes of the network which are in
            this subsystem. Nodes can be specified either as indices or as
            labels if the |Network| was passed ``node_labels``. If this is
            ``None`` then the full network will be used.
        cut (Cut): The unidirectional |Cut| to apply to this subsystem.

    Attributes:
        network (Network): The network the subsystem belongs to.
        tpm (pyphi.tpm.ExplicitTPM): The TPM conditioned on the state
            of the external nodes.
        cm (np.ndarray): The connectivity matrix after applying the cut.
        state (tuple[int]): The state of the network.
        node_indices (tuple[int]): The indices of the nodes in the subsystem.
        cut (Cut): The cut that has been applied to this subsystem. Defaults to
            the null cut.
    """

    def __init__(
        self,
        network,
        state,
        nodes=None,
        cut=None,
        # TODO(4.0): refactor repertoire caches
        repertoire_cache=None,
        single_node_repertoire_cache=None,
        forward_repertoire_cache=None,
        unconstrained_forward_repertoire_cache=None,
        backward_tpm=False,
        _external_indices=None,
    ):
        # The network this subsystem belongs to.
        validate.is_network(network)
        network._tpm = network.tpm
        self.network = network

        self.node_labels = network.node_labels
        # Remove duplicates, sort, and ensure native Python `int`s
        # (for JSON serialization).
        self.node_indices = self.node_labels.coerce_to_indices(nodes)

        validate.state_length(state, self.network.size)

        # The state of the network.
        self.state = tuple(state)

        # Get the external node indices.
        # TODO: don't expose this as an attribute?
        if _external_indices is None:
            self.external_indices = tuple(
                set(network.node_indices) - set(self.node_indices)
            )
        else:
            self.external_indices = _external_indices

        # Get the TPM conditioned on the state of the external nodes.
        external_state = utils.state_of(self.external_indices, self.state)
        background_conditions = dict(zip(self.external_indices, external_state))
        self.backward_tpm = backward_tpm
        if self.backward_tpm:
            self.tpm = _backward_tpm(self.network.tpm, state, self.node_indices)
        else:
            self.tpm = self.network.tpm.condition_tpm(background_conditions)
        # The TPM for just the nodes in the subsystem.
        self.proper_tpm = self.tpm.squeeze()[..., list(self.node_indices)]

        # The unidirectional cut applied for phi evaluation
        self.cut = (
            cut if cut is not None else NullCut(self.node_indices, self.node_labels)
        )

        # The network's connectivity matrix with cut applied
        self.cm = self.cut.apply_cut(network.cm)
        # The subsystem's connectivity matrix with the cut applied
        self.proper_cm = connectivity.subadjacency(self.cm, self.node_indices)

        # Cause & effect repertoire caches
        # TODO: if repertoire caches are never reused, there's no reason to
        # have an accesible object-level cache. Just use a simple memoizer
        self._single_node_repertoire_cache = (
            single_node_repertoire_cache or cache.DictCache()
        )
        self._repertoire_cache = repertoire_cache or cache.DictCache()
        self._forward_repertoire_cache = forward_repertoire_cache or cache.DictCache()
        self._unconstrained_forward_repertoire_cache = (
            unconstrained_forward_repertoire_cache or cache.DictCache()
        )

        self.nodes = generate_nodes(
            self.tpm, self.cm, self.state, self.node_indices, self.node_labels
        )

        validate.subsystem(self)

    @property
    def nodes(self):
        """tuple[Node]: The nodes in this |Subsystem|."""
        return self._nodes

    @nodes.setter
    def nodes(self, value):
        """Remap indices to nodes whenever nodes are changed, e.g. in the
        `macro` module.
        """
        # pylint: disable=attribute-defined-outside-init
        self._nodes = value
        self._index2node = {node.index: node for node in self._nodes}

    @property
    def proper_state(self):
        """tuple[int]: The state of the subsystem.

        ``proper_state[i]`` gives the state of the |ith| node **in the
        subsystem**. Note that this is **not** the state of ``nodes[i]``.
        """
        return utils.state_of(self.node_indices, self.state)

    @property
    def connectivity_matrix(self):
        """np.ndarray: Alias for |Subsystem.cm|."""
        return self.cm

    @property
    def size(self):
        """int: The number of nodes in the subsystem."""
        return len(self.node_indices)

    @property
    def is_cut(self):
        """bool: ``True`` if this Subsystem has a cut applied to it."""
        return not self.cut.is_null

    @property
    def cut_indices(self):
        """tuple[int]: The nodes of this subsystem to cut for |big_phi|
        computations.

        This was added to support ``MacroSubsystem``, which cuts indices other
        than ``node_indices``.

        Yields:
            tuple[int]
        """
        return self.node_indices

    @property
    def cut_mechanisms(self):
        """list[tuple[int]]: The mechanisms that are cut in this system."""
        return self.cut.all_cut_mechanisms()

    @property
    def cut_node_labels(self):
        """``NodeLabels``: Labels for the nodes of this system that will be
        cut.
        """
        return self.node_labels

    @property
    def tpm_size(self):
        """int: The number of nodes in the TPM."""
        return self.tpm.shape[-1]

    def cache_info(self):
        """Report repertoire cache statistics."""
        return {
            "single_node_repertoire": self._single_node_repertoire_cache.info(),
            "repertoire": self._repertoire_cache.info(),
        }

    def clear_caches(self):
        """Clear the mice and repertoire caches."""
        self._single_node_repertoire_cache.clear()
        self._repertoire_cache.clear()

    def __repr__(self):
        return "Subsystem(" + ", ".join(map(repr, self.nodes)) + ")"

    def __str__(self):
        return repr(self)

    def __bool__(self):
        """Return ``False`` if the Subsystem has no nodes, ``True``
        otherwise.
        """
        return bool(self.nodes)

    def __eq__(self, other):
        """Return whether this Subsystem is equal to the other object.

        Two Subsystems are equal if their sets of nodes, networks, and cuts are
        equal.
        """
        if not isinstance(other, Subsystem):
            return False

        return (
            set(self.node_indices) == set(other.node_indices)
            and self.state == other.state
            and self.network == other.network
            and self.cut == other.cut
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        """Return whether this subsystem has fewer nodes than the other."""
        return len(self.nodes) < len(other.nodes)

    def __gt__(self, other):
        """Return whether this subsystem has more nodes than the other."""
        return len(self.nodes) > len(other.nodes)

    def __le__(self, other):
        return len(self.nodes) <= len(other.nodes)

    def __ge__(self, other):
        return len(self.nodes) >= len(other.nodes)

    def __len__(self):
        """Return the number of nodes in this Subsystem."""
        return len(self.node_indices)

    def __hash__(self):
        return hash((self.network, self.node_indices, self.state, self.cut))

    def to_json(self):
        """Return a JSON-serializable representation."""
        return {
            "network": self.network,
            "state": self.state,
            "nodes": self.node_indices,
            "cut": self.cut,
        }

    def apply_cut(self, cut):
        """Return a cut version of this |Subsystem|.

        Args:
            cut (Cut): The cut to apply to this |Subsystem|.

        Returns:
            Subsystem: The cut subsystem.
        """
        return Subsystem(
            self.network,
            self.state,
            self.node_indices,
            cut=cut,
            backward_tpm=self.backward_tpm,
        )

    def indices2nodes(self, indices):
        """Return |Nodes| for these indices.

        Args:
            indices (tuple[int]): The indices in question.

        Returns:
            tuple[Node]: The |Node| objects corresponding to these indices.

        Raises:
            ValueError: If requested indices are not in the subsystem.
        """
        if set(indices) - set(self.node_indices):
            raise ValueError("`indices` must be a subset of the Subsystem's indices.")
        return tuple(self._index2node[n] for n in indices)

    # TODO extend to nonbinary nodes
    @cache.method("_single_node_repertoire_cache", Direction.CAUSE)
    def _single_node_cause_repertoire(self, mechanism_node_index, purview):
        # pylint: disable=missing-docstring
        mechanism_node = self._index2node[mechanism_node_index]
        # We're conditioning on this node's state, so take the TPM for the node
        # being in that state.
        tpm = mechanism_node.tpm[..., mechanism_node.state]
        # Marginalize-out all parents of this mechanism node that aren't in the
        # purview.
        return tpm.marginalize_out((mechanism_node.inputs - purview)).tpm

    # TODO extend to nonbinary nodes
    @cache.method("_repertoire_cache", Direction.CAUSE)
    def _cause_repertoire(self, mechanism, purview):
        # Use a frozenset so the arguments to `_single_node_cause_repertoire`
        # can be hashed and cached.
        purview = frozenset(purview)
        # Preallocate the repertoire with the proper shape, so that
        # probabilities are broadcasted appropriately.
        joint = np.ones(repertoire_shape(self.network.node_indices, purview))
        # The cause repertoire is the product of the cause repertoires of the
        # individual nodes.
        joint *= functools.reduce(
            np.multiply,
            [self._single_node_cause_repertoire(m, purview) for m in mechanism],
        )
        # The resulting joint distribution is over previous states, which are
        # rows in the TPM, so the distribution is a column. The columns of a
        # TPM don't necessarily sum to 1, so we normalize.
        return distribution.normalize(joint)

    def cause_repertoire(self, mechanism, purview, **kwargs):
        """Return the cause repertoire of a mechanism over a purview.

        Args:
            mechanism (tuple[int]): The mechanism for which to calculate the
                cause repertoire.
            purview (tuple[int]): The purview over which to calculate the
                cause repertoire.

        Returns:
            np.ndarray: The cause repertoire of the mechanism over the purview.

        .. note::
            The returned repertoire is a distribution over purview node states,
            not the states of the whole network.
        """
        # If the purview is empty, the distribution is empty; return the
        # multiplicative identity.
        if not purview:
            return np.array([1.0])
        # If the mechanism is empty, nothing is specified about the previous
        # state of the purview; return the purview's maximum entropy
        # distribution.
        if not mechanism:
            return max_entropy_distribution(self.node_indices, purview)
        # Drop kwargs
        return self._cause_repertoire(mechanism, purview)

    # TODO extend to nonbinary nodes
    @cache.method("_single_node_repertoire_cache", Direction.EFFECT)
    def _single_node_effect_repertoire(
        self,
        condition: FrozenMap[int, int],
        purview_node_index: int,
    ):
        # pylint: disable=missing-docstring
        purview_node = self._index2node[purview_node_index]
        # Condition on the state of the purview inputs that are in the mechanism
        purview_node.tpm = purview_node.tpm
        tpm = purview_node.tpm.condition_tpm(condition)
        # TODO(4.0) remove reference to TPM
        # Marginalize-out the inputs that aren't in the mechanism.
        nonmechanism_inputs = purview_node.inputs - set(condition)
        tpm = tpm.marginalize_out(nonmechanism_inputs)
        # Reshape so that the distribution is over next states.
        return tpm.reshape(
            repertoire_shape(self.network.node_indices, (purview_node_index,))
        ).tpm

    @cache.method("_repertoire_cache", Direction.EFFECT)
    def _effect_repertoire(
        self,
        condition: FrozenMap[int, int],
        purview: Tuple[int],
    ):
        # Preallocate the repertoire with the proper shape, so that
        # probabilities are broadcasted appropriately.
        joint = np.ones(repertoire_shape(self.network.node_indices, purview))
        # The effect repertoire is the product of the effect repertoires of the
        # individual nodes.
        # TODO(tpm) Currently the single-node repertoires need to be bare numpy
        # arrays here because reducing with np.multiply throws an error; this
        # should be fixed
        return joint * functools.reduce(
            np.multiply,
            [self._single_node_effect_repertoire(condition, p) for p in purview],
        )

    def effect_repertoire(self, mechanism, purview, mechanism_state=None):
        """Return the effect repertoire of a mechanism over a purview.

        Args:
            mechanism (tuple[int]): The mechanism for which to calculate the
                effect repertoire.
            purview (tuple[int]): The purview over which to calculate the
                effect repertoire.

        Returns:
            np.ndarray: The effect repertoire of the mechanism over the
            purview.

        .. note::
            The returned repertoire is a distribution over purview node states,
            not the states of the whole network.
        """
        if not purview:
            # If the purview is empty, the distribution is empty, so return the
            # multiplicative identity.
            return np.array([1.0])
        if mechanism_state is None:
            mechanism_state = utils.state_of(mechanism, self.state)
        condition = FrozenMap(zip(mechanism, mechanism_state))
        return self._effect_repertoire(condition, purview)

    def repertoire(self, direction, mechanism, purview, **kwargs):
        """Return the cause or effect repertoire based on a direction.

        Args:
            direction (Direction): |CAUSE| or |EFFECT|.
            mechanism (tuple[int]): The mechanism for which to calculate the
                repertoire.
            purview (tuple[int]): The purview over which to calculate the
                repertoire.

        Returns:
            np.ndarray: The cause or effect repertoire of the mechanism over
            the purview.

        Raises:
            ValueError: If ``direction`` is invalid.
        """
        if direction == Direction.CAUSE:
            return self.cause_repertoire(mechanism, purview, **kwargs)
        elif direction == Direction.EFFECT:
            return self.effect_repertoire(mechanism, purview, **kwargs)
        return validate.direction(direction)

    def unconstrained_repertoire(self, direction, purview, **kwargs):
        """Return the unconstrained cause/effect repertoire over a purview."""
        return self.repertoire(direction, (), purview, **kwargs)

    def unconstrained_cause_repertoire(self, purview, **kwargs):
        """Return the unconstrained cause repertoire for a purview.

        This is just the cause repertoire in the absence of any mechanism.
        """
        return self.unconstrained_repertoire(Direction.CAUSE, purview, **kwargs)

    def unconstrained_effect_repertoire(self, purview, **kwargs):
        """Return the unconstrained effect repertoire for a purview.

        This is just the effect repertoire in the absence of any mechanism.
        """
        return self.unconstrained_repertoire(Direction.EFFECT, purview, **kwargs)

    def partitioned_repertoire(
        self, direction, partition, repertoire_distance=None, **kwargs
    ):
        """Compute the repertoire of a partitioned mechanism and purview."""
        repertoire_distance = fallback(repertoire_distance, config.REPERTOIRE_DISTANCE)
        if repertoire_distance == "GENERALIZED_INTRINSIC_DIFFERENCE":
            if "state" not in kwargs:
                raise ValueError(
                    "must provide purview state for generalized intrinsic difference"
                )
            purview_state = kwargs.pop("state")
            prs = [
                self.forward_probability(
                    direction,
                    part.mechanism,
                    part.purview,
                    purview_state=utils.substate(
                        partition.purview, purview_state, part.purview
                    ),
                    **kwargs,
                )
                for part in partition
            ]
            return np.prod(prs)
        else:
            repertoires = [
                self.repertoire(direction, part.mechanism, part.purview, **kwargs)
                for part in partition
            ]
        return functools.reduce(np.multiply, repertoires)

    def forward_probability(
        self,
        direction: Direction,
        mechanism: Tuple[int],
        purview: Tuple[int],
        purview_state: Tuple[int],
        **kwargs,
    ) -> float:
        if direction == Direction.CAUSE:
            return self.forward_cause_probability(
                mechanism, purview, purview_state, **kwargs
            )
        elif direction == Direction.EFFECT:
            return self.forward_effect_probability(
                mechanism, purview, purview_state, **kwargs
            )
        return validate.direction(direction)

    def forward_effect_probability(
        self,
        mechanism: Tuple[int],
        purview: Tuple[int],
        purview_state: Tuple[int],
        **kwargs,
    ) -> float:
        return _repertoire.forward_effect_probability(
            self, mechanism, purview, purview_state, **kwargs
        )

    def forward_cause_probability(
        self,
        mechanism: Tuple[int],
        purview: Tuple[int],
        purview_state: Tuple[int],
        **kwargs,
    ) -> float:
        return _repertoire.forward_cause_probability(
            self, mechanism, purview, purview_state, **kwargs
        )

    def forward_repertoire(
        self, direction: Direction, mechanism: Tuple[int], purview: Tuple[int], **kwargs
    ) -> ArrayLike:
        if direction == Direction.CAUSE:
            return self.forward_cause_repertoire(mechanism, purview)
        elif direction == Direction.EFFECT:
            return self.forward_effect_repertoire(mechanism, purview, **kwargs)
        return validate.direction(direction)

    @cache.method("_forward_repertoire_cache", Direction.CAUSE)
    def forward_cause_repertoire(
        self, mechanism: Tuple[int], purview: Tuple[int]
    ) -> ArrayLike:
        return _repertoire.forward_cause_repertoire(self, mechanism, purview)

    # NOTE: No caching is required here because the forward effect repertoire is
    # the same as the effect repertoire.
    def forward_effect_repertoire(
        self, mechanism: Tuple[int], purview: Tuple[int], **kwargs
    ) -> ArrayLike:
        return _repertoire.forward_effect_repertoire(self, mechanism, purview, **kwargs)

    def unconstrained_forward_repertoire(
        self, direction: Direction, mechanism: Tuple[int], purview: Tuple[int]
    ) -> ArrayLike:
        if direction == Direction.CAUSE:
            return self.unconstrained_forward_cause_repertoire(mechanism, purview)
        elif direction == Direction.EFFECT:
            return self.unconstrained_forward_effect_repertoire(mechanism, purview)
        return validate.direction(direction)

    @cache.method("_unconstrained_forward_repertoire_cache", Direction.EFFECT)
    def unconstrained_forward_effect_repertoire(
        self, mechanism: Tuple[int], purview: Tuple[int]
    ) -> ArrayLike:
        return _repertoire.unconstrained_forward_effect_repertoire(
            self, mechanism, purview
        )

    @cache.method("_unconstrained_forward_repertoire_cache", Direction.CAUSE)
    def unconstrained_forward_cause_repertoire(
        self, mechanism: Tuple[int], purview: Tuple[int]
    ) -> ArrayLike:
        return _repertoire.unconstrained_forward_cause_repertoire(
            self, mechanism, purview
        )

    def expand_repertoire(self, direction, repertoire, new_purview=None):
        """Distribute an effect repertoire over a larger purview.

        Args:
            direction (Direction): |CAUSE| or |EFFECT|.
            repertoire (np.ndarray): The repertoire to expand.

        Keyword Args:
            new_purview (tuple[int]): The new purview to expand the repertoire
                over. If ``None`` (the default), the new purview is the entire
                network.

        Returns:
            np.ndarray: A distribution over the new purview, where probability
            is spread out over the new nodes.

        Raises:
            ValueError: If the expanded purview doesn't contain the original
                purview.
        """
        if repertoire is None:
            return None

        purview = distribution.purview(repertoire)

        if new_purview is None:
            new_purview = self.node_indices  # full subsystem

        if not set(purview).issubset(new_purview):
            raise ValueError("Expanded purview must contain original purview.")

        # Get the unconstrained repertoire over the other nodes in the network.
        non_purview_indices = tuple(set(new_purview) - set(purview))
        uc = self.unconstrained_repertoire(direction, non_purview_indices)
        # Multiply the given repertoire by the unconstrained one to get a
        # distribution over all the nodes in the network.
        expanded_repertoire = repertoire * uc

        return distribution.normalize(expanded_repertoire)

    def expand_cause_repertoire(self, repertoire, new_purview=None):
        """Alias for |expand_repertoire()| with ``direction`` set to |CAUSE|."""
        return self.expand_repertoire(Direction.CAUSE, repertoire, new_purview)

    def expand_effect_repertoire(self, repertoire, new_purview=None):
        """Alias for |expand_repertoire()| with ``direction`` set to |EFFECT|."""
        return self.expand_repertoire(Direction.EFFECT, repertoire, new_purview)

    def cause_info(self, mechanism, purview, **kwargs):
        """Return the cause information for a mechanism over a purview."""
        return _repertoire_distance(
            self.cause_repertoire(mechanism, purview),
            self.unconstrained_cause_repertoire(purview),
            direction=Direction.CAUSE,
            **kwargs,
        )

    def effect_info(self, mechanism, purview, **kwargs):
        """Return the effect information for a mechanism over a purview."""
        return _repertoire_distance(
            self.effect_repertoire(mechanism, purview),
            self.unconstrained_effect_repertoire(purview),
            direction=Direction.EFFECT,
            **kwargs,
        )

    def cause_effect_info(self, mechanism, purview, **kwargs):
        """Return the cause-effect information for a mechanism over a purview.

        This is the minimum of the cause and effect information.
        """
        return min(
            self.cause_info(mechanism, purview, **kwargs),
            self.effect_info(mechanism, purview, **kwargs),
        )

    # MIP methods
    # =========================================================================

    def evaluate_partition(
        self,
        direction,
        mechanism,
        purview,
        partition,
        repertoire=None,
        partitioned_repertoire=None,
        repertoire_distance=None,
        partitioned_repertoire_kwargs=None,
        **kwargs,
    ):
        """Return the |small_phi| of a mechanism over a purview for the given
        partition.

        Args:
            direction (Direction): |CAUSE| or |EFFECT|.
            mechanism (tuple[int]): The nodes in the mechanism.
            purview (tuple[int]): The nodes in the purview.
            partition (Bipartition): The partition to evaluate.

        Keyword Args:
            repertoire (np.array): The unpartitioned repertoire.
                If not supplied, it will be computed.

        Returns:
            tuple[int, np.ndarray]: The distance between the unpartitioned and
            partitioned repertoires, and the partitioned repertoire.
        """
        repertoire_distance = fallback(repertoire_distance, config.REPERTOIRE_DISTANCE)
        # TODO(4.0) refactor
        # TODO(4.0) consolidate logic with system level partitions
        if repertoire is None:
            repertoire = self.repertoire(direction, mechanism, purview)
        # TODO(4.0) use same partitioned_repertoire func
        if repertoire_distance == "GENERALIZED_INTRINSIC_DIFFERENCE":
            purview_state = kwargs["state"].state
            selectivity = repertoire.squeeze()[purview_state]
            forward_pr = self.forward_probability(
                direction, mechanism, purview, purview_state
            )
            if partitioned_repertoire is None:
                partitioned_pr = self.partitioned_repertoire(
                    direction, partition, state=purview_state
                )
            else:
                partitioned_pr = partitioned_repertoire
            phi = metrics.distribution.generalized_intrinsic_difference(
                forward_repertoire=forward_pr,
                partitioned_forward_repertoire=partitioned_pr,
                selectivity_repertoire=selectivity,
            )
            repertoire = forward_pr
            # TODO(4.0) refactor
            partitioned_repertoire = partitioned_pr
        else:
            if partitioned_repertoire is None:
                partitioned_repertoire_kwargs = partitioned_repertoire_kwargs or dict()
                partitioned_repertoire = self.partitioned_repertoire(
                    direction, partition, **partitioned_repertoire_kwargs
                )
            phi = _repertoire_distance(
                repertoire,
                partitioned_repertoire,
                direction=direction,
                repertoire_distance=repertoire_distance,
                **kwargs,
            )
            selectivity = None
        return RepertoireIrreducibilityAnalysis(
            phi=phi,
            direction=direction,
            mechanism=mechanism,
            purview=purview,
            partition=partition,
            repertoire=repertoire,
            partitioned_repertoire=partitioned_repertoire,
            mechanism_state=state_of(mechanism, self.state),
            purview_state=state_of(purview, self.state),
            # TODO(4.0) refactor
            specified_state=kwargs.get("state"),
            node_labels=self.node_labels,
            selectivity=selectivity,
        )

    def _find_mip_single_state(
        self,
        specified_state,
        direction,
        mechanism,
        purview,
        repertoire,
        partitions,
        parallel_kwargs,
        **kwargs,
    ):
        # TODO(4.0) allow IIT 3.0 calculations
        partitions = fallback(
            partitions, mip_partitions(mechanism, purview, self.node_labels)
        )

        # TODO(ties) refactor: make partition the first positional arg and use
        # map_kwargs in MapReduce
        def _evaluate_partition(partition):
            return self.evaluate_partition(
                direction,
                mechanism,
                purview,
                partition,
                repertoire=repertoire,
                state=specified_state,
                **kwargs,
            )

        candidate_mips = MapReduce(
            _evaluate_partition,
            partitions,
            shortcircuit_func=utils.is_falsy,
            desc="Evaluating mechanism partitions",
            **parallel_kwargs,
        ).run()

        ties = tuple(
            resolve_ties.partitions(
                candidate_mips,
                default=_null_ria(
                    direction,
                    mechanism,
                    purview,
                    phi=0,
                    specified_state=specified_state,
                ),
            )
        )
        for tie in ties:
            # TODO(ties) do this assignment in resolve_ties
            tie.set_partition_ties(ties)
        return ties[0]

    def find_mip(
        self, direction, mechanism, purview, partitions=None, state=None, **kwargs
    ):
        """Return the minimum information partition for a mechanism over a
        purview.

        Args:
            direction (Direction): |CAUSE| or |EFFECT|.
            mechanism (tuple[int]): The nodes in the mechanism.
            purview (tuple[int]): The nodes in the purview.

        Keyword Args:
            **kwargs: MapReduce kwargs control parallelization; others are
                passed to |evaluate_partition|.

        Returns:
            RepertoireIrreducibilityAnalysis: The irreducibility analysis for
            the mininum-information partition in one temporal direction.

        """

        def null_mip(**kwargs):
            return _null_ria(direction, mechanism, purview, specified_state=state)

        if not purview:
            return null_mip(reasons=(ShortCircuitConditions.EMPTY_PURVIEW,))

        # Calculate the unpartitioned repertoire to compare against the
        # partitioned ones.
        repertoire = self.repertoire(direction, mechanism, purview)

        # State is unreachable - return 0 instead of giving nonsense results
        # TODO(4.0) re-evaluate this with the GID
        if direction == Direction.CAUSE and np.all(repertoire == 0):
            return null_mip(reasons=(ShortCircuitConditions.UNREACHABLE_STATE,))

        if partitions is not None:
            # NOTE: Must convert to list to allow for multiple iterations in
            # case of tied states
            partitions = list(partitions)

        parallel_kwargs = conf.parallel_kwargs(
            config.PARALLEL_MECHANISM_PARTITION_EVALUATION, **kwargs
        )
        if config.IIT_VERSION == 4:
            if state is None:
                specified_states = self.intrinsic_information(
                    direction, mechanism, purview
                ).ties
            else:
                specified_states = [state]

            mips = MapReduce(
                self._find_mip_single_state,
                specified_states,
                map_kwargs=dict(
                    direction=direction,
                    mechanism=mechanism,
                    purview=purview,
                    repertoire=repertoire,
                    partitions=partitions,
                    parallel_kwargs=parallel_kwargs,
                ),
                desc="Finding MIP for maximum intrinsic information states",
                **parallel_kwargs,
            ).run()
        elif config.IIT_VERSION == 3:
            if state is not None:
                raise ValueError("passing `state` is not supported with IIT 3.0")
            return self._find_mip_single_state(
                None,
                direction,
                mechanism,
                purview,
                repertoire,
                partitions,
                parallel_kwargs,
            )
        else:
            raise NotImplementedError

        ties = tuple(resolve_ties.states(mips))
        for tie in ties:
            tie.set_state_ties(ties)
        return ties[0]

    def cause_mip(self, mechanism, purview, **kwargs):
        """Return the irreducibility analysis for the cause MIP.

        Alias for |find_mip()| with ``direction`` set to |CAUSE|.
        """
        return self.find_mip(Direction.CAUSE, mechanism, purview, **kwargs)

    def effect_mip(self, mechanism, purview, **kwargs):
        """Return the irreducibility analysis for the effect MIP.

        Alias for |find_mip()| with ``direction`` set to |EFFECT|.
        """
        return self.find_mip(Direction.EFFECT, mechanism, purview, **kwargs)

    def phi_cause_mip(self, mechanism, purview, **kwargs):
        """Return the |small_phi| of the cause MIP.

        This is the distance between the unpartitioned cause repertoire and the
        MIP cause repertoire.
        """
        mip = self.cause_mip(mechanism, purview, **kwargs)
        return mip.phi if mip else 0

    def phi_effect_mip(self, mechanism, purview, **kwargs):
        """Return the |small_phi| of the effect MIP.

        This is the distance between the unpartitioned effect repertoire and
        the MIP cause repertoire.
        """
        mip = self.effect_mip(mechanism, purview, **kwargs)
        return mip.phi if mip else 0

    def phi(self, mechanism, purview, **kwargs):
        """Return the |small_phi| of a mechanism over a purview."""
        return min(
            self.phi_cause_mip(mechanism, purview, **kwargs),
            self.phi_effect_mip(mechanism, purview, **kwargs),
        )

    # Maximal state methods
    # =========================================================================

    def intrinsic_information(
        self,
        direction: Direction,
        mechanism: Tuple[int],
        purview: Tuple[int],
        repertoire_distance: str = None,
        states: Iterable[Iterable[int]] = None,
    ):
        repertoire_distance = fallback(
            repertoire_distance, config.REPERTOIRE_DISTANCE_INFORMATION
        )
        if states is None:
            states = utils.all_states(len(purview))

        # TODO(4.0) refactor for consistent API across metrics
        if repertoire_distance == "GENERALIZED_INTRINSIC_DIFFERENCE":
            # TODO(4.0) include selectivity_repertoire in StateSpecification
            selectivity_repertoire = self.repertoire(
                direction,
                mechanism,
                purview,
            )
            repertoire = forward_repertoire(
                direction,
                self,
                mechanism,
                purview,
            )
            unconstrained_repertoire = unconstrained_forward_repertoire(
                direction,
                self,
                mechanism,
                purview,
            )
            gid = metrics.distribution.generalized_intrinsic_difference(
                repertoire,
                unconstrained_repertoire,
                selectivity_repertoire,
            )
            # Remove singleton dimensions since we'll index with purview state
            gid = gid.squeeze()

            def evaluate_state(state):
                return gid[state]

        else:
            repertoire = self.repertoire(
                direction,
                mechanism,
                purview,
            )
            unconstrained_repertoire = self.unconstrained_repertoire(
                direction,
                purview,
            )

            def evaluate_state(state):
                return _repertoire_distance(
                    repertoire, unconstrained_repertoire, state=state
                )

        # TODO(4.0): compute arraywise once, then find max; requires refactoring state kwarg to metrics
        # TODO(ties): use resolve_ties here
        state_to_information = {state: evaluate_state(state) for state in states}
        max_information = max(state_to_information.values())
        # Return all tied states
        ties = [
            StateSpecification(
                direction=direction,
                purview=purview,
                state=state,
                intrinsic_information=information,
                repertoire=repertoire,
                unconstrained_repertoire=unconstrained_repertoire,
            )
            for state, information in state_to_information.items()
            if information == max_information
        ]
        for tie in ties:
            tie.set_ties(ties)
        return ties[0]

    # Phi_max methods
    # =========================================================================

    def potential_purviews(self, direction, mechanism, purviews=None):
        """Return all purviews that could belong to the |MIC|/|MIE|.

        Filters out trivially-reducible purviews.

        Args:
            direction (Direction): |CAUSE| or |EFFECT|.
            mechanism (tuple[int]): The mechanism of interest.

        Keyword Args:
            purviews (tuple[int]): Optional subset of purviews of interest.
        """
        # TODO(4.0) return set from network.potential_purviews?
        _potential_purviews = set(self.network.potential_purviews(direction, mechanism))
        if purviews is None:
            purviews = _potential_purviews
        else:
            # Restrict to given purviews
            purviews = _potential_purviews & set(purviews)
        # Restrict to purviews within the subsystem
        purviews = [
            purview for purview in purviews if set(purview).issubset(self.node_indices)
        ]
        # Purviews are already filtered in network.potential_purviews
        # over the full network connectivity matrix. However, since the cm
        # is cut/smaller we check again here.
        return irreducible_purviews(self.cm, direction, mechanism, purviews)

    def find_mice(self, direction, mechanism, purviews=None, **kwargs):
        """Return the |MIC| or |MIE| for a mechanism.

        Args:
            direction (Direction): |CAUSE| or |EFFECT|.
            mechanism (tuple[int]): The mechanism to be tested for
                irreducibility.

        Keyword Args:
            purviews (tuple[int]): Optionally restrict the possible purviews
                to a subset of the subsystem. This may be useful for _e.g._
                finding only concepts that are "about" a certain subset of
                nodes.

        Returns:
            MaximallyIrreducibleCauseOrEffect: The |MIC| or |MIE|.
        """
        purviews = self.potential_purviews(direction, mechanism, purviews)

        if direction == Direction.CAUSE:
            mice_class = MaximallyIrreducibleCause
        elif direction == Direction.EFFECT:
            mice_class = MaximallyIrreducibleEffect
        else:
            validate.direction(direction)

        no_purviews = mice_class(
            _null_ria(
                direction,
                mechanism,
                (),
                reasons=(ShortCircuitConditions.NO_PURVIEWS,),
            )
        )

        if not purviews:
            return no_purviews

        # TODO put purview first in signature to avoid
        def _find_mip(purview):
            return self.find_mip(direction, mechanism, purview)

        parallel_kwargs = conf.parallel_kwargs(
            config.PARALLEL_PURVIEW_EVALUATION, **kwargs
        )
        map_reduce = MapReduce(
            _find_mip,
            purviews,
            total=len(purviews),
            desc="Evaluating purviews",
            **parallel_kwargs,
        )

        all_mice = map(mice_class, map_reduce.run())
        # Record purview ties
        ties = tuple(resolve_ties.purviews(all_mice, default=no_purviews))
        # TODO(ties) refactor this into `resolve_ties.purviews`?
        for tie in ties:
            tie.set_purview_ties(ties)
        return ties[0]

    def mic(self, mechanism, purviews=None, **kwargs):
        """Return the mechanism's maximally-irreducible cause (|MIC|).

        Alias for |find_mice()| with ``direction`` set to |CAUSE|.
        """
        return self.find_mice(Direction.CAUSE, mechanism, purviews=purviews, **kwargs)

    def mie(self, mechanism, purviews=None, **kwargs):
        """Return the mechanism's maximally-irreducible effect (|MIE|).

        Alias for |find_mice()| with ``direction`` set to |EFFECT|.
        """
        return self.find_mice(Direction.EFFECT, mechanism, purviews=purviews, **kwargs)

    def phi_max(self, mechanism):
        """Return the |small_phi_max| of a mechanism.

        This is the maximum of |small_phi| taken over all possible purviews.
        """
        return min(self.mic(mechanism).phi, self.mie(mechanism).phi)

    # Big Phi methods
    # =========================================================================

    @property
    def null_concept(self):
        """Return the null concept of this subsystem.

        The null concept is a point in concept space identified with
        the unconstrained cause and effect repertoire of this subsystem.
        """
        # Unconstrained cause repertoire.
        cause_repertoire = self.cause_repertoire((), ())
        # Unconstrained effect repertoire.
        effect_repertoire = self.effect_repertoire((), ())

        # Null cause.
        cause = MaximallyIrreducibleCause(
            _null_ria(Direction.CAUSE, (), (), cause_repertoire)
        )
        # Null effect.
        effect = MaximallyIrreducibleEffect(
            _null_ria(Direction.EFFECT, (), (), effect_repertoire)
        )

        # All together now...
        return Concept(
            mechanism=(),
            cause=cause,
            effect=effect,
        )

    def concept(
        self,
        mechanism,
        purviews=None,
        cause_purviews=None,
        effect_purviews=None,
        **kwargs,
    ):
        """Return the concept specified by a mechanism within this subsytem.

        Args:
            mechanism (tuple[int]): The candidate set of nodes.

        Keyword Args:
            purviews (tuple[tuple[int]]): Restrict the possible purviews to
                those in this list.
            cause_purviews (tuple[tuple[int]]): Restrict the possible cause
                purviews to those in this list. Takes precedence over
                ``purviews``.
            effect_purviews (tuple[tuple[int]]): Restrict the possible effect
                purviews to those in this list. Takes precedence over
                ``purviews``.

        Returns:
            Concept: The pair of maximally irreducible cause/effect repertoires
            that constitute the concept specified by the given mechanism.
        """
        log.debug("Computing concept %s...", mechanism)

        # If the mechanism is empty, there is no concept.
        if not mechanism:
            log.debug("Empty concept; returning null concept")
            return self.null_concept

        cause_purviews = cause_purviews if cause_purviews is not None else purviews
        cause = self.mic(mechanism, purviews=cause_purviews, **kwargs)

        effect_purviews = effect_purviews if effect_purviews is not None else purviews
        effect = self.mie(mechanism, purviews=effect_purviews, **kwargs)

        log.debug("Found concept %s", mechanism)
        # NOTE: Make sure to expand the repertoires to the size of the
        # subsystem when calculating concept distance. For now, they must
        # remain un-expanded so the concept doesn't depend on the subsystem.
        return Concept(
            mechanism=mechanism,
            cause=cause,
            effect=effect,
        )
