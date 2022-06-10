#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# models/mechanism.py

"""Mechanism-level objects."""

import numpy as np

from .. import connectivity, utils, validate
from ..direction import Direction
from ..exceptions import WrongDirectionError
from ..metrics import distribution
from . import cmp, fmt

_ria_attributes = [
    "phi",
    "direction",
    "mechanism",
    "purview",
    "partition",
    "repertoire",
    "partitioned_repertoire",
]


class RepertoireIrreducibilityAnalysis(cmp.Orderable):
    """An analysis of the irreducibility (|small_phi|) of a mechanism over a
    purview, for a given partition, in one temporal direction.

    These can be compared with the built-in Python comparison operators (``<``,
    ``>``, etc.). Comparison is based on |small_phi| value, then mechanism size.
    """

    def __init__(
        self,
        phi,
        direction,
        mechanism,
        purview,
        partition,
        repertoire,
        partitioned_repertoire,
        specified_index=None,
        specified_state=None,
        mechanism_state=None,
        purview_state=None,
        node_labels=None,
    ):
        self._phi = phi
        self._direction = direction
        self._mechanism = mechanism
        self._purview = purview
        self._partition = partition
        self._mechanism_state = mechanism_state
        self._purview_state = purview_state

        def _repertoire(repertoire):
            if repertoire is None:
                return None
            return np.array(repertoire)

        self._repertoire = _repertoire(repertoire)
        self._partitioned_repertoire = _repertoire(partitioned_repertoire)

        # TODO(4.0)
        # - use DistanceResult?
        if self._partitioned_repertoire is None:
            self._specified_index = None
            self._specified_state = None
        else:
            self._specified_index = (
                distribution.specified_index(
                    self.repertoire, self.partitioned_repertoire
                )
                if specified_index is None
                else specified_index
            )
            self._specified_state = (
                distribution.specified_state(
                    self.repertoire, self.partitioned_repertoire
                )
                if specified_state is None
                else specified_state
            )

        # Optional labels - only used to generate nice labeled reprs
        self._node_labels = node_labels

    @property
    def phi(self):
        """float: This is the difference between the mechanism's unpartitioned
        and partitioned repertoires.
        """
        return self._phi

    @property
    def direction(self):
        """Direction: |CAUSE| or |EFFECT|."""
        return self._direction

    @property
    def mechanism(self):
        """tuple[int]: The mechanism that was analyzed."""
        return self._mechanism

    @property
    def mechanism_state(self):
        """tuple[int]: The current state of the mechanism."""
        return self._mechanism_state

    @property
    def purview(self):
        """tuple[int]: The purview over which the the mechanism was
        analyzed.
        """
        return self._purview

    @property
    def purview_state(self):
        """tuple[int]: The current state of the purview."""
        return self._purview_state

    @property
    def partition(self):
        """KPartition: The partition of the mechanism-purview pair that was
        analyzed.
        """
        return self._partition

    @property
    def repertoire(self):
        """np.ndarray: The repertoire of the mechanism over the purview."""
        return self._repertoire

    @property
    def partitioned_repertoire(self):
        """np.ndarray: The partitioned repertoire of the mechanism over the
        purview. This is the product of the repertoires of each part of the
        partition.
        """
        return self._partitioned_repertoire

    @property
    def specified_index(self):
        """The state(s) with the maximal absolute intrinsic difference
        between the unpartitioned and partitioned repertoires."""
        return self._specified_index

    @property
    def specified_state(self):
        """The state(s) with the maximal absolute intrinsic difference
        between the unpartitioned and partitioned repertoires."""
        return self._specified_state

    # TODO(4.0) clean up specified state logic once it stabilizes
    def set_specified_state(self, state):
        self._specified_state = state

    def state_ties(self):
        for index, state in zip(self.specified_index, self.specified_state):
            yield RepertoireIrreducibilityAnalysis(
                self.phi,
                self.direction,
                self.mechanism,
                self.purview,
                self.partition,
                self.repertoire,
                self.partitioned_repertoire,
                specified_index=index.reshape(1, -1),
                specified_state=state.reshape(1, -1),
                mechanism_state=self.mechanism_state,
                purview_state=self.purview_state,
                node_labels=self.node_labels,
            )

    @property
    def node_labels(self):
        """|NodeLabels| for this system."""
        return self._node_labels

    def order_by(self):
        return (self.phi, len(self.mechanism))

    def __eq__(self, other):
        # We don't consider the partition and partitioned repertoire in
        # checking for RIA equality.
        attrs = ["phi", "direction", "mechanism", "purview", "repertoire"]
        return cmp.general_eq(self, other, attrs)

    def __bool__(self):
        """A |RepertoireIrreducibilityAnalysis| is ``True`` if it has
        |small_phi > 0|.
        """
        return not utils.eq(self.phi, 0)

    def __hash__(self):
        return hash(
            (
                self.phi,
                self.direction,
                self.mechanism,
                self.purview,
                utils.np_hash(self.repertoire),
            )
        )

    def __repr__(self):
        return fmt.make_repr(self, _ria_attributes)

    def __str__(self):
        return "Repertoire irreducibility analysis\n" + fmt.indent(fmt.fmt_ria(self))

    def to_json(self):
        return {attr: getattr(self, attr) for attr in _ria_attributes}


def _null_ria(direction, mechanism, purview, repertoire=None, phi=0.0):
    """The irreducibility analysis for a reducible mechanism."""
    # TODO Use properties here to infer mechanism and purview from
    # partition yet access them with .mechanism and .partition
    return RepertoireIrreducibilityAnalysis(
        direction=direction,
        mechanism=mechanism,
        purview=purview,
        partition=None,
        repertoire=repertoire,
        partitioned_repertoire=None,
        phi=phi,
    )


# =============================================================================


class MaximallyIrreducibleCauseOrEffect(cmp.Orderable):
    """A maximally irreducible cause or effect (MICE).

    These can be compared with the built-in Python comparison operators (``<``,
    ``>``, etc.). Comparison is based on |small_phi| value, then mechanism size.
    """

    def __init__(self, ria, ties=None):
        self._ria = ria
        self._all_ties = ties

    @property
    def phi(self):
        """float: The difference between the mechanism's unpartitioned and
        partitioned repertoires.
        """
        return self._ria.phi

    @property
    def direction(self):
        """Direction: |CAUSE| or |EFFECT|."""
        return self._ria.direction

    @property
    def mechanism(self):
        """list[int]: The mechanism for which the MICE is evaluated."""
        return self._ria.mechanism

    @property
    def mechanism_state(self):
        """tuple[int]: The current state of the mechanism."""
        return self._ria.mechanism_state

    @property
    def purview(self):
        """list[int]: The purview over which this mechanism's |small_phi| is
        maximal.
        """
        return self._ria.purview

    @property
    def purview_state(self):
        """tuple[int]: The current state of the purview."""
        return self._ria.purview_state

    @property
    def mip(self):
        """KPartition: The partition that makes the least difference to the
        mechanism's repertoire.
        """
        return self._ria.partition

    @property
    def repertoire(self):
        """np.ndarray: The unpartitioned repertoire of the mechanism over the
        purview.
        """
        return self._ria.repertoire

    @property
    def partitioned_repertoire(self):
        """np.ndarray: The partitioned repertoire of the mechanism over the
        purview.
        """
        return self._ria.partitioned_repertoire

    @property
    def specified_index(self):
        """The state(s) with the maximal absolute intrinsic difference
        between the unpartitioned and partitioned repertoires."""
        return self._ria.specified_index

    @property
    def specified_state(self):
        """The state(s) with the maximal absolute intrinsic difference
        between the unpartitioned and partitioned repertoires."""
        return self._ria.specified_state

    @property
    def ria(self):
        """RepertoireIrreducibilityAnalysis: The irreducibility analysis for
        this mechanism.
        """
        return self._ria

    def set_ties(self, ties):
        """Set the ``purview_partition_ties`` attribute."""
        self._all_ties = ties

    @property
    def purview_ties(self):
        """list: MICE objects for any other purviews that are maximal.

        NOTE: Partition ties are resolved arbitrarily.
        """
        seen = set()
        for tie in self._all_ties:
            if tie.purview not in seen:
                yield tie
                seen.add(tie.purview)

    @property
    def partition_ties(self):
        """list: MICE objects for any other purviews that are maximal.

        NOTE: Partition ties are resolved arbitrarily.
        """
        seen = set()
        for tie in self._all_ties:
            if tie.purview == self.purview and tie.mip not in seen:
                yield tie
                seen.add(tie.mip)

    @property
    def state_ties(self):
        cls = type(self)
        for ria in self.ria.state_ties():
            yield cls(ria, ties=self._all_ties)

    def ties(self, purview=True, state=True, partition=True):
        """Return MICE for any other purviews, partitions, or states that are maximal."""
        if purview and partition:
            ties = []
            # TODO(4.0) Currently the logic in subsystem.find_mice will add
            # duplicate MIPE to the ties, one for each state. This should be
            # made more sensible when the logic stabilizes, but for now we just
            # filter out the duplicates.
            seen = set()
            for tie in self._all_ties:
                if tie not in seen:
                    ties.append(tie)
                    seen.add(tie)
        elif purview:
            ties = self.purview_ties
        elif partition:
            ties = self.partition_ties
        else:
            ties = [self]

        if state:
            for mice in ties:
                yield from mice.state_ties
        else:
            yield from ties

    @property
    def is_tied(self):
        """Whether this MICE is non-unique."""
        return len(self._all_ties) > 1

    def flip(self):
        """Return the linked MICE in the other direction."""
        return self.parent.mice(self.direction.flip())

    def __repr__(self):
        return fmt.make_repr(self, ["ria"])

    def __str__(self):
        return "Maximally-irreducible {}\n".format(
            str(self.direction).lower()
        ) + fmt.indent(fmt.fmt_ria(self.ria, mip=True))

    unorderable_unless_eq = RepertoireIrreducibilityAnalysis.unorderable_unless_eq

    def order_by(self):
        return self.ria.order_by()

    def __eq__(self, other):
        return self.ria == other.ria

    def __hash__(self):
        return hash(self._ria)

    def to_json(self):
        return {"ria": self.ria}

    def _relevant_connections(self, subsystem):
        """Identify connections that “matter” to this concept.

        For a |MIC|, the important connections are those which connect the
        purview to the mechanism; for a |MIE| they are the connections from the
        mechanism to the purview.

        Returns an |N x N| matrix, where `N` is the number of nodes in this
        corresponding subsystem, that identifies connections that “matter” to
        this MICE:

        ``direction == Direction.CAUSE``:
            ``relevant_connections[i,j]`` is ``1`` if node ``i`` is in the
            cause purview and node ``j`` is in the mechanism (and ``0``
            otherwise).

        ``direction == Direction.EFFECT``:
            ``relevant_connections[i,j]`` is ``1`` if node ``i`` is in the
            mechanism and node ``j`` is in the effect purview (and ``0``
            otherwise).

        Args:
            subsystem (Subsystem): The |Subsystem| of this MICE.

        Returns:
            np.ndarray: A |N x N| matrix of connections, where |N| is the size
            of the network.

        Raises:
            ValueError: If ``direction`` is invalid.
        """
        _from, to = self.direction.order(self.mechanism, self.purview)
        return connectivity.relevant_connections(subsystem.network.size, _from, to)

    # TODO: pass in `cut` instead? We can infer
    # subsystem indices from the cut itself, validate, and check.
    def damaged_by_cut(self, subsystem):
        """Return ``True`` if this MICE is affected by the subsystem's cut.

        The cut affects the MICE if it either splits the MICE's mechanism
        or splits the connections between the purview and mechanism.
        """
        return subsystem.cut.splits_mechanism(self.mechanism) or np.any(
            self._relevant_connections(subsystem)
            * subsystem.cut.cut_matrix(subsystem.network.size)
            == 1
        )


class MaximallyIrreducibleCause(MaximallyIrreducibleCauseOrEffect):
    """A maximally irreducible cause (MIC).

    These can be compared with the built-in Python comparison operators (``<``,
    ``>``, etc.). Comparison is based on |small_phi| value, then mechanism size.
    """

    def __init__(self, ria, ties=None):
        if ria.direction != Direction.CAUSE:
            raise WrongDirectionError(
                "A MIC must be initialized with a RIA " "in the cause direction."
            )
        super().__init__(ria, ties=ties)

    def order_by(self):
        return self.ria.order_by()

    @property
    def direction(self):
        """Direction: |CAUSE|."""
        return self._ria.direction


class MaximallyIrreducibleEffect(MaximallyIrreducibleCauseOrEffect):
    """A maximally irreducible effect (MIE).

    These can be compared with the built-in Python comparison operators (``<``,
    ``>``, etc.). Comparison is based on |small_phi| value, then mechanism size.
    """

    def __init__(self, ria, ties=None):
        if ria.direction != Direction.EFFECT:
            raise WrongDirectionError(
                "A MIE must be initialized with a RIA in the effect direction."
            )
        super().__init__(ria, ties=ties)

    @property
    def direction(self):
        """Direction: |EFFECT|."""
        return self._ria.direction


# =============================================================================

_concept_attributes = ["phi", "mechanism", "cause", "effect", "subsystem"]


# TODO: make mechanism a property
# TODO: make phi a property
class Concept(cmp.Orderable):
    """The maximally irreducible cause and effect specified by a mechanism.

    These can be compared with the built-in Python comparison operators (``<``,
    ``>``, etc.). Comparison is based on |small_phi| value, then mechanism size.

    Attributes:
        mechanism (tuple[int]): The mechanism that the concept consists of.
        cause (MaximallyIrreducibleCause): The |MIC| representing the
            maximally-irreducible cause of this concept.
        effect (MaximallyIrreducibleEffect): The |MIE| representing the
            maximally-irreducible effect of this concept.
        subsystem (Subsystem): This concept's parent subsystem.
        time (float): The number of seconds it took to calculate.
    """

    def __init__(
        self,
        mechanism=None,
        cause=None,
        effect=None,
        subsystem=None,
    ):
        self.mechanism = mechanism
        self.cause = cause
        self.effect = effect
        self.subsystem = subsystem
        self.node_labels = subsystem.node_labels
        # Attach references to this object on the cause and effect
        # TODO(4.0) document this
        self.cause.parent = self
        self.effect.parent = self

    def __repr__(self):
        return fmt.make_repr(self, _concept_attributes)

    def __str__(self):
        return fmt.fmt_concept(self)

    @property
    def phi(self):
        """float: The size of the concept.

        This is the minimum of the |small_phi| values of the concept's |MIC|
        and |MIE|.
        """
        return min(self.cause.phi, self.effect.phi)

    # TODO(4.0) rename?
    def mice(self, direction):
        if direction is Direction.CAUSE:
            return self.cause
        if direction is Direction.EFFECT:
            return self.effect
        validate.direction(direction)

    @property
    def cause_purview(self):
        """tuple[int]: The cause purview."""
        return getattr(self.cause, "purview", None)

    @property
    def effect_purview(self):
        """tuple[int]: The effect purview."""
        return getattr(self.effect, "purview", None)

    @property
    def cause_repertoire(self):
        """np.ndarray: The cause repertoire."""
        return getattr(self.cause, "repertoire", None)

    @property
    def effect_repertoire(self):
        """np.ndarray: The effect repertoire."""
        return getattr(self.effect, "repertoire", None)

    @property
    def mechanism_state(self):
        """tuple(int): The state of this mechanism."""
        return utils.state_of(self.mechanism, self.subsystem.state)

    def purview(self, direction):
        """Return the purview in the given direction."""
        if direction == Direction.CAUSE:
            return self.cause.purview
        if direction == Direction.EFFECT:
            return self.effect.purview
        raise ValueError("invalid direction")

    unorderable_unless_eq = ["subsystem"]

    def order_by(self):
        return [self.phi, len(self.mechanism)]

    def __eq__(self, other):
        return (
            self.phi == other.phi
            and self.mechanism == other.mechanism
            and self.mechanism_state == other.mechanism_state
            and self.cause_purview == other.cause_purview
            and self.effect_purview == other.effect_purview
            and self.eq_repertoires(other)
            and self.subsystem.network == other.subsystem.network
        )

    def __hash__(self):
        return hash(
            (
                self.phi,
                self.mechanism,
                self.mechanism_state,
                self.cause_purview,
                self.effect_purview,
                utils.np_hash(self.cause_repertoire),
                utils.np_hash(self.effect_repertoire),
                self.subsystem.network,
            )
        )

    def __bool__(self):
        """A concept is ``True`` if |small_phi > 0|."""
        return not utils.eq(self.phi, 0)

    def eq_repertoires(self, other):
        """Return whether this concept has the same repertoires as another.

        .. warning::
            This only checks if the cause and effect repertoires are equal as
            arrays; mechanisms, purviews, or even the nodes that the mechanism
            and purview indices refer to, might be different.
        """
        return np.array_equal(
            self.cause_repertoire, other.cause_repertoire
        ) and np.array_equal(self.effect_repertoire, other.effect_repertoire)

    def emd_eq(self, other):
        """Return whether this concept is equal to another in the context of
        an EMD calculation.
        """
        return (
            self.phi == other.phi
            and self.mechanism == other.mechanism
            and self.eq_repertoires(other)
        )

    # These methods are used by phiserver
    # TODO Rename to expanded_cause_repertoire, etc
    def expand_cause_repertoire(self, new_purview=None):
        """See |Subsystem.expand_repertoire()|."""
        return self.subsystem.expand_cause_repertoire(
            self.cause.repertoire, new_purview
        )

    def expand_effect_repertoire(self, new_purview=None):
        """See |Subsystem.expand_repertoire()|."""
        return self.subsystem.expand_effect_repertoire(
            self.effect.repertoire, new_purview
        )

    def expand_partitioned_cause_repertoire(self):
        """See |Subsystem.expand_repertoire()|."""
        return self.subsystem.expand_cause_repertoire(
            self.cause.ria.partitioned_repertoire
        )

    def expand_partitioned_effect_repertoire(self):
        """See |Subsystem.expand_repertoire()|."""
        return self.subsystem.expand_effect_repertoire(
            self.effect.ria.partitioned_repertoire
        )

    def to_json(self):
        """Return a JSON-serializable representation."""
        return {attr: getattr(self, attr) for attr in _concept_attributes}

    @classmethod
    def from_json(cls, dct):
        del dct["phi"]
        return cls(**dct)
