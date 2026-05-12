# models/mice.py
"""Maximally irreducible cause/effect (MICE) wrapper objects.

A MICE wraps the :class:`RepertoireIrreducibilityAnalysis` of a mechanism
over its maximally-specifying purview in one direction. The two concrete
subclasses (:class:`MaximallyIrreducibleCause` and
:class:`MaximallyIrreducibleEffect`) enforce a direction invariant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from more_itertools import flatten
from toolz import concat

from pyphi import connectivity
from pyphi.direction import Direction
from pyphi.exceptions import WrongDirectionError

from .pandas import ToDictFromExplicitAttrsMixin
from .pandas import ToPandasMixin
from .ria import RepertoireIrreducibilityAnalysis
from .ria import _ria_dict_attrs

if TYPE_CHECKING:
    from .concept import Concept

from . import cmp


# TODO(4.0) implement as a subclass of RIA?
class MaximallyIrreducibleCauseOrEffect(
    cmp.Orderable, ToDictFromExplicitAttrsMixin, ToPandasMixin
):
    """A maximally irreducible cause or effect (MICE).

    These can be compared with the built-in Python comparison operators (``<``,
    ``>``, etc.). Comparison is based on |small_phi| value, then mechanism size.
    """

    parent: Concept  # Set by Concept.__init__

    def __init__(self, ria):
        self._ria = ria
        self._state_ties = None
        self._partition_ties = None
        self._purview_ties = None

    @property
    def phi(self):
        """float: The difference between the mechanism's unpartitioned and
        partitioned repertoires.
        """
        return self._ria.phi

    @property
    def normalized_phi(self):
        """float: Normalized |small_phi| value."""
        return self._ria.normalized_phi

    @property
    def direction(self):
        """Direction: |CAUSE| or |EFFECT|."""
        return self._ria.direction

    @property
    def mechanism(self):
        """list[int]: The mechanism for which the MICE is evaluated."""
        return self._ria.mechanism

    @property
    def mechanism_label(self):
        """list[int]: The mechanism for which the MICE is evaluated."""
        return self._ria.mechanism_label

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
    def purview_label(self):
        return self.ria.purview_label

    @property
    def purview_units(self):
        return self.ria.purview_units

    # TODO(4.0) remove or rename to "purview_current_state"
    @property
    def purview_state(self):
        """tuple[int]: The current state of the purview."""
        return self._ria.purview_state

    @property
    def mip(self):
        """JointPartition: The partition that makes the least difference to the
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
    def selectivity(self):
        """float: The selectivity factor."""
        return self._ria.selectivity

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

    @property
    def node_labels(self):
        return self.ria.node_labels

    @property
    def partition(self):
        return self.ria.partition

    @property
    def reasons(self):
        return self.ria.reasons

    @property
    def state_ties(self):
        if self._state_ties is None:
            self._state_ties = (
                self,
                *tuple(
                    type(self)(tie) for tie in self.ria.state_ties if tie is not self.ria
                ),
            )
        return self._state_ties

    def set_state_ties(self, ties):
        # Update state ties on other tied objects
        ties = tuple(ties)
        self._state_ties = ties
        # Update state ties on other tied objects
        for tie in concat(filter(None, [self.partition_ties, self.purview_ties])):
            tie._state_ties = ties

    @property
    def num_state_ties(self):
        return self.ria.num_state_ties

    @property
    def partition_ties(self):
        if self._partition_ties is None:
            self._partition_ties = (
                self,
                *tuple(
                    type(self)(tie)
                    for tie in self.ria.partition_ties
                    if tie is not self.ria
                ),
            )
        return self._partition_ties

    def set_partition_ties(self, ties):
        ties = tuple(ties)
        self._partition_ties = ties
        # Update partition ties on other tied objects
        for tie in concat(filter(None, [self.state_ties, self.purview_ties])):
            tie._partition_ties = ties

    @property
    def num_partition_ties(self):
        return self.ria.num_partition_ties

    @property
    def purview_ties(self):
        """tuple[MaximallyIrreducibleCauseOrEffect]: The purviews that are tied
        for maximal |small_phi| value.
        """
        return self._purview_ties

    def set_purview_ties(self, ties):
        """Set the ties."""
        self._purview_ties = tuple(ties)
        # Update purview ties on other tied objects
        for tie in flatten([self.state_ties, self.partition_ties]):
            tie._purview_ties = ties

    @property
    def num_purview_ties(self):
        if self._purview_ties is None:
            return np.nan
        return len(self._purview_ties) - 1

    def flip(self):
        """Return the linked MICE in the other direction."""
        return self.parent.mice(self.direction.flip())

    def is_congruent(self, specified_state):
        """Return whether the state specified by this MICE is congruent."""
        return self.ria.is_congruent(specified_state)

    def _repr_columns(self):
        return [
            *self.ria._repr_columns(),
            ("#(partition ties)", self.num_partition_ties),
        ]

    def __repr__(self):
        # TODO just use normal repr when subclass of RIA
        title = f"Maximally-irreducible {str(self.direction).lower()}"
        columns = [*self.ria._repr_columns(), ("Purview ties", self.num_partition_ties)]
        return self.ria.make_repr(title=title, columns=columns)

    def __str__(self):
        return self.__repr__()

    unorderable_unless_eq = RepertoireIrreducibilityAnalysis.unorderable_unless_eq

    def order_by(self):
        return self.ria.order_by()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MaximallyIrreducibleCauseOrEffect):
            return NotImplemented
        return self.ria == other.ria

    def __hash__(self):
        return hash(self._ria)

    _dict_attrs = _ria_dict_attrs

    def to_dict(self):
        dct = super().to_dict()
        dct["is_mice"] = True
        return dct

    def to_json(self):
        return {"ria": self.ria}

    # TODO(to_pandas): This is currently broken; MICE should become a subclass
    # of RIA, and then a consistent implementation of `from_json` can be used
    # there
    @classmethod
    def from_json(cls, data):
        instance = cls(data["ria"])
        instance._purview_ties = (instance,)
        return instance

    def _relevant_connections(self, system):
        """Identify connections that “matter” to this concept.

        For a |MIC|, the important connections are those which connect the
        purview to the mechanism; for a |MIE| they are the connections from the
        mechanism to the purview.

        Returns an |N x N| matrix, where `N` is the number of nodes in this
        corresponding system, that identifies connections that “matter” to
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
            system (System): The |System| of this MICE.

        Returns:
            np.ndarray: A |N x N| matrix of connections, where |N| is the size
            of the substrate.

        Raises:
            ValueError: If ``direction`` is invalid.
        """
        _from, to = self.direction.order(self.mechanism, self.purview)
        return connectivity.relevant_connections(system.substrate.size, _from, to)

    # TODO: pass in `cut` instead? We can infer
    # system indices from the cut itself, validate, and check.
    def damaged_by_cut(self, system):
        """Return ``True`` if this MICE is affected by the system's cut.

        The cut affects the MICE if it either splits the MICE's mechanism
        or splits the connections between the purview and mechanism.
        """
        return system.partition.splits_mechanism(self.mechanism) or np.any(
            self._relevant_connections(system)
            * system.partition.cut_matrix(system.substrate.size)
            == 1
        )

    def __getstate__(self):
        dct = self.__dict__.copy()
        dct["parent"] = None
        return dct


class MaximallyIrreducibleCause(MaximallyIrreducibleCauseOrEffect):
    """A maximally irreducible cause (MIC).

    These can be compared with the built-in Python comparison operators (``<``,
    ``>``, etc.). Comparison is based on |small_phi| value, then mechanism size.
    """

    def __init__(self, ria):
        if ria.direction != Direction.CAUSE:
            raise WrongDirectionError(
                "A MIC must be initialized with a RIA in the cause direction."
            )
        super().__init__(ria)

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

    def __init__(self, ria):
        if ria.direction != Direction.EFFECT:
            raise WrongDirectionError(
                "A MIE must be initialized with a RIA in the effect direction."
            )
        super().__init__(ria)

    @property
    def direction(self):
        """Direction: |EFFECT|."""
        return self._ria.direction
