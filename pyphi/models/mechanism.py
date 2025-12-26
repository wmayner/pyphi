# models/mechanism.py
"""Mechanism-level objects."""

from dataclasses import dataclass
from enum import Enum, auto
from enum import unique as unique_enum
from functools import cached_property, total_ordering
from typing import Iterable, Tuple

import numpy as np
from more_itertools import flatten
from numpy.typing import ArrayLike
from toolz import concat, unique

from pyphi.models.cuts import KPartition

from .. import connectivity, utils, validate
from ..conf import config
from ..data_structures import PyPhiFloat
from ..direction import Direction
from ..exceptions import WrongDirectionError
from ..models import fmt
from ..registry import Registry
from ..warnings import warn_about_tie_serialization
from . import cmp, fmt
from .pandas import ToDictFromExplicitAttrsMixin, ToDictMixin, ToPandasMixin


@total_ordering
@dataclass(frozen=True)
class Unit:
    """A unit in a state."""

    index: int
    state: int
    label: str = None

    def __hash__(self):
        return hash((self.index, self.state))

    def __eq__(self, other):
        return (self.index, self.state) == (other.index, other.state)

    def __lt__(self, other):
        return (self.index, self.state) < (other.index, other.state)

    def __repr__(self):
        label = str(self.index) if self.label is None else self.label
        return label.lower() if self.state == 0 else label.upper()


@dataclass
class StateSpecification(ToDictMixin, ToPandasMixin):
    direction: Direction
    purview: Tuple[int]
    state: Tuple[int]
    intrinsic_information: PyPhiFloat
    repertoire: ArrayLike
    unconstrained_repertoire: ArrayLike

    def __post_init__(self):
        self.intrinsic_information = PyPhiFloat(self.intrinsic_information)

    def set_ties(self, ties: Iterable):
        self._ties = ties

    @property
    def ties(self):
        return self._ties

    def __getitem__(self, i):
        return self.state[i]

    def __eq__(self, other):
        return cmp.general_eq(
            self,
            other,
            [
                "direction",
                "purview",
                "state",
                "intrinsic_information",
                "repertoire",
                "unconstrained_repertoire",
            ],
        )

    def __hash__(self):
        return hash(
            (self.direction, self.purview, self.state, self.intrinsic_information)
        )

    def _repr_columns(self, prefix=""):
        # TODO(fmt) include purview
        return [
            (f"{prefix}{self.direction}", fmt.state(self.state)),
            (
                f"{prefix}II_{str(self.direction)[:1].lower()}",
                self.intrinsic_information,
            ),
        ]

    def __repr__(self):
        body = "\n".join(fmt.align_columns(self._repr_columns()))
        body = fmt.header(
            f"Specified {self.direction}", body, under_char=fmt.HEADER_BAR_3
        )
        return fmt.box(fmt.center(body))

    def is_congruent(self, other):
        ours = dict(zip(self.purview, self.state))
        theirs = dict(zip(other.purview, other.state))
        mutual = set(ours.keys()) & set(theirs.keys())
        return self.direction == other.direction and all(
            ours[purview_node] == theirs[purview_node] for purview_node in mutual
        )

    def to_json(self):
        warn_about_tie_serialization(self.__class__.__name__, serialize=True)
        dct = self.to_dict()
        return dct

    @classmethod
    def from_json(cls, data):
        warn_about_tie_serialization(cls.__name__, deserialize=True)
        for key in ["repertoire", "unconstrained_repertoire"]:
            data[key] = np.array(data[key])
        instance = cls(**data)
        instance._ties = (instance,)
        return instance


class DistinctionPhiNormalizationRegistry(Registry):
    """Storage for distinction |small_phi| normalizations."""

    desc = "functions for normalizing distinction |small_phi| values"


distinction_phi_normalizations = DistinctionPhiNormalizationRegistry()


@distinction_phi_normalizations.register("NONE")
def _(partition):
    return 1


@distinction_phi_normalizations.register("NUM_CONNECTIONS_CUT")
def _(partition):
    try:
        return 1 / partition.num_connections_cut()
    except ZeroDivisionError:
        return 1
    except AttributeError:
        return None


def normalization_factor(partition):
    return distinction_phi_normalizations[config.DISTINCTION_PHI_NORMALIZATION](
        partition
    )


@unique_enum
class ShortCircuitConditions(Enum):
    # MICE level reasons
    NO_PURVIEWS = auto()
    NO_PARTITIONS = auto()
    # MIP level reasons
    EMPTY_PURVIEW = auto()
    UNREACHABLE_STATE = auto()


_ria_dict_attrs = [
    "phi",
    "direction",
    "mechanism",
    "mechanism_label",
    "mechanism_state",
    "purview",
    "purview_label",
    "purview_state",
    "partition",
    "repertoire",
    "partitioned_repertoire",
    "specified_state",
    "node_labels",
]


class RepertoireIrreducibilityAnalysis(
    cmp.OrderableByPhi, ToDictFromExplicitAttrsMixin, ToPandasMixin
):
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
        specified_state=None,
        mechanism_state=None,
        purview_state=None,
        node_labels=None,
        selectivity=None,
        reasons=None,
    ):
        self._phi = PyPhiFloat(phi)
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
        self._specified_state = specified_state
        self._partition_ties = (self,)
        self._state_ties = (self,)
        self._selectivity = selectivity
        self._reasons = reasons

        norm = normalization_factor(self._partition)

        if norm is None:
            self._normalized_phi = None
        else:
            self._normalized_phi = PyPhiFloat(self._phi * norm)

        # Optional labels - only used to generate nice labeled reprs
        self._node_labels = node_labels

    @property
    def phi(self):
        """PyPhiFloat: This is the difference between the mechanism's unpartitioned
        and partitioned repertoires.
        """
        return self._phi

    @property
    def normalized_phi(self):
        """float: Normalized |small_phi| value."""
        return self._normalized_phi

    @property
    def direction(self):
        """Direction: |CAUSE| or |EFFECT|."""
        return self._direction

    @property
    def mechanism(self):
        """tuple[int]: The mechanism that was analyzed."""
        return self._mechanism

    @property
    def mechanism_label(self):
        """tuple[str]: The labels of the mechanism nodes."""
        return self.node_labels.label_string(self.mechanism, self.mechanism_state)

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
    def purview_label(self):
        """tuple[str]: The labels of the mechanism nodes."""
        return self.node_labels.label_string(self.purview, self.purview_state)

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
    def selectivity(self):
        """float: The selectivity factor."""
        return self._selectivity

    @property
    def reasons(self):
        """Reasons why the computation short-circuited."""
        return self._reasons

    @property
    def specified_state(self):
        """The state with the maximal absolute intrinsic difference between
        the unpartitioned and partitioned repertoires among all ties."""
        return self._specified_state

    @property
    def purview_units(self):
        return frozenset(
            (
                Unit(index, state, label=self.node_labels.index2label(index))
                for index, state in zip(
                    self.specified_state.purview, self.specified_state
                )
            )
        )

    def is_congruent(self, specified_state):
        """Whether the state specified by this RIA is congruent to the given one."""
        return self.specified_state.is_congruent(specified_state)

    @property
    def state_ties(self):
        return self._state_ties

    def set_state_ties(self, ties):
        ties = tuple(ties)
        self._state_ties = ties
        # Update tie references in partition ties
        for tie in self.partition_ties:
            tie._state_ties = ties

    @property
    def partition_ties(self):
        return self._partition_ties

    def set_partition_ties(self, ties):
        ties = tuple(ties)
        self._partition_ties = ties
        # Update tie references in state ties
        for tie in self.state_ties:
            tie._partition_ties = ties

    @property
    def ties(self):
        # TODO(ties) check unique usage here
        return unique(concat([self._state_ties, self._partition_ties]))

    @property
    def num_state_ties(self):
        if self._state_ties is None:
            return np.nan
        return len(self._state_ties) - 1

    @property
    def num_partition_ties(self):
        if self._partition_ties is None:
            return np.nan
        return len(self._partition_ties) - 1

    @property
    def node_labels(self):
        """|NodeLabels| for this system."""
        return self._node_labels

    def __eq__(self, other):
        # We don't consider the partition and partitioned repertoire in
        # checking for RIA equality.
        attrs = ["phi", "direction", "mechanism", "purview", "repertoire"]
        return cmp.general_eq(self, other, attrs)

    def __bool__(self):
        """A |RepertoireIrreducibilityAnalysis| is ``True`` if it has
        |small_phi > 0|.
        """
        return utils.is_positive(self.phi)

    def __hash__(self):
        return hash(
            (
                self.phi,
                self.direction,
                self.mechanism,
                self.purview,
                self.specified_state,
                utils.np_hash(self.repertoire),
            )
        )

    def _repr_columns(self):
        cols = [
            (fmt.SMALL_PHI, self.phi),
            (f"Normalized {fmt.SMALL_PHI}", self.normalized_phi),
            ("Mechanism", fmt.fmt_mechanism(self.mechanism, self.node_labels)),
            ("Purview", fmt.fmt_mechanism(self.purview, self.node_labels)),
        ]

        if self.specified_state is not None:
            cols.append(("Specified state", str(self.specified_state)))

        if self.selectivity is not None:
            cols.append(("Selectivity", self.selectivity))

        if self.repertoire is not None:
            if self.specified_state is not None:
                mark_states = [
                    specified.state for specified in self.specified_state.ties
                ]
            else:
                mark_states = []
            if self.repertoire.size == 1:
                repertoire_str = self.repertoire
                repertoire = ("Forward Pr", repertoire_str)
                partitioned_repertoire_str = self.partitioned_repertoire
                partitioned_repertoire = (
                    "Partitioned forward Pr",
                    partitioned_repertoire_str,
                )
            else:
                repertoire_str = fmt.fmt_repertoire(
                    self.repertoire, mark_states=mark_states
                )
                repertoire = ("Repertoire", repertoire_str)
                partitioned_repertoire_str = fmt.fmt_repertoire(
                    self.partitioned_repertoire, mark_states=mark_states
                )
                partitioned_repertoire = (
                    "Partitioned repertoire",
                    partitioned_repertoire_str,
                )
            cols.append(repertoire)
            cols.append(partitioned_repertoire)

        if self.partition:
            partition_str = fmt.fmt_partition(self.partition)
        else:
            partition_str = "empty"
        cols.append(("Partition", partition_str))

        if self.reasons is not None:
            cols.append(("Reasons", ", ".join(map(str, self.reasons))))

        cols += [
            ("State ties", self.num_state_ties),
            ("Partition ties", self.num_partition_ties),
        ]

        return cols

    def make_repr(self, title=None, columns=None):
        if title is None:
            title = self.__class__.__name__
        if columns is None:
            columns = self._repr_columns()
        lines = fmt.align_columns(columns)
        body = "\n".join(lines)
        body = fmt.header(title, body, under_char=fmt.HEADER_BAR_2, center=True)
        return fmt.box(body)

    def __repr__(self):
        return self.make_repr()

    def __str__(self):
        return repr(self)

    _dict_attrs = _ria_dict_attrs

    def to_json(self):
        # TODO(ties) implement serialization of ties
        warn_about_tie_serialization(self.__class__.__name__, serialize=True)
        return {
            attr: getattr(self, attr)
            for attr in self._dict_attrs
            if attr not in {"mechanism_label", "purview_label"}
        }

    @classmethod
    def from_json(cls, data):
        # TODO(ties) implement serialization of ties
        warn_about_tie_serialization(cls.__name__, deserialize=True)
        instance = cls(**data)
        instance._partition_ties = (instance,)
        instance._state_ties = (instance,)
        return instance


def _null_ria(direction, mechanism, purview, repertoire=None, phi=0.0, **kwargs):
    """The irreducibility analysis for a reducible mechanism."""
    # TODO Use properties here to infer mechanism and purview from
    # partition yet access them with .mechanism and .partition
    return RepertoireIrreducibilityAnalysis(
        direction=direction,
        mechanism=mechanism,
        purview=purview,
        partition=KPartition(),
        repertoire=repertoire,
        partitioned_repertoire=None,
        phi=phi,
        **kwargs,
    )


# =============================================================================


# TODO(4.0) implement as a subclass of RIA?
class MaximallyIrreducibleCauseOrEffect(
    cmp.Orderable, ToDictFromExplicitAttrsMixin, ToPandasMixin
):
    """A maximally irreducible cause or effect (MICE).

    These can be compared with the built-in Python comparison operators (``<``,
    ``>``, etc.). Comparison is based on |small_phi| value, then mechanism size.
    """

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
            self._state_ties = (self,) + tuple(
                type(self)(tie) for tie in self.ria.state_ties if tie is not self.ria
            )
        return self._state_ties

    def set_state_ties(self, ties):
        # Update state ties on other tied objects
        ties = tuple(ties)
        self._state_ties = ties
        # Update state ties on other tied objects
        for tie in flatten([self.partition_ties, self.purview_ties]):
            tie._state_ties = ties

    @property
    def num_state_ties(self):
        return self.ria.num_state_ties

    @property
    def partition_ties(self):
        if self._partition_ties is None:
            self._partition_ties = (self,) + tuple(
                type(self)(tie)
                for tie in self.ria.partition_ties
                if tie is not self.ria
            )
        return self._partition_ties

    def set_partition_ties(self, ties):
        ties = tuple(ties)
        self._partition_ties = ties
        # Update partition ties on other tied objects
        for tie in flatten([self.state_ties, self.purview_ties]):
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
        return self.ria._repr_columns() + [
            ("#(partition ties)", self.num_partition_ties),
        ]

    def __repr__(self):
        # TODO just use normal repr when subclass of RIA
        title = f"Maximally-irreducible {str(self.direction).lower()}"
        columns = self.ria._repr_columns() + [("Purview ties", self.num_partition_ties)]
        return self.ria.make_repr(title=title, columns=columns)

    def __str__(self):
        return self.__repr__()

    unorderable_unless_eq = RepertoireIrreducibilityAnalysis.unorderable_unless_eq

    def order_by(self):
        return self.ria.order_by()

    def __eq__(self, other):
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


# =============================================================================

_concept_attributes = [
    "phi",
    "mechanism",
    "mechanism_state",
    "mechanism_label",
    "cause",
    "effect",
]


# TODO: make mechanism a property
# TODO: make phi a property
class Concept(cmp.OrderableByPhi, ToDictFromExplicitAttrsMixin, ToPandasMixin):
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
    ):
        self.mechanism = mechanism
        self.cause = cause
        self.effect = effect
        # Attach references to this object on the cause and effect
        # TODO(4.0) document this
        self.cause.parent = self
        self.effect.parent = self

    def __repr__(self):
        return fmt.make_repr(self, _concept_attributes)

    def __str__(self):
        return fmt.fmt_concept(self)

    # TODO use cached_property
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

    @cached_property
    def both_purview_unit_sets(self):
        return [
            set(self.mice(direction).purview_units) for direction in Direction.both()
        ]

    @cached_property
    def purview_union(self):
        return set.union(*self.both_purview_unit_sets)

    @cached_property
    def purview_intersection(self):
        return set.intersection(*self.both_purview_unit_sets)

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
        if self.cause.mechanism_state != self.effect.mechanism_state:
            raise ValueError("Inconsistent cause and effect mechanism states!")
        return self.cause.mechanism_state

    @cached_property
    def mechanism_label(self):
        """tuple[str]: The labels of the mechanism nodes."""
        return self.node_labels.label_string(self.mechanism, self.mechanism_state)

    def purview(self, direction):
        """Return the purview in the given direction."""
        if direction == Direction.CAUSE:
            return self.cause.purview
        if direction == Direction.EFFECT:
            return self.effect.purview
        raise ValueError("invalid direction")

    @property
    def node_labels(self):
        if self.cause.node_labels != self.effect.node_labels:
            raise ValueError("Inconsistent cause and effect node labels!")
        return self.cause.node_labels

    unorderable_unless_eq = ["subsystem"]

    def __eq__(self, other):
        try:
            return (
                self.phi == other.phi
                and self.mechanism == other.mechanism
                and self.mechanism_state == other.mechanism_state
                and self.cause_purview == other.cause_purview
                and self.effect_purview == other.effect_purview
                and self.eq_repertoires(other)
            )
        except AttributeError:
            return False

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
            )
        )

    def __bool__(self):
        """A concept is ``True`` if |small_phi > 0|."""
        return utils.is_positive(self.phi)

    def is_congruent(self, system_state):
        return all(
            self.mice(direction).is_congruent(system_state[direction])
            for direction in Direction.both()
        )

    # TODO(ties) refactor
    def resolve_congruence(self, system_state):
        """Choose the MIC/MIE that are congruent, if any."""
        cause, effect = [
            next(
                filter(
                    lambda mice: mice.is_congruent(system_state[direction]),
                    flatten(
                        [
                            self.mice(direction).state_ties,
                            self.mice(direction).purview_ties,
                        ]
                    ),
                ),
                None,
            )
            for direction in Direction.both()
        ]
        if cause is None or effect is None:
            return None
        return type(self)(
            mechanism=self.mechanism,
            cause=cause,
            effect=effect,
        )

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

    # TODO(4.0) REMOVE
    # # These methods are used by phiserver
    # # TODO Rename to expanded_cause_repertoire, etc
    # def expand_cause_repertoire(self, new_purview=None):
    #     """See |Subsystem.expand_repertoire()|."""
    #     return self.subsystem.expand_cause_repertoire(
    #         self.cause.repertoire, new_purview
    #     )

    # def expand_effect_repertoire(self, new_purview=None):
    #     """See |Subsystem.expand_repertoire()|."""
    #     return self.subsystem.expand_effect_repertoire(
    #         self.effect.repertoire, new_purview
    #     )

    # def expand_partitioned_cause_repertoire(self):
    #     """See |Subsystem.expand_repertoire()|."""
    #     return self.subsystem.expand_cause_repertoire(
    #         self.cause.ria.partitioned_repertoire
    #     )

    # def expand_partitioned_effect_repertoire(self):
    #     """See |Subsystem.expand_repertoire()|."""
    #     return self.subsystem.expand_effect_repertoire(
    #         self.effect.ria.partitioned_repertoire
    #     )

    _dict_attrs = _concept_attributes

    def to_json(self):
        """Return a JSON-serializable representation."""
        return dict(
            mechanism=self.mechanism,
            cause=self.cause,
            effect=self.effect,
        )

    @classmethod
    def from_json(cls, dct):
        instance = cls(**dct)
        # Restore parent references to MICEs
        instance.cause.parent = instance
        instance.effect.parent = instance
        return instance

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Restore parent references to MICEs
        self.cause.parent = self
        self.effect.parent = self
        return self
