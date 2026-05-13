# models/ria.py
"""Mechanism-level repertoire irreducibility analysis (RIA).

A :class:`RepertoireIrreducibilityAnalysis` records the result of testing a
mechanism's irreducibility against a single partition in one temporal
direction. It carries both the canonical ``|·|+``-clamped ``phi`` (Eqs.
19-20) and the raw ``signed_phi`` for diagnostic use.

:class:`ShortCircuitConditions` enumerates reasons the analysis returned a
trivial null result. ``_null_ria`` is the convenience constructor used when
short-circuiting.
"""

from __future__ import annotations

from enum import Enum
from enum import auto
from enum import unique as unique_enum
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray
from toolz import concat
from toolz import unique

from pyphi import utils
from pyphi.data_structures import PyPhiFloat
from pyphi.direction import Direction
from pyphi.metrics.distribution import DistanceResult
from pyphi.models.partitions import JointPartition
from pyphi.warnings import warn_about_tie_serialization

from . import cmp
from . import fmt
from .pandas import ToDictFromExplicitAttrsMixin
from .pandas import ToPandasMixin
from .state_specification import StateSpecification
from .state_specification import UnitState
from .state_specification import normalization_factor

if TYPE_CHECKING:
    from pyphi.labels import NodeLabels


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

    _phi: PyPhiFloat
    _signed_phi: PyPhiFloat | DistanceResult
    _direction: Direction
    _mechanism: tuple[int, ...]
    _purview: tuple[int, ...]
    _partition: JointPartition
    _mechanism_state: tuple[int, ...] | None
    _purview_state: tuple[int, ...] | None
    _repertoire: NDArray[np.float64] | None
    _partitioned_repertoire: NDArray[np.float64] | None
    _specified_state: StateSpecification | None
    _partition_ties: tuple[RepertoireIrreducibilityAnalysis, ...]
    _state_ties: tuple[RepertoireIrreducibilityAnalysis, ...]
    _selectivity: float | None
    _reasons: list[ShortCircuitConditions] | None
    _normalized_phi: PyPhiFloat | None
    _signed_normalized_phi: PyPhiFloat | None
    _node_labels: NodeLabels | None

    def __init__(
        self,
        phi: float,
        direction: Direction,
        mechanism: tuple[int, ...],
        purview: tuple[int, ...],
        partition: JointPartition,
        repertoire: ArrayLike | None,
        partitioned_repertoire: ArrayLike | None,
        specified_state: StateSpecification | None = None,
        mechanism_state: tuple[int, ...] | None = None,
        purview_state: tuple[int, ...] | None = None,
        node_labels: NodeLabels | None = None,
        selectivity: float | None = None,
        reasons: list[ShortCircuitConditions] | None = None,
        signed_phi: float | DistanceResult | None = None,
    ) -> None:
        # ``signed_phi`` is the raw integration value, possibly negative
        # under preventative-cause semantics. ``phi`` exposes the ``|·|+``
        # clamp (Eqs. 19-20). Construction accepts the signed value as
        # ``phi``; if ``signed_phi`` is not supplied explicitly it is
        # snapshotted from ``phi`` before the clamp is applied.
        if signed_phi is None:
            signed_phi = phi
        clamped_phi = utils.positive_part(signed_phi)
        if isinstance(phi, DistanceResult):
            self._phi = type(phi)(clamped_phi, **phi._public_aux_data())  # type: ignore[assignment]
        else:
            self._phi = PyPhiFloat(clamped_phi)
        if isinstance(signed_phi, DistanceResult):
            self._signed_phi = signed_phi
        else:
            self._signed_phi = PyPhiFloat(signed_phi)
        self._direction = direction
        self._mechanism = mechanism
        self._purview = purview
        self._partition = partition
        self._mechanism_state = mechanism_state
        self._purview_state = purview_state

        def _repertoire(repertoire: ArrayLike | None) -> NDArray[np.float64] | None:
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
            self._signed_normalized_phi = None
        else:
            # Compute the signed normalized phi (raw) first, then derive
            # the clamped canonical value.
            if isinstance(signed_phi, DistanceResult):
                signed_norm = float(signed_phi) * norm
            else:
                signed_norm = signed_phi * norm
            self._signed_normalized_phi = PyPhiFloat(signed_norm)
            self._normalized_phi = PyPhiFloat(utils.positive_part(signed_norm))

        # Optional labels - only used to generate nice labeled reprs
        self._node_labels = node_labels

    @property
    def phi(self) -> PyPhiFloat:  # type: ignore[override]
        """PyPhiFloat: Canonical |small_phi| value (|·|+ clamped).

        This is ``positive_part(signed_phi)`` — the integrated information
        value with the |·|+ operator applied (Eqs. 19-20 of the IIT 4.0
        paper). Always non-negative. For the raw value before clamping
        (which may be negative under preventative-cause semantics), see
        ``signed_phi``.
        """
        return self._phi

    @property
    def signed_phi(self) -> PyPhiFloat | DistanceResult:
        """The raw |small_phi| before the |·|+ clamp.

        When negative, flags preventative-cause structure that the
        clamped ``phi`` hides. Surfaced for diagnostic inspection of
        substrates with preventative mechanisms.
        """
        return self._signed_phi

    @property
    def normalized_phi(self):
        """float: Canonical normalized |small_phi| (|·|+ clamped)."""
        return self._normalized_phi

    @property
    def signed_normalized_phi(self):
        """float: Raw normalized |small_phi| before the |·|+ clamp."""
        return self._signed_normalized_phi

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
        assert self.node_labels is not None
        return self.node_labels.label_string(
            self.mechanism,
            self.mechanism_state,  # type: ignore[arg-type]
        )

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
        assert self.node_labels is not None
        return self.node_labels.label_string(
            self.purview,
            self.purview_state,  # type: ignore[arg-type]
        )

    @property
    def purview_state(self):
        """tuple[int]: The current state of the purview."""
        return self._purview_state

    @property
    def partition(self):
        """JointPartition: The partition of the mechanism-purview pair that was
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
        assert self.node_labels is not None
        assert self.specified_state is not None
        return frozenset(
            (
                UnitState(index, state, label=self.node_labels.index2label(index))
                for index, state in zip(
                    self.specified_state.purview,
                    self.specified_state.state,
                    strict=False,
                )
            )
        )

    def is_congruent(self, specified_state):
        """Whether the state specified by this RIA is congruent to the given one."""
        assert self.specified_state is not None
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


def _null_ria(
    direction: Direction,
    mechanism: tuple[int, ...],
    purview: tuple[int, ...],
    repertoire: ArrayLike | None = None,
    phi: float = 0.0,
    **kwargs: Any,
) -> RepertoireIrreducibilityAnalysis:
    """The irreducibility analysis for a reducible mechanism."""
    # TODO Use properties here to infer mechanism and purview from
    # partition yet access them with .mechanism and .partition
    return RepertoireIrreducibilityAnalysis(
        direction=direction,
        mechanism=mechanism,
        purview=purview,
        partition=JointPartition(),
        repertoire=repertoire,
        partitioned_repertoire=None,
        phi=phi,
        **kwargs,
    )
