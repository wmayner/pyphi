# models/ria.py
"""Mechanism-level repertoire irreducibility analysis (RIA).

A :class:`RepertoireIrreducibilityAnalysis` records the result of testing a
mechanism's irreducibility against a single partition in one temporal
direction. It carries both the canonical ``|·|+``-clamped ``phi`` (Eqs.
19-20) and the raw ``signed_phi`` for diagnostic use.

:class:`~pyphi.models.explanation.NullResultReason` enumerates reasons the
analysis returned a trivial null result. ``_null_ria`` is the convenience
constructor used when short-circuiting.
"""

from __future__ import annotations

import contextvars
import functools
from itertools import chain
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray
from toolz import unique

from pyphi import utils
from pyphi.data_structures import PyPhiFloat
from pyphi.direction import Direction
from pyphi.display import Description
from pyphi.display import Displayable
from pyphi.display import Row
from pyphi.display import Section
from pyphi.display import Table
from pyphi.display import tone_of
from pyphi.display.mixin import FULL
from pyphi.display.mixin import HIGH
from pyphi.display.mixin import LOW
from pyphi.display.numbers import format_value
from pyphi.measures.distribution import DistanceResult
from pyphi.models.explanation import Explanation
from pyphi.models.explanation import Finding
from pyphi.models.explanation import NullResultReason
from pyphi.models.partitions import JointPartition
from pyphi.models.partitions import _cut_grid
from pyphi.models.partitions import concise_partition

from . import cmp
from . import fmt
from .diff import Change
from .diff import ResultDiff
from .diff import _diff_common
from .pandas import ToDictFromExplicitAttrsMixin
from .pandas import ToPandasMixin
from .state_specification import StateSpecification
from .state_specification import UnitState
from .state_specification import normalization_factor

_SERIALIZING_AS_TIE_PEER: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "ria_serializing_as_tie_peer", default=False
)

if TYPE_CHECKING:
    from pyphi.labels import NodeLabels


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


def _repertoire_table(
    repertoire: NDArray, partitioned: NDArray | None, mark_states: list
) -> Table:
    """Distribution grid for a repertoire: rows = purview states, columns = ``Pr``
    (and ``Pr (cut)`` when a partitioned repertoire of matching shape is given).

    Tied specified states are marked with ``*``.
    """
    r = repertoire.squeeze()
    p = partitioned.squeeze() if partitioned is not None else None
    paired = p is not None and p.shape == r.shape
    headers = ("state", "Pr", "Pr (cut)") if paired else ("state", "Pr")
    marks = set(mark_states or [])
    rows = []
    for state in utils.all_states(r.shape):
        label = "(" + ",".join(map(str, state)) + ")"
        if state in marks:
            label += " *"
        cells: list[Any] = [label, float(r[state])]
        if paired:
            cells.append(float(p[state]))  # type: ignore[index]
        rows.append(tuple(cells))
    return Table(headers=headers, rows=tuple(rows), grid=True)


class RepertoireIrreducibilityAnalysis(
    Displayable, cmp.OrderableByPhi, ToDictFromExplicitAttrsMixin, ToPandasMixin
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
    _reasons: list[NullResultReason] | None
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
        reasons: list[NullResultReason] | None = None,
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

        # ``norm`` is None only for a null/unconstrained analysis (no
        # partition); such an analysis has no normalized phi.
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

    def _findings(self) -> tuple[Finding, ...]:
        findings = [
            Finding(kind="null_result", label="Null result", value=reason)
            for reason in (self.reasons or [])
        ]
        if self.purview:
            findings.append(Finding(kind="purview", label="Purview", value=self.purview))
        if self.partition is not None:
            findings.append(
                Finding(
                    kind="winning_partition",
                    label="MIP",
                    value=concise_partition(self.partition),
                )
            )
        return tuple(findings)

    def explain(self) -> Explanation:
        """A typed account of why this |small_phi| value came out as it did."""
        return Explanation(
            subject=f"φ = {format_value(self.phi)}",
            level="mechanism",
            findings=self._findings(),
        )

    def diff(self, other) -> ResultDiff:
        """Structured delta from this analysis to ``other`` (``a.diff(b)``).

        A mechanism-level result carries no :class:`ConfigSnapshot`, so
        ``config_diff`` is always empty.
        """
        if not isinstance(other, RepertoireIrreducibilityAnalysis):
            raise TypeError(
                f"cannot diff {type(self).__name__} against {type(other).__name__}"
            )
        common = _diff_common(self, other)
        changes = []
        if self.purview != other.purview:
            changes.append(
                Change("purview_changed", self.mechanism, self.purview, other.purview)
            )
        return ResultDiff(
            subject=f"Δφ = {format_value(common['delta_phi'])}",
            level="mechanism",
            delta_phi=common["delta_phi"],
            mip_changed=common["mip_changed"],
            changes=tuple(changes),
            config_diff=common["config_diff"],
            substrate_note=common["substrate_note"],
        )

    @property
    def specified_state(self):
        """The state with the maximal absolute intrinsic difference between
        the unpartitioned and partitioned repertoires among all ties."""
        return self._specified_state

    @functools.cached_property
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
        return unique(chain.from_iterable([self._state_ties, self._partition_ties]))

    @property
    def num_state_ties(self):
        return len(self._state_ties) - 1

    @property
    def num_partition_ties(self):
        return len(self._partition_ties) - 1

    @property
    def node_labels(self):
        """|NodeLabels| for this system."""
        return self._node_labels

    def __eq__(self, other: object) -> bool:  # noqa: PLR0911
        # The partition and partitioned repertoire are not considered.
        if not isinstance(other, RepertoireIrreducibilityAnalysis):
            return NotImplemented
        if self.direction != other.direction:
            return False
        if self.mechanism != other.mechanism:
            return False
        if self.purview != other.purview:
            return False
        if self.specified_state != other.specified_state:
            return False
        if not utils.eq(self.phi, other.phi):
            return False
        return cmp.numpy_aware_eq(self.repertoire, other.repertoire)

    def __bool__(self):
        """A |RepertoireIrreducibilityAnalysis| is ``True`` if it has
        |small_phi > 0|.
        """
        return utils.is_positive(self.phi)

    def __hash__(self) -> int:
        # specified_state must be in __hash__ (not just __eq__): RIA.ties uses
        # toolz.unique (set-backed) to distinguish state-ties that share
        # (direction, mechanism, purview) but differ in specified_state.
        return hash(
            (
                self.direction,
                self.mechanism,
                self.purview,
                self.specified_state,
            )
        )

    def _describe(self, verbosity: int) -> Description:
        cls = type(self).__name__
        compact = f"{cls}({fmt.SMALL_PHI}={format_value(self.phi)})"
        if verbosity == LOW:
            return Description(title=cls, compact=compact)

        mech = fmt.fmt_mechanism(self.mechanism, self.node_labels)
        purv = fmt.fmt_mechanism(self.purview, self.node_labels)
        summary: list[Row] = [
            Row(fmt.SMALL_PHI, self.phi),
            Row(f"Normalized {fmt.SMALL_PHI}", self.normalized_phi),
            Row(
                "Direction",
                self.direction.name if self.direction is not None else None,
                tone=tone_of(self.direction),
            ),
            Row("Mechanism", mech),
            Row("Purview", purv),
        ]
        if self.specified_state is not None:
            ss = self.specified_state
            summary.append(Row("Specified state", ss.state))
            summary.append(Row("Intrinsic information", ss.intrinsic_information))
        if self.selectivity is not None:
            summary.append(Row("Selectivity", self.selectivity))
        sections = [Section(rows=tuple(summary))]

        # Section order: Repertoire, MIP, Ties.
        if verbosity >= HIGH and self.repertoire is not None:
            sections.append(self._repertoire_section())

        # MIP: the concise partition headline, plus its cut grid at FULL.
        partition_str = concise_partition(self.partition) if self.partition else "empty"
        mip_rows = [Row("Partition", partition_str)]
        if self.reasons is not None:
            mip_rows.append(Row("Reasons", ", ".join(map(str, self.reasons))))
        mip_body = (
            (_cut_grid(self.partition),)
            if verbosity >= FULL
            and self.partition
            and self.partition.num_connections_cut()
            else ()
        )
        sections.append(Section(label="MIP", rows=tuple(mip_rows), body=mip_body))

        sections.append(self._ties_section())

        return Description(title=cls, sections=tuple(sections), compact=compact)

    def _ties_section(self) -> Section:
        return Section(
            label="Ties",
            rows=(
                Row("State ties", self.num_state_ties),
                Row("Partition ties", self.num_partition_ties),
            ),
        )

    def _repertoire_section(self) -> Section:
        repertoire = self.repertoire
        assert repertoire is not None  # guarded by the caller
        mark_states = (
            [s.state for s in self.specified_state.ties]
            if self.specified_state is not None
            else []
        )
        if repertoire.size == 1:
            rows = [Row("Forward probability", repertoire.item())]
            if self.partitioned_repertoire is not None:
                rows.append(
                    Row(
                        "Partitioned forward probability",
                        self.partitioned_repertoire.item(),
                    )
                )
            return Section(label="Repertoire", rows=tuple(rows))
        table = _repertoire_table(repertoire, self.partitioned_repertoire, mark_states)
        return Section(label="Repertoire", body=(table,))

    _dict_attrs = _ria_dict_attrs

    def to_json(self):
        dct = {
            attr: getattr(self, attr)
            for attr in self._dict_attrs
            if attr not in {"mechanism_label", "purview_label"}
        }
        if _SERIALIZING_AS_TIE_PEER.get():
            return dct
        partition_peers = tuple(t for t in self._partition_ties if t is not self)
        state_peers = tuple(t for t in self._state_ties if t is not self)
        if partition_peers or state_peers:
            from pyphi.jsonify import jsonify

            token = _SERIALIZING_AS_TIE_PEER.set(True)
            try:
                if partition_peers:
                    dct["_partition_tie_peers"] = [
                        jsonify(p.to_json()) for p in partition_peers
                    ]
                if state_peers:
                    dct["_state_tie_peers"] = [jsonify(p.to_json()) for p in state_peers]
            finally:
                _SERIALIZING_AS_TIE_PEER.reset(token)
        return dct

    def _pandas_record(self):
        labels = self.node_labels

        def labelled(nodes):
            if labels is None:
                return tuple(nodes)
            return tuple(labels.coerce_to_labels(nodes))

        return {
            "phi": float(self.phi),
            "direction": str(self.direction),
            "mechanism": labelled(self.mechanism),
            "purview": labelled(self.purview),
            "mechanism_state": (
                None if self.mechanism_state is None else tuple(self.mechanism_state)
            ),
            "purview_state": (
                None if self.purview_state is None else tuple(self.purview_state)
            ),
            "specified_state": self.specified_state,
        }

    @classmethod
    def from_json(cls, data):
        partition_peers_raw: Any = data.pop("_partition_tie_peers", ())
        state_peers_raw: Any = data.pop("_state_tie_peers", ())
        partition_peers: tuple[RepertoireIrreducibilityAnalysis, ...] = tuple(
            cls(**dict(p)) for p in partition_peers_raw
        )
        state_peers: tuple[RepertoireIrreducibilityAnalysis, ...] = tuple(
            cls(**dict(p)) for p in state_peers_raw
        )
        instance = cls(**data)
        if partition_peers:
            partition_tied: tuple[RepertoireIrreducibilityAnalysis, ...] = (
                instance,
                *partition_peers,
            )
            instance._partition_ties = partition_tied
            for peer in partition_peers:
                peer._partition_ties = partition_tied
        if state_peers:
            state_tied: tuple[RepertoireIrreducibilityAnalysis, ...] = (
                instance,
                *state_peers,
            )
            instance._state_ties = state_tied
            for peer in state_peers:
                peer._state_ties = state_tied
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
