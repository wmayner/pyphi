# models/sia.py
"""IIT 3.0 ``SystemIrreducibilityAnalysis`` (system-level Φ result).

The IIT 4.0 system-level analysis lives in
:mod:`pyphi.formalism.iit4` (under the same name); the class here is the
IIT 3.0 result type that ``compute.system.sia`` produces.
"""

from __future__ import annotations

import contextvars
from typing import Any

from pyphi import utils

from . import cmp
from . import fmt
from .distinctions import _null_ces

_SERIALIZING_AS_TIE_PEER: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "iit3_sia_serializing_as_tie_peer", default=False
)


class IIT3SystemIrreducibilityAnalysis(cmp.OrderableByPhi):
    """An analysis of system irreducibility (|big_phi|).

    Contains the |big_phi| value of the |System| and the intermediate
    results obtained in the course of computing it.

    These can be compared with the built-in Python comparison operators (``<``,
    ``>``, etc.). First, |big_phi| values are compared. Then, if these are
    equal up to |PRECISION|, the one with the larger system is greater.

    Attributes:
        phi (float): The |big_phi| value for the system, *i.e.* the distance
            between the unpartitioned and partitioned cause-effect structures
            under the minimum-information partition.
        partitioned_distinctions (Distinctions): The cause-effect structure
            when the system is partitioned according to the MIP.
        partition (DirectedBipartition): The minimum-information partition.
        node_indices (tuple[int, ...]): Indices of the nodes the analysis
            was computed over.
        node_labels (NodeLabels): Labels corresponding to ``node_indices``.
        current_state (tuple[int, ...]): The system state at the time of
            analysis.
    """

    phi: float  # Override parent to allow None during init

    def __init__(
        self,
        phi=None,
        partitioned_distinctions=None,
        partition=None,
        node_indices=None,
        node_labels=None,
        current_state=None,
        config=None,
    ):
        if phi is None:
            self.phi = phi  # type: ignore[assignment]
        else:
            from pyphi.data_structures.pyphi_float import PyPhiFloat
            from pyphi.measures.distribution import DistanceResult

            if isinstance(phi, DistanceResult):
                self.phi = phi  # type: ignore[assignment]
            else:
                self.phi = PyPhiFloat(phi)  # type: ignore[assignment]
        self.partitioned_distinctions = partitioned_distinctions
        self.partition = partition
        self.node_indices = node_indices
        self.node_labels = node_labels
        self.current_state = current_state
        if config is None:
            from pyphi.conf import config as _global

            config = _global.snapshot()
        self.config = config

    def _repr_columns(self):
        return fmt.fmt_sia_columns(self)

    def _repr_html_(self) -> str:
        return fmt.html_columns(self._repr_columns(), title=self.__class__.__name__)

    def __repr__(self):
        body = "\n".join(fmt.align_columns(self._repr_columns()))
        body = fmt.header(self.__class__.__name__, body, under_char=fmt.HEADER_BAR_2)
        body = fmt.center(body)
        return fmt.box(body)

    def __str__(self):
        return fmt.fmt_sia(self)

    def print(self):
        """Print this SystemIrreducibilityAnalysis."""
        import sys

        sys.stdout.write(str(self) + "\n")

    def __eq__(self, other: object) -> bool:  # noqa: PLR0911
        if not isinstance(other, IIT3SystemIrreducibilityAnalysis):
            return NotImplemented
        if self.partitioned_distinctions != other.partitioned_distinctions:
            return False
        if self.partition != other.partition:
            return False
        if self.node_indices != other.node_indices:
            return False
        if self.node_labels != other.node_labels:
            return False
        if self.current_state != other.current_state:
            return False
        return utils.eq(self.phi, other.phi)

    def __bool__(self):
        """A |SystemIrreducibilityAnalysis| is ``True`` if it has
        |big_phi > 0|.
        """
        return not utils.eq(self.phi, 0)

    def __hash__(self) -> int:
        return hash(
            (
                self.partitioned_distinctions,
                self.partition,
                self.node_indices,
                self.node_labels,
                self.current_state,
            )
        )

    def order_by(self) -> tuple[Any, bytes]:
        """Sort key: ``(phi, partition.lex_key())``."""
        if self.partition is None:
            return (self.phi, b"")
        return (self.phi, self.partition.lex_key())

    @property
    def ties(self) -> list[Any]:
        """The full tied set this SIA belongs to (winner first, peers after).

        For an SIA without ties, returns a single-element list containing self.
        Populated by ``set_ties`` after partition evaluation; preserved across
        JSON round-trips via ``_tie_peers``.
        """
        try:
            return self._ties
        except AttributeError:
            self._ties = [self]
            return self._ties

    def set_ties(self, ties: list[Any]) -> None:
        self._ties = list(ties)

    def to_json(self):
        """Return a JSON-serializable representation."""
        dct = {
            attr: getattr(self, attr)
            for attr in (
                "phi",
                "partitioned_distinctions",
                "partition",
                "node_indices",
                "node_labels",
                "current_state",
            )
        }
        if _SERIALIZING_AS_TIE_PEER.get():
            return dct
        peers = tuple(t for t in self.ties if t is not self)
        if peers:
            from pyphi.jsonify import jsonify

            token = _SERIALIZING_AS_TIE_PEER.set(True)
            try:
                dct["_tie_peers"] = [jsonify(p.to_json()) for p in peers]
            finally:
                _SERIALIZING_AS_TIE_PEER.reset(token)
        return dct

    @classmethod
    def from_json(cls, dct):
        peers_raw: Any = dct.pop("_tie_peers", ())
        peers: tuple[IIT3SystemIrreducibilityAnalysis, ...] = tuple(
            cls(**dict(p)) for p in peers_raw
        )
        instance = cls(**dct)
        if peers:
            tied: list[IIT3SystemIrreducibilityAnalysis] = [instance, *peers]
            instance._ties = tied
            for peer in peers:
                peer._ties = tied
        return instance


def _null_sia(system, phi=0.0):
    """Return an IIT3SystemIrreducibilityAnalysis with zero phi.

    This is the analysis result for a reducible system.
    """
    return IIT3SystemIrreducibilityAnalysis(
        phi=phi,
        partitioned_distinctions=_null_ces(),
        partition=system.partition,
        node_indices=system.node_indices,
        node_labels=system.substrate.node_labels,
        current_state=system.state,
    )
