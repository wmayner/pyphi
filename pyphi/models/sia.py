# models/sia.py
"""IIT 3.0 ``SystemIrreducibilityAnalysis`` (system-level Φ result).

The IIT 4.0 system-level analysis lives in
:mod:`pyphi.formalism.iit4` (under the same name); the class here is the
IIT 3.0 result type that ``compute.system.sia`` produces.
"""

from __future__ import annotations

from typing import Any
from typing import ClassVar

from pyphi import utils

from . import cmp
from . import fmt
from .distinctions import _null_ces

_sia_attributes = [
    "phi",
    "partitioned_distinctions",
    "partition",
    "node_indices",
    "node_labels",
    "current_state",
]


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

    unorderable_unless_eq: ClassVar[list[str]] = []

    def __eq__(self, other):
        if type(other) is not type(self):
            return NotImplemented
        return cmp.general_eq(self, other, _sia_attributes)

    def __bool__(self):
        """A |SystemIrreducibilityAnalysis| is ``True`` if it has
        |big_phi > 0|.
        """
        return not utils.eq(self.phi, 0)

    def __hash__(self):
        return hash(
            (
                self.phi,
                self.partitioned_distinctions,
                self.partition,
                self.node_indices,
                self.current_state,
            )
        )

    def order_by(self) -> tuple[Any, bytes]:
        """Sort key: ``(phi, partition.lex_key())``."""
        if self.partition is None:
            return (self.phi, b"")
        return (self.phi, self.partition.lex_key())

    def to_json(self):
        """Return a JSON-serializable representation."""
        return {attr: getattr(self, attr) for attr in _sia_attributes}

    @classmethod
    def from_json(cls, dct):
        return cls(**dct)


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
