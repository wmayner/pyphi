# models/complex.py
"""The |Complex| — an irreducible system selected as a local maximum of
|big_phi| under the exclusion postulate — and the lightweight record of a
candidate excluded in its favor."""

from __future__ import annotations

from typing import Any

from pyphi import utils

from . import fmt

_excluded_candidate_attributes = ["node_indices", "phi"]


class ExcludedCandidate:
    """A candidate system excluded from being a complex in favor of an
    overlapping complex with greater-or-equal |big_phi|.

    Holds plain values only (units and |big_phi|), never a back-reference to
    the excluding |Complex|, so the heavy analysis graph is not retained.

    Attributes:
        node_indices (tuple[int, ...]): The excluded candidate's units.
        phi (float): The candidate's |big_phi| value.
    """

    def __init__(self, node_indices: Any, phi: Any) -> None:
        self.node_indices: tuple[int, ...] = tuple(node_indices)
        self.phi: float = float(phi)

    def __repr__(self) -> str:
        return fmt.make_repr(self, _excluded_candidate_attributes)

    def __str__(self) -> str:
        return f"ExcludedCandidate(node_indices={self.node_indices}, phi={self.phi})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExcludedCandidate):
            return NotImplemented
        return self.node_indices == other.node_indices and utils.eq(self.phi, other.phi)

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __hash__(self) -> int:
        return hash(self.node_indices)

    def to_json(self) -> dict[str, Any]:
        return {"node_indices": list(self.node_indices), "phi": self.phi}

    @classmethod
    def from_json(cls, dct: dict[str, Any]) -> ExcludedCandidate:
        return cls(node_indices=dct["node_indices"], phi=dct["phi"])
