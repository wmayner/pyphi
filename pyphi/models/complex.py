# models/complex.py
"""The |Complex| — an irreducible system selected as a local maximum of
|big_phi| under the exclusion postulate — and the lightweight record of a
candidate excluded in its favor."""

from __future__ import annotations

from typing import Any

from pyphi import utils

from . import cmp
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


_complex_attributes = ["node_indices", "phi", "is_maximal", "excluded"]


class Complex(cmp.OrderableByPhi):
    """An irreducible system selected as a complex: a local maximum of
    |big_phi| over overlapping candidate systems (the exclusion postulate).

    Wraps the system irreducibility analysis (IIT 3.0 or 4.0) and records
    whether it is the |big_phi|-maximal complex of its substrate, the
    candidates excluded in its favor, and the substrate that selected it.
    Ordered by |big_phi| like the wrapped analysis.

    Attributes:
        sia: The wrapped system irreducibility analysis.
        substrate (Substrate): The substrate this complex was selected from.
        is_maximal (bool): Whether this is the |big_phi|-maximal complex.
        excluded (tuple[ExcludedCandidate, ...]): Overlapping candidates
            excluded in this complex's favor.
    """

    def __init__(
        self,
        sia: Any,
        substrate: Any,
        is_maximal: bool = False,
        excluded: Any = (),
    ) -> None:
        self.sia = sia
        self.substrate = substrate
        self.is_maximal = bool(is_maximal)
        self.excluded: tuple[ExcludedCandidate, ...] = tuple(excluded)

    @property
    def node_indices(self) -> tuple[int, ...]:
        """The units of this complex (``()`` for a null complex)."""
        from pyphi.substrate import _sia_node_indices

        return _sia_node_indices(self.sia) or ()

    @property
    def phi(self) -> Any:  # type: ignore[override]
        """The |big_phi| value of this complex."""
        return self.sia.phi

    def order_by(self) -> Any:
        return self.sia.order_by()

    def __bool__(self) -> bool:
        """``True`` iff |big_phi > 0| (a null complex is falsy)."""
        return not utils.eq(self.phi, 0)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Complex):
            return NotImplemented
        return (
            self.sia == other.sia
            and self.is_maximal == other.is_maximal
            and self.excluded == other.excluded
        )

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __hash__(self) -> int:
        return hash((self.node_indices, self.is_maximal))

    def __repr__(self) -> str:
        return fmt.make_repr(self, _complex_attributes)

    def __str__(self) -> str:
        return self.__repr__()

    def to_json(self) -> dict[str, Any]:
        return {
            "sia": self.sia,
            "substrate": self.substrate,
            "is_maximal": self.is_maximal,
            "excluded": list(self.excluded),
        }

    @classmethod
    def from_json(cls, dct: dict[str, Any]) -> Complex:
        return cls(
            sia=dct["sia"],
            substrate=dct["substrate"],
            is_maximal=dct["is_maximal"],
            excluded=dct["excluded"],
        )
