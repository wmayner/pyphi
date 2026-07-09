# models/complex.py
"""The :class:`~pyphi.models.complex.Complex` — an irreducible system selected
as a local maximum of Φ under the exclusion postulate — and the lightweight
record of a candidate excluded in its favor."""

from __future__ import annotations

from typing import Any

from pyphi import utils
from pyphi.display import Description
from pyphi.display import Displayable
from pyphi.display import Row
from pyphi.display import Section
from pyphi.display.numbers import format_value
from pyphi.serializable import Serializable

from . import cmp
from .pandas import ToPandasMixin


class ExcludedCandidate(Displayable, ToPandasMixin):
    """A candidate system excluded from being a complex in favor of an
    overlapping complex with greater-or-equal Φ.

    Holds plain values only (units and Φ), never a back-reference to the
    excluding :class:`~pyphi.models.complex.Complex`, so the heavy analysis
    graph is not retained.

    Attributes
    ----------
    node_indices : tuple[int, ...]
        The excluded candidate's units.
    phi : float
        The candidate's Φ value.
    """

    def __init__(self, node_indices: Any, phi: Any) -> None:
        self.node_indices: tuple[int, ...] = tuple(node_indices)
        self.phi: float = float(phi)

    def _pandas_record(self) -> dict[str, Any]:
        return {"node_indices": self.node_indices, "phi": float(self.phi)}

    def _describe(self, verbosity: int) -> Description:  # noqa: ARG002
        cls = type(self).__name__
        return Description(
            title=cls,
            compact=f"{cls}({self.node_indices}, φ={format_value(self.phi)})",
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExcludedCandidate):
            return NotImplemented
        return self.node_indices == other.node_indices and utils.eq(self.phi, other.phi)

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __hash__(self) -> int:
        return hash(self.node_indices)


class Complex(Displayable, cmp.OrderableByPhi, ToPandasMixin, Serializable):
    """An irreducible system selected as a complex: a local maximum of Φ over
    overlapping candidate systems (the exclusion postulate).

    Wraps the system irreducibility analysis (IIT 3.0 or 4.0) and records
    whether it is the Φ-maximal complex of its substrate, the candidates
    excluded in its favor, and the substrate that selected it. Ordered by Φ
    like the wrapped analysis.

    Attributes
    ----------
    sia
        The wrapped system irreducibility analysis.
    substrate : Substrate
        The substrate this complex was selected from.
    is_maximal : bool
        Whether this is the Φ-maximal complex.
    excluded : tuple[ExcludedCandidate, ...]
        Overlapping candidates excluded in this complex's favor.
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
        from pyphi.condensation import _sia_node_indices

        return _sia_node_indices(self.sia) or ()

    @property
    def phi(self) -> Any:  # type: ignore[override]
        """The Φ value of this complex."""
        return self.sia.phi

    def _pandas_record(self) -> dict[str, Any]:
        record = dict(self.sia._pandas_record())
        record["is_maximal"] = self.is_maximal
        return record

    def order_by(self) -> Any:
        return self.sia.order_by()

    def __bool__(self) -> bool:
        """``True`` iff Φ > 0 (a null complex is falsy)."""
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

    def _describe(self, verbosity: int) -> Description:  # noqa: ARG002
        cls = type(self).__name__
        num_excluded = len(self.excluded)
        return Description(
            title=cls,
            sections=(
                Section(
                    rows=(
                        Row("Φ", self.phi),
                        Row("Nodes", str(self.node_indices)),
                        Row("Is maximal", self.is_maximal),
                        Row("Excluded candidates", num_excluded),
                    )
                ),
            ),
            compact=(
                f"{cls}({self.node_indices}, Φ={format_value(self.phi)},"
                f" is_maximal={self.is_maximal})"
            ),
        )
