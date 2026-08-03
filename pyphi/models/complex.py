# models/complex.py
"""The :class:`~pyphi.models.complex.Complex` — an irreducible system selected
as a local maximum of system irreducibility (φₛ under IIT 4.0, Φ under IIT 3.0)
under the exclusion postulate — and the lightweight record of a candidate
excluded in its favor."""

from __future__ import annotations

from typing import Any

from pyphi import numerics
from pyphi.display import Description
from pyphi.display import Displayable
from pyphi.display import Row
from pyphi.display import Section
from pyphi.display import system_phi_label
from pyphi.display.numbers import format_value
from pyphi.serializable import Serializable

from . import cmp
from .pandas import ToPandasMixin


class ExcludedCandidate(Displayable, ToPandasMixin):
    """A candidate system that overlaps a complex and is not itself a
    complex: it was beaten (or Φ-outranked) by an overlapping accepted
    complex, or belonged to a Φ-tied clique that failed exclusion.

    An excluded candidate may carry higher φₛ than a complex whose record
    it appears in, when it was carved away by a different overlapping
    complex. Holds plain values only, never a back-reference to the
    excluding :class:`~pyphi.models.complex.Complex`, so the heavy analysis
    graph is not retained.

    Attributes
    ----------
    node_indices : tuple[int, ...]
        The excluded candidate's micro units.
    phi : float or None
        The candidate's φₛ value, or ``None`` for a gated candidate —
        one whose partition sweep was skipped because its
        intrinsic-information ceiling is certifiably below the excluding
        complex's φₛ.
    units : tuple or None
        The candidate's macro unit structure; ``None`` for a candidate
        system of micro units.
    ii_ceiling : float or None
        The certified upper bound on φₛ, when the certified prune
        computed one.
    gated : bool
        Whether the candidate was gated; ``phi`` is ``None`` exactly
        when this is True.
    """

    def __init__(
        self,
        node_indices: Any,
        phi: Any,
        units: Any = None,
        ii_ceiling: Any = None,
        gated: bool = False,
    ) -> None:
        self.node_indices: tuple[int, ...] = tuple(node_indices)
        self.phi: float | None = float(phi) if phi is not None else None
        self.units: tuple[Any, ...] | None = tuple(units) if units is not None else None
        self.ii_ceiling: float | None = (
            float(ii_ceiling) if ii_ceiling is not None else None
        )
        self.gated: bool = bool(gated)

    def _pandas_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "node_indices": self.node_indices,
            "phi": float(self.phi) if self.phi is not None else None,
        }
        if self.ii_ceiling is not None:
            record["ii_ceiling"] = self.ii_ceiling
        if self.gated:
            record["gated"] = True
        if self.units is not None:
            record["units"] = self.units
        return record

    def _describe(self, verbosity: int) -> Description:  # noqa: ARG002
        cls = type(self).__name__
        if self.phi is None:
            value = (
                f"φ ≤ {format_value(self.ii_ceiling)}"
                if self.ii_ceiling is not None
                else "gated"
            )
        else:
            value = f"φ={format_value(self.phi)}"
        return Description(
            title=cls,
            compact=f"{cls}({self.node_indices}, {value})",
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExcludedCandidate):
            return NotImplemented
        if self.node_indices != other.node_indices or (self.phi is None) != (
            other.phi is None
        ):
            return False
        if self.phi is not None:
            assert other.phi is not None
            return numerics.eq(self.phi, other.phi)
        if (self.ii_ceiling is None) != (other.ii_ceiling is None):
            return False
        if self.ii_ceiling is None:
            return True
        assert other.ii_ceiling is not None
        return numerics.eq(self.ii_ceiling, other.ii_ceiling)

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
        units: Any = None,
        node_indices: Any = None,
    ) -> None:
        self.sia = sia
        self.substrate = substrate
        self.is_maximal = bool(is_maximal)
        self.excluded: tuple[ExcludedCandidate, ...] = tuple(excluded)
        self.units: tuple[Any, ...] | None = tuple(units) if units is not None else None
        self._node_indices: tuple[int, ...] | None = (
            tuple(node_indices) if node_indices is not None else None
        )

    @property
    def node_indices(self) -> tuple[int, ...]:
        """The micro units of this complex (``()`` for a null complex).

        For a complex of macro units this is the union of the units' micro
        constituents, not the macro units' own indices.
        """
        if self._node_indices is not None:
            return self._node_indices
        from pyphi.condensation import _sia_node_indices

        return _sia_node_indices(self.sia) or ()

    @property
    def phi(self) -> Any:  # type: ignore[override]
        """The system irreducibility value of this complex: φₛ under IIT 4.0,
        Φ under IIT 3.0."""
        return self.sia.phi

    @property
    def exclusion_margin(self) -> float | None:
        """The gap in φₛ between this complex and the best overlapping
        rival it beat, or ``None`` when it beat none.

        Rivals are the excluded candidates whose φₛ is less than or
        precision-equal to this complex's own. Because condensation is
        recursive, ``excluded`` may also contain overlapping candidates
        with higher φₛ — carved away by a different complex before this
        one was accepted — and those do not enter the margin. A margin
        of zero indicates an overlapping rival within ``precision`` of
        this complex's own φₛ: the selection was decided beyond φₛ,
        either by escalation within the tie clique or by the rival's
        overlap with another complex. Gated rivals — certified strictly
        below this complex's φₛ, with no exact value — do not enter the
        margin.
        """
        phi = float(self.phi)
        rivals = [
            float(c.phi)
            for c in self.excluded
            # numerics: exact — composed with eq into a tolerant ≤ for a reported margin
            if c.phi is not None and (c.phi < phi or numerics.eq(c.phi, phi))
        ]
        if not rivals:
            return None
        return max(0.0, phi - max(rivals))

    @property
    def effectively_tied(self) -> bool:
        """Whether the exclusion margin is within ``precision`` of zero.

        ``False`` when the margin is ``None`` (no beaten rival).
        """
        margin = self.exclusion_margin
        return margin is not None and numerics.is_zero(margin)

    def _pandas_record(self) -> dict[str, Any]:
        record = dict(self.sia._pandas_record())
        record["is_maximal"] = self.is_maximal
        record["exclusion_margin"] = self.exclusion_margin
        record["effectively_tied"] = self.effectively_tied
        return record

    def order_by(self) -> Any:
        return self.sia.order_by()

    def __bool__(self) -> bool:
        """``True`` iff Φ > 0 (a null complex is falsy)."""
        return not numerics.eq(self.phi, 0)

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
        phi_label = system_phi_label(getattr(self.sia, "config", None))
        rows = [
            Row(phi_label, self.phi),
            Row("Nodes", str(self.node_indices)),
            Row("Is maximal", self.is_maximal),
            Row("Excluded candidates", num_excluded),
        ]
        margin = self.exclusion_margin
        if margin is not None:
            rows.append(Row("Selection margin", margin))
            rows.append(Row("Effectively tied", self.effectively_tied))
        if self.units is not None:
            rows.insert(2, Row("Units", len(self.units)))
        return Description(
            title=cls,
            sections=(Section(rows=tuple(rows)),),
            compact=(
                f"{cls}({self.node_indices}, {phi_label}={format_value(self.phi)},"
                f" is_maximal={self.is_maximal})"
            ),
        )
