# models/sia.py
"""IIT 3.0 ``SystemIrreducibilityAnalysis`` (system-level Φ result).

The IIT 4.0 system-level analysis lives in
:mod:`pyphi.formalism.iit4` (under the same name); the class here is the
IIT 3.0 result type that ``compute.system.sia`` produces.
"""

from __future__ import annotations

from typing import Any

from pyphi import utils
from pyphi.display import PROVENANCE
from pyphi.display import Description
from pyphi.display import Displayable
from pyphi.display import Row
from pyphi.display import Section
from pyphi.display.numbers import format_value
from pyphi.provenance import HasProvenance

from . import cmp
from .diff import ResultDiff
from .diff import _diff_common
from .distinctions import Distinctions
from .distinctions import _null_ces
from .explanation import Explanation
from .explanation import Finding
from .partitions import _cut_grid
from .partitions import concise_partition


class IIT3SystemIrreducibilityAnalysis(HasProvenance, Displayable, cmp.OrderableByPhi):
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
        distinctions (Distinctions): The cause-effect structure of the
            unpartitioned system, computed in the course of the |big_phi|
            analysis. ``None`` when the analysis short-circuited to zero
            |big_phi| before the cause-effect structure was computed (*e.g.*
            the system is not strongly connected); retrieve it via
            :func:`pyphi.formalism.iit3.ces` in that case.
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
        distinctions: Distinctions | None = None,
        partitioned_distinctions=None,
        partition=None,
        node_indices=None,
        node_labels=None,
        current_state=None,
        config=None,
        provenance=None,
        reasons=None,
        runner_up=None,
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
        self.distinctions = distinctions
        self.partitioned_distinctions = partitioned_distinctions
        self.partition = partition
        self.node_indices = node_indices
        self.node_labels = node_labels
        self.current_state = current_state
        self.reasons = reasons or []
        self.runner_up = runner_up
        if config is None:
            from pyphi.conf import config as _global

            config = _global.snapshot()
        self.config = config
        if provenance is None:
            from pyphi.provenance import Provenance

            provenance = Provenance.capture()
        self.provenance = provenance

    def _system_label(self) -> str | None:
        node_indices = self.node_indices
        node_labels = self.node_labels
        if node_labels is not None and node_indices is not None:
            return ",".join(
                str(label) for label in node_labels.coerce_to_labels(node_indices)
            )
        if node_indices is not None:
            return ",".join(str(i) for i in node_indices)
        return None

    def _describe(self, verbosity: int) -> Description:
        cls = type(self).__name__
        sections = [
            Section(
                rows=(
                    Row("Φ", self.phi),
                    Row("System", self._system_label()),
                    Row("Current state", self.current_state),
                )
            )
        ]
        if self.partition is not None:
            mip_rows = (Row("Partition", concise_partition(self.partition)),)
            mip_body = (
                (_cut_grid(self.partition),)
                if self.partition.num_connections_cut()
                else ()
            )
            sections.append(Section(label="MIP", rows=mip_rows, body=mip_body))
        if verbosity >= PROVENANCE and self.provenance is not None:
            from pyphi.display.provenance import provenance_section

            sections.append(provenance_section(self.provenance))
        return Description(
            title=cls,
            sections=tuple(sections),
            compact=f"{cls}(Φ={format_value(self.phi)})",
        )

    def _findings(self) -> tuple[Finding, ...]:
        findings = [
            Finding(kind="null_result", label="Null result", value=reason)
            for reason in (self.reasons or [])
        ]
        if self.partition is not None and self.phi is not None and self.phi > 0:
            findings.append(
                Finding(
                    kind="winning_partition",
                    label="MIP",
                    value=concise_partition(self.partition),
                )
            )
        if self.runner_up is not None:
            findings.append(
                Finding(
                    kind="gap",
                    label="Φ-gap to runner-up",
                    value=float(self.runner_up.phi) - float(self.phi),
                )
            )
        return tuple(findings)

    def explain(self) -> Explanation:
        """A typed account of why this Φ value came out as it did."""
        return Explanation(
            subject=f"Φ = {format_value(self.phi)}",
            level="system",
            findings=self._findings(),
        )

    def diff(self, other) -> ResultDiff:
        """Structured delta from this SIA to ``other`` (``a.diff(b)``)."""
        if not isinstance(other, IIT3SystemIrreducibilityAnalysis):
            raise TypeError(
                f"cannot diff {type(self).__name__} against {type(other).__name__}"
            )
        common = _diff_common(self, other)
        return ResultDiff(
            subject=f"ΔΦ = {format_value(common['delta_phi'])}",
            level="system",
            delta_phi=common["delta_phi"],
            mip_changed=common["mip_changed"],
            config_diff=common["config_diff"],
            substrate_note=common["substrate_note"],
        )

    def print(self):
        """Print this SystemIrreducibilityAnalysis."""
        import sys

        sys.stdout.write(str(self) + "\n")

    def __eq__(self, other: object) -> bool:  # noqa: PLR0911
        if not isinstance(other, IIT3SystemIrreducibilityAnalysis):
            return NotImplemented
        if self.distinctions != other.distinctions:
            return False
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
                self.distinctions,
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


def _null_sia(system, phi=0.0, reasons=None):
    """Return an IIT3SystemIrreducibilityAnalysis with zero phi.

    This is the analysis result for a reducible system. ``reasons`` records
    why (a list of :class:`~pyphi.models.explanation.NullResultReason`).
    """
    return IIT3SystemIrreducibilityAnalysis(
        phi=phi,
        partitioned_distinctions=_null_ces(),
        partition=system.partition,
        node_indices=system.node_indices,
        node_labels=system.substrate.node_labels,
        current_state=system.state,
        reasons=reasons,
    )
