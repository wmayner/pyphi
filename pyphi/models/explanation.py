# models/explanation.py
"""Typed explanations of why a result came out as it did (``result.explain()``).

:class:`NullResultReason` enumerates the conditions under which an analysis
yields a trivial (|big_phi| = 0 / |alpha| = 0) result. :class:`Finding` and
:class:`Explanation` are the typed account ``.explain()`` returns;
:class:`RunnerUp` is the lightweight record of the second-best partition
retained at MIP selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from enum import auto
from enum import unique
from typing import Any

from pyphi import utils
from pyphi.display import Description
from pyphi.display import Displayable
from pyphi.display import Row
from pyphi.display import Section


@unique
class NullResultReason(Enum):
    """A condition under which an analysis returns a trivial null result."""

    # System level
    NO_SYSTEM = auto()
    NO_STRONG_CONNECTIVITY = auto()
    NO_WEAK_CONNECTIVITY = auto()
    MONAD_WITH_NO_SELFLOOP = auto()
    MONAD_WITH_SELFLOOP_DEFINED_TO_BE_ZERO_PHI = auto()
    NO_VALID_PARTITIONS = auto()
    NO_CAUSE = auto()
    NO_EFFECT = auto()
    EMPTY_CAUSE_EFFECT_STRUCTURE = auto()
    # Mechanism level
    NO_PURVIEWS = auto()
    NO_PARTITIONS = auto()
    EMPTY_PURVIEW = auto()
    UNREACHABLE_STATE = auto()
    REDUCIBLE_OVER_PARTITION = auto()

    @property
    def level(self) -> str:
        """The structural level the reason arises at: ``"system"`` or
        ``"mechanism"``."""
        return _LEVEL_OF[self]


_MECHANISM_REASONS = frozenset(
    {
        NullResultReason.NO_PURVIEWS,
        NullResultReason.NO_PARTITIONS,
        NullResultReason.EMPTY_PURVIEW,
        NullResultReason.UNREACHABLE_STATE,
        NullResultReason.REDUCIBLE_OVER_PARTITION,
    }
)

_LEVEL_OF: dict[NullResultReason, str] = {
    reason: ("mechanism" if reason in _MECHANISM_REASONS else "system")
    for reason in NullResultReason
}


@dataclass(frozen=True)
class RunnerUp:
    """The second-best partition at MIP selection.

    The lowest-|small_phi| candidate whose value is strictly greater than the
    MIP's. ``partition`` is the cut; ``phi`` is its (clamped) integrated
    information.
    """

    partition: Any
    phi: Any


def runner_up_from_candidates(candidates: Any, mip_phi: Any) -> RunnerUp | None:
    """The lowest-phi candidate whose phi is *strictly* greater than ``mip_phi``.

    Candidates that tie the MIP (within :func:`pyphi.utils.eq`) are tied peers,
    not runners-up, so they are excluded. Returns ``None`` when the MIP is the
    unique phi value. Each candidate must expose ``.phi`` and ``.partition``.
    """
    mip = float(mip_phi)
    best = None
    for candidate in candidates:
        phi = float(candidate.phi)
        if (
            phi > mip
            and not utils.eq(phi, mip)
            and (best is None or phi < float(best.phi))
        ):
            best = candidate
    if best is None:
        return None
    return RunnerUp(partition=best.partition, phi=best.phi)


@dataclass(frozen=True)
class Finding:
    """One element of an explanation.

    ``kind`` is a stable machine key (``"null_result"``, ``"winning_partition"``,
    ``"runner_up"``, ``"gap"``, ``"binding_direction"``, ...); ``label`` is the
    human-readable summary; ``value`` is the quantity it concerns; ``detail``
    holds optional supporting fields; ``tone`` is an optional semantic accent
    (``"cause"`` / ``"effect"``) that HTML rendering colors.
    """

    kind: str
    label: str
    value: Any = None
    detail: tuple[tuple[str, Any], ...] = ()
    tone: str | None = None


def _reason_value(value: Any) -> Any:
    """Render a :class:`NullResultReason` by its name; pass other values
    through unchanged."""
    return value.name if isinstance(value, NullResultReason) else value


@dataclass(frozen=True)
class Explanation(Displayable):
    """A typed account of why a result came out as it did.

    ``subject`` names the quantity being explained (*e.g.* ``"Φ_s = 0.0"``);
    ``level`` is ``"system"`` or ``"mechanism"``; ``findings`` is the ordered
    account.
    """

    subject: str
    level: str
    findings: tuple[Finding, ...] = ()

    def _describe(self, verbosity: int) -> Description:  # noqa: ARG002
        rows = tuple(
            Row(finding.label, _reason_value(finding.value), tone=finding.tone)
            for finding in self.findings
        )
        return Description(
            title=self.subject,
            sections=(Section(label="Why", rows=rows),),
            compact=self.subject,
        )

    def to_pandas(self):
        """A tidy, one-row-per-finding ``DataFrame`` (B8 / P14d export)."""
        from pyphi.models.pandas import records_to_frame

        return records_to_frame(
            (
                {
                    "level": self.level,
                    "kind": finding.kind,
                    "label": finding.label,
                    "value": _reason_value(finding.value),
                }
                for finding in self.findings
            ),
            columns=["level", "kind", "label", "value"],
        )
