# models/explanation.py
"""Typed explanations of why a result came out as it did (``result.explain()``).

:class:`NullResultReason` enumerates the conditions under which an analysis
yields a trivial (Φ = 0 / α = 0) result. :class:`Finding` and
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

from pyphi import numerics
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
    NONUNIQUE_SYSTEM_STATE = auto()
    EMPTY_CAUSE_EFFECT_STRUCTURE = auto()
    # Mechanism level
    NO_PURVIEWS = auto()
    NO_POSITIVE_ALPHA = auto()
    NO_PARTITIONS = auto()
    EMPTY_PURVIEW = auto()
    UNREACHABLE_STATE = auto()
    REDUCIBLE_OVER_PARTITION = auto()
    OTHER_DIRECTION_REDUCIBLE = auto()

    @property
    def level(self) -> str:
        """The structural level the reason arises at: ``"system"`` or
        ``"mechanism"``."""
        return _LEVEL_OF[self]


_MECHANISM_REASONS = frozenset(
    {
        NullResultReason.NO_PURVIEWS,
        NullResultReason.NO_POSITIVE_ALPHA,
        NullResultReason.NO_PARTITIONS,
        NullResultReason.EMPTY_PURVIEW,
        NullResultReason.UNREACHABLE_STATE,
        NullResultReason.REDUCIBLE_OVER_PARTITION,
        NullResultReason.OTHER_DIRECTION_REDUCIBLE,
    }
)

_LEVEL_OF: dict[NullResultReason, str] = {
    reason: ("mechanism" if reason in _MECHANISM_REASONS else "system")
    for reason in NullResultReason
}


@dataclass(frozen=True)
class RunnerUp:
    """The second-best partition at MIP selection.

    The candidate ranked next after the MIP by the same quantity that
    selected the MIP. ``partition`` is the cut; ``phi`` is its (clamped)
    integrated information. ``normalized_phi`` is set only when the
    ranking quantity was normalized φ.
    """

    partition: Any
    phi: Any
    normalized_phi: Any = None


_NORMALIZED_STRATEGIES = frozenset({"NORMALIZED_PHI", "NEGATIVE_NORMALIZED_PHI"})
_PHI_VALUED_STRATEGIES = _NORMALIZED_STRATEGIES | {"PHI", "NEGATIVE_PHI"}


def sia_runner_up_key(strategy: Any) -> tuple[Any, bool]:
    """The ranking key matching an SIA tie-resolution strategy.

    Returns ``(key, normalized)``: ``key`` is the strategy function of the
    first φ-valued component of ``strategy`` (so the runner-up is ranked by
    the same quantity that selected the MIP), and ``normalized`` is whether
    that quantity is normalized φ. Falls back to raw φ when no component is
    φ-valued (e.g. a bare ``"NONE"`` or ``"PARTITION_LEX"``).
    """
    from pyphi.resolve_ties import phi_object_tie_resolution_strategies

    components = (strategy,) if isinstance(strategy, str) else tuple(strategy)
    for component in components:
        if component in _PHI_VALUED_STRATEGIES:
            return (
                phi_object_tie_resolution_strategies[component],
                component in _NORMALIZED_STRATEGIES,
            )
    return (lambda m: m.phi), False


def runner_up_from_candidates(
    candidates: Any,
    mip_value: Any,
    key: Any = None,
    normalized: bool = False,
) -> RunnerUp | None:
    """The candidate ranked next after the MIP under ``key``.

    ``key`` maps a candidate to the quantity minimized at MIP selection
    (raw φ by default; pass the result of :func:`sia_runner_up_key` to
    match a configured tie-resolution strategy). ``mip_value`` is the
    MIP's own key value. Candidates that tie the MIP (within
    :func:`pyphi.numerics.eq`) are tied peers, not runners-up, so they are
    excluded. Returns ``None`` when the MIP's value is unique. Candidates
    tied for next-best are ordered by ``partition.lex_key()`` so the choice
    does not depend on iteration order. Each candidate must expose
    ``.partition`` and whatever ``key`` reads (``.phi`` by default).
    """
    if key is None:
        key = lambda c: c.phi  # noqa: E731
    mip = float(mip_value)
    best = None
    for candidate in candidates:
        value = float(key(candidate))
        if value <= mip or numerics.eq(value, mip):
            continue  # the MIP itself or a tied peer, not a runner-up
        if best is None:
            best = candidate
            continue
        best_value = float(key(best))
        if numerics.eq(value, best_value):
            if candidate.partition.lex_key() < best.partition.lex_key():
                best = candidate
        elif value < best_value:
            best = candidate
    if best is None:
        return None
    return RunnerUp(
        partition=best.partition,
        phi=best.phi,
        normalized_phi=getattr(best, "normalized_phi", None) if normalized else None,
    )


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


def binding_direction_finding(cause_phi: Any, effect_phi: Any) -> Finding:
    """The Finding naming which direction binds ``min(φ_c, φ_e)``.

    Reports ``"TIED"`` (with no display tone) when the two values are equal up
    to ``config.numerics.precision``; otherwise ``"CAUSE"`` or ``"EFFECT"`` for
    the strictly smaller side.
    """
    if numerics.eq(float(cause_phi), float(effect_phi)):
        value, tone = "TIED", None
    elif float(cause_phi) < float(effect_phi):
        value, tone = "CAUSE", "cause"
    else:
        value, tone = "EFFECT", "effect"
    return Finding(
        kind="binding_direction",
        label="Binding direction",
        value=value,
        detail=(("φ_cause", cause_phi), ("φ_effect", effect_phi)),
        tone=tone,
    )


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
        """A tidy ``DataFrame`` with one row per finding.

        Columns: ``level``, ``kind``, ``label``, ``value``. A
        :class:`NullResultReason` value renders by its enum name.
        """
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
