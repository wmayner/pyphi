# models/diff.py
"""Structured deltas between two results (``result.diff()``).

``a.diff(b)`` returns a :class:`ResultDiff`: the signed Δφ, whether the MIP
genuinely changed (not a co-optimal tie-reshuffle), the distinctions /
relations / account-links gained, lost, or changed, and — composing
:meth:`pyphi.conf.snapshot.ConfigSnapshot.diff` — which config differences
could explain the change. Pairs with :mod:`pyphi.models.explanation`:
``explain`` says why one result is what it is; ``diff`` says what changed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pyphi.display import Description
from pyphi.display import Displayable
from pyphi.display import Row
from pyphi.display import Section
from pyphi.display import Table


@dataclass(frozen=True)
class Change:
    """One element-level delta between two results.

    ``kind`` is a stable machine key (``"distinction_gained"`` /
    ``"distinction_lost"`` / ``"distinction_changed"`` / ``"relation_gained"`` /
    ``"relation_lost"`` / ``"link_gained"`` / ``"link_lost"`` /
    ``"link_changed"`` / ``"purview_changed"``); ``key`` identifies the element
    (mechanism, relata, or link); ``a_value`` / ``b_value`` are the per-side
    quantities (``None`` on the side where the element is absent); ``tone`` is
    an optional semantic accent that HTML rendering colors.
    """

    kind: str
    key: Any
    a_value: Any = None
    b_value: Any = None
    tone: str | None = None


@dataclass(frozen=True)
class ResultDiff(Displayable):
    """A typed delta from result ``a`` to result ``b`` (``a.diff(b)``).

    ``subject`` names the delta (*e.g.* ``"ΔΦ_s = +0.10"``); ``level`` is
    ``"system"`` or ``"mechanism"``; ``delta_phi`` is ``b``'s ordering value
    minus ``a``'s (φ for IIT, alpha for actual causation); ``mip_changed`` is
    ``True`` only for a genuine MIP change, not a co-optimal tie-reshuffle;
    ``binding_direction_changed`` is ``None`` where the concept does not apply;
    ``changes`` is the element-level delta; ``config_diff`` maps the differing
    config paths to their ``(a, b)`` values; ``substrate_note`` records a
    substrate mismatch when the two results come from different substrates.
    """

    subject: str
    level: str
    delta_phi: Any
    mip_changed: bool
    binding_direction_changed: bool | None = None
    changes: tuple[Change, ...] = ()
    config_diff: dict[str, tuple[Any, Any]] | None = None
    substrate_note: str | None = None

    def _describe(self, verbosity: int) -> Description:  # noqa: ARG002
        summary = [
            Row("Δφ", self.delta_phi),
            Row("MIP changed", self.mip_changed),
        ]
        if self.binding_direction_changed is not None:
            summary.append(
                Row("Binding direction changed", self.binding_direction_changed)
            )
        if self.substrate_note:
            summary.append(Row("Note", self.substrate_note))
        sections = [Section(label="Summary", rows=tuple(summary))]
        if self.changes:
            sections.append(
                Section(
                    label="Changes",
                    body=(
                        Table(
                            headers=("Change", "Key", "a", "b"),
                            rows=tuple(
                                (c.kind, c.key, c.a_value, c.b_value)
                                for c in self.changes
                            ),
                            row_tones=tuple(
                                (c.tone, None, None, None) for c in self.changes
                            ),
                        ),
                    ),
                )
            )
        if self.config_diff:
            sections.append(
                Section(
                    label="Config differences",
                    body=(
                        Table(
                            headers=("Setting", "a", "b"),
                            rows=tuple(
                                (path, a, b) for path, (a, b) in self.config_diff.items()
                            ),
                        ),
                    ),
                )
            )
        return Description(
            title=self.subject, sections=tuple(sections), compact=self.subject
        )

    def to_pandas(self):
        """A tidy ``(category, key, a, b)`` frame of every delta."""
        from pyphi.models.pandas import records_to_frame

        rows: list[dict[str, Any]] = [
            {"category": "delta_phi", "key": None, "a": None, "b": self.delta_phi},
            {"category": "mip_changed", "key": None, "a": None, "b": self.mip_changed},
        ]
        rows.extend(
            {"category": c.kind, "key": c.key, "a": c.a_value, "b": c.b_value}
            for c in self.changes
        )
        if self.config_diff:
            rows.extend(
                {"category": "config", "key": path, "a": a, "b": b}
                for path, (a, b) in self.config_diff.items()
            )
        return records_to_frame(rows, columns=["category", "key", "a", "b"])
