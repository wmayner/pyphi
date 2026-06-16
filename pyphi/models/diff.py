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

from pyphi import utils
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


def _phi_of(result: Any) -> Any:
    """The scalar this result is ordered by (φ for IIT, alpha for AC)."""
    return getattr(result, "alpha", None) if hasattr(result, "alpha") else result.phi


def _mip_changed(a: Any, b: Any) -> bool:
    """True iff b's MIP partition is not one a could co-optimally have chosen.

    Uses ``lex_key`` over a's tie set (which already encodes EQUALITY_TOLERANCE
    from tie resolution). Falls back to a tolerance-aware lex_key inequality for
    results without a ``.ties`` set.
    """
    a_part = getattr(a, "partition", None)
    b_part = getattr(b, "partition", None)
    if a_part is None or b_part is None:
        return a_part is not b_part
    ties = getattr(a, "ties", None)
    if ties:
        a_tie_keys = {t.partition.lex_key() for t in ties if t.partition is not None}
        return b_part.lex_key() not in a_tie_keys
    # Fallback: a real change only if the partition differs AND phi differs.
    if b_part.lex_key() == a_part.lex_key():
        return False
    return not utils.eq(float(_phi_of(a)), float(_phi_of(b)))


def _config_diff(a: Any, b: Any) -> dict[str, tuple[Any, Any]]:
    a_cfg = getattr(a, "config", None)
    b_cfg = getattr(b, "config", None)
    if a_cfg is None or b_cfg is None:
        return {}
    return a_cfg.diff(b_cfg)


def _substrate_note(a: Any, b: Any) -> str | None:
    a_idx = getattr(a, "node_indices", None)
    b_idx = getattr(b, "node_indices", None)
    if a_idx is not None and b_idx is not None and a_idx != b_idx:
        return f"substrates differ ({a_idx} vs {b_idx}); deltas keyed by mechanism"
    return None


def _diff_common(a: Any, b: Any) -> dict[str, Any]:
    """Shared scalar deltas every result type's diff() reuses."""
    from pyphi.data_structures import PyPhiFloat

    delta = PyPhiFloat(float(_phi_of(b)) - float(_phi_of(a)))
    return {
        "delta_phi": delta,
        "mip_changed": _mip_changed(a, b),
        "config_diff": _config_diff(a, b),
        "substrate_note": _substrate_note(a, b),
    }
