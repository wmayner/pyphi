"""Declarative, backend-independent description of how to display a result.

A result type's ``_describe()`` returns a ``Description``; a renderer turns it
into ASCII or HTML. This is the single source of truth for *what* to show.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def system_phi_label(config: Any) -> str:
    """The label for a system irreducibility value under ``config``'s formalism.

    Under IIT 3.0 the system-level irreducibility value *is* big phi, so it is
    labelled ``Φ``. Under IIT 4.0 it is φₛ, and ``Φ`` names a different
    quantity — the structure integrated information, the sum of φ over the
    Φ-structure's distinctions and relations. Labelling φₛ as Φ under IIT 4.0
    reports one quantity as if it were the other.

    Parameters
    ----------
    config : ConfigSnapshot or None
        The snapshot carried by the result being displayed. ``None`` falls back
        to the IIT 4.0 label.

    Returns
    -------
    str
        ``"Φ"`` under IIT 3.0, ``"φ_s"`` otherwise.
    """
    version = getattr(
        getattr(getattr(config, "formalism", None), "iit", None), "version", None
    )
    return "Φ" if version == "IIT_3_0" else "φ_s"


@dataclass(frozen=True)
class Row:
    """One aligned key/value line with optional trailing extra fields.

    ``tone`` is an optional semantic accent (``"cause"`` / ``"effect"``) that
    HTML rendering colors; the ASCII backend ignores it.
    """

    label: str
    value: Any
    extra: tuple[tuple[str, Any], ...] = ()
    tone: str | None = None


@dataclass(frozen=True)
class Table:
    """A tabular list (distinctions, relations, account links).

    ``overflow`` is the number of rows omitted from ``rows`` (the collection
    was larger than the display cap); renderers show a "… N more" indicator.
    """

    headers: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    overflow: int = 0
    grid: bool = False  # matrix-style (e.g. a cut grid): tight, center-aligned
    # Optional per-column semantic tone for the header cells (HTML colors them).
    header_tones: tuple[str | None, ...] = ()
    # Optional per-cell semantic tone, aligned with ``rows`` (HTML colors them).
    row_tones: tuple[tuple[str | None, ...], ...] = ()


@dataclass(frozen=True)
class Inline:
    """A pre-formatted fragment owned by the source type.

    ``text`` is the ASCII form; ``html`` optionally overrides the HTML form.
    """

    text: str
    html: str | None = None


@dataclass(frozen=True)
class Nested:
    """A child result rendered compactly (one line), never as a recursive box."""

    description: Description


Component = Row | Table | Inline | Nested


@dataclass(frozen=True)
class Section:
    """A named group rendered with a rule divider.

    ``rows`` are key/value lines; ``body`` holds richer components.
    """

    label: str | None = None
    rows: tuple[Row, ...] = ()
    body: tuple[Component, ...] = ()
    tone: str | None = None


@dataclass(frozen=True)
class Description:
    """The full description of a displayable object."""

    title: str
    subtitle: str | None = None
    sections: tuple[Section, ...] = ()
    compact: str | None = None
    tone: str | None = None
