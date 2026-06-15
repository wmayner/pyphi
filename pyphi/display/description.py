"""Declarative, backend-independent description of how to display a result.

A result type's ``_describe()`` returns a ``Description``; a renderer turns it
into ASCII or HTML. This is the single source of truth for *what* to show.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Row:
    """One aligned key/value line with optional trailing extra fields."""

    label: str
    value: Any
    extra: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class Table:
    """A tabular list (distinctions, relations, account links)."""

    headers: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]


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


@dataclass(frozen=True)
class Description:
    """The full description of a displayable object."""

    title: str
    subtitle: str | None = None
    sections: tuple[Section, ...] = ()
    compact: str | None = None
