"""Render a Provenance record as a display Section."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyphi.display.description import Row
from pyphi.display.description import Section

if TYPE_CHECKING:
    from pyphi.provenance import Provenance


def provenance_section(prov: Provenance) -> Section:
    """A 'Provenance' Section with one Row per recorded field."""
    return Section(
        label="Provenance",
        rows=tuple(Row(label, value) for label, value in prov.display_rows()),
    )
