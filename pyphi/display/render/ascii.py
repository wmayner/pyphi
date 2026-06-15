"""ASCII/Unicode boxed-card backend for the display description vocabulary."""

from __future__ import annotations

from pyphi.display.description import Row
from pyphi.display.description import Table
from pyphi.display.numbers import format_value

H = "─"
V = "│"
TL, TR, BL, BR = "╭", "╮", "╰", "╯"
ML, MR = "├", "┤"


def _vis_len(text: str) -> int:
    """Visible length of a single-line string (one column per code point)."""
    return len(text)


def _pad(text: str, width: int) -> str:
    """Right-pad ``text`` with spaces to ``width`` (no truncation)."""
    pad = width - _vis_len(text)
    return text + " " * pad if pad > 0 else text


def _framed(content: str, inner_width: int) -> str:
    """Wrap one content line in vertical borders, padded to ``inner_width``."""
    return f"{V} {_pad(content, inner_width)} {V}"


def _format_rows(rows: tuple[Row, ...]) -> list[str]:
    """Render key/value rows with labels aligned to a common width."""
    if not rows:
        return []
    label_w = max(_vis_len(row.label) for row in rows)
    lines = []
    for row in rows:
        parts = [f"{_pad(row.label, label_w)}   {format_value(row.value)}"]
        for name, val in row.extra:
            parts.append(f"{name} {format_value(val)}")
        lines.append("   ".join(parts))
    return lines


def _format_table(table: Table) -> list[str]:
    """Render a table with each column padded to its widest cell."""
    cells = [list(table.headers)] + [
        [format_value(c) for c in row] for row in table.rows
    ]
    ncols = len(table.headers)
    widths = [max(_vis_len(row[c]) for row in cells) for c in range(ncols)]
    return [
        "   ".join(_pad(row[c], widths[c]) for c in range(ncols)).rstrip()
        for row in cells
    ]
