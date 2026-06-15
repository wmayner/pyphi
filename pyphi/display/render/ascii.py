"""ASCII/Unicode boxed-card backend for the display description vocabulary."""

from __future__ import annotations

from pyphi.display.description import Row
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
