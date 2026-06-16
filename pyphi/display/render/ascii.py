"""ASCII/Unicode boxed-card backend for the display description vocabulary."""

from __future__ import annotations

from pyphi.display.description import Description
from pyphi.display.description import Inline
from pyphi.display.description import Nested
from pyphi.display.description import Row
from pyphi.display.description import Section
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
    lines = [
        "   ".join(_pad(row[c], widths[c]) for c in range(ncols)).rstrip()
        for row in cells
    ]
    if table.overflow:
        lines.append(f"… {table.overflow} more")
    return lines


def _compact(description: Description) -> str:
    """One-line summary for a nested child."""
    if description.compact is not None:
        return description.compact
    if description.subtitle:
        return f"{description.title}  {description.subtitle}"
    return description.title


def _section_lines(section: Section) -> list[str]:
    """Flatten a section's rows and body components into content lines."""
    lines = list(_format_rows(section.rows))
    for comp in section.body:
        if isinstance(comp, Table):
            lines.extend(_format_table(comp))
        elif isinstance(comp, Inline):
            lines.extend(comp.text.splitlines())
        elif isinstance(comp, Nested):
            lines.append(_compact(comp.description))
    return lines


def _top_border(title: str, inner_width: int) -> str:
    prefix = f"{TL}{H} {title} "
    fill = inner_width + 4 - _vis_len(prefix) - 1
    return f"{prefix}{H * max(fill, 0)}{TR}"


def _divider(label: str | None, inner_width: int) -> str:
    if label is None:
        return f"{ML}{H * (inner_width + 2)}{MR}"
    prefix = f"{ML}{H} {label} "
    fill = inner_width + 4 - _vis_len(prefix) - 1
    return f"{prefix}{H * max(fill, 0)}{MR}"


def render(description: Description, verbosity: int) -> str:  # noqa: ARG001
    """Render a description as a boxed card.

    If the description has no sections, renders as a single compact line.
    """
    if not description.sections:
        return description.compact or description.title

    blocks: list[tuple[str | None, list[str]]] = []
    if description.subtitle:
        blocks.append((None, [description.subtitle]))
    for section in description.sections:
        if section.label is None and not blocks:
            blocks.append((None, _section_lines(section)))
        else:
            blocks.append((section.label, _section_lines(section)))

    content_lines = [line for _, block in blocks for line in block]
    inner = max(
        [_vis_len(line) for line in content_lines] + [_vis_len(description.title) + 2],
        default=0,
    )

    out = [_top_border(description.title, inner)]
    for idx, (label, block) in enumerate(blocks):
        if not (idx == 0 and label is None):
            out.append(_divider(label, inner))
        out.extend(_framed(line, inner) for line in block)
    out.append(f"{BL}{H * (inner + 2)}{BR}")
    return "\n".join(out)
