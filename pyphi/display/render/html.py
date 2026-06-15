"""Styled-HTML backend for the display description vocabulary."""

from __future__ import annotations

from html import escape

from pyphi.display.description import Description
from pyphi.display.description import Inline
from pyphi.display.description import Nested
from pyphi.display.description import Row
from pyphi.display.description import Section
from pyphi.display.description import Table
from pyphi.display.numbers import format_value

_STYLE = """\
<style>
.pyphi-card{display:inline-block;border:1px solid #b0b0b0;border-radius:8px;
 font-family:ui-monospace,Menlo,Consolas,monospace;font-size:0.85em;margin:2px 0}
.pyphi-card .pyphi-title{font-weight:600;padding:4px 10px;
 border-bottom:1px solid #b0b0b0}
.pyphi-card .pyphi-subtitle{padding:2px 10px;color:#555}
.pyphi-card .pyphi-section{padding:4px 10px;border-top:1px solid #e0e0e0}
.pyphi-card .pyphi-section-label{font-weight:600;color:#444}
.pyphi-card table{border-collapse:collapse;margin:2px 0}
.pyphi-card td,.pyphi-card th{padding:1px 8px 1px 0;text-align:left}
.pyphi-card .pyphi-key{color:#444;padding-right:10px}
</style>"""


def _row_html(row: Row) -> str:
    extra = "".join(
        f' <span class="pyphi-extra">{escape(name)} {escape(format_value(val))}</span>'
        for name, val in row.extra
    )
    return (
        f'<tr><td class="pyphi-key">{escape(row.label)}</td>'
        f"<td>{escape(format_value(row.value))}{extra}</td></tr>"
    )


def _table_html(table: Table) -> str:
    head = "".join(f"<th>{escape(h)}</th>" for h in table.headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(format_value(c))}</td>" for c in row) + "</tr>"
        for row in table.rows
    )
    return f"<table><tr>{head}</tr>{body}</table>"


def _section_html(section: Section) -> str:
    parts = []
    if section.label:
        parts.append(f'<div class="pyphi-section-label">{escape(section.label)}</div>')
    if section.rows:
        parts.append(
            "<table>" + "".join(_row_html(r) for r in section.rows) + "</table>"
        )
    for comp in section.body:
        if isinstance(comp, Table):
            parts.append(_table_html(comp))
        elif isinstance(comp, Inline):
            parts.append(comp.html or f"<pre>{escape(comp.text)}</pre>")
        elif isinstance(comp, Nested):
            sub = comp.description
            label = sub.compact or sub.title
            parts.append(f'<div class="pyphi-nested">{escape(label)}</div>')
    return f'<div class="pyphi-section">{"".join(parts)}</div>'


def render(description: Description, verbosity: int) -> str:  # noqa: ARG001
    """Render a description as a styled HTML card (style block included)."""
    parts = [_STYLE, '<div class="pyphi-card">']
    parts.append(f'<div class="pyphi-title">{escape(description.title)}</div>')
    if description.subtitle:
        parts.append(f'<div class="pyphi-subtitle">{escape(description.subtitle)}</div>')
    parts.extend(_section_html(section) for section in description.sections)
    parts.append("</div>")
    return "".join(parts)
