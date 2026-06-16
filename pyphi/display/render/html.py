"""Styled-HTML backend for the display description vocabulary.

Unlike the ASCII backend, this renders HTML-native structure: a header with a
metric badge, sections as flex-wrapping panels (so cause/effect sit
side-by-side), key/value grids, and real ``<table>`` elements for collections.
"""

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
.pyphi-card{display:inline-block;background:#fff;color:#1f2328;
 border:1px solid #d0d7de;border-radius:10px;overflow:hidden;
 box-shadow:0 1px 3px rgba(0,0,0,.07);font-size:13px;
 font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.pyphi-leaf{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
 font-size:13px}
.pyphi-head{display:flex;align-items:baseline;justify-content:space-between;
 gap:18px;padding:7px 14px;background:#f6f8fa;border-bottom:1px solid #e4e8ec}
.pyphi-title{font-weight:600}
.pyphi-badge{background:#eef2ff;color:#3538cd;border-radius:6px;
 padding:2px 8px;font-size:12px;white-space:nowrap;
 font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.pyphi-body{display:block}
.pyphi-section{padding:8px 14px;border-top:1px solid #eef0f2}
.pyphi-label{font-weight:600;color:#57606a;font-size:10px;
 text-transform:uppercase;letter-spacing:.05em;margin-bottom:5px}
.pyphi-kv{display:grid;grid-template-columns:auto 1fr;gap:3px 14px}
.pyphi-k{color:#8b949e}
.pyphi-v{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.pyphi-extra{color:#8b949e;margin-left:10px;font-size:12px}
table.pyphi-table{border-collapse:collapse;width:100%;font-size:12px}
table.pyphi-table th{text-align:left;color:#57606a;font-weight:600;
 border-bottom:1px solid #d0d7de;padding:3px 12px 3px 0}
table.pyphi-table td{padding:3px 12px 3px 0;border-bottom:1px solid #f0f2f4;
 font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.pyphi-scroll{max-height:18em;overflow:auto}
.pyphi-more{color:#8b949e;font-size:12px;padding:3px 0}
.pyphi-cause{color:#D55C00}
.pyphi-effect{color:#009E73}
</style>"""


def _tone_cls(tone: str | None) -> str:
    """Space-prefixed CSS class for a semantic tone, for appending to a class."""
    return f" pyphi-{tone}" if tone in ("cause", "effect") else ""


def _value_html(value: object, extra: tuple[tuple[str, object], ...]) -> str:
    parts = [f'<span class="pyphi-v">{escape(format_value(value))}</span>']
    for name, val in extra:
        parts.append(
            f'<span class="pyphi-extra">{escape(name)} '
            f"{escape(format_value(val))}</span>"
        )
    return "".join(parts)


def _kv_html(rows: tuple[Row, ...]) -> str:
    cells = []
    for row in rows:
        cells.append(f'<span class="pyphi-k">{escape(row.label)}</span>')
        val = _value_html(row.value, row.extra)
        cells.append(f'<span class="pyphi-vcell{_tone_cls(row.tone)}">{val}</span>')
    return f'<div class="pyphi-kv">{"".join(cells)}</div>'


def _table_html(table: Table) -> str:
    head = "".join(f"<th>{escape(h)}</th>" for h in table.headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(format_value(c))}</td>" for c in row) + "</tr>"
        for row in table.rows
    )
    html = (
        f'<div class="pyphi-scroll">'
        f'<table class="pyphi-table"><tr>{head}</tr>{body}</table></div>'
    )
    if table.overflow:
        html += f'<div class="pyphi-more">… {table.overflow} more</div>'
    return html


def _section_html(section: Section) -> str:
    parts = []
    if section.label:
        cls = f"pyphi-label{_tone_cls(section.tone)}"
        parts.append(f'<div class="{cls}">{escape(section.label)}</div>')
    if section.rows:
        parts.append(_kv_html(section.rows))
    for comp in section.body:
        if isinstance(comp, Table):
            parts.append(_table_html(comp))
        elif isinstance(comp, Inline):
            parts.append(comp.html or f"<pre>{escape(comp.text)}</pre>")
        elif isinstance(comp, Nested):
            sub = comp.description
            label = sub.compact or sub.title
            parts.append(f'<span class="pyphi-extra">{escape(label)}</span>')
    return f'<div class="pyphi-section">{"".join(parts)}</div>'


def render(description: Description, verbosity: int) -> str:  # noqa: ARG001
    """Render a description as an HTML-native styled card.

    If the description has no sections, renders as a small inline leaf element
    instead of a full card.
    """
    if not description.sections:
        text = description.compact or description.title
        return _STYLE + f'<span class="pyphi-leaf">{escape(text)}</span>'

    title_cls = f"pyphi-title{_tone_cls(description.tone)}"
    head = [f'<span class="{title_cls}">{escape(description.title)}</span>']
    if description.subtitle:
        head.append(f'<span class="pyphi-badge">{escape(description.subtitle)}</span>')
    sections = "".join(_section_html(section) for section in description.sections)
    return (
        _STYLE
        + '<div class="pyphi-card">'
        + f'<div class="pyphi-head">{"".join(head)}</div>'
        + f'<div class="pyphi-body">{sections}</div></div>'
    )
