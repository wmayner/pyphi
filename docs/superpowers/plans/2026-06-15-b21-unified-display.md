# B21 — Unified Object Display Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace PyPhi's scattered `fmt.py`-based object formatting with one declarative `_describe()` description per result type, rendered by pluggable ASCII and styled-HTML backends, giving every user-facing result a consistent, redesigned repr/str/HTML.

**Architecture:** A new top-level `pyphi/display/` package defines a frozen-dataclass description vocabulary, an ASCII (boxed-card) backend, a styled-HTML backend, and a `Displayable` mixin that wires `__repr__`/`__str__`/`_repr_html_`/`_repr_mimebundle_` to `_describe()` + the active backend (the one site that reads `repr_verbosity`). Each result type implements `_describe()`; the high-level `fmt_*(obj)` composers are deleted and the low-level string primitives move into the ASCII backend.

**Tech Stack:** Python 3.12+, frozen dataclasses, pytest (`uv run pytest`), pyright, ruff. No new runtime dependencies (the `rich` backend is deferred behind the backend seam).

**Spec:** `docs/superpowers/specs/2026-06-15-b21-unified-display-design.md`

---

## Conventions for the executor

- Always run Python via `uv run` (e.g. `uv run pytest`, `uv run python`).
- `repr_verbosity` is read **only** through `config.infrastructure.repr_verbosity`. The constants are `LOW=0`, `MEDIUM=1`, `HIGH=2` (default `2`).
- Commit after every task with the message shown in that task's final step. Stage **only** the files that task touched — the working tree contains unrelated untracked files from other contributors; never `git add -A`, never revert files you did not change.
- Do **not** run `git push`; do **not** pass `--no-verify`. If a commit is blocked by a pre-commit hook, read the hook output, fix the cause, re-stage, and re-commit.
- The "redesign output" decision is intentional: rendered strings change. Golden snapshots are captured *from actual output* once the look is settled (Task 16), so the mockups in the spec are illustrative, not assertions.

---

## File structure

| File | Responsibility |
|---|---|
| `pyphi/display/__init__.py` | Public exports: `Displayable`, the description vocabulary, `render` |
| `pyphi/display/numbers.py` | `format_value()` — display rounding |
| `pyphi/display/description.py` | Frozen-dataclass vocabulary (`Description`, `Section`, `Row`, `Table`, `Inline`, `Nested`) |
| `pyphi/display/render/__init__.py` | Backend registry + `render(description, fmt="ascii"/"html", verbosity)` |
| `pyphi/display/render/ascii.py` | ASCII/Unicode boxed-card backend |
| `pyphi/display/render/html.py` | Styled-HTML backend |
| `pyphi/display/mixin.py` | `Displayable` mixin; the only `repr_verbosity` read site |
| `test/test_display.py` | Unit tests for vocabulary/backends/mixin + per-type golden snapshots |
| (migrated) result-type modules | gain `_describe()`, inherit `Displayable`, drop ad-hoc reprs |
| `pyphi/models/fmt.py` | low-level primitives migrate to `render/ascii.py`; high-level composers deleted |

---

## Phase 0 — Package scaffold & number formatting

### Task 1: Create the `pyphi/display/` package and `format_value`

**Files:**
- Create: `pyphi/display/__init__.py`
- Create: `pyphi/display/numbers.py`
- Test: `test/test_display.py`

- [ ] **Step 1: Write the failing test**

```python
# test/test_display.py
import numpy as np

from pyphi.display.numbers import format_value


def test_format_value_rounds_floats_to_6_sig_figs():
    assert format_value(0.41503749927884376) == "0.415037"


def test_format_value_handles_numpy_scalars():
    assert format_value(np.float64(3.0)) == "3"


def test_format_value_passes_through_non_numbers():
    assert format_value((1, 0, 0)) == "(1, 0, 0)"
    assert format_value(None) == "None"
    assert format_value("A,B,C") == "A,B,C"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_display.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pyphi.display'`

- [ ] **Step 3: Write minimal implementation**

```python
# pyphi/display/__init__.py
"""Unified object display: declarative descriptions rendered by pluggable backends."""
```

```python
# pyphi/display/numbers.py
"""Display-time formatting of numeric values."""

from numbers import Real

SIG_FIGS = 6


def format_value(value, sig_figs: int = SIG_FIGS) -> str:
    """Format a value for display.

    Real numbers are rounded to ``sig_figs`` significant figures; everything
    else is rendered with ``str``. The exact numeric value remains available on
    the source object's attribute.
    """
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, Real):
        return f"{float(value):.{sig_figs}g}"
    return str(value)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_display.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add pyphi/display/__init__.py pyphi/display/numbers.py test/test_display.py
git commit -m "Add pyphi.display package scaffold and format_value"
```

---

## Phase 1 — Description vocabulary

### Task 2: Define the description dataclasses

**Files:**
- Create: `pyphi/display/description.py`
- Test: `test/test_display.py`

- [ ] **Step 1: Write the failing test**

```python
# append to test/test_display.py
from pyphi.display.description import (
    Description,
    Inline,
    Nested,
    Row,
    Section,
    Table,
)


def test_description_is_frozen_and_composes():
    d = Description(
        title="Demo",
        subtitle="x 1",
        sections=(
            Section(label=None, rows=(Row("System", "A,B,C"),)),
            Section(label="Cause", rows=(Row("purview", "(1,1,0)", (("II_c", 3.0),)),)),
            Section(label="List", body=(Table(headers=("a", "b"), rows=(("1", "2"),)),)),
        ),
        compact="Demo(x=1)",
    )
    assert d.title == "Demo"
    assert d.sections[1].rows[0].extra == (("II_c", 3.0),)
    assert d.sections[2].body[0].headers == ("a", "b")
    import dataclasses

    with __import__("pytest").raises(dataclasses.FrozenInstanceError):
        d.title = "no"  # type: ignore[misc]


def test_nested_and_inline_are_components():
    inner = Description(title="Inner", compact="Inner()")
    s = Section(label=None, body=(Nested(inner), Inline(text="x ─── y")))
    assert isinstance(s.body[0], Nested)
    assert s.body[1].text == "x ─── y"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_display.py -k description -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pyphi.display.description'`

- [ ] **Step 3: Write minimal implementation**

```python
# pyphi/display/description.py
"""Declarative, backend-independent description of how to display a result.

A result type's ``_describe()`` returns a ``Description``; a renderer turns it
into ASCII or HTML. This is the single source of truth for *what* to show.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union


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

    description: "Description"


Component = Union[Row, Table, Inline, Nested]


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
```

(`field` import is unused; remove it — kept here only to remind you ruff will flag unused imports. Final file imports only `dataclass`, `Any`, `Union`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_display.py -k "description or nested" -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add pyphi/display/description.py test/test_display.py
git commit -m "Add display description vocabulary"
```

---

## Phase 2 — ASCII backend

The ASCII backend renders a `Description` as a rounded-corner boxed card with
sectioned dividers. Build it in layers (helpers → rows → sections/tables →
assemble) so each piece is testable.

### Task 3: ASCII layout helpers

**Files:**
- Create: `pyphi/display/render/__init__.py` (empty for now — just makes `render` a package)
- Create: `pyphi/display/render/ascii.py`
- Test: `test/test_display.py`

- [ ] **Step 1: Write the failing test**

```python
# append to test/test_display.py
from pyphi.display.render import ascii as ascii_backend


def test_ascii_visual_len_counts_codepoints():
    assert ascii_backend._vis_len("φ_s") == 3
    assert ascii_backend._vis_len("A,B,C") == 5


def test_ascii_pad_right():
    assert ascii_backend._pad("ab", 5) == "ab   "
    assert ascii_backend._pad("abcde", 3) == "abcde"


def test_ascii_framed_line_has_borders_and_width():
    line = ascii_backend._framed("hi", 6)
    assert line.startswith("│ ") and line.endswith(" │")
    assert ascii_backend._vis_len(line) == 6 + 4  # content width + "│ " + " │"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_display.py -k ascii -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pyphi.display.render'`

- [ ] **Step 3: Write minimal implementation**

```python
# pyphi/display/render/__init__.py
"""Display backends."""
```

```python
# pyphi/display/render/ascii.py
"""ASCII/Unicode boxed-card backend for the display description vocabulary."""

from __future__ import annotations

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_display.py -k ascii -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyphi/display/render/__init__.py pyphi/display/render/ascii.py test/test_display.py
git commit -m "Add ASCII backend layout helpers"
```

### Task 4: ASCII row & section formatting

**Files:**
- Modify: `pyphi/display/render/ascii.py`
- Test: `test/test_display.py`

- [ ] **Step 1: Write the failing test**

```python
# append to test/test_display.py
def test_ascii_format_rows_aligns_labels():
    from pyphi.display.description import Row

    rows = (Row("System", "A,B,C"), Row("φ_s", 0.41503749927884376))
    lines = ascii_backend._format_rows(rows)
    # labels right-padded to the same width; values formatted
    assert lines[0] == "System   A,B,C"
    assert lines[1] == "φ_s      0.415037"


def test_ascii_row_extra_fields_appended():
    from pyphi.display.description import Row

    rows = (Row("purview", "(1,1,0)", (("II_c", 3.0), ("int.diff", 0.0))),)
    lines = ascii_backend._format_rows(rows)
    assert lines[0] == "purview   (1,1,0)   II_c 3   int.diff 0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_display.py -k "format_rows or extra_fields" -v`
Expected: FAIL — `AttributeError: module ... has no attribute '_format_rows'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to pyphi/display/render/ascii.py
from pyphi.display.description import Row
from pyphi.display.numbers import format_value


def _format_rows(rows: tuple[Row, ...]) -> list[str]:
    """Render key/value rows with labels aligned to a common width."""
    if not rows:
        return []
    label_w = max(_vis_len(r.label) for r in rows)
    lines = []
    for r in rows:
        parts = [f"{_pad(r.label, label_w)}   {format_value(r.value)}"]
        for name, val in r.extra:
            parts.append(f"{name} {format_value(val)}")
        lines.append("   ".join(parts))
    return lines
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_display.py -k "format_rows or extra_fields" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyphi/display/render/ascii.py test/test_display.py
git commit -m "Add ASCII row formatting with aligned labels"
```

### Task 5: ASCII table formatting

**Files:**
- Modify: `pyphi/display/render/ascii.py`
- Test: `test/test_display.py`

- [ ] **Step 1: Write the failing test**

```python
# append to test/test_display.py
def test_ascii_format_table_aligns_columns():
    from pyphi.display.description import Table

    t = Table(headers=("dir", "purview", "α"), rows=(("CAUSE", "OR", 0.415037),
                                                      ("EFFECT", "AND", 0.415037)))
    lines = ascii_backend._format_table(t)
    assert lines[0] == "dir      purview   α"
    assert lines[1] == "CAUSE    OR        0.415037"
    assert lines[2] == "EFFECT   AND       0.415037"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_display.py -k format_table -v`
Expected: FAIL — no attribute `_format_table`

- [ ] **Step 3: Write minimal implementation**

```python
# add to pyphi/display/render/ascii.py
from pyphi.display.description import Table


def _format_table(table: Table) -> list[str]:
    """Render a table with each column padded to its widest cell."""
    cells = [list(table.headers)] + [
        [format_value(c) for c in row] for row in table.rows
    ]
    ncols = len(table.headers)
    widths = [max(_vis_len(row[c]) for row in cells) for c in range(ncols)]
    lines = []
    for row in cells:
        lines.append("   ".join(_pad(row[c], widths[c]) for c in range(ncols)).rstrip())
    return lines
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_display.py -k format_table -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyphi/display/render/ascii.py test/test_display.py
git commit -m "Add ASCII table formatting"
```

### Task 6: ASCII card assembly (`render`)

**Files:**
- Modify: `pyphi/display/render/ascii.py`
- Test: `test/test_display.py`

- [ ] **Step 1: Write the failing test**

```python
# append to test/test_display.py
def test_ascii_render_full_card():
    from pyphi.display.description import Description, Inline, Row, Section, Table

    d = Description(
        title="Demo",
        subtitle="φ_s 0.415037",
        sections=(
            Section(label=None, rows=(Row("System", "A,B,C"),)),
            Section(label="Cause", rows=(Row("purview", "(1,1,0)", (("II_c", 3.0),)),)),
            Section(label="Links", body=(Table(("a", "b"), (("1", "2"),)),)),
            Section(label="MIP", body=(Inline("{A,BC}"),)),
        ),
    )
    out = ascii_backend.render(d, verbosity=2)
    lines = out.splitlines()
    # top border carries the title; every line is the same visible width
    assert lines[0].startswith("╭─ Demo ") and lines[0].endswith("╮")
    assert lines[-1].startswith("╰") and lines[-1].endswith("╯")
    widths = {ascii_backend._vis_len(line) for line in lines}
    assert len(widths) == 1  # all lines equal width
    assert "├─ Cause " in out
    assert "II_c 3" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_display.py -k render_full_card -v`
Expected: FAIL — no attribute `render`

- [ ] **Step 3: Write minimal implementation**

```python
# add to pyphi/display/render/ascii.py
from pyphi.display.description import Description, Inline, Nested, Section


def _section_lines(section: Section) -> list[str]:
    lines = list(_format_rows(section.rows))
    for comp in section.body:
        if isinstance(comp, Table):
            lines.extend(_format_table(comp))
        elif isinstance(comp, Inline):
            lines.extend(comp.text.splitlines())
        elif isinstance(comp, Nested):
            lines.append(_compact(comp.description))
    return lines


def _compact(description: Description) -> str:
    """One-line summary for a nested child."""
    if description.compact is not None:
        return description.compact
    if description.subtitle:
        return f"{description.title}  {description.subtitle}"
    return description.title


def _divider(label: str | None, inner_width: int) -> str:
    if label is None:
        return f"{ML}{H * (inner_width + 2)}{MR}"
    prefix = f"{ML}{H} {label} "
    fill = inner_width + 4 - _vis_len(prefix) - 1
    return f"{prefix}{H * max(fill, 0)}{MR}"


def _top_border(title: str, inner_width: int) -> str:
    prefix = f"{TL}{H} {title} "
    fill = inner_width + 4 - _vis_len(prefix) - 1
    return f"{prefix}{H * max(fill, 0)}{TR}"


def render(description: Description, verbosity: int) -> str:
    """Render a description as a boxed card."""
    blocks: list[tuple[str | None, list[str]]] = []
    if description.subtitle:
        blocks.append((None, [description.subtitle]))
    for i, section in enumerate(description.sections):
        # the first label-less section shares the header block (no divider)
        label = section.label
        if label is None and not blocks:
            blocks.append((None, _section_lines(section)))
        else:
            blocks.append((label, _section_lines(section)))

    content_lines = [ln for _, block in blocks for ln in block]
    inner = max(
        [_vis_len(ln) for ln in content_lines]
        + [_vis_len(description.title) + 2],
        default=0,
    )

    out = [_top_border(description.title, inner)]
    for idx, (label, block) in enumerate(blocks):
        if idx == 0 and label is None:
            pass  # header block: no divider
        else:
            out.append(_divider(label, inner))
        out.extend(_framed(ln, inner) for ln in block)
    out.append(f"{BL}{H * (inner + 2)}{BR}")
    return "\n".join(out)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_display.py -k render_full_card -v`
Expected: PASS. If width assertions fail, adjust `_divider`/`_top_border` fill arithmetic until all lines share one width — this is the invariant to satisfy.

- [ ] **Step 5: Commit**

```bash
git add pyphi/display/render/ascii.py test/test_display.py
git commit -m "Add ASCII boxed-card assembly"
```

---

## Phase 3 — HTML backend

### Task 7: Styled HTML backend

**Files:**
- Create: `pyphi/display/render/html.py`
- Test: `test/test_display.py`

- [ ] **Step 1: Write the failing test**

```python
# append to test/test_display.py
from pyphi.display.render import html as html_backend


def test_html_render_has_scoped_panel_and_escapes():
    from pyphi.display.description import Description, Row, Section

    d = Description(
        title="Demo<>",
        subtitle="φ_s 0.415037",
        sections=(Section(label="Cause", rows=(Row("purview", "(1,1,0) & <x>"),)),),
    )
    out = html_backend.render(d, verbosity=2)
    assert 'class="pyphi-card"' in out
    assert "pyphi-section" in out
    assert "Demo&lt;&gt;" in out          # title escaped
    assert "&lt;x&gt;" in out             # value escaped
    assert "<style" in out                # style block present


def test_html_style_injected_once_per_render():
    from pyphi.display.description import Description

    out = html_backend.render(Description(title="A"), verbosity=2)
    assert out.count("<style") == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_display.py -k html -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pyphi.display.render.html'`

- [ ] **Step 3: Write minimal implementation**

```python
# pyphi/display/render/html.py
"""Styled-HTML backend for the display description vocabulary."""

from __future__ import annotations

from html import escape

from pyphi.display.description import (
    Description,
    Inline,
    Nested,
    Row,
    Section,
    Table,
)
from pyphi.display.numbers import format_value

_STYLE = """\
<style>
.pyphi-card{display:inline-block;border:1px solid #b0b0b0;border-radius:8px;
 font-family:ui-monospace,Menlo,Consolas,monospace;font-size:0.85em;margin:2px 0}
.pyphi-card .pyphi-title{font-weight:600;padding:4px 10px;border-bottom:1px solid #b0b0b0}
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
        parts.append("<table>" + "".join(_row_html(r) for r in section.rows) + "</table>")
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


def render(description: Description, verbosity: int) -> str:
    """Render a description as a styled HTML card (style block included)."""
    parts = [_STYLE, '<div class="pyphi-card">']
    parts.append(f'<div class="pyphi-title">{escape(description.title)}</div>')
    if description.subtitle:
        parts.append(f'<div class="pyphi-subtitle">{escape(description.subtitle)}</div>')
    for section in description.sections:
        parts.append(_section_html(section))
    parts.append("</div>")
    return "".join(parts)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_display.py -k html -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyphi/display/render/html.py test/test_display.py
git commit -m "Add styled HTML display backend"
```

### Task 8: Backend registry

**Files:**
- Modify: `pyphi/display/render/__init__.py`
- Test: `test/test_display.py`

- [ ] **Step 1: Write the failing test**

```python
# append to test/test_display.py
from pyphi.display import render as render_pkg


def test_render_dispatches_by_backend_name():
    from pyphi.display.description import Description

    d = Description(title="Demo")
    assert render_pkg.render(d, backend="ascii", verbosity=2).startswith("╭")
    assert 'class="pyphi-card"' in render_pkg.render(d, backend="html", verbosity=2)


def test_render_unknown_backend_raises():
    from pyphi.display.description import Description

    with __import__("pytest").raises(KeyError):
        render_pkg.render(Description(title="x"), backend="rich", verbosity=2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_display.py -k "dispatches or unknown_backend" -v`
Expected: FAIL — `render` package has no `render` function

- [ ] **Step 3: Write minimal implementation**

```python
# pyphi/display/render/__init__.py
"""Display backends and the backend registry."""

from pyphi.display.description import Description
from pyphi.display.render import ascii as _ascii
from pyphi.display.render import html as _html

_BACKENDS = {"ascii": _ascii.render, "html": _html.render}


def render(description: Description, backend: str = "ascii", verbosity: int = 2) -> str:
    """Render ``description`` with the named backend.

    A future ``rich`` backend registers here without touching call sites.
    """
    return _BACKENDS[backend](description, verbosity)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_display.py -k "dispatches or unknown_backend" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyphi/display/render/__init__.py test/test_display.py
git commit -m "Add display backend registry"
```

---

## Phase 4 — Displayable mixin

### Task 9: `Displayable` mixin

**Files:**
- Create: `pyphi/display/mixin.py`
- Modify: `pyphi/display/__init__.py` (export the public surface)
- Test: `test/test_display.py`

- [ ] **Step 1: Write the failing test**

```python
# append to test/test_display.py
import pyphi
from pyphi.display import Displayable
from pyphi.display.description import Description, Row, Section


class _Demo(Displayable):
    def _describe(self, verbosity):
        return Description(
            title="Demo",
            subtitle="x 1",
            sections=(Section(label=None, rows=(Row("x", 1),)),),
            compact="Demo(x=1)",
        )


def test_displayable_repr_and_str_match_ascii_card():
    obj = _Demo()
    assert repr(obj) == str(obj)
    assert repr(obj).startswith("╭─ Demo")


def test_displayable_low_verbosity_uses_compact():
    obj = _Demo()
    with pyphi.config.override(repr_verbosity=0):
        assert repr(obj) == "Demo(x=1)"


def test_displayable_html_and_mimebundle():
    obj = _Demo()
    assert 'class="pyphi-card"' in obj._repr_html_()
    bundle = obj._repr_mimebundle_()
    assert set(bundle) == {"text/plain", "text/html"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_display.py -k displayable -v`
Expected: FAIL — `ImportError: cannot import name 'Displayable'`

- [ ] **Step 3: Write minimal implementation**

```python
# pyphi/display/mixin.py
"""The Displayable mixin: the single site that reads repr_verbosity."""

from __future__ import annotations

from pyphi.conf import config
from pyphi.display.description import Description
from pyphi.display.render import render

LOW, MEDIUM, HIGH = 0, 1, 2


def _verbosity() -> int:
    return config.infrastructure.repr_verbosity


class Displayable:
    """Provides repr/str/HTML from a subclass ``_describe()`` hook."""

    def _describe(self, verbosity: int) -> Description:  # pragma: no cover
        raise NotImplementedError

    def __repr__(self) -> str:
        v = _verbosity()
        description = self._describe(v)
        if v == LOW and description.compact is not None:
            return description.compact
        return render(description, backend="ascii", verbosity=v)

    __str__ = __repr__

    def _repr_html_(self) -> str:
        v = _verbosity()
        return render(self._describe(v), backend="html", verbosity=v)

    def _repr_mimebundle_(self, include=None, exclude=None):
        return {"text/plain": str(self), "text/html": self._repr_html_()}
```

```python
# pyphi/display/__init__.py  (replace the module docstring file with this)
"""Unified object display: declarative descriptions rendered by pluggable backends."""

from pyphi.display.description import (
    Description,
    Inline,
    Nested,
    Row,
    Section,
    Table,
)
from pyphi.display.mixin import Displayable
from pyphi.display.render import render

__all__ = [
    "Description",
    "Displayable",
    "Inline",
    "Nested",
    "Row",
    "Section",
    "Table",
    "render",
]
```

Note: confirm `from pyphi.conf import config` is the correct import (check an existing module, e.g. `pyphi/models/fmt.py` line ~10, which currently does `from pyphi.conf import config` or similar). Match the existing pattern exactly to avoid a circular import; if `pyphi.conf` import at module load causes a cycle, import `config` lazily inside `_verbosity()`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_display.py -k displayable -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyphi/display/mixin.py pyphi/display/__init__.py test/test_display.py
git commit -m "Add Displayable mixin wiring repr/str/html to _describe"
```

---

## Phase 5 — First type end-to-end (IIT 4.0 SIA)

This proves the vocabulary on the flagship type and locks the visual language
before the batch migration.

### Task 10: Migrate IIT 4.0 `SystemIrreducibilityAnalysis`

**Files:**
- Modify: `pyphi/formalism/iit4/__init__.py` (class around line 286–329 — replace `_repr_columns`, `_repr_html_`, `__repr__`)
- Test: `test/test_display.py`

Faithful field mapping (from the current `_repr_columns` at `iit4/__init__.py:286` and the live capture): System (node labels), Current state, φ_s, Normalized φ_s, Int. diff. CAUSE, Int. diff. EFFECT, the `system_state` columns (CAUSE purview/state + II_c, EFFECT purview/state + II_e — from `self.system_state._repr_columns()`), #(tied MIPs) = `len(self.ties) - 1`, the partition (as an `Inline`), and Reasons if present.

- [ ] **Step 1: Write the failing test**

```python
# append to test/test_display.py
def test_iit4_sia_describe_structure():
    s = pyphi.examples.basic_system()
    sia = s.sia()
    d = sia._describe(2)
    assert d.title == "SystemIrreducibilityAnalysis"
    labels = [r.label for sec in d.sections for r in sec.rows]
    assert "System" in labels
    assert "φ_s" in labels
    section_labels = [sec.label for sec in d.sections]
    assert "Cause" in section_labels and "Effect" in section_labels
    out = repr(sia)
    assert out.startswith("╭─ SystemIrreducibilityAnalysis")
    assert "0.41503749927884376" not in out  # numbers are rounded
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_display.py -k iit4_sia_describe -v`
Expected: FAIL — `_describe` not defined / `AttributeError`

- [ ] **Step 3: Write the implementation**

Make the class inherit `Displayable` (add to its bases) and replace the three
display methods with a single `_describe`. Read the surrounding class first to
confirm attribute names (`self.phi`, `self.normalized_phi`,
`self.intrinsic_differentiation`, `self.system_state`, `self.current_state`,
`self.node_indices`, `self.node_labels` or how labels are produced in
`fmt_sia_columns`, `self.partition`, `self.ties`, `self.reasons`).

```python
# in pyphi/formalism/iit4/__init__.py, on the SIA class
from pyphi.display import Description, Displayable, Inline, Row, Section
from pyphi.core.direction import Direction  # use the existing Direction import in this file

def _describe(self, verbosity):
    idiff = self.intrinsic_differentiation
    cause_rows = (
        Row("purview", self.system_state.cause if self.system_state else None,
            (("II_c", self.system_state.ii_cause if self.system_state else None),
             ("int.diff", idiff[Direction.CAUSE] if idiff else None))),
    )
    effect_rows = (
        Row("purview", self.system_state.effect if self.system_state else None,
            (("II_e", self.system_state.ii_effect if self.system_state else None),
             ("int.diff", idiff[Direction.EFFECT] if idiff else None))),
    )
    header_rows = (
        Row("System", <node-labels-expression>),
        Row("Current state", self.current_state),
        Row("φ_s", self.phi, (("norm", self.normalized_phi),)),
    )
    sections = [
        Section(label=None, rows=header_rows),
        Section(label="Cause", rows=cause_rows),
        Section(label="Effect", rows=effect_rows),
        Section(label="MIP", rows=(Row("partition",
                                       <readable-partition-string>),
                                   Row("tied", len(self.ties) - 1))),
    ]
    if self.reasons:
        sections.append(Section(label="Reasons",
                                rows=(Row("", ", ".join(self.reasons)),)))
    return Description(
        title=type(self).__name__,
        subtitle=f"φ_s {format_value(self.phi)}",
        sections=tuple(sections),
        compact=f"{type(self).__name__}(φ_s={format_value(self.phi)})",
    )
```

Resolve the three `<...>` placeholders by reading the existing code:
- `<node-labels-expression>`: reuse exactly what `fmt.fmt_sia_columns` used for the "System" value (open `pyphi/models/fmt.py` `fmt_sia_columns`).
- `<readable-partition-string>`: the human-readable partition (e.g. `str(self.partition)` if its repr is already readable after Task 13, or `fmt.fmt_partition(self.partition)` until then; the raw numpy matrix is excluded). For this task, use the partition's `__str__`'s first line (the `{A,BC}` form) — confirm against the partition class.
- `system_state.ii_cause`/`ii_effect`/`cause`/`effect`: confirm the attribute names on the `SystemStateSpecification` class (`pyphi/models/state_specification.py`); use the names its own `_repr_columns` exposes.

Import `format_value` from `pyphi.display.numbers`. Delete the old `_repr_columns`, `_repr_html_`, and `__repr__` on this class.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_display.py -k iit4_sia_describe -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyphi/formalism/iit4/__init__.py test/test_display.py
git commit -m "Migrate IIT 4.0 SIA to the unified display model"
```

### Task 11: Lock the look — manual review checkpoint

- [ ] **Step 1: Print the rendered SIA and CES-less output**

Run:
```bash
uv run python - <<'PY'
import pyphi
pyphi.config.progress_bars = False
s = pyphi.examples.basic_system()
print(repr(s.sia()))
PY
```

- [ ] **Step 2: Eyeball it against the spec §4 mockup.** Confirm: rounded numbers, aligned labels, sectioned dividers, no raw numpy matrix, all lines equal width. Adjust `_describe`/`ascii.render` if needed; re-run Task 10's test. **Do not capture goldens yet** (Task 16 does that once all types are migrated). No commit unless code changed; if it did:

```bash
git add -A pyphi/display pyphi/formalism/iit4/__init__.py
git commit -m "Tune ASCII card layout after first-type review"
```

(Stage only `pyphi/display` and the SIA file — not unrelated working-tree files.)

---

## Phase 6 — Migrate remaining result types

Each task below follows the **same recipe** as Task 10:
1. Add `Displayable` to the class bases (or its display base).
2. Implement `_describe(self, verbosity)` returning a `Description`, mapping the
   fields the class's current `_repr_columns`/`__repr__` showed (read the
   existing method first; preserve which fields appear, redesign only the
   layout).
3. Delete the class's ad-hoc `__repr__`/`__str__`/`_repr_columns`/`_repr_html_`.
4. Add a structural test in `test/test_display.py`: assert `_describe(...).title`,
   that key labels appear, that `repr(obj).startswith("╭")`, and that
   `obj._repr_html_()` contains `pyphi-card`.
5. Commit, staging only the touched module + the test file.

> The field choices are mechanical but require reading each class — they are not
> spelled out as literal code here to avoid drift against the real attribute
> names. The recipe + the Task 10 exemplar are the template. Where a type is a
> *collection* (Distinctions, Relations, Account), render its members as a
> `Table` (see Task 5 / the Account mockup in spec §4) or compact `Nested` rows —
> never recursive boxes.

### Task 12: IIT 3.0 SIA + CauseEffectStructure + PhiFold

**Files:**
- Modify: `pyphi/models/sia.py:82-95` (IIT3 SIA — `_repr_columns`/`_repr_html_`/`__repr__`/`__str__`)
- Modify: `pyphi/models/ces.py:89-...` (`CauseEffectStructure`: add `__str__`; render distinctions as a `Table`, **not** an embedded SIA box; `PhiFold`: add full display)
- Test: `test/test_display.py`

Follow the recipe. CES `_describe`: header section (Φ, #distinctions, Σφ_d, #relations, Σφ_r) + a "Distinctions" section whose body is a `Table` with columns (mechanism, φ_d, cause purview, effect purview). Commit: `"Migrate IIT 3.0 SIA, CES, and PhiFold to the unified display model"`.

### Task 13: RIA, MICE, partitions & cuts, repertoires

**Files:**
- Modify: `pyphi/models/ria.py:387-455`
- Modify: `pyphi/models/mice.py:229`
- Modify: `pyphi/models/partitions.py` (all `_PartitionBase` subclasses — give them `_describe`; the readable `{A,BC}`/arrow form as the body, raw matrix only at HIGH verbosity)
- Modify: repertoire formatting (currently `fmt.fmt_repertoire`; expose via the owning result types' `Inline`, or a small `Displayable` wrapper if repertoires are bare arrays — confirm how repertoires reach the user)
- Test: `test/test_display.py`

Note the load-bearing assertions to update **deliberately** in this task:
`test/test_models.py:727` and `:746` (partition/tripartition exact strings) and
`test/test_models.py:367` (`NullCut((2, 3))`). Update them to the new rendered
form and confirm the new form is intentional. Commit:
`"Migrate RIA, MICE, partitions, and repertoires to the unified display model"`.

### Task 14: System, Substrate, Node, state specifications

**Files:**
- Modify: `pyphi/system.py` (`System`: add `__repr__`; it currently has only `__str__` → `"System(B, C)"`)
- Modify: `pyphi/substrate.py` (`Substrate`)
- Modify: `pyphi/models/node.py` or wherever `Node` lives (confirm via grep)
- Modify: `pyphi/models/state_specification.py:117,234` (`StateSpecification`, `SystemStateSpecification`, `UnitState`)
- Test: `test/test_display.py`

Update `test/test_system.py:163` (`str(system) == "System(B, C)"`) deliberately
if the new form differs — keep `System(B, C)` as the `compact`/LOW form so that
assertion can stay green by setting `repr_verbosity=0` there, or update it to the
new default rendering. Decide and make it explicit. Commit:
`"Migrate System, Substrate, Node, and state specs to the unified display model"`.

### Task 15: Actual causation (AcRIA, CausalLink, Account, AcSIA) + Relations

**Files:**
- Modify: `pyphi/models/actual_causation.py:497-500` (`AcSIA._repr_columns`/`_repr_html_`) and `AcRIA`/`CausalLink`/`Account`
- Modify: `pyphi/relations.py:104,194,293` (`Relation`, `RelationFace`, `Relations` — these hand-roll boxed reprs; route through `_describe`. `Relations`/`Account` render members as a `Table`.)
- Modify: `pyphi/actual.py` (any `fmt.*` call sites)
- Test: `test/test_display.py`

Commit: `"Migrate actual-causation results and relations to the unified display model"`.

---

## Phase 7 — Teardown, goldens, and verification

### Task 16: Capture golden snapshots

**Files:**
- Create: `test/data/display_goldens/` (one `.txt` per type) **or** inline expected strings in `test/test_display.py`
- Test: `test/test_display.py`

- [ ] **Step 1:** Write a parametrized golden test that builds one example of each
  migrated type (use `pyphi.examples.basic_system()` and its `.sia()`/`.ces()`,
  plus an actual-causation example and a relations example), renders
  `repr(obj)`, and compares to a stored golden. On first run, generate the
  goldens by printing each and saving the output verbatim (review each by eye
  before saving — this is the redesign's acceptance gate).

```python
# sketch
import pytest

@pytest.mark.parametrize("key", ["iit4_sia", "iit3_sia", "ces", "distinction",
                                 "account", "relations", "partition", "system"])
def test_display_golden(key, display_examples, display_goldens):
    obj = display_examples[key]
    assert repr(obj) == display_goldens[key]
```

- [ ] **Step 2:** Run `uv run pytest test/test_display.py -k golden -v`; confirm PASS after goldens are saved.

- [ ] **Step 3: Commit**

```bash
git add test/test_display.py test/data/display_goldens
git commit -m "Add golden snapshots for unified display output"
```

### Task 17: Add the `Displayable` coverage invariant

**Files:**
- Test: `test/test_display.py`

- [ ] **Step 1:** Add a parametrized test over every migrated result type asserting
  it is a `Displayable` subclass, `_describe(2)` returns a `Description`, and
  `repr`, `str`, `_repr_html_` all return non-empty strings without raising.

- [ ] **Step 2:** Run it; PASS.

- [ ] **Step 3: Commit**

```bash
git add test/test_display.py
git commit -m "Assert all result types are Displayable"
```

### Task 18: Delete `fmt.py` high-level composers; move primitives

**Files:**
- Modify: `pyphi/models/fmt.py` (delete `make_repr`, `fmt_sia`, `fmt_sia_4`, `fmt_sia_columns`, `fmt_ces`, `fmt_ces_columns`, `fmt_distinction`, `fmt_ria`, `fmt_ac_sia`, `fmt_ac_ria`, `fmt_account`, `fmt_causal_link`, `fmt_relation*`, `fmt_phi_structure`, `html_columns`, and the `LOW/MEDIUM/HIGH` constants now owned by `display.mixin`)
- Move the low-level primitives still referenced (`box`, `header`, `align_columns`, `side_by_side`, `indent`, `center`, `margin`, decimal-alignment helpers, Unicode constants) into `pyphi/display/render/ascii.py` as internals, **only if** the new backend still needs them; otherwise delete. The new backend (Tasks 3–6) is self-contained, so most should be deletable.
- Update remaining importers: `pyphi/measures/distribution.py` and any other file from the caller list still importing `fmt`.

- [ ] **Step 1:** Grep for residual usage: `grep -rn --include='*.py' 'fmt\.' pyphi/` and `grep -rn 'import fmt' pyphi/`. Every hit must resolve to a surviving primitive or be removed.

- [ ] **Step 2:** Run the targeted suites: `uv run pytest test/test_display.py test/test_models.py test/test_result_protocols.py test/test_system.py -v`. Expected: PASS (with the deliberately-updated assertions from Tasks 13–14).

- [ ] **Step 3: Commit**

```bash
git add pyphi/models/fmt.py pyphi/display/render/ascii.py pyphi/measures/distribution.py
git commit -m "Remove fmt.py composers; consolidate display in pyphi.display"
```

### Task 19: Extend HTML structural tests to all types

**Files:**
- Modify: `test/test_result_protocols.py`

- [ ] **Step 1:** Generalize the existing 4 `_repr_html_` tests into a parametrized
  test over all migrated types asserting `pyphi-card` and the type's key labels
  appear in `obj._repr_html_()`.

- [ ] **Step 2:** Run `uv run pytest test/test_result_protocols.py -v`; PASS.

- [ ] **Step 3: Commit**

```bash
git add test/test_result_protocols.py
git commit -m "Extend HTML display tests to all result types"
```

### Task 20: Full verification + changelog + roadmap

**Files:**
- Create: `changelog.d/b21.refactor.md`
- Modify: `ROADMAP.md` (Status Dashboard B21 row → ✅ landed; update the Wave 2 prose)

- [ ] **Step 1: Full suite (no path arg, so the doctest sweep runs)**

Run: `uv run pytest`
Expected: PASS. Investigate any doctest failures in `pyphi/` source modules
(rendered-output doctests, if any survive) and update them deliberately to the
new output. Run the slow lane too if touched: `uv run pytest test/test_invariants_hypothesis.py` (background; see CLAUDE.md parallel guidance).

- [ ] **Step 2: Type check**

Run: `uv run pyright pyphi/display`
Expected: no new errors. (Note: a session-local `typeCheckingMode = "off"` may be set in `pyproject.toml` — if pyright reports nothing, trust the pre-commit hook instead.)

- [ ] **Step 3: Write the changelog fragment**

```bash
echo 'Unified all result-object display behind a single \`_describe()\` description model with pluggable ASCII and HTML renderers, giving every result type a consistent, redesigned `repr`/`str`/`_repr_html_`.' > changelog.d/b21.refactor.md
```

- [ ] **Step 4: Update ROADMAP.md** — set the B21 dashboard row to ✅ landed and
  reconcile the Wave 2 prose, matching the style of the already-landed rows.

- [ ] **Step 5: Commit**

```bash
git add changelog.d/b21.refactor.md ROADMAP.md
git commit -m "B21: changelog + roadmap status for unified display"
```

---

## Self-review notes (for the executor)

- **Spec coverage:** vocabulary (Task 2) · ASCII backend (3–6) · HTML backend (7) · backend seam incl. deferred rich (8) · Displayable + one verbosity site (9) · rounding (1) · first-type proof + look-lock (10–11) · full coverage (10, 12–15) · partitions readable / no numpy (13) · fmt.py teardown (18) · goldens (16) · Displayable invariant (17) · HTML coverage (19) · doctest sweep + changelog + roadmap (20).
- **The `<...>` placeholders in Task 10** are the only intentional read-the-code gaps; they are explicitly resolved by reading named existing methods. Do not invent attribute names.
- **Number display:** `format_value` uses `:.6g`, so whole floats print without a decimal (`1.0` → `1`); goldens are captured from real output, so this is consistent by construction.
- **Circular imports:** `pyphi.display.mixin` imports `config`; if importing at module top causes a cycle when result modules import `Displayable`, switch to a lazy import inside `_verbosity()`.
