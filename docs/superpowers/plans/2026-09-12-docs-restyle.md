# Documentation Restyle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the approved visual system to the docs site (palette, IBM Plex, Selenized code blocks, restyled cards, landing page, tables, admonitions, release chrome) and make the IIT 4.0 demo notebook page the notebook itself with stored, drift-guarded outputs.

**Architecture:** One stylesheet (`docs/_static/custom.css`) holds the tokens and component rules; `docs/conf.py` gains theme options and registers two Pygments styles from `docs/_ext/selenized.py`; one display-module change gives every card a single label-column width; the demo notebook is executed once by a script that stores a code hash, committed with outputs, rendered without execution, and guarded by a fast hash test and a slow re-execution test.

**Tech Stack:** pydata-sphinx-theme 0.19, MyST-NB, sphinx_design (Octicons), Pygments 2.20, nbclient/nbformat, pytest.

**Spec:** `docs/superpowers/specs/2026-09-12-docs-restyle-design.md`

## Global Constraints

- Public prose follows the `writing-naturally` skill; "intrinsic-information requirement", never "cap".
- Every colour is a token defined for both `html[data-theme="light"]` and `html[data-theme="dark"]`; no component rule names a literal colour except the cause and effect tones already in the card CSS.
- `uv run ruff format <specific .py files>`; `uv run ruff check pyphi test docs/_ext scripts`; pre-commit runs ruff and pyright; never `--no-verify`.
- pytest: redirect to a scratch file, echo `$?`, read the summary line. Bare `uv run pytest` for the final verification. Docs: `just docs` must end in "build succeeded" (delete a stale `docs/reference/_autosummary/` first if the build reports missing attributes).
- Environment for any pyphi script: `PYPHI_WELCOME_OFF=1 PYPHI_AGENT_NOTE_OFF=1`.
- Commit messages end with the session's attribution trailer. No push, no tag.
- Check `git status` before every commit: whatever is already staged rides into it.

---

### Task 1: Worktree

- [ ] **Step 1: Create the worktree**

```bash
cd /Users/will/projects/pyphi
git worktree add .claude/worktrees/docs-style -b docs-style main
cd .claude/worktrees/docs-style
uv sync --all-extras > /dev/null 2>&1; echo "sync exit: $?"
```

- [ ] **Step 2: Baseline**

```bash
export PYPHI_WELCOME_OFF=1 PYPHI_AGENT_NOTE_OFF=1
uv run pytest test/display -q > /tmp/baseline.log 2>&1; echo $?; tail -1 /tmp/baseline.log
```
Expected: exit 0.

---

### Task 2: One label-column width per card (display module)

**Files:**
- Modify: `pyphi/display/render/ascii.py` (`_format_rows`, `_section_lines`, `render`)
- Modify: `pyphi/display/render/html.py` (`_STYLE` `.pyphi-kv` rule, `_kv_html`, `_section_html`, `render`)
- Test: `test/display/test_display.py` (append two tests; existing tests keep passing because the new parameter defaults to per-section behavior)

**Interfaces:**
- Produces: `ascii._format_rows(rows, label_w=None)`, `ascii._section_lines(section, label_w=None)`, `html._kv_html(rows)` unchanged, the card `<div class="pyphi-card" style="--pc-kcol:Nch">`, and a module-level helper `card_label_width(description) -> int` in `pyphi/display/description.py` used by both backends.

- [ ] **Step 1: Write the failing tests**

Append to `test/display/test_display.py`:

```python
def test_ascii_card_shares_one_label_column_across_sections():
    """Values line up down the whole card, not per section."""
    d = Description(
        title="Demo",
        sections=(
            Section(label=None, rows=(Row("Φ", 1.5), Row("φ_s", 0.4))),
            Section(label="System", rows=(Row("Requirement binds", "x"),)),
        ),
    )
    out = ascii_backend.render(d, verbosity=2)
    value_cols = [line.index("1.5") for line in out.splitlines() if "1.5" in line]
    value_cols += [line.index("x") for line in out.splitlines() if line.rstrip().endswith("x │")]
    assert len(set(value_cols)) == 1


def test_html_card_emits_label_column_width():
    d = Description(
        title="Demo",
        sections=(
            Section(label=None, rows=(Row("Φ", 1.5),)),
            Section(label="System", rows=(Row("Requirement binds", "x"),)),
        ),
    )
    out = html_backend.render(d, verbosity=2)
    assert 'class="pyphi-card" style="--pc-kcol:17ch"' in out
    assert "grid-template-columns:var(--pc-kcol,auto) 1fr" in out
```

(`html_backend` is imported at the top of the test module the same way `ascii_backend` is; check the import line and add it if absent.)

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest test/display/test_display.py -q -k "label_column" > /tmp/t2a.log 2>&1; echo $?; tail -3 /tmp/t2a.log
```
Expected: FAIL (values at different columns; no `--pc-kcol` in the HTML).

- [ ] **Step 3: Implement**

In `pyphi/display/description.py`, after the `Description` class:

```python
def card_label_width(description: "Description") -> int:
    """The widest key/value label in a card, in visible characters.

    Every key/value section of a card aligns its values to this width, so
    values line up down the whole card rather than per section.
    """
    labels: list[str] = []
    for section in description.sections:
        labels.extend(row.label for row in section.rows)
        labels.extend(comp.label for comp in section.body if isinstance(comp, Row))
    return max((len(label) for label in labels), default=0)
```

In `pyphi/display/render/ascii.py`:

```python
def _format_rows(rows: tuple[Row, ...], label_w: int | None = None) -> list[str]:
    """Render key/value rows with labels aligned to a common width.

    ``label_w`` is the card-wide label width; when None the rows' own widest
    label is used.
    """
    if not rows:
        return []
    if label_w is None:
        label_w = max(_vis_len(row.label) for row in rows)
    lines = []
    for row in rows:
        parts = [f"{_pad(row.label, label_w)}   {format_value(row.value)}"]
        for name, val in row.extra:
            parts.append(f"{name} {format_value(val)}")
        lines.append("   ".join(parts))
    return lines


def _section_lines(section: Section, label_w: int | None = None) -> list[str]:
    """Flatten a section's rows and body components into content lines."""
    lines = list(_format_rows(section.rows, label_w))
    for comp in section.body:
        if isinstance(comp, Table):
            lines.extend(_format_table(comp))
        elif isinstance(comp, Inline):
            lines.extend(comp.text.splitlines())
        elif isinstance(comp, Row):
            lines.extend(_format_rows((comp,), label_w))
        elif isinstance(comp, Nested):
            lines.append(_compact(comp.description))
    return lines
```

and in `render`, compute `label_w = card_label_width(description)` (import it from `pyphi.display.description`) before the loop and pass it to both `_section_lines` calls. `_vis_len` counts code points, as `card_label_width` does with `len`, so the two agree.

In `pyphi/display/render/html.py`: change the `.pyphi-kv` rule in `_STYLE` to

```
.pyphi-kv{{display:grid;grid-template-columns:var(--pc-kcol,auto) 1fr;gap:3px 14px}}
```

and in `render`, replace `'<div class="pyphi-card">'` with

```python
        + f'<div class="pyphi-card" style="--pc-kcol:{card_label_width(description)}ch">'
```

- [ ] **Step 4: Run**

```bash
uv run pytest test/display test/test_analyze.py test/mcp -q > /tmp/t2.log 2>&1; echo $?; tail -1 /tmp/t2.log
```
Expected: exit 0. If an existing assertion pins a per-section width (search the log for `assert` failures on card text), the card-wide width is the intended value; update that expectation.

- [ ] **Step 5: Commit**

```bash
cat > changelog.d/card-label-column.change.md <<'EOF2'
Result cards align every key/value section to one label-column width, so values line up down the whole card in both the text and the HTML rendering.
EOF2
uv run ruff format pyphi/display/description.py pyphi/display/render/ascii.py pyphi/display/render/html.py test/display/test_display.py && uv run ruff check pyphi test
git status --short
git add pyphi/display/description.py pyphi/display/render/ascii.py pyphi/display/render/html.py test/display/test_display.py changelog.d/card-label-column.change.md
git commit -m "Align every card section to one label-column width"
```

---

### Task 3: Selenized Pygments styles

**Files:**
- Create: `docs/_ext/selenized.py`
- Modify: `docs/conf.py` (sys.path, registration, theme options)
- Test: `test/docs/test_selenized.py` (new directory with `__init__.py`)

- [ ] **Step 1: Failing test**

```python
"""The Selenized Pygments styles the docs register."""

import sys
from pathlib import Path

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer
from pygments.token import Name

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs" / "_ext"))

from selenized import SelenizedDarkStyle, SelenizedLightStyle  # noqa: E402


def test_plain_names_have_an_explicit_colour():
    """A block must not depend on the page's text colour for plain names."""
    for style in (SelenizedLightStyle, SelenizedDarkStyle):
        assert style.styles[Name]  # non-empty colour spec
        html = highlight("pyphi.config", PythonLexer(), HtmlFormatter(style=style, noclasses=True))
        assert style.styles[Name].split()[-1] in html
```

- [ ] **Step 2: Implement `docs/_ext/selenized.py`**

```python
"""Pygments styles from the Selenized palettes (Jan Warchoł), light and dark.

Plain names carry the palette's foreground explicitly so a highlighted block
never inherits the page's text colour: a dark block on a light page (or the
reverse) stays readable.
"""

from pygments.style import Style
from pygments.token import (
    Comment,
    Keyword,
    Name,
    Number,
    Operator,
    Punctuation,
    String,
    Text,
)


def _selenized(bg, fg, dim, green, yellow, blue, magenta, cyan, orange, violet):
    return {
        "background_color": bg,
        "styles": {
            Text: fg,
            Name: fg,
            Comment: f"italic {dim}",
            Keyword: f"bold {green}",
            Keyword.Constant: yellow,
            Name.Builtin: blue,
            Name.Function: blue,
            Name.Class: f"bold {yellow}",
            Name.Decorator: violet,
            String: cyan,
            String.Interpol: orange,
            Number: magenta,
            Operator: fg,
            Operator.Word: f"bold {green}",
            Punctuation: fg,
        },
    }


class SelenizedLightStyle(Style):
    """Selenized light: cream ground, muted ink."""

    _p = _selenized("#fbf3db", "#53676d", "#909995", "#489100", "#ad8900", "#0072d4", "#ca4898", "#009c8f", "#c25d1e", "#8762c6")
    background_color = _p["background_color"]
    styles = _p["styles"]


class SelenizedDarkStyle(Style):
    """Selenized dark: blue-teal ground."""

    _p = _selenized("#103c48", "#adbcbc", "#72898f", "#75b938", "#dbb32d", "#4695f7", "#f275be", "#41c7b9", "#ed8649", "#af88eb")
    background_color = _p["background_color"]
    styles = _p["styles"]
```

- [ ] **Step 3: Register in `docs/conf.py`**

After the `import os` block:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "_ext"))

# Register the Selenized styles under the names the theme options use.
from pygments import styles as _pygments_styles  # noqa: E402

_pygments_styles._STYLE_NAME_TO_MODULE_MAP["selenized-light"] = ("selenized", "SelenizedLightStyle")
_pygments_styles._STYLE_NAME_TO_MODULE_MAP["selenized-dark"] = ("selenized", "SelenizedDarkStyle")
```

and in `html_theme_options` add `"pygments_light_style": "selenized-light"` and `"pygments_dark_style": "selenized-dark"`.

- [ ] **Step 4: Run and commit**

```bash
mkdir -p test/docs && touch test/docs/__init__.py
uv run pytest test/docs -q > /tmp/t3.log 2>&1; echo $?; tail -1 /tmp/t3.log
rm -rf docs/reference/_autosummary; just docs > /tmp/t3docs.log 2>&1; echo $?; grep -n "build succeeded" /tmp/t3docs.log
grep -c "fbf3db" docs/_build/html/_static/pygments.css   # the light style landed
uv run ruff format docs/_ext/selenized.py test/docs/test_selenized.py docs/conf.py && uv run ruff check docs/_ext test
git add docs/_ext/selenized.py docs/conf.py test/docs
git commit -m "Register Selenized light and dark as the docs code-block styles"
```

`ruff check` on `docs/conf.py` may already be excluded by the project's ruff config; if so, leave it.

---

### Task 4: The stylesheet

**Files:**
- Modify: `docs/_static/custom.css` (replace; keep the existing dark-mode rule for card backdrops)
- Modify: `docs/conf.py` (`html_css_files` gains the Google Fonts URL)

- [ ] **Step 1: Fonts**

In `docs/conf.py`:

```python
html_css_files = [
    "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400;1,600&family=IBM+Plex+Serif:wght@500;600&family=IBM+Plex+Mono:ital,wght@0,400;0,500;1,400&display=swap",
    "custom.css",
]
```

- [ ] **Step 2: Write `docs/_static/custom.css`**

```css
/* PyPhi docs: tokens and component rules. Every colour is a token with a
   light and a dark value; components never name a colour directly. */

html[data-theme="light"] {
  --pp-ground: #fbfaf7;
  --pp-raised: #f2f0ea;
  --pp-text: #1d1f22;
  --pp-muted: #5b6068;
  --pp-rule: #d9d6ce;
  --pp-primary: #0072b2;
  --pp-link: #0b6e9e;
  --pp-tint: #e4f0f7;
  --pp-cause: #d55c00;
  --pp-effect: #009e73;
  --pp-code-ground: #fbf3db;

  --pst-color-background: var(--pp-ground);
  --pst-color-on-background: var(--pp-raised);
  --pst-color-surface: var(--pp-raised);
  --pst-color-on-surface: var(--pp-text);
  --pst-color-text-base: var(--pp-text);
  --pst-color-text-muted: var(--pp-muted);
  --pst-color-border: var(--pp-rule);
  --pst-color-border-muted: var(--pp-rule);
  --pst-color-primary: var(--pp-primary);
  --pst-color-primary-bg: var(--pp-tint);
  --pst-color-secondary: var(--pp-link);
  --pst-color-link: var(--pp-link);
  --pst-color-link-hover: var(--pp-primary);
  --pst-color-inline-code: var(--pp-text);
  --pst-color-inline-code-links: var(--pp-link);
  --pst-color-table-heading-bg: var(--pp-raised);
  --pst-color-table-row-zebra-low-bg: var(--pp-raised);
}

html[data-theme="dark"] {
  --pp-ground: #15171a;
  --pp-raised: #1d2024;
  --pp-text: #e8e6e1;
  --pp-muted: #a2a7ae;
  --pp-rule: #33373d;
  --pp-primary: #5db4dd;
  --pp-link: #5db4dd;
  --pp-tint: #14323f;
  --pp-cause: #d55c00;
  --pp-effect: #009e73;
  --pp-code-ground: #103c48;

  --pst-color-background: var(--pp-ground);
  --pst-color-on-background: var(--pp-raised);
  --pst-color-surface: var(--pp-raised);
  --pst-color-on-surface: var(--pp-text);
  --pst-color-text-base: var(--pp-text);
  --pst-color-text-muted: var(--pp-muted);
  --pst-color-border: var(--pp-rule);
  --pst-color-border-muted: var(--pp-rule);
  --pst-color-primary: var(--pp-primary);
  --pst-color-primary-bg: var(--pp-tint);
  --pst-color-secondary: var(--pp-link);
  --pst-color-link: var(--pp-link);
  --pst-color-link-hover: var(--pp-primary);
  --pst-color-inline-code: var(--pp-text);
  --pst-color-inline-code-links: var(--pp-link);
  --pst-color-table-heading-bg: var(--pp-raised);
  --pst-color-table-row-zebra-low-bg: var(--pp-raised);
}

/* ---- Typography ---------------------------------------------------------- */

html {
  --pst-font-family-base: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --pst-font-family-heading: "IBM Plex Serif", Georgia, "Times New Roman", serif;
  --pst-font-family-monospace: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
  --pst-font-size-base: 16px;
  --pst-font-weight-heading: 600;
  --pst-font-size-h1: 2rem;
  --pst-font-size-h2: 1.5rem;
  --pst-font-size-h3: 1.2rem;
  --pst-font-size-h4: 1rem;
}
body { line-height: 1.6; }
h1, h2, h3, h4 { text-wrap: balance; }
h1 { margin-bottom: 1.25rem; }
code, kbd, pre, samp { font-size: 0.9em; }
pre code { font-size: 1em; }

/* ---- Code blocks --------------------------------------------------------- */

div.highlight pre, .cell_input div.highlight pre {
  border: 1px solid var(--pp-rule);
  border-radius: 6px;
  padding: 0.8rem 1rem;
}
.cell_output .output.text_plain pre { border-radius: 6px; }

/* ---- Result cards -------------------------------------------------------- */

.pyphi-card {
  --pc-bg: var(--pp-ground);
  --pc-fg: var(--pp-text);
  --pc-line: var(--pp-rule);
  --pc-soft: var(--pp-raised);
  --pc-head: var(--pp-raised);
  --pc-muted: var(--pp-muted);
  --pc-faint: var(--pp-muted);
  --pc-badge-bg: var(--pp-tint);
  --pc-badge-fg: var(--pp-primary);
  --pc-shadow: rgba(0, 0, 0, 0.06);
  border-radius: 6px;
  font-family: var(--pst-font-family-base);
}
.pyphi-card .pyphi-v,
.pyphi-card .pyphi-badge,
.pyphi-card table.pyphi-table td { font-family: var(--pst-font-family-monospace); }
.pyphi-card .pyphi-label { letter-spacing: 0.04em; }
/* The cards set their own colours, so drop the light backdrop the theme
   adds behind HTML outputs in dark mode. */
html[data-theme="dark"] .bd-content div.cell_output .output.text_html:has(.pyphi-card) {
  background-color: transparent;
}

/* ---- Tables -------------------------------------------------------------- */

.bd-content table.table, .bd-content table.docutils, .bd-content .dataframe {
  width: auto;
  max-width: 100%;
  font-variant-numeric: tabular-nums;
  border-collapse: collapse;
}
.bd-content table.table thead th, .bd-content table.docutils thead th, .bd-content .dataframe thead th {
  font-size: 0.8rem;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--pp-muted);
  background: transparent;
  border-bottom: 2px solid var(--pp-rule);
  text-align: left;
}
.bd-content table.table td, .bd-content table.docutils td, .bd-content .dataframe td {
  border-bottom: 1px solid var(--pp-rule);
  padding: 0.45rem 0.6rem;
}
.bd-content table.table tbody tr:nth-child(even) td,
.bd-content table.docutils tbody tr:nth-child(even) td,
.bd-content .dataframe tbody tr:nth-child(even) td { background: var(--pp-raised); }
.bd-content .table-responsive, .bd-content .pst-scrollable-table-container { overflow-x: auto; }

/* ---- Admonitions --------------------------------------------------------- */

.admonition, div.admonition {
  border: 0;
  border-left: 4px solid var(--pp-primary);
  border-radius: 0 6px 6px 0;
  background: var(--pp-tint);
  box-shadow: none;
}
.admonition > .admonition-title { background: transparent; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; font-size: 0.85rem; }
.admonition > .admonition-title::after { display: none; }
.admonition.warning, .admonition.danger, .admonition.caution {
  border-left-color: var(--pp-cause);
  background: color-mix(in srgb, var(--pp-cause) 12%, var(--pp-ground));
}
.admonition.tip, .admonition.hint {
  border-left-color: var(--pp-effect);
  background: color-mix(in srgb, var(--pp-effect) 12%, var(--pp-ground));
}

/* ---- Landing page -------------------------------------------------------- */

.pp-tagline { font-size: 1.15rem; color: var(--pp-muted); max-width: 60ch; margin: 0 0 1.5rem; }
.sd-card { border-color: var(--pp-rule); background: var(--pp-raised); border-radius: 6px; }
.sd-card .sd-card-title .octicon { color: var(--pp-primary); margin-right: 0.4rem; }
.pp-cite { font-size: 0.9rem; color: var(--pp-muted); border-top: 1px solid var(--pp-rule); margin-top: 2rem; padding-top: 1rem; }
.pp-cite p { margin: 0.25rem 0; }

/* ---- Sidebar ------------------------------------------------------------- */

.pp-sidebar-cite { font-size: 0.8rem; color: var(--pp-muted); padding: 1rem 0; border-top: 1px solid var(--pp-rule); }
```

The theme's variable names above are those of pydata-sphinx-theme 0.19; if a build shows one unstyled (for example the table zebra variable), check the installed theme's `pydata-sphinx-theme.css` for the exact name and correct it. The selectors for tables and admonitions target the theme's own classes; verify each against the built HTML of `reference/configuration.html` (a DataFrame), `migration/migration-2.0.html` (Markdown tables), and `theory/intrinsic-information.html`.

- [ ] **Step 3: Build and look**

```bash
rm -rf docs/reference/_autosummary; just docs > /tmp/t4docs.log 2>&1; echo $?; grep -n "build succeeded" /tmp/t4docs.log
```

Open `docs/_build/html/index.html`, `howto/read-result.html`, `howto/estimate-cost.html`, `theory/intrinsic-information.html`, `reference/configuration.html`, `migration/migration-2.0.html` in a browser and toggle the theme on each: fonts loaded (serif headings), links in the primary blue, code blocks Selenized in both themes, cards on the page ground with a raised header, tables striped, admonitions tinted. Fix selectors until every item holds; no page scrolls horizontally at 400 px.

- [ ] **Step 4: Commit**

```bash
git add docs/_static/custom.css docs/conf.py
git commit -m "Style the docs: palette, IBM Plex, code blocks, cards, tables, admonitions"
```

---

### Task 5: Landing page and chrome

**Files:**
- Modify: `docs/index.md`
- Modify: `docs/conf.py` (`html_theme_options`: announcement, switcher, navbar_end, primary_sidebar_end)
- Create: `docs/_static/switcher.json`, `docs/_templates/sidebar-cite.html`

- [ ] **Step 1: `docs/index.md`**

Replace the file's body above the hidden toctree with:

````markdown
# PyPhi

```{raw} html
<p class="pp-tagline">Integrated information, computed. Given a substrate and a state, PyPhi finds the complexes, their φ<sub>s</sub>, and the Φ-structure each one specifies.</p>
```

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} {octicon}`rocket` Getting started
:link: getting-started/index
:link-type: doc
Install PyPhi and compute your first φ.
:::

:::{grid-item-card} {octicon}`book` Tutorials
:link: tutorials/index
:link-type: doc
Learn the library through worked, executable examples.
:::

:::{grid-item-card} {octicon}`tools` How-to guides
:link: howto/index
:link-type: doc
Build a substrate, read a result, size a run, configure, parallelize, export.
:::

:::{grid-item-card} {octicon}`beaker` Theory
:link: theory/index
:link-type: doc
How IIT 4.0's mathematics maps onto PyPhi's types and functions.
:::

:::{grid-item-card} {octicon}`list-unordered` Reference
:link: reference/index
:link-type: doc
The API reference, the glossary, configuration options, and conventions.
:::

:::{grid-item-card} {octicon}`arrow-switch` Migration
:link: migration/index
:link-type: doc
Moving to PyPhi 2.0 from earlier versions and related tools.
:::
::::

```{raw} html
<div class="pp-cite">
<p><b>Cite.</b> If you use PyPhi in your research, please cite the software paper, and the theory papers for the formalism you used.</p>
<p>Mayner WGP, Marshall W, Albantakis L, Findlay G, Marchman R, Tononi G. (2018). PyPhi: A toolbox for integrated information theory. <i>PLOS Computational Biology</i> 14(7): e1006343. <a href="https://doi.org/10.1371/journal.pcbi.1006343">doi:10.1371/journal.pcbi.1006343</a></p>
<p>Albantakis L, Barbosa L, Findlay G, Grasso M, … Tononi G. (2023). Integrated information theory (IIT) 4.0. <i>PLoS Computational Biology</i> 19(10): e1011465. <a href="https://doi.org/10.1371/journal.pcbi.1011465">doi:10.1371/journal.pcbi.1011465</a></p>
<p>Mayner WGP, Marshall W, Tononi G. (2026). Intrinsic cause–effect power: the tradeoff between differentiation and specification. <i>Entropy</i> 28(4): 410. <a href="https://doi.org/10.3390/e28040410">doi:10.3390/e28040410</a></p>
<p>Issues: <a href="https://github.com/wmayner/pyphi/issues">GitHub issue tracker</a>. Discussion: <a href="https://groups.google.com/forum/#!forum/pyphi-users">pyphi-users group</a>.</p>
</div>
```
````

Keep the hidden toctree exactly as it is (it still lists `whats-new-in-2.0` and `contributing`). The what's-new card is dropped; the announcement bar carries that link.

- [ ] **Step 2: Chrome in `docs/conf.py`**

```python
html_theme_options = {
    "github_url": "https://github.com/wmayner/pyphi",
    "navbar_align": "left",
    "header_links_before_dropdown": 6,
    "logo": {
        "image_light": "_static/pyphi-logo-text-noborder-776x196.png",
        "image_dark": "_static/pyphi-logo-text-white-noborder-776x196.png",
    },
    "pygments_light_style": "selenized-light",
    "pygments_dark_style": "selenized-dark",
    "announcement": (
        'PyPhi 2.0 is released: <a href="whats-new-in-2.0.html">what\'s new</a>, '
        'and the <a href="migration/migration-2.0.html">migration guide</a> for 1.x users.'
    ),
    "switcher": {
        "json_url": "https://pyphi.readthedocs.io/en/latest/_static/switcher.json",
        "version_match": os.environ.get("READTHEDOCS_VERSION", "latest"),
    },
    "check_switcher": False,
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    "primary_sidebar_end": ["sidebar-cite"],
}
```

`docs/_static/switcher.json`:

```json
[
  {"name": "stable", "version": "stable", "url": "https://pyphi.readthedocs.io/en/stable/", "preferred": true},
  {"name": "latest", "version": "latest", "url": "https://pyphi.readthedocs.io/en/latest/"},
  {"name": "2.0.0", "version": "v2.0.0", "url": "https://pyphi.readthedocs.io/en/v2.0.0/"}
]
```

`docs/_templates/sidebar-cite.html`:

```html
<div class="pp-sidebar-cite">
  <p>Cite: Mayner et al. (2018), <i>PLOS Comput Biol</i> 14(7): e1006343.</p>
  <p><a href="https://github.com/wmayner/pyphi">Source on GitHub</a></p>
</div>
```

`templates_path = ["_templates"]` is already set. The announcement links are relative to the site root, which is right for the landing page and wrong for nested pages; use `{{ pathto("whats-new-in-2.0") }}` if the theme renders the announcement as a template (0.19 does not), otherwise link with absolute site paths on RTD (`https://pyphi.readthedocs.io/en/stable/whats-new-in-2.0.html`). Use the absolute form.

- [ ] **Step 3: Build, look, commit**

```bash
rm -rf docs/reference/_autosummary; just docs > /tmp/t5docs.log 2>&1; echo $?; grep -n "build succeeded" /tmp/t5docs.log
```
Open `index.html`: announcement bar, tagline, six cards with glyphs, cite strip; the sidebar footer on any inner page; the version dropdown present (it shows "latest" locally).

```bash
git add docs/index.md docs/conf.py docs/_static/switcher.json docs/_templates/sidebar-cite.html
git commit -m "Reorder the landing page and add the release chrome"
```

---

### Task 6: The demo notebook page with stored, guarded outputs

**Files:**
- Create: `scripts/execute_notebook.py`
- Modify: `docs/examples/IIT_4.0_demo.ipynb` (first cell, pip cell tag, outputs, metadata)
- Modify: `docs/conf.py` (exclude patterns), `docs/tutorials/index.md`, `justfile`, `RELEASING.md`
- Delete: `docs/tutorials/iit-4.0-demo.md`, `docs/tutorials/iit-4.0-demo.ipynb`
- Test: `test/docs/test_demo_notebook.py`

- [ ] **Step 1: The execution script**

`scripts/execute_notebook.py`:

```python
"""Execute a notebook in place and record a hash of its code cells.

The stored outputs are what the docs render; the hash lets a fast test detect
code edits made without re-executing. Cells tagged ``skip-execution`` are
left untouched.
"""

import hashlib
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def code_hash(nb) -> str:
    h = hashlib.sha256()
    for cell in nb.cells:
        if cell.cell_type == "code":
            h.update(cell.source.encode())
            h.update(b"\0")
    return h.hexdigest()


def main(path: str) -> None:
    file = Path(path)
    nb = nbformat.read(file, as_version=4)
    client = NotebookClient(
        nb,
        timeout=1800,
        kernel_name="python3",
        resources={"metadata": {"path": str(file.parent)}},
    )
    client.execute()
    nb.metadata.setdefault("pyphi", {})["code_hash"] = code_hash(nb)
    nbformat.write(nb, file)
    print(f"executed {sum(c.cell_type == 'code' for c in nb.cells)} code cells; hash {nb.metadata['pyphi']['code_hash'][:12]}")


if __name__ == "__main__":
    main(sys.argv[1])
```

- [ ] **Step 2: Prepare the notebook**

With a short Python script (nbformat): tag the `!python -m pip install ...` cell with `"tags": ["skip-execution"]` and change its source's first line to a comment `# Colab: install PyPhi first.` above the command; insert as the first cell a Markdown cell with the summary from `docs/tutorials/iit-4.0-demo.md` (the paragraphs from "This notebook is the supplement..." to "...see A complete worked example", with the `{doc}` roles turned into relative links `../theory/intrinsic-information.md`, `../theory/index.md`, `../tutorials/worked-example.md`) preceded by the heading `# The IIT 4.0 demo notebook` and the two badges (download link: "Download the notebook", `IIT_4.0_demo.ipynb`; the Colab badge as it is). Then:

```bash
export PYPHI_WELCOME_OFF=1 PYPHI_AGENT_NOTE_OFF=1
uv run python scripts/execute_notebook.py docs/examples/IIT_4.0_demo.ipynb
```
Expected: "executed 44 code cells" (45 minus the skipped install cell) in about 15 minutes.

- [ ] **Step 3: Wire the page**

`docs/conf.py`: remove `"examples/IIT_4.0_demo.ipynb"` from `exclude_patterns`; add

```python
nb_execution_excludepatterns = ["examples/IIT_4.0_demo.ipynb"]
```

`docs/tutorials/index.md` toctree: replace `iit-4.0-demo` with `../examples/IIT_4.0_demo`. Delete `docs/tutorials/iit-4.0-demo.md` and its `.ipynb` pair (`git rm`). Check `docs/` for links to `tutorials/iit-4.0-demo` (`grep -rn "iit-4.0-demo" docs README.md pyphi/mcp/content`) and repoint them to `examples/IIT_4.0_demo`.

`justfile`:

```make
# Re-execute the demo notebook and store its outputs (about 15 minutes)
notebook-outputs:
    PYPHI_WELCOME_OFF=1 PYPHI_AGENT_NOTE_OFF=1 uv run python scripts/execute_notebook.py docs/examples/IIT_4.0_demo.ipynb
```

`RELEASING.md` gate 4: replace the paragraph with "**Demo notebook**: the stored outputs are guarded by `test/docs/test_demo_notebook.py` (a fast code-hash check in the default suite, and a full re-execution diff in the slow lane). To refresh them after a change, run `just notebook-outputs` and commit the notebook."

- [ ] **Step 4: The two guards**

`test/docs/test_demo_notebook.py`:

```python
"""The demo notebook's stored outputs must match its code and the library."""

import re
import sys
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "docs" / "examples" / "IIT_4.0_demo.ipynb"
sys.path.insert(0, str(ROOT / "scripts"))

from execute_notebook import code_hash  # noqa: E402


def test_stored_outputs_match_the_code():
    """Editing a code cell without re-executing leaves a stale hash."""
    nb = nbformat.read(NOTEBOOK, as_version=4)
    assert nb.metadata["pyphi"]["code_hash"] == code_hash(nb), (
        "the notebook's code changed since its outputs were stored; run `just notebook-outputs`"
    )


def _text_outputs(cell) -> list[str]:
    out = []
    for o in cell.get("outputs", []):
        if o.output_type == "stream" and o.name == "stdout":
            out.append(o.text)
        elif o.output_type in ("execute_result", "display_data"):
            out.append(o.data.get("text/plain", ""))
    text = "\n".join(out)
    text = re.sub(r"<[^>]+ at 0x[0-9a-f]+>", "<obj>", text)  # object addresses
    return [line for line in text.splitlines() if not re.search(r"\b(s|ms|it/s)\b", line)]


@pytest.mark.slow
def test_stored_outputs_match_a_fresh_execution():
    """Library changes that alter a printed value must be caught before release."""
    from nbclient import NotebookClient

    stored = nbformat.read(NOTEBOOK, as_version=4)
    fresh = nbformat.read(NOTEBOOK, as_version=4)
    NotebookClient(
        fresh, timeout=1800, kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK.parent)}},
    ).execute()
    for i, (a, b) in enumerate(zip(stored.cells, fresh.cells, strict=True)):
        if a.cell_type != "code" or "skip-execution" in a.metadata.get("tags", []):
            continue
        assert _text_outputs(a) == _text_outputs(b), f"cell {i} output drifted"
```

The slow test respects the root conftest's `--slow` gate like every other `slow` test. Run the fast one now; run the slow one once in Task 7.

- [ ] **Step 5: Build, test, commit**

```bash
uv run pytest test/docs -q > /tmp/t6.log 2>&1; echo $?; tail -1 /tmp/t6.log
rm -rf docs/reference/_autosummary; just docs > /tmp/t6docs.log 2>&1; echo $?; grep -n "build succeeded" /tmp/t6docs.log
```
Open `docs/_build/html/examples/IIT_4.0_demo.html`: the notebook renders with outputs, the install cell shows no output, the download badge reads "Download the notebook".

```bash
cat > changelog.d/demo-notebook-page.doc.md <<'EOF2'
The IIT 4.0 demo notebook is now rendered in full on its tutorial page from stored outputs, with a fast test that detects code edits made without re-execution and a slow-lane test that re-executes it and compares every output.
EOF2
uv run ruff format scripts/execute_notebook.py test/docs/test_demo_notebook.py && uv run ruff check scripts test
git status --short
git add scripts/execute_notebook.py docs/examples/IIT_4.0_demo.ipynb docs/conf.py docs/tutorials/index.md justfile RELEASING.md test/docs/test_demo_notebook.py changelog.d/demo-notebook-page.doc.md
git rm -q docs/tutorials/iit-4.0-demo.md docs/tutorials/iit-4.0-demo.ipynb
git commit -m "Render the demo notebook from stored, guarded outputs"
```

---

### Task 7: Verification and merge

- [ ] **Step 1: Suites and build**

```bash
export PYPHI_WELCOME_OFF=1 PYPHI_AGENT_NOTE_OFF=1
uv run pytest -q > /tmp/final-fast.log 2>&1; echo $?; tail -1 /tmp/final-fast.log
uv run pytest test/docs -m slow --slow -q > /tmp/final-nb.log 2>&1; echo $?; tail -1 /tmp/final-nb.log
rm -rf docs/reference/_autosummary; just docs > /tmp/final-docs.log 2>&1; echo $?; grep -n "build succeeded" /tmp/final-docs.log
```
Expected: all exit 0. The full slow lane is unchanged by this work apart from the one new test, so the notebook test alone is the slow check here.

- [ ] **Step 2: Both themes, seven pages**

In a browser, light and dark: `index.html`, `getting-started/index.html`, `howto/read-result.html`, `howto/estimate-cost.html`, `theory/intrinsic-information.html`, `reference/configuration.html`, `examples/IIT_4.0_demo.html`. Check fonts, links, code blocks, cards, tables, admonitions, the announcement, the version dropdown, the sidebar footer, and that the narrowest window (400 px) does not scroll horizontally. Fix and re-commit anything off.

- [ ] **Step 3: Merge (no push)**

```bash
cd /Users/will/projects/pyphi
git merge --no-ff docs-style -m "Merge the documentation restyle"
git worktree remove .claude/worktrees/docs-style
git branch -d docs-style
```

Then fold the two fragments into the 2.0.0 changelog section (`.change.md` at the end of "API changes", `.doc.md` at the end of "Documentation", wrapped bullets with the fragment name in parentheses), `git rm` them, and commit "Fold the restyle fragments into the 2.0.0 changelog".
