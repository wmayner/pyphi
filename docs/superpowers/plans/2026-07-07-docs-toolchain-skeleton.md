# Docs Toolchain & Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the Sphinx toolchain and site skeleton per the documentation-overhaul design (`docs/superpowers/specs/2026-07-07-documentation-overhaul-design.md` §2, §3, §7 item 1): a green `-W` build of a six-section site with a build-time-generated API reference, executable-page machinery, docs CI, and an updated ReadTheDocs build.

**Architecture:** Sphinx with pydata-sphinx-theme and myst-nb; all new pages are MyST Markdown. The hand-written `docs/api/*.rst` stubs are deleted and replaced by autosummary pages generated at build time from a curated module list. Retained legacy `.rst` pages stay at their current paths and are wired into the new section toctrees; later content plans rewrite them in place.

**Tech Stack:** Sphinx, pydata-sphinx-theme, myst-nb, jupytext, sphinx-copybutton, sphinx-design, autodoc/napoleon/autosummary, GitHub Actions, ReadTheDocs, uv.

## Global Constraints

- Python 3.13+ only; no backward compatibility with older Pythons.
- All python commands run through `uv run`; docs dependencies live in a `docs` dependency group (PEP 735), not an extra.
- The docs build must pass `sphinx-build -W --keep-going` (warnings are errors) from Task 2 onward.
- Never `git add -A`; stage only the files named in each commit step. Unrelated working-tree changes (e.g. `AGENTS.md`) belong to concurrent sessions — leave them alone.
- No planning-artifact markers (P-numbers, B-numbers, "Wave") in committed docs content or changelog fragments.
- `docs/superpowers/**` (specs/plans) must be excluded from the Sphinx build.
- Legacy pages kept for later porting (`installation.rst`, `macos_installation.rst`, `configuration.rst`, `caching.rst`, `conventions.rst`, `tiebreaking.rst`, `examples/index.rst`, `examples/xor.rst`, `examples/macro.rst`, `examples/actual_causation.rst`, `examples/conditional_independence.rst`, `migration/from-substrate-modeler.md`) are NOT rewritten in this plan — only re-homed in the navigation.

---

## File structure

```
pyproject.toml                     modified: docs dependency group
justfile                           modified: docs / serve-docs recipes
.gitignore                         modified: generated API pages, jupyter cache
.readthedocs.yml                   rewritten: uv-based build
.pre-commit-config.yaml            modified: jupytext sync hook
.github/workflows/docs.yml         new: docs CI job
docs/conf.py                       rewritten from scratch
docs/index.md                      new landing page (replaces index.rst)
docs/getting-started/index.md      new section hub
docs/getting-started/first-computation.md   new executable stub page
docs/getting-started/first-computation.ipynb  jupytext-generated pair
docs/tutorials/index.md            new section hub
docs/howto/index.md                new section hub
docs/theory/index.md               new section hub
docs/reference/index.md            new section hub
docs/reference/api.md              new curated autosummary index
docs/migration/index.md            new section hub
docs/_templates/autosummary/module.rst   new autosummary template
docs/api/*.rst                     deleted (31 files)
docs/examples/{2014paper,residue,magic_cut}.rst   deleted
docs/_themes/                      deleted
docs/_templates/layout.html        deleted
docs/Makefile, docs/_static/Makefile, docs/index.rst   deleted
changelog.d/docs-toolchain.doc.md  new changelog fragment
ROADMAP.md                         modified: dashboard row
```

---

### Task 1: The `docs` dependency group

**Files:**
- Modify: `pyproject.toml` (the `[dependency-groups]` table, ~line 55)

**Interfaces:**
- Produces: a `docs` dependency group installable with `uv sync --group docs`; every later task runs Sphinx via `uv run --group docs sphinx-build …`.

- [ ] **Step 1: Edit the dependency groups**

In `pyproject.toml` under `[dependency-groups]`, remove `"sphinx"` and `"sphinx-rtd-theme"` from the `dev` list, and add a new group after `dev`:

```toml
docs = [
    "jupytext",
    "myst-nb",
    "pydata-sphinx-theme",
    "sphinx",
    "sphinx-copybutton",
    "sphinx-design",
]
```

- [ ] **Step 2: Lock and sync**

Run: `uv lock && uv sync --group docs --all-extras`
Expected: resolves and installs without error.

- [ ] **Step 3: Verify the toolchain imports**

Run: `uv run --group docs python -c "import myst_nb, pydata_sphinx_theme, sphinx_copybutton, sphinx_design, jupytext; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "Add a docs dependency group for the documentation toolchain"
```

---

### Task 2: Site skeleton replacement

Deletes the dead content, rewrites `conf.py`, and builds the six-section Diátaxis skeleton. Ends with a green `-W` build.

**Files:**
- Delete: `docs/api/` (entire directory), `docs/examples/2014paper.rst`, `docs/examples/residue.rst`, `docs/examples/magic_cut.rst`, `docs/_themes/` (entire directory), `docs/_templates/layout.html`, `docs/Makefile`, `docs/_static/Makefile`, `docs/index.rst`
- Rewrite: `docs/conf.py`
- Create: `docs/index.md`, `docs/getting-started/index.md`, `docs/tutorials/index.md`, `docs/howto/index.md`, `docs/theory/index.md`, `docs/reference/index.md`, `docs/migration/index.md`

**Interfaces:**
- Consumes: the `docs` dependency group (Task 1).
- Produces: `docs/conf.py` with `autosummary_generate = True`, `nb_execution_mode = "cache"`, `nb_execution_raise_on_error = True` already set (Tasks 3 and 4 rely on these); section hub pages whose toctrees Tasks 3 and 4 append to; the build command `uv run --group docs sphinx-build -W --keep-going -b html docs docs/_build/html`.

- [ ] **Step 1: Delete dead and dropped content**

```bash
git rm -r docs/api docs/_themes
git rm docs/examples/2014paper.rst docs/examples/residue.rst docs/examples/magic_cut.rst
git rm docs/_templates/layout.html docs/Makefile docs/_static/Makefile docs/index.rst
```

- [ ] **Step 2: Check whether `custom.css` is still referenced**

Run: `grep -rn "custom.css" docs --include="*.rst" --include="*.py" --include="*.md" | grep -v _build`
Expected: no hits outside the old `conf.py` (being rewritten this task). If none: `git rm docs/_static/custom.css`. If a retained page references it, keep the file.

- [ ] **Step 3: Rewrite `docs/conf.py`**

Replace the entire file with:

```python
"""Sphinx configuration for the PyPhi documentation."""

import os
from importlib.metadata import metadata

# Keep the import-time welcome banner out of autodoc's import of pyphi.
os.environ["PYPHI_WELCOME_OFF"] = "1"

project = "PyPhi"
author = "Will Mayner"
copyright = "2014–2026, Will Mayner and contributors"
release = metadata("pyphi")["Version"]
version = release

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "superpowers/**",
    "**/.ipynb_checkpoints",
    "examples/IIT_4.0_demo.ipynb",
    "examples/serialize_demo.ipynb",
]

# --- MyST / executable pages ------------------------------------------------

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "substitution",
]
nb_execution_mode = "cache"
nb_execution_timeout = 300
nb_execution_raise_on_error = True

# --- API reference ----------------------------------------------------------

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_use_rtype = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
}

# --- HTML output ------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_logo = "_static/pyphi-logo-text-776x196.png"
html_favicon = "_static/phi_144x144.png"
html_theme_options = {
    "github_url": "https://github.com/wmayner/pyphi",
    "navbar_align": "left",
    "header_links_before_dropdown": 6,
}
```

- [ ] **Step 4: Create the landing page `docs/index.md`**

```markdown
# PyPhi

PyPhi is a Python library for computing integrated information.

The formalism it implements (IIT 4.0) is described in:

> Albantakis L, Barbosa L, Findlay G, Grasso M, … Tononi G. (2023).
> Integrated information theory (IIT) 4.0: formulating the properties of
> phenomenal existence in physical terms.
> *PLoS Computational Biology* 19(10): e1011465.
> <https://doi.org/10.1371/journal.pcbi.1011465>

If you use this software in your research, please cite the software paper:

> Mayner WGP, Marshall W, Albantakis L, Findlay G, Marchman R, Tononi G. (2018).
> PyPhi: A toolbox for integrated information theory.
> *PLOS Computational Biology* 14(7): e1006343.
> <https://doi.org/10.1371/journal.pcbi.1006343>

To report issues, use the [issue tracker](https://github.com/wmayner/pyphi/issues).
For general discussion, join the [pyphi-users group](https://groups.google.com/forum/#!forum/pyphi-users).

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} Getting started
:link: getting-started/index
:link-type: doc
Install PyPhi and compute your first φ.
:::

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc
Learn the library through worked, executable examples.
:::

:::{grid-item-card} How-to guides
:link: howto/index
:link-type: doc
Recipes for specific tasks: configuration, parallelism, caching, export.
:::

:::{grid-item-card} Theory
:link: theory/index
:link-type: doc
How IIT 4.0's mathematics maps onto PyPhi's types and functions.
:::

:::{grid-item-card} Reference
:link: reference/index
:link-type: doc
The API reference, configuration options, and conventions.
:::

:::{grid-item-card} Migration
:link: migration/index
:link-type: doc
Moving to PyPhi 2.0 from earlier versions and related tools.
:::
::::

```{toctree}
:hidden:
:maxdepth: 1

getting-started/index
tutorials/index
howto/index
theory/index
reference/index
migration/index
```
```

- [ ] **Step 5: Create the section hubs**

`docs/getting-started/index.md`:

````markdown
# Getting started

```{toctree}
:maxdepth: 1

../installation
../macos_installation
```
````

`docs/tutorials/index.md` (the four `examples/` entries are legacy pages pending rewrite; the tutorials content plan replaces them):

````markdown
# Tutorials

```{toctree}
:maxdepth: 1

../examples/index
../examples/xor
../examples/macro
../examples/actual_causation
```
````

`docs/howto/index.md`:

````markdown
# How-to guides

```{toctree}
:maxdepth: 1

../configuration
../caching
../tiebreaking
```
````

`docs/theory/index.md`:

````markdown
# Theory

```{toctree}
:maxdepth: 1

../examples/conditional_independence
```
````

`docs/reference/index.md`:

````markdown
# Reference

```{toctree}
:maxdepth: 1

../conventions
```
````

`docs/migration/index.md`:

````markdown
# Migration

```{toctree}
:maxdepth: 1

from-substrate-modeler
```
````

- [ ] **Step 6: Build and drive warnings to zero**

Run: `uv run --group docs sphinx-build -W --keep-going -b html docs docs/_build/html`

Expected on first run: possible warnings from the retained legacy pages. Fix each until the build exits 0. Known likely categories:

- *"document isn't included in any toctree"* — a retained page not wired into a section hub: add it to the appropriate hub's toctree (or, for a file that should have been dropped, `git rm` it).
- *Unknown-document `:doc:`/`:ref:` targets* in retained pages pointing at the deleted `2014paper`/`residue`/`magic_cut` pages — replace the cross-reference with a citation of the relevant paper (Oizumi et al. 2014 for `2014paper` and `residue`; Hoel et al. 2013 for macro-level references) as plain text.
- *Image not found* — a deleted static file still referenced: restore that one file from HEAD~ (`git checkout HEAD~1 -- <path>`) rather than deleting the reference.
- *Lexing/highlighting warnings* in legacy code blocks — change the block's language to `text`.

Do NOT rewrite legacy page prose in this task; the minimal mechanical fix only.

- [ ] **Step 7: Verify the section structure renders**

Run: `ls docs/_build/html/getting-started/index.html docs/_build/html/tutorials/index.html docs/_build/html/howto/index.html docs/_build/html/theory/index.html docs/_build/html/reference/index.html docs/_build/html/migration/index.html`
Expected: all six files exist.

- [ ] **Step 8: Commit**

```bash
git add -u -- docs ':(exclude)docs/superpowers'
git add docs/conf.py docs/index.md docs/getting-started docs/tutorials docs/howto docs/theory docs/reference docs/migration
git commit -m "Replace the documentation skeleton

Six-section layout (getting started / tutorials / how-to / theory /
reference / migration) built on pydata-sphinx-theme and MyST. The
hand-written api/*.rst stubs, the pages superseded by the theory
section (2014paper, residue, magic_cut), and the 2014-era Sphinx
configuration and theme are removed. Retained pages are re-homed in
the new navigation pending rewrite."
```

(`git add -u docs` stages the deletions; confirm with `git status` that nothing outside `docs/` is staged.)

---

### Task 3: Build-time API reference

**Files:**
- Create: `docs/_templates/autosummary/module.rst`, `docs/reference/api.md`
- Modify: `docs/reference/index.md`, `.gitignore`

**Interfaces:**
- Consumes: `autosummary_generate = True` and `templates_path = ["_templates"]` from Task 2's `conf.py`.
- Produces: generated module pages under `docs/reference/_autosummary/` (gitignored, rebuilt every build).

- [ ] **Step 1: Create the autosummary module template**

`docs/_templates/autosummary/module.rst`:

```rst
{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}
   :members:
   :show-inheritance:

{% block modules %}
{% if modules %}
.. rubric:: Submodules

.. autosummary::
   :toctree:
   :recursive:
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}
```

- [ ] **Step 2: Create the curated API index `docs/reference/api.md`**

The module list below is the curated public surface; adding a top-level module to `pyphi/` requires adding a line here (the build fails loudly if a listed module disappears).

````markdown
# API reference

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :recursive:

   pyphi.actual
   pyphi.analyze
   pyphi.automorphism
   pyphi.cache
   pyphi.combinatorics
   pyphi.compositional_state
   pyphi.conf
   pyphi.connectivity
   pyphi.constants
   pyphi.convert
   pyphi.core
   pyphi.data_structures
   pyphi.direction
   pyphi.display
   pyphi.distribution
   pyphi.dynamics
   pyphi.examples
   pyphi.exceptions
   pyphi.formalism
   pyphi.graph
   pyphi.labels
   pyphi.log
   pyphi.macro
   pyphi.matching
   pyphi.measures
   pyphi.models
   pyphi.node
   pyphi.parallel
   pyphi.partition
   pyphi.protocols
   pyphi.provenance
   pyphi.registry
   pyphi.relations
   pyphi.resolve_ties
   pyphi.serializable
   pyphi.serialize
   pyphi.substrate
   pyphi.substrate_generator
   pyphi.sweep
   pyphi.system
   pyphi.timescale
   pyphi.types
   pyphi.utils
   pyphi.validate
   pyphi.visualize
   pyphi.warnings
```
````

- [ ] **Step 3: Wire it into the Reference hub and gitignore the output**

In `docs/reference/index.md`, change the toctree to:

````markdown
```{toctree}
:maxdepth: 1

api
../conventions
```
````

Append to `.gitignore`:

```
docs/reference/_autosummary/
```

- [ ] **Step 4: Build and fix autodoc warnings**

Run: `uv run --group docs sphinx-build -W --keep-going -b html docs docs/_build/html`

Expected on first run: possible autodoc warnings. Fix until the build exits 0. Known likely categories:

- *Failed to import a listed module* — if the module genuinely doesn't exist under that name, correct the line in `api.md`; do not delete listed public modules to silence the build.
- *Duplicate object description* (the same object documented on both a package page and its home module's page, via re-exports in `__init__.py`) — the home module's page is canonical. In order of preference: (1) rely on the package's `__all__` — autodoc documents only `__all__` members, so a package whose `__all__` excludes re-exports produces no duplicates; (2) for a specific package that still collides, hand-write that one package's page as a checked-in file under `docs/reference/` using `automodule` with `:exclude-members:` naming the re-exports, and remove that package from the `api.md` autosummary list; (3) if collisions are pervasive rather than isolated, stop and raise it rather than picking a broad mechanism unilaterally. If no duplicate warnings appear, change nothing.
- *Docstring formatting warnings* (bad indentation, unexpected section) — fix the docstring in `pyphi/` source; these are genuine docstring bugs and in scope.

- [ ] **Step 5: Verify coverage and currency**

```bash
ls docs/reference/_autosummary/ | wc -l          # expected: > 100 generated stubs
ls docs/reference/_autosummary/pyphi.substrate.rst docs/reference/_autosummary/pyphi.system.rst docs/reference/_autosummary/pyphi.formalism.rst
grep -rl "pyphi\.network\b\|pyphi\.subsystem\b" docs/_build/html/reference/ | wc -l   # expected: 0
```

- [ ] **Step 6: Commit**

```bash
git add docs/_templates/autosummary/module.rst docs/reference/api.md docs/reference/index.md .gitignore
git add -u pyphi   # only if Step 4 fixed docstrings; verify each staged hunk is yours
git commit -m "Generate the API reference at build time

A curated module index drives recursive autosummary generation; the
generated pages are build artifacts, not committed. A listed module
that disappears fails the -W build, so the reference cannot silently
drift from the package layout."
```

---

### Task 4: Executable-page machinery with notebook pairing

**Files:**
- Create: `docs/getting-started/first-computation.md` (+ generated `docs/getting-started/first-computation.ipynb`)
- Modify: `docs/getting-started/index.md`, `.pre-commit-config.yaml`, `.gitignore`

**Interfaces:**
- Consumes: `nb_execution_mode = "cache"` and `nb_execution_raise_on_error = True` from Task 2's `conf.py`.
- Produces: the pairing pattern (jupytext front matter + committed output-free `.ipynb` + download link + Colab badge) that every tutorial page in the tutorials content plan follows.

- [ ] **Step 1: Create the executable stub page**

`docs/getting-started/first-computation.md`. This is a machinery-proving stub; the full ten-minute walkthrough is written by the tutorials content plan.

````markdown
---
jupytext:
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Your first computation

{download}`Download this page as a Jupyter notebook <first-computation.ipynb>`
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wmayner/pyphi/blob/main/docs/getting-started/first-computation.ipynb)

This page verifies your installation by constructing the three-node example
system used throughout the documentation.

```{code-cell} python
import pyphi

pyphi.config.progress_bars = False
system = pyphi.examples.basic_system()
system
```

A full walkthrough — from a transition probability matrix to a φ-structure —
is being written; see the tutorials section in the meantime.
````

- [ ] **Step 2: Generate the paired notebook**

Run: `uv run --group docs jupytext --sync docs/getting-started/first-computation.md`
Expected: creates `docs/getting-started/first-computation.ipynb`. Verify it has no outputs: `grep -c '"outputs": \[\]' docs/getting-started/first-computation.ipynb` ≥ 1 and `grep -c 'execution_count": [0-9]' docs/getting-started/first-computation.ipynb` = 0.

- [ ] **Step 3: Wire into the hub and ignore the execution cache**

`docs/getting-started/index.md` toctree becomes:

````markdown
```{toctree}
:maxdepth: 1

../installation
../macos_installation
first-computation
```
````

Append to `.gitignore`:

```
docs/_build/
.jupyter_cache/
```

The paired `.ipynb` must NOT be gitignored — Colab opens it from the repository. Exclude the paired notebook from Sphinx (the `.md` is the rendered source; without this the build sees the same content twice). In `docs/conf.py` `exclude_patterns`, add:

```python
    "getting-started/first-computation.ipynb",
```

- [ ] **Step 4: Add the jupytext sync hook**

In `.pre-commit-config.yaml`, add as a new repo entry:

```yaml
  - repo: https://github.com/mwouts/jupytext
    rev: v1.17.2
    hooks:
      - id: jupytext
        args: [--sync]
        files: ^docs/.*\.md$
        require_serial: true
```

(If `pre-commit` reports the rev doesn't exist, run `pre-commit autoupdate --repo https://github.com/mwouts/jupytext` and use the rev it pins.) The `files` pattern is safe for non-paired pages: jupytext only syncs files carrying pairing front matter and leaves the rest untouched.

- [ ] **Step 5: Build and verify execution happened**

Run: `uv run --group docs sphinx-build -W --keep-going -b html docs docs/_build/html`
Expected: exit 0.

Run: `grep -c "cell_output" docs/_build/html/getting-started/first-computation.html`
Expected: ≥ 1 — myst-nb wraps executed-cell output in a `cell_output` container, proving build-time execution. Also verify the badge survived rendering: `grep -c "colab.research.google.com" docs/_build/html/getting-started/first-computation.html` ≥ 1.

- [ ] **Step 6: Verify the sync hook round-trips clean**

```bash
uv run --group docs jupytext --sync docs/getting-started/first-computation.md
git diff --stat docs/getting-started/
```
Expected: no diff (sync is idempotent).

- [ ] **Step 7: Commit**

```bash
git add docs/getting-started/first-computation.md docs/getting-started/first-computation.ipynb docs/getting-started/index.md docs/conf.py .pre-commit-config.yaml .gitignore
git commit -m "Add executable-page machinery with jupytext notebook pairing

Tutorial pages are MyST Markdown whose code cells execute during the
Sphinx build; each page is jupytext-paired with a committed output-free
notebook served as a download and a Colab link. A pre-commit hook keeps
pairs in sync."
```

---

### Task 5: CI, ReadTheDocs, recipes, and bookkeeping

**Files:**
- Create: `.github/workflows/docs.yml`, `changelog.d/docs-toolchain.doc.md`
- Rewrite: `.readthedocs.yml`
- Modify: `justfile` (the `docs` and `serve-docs` recipes), `ROADMAP.md` (dashboard row)

**Interfaces:**
- Consumes: the build command from Task 2; the `docs` group from Task 1.

- [ ] **Step 1: Update the justfile recipes**

Replace the current `docs` and `serve-docs` recipes with:

```make
# Build documentation (warnings are errors, matching CI)
docs:
    uv run --group docs sphinx-build -W --keep-going -b html docs docs/_build/html

# Serve documentation locally
serve-docs port="1337": docs
    cd docs/_build/html && uv run python -m http.server {{ port }}
```

Run: `just docs`
Expected: exit 0.

- [ ] **Step 2: Create `.github/workflows/docs.yml`**

```yaml
name: Docs

on:
  push:
    branches: [main]
    paths:
      - "docs/**"
      - "pyphi/**"
      - "pyproject.toml"
      - "uv.lock"
      - ".github/workflows/docs.yml"
  pull_request:
    branches: [main]
    paths:
      - "docs/**"
      - "pyphi/**"
      - "pyproject.toml"
      - "uv.lock"
      - ".github/workflows/docs.yml"

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Check out repository
        uses: actions/checkout@v4

      - name: Set up uv
        uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true
          cache-dependency-glob: "uv.lock"

      - name: Set up Python 3.13
        run: uv python install 3.13

      - name: Install dependencies
        run: uv sync --python 3.13 --group docs --all-extras

      - name: Build documentation (warnings are errors, cells execute)
        run: uv run --group docs sphinx-build -W --keep-going -b html docs docs/_build/html
```

Verify the YAML parses: `uv run python -c "import yaml, pathlib; yaml.safe_load(pathlib.Path('.github/workflows/docs.yml').read_text()); print('ok')"`
Expected: `ok`

- [ ] **Step 3: Rewrite `.readthedocs.yml`**

The current file installs from a `docs/requirements.txt` that does not exist (the RTD build is broken) and pins Python 3.12, below the package floor. Replace the whole file with the uv-based command build:

```yaml
# Read the Docs configuration file
# See https://docs.readthedocs.io/en/stable/config-file/v2.html for details
version: 2

build:
  os: ubuntu-24.04
  tools:
    python: "3.13"
  commands:
    - asdf plugin add uv
    - asdf install uv latest
    - asdf global uv latest
    - uv sync --group docs --all-extras
    - uv run --group docs sphinx-build -W --keep-going -b html docs $READTHEDOCS_OUTPUT/html
```

Verify the YAML parses: `uv run python -c "import yaml, pathlib; yaml.safe_load(pathlib.Path('.readthedocs.yml').read_text()); print('ok')"`
Expected: `ok`
(The RTD build itself can only be verified after a push; the command line is identical to the local `just docs` invocation apart from the output path.)

- [ ] **Step 4: Changelog fragment**

```bash
echo "Rebuilt the documentation toolchain: pydata-sphinx-theme, MyST Markdown with build-time-executed code cells, notebook pairing with Colab links, an API reference generated from the current module layout on every build, and a docs CI job that fails on any warning or failed cell." > changelog.d/docs-toolchain.doc.md
```

- [ ] **Step 5: Update the ROADMAP dashboard row**

In `ROADMAP.md`, change the "Documentation overhaul (2.0 / IIT 4.0)" dashboard row status from `⬜ open` to `🟡 partial` and prepend to its notes: "Toolchain & skeleton landed (pydata-sphinx-theme + myst-nb, six-section layout, build-time API reference, docs CI, RTD build fixed); remaining: docstring sweep, theory narrative, tutorials/how-tos, migration guide." Keep the rest of the row.

- [ ] **Step 6: Full verification**

```bash
just docs
uv run --all-extras pytest -q test/ -x -m "not slow" 2>&1 | tail -3
```
Expected: docs build exit 0; test lane green (docs changes shouldn't affect tests — this catches accidental `pyphi/` damage from Task 3's docstring fixes).

- [ ] **Step 7: Commit**

```bash
git add .github/workflows/docs.yml .readthedocs.yml justfile changelog.d/docs-toolchain.doc.md ROADMAP.md
git commit -m "Add docs CI and fix the ReadTheDocs build

CI builds the site with warnings-as-errors and executed cells on any
change to docs/ or pyphi/. The ReadTheDocs build previously referenced
a nonexistent requirements file and a Python below the package floor;
it now installs the docs dependency group via uv."
```

---

## As-built deviations

The landed implementation deviates from the task text above as follows:

- **API reference generation is fully recursive, not a curated list.** Task 3
  as written used a hand-maintained module list in `api.md` plus five
  checked-in override pages for the pure re-export packages
  (`pyphi.formalism`, `pyphi.models`, `pyphi.matching`, `pyphi.macro`,
  `pyphi.display`) whose `__all__` re-exports collided with recursive
  autosummary. A follow-up reworked this to the standard recursive
  per-object autosummary pattern: `api.md` is a single `:recursive:` entry
  on the root `pyphi` package, per-object page templates
  (`docs/_templates/autosummary/{module,class}.rst`) render members as
  autosummary tables so re-exports resolve to one canonical page, and the
  five override pages are deleted. No hand-maintained module list remains;
  new modules appear automatically. A `conf.py` `autosummary_filename_map`
  entry disambiguates the `relation`/`Relation` and `relations`/`Relations`
  page names, which collide on case-insensitive filesystems.
- `docs/conf.py` additionally carries the legacy `rst_prolog` substitution
  block (the retained `.rst` pages and 16 `pyphi/` docstrings depend on
  `|big_phi|`-style substitutions) and `napoleon_use_ivar = True` (resolves
  duplicate-attribute warnings from dataclass docstrings under autodoc). The
  substitutions are slated for removal: the docstring sweep replaces them in
  `pyphi/` with literal Unicode symbols, each content plan's rewrites remove
  them from its pages, and the final content plan deletes `rst_prolog`.
- The jupytext pre-commit hook is pinned to the `uv.lock`-resolved jupytext
  version (paired-page front matter embeds `jupytext_version`; a mismatched
  hook rewrites pairs on every commit). The two must move together.
- The `justfile` docs recipe uses `--all-extras` (CI and RTD already did):
  the API reference imports `pyphi.visualize`, which raises without the
  visualize extras.

## Self-review checklist (run after writing, before execution)

- Spec §2 coverage: conf.py rewrite ✓ (T2), theme ✓ (T2), extensions ✓ (T2), build-time API generation ✓ (T3), notebook delivery/pairing ✓ (T4), docs deps + RTD ✓ (T1, T5), `just docs` ✓ (T5).
- Spec §3 coverage: `-W` + execution in CI ✓ (T5), execution cache ✓ (T2 conf + T4 gitignore).
- Spec §7.1 deliverable: green `-W` build of complete-but-thin site with full current API reference ✓ (T2+T3).
- Out of scope, deferred to content plans: installation-page merge, ten-minute walkthrough content, legacy ports, canonical notebook rewrite, migration-2.0.md.
