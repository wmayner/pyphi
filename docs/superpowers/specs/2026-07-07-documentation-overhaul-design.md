# Documentation overhaul (2.0 / IIT 4.0) — design

An umbrella design for rebuilding PyPhi's documentation as a comprehensive
resource for 2.0 and IIT 4.0. This spec fixes the site's information
architecture, toolchain, verification policy, and content inventory; the work
is executed as five separate implementation plans (§7), each planned against
this document.

## Current state

- All 31 `docs/api/*.rst` files use `automodule::` against the pre-2.0 module
  layout (`pyphi.network`, `pyphi.subsystem`, `pyphi.models.*`,
  `pyphi.compute.*`); most of those modules no longer exist. Nothing documents
  `substrate`, `system`, `formalism/`, `core/`, `conf/`, `analyze`, `sweep`,
  `serialize`, or `display`.
- Roughly 2,000 lines of `.rst` walkthroughs are written against the pre-2.0
  API. Their doctest blocks are not collected by pytest (`docs/` is not in
  testpaths), so the breakage is silent.
- `docs/conf.py` dates from sphinx-quickstart 2014: sphinx-rtd-theme, MathJax
  2.7 from a CDN, no intersphinx, and a latent `NameError` (`datetime` is used
  without being imported).
- The two notebooks (`IIT_4.0_demo.ipynb`, `serialize_demo.ipynb`) are not
  rendered into the site; the demo notebook is reachable only through a Colab
  link that points at the retired `feature/iit-4.0` branch.
- Migration content: `docs/migration/from-substrate-modeler.md` exists; the
  2.0 migration guide (`migration-2.0.md`, ship criterion #5) does not.

## Decisions

1. **Umbrella spec + per-piece plans.** One design (this document) fixes the
   site architecture; each sub-project (§7) gets its own implementation plan.
2. **Modern Sphinx stack, MyST-hybrid authoring.** Sphinx remains the engine
   (autodoc/napoleon is the only toolchain that handles PyPhi's lifted,
   PEP-562-lazy public surface reliably; ReadTheDocs stays turnkey). All new
   narrative content is authored in MyST Markdown; `.rst` survives only where
   legacy content is retained unconverted. This is the numpy/scipy/pandas
   pattern.
3. **Legacy walkthroughs: port the keepers, drop the rest.** Ported to the
   2.0 API: `actual_causation.rst`, `macro.rst`, `conditional_independence.rst`,
   `xor.rst`. Dropped: `2014paper.rst`, `residue.rst`, `magic_cut.rst` (IIT
   3.0-paper pedagogy; the Theory section's IIT 3.0 page cites the paper
   instead).
4. **Everything executes in CI.** Tutorial and how-to pages are executable;
   their code runs during the Sphinx build, and CI builds with warnings as
   errors. There is no separate docs-doctest harness (§3).
5. **Teaching content is text-source, notebook-delivery.** Tutorials are MyST
   Markdown pages whose code cells execute at build time. Each tutorial page
   is jupytext-paired with a committed, output-free `.ipynb`, exposed as a
   download button and a Colab badge. The MyST page is the source of truth;
   pairing is enforced mechanically (§2). Only the canonical IIT 4.0 demo is
   a notebook-first pairing (§4).
6. **Diátaxis information architecture** (§1).

## 1. Information architecture

Six top-level sections, mapped to the theme's header navigation:

- **Getting started** — a single installation page (merging
  `installation.rst` and `macos_installation.rst`, with platform tabs) and a
  ten-minute first-computation walkthrough ending at `pyphi.analyze(...)` on a
  three-node system.
- **Tutorials** — executable teaching pages, each with a notebook download and
  Colab badge:
  - the canonical IIT 4.0 walkthrough (paired with the demo notebook, §4);
  - computing a cause–effect structure (distinctions and relations);
  - macro systems and blackboxing (ported from `macro.rst`);
  - actual causation (ported from `actual_causation.rst`);
  - a small complete worked example (ported from `xor.rst`).
- **How-to guides** — one task per page: configure PyPhi (rewrite of
  `configuration.rst`); run computations in parallel; cache results (rewrite
  of `caching.rst`); save and load results (content of
  `serialize_demo.ipynb`); export to pandas/xarray/DBN; sweep parameter
  landscapes; control tie-breaking (from `tiebreaking.rst`).
- **Theory** — the IIT 4.0 narrative:
  - what IIT 4.0 computes: postulates → φ-structure, with equation citations
    to Albantakis et al. 2023 and the 2026 intrinsic-information cap;
  - the paper-to-code map: every named quantity in the paper → the runtime
    type/function that implements it;
  - the architecture guide: `Substrate → System → formalism → φ-structure`
    layering;
  - formalism versions: IIT 4.0 (2023) vs IIT 4.0 (2026) vs IIT 3.0 vs actual
    causation, and how configuration selects among them;
  - conditional independence (ported from `conditional_independence.rst`);
  - IIT 3.0 overview, citing Oizumi et al. 2014 for depth.
- **Reference** — the auto-generated API pages (§2), the configuration
  reference, conventions (state ordering / little-endian indexing), and the
  changelog.
- **Migration** — `migration-2.0.md` (new, §6) and
  `from-substrate-modeler.md` (moved under this section).

## 2. Toolchain

- **`conf.py` rewritten from scratch.** The 2014 quickstart file is replaced
  by a minimal configuration for the stack below.
- **Theme: pydata-sphinx-theme.** Header navigation carries the six sections;
  the version switcher integrates with ReadTheDocs.
- **Extensions:** myst-nb (MyST parsing, executed code cells, notebook
  rendering), autodoc + napoleon, autosummary, intersphinx (python, numpy,
  scipy, pandas, xarray), sphinx-copybutton, sphinx-design, MathJax 3
  (Sphinx-bundled, replacing the pinned 2.7 CDN).
- **API reference generated at build time.** The hand-written `docs/api/*.rst`
  files are deleted. A single curated reference index lists the public
  subpackages and modules; autosummary recursively generates per-module pages
  into a gitignored directory during the build. Generated pages cannot
  reference dead modules without failing the build, which is the property the
  hand-maintained stubs lacked.
- **Notebook delivery:** each tutorial page's MyST source is jupytext-paired
  with an output-free `.ipynb` committed alongside it (Colab can only open
  notebooks that exist at a GitHub URL, so the notebook artifacts must be in
  the repository). A pre-commit hook (`jupytext --sync` + check) keeps pairs
  in sync and outputs stripped. Each rendered page carries a download button
  and a Colab badge targeting the paired notebook on `main`.
- **Dependencies:** a `docs` dependency group in `pyproject.toml`;
  `.readthedocs.yml` updated to install it via uv. The `just docs` recipe
  updated accordingly (the manual CSS/PNG copy steps retire with the theme).

## 3. Verification

One mechanism: **CI builds the documentation with `-W --keep-going` and
notebook execution enabled.** Any failed code cell, unresolvable
cross-reference, or autodoc import failure fails the build. A GitHub Actions
docs job runs on pull requests touching `docs/` or `pyphi/`; myst-nb's
execution cache keeps incremental builds fast.

Content constraint: executed pages must run in minutes. Computations too heavy
for the build load pre-computed committed artifacts instead of computing live,
and say so in the page text.

Docstring doctests are unaffected: they already run under pytest via
testpaths.

## 4. The canonical notebook

`docs/examples/IIT_4.0_demo.ipynb` is rewritten against the 2.0 API as the
paper-supplement notebook. It remains a true `.ipynb` at a stable path
(external citations point to it), jupytext-paired with a MyST page so the site
renders it and diffs review as text. `serialize_demo.ipynb` is retired; its
content becomes the save/load how-to page.

## 5. Docstring sweep

Every docstring under `pyphi/**` rewritten in final-state voice: describe what
the object is and does, not what it was, replaces, or how it came to be.
Remove migration-journey phrasing and design-decision narrative. This is the
text autodoc renders, so it is content work for the Reference section, not
cleanup.

## 6. Migration guide

`migration-2.0.md` documents every API change for users of pre-2.0 PyPhi
(1.x/PyPhi-paper era and the IIT 4.0 feature-branch era): the renames
(`Network → Substrate`, `Subsystem → System`, `cause_tpm`/`effect_tpm`, module
moves), formalism selection and dispatch, the layered configuration format
(and the rejection of legacy flat YAML with a rename map), the
jsonify → msgspec serialization break with the migration tool, and changed
defaults. Organized as a rename table plus per-topic prose, written from the
change history rather than memory.

## 7. Sub-projects and ordering

Five implementation plans, in dependency order:

1. **Toolchain & skeleton** — conf.py, theme, extensions, six-section
   skeleton with stub pages, build-time API generation, deletion of dead
   `api/*.rst`, docs CI job, ReadTheDocs config, `just docs`. Deliverable: a
   green `-W` build of a complete-but-thin site with a full, current API
   reference. Everything else depends on this.
2. **Docstring sweep** (§5) — independent of #1; can run in parallel with it.
3. **Theory narrative** (§1, Theory section).
4. **Tutorials & how-tos** (§1, Tutorials + How-to sections, including the
   canonical notebook rewrite and the four legacy ports).
5. **Migration guide** (§6) **+ changelog condense** — the changelog-fragment
   condense to first-encounter voice rides with this plan.

Plans 3–5 are content-independent of one another and may land in any order.

## Success criteria

- The docs CI job is green with `-W` and executed cells.
- Every public module has a generated reference page; no page anywhere
  references the pre-2.0 API.
- All six sections are populated; the four legacy ports execute against the
  2.0 API; the dropped pages are deleted.
- `migration-2.0.md` ships (ship criterion #5), and the ROADMAP dashboard row
  for the documentation overhaul is updated as each sub-project lands.
