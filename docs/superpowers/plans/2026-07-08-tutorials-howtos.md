# Tutorials & How-tos Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the teaching content of the site — the getting-started
walkthrough, five tutorial pages (including the paper-supplement notebook), and
seven how-to recipes — per
`docs/superpowers/specs/2026-07-08-tutorials-howtos-design.md`, retiring the
last legacy `.rst` pages and the `rst_prolog` block. Completes the
documentation overhaul.

**Architecture:** Executable MyST pages under `docs/getting-started/`,
`docs/tutorials/`, `docs/howto/`. Code cells run at build time (myst-nb).
Tutorial pages are jupytext-paired with committed, output-free `.ipynb` files
and carry a Colab badge; how-to pages execute but are not paired. The
paper-supplement notebook stays a true `.ipynb` at its exact path. Each page is
authored, then verified against the code and the build, before it is accepted.

**Tech Stack:** MyST Markdown, myst-nb (executed cells), jupytext, Sphinx `-W`.

## Global Constraints

- **Every code cell executes** under `uv run --all-extras --group docs
  sphinx-build -W --keep-going -b html docs docs/_build/html`. Set
  `pyphi.config.progress_bars = False` in the first cell of every executable
  page.
- **Executable page front matter** (required for myst-nb to run a page):
  ```
  ---
  jupytext:
    text_representation:
      extension: .md
      format_name: myst
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  ---
  ```
- **Tutorial pages are jupytext-paired** with a committed output-free `.ipynb`
  and a Colab badge; use the `md:myst,ipynb` formats front matter and the badge
  pattern from `docs/getting-started/first-computation.md`. Colab links target
  `main`. How-to pages use the plain executable front matter above (no `formats`,
  so no pairing).
- **LaTeX math** for equations (`$...$` / `$$...$$`); Unicode is fine in prose.
- **2.0 API only, verified.** `Network → Substrate`, `Subsystem → System`,
  `pyphi.analyze`. Every call shown is confirmed against the current code before
  it appears (`env -u VIRTUAL_ENV uv run python -c "…"`).
- **The paper-supplement notebook stays at the exact path**
  `docs/examples/IIT_4.0_demo.ipynb` — do not move or rename it.
- Commit trailer on every commit; never `--no-verify`; never `git add -A` (stage
  only the page(s) named); `docs/conf.py` and `AGENTS.md` may carry concurrent
  edits — stage only your hunks. No planning-artifact markers in content.

## Verified entry points (confirmed against the code)

`pyphi.analyze(substrate, state)` → `Analysis` (`.phi`, `.ces`, `.sia`,
`.system`); `pyphi.System(substrate, state, node_indices=…)`;
`pyphi.sweep`; `pyphi.macro` (`coarse_grain`, `blackbox`, `MacroSystem`);
`pyphi.actual` (`Transition`, `account`, `events`, `nexus`);
`analysis.ces.to_pandas()`; `pyphi.save` / `pyphi.load` (on serializable result
types, e.g. `analysis.ces`); `pyphi.config.infrastructure.parallel`.

---

## Shared asset: the page-authoring prompt

Dispatched per page with `{TITLE}`, `{FILE}`, `{SECTION}`, `{KIND}` (tutorial or
howto), `{SOURCE}` (legacy page to port, or "new"), `{SCOPE}`, `{REPORT_FILE}`.

```
You are writing one page of PyPhi's documentation. Repo: /Users/will/projects/pyphi

Page: {TITLE}
File to create: {FILE}   (MyST Markdown)
Section: {SECTION}   Kind: {KIND}

## What this page covers
{SCOPE}

## Source
{SOURCE}
If porting a legacy .rst page: convert it to MyST, update every code example to
the 2.0 API (Network→Substrate, Subsystem→System, pyphi.analyze), turn its
examples into executable {code-cell} blocks, and drop any |substitution|
markup in favor of LaTeX math or plain prose. Preserve the pedagogical content;
do not invent new material beyond clarifying updates.

## Requirements
- Executable MyST with the required front matter (see the plan's Global
  Constraints). First code cell: `import pyphi` then
  `pyphi.config.progress_bars = False`.
- If Kind is tutorial: add the jupytext `formats: md:myst,ipynb` front matter,
  a {download} link and a Colab badge (copy the pattern from
  docs/getting-started/first-computation.md), then run
  `uv run --group docs jupytext --sync {FILE}` to generate the paired .ipynb and
  confirm it is output-free.
- 2.0 API only. Before showing a call, confirm it against the code
  (env -u VIRTUAL_ENV uv run python -c "…"). Never show a call you have not run.
- Plain, precise prose (house style: no compressed shorthand). LaTeX for math.
- Build the page: env -u VIRTUAL_ENV uv run --all-extras --group docs
  sphinx-build -W --keep-going -b html docs docs/_build/html → exit 0, cells
  executed (grep the built HTML for expected output).

Do NOT run git. Report to {REPORT_FILE}: files created, every 2.0 call shown and
its confirmed output, build result, anything you could not port and why.
```

## Shared asset: the verify prompt

```
You are verifying one documentation page. File: {FILE}   Report: {REPORT_FILE}
Repo: /Users/will/projects/pyphi

Verify:
1. Every code call in the page is a real 2.0 API call that runs and produces the
   output shown (run it: env -u VIRTUAL_ENV uv run python -c "…").
2. No pre-2.0 API (Network, Subsystem, compute.*, cause_tpm, jsonify) and no
   |substitution| markup remain.
3. The page builds under -W with cells executed (build to a scratch dir if
   needed) and the prose matches the rendered output.
4. If a tutorial: the paired .ipynb exists, is output-free, and the Colab badge
   targets the correct path on main.
5. Prose is plain and precise; a ported page preserves the original's teaching
   content.
Read-only; do not modify or run git. Report findings (Critical/Important/Minor,
file:line, fix) and a verdict: Approved | Needs fixes.
```

## Page table

Each row (Tasks 2–14) is: author → build → verify → fix → accept → commit the
page (stage the page file, and for tutorials its paired `.ipynb`). Author and
verify on the standard model; the notebook task (Task 7) is called out
separately for extra care.

| # | Page / file | Section | Kind | Source | Scope |
|---|-------------|---------|------|--------|-------|
| 2 | `getting-started/first-computation.md` | Getting started | tutorial | expand existing stub | The ~10-minute first computation: install note, build `basic_substrate`, `analyze`, read `.phi` and the Φ-structure, save `analysis.ces`. Ends pointing to the tutorials. |
| 3 | `tutorials/cause-effect-structure.md` | Tutorials | tutorial | new | Hands-on distinctions and relations: from `analysis.ces`, iterate distinctions (`.mechanism`, `.phi`, purviews, a repertoire), iterate relations, `ces.to_pandas()`. Deeper than getting-started; links to the theory distinctions-and-relations page. |
| 4 | `tutorials/macro.md` | Tutorials | tutorial | `docs/examples/macro.rst` | Macro systems and blackboxing: `pyphi.macro` `coarse_grain` / `blackbox` / `MacroSystem`, updated to 2.0. |
| 5 | `tutorials/actual-causation.md` | Tutorials | tutorial | `docs/examples/actual_causation.rst` | Actual causation: `pyphi.actual` `Transition`, `account`, `events`; keeps the AC formalism, updated to 2.0. |
| 6 | `tutorials/worked-example.md` | Tutorials | tutorial | `docs/examples/xor.rst` | A small complete worked example (the XOR system), updated to 2.0 and IIT 4.0. |
| 7 | `tutorials/iit-4.0-demo.md` + `docs/examples/IIT_4.0_demo.ipynb` | Tutorials | notebook | update existing `.ipynb` | The paper supplement (see Task 7 detail below). |
| 8 | `howto/configure.md` | How-to | howto | `docs/configuration.rst` | Configure PyPhi: the layered config, `pyphi.config…`, `config.override(...)`, presets. |
| 9 | `howto/parallel.md` | How-to | howto | new | Run in parallel: `config.infrastructure.parallel` and the per-level `parallel_*_evaluation` knobs. |
| 10 | `howto/cache.md` | How-to | howto | `docs/caching.rst` | Cache results: in-memory and disk caches, the relevant config. |
| 11 | `howto/save-load.md` | How-to | howto | `docs/examples/serialize_demo.ipynb` | Save and load results: `pyphi.save` / `load`, `.save()` / `.load()`, formats (`.json` / `.mpk` / `.gz`). |
| 12 | `howto/export.md` | How-to | howto | new | Export results: `to_pandas`, xarray, and the DBN export (`pyphi.graph` / `substrate_to_dbn`). |
| 13 | `howto/sweep.md` | How-to | howto | new | Sweep parameter landscapes with `pyphi.sweep`, with optional parallelism. |
| 14 | `howto/tie-breaking.md` | How-to | howto | `docs/tiebreaking.rst` | Control tie-breaking: the tie-resolution config and its effect. |

Wire each page into its section's toctree (`docs/getting-started/index.md`,
`docs/tutorials/index.md`, `docs/howto/index.md`) as it lands, replacing the
legacy `../examples/*` / `../configuration` / `../caching` / `../tiebreaking`
entries.

## Task 1: Section toctree scaffolding

- [ ] **Step 1:** Create stub pages for all new files in Tasks 2–14 (title
  only), and rewrite the three section `index.md` toctrees to list the new page
  names (getting-started: `../installation`, `../macos_installation`,
  `first-computation`; tutorials: the five tutorial pages; howto: the seven
  how-to pages). Do not yet reference the legacy pages.
- [ ] **Step 2:** Mark every legacy page that is now orphaned (`examples/*.rst`
  not otherwise referenced, `configuration.rst`, `caching.rst`, `tiebreaking.rst`)
  with `:orphan:` so the `-W` build stays green until each is ported and deleted.
- [ ] **Step 3:** Build `-W` → exit 0. Commit the scaffolding.

## Task 7 detail: the paper-supplement notebook

**Files:** `docs/examples/IIT_4.0_demo.ipynb` (update in place — exact path),
`docs/tutorials/iit-4.0-demo.md` (a MyST page that renders/points to it).

- [ ] **Step 1:** Read the existing notebook's structure and update every cell
  to the 2.0 API. Preserve the pedagogical level and structure: installation,
  defining a substrate, the SIA unfolded by postulate (intrinsicality,
  information, integration $\varphi_s$, exclusion / first complex), composition
  into the Φ-structure (distinctions, relations, big Φ), and the reproduction of
  the IIT 4.0 paper Figures 1, 2, 4 with their derivations.
- [ ] **Step 2:** Apply the three additive improvements: (a) open with
  `analysis = pyphi.analyze(substrate, state)` as the one call that runs the
  whole pipeline, then unfold it; (b) cross-link each postulate section to its
  theory page; (c) close with a short "what next" (save a result, `to_pandas`)
  pointing to the how-to guides.
- [ ] **Step 3:** Ensure the notebook executes end to end
  (`env -u VIRTUAL_ENV uv run --group docs jupytext` is not needed — it is a
  true `.ipynb`; execute it via nbclient or by building the tutorials page that
  includes it). Keep it at the exact path; do not strip its identity as the
  supplement.
- [ ] **Step 4:** Create `docs/tutorials/iit-4.0-demo.md` that introduces the
  notebook and links to it (download + Colab badge to
  `docs/examples/IIT_4.0_demo.ipynb` on `main`), so the Tutorials section
  surfaces it.
- [ ] **Step 5:** Build `-W`; confirm the notebook renders and its cells show
  executed output. Commit the notebook + the tutorials page.

## Task 15: Retire legacy pages, delete rst_prolog, final gate

- [ ] **Step 1:** `git rm` the ported legacy pages: `docs/examples/macro.rst`,
  `docs/examples/actual_causation.rst`, `docs/examples/xor.rst`,
  `docs/examples/index.rst`, `docs/configuration.rst`, `docs/caching.rst`,
  `docs/tiebreaking.rst`, and `docs/examples/serialize_demo.ipynb`. Re-home any
  `:ref:` / `:doc:` targets that pointed at them (as the theory port did for
  `conditional-independence`).
- [ ] **Step 2:** Confirm no `|substitution|` markup remains in `docs/`:
  `grep -rn '|[A-Za-z]' docs --include="*.md" --include="*.rst" | grep -v superpowers`.
  Then delete the `rst_prolog` block from `docs/conf.py` (stage only that hunk).
- [ ] **Step 3:** Clean-slate `-W` build (`rm -rf docs/_build
  docs/reference/_autosummary` then build) → exit 0, every cell executed. If the
  flaky "search index couldn't be loaded" warning appears, rebuild once (it is a
  first-build artifact).
- [ ] **Step 4:** Confirm the tutorial notebooks are committed and output-free,
  and the demo notebook is at its exact path with a working badge.
- [ ] **Step 5:** Update the ROADMAP: mark the Documentation-overhaul row **done**
  (all five sub-projects landed) and note the overhaul complete. Commit.

## Self-review checklist

- Spec pages (getting-started, 5 tutorials, 7 how-tos) → Tasks 2–14 one-to-one. ✓
- Spec notebook (faithful update + 3 improvements + exact path) → Task 7. ✓
- Spec pairing (tutorials paired, how-tos not) → Global Constraints + page table
  Kind column. ✓
- Spec legacy retirement + rst_prolog deletion → Task 15. ✓
- Spec accuracy (every cell executes, 2.0 verified) → authoring + verify prompts,
  Global Constraints. ✓
- Spec success (build green, notebook at path, legacy retired, prolog gone,
  ROADMAP done) → Task 15. ✓
- Placeholder scan: page scopes name concrete 2.0 entry points (verified in
  "Verified entry points"); exact prose is written at execution. No TBDs.
