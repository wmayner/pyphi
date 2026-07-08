# Tutorials & how-tos — design

The fifth and final sub-project of the documentation overhaul
(`docs/superpowers/specs/2026-07-07-documentation-overhaul-design.md` §1,
Tutorials and How-to sections). Fills the getting-started walkthrough, the
tutorial pages (including the paper-supplement notebook), and the how-to
recipes — the teaching content of the site. This retires the last legacy
`.rst` example pages and the two example notebooks.

## Pages

**Getting started.** Flesh out the existing `docs/getting-started/first-computation.md`
stub into a real ~10-minute first walkthrough: install → build a three-unit
substrate → `pyphi.analyze` → read `.phi` and the Φ-structure → save a result.
The reader's first successful computation, end to end.

**Tutorials** — executable MyST pages, each jupytext-paired with a committed,
output-free `.ipynb` and carrying a Colab badge:

- **The IIT 4.0 demo** — the paper-supplement notebook (§3), rendered here.
- **Computing a cause–effect structure** — distinctions and relations hands-on,
  going deeper than getting-started.
- **Macro and blackboxing** — ported from `docs/examples/macro.rst`.
- **Actual causation** — ported from `docs/examples/actual_causation.rst`.
- **A complete worked example** — ported from `docs/examples/xor.rst`.

**How-to guides** — one task per page, executable MyST (not paired to notebooks;
they are short recipes):

- **Configure PyPhi** — rewrite of `docs/configuration.rst`.
- **Run computations in parallel.**
- **Cache results** — rewrite of `docs/caching.rst`.
- **Save and load results** — the content of `docs/examples/serialize_demo.ipynb`.
- **Export results** — to pandas, xarray, and a dynamic Bayesian network.
- **Sweep parameter landscapes.**
- **Control tie-breaking** — from `docs/tiebreaking.rst`.

## Executability and notebook delivery

Consistent with the toolchain (`docs/getting-started/first-computation.md` is
the established pattern): tutorial and how-to pages are executable MyST — code
cells run at build time via myst-nb. Tutorial pages additionally are
jupytext-paired with a committed, output-free `.ipynb` and carry a Colab badge,
so a reader can open any tutorial as a runnable notebook; how-to pages execute
but are not paired.

## The paper-supplement notebook

`docs/examples/IIT_4.0_demo.ipynb` is a **faithful update of the existing
notebook** — the same pedagogical level and structure, updated to the 2.0 API
and clarified — kept as a true `.ipynb` at the **exact current path** so the
published paper-supplement link resolves to the new version. Its structure is
preserved: installation, defining a substrate, the system irreducibility
analysis unfolded postulate by postulate (intrinsicality, information,
integration $\varphi_s$, exclusion / the first complex), composition into the
Φ-structure (distinctions, relations, big Φ), and the reproduction of the IIT
4.0 paper's Figures 1, 2, and 4 with their detailed derivations.

Three additive improvements, none removing content or lowering the level:

1. **Lead with `pyphi.analyze()`, then unfold it.** Open with the single call
   that runs the whole pipeline, then take it apart postulate by postulate —
   the easy path first, then the depth. This showcases the 2.0 entry point.
2. **Cross-link each section to the theory pages.** The notebook now has a
   companion theory section; each step links to its theory page for the deeper
   *why*, while the notebook keeps its own explanations and self-containment.
3. **A short "what next" close** — save a result, export to a DataFrame —
   pointing to the how-to guides.

The notebook is jupytext-paired with a MyST page so the Tutorials section
renders it; the `.ipynb` at the exact path remains the source of truth for the
external link.

## Accuracy and legacy ports

Every code cell executes at build time (the guarantee the theory pages already
meet), so the pages cannot drift from the code. The three tutorial ports
(`macro`, `actual_causation`, `xor`) and the how-to rewrites are updated to the
2.0 API — `Network → Substrate`, `Subsystem → System`, `pyphi.analyze` — and
each is verified to run. Actual causation keeps the actual-causation formalism;
the IIT examples use the IIT 4.0 default, naming the formalism explicitly where
a page teaches 3.0-specific behavior.

## Retiring the legacy pages

Porting these pages removes the last consumers of the `rst_prolog` substitution
block in `docs/conf.py`: after `macro.rst`, `actual_causation.rst`, `xor.rst`,
`configuration.rst`, `caching.rst`, `tiebreaking.rst`, and `examples/index.rst`
are ported and deleted, and the two example notebooks retired, the `rst_prolog`
block is deleted (it exists only for those retained `.rst` pages). This
completes the substitution-removal begun in the docstring sweep.

## Success criteria

- The getting-started, tutorial, and how-to pages are all present, wired into
  their section toctrees, and build clean under `-W` with every cell executed.
- The paper-supplement notebook is a faithful, 2.0-updated version at the exact
  path `docs/examples/IIT_4.0_demo.ipynb`, with a working Colab badge, rendered
  in the Tutorials section.
- The legacy `docs/examples/*.rst`, `docs/configuration.rst`, `docs/caching.rst`,
  `docs/tiebreaking.rst`, and `docs/examples/serialize_demo.ipynb` are retired,
  their content re-homed in the new pages.
- The `rst_prolog` block is removed from `docs/conf.py`, and no substitution
  markup remains in the docs.
- The tutorial notebooks are committed, output-free, and jupytext-paired.
- The documentation overhaul is complete; the ROADMAP row is updated to done.
