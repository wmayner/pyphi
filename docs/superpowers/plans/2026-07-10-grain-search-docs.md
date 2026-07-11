# Grain-Search Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Document the grain search across the Diátaxis framework: a how-to recipe, a theory page, blackboxing + temporal-grain tutorial sections, and a complexity-page cost section.

**Architecture:** Four independent page tasks (author → execute-verify → build-verify → commit), following the docs conventions established by the documentation overhaul: how-tos are executable MyST (unpaired), tutorials are jupytext-paired with output-free `.ipynb`, theory pages are prose with verified citations. Every number shown comes from an executed cell or a value verified live in this plan.

**Tech Stack:** MyST Markdown, myst-nb, jupytext, Sphinx.

**Spec:** `docs/superpowers/specs/2026-07-10-grain-search-docs-design.md`

## Global Constraints

- `uv run` for all python (`env -u VIRTUAL_ENV uv run --all-extras --group docs …` for sphinx). Never `--no-verify`. Stage only files the task touches. Pre-commit hooks must pass (jupytext hook re-syncs paired notebooks — stage the `.ipynb` when it changes).
- Plain, precise prose (house style: no compressed shorthand, no planning artifacts). Unicode symbols in prose (φₛ, Φ, τ); LaTeX math in MyST where needed.
- **Never cite a number from memory**: every equation/figure number verified against `papers/2024__marshall-et-al__intrinsic-units.pdf`; every quantitative claim in prose matches an executed cell or a plan-verified value.
- All demo cells run under `with pyphi.config.override(**presets.iit4_2023):` (import `from pyphi.conf import presets`), with `pyphi.config.progress_bars = False` in the setup cell.
- **Build verification recipe (per page):** a cold full `-W` build currently fails on a PRE-EXISTING `pyphi.mcp.content.topics` autosummary warning unrelated to this work. Verify pages with:
  `env -u VIRTUAL_ENV uv run --all-extras --group docs sphinx-build -b html docs docs/_build/html 2>/tmp/sphinx-warnings.log; echo exit=$?` → exit 0, then `grep <your-page-filename> /tmp/sphinx-warnings.log` → no output (no NEW warnings mention your page), then grep the built page's HTML for an executed output value named in the task.
- Changelog: one doc fragment for the whole plan (created in Task 1; later tasks do not add more).

## Verified reference values (all verified live, 2026-07-10, presets.iit4_2023)

- **min-substrate** (`test/macro/test_macro_criteria.py::min_substrate`, n = 2, state `(0, 0)`; for docs pages REBUILD it inline — docs cannot import from `test/`): `tpm = np.array([[0.05, 0.05], [0.05, 0.35], [0.35, 0.05], [0.85, 0.85]])` — VERIFY this reproduces `min_substrate()` before using (Task 1 Step 1 shows how). Default grain sweep: 8 systems evaluated, winner is the both-on coarse-grain of `{0, 1}`, φₛ = 0.7883339770634884 at 1e-13. `SearchBounds().estimate(substrate)`: `distinct_systems_upper_bound == 8` (worst case achieved), `systems_by_unit_count == {1: 7, 2: 1}`, `partition_sweeps_upper_bound == 10`, `is_exact False`.
- **Bare-state error** under τ > 1 bounds: `ValueError: micro_history must be a sequence of 2 universe states (oldest first); got a bare state`.
- **Temporal specimen (tutorial):** fn_table `[2, 3, 4, 3, 0, 0, 5, 0]` (state index → next state index, little-endian, A = bit 0), history `[(0, 0, 1), (0, 0, 0)]`, `SearchBounds(max_update_grain=2)`. Result: exactly two complexes, both temporal — `{A}` at τ=2, mapping `(0, 0, 1, 1)`, φₛ = 0.508305 (6 s.f.), `exclusion_margin` = 0.300787, with the full micro universe (φₛ = 0.2075) in its `excluded`; `{B, C}` at τ=2, φₛ = 0.462996, margin = 0.199962.
- **Mixed-grain mention:** fn_table `[3, 1, 1, 7, 1, 2, 2, 7]`, history `[(1, 0, 0), (1, 0, 0)]` → whole-universe winner with grains (1, 2, 1), φₛ = 0.314099 vs all-micro 0.276692.
- **Hit rate:** 23 of 120 seeded random runs (`experiments/grain_tau_experiments/README.md`).
- **Partition counts** (`DIRECTED_SET_PARTITION`): m = 1 → 1, 2 → 3, 3 → 22, 4 → 150, 5 → 1,061, 6 → 7,896.
- **Mapping counts:** surjective tables for |V| constituents at update grain τ′: 2^(2^(τ′·|V|)−1) − 1. The paper's Fig. 3E numbers MUST be re-verified against the PDF in Task 2/4 before citing.
- **Measured anatomy** (exploration §2.2, n = 4 Example-1 substrate, default bounds): 80 systems, ≈ 0.85 s, ≈ 92% SIA / 8% construction — re-verify order of magnitude in Task 4 before citing wall-clock.

---

### Task 1: `docs/howto/grain-search.md`

**Files:**
- Create: `docs/howto/grain-search.md`
- Modify: `docs/howto/index.md` (toctree: add `grain-search` after `sweep`)
- Create: `changelog.d/grain-search-docs.doc.md`

**Interfaces:**
- Consumes: `pyphi.analyze(substrate, state, grains=True | SearchBounds(...))` → `ComplexesResult`; `SearchBounds.estimate(substrate)` → `SearchEstimate`; `pyphi.macro.complexes`; `Complex.units/.phi/.exclusion_margin/.effectively_tied/.excluded`; `ComplexesResult.records/.ties/.maximal_complex`.

- [ ] **Step 1: Verify the inline substrate reproduces the fixture** — run:

```bash
uv run python -c "
import sys; sys.path.insert(0, 'test')
import numpy as np
from test.macro.test_macro_criteria import min_substrate
print(np.asarray(min_substrate().tpm.to_numpy() if hasattr(min_substrate().tpm, 'to_numpy') else min_substrate().tpm))"
```

Record the printed TPM; the page's inline `np.array` MUST match it exactly. (If the attribute access differs, inspect `min_substrate` in `test/macro/test_macro_criteria.py` and print its TPM however it exposes it.)

- [ ] **Step 2: Author the page.** Front matter: the executable-MyST header used by `docs/howto/tie-breaking.md` (jupytext text_representation only — how-tos are NOT paired). Title: `# Search across grains`. Required content, in order, each demo an executed `{code-cell}`:

1. One-paragraph framing: the intrinsic-units search asks which units, at which spatial and temporal grain, are intrinsic; it is combinatorial, so pre-flight first. Link `{doc}`../theory/macro-units`` and the macro tutorial.
2. **Pre-flight the cost**: build the two-unit substrate inline (verified TPM from Step 1), `bounds = SearchBounds()`, `est = bounds.estimate(substrate)`, print `est.distinct_systems_upper_bound`; prose explains `is_exact` (exact only at `max_depth=0`; enumeration counts, not records), `truncated`, `partitions_capped`, and that the worst case assumes every candidate passes.
3. **Run the search**: `result = pyphi.analyze(substrate, (0, 0), grains=True)`; prose: `grains=True` is default `SearchBounds()`; `grains=SearchBounds(...)` for control; `pyphi.macro.complexes(substrate, state, bounds)` is the underlying driver. Print the winner's φₛ (rounded, must show 0.788334).
4. **Supply a micro history for τ > 1**: show that `pyphi.analyze(substrate, (0, 0), grains=SearchBounds(max_update_grain=2))` raises, using a `{code-cell}` with the `raises-exception` tag OR a try/except printing the message; the message shown must be the verified bare-state error. Then show the correct call with a two-state history (`[(0, 0), (0, 0)]`) succeeding. Prose: the required length is `max_update_grain ** max_depth`, oldest first.
5. **Read the result**: iterate `result.complexes` printing `node_indices`, φₛ, unit grains; show `winner.exclusion_margin` and `winner.effectively_tied`; show one `excluded` record; prose notes shadows (higher-φₛ excluded candidates, link the recursive-exclusion tutorial), `result.records` (every evaluated system — compare to the estimate), `result.ties`.
6. **Parallelize**: prose + a NON-executed code block (```` ```python ````, not `{code-cell}`) showing `parallel_kwargs={...}` forwarded, linking `{doc}`parallel``.
7. **Bound the search**: a short table or list of the `SearchBounds` knobs (`max_constituents`, `max_update_grain`, `max_depth`, `mappings`, `exhaustive_cap`, `apportionment`/`max_background`) and the cost each drives, linking `{doc}`../theory/computational-complexity``.

- [ ] **Step 3: Add the toctree entry** in `docs/howto/index.md` (after `sweep`).

- [ ] **Step 4: Verify execution and build.** Run the page's cells end to end via a scratch script mirroring them exactly (`uv run python`), confirming: estimate prints 8; winner φₛ prints 0.788334; the τ error message matches; the history call succeeds. Then the build recipe from Global Constraints; grep the built `docs/_build/html/howto/grain-search.html` for `0.788334`.

- [ ] **Step 5: Changelog fragment**

```bash
echo 'Documented the grain search across the docs: a how-to recipe (`docs/howto/grain-search.md`), a theory page on macro units and exclusion across grains, blackboxing and temporal-grain tutorial sections (with a substrate whose maximal complex is a τ=2 unit), and a cost section on the complexity page.' > changelog.d/grain-search-docs.doc.md
```

- [ ] **Step 6: Commit**

```bash
git add docs/howto/grain-search.md docs/howto/index.md changelog.d/grain-search-docs.doc.md
git commit -m "Add the grain-search how-to guide"
```

### Task 2: `docs/theory/macro-units.md`

**Files:**
- Create: `docs/theory/macro-units.md`
- Modify: `docs/theory/index.md` (toctree: add `macro-units` after `phi-structure`)

**Interfaces:**
- Consumes: the concepts and types only — `MacroUnit`, `MacroSystem`, `SearchBounds`, `ComplexesResult`, `SearchEstimate`, `judge_candidate`/`UnitVerdict`/`Reason`, `pyphi/condensation.py`; `papers/2024__marshall-et-al__intrinsic-units.pdf` for every citation.

- [ ] **Step 1: Verify the citations.** Read the relevant parts of `papers/2024__marshall-et-al__intrinsic-units.pdf` and record, with page/figure/equation numbers actually seen: the unit-definition equations (the spec references Eqs. 15–16 for the criteria, Eq. 18 for disjoint assembly, Eq. 19 for exclusion, Eq. 13 for the mapping count, Fig. 3D grain raising, Fig. 3E mapping-count illustration — confirm each number or correct it). Do not write any equation number you did not see in the PDF.

- [ ] **Step 2: Author the page.** Prose theory page (no front matter needed beyond what sibling theory pages use — check `docs/theory/system-integration.md`'s header and match it). Title: `# Macro units and grains`. Required sections:

1. **Units at a grain** — a macro unit is a state mapping over a window of its micro constituents' sequence states; constituents, update grain τ′, and the mapping; coarse-graining (grain-1 on-count classes) vs blackboxing (output subsets, any grain); grains compose multiplicatively down the hierarchy (`max_update_grain ** max_depth`). Map to `MacroUnit` (`constituents`, `update_grain`, `mapping`), `coarse_grain`, `blackbox`, `micro_unit`.
2. **The intrinsic-unit criteria** — the candidate decomposition must be integrated (its own system φₛ > 0) and no competitor over the same footprint may beat it; one verdict covers every mapped and grained variant. Map to `judge_candidate`, `UnitVerdict`, `Reason` (name `NOT_INTEGRATED` as the gate the 4-cycle example in the tutorial hits).
3. **Exclusion across grains** — candidate systems at every grain compete in ONE exclusion cascade on micro footprints, applied recursively (an excluded candidate cannot exclude others); shadows (excluded candidates with higher φₛ) and selection margins (`Complex.exclusion_margin`); link `{doc}`../tutorials/recursive-exclusion`` and `{doc}`../howto/tie-breaking``. Map to `pyphi/condensation.py`, `ComplexesResult`.
4. **From theory to the library** — a short table: quantity/notion → type/function (`MacroUnit`, `MacroSystem`, `SearchBounds`, `pyphi.macro.complexes` / `analyze(grains=...)`, `ComplexesResult`, `SearchEstimate`).
5. **References** — Marshall et al. (2024) with the verified venue string (copy the exact form used by `docs/tutorials/macro.md`'s references), and Albantakis et al. (2023) for the exclusion postulate.

No uncertified exploration claims (no SLEM cap, no ii-gate, no cost conjectures).

- [ ] **Step 3: Add the toctree entry** in `docs/theory/index.md` after `phi-structure`.

- [ ] **Step 4: Build-verify** per the Global Constraints recipe; grep the warnings log for `macro-units` → no output; open-check the built HTML contains the section headings.

- [ ] **Step 5: Commit**

```bash
git add docs/theory/macro-units.md docs/theory/index.md
git commit -m "Add the macro-units theory page"
```

### Task 3: Tutorial sections — blackboxing and temporal grains

**Files:**
- Modify: `docs/tutorials/macro.md` (+ paired `docs/tutorials/macro.ipynb` via jupytext)

**Interfaces:**
- Consumes: `pyphi.macro.blackbox(num_constituents, update_grain, outputs)`, `coarse_grain(num_constituents, on_counts)`, `pyphi.analyze(..., grains=SearchBounds(max_update_grain=2))`; the temporal specimen from the reference values.

- [ ] **Step 1: Author the blackboxing section** — insert `## Blackboxing` after the "Rediscovering the paper's coarse-graining example" section. Content: prose contrast (coarse-graining pools states by on-count; blackboxing reads an output subset, hiding the rest — and it is the family that extends to update grains above 1); one `{code-cell}` printing `coarse_grain(2, (0, 2))` and `blackbox(2, 1, (0,))` side by side with a sentence interpreting each table; one `{code-cell}` building a blackboxed `MacroUnit` on the tutorial's existing example substrate and showing its analysis runs (reuse the page's existing substrate and state variables — read the page first and fit in).

- [ ] **Step 2: Author the temporal section** — insert `## Temporal grains` after Blackboxing. Content, with each claim backed by the executed cells:

1. Prose: units may also exist over several micro updates; a τ=2 unit's state is a mapping over two-step sequences; analyses then need a two-state micro history.
2. `{code-cell}`: build the specimen substrate from the function table (show the table as data — `fn_table = [2, 3, 4, 3, 0, 0, 5, 0]` — with a comment explaining state-index → next-state-index, little-endian, A = bit 0); construct the 8×3 TPM with a loop; `history = [(0, 0, 1), (0, 0, 0)]`.
3. `{code-cell}`: `result = pyphi.analyze(substrate, history, grains=SearchBounds(max_update_grain=2))`; print each complex's `node_indices`, φₛ (4 decimals), and its units' `micro_grain`s. Expected printed values: two complexes, φₛ 0.5083 (grains (2,)) and 0.4630 (grains (2,)).
4. `{code-cell}`: show the winner's unit (`result.maximal_complex.units[0]`) and `exclusion_margin`; show that the full micro universe appears in `excluded` with φₛ 0.2075 (loop over `excluded` printing the record for footprint `(0, 1, 2)` at grain 1).
5. Prose (honest framing): on symmetric substrates such as a deterministic rotation the integration criterion rejects pair decompositions before any temporal variant is built, while asymmetric substrates make temporal units win outright — a seeded random search found such wins in roughly one run in five; the winning unit here reads A every second step, and that view of the substrate has more integrated information than the whole micro system. One-sentence pointer: the mixed-grain case also occurs (units of one system at different grains).

- [ ] **Step 3: Sync and verify.** `uv run jupytext --sync docs/tutorials/macro.md`; run the new cells end to end in order via a scratch script; confirm printed φₛ values 0.5083 / 0.4630 and the 0.2075 excluded record; build-verify per the recipe; grep the built tutorial HTML for `0.5083`.

- [ ] **Step 4: Commit**

```bash
git add docs/tutorials/macro.md docs/tutorials/macro.ipynb
git commit -m "Teach blackboxing and temporal grains in the macro tutorial"
```

### Task 4: Complexity page — the cost of the grain search

**Files:**
- Modify: `docs/theory/computational-complexity.md` (new `## The cost of the grain search` between "The cost of IIT 4.0" and "Reducing the cost")

**Interfaces:**
- Consumes: `SearchBounds.estimate`; the reference values (partition counts, mapping-count formula); `papers/2024__marshall-et-al__intrinsic-units.pdf` for the Fig. 3E numbers.

- [ ] **Step 1: Verify the two citable measurements.**
  (a) Fig. 3E mapping counts: read the figure in the PDF; record the exact numbers it states (the exploration quotes 32,727 for 4 micro constituents at τ′ = 1 — confirm or correct against the figure).
  (b) Order-of-magnitude anatomy: run the n = 4 Example-1 sweep once (`pyphi.examples` — find the 2024-paper example substrate; if none exists under that name, reconstruct from `test/macro/` fixtures or SKIP the wall-clock sentence entirely and keep only the systems count, which `estimate` can state) and record systems evaluated + wall time. Cite only what you measured.

- [ ] **Step 2: Author the section.** Match the page's existing voice (read the neighboring sections first). Required content:

1. **The axes of the sweep** — decompositions (set-partition growth, capped by `max_constituents`) × mappings (2^(2^(τ′·|V|)−1) − 1 surjective tables, citing the verified equation number; the doubly-exponential axis, with the verified Fig. 3E illustration) × update grains (multiplicative composition) × assemblies (disjoint unit sets). Note the structural relief: the criteria judge a decomposition once, covering all its mapped and grained variants.
2. **What one candidate costs** — macro-TPM construction Θ(τ·4ⁿ) per distinct (footprint, grain) key; the SIA partition sweep growing with macro unit count m — the live-verifiable table m = 3 → 22, 4 → 150, 5 → 1,061, 6 → 7,896 under `DIRECTED_SET_PARTITION`.
3. **Measured shape** — whatever Step 1(b) verified, stated as measured on this hardware and version.
4. **Pre-flight before running** — a `{code-cell}` (this page already executes cells — CHECK; if the page is prose-only RST-style MyST without kernel, present the estimate as a non-executed literal block with its verified output inline): `SearchBounds().estimate(substrate)` on the how-to's two-unit substrate showing `distinct_systems_upper_bound == 8`; prose: the bound is an exact worst case (all-pass); level-1 judgments exact; `limit` and truncation.

- [ ] **Step 3: Build-verify** per the recipe; grep the warnings log for `computational-complexity` → no output.

- [ ] **Step 4: Commit**

```bash
git add docs/theory/computational-complexity.md
git commit -m "Add the grain-search cost section to the complexity page"
```

### Task 5: Full verification

- [ ] **Step 1:** `uv run pytest` (NO path argument) — all green (docs-only changes; the doctest sweep guards the tutorial).
- [ ] **Step 2:** One final full docs build per the recipe; confirm no warnings mention any of the four touched pages.
- [ ] **Step 3: If anything fails,** fix within the task that introduced it; do not proceed with failures.
