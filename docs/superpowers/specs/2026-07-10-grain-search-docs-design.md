# Grain-search documentation — design

**Date:** 2026-07-10
**Status:** Approved design, pending implementation plan
**Depends on:** complex unification, exclusion margins, `SearchBounds.estimate`, the `analyze()` grain axis (all landed)
**Relates to:** the documentation overhaul (`2026-07-07-documentation-overhaul-design.md`); the grain exploration (`2026-07-07-grain-discovery.md`)

## Goal

Document the grain search across the Diátaxis framework: a how-to recipe,
a theory page (the theory section currently has zero macro coverage),
tutorial sections for blackboxing and temporal grains, and a complexity
subsection with the verified cost model.

## Verified specimen (temporal grains)

The exploration's promised temporal demo (deterministic 4-cycle, τ=2 at
φₛ=2.0) does **not** reproduce through the real pipeline: the integration
criterion (Eq. 15) rejects every pair decomposition on the 4-cycle
(`NOT_INTEGRATED`), so no temporal variant is ever built — the exploration
measured raw mapped variants bypassing the criteria. A seeded prospecting
sweep (`experiments/grain_tau_experiments/prospect.py --seed 42 --n 3 --trials 60`)
found τ-wins are common in asymmetric substrates (23/120 runs). The
tutorial's specimen, verified end-to-end via
`pyphi.analyze(substrate, history, grains=SearchBounds(max_update_grain=2))`
under `presets.iit4_2023`:

- **fn_table `[2, 3, 4, 3, 0, 0, 5, 0]`** (state index → next state index,
  little-endian, A = bit 0; n = 3, deterministic), history
  `[(0, 0, 1), (0, 0, 0)]` (oldest first). The substrate condenses into
  **two temporal complexes and nothing else**: `{A}` at τ=2 with mapping
  `(0, 0, 1, 1)` (macro state = A at the window end — A sampled every
  second step), φₛ = 0.508305, `exclusion_margin` = 0.300787, excluding
  the full micro universe (φₛ = 0.2075); and `{B, C}` at τ=2 blackboxed
  through B, φₛ = 0.462996, margin = 0.199962.
- Secondary mention: fn_table `[3, 1, 1, 7, 1, 2, 2, 7]`, history
  `[(1, 0, 0), (1, 0, 0)]` — the whole-universe winner is a
  **mixed-grain** system (A, C at grain 1; B at grain 2), φₛ = 0.314099
  vs the all-micro universe's 0.276692.

Provenance: commit `experiments/grain_tau_experiments/README.md` and `prospect.py`
(seeded, saves raw per-candidate records, no-clobber filenames); the
results JSON stays untracked (496K, regenerable from the seed).

## Deliverables

### 1. `docs/howto/grain-search.md`

Executable MyST how-to (unpaired, like the other how-to pages), added to
the howto toctree after `sweep`. One task: run a grain search and read the
result. Sections:

- **Pre-flight the cost** — `SearchBounds.estimate(substrate)`: the
  headline counts, what `is_exact` / `truncated` / `partitions_capped`
  mean, skipping a sweep that is too big.
- **Run the search** — `analyze(substrate, state, grains=True)` as the
  primary spelling; `grains=SearchBounds(...)` for control;
  `pyphi.macro.complexes` as the underlying driver.
- **Supply a micro history for τ > 1** — the untaught requirement:
  `max_update_grain ** max_depth` universe states, oldest first; show the
  driver's error message for a bare state.
- **Read the result** — winners are `Complex` objects: `units` (with
  grains and mappings), φₛ, `exclusion_margin` / `effectively_tied`,
  `excluded` (including higher-φₛ shadows, cross-linking the
  recursive-exclusion tutorial); `records` (every evaluated system);
  `ties` (exclusion failures).
- **Parallelize** — `parallel_kwargs` pass-through, cross-linking the
  parallel how-to.
- **Bound the search** — what each `SearchBounds` knob does to the sweep,
  cross-linking the complexity page's grain section.

Demos run on the min-substrate (n = 2, seconds; numbers verified: default
sweep evaluates 8 systems, winner φₛ = 0.7883339770634884 at 1e-13).

### 2. `docs/theory/macro-units.md`

Prose theory page, toctree entry after `phi-structure`. The theory
section's frame is "how IIT 4.0's quantities map onto PyPhi's types";
this page covers, with equations verified against
`papers/2024__marshall-et-al__intrinsic-units.pdf`:

- **What a macro unit is** — a state mapping over a sequence-state window
  of micro constituents; coarse-graining vs blackboxing; spatial and
  temporal grains; grains compose down the hierarchy.
- **The intrinsic-unit criteria** — Eqs. 15–16 (integration of the
  candidate decomposition; no stronger competitor over the same
  footprint) → `pyphi.macro.judge_candidate`, `UnitVerdict`, `Reason`.
- **Exclusion across grains** — Eq. 19 applied recursively (macro and
  micro candidates compete in one cascade on micro footprints;
  `pyphi/condensation.py`); shadows and selection margins, cross-linking
  the recursive-exclusion tutorial and the tie-breaking how-to.
- **The type map** — `MacroUnit`, `MacroSystem`, `SearchBounds`,
  `ComplexesResult`, `SearchEstimate`, and where `analyze(grains=...)`
  sits.

No uncertified exploration claims (no SLEM cap, no ii-gate).

### 3. Tutorial additions: `docs/tutorials/macro.md`

Two new sections after "Rediscovering the paper's coarse-graining
example", jupytext-paired notebook re-synced:

- **Blackboxing** — the page currently builds only coarse-grains; show
  `blackbox()` (output-subset mappings), contrast with `coarse_grain()`
  on the same constituents, and note blackboxing is the family that
  extends to update grains above 1.
- **Temporal grains** — the rand:22 specimen: build the three-unit
  deterministic TPM from the function table, show the two-state history
  requirement (and the error a bare state raises), run
  `analyze(substrate, history, grains=SearchBounds(max_update_grain=2))`,
  and read the result: the substrate condenses into two temporal
  complexes; the τ=2 unit over `{A}` excludes the entire micro universe.
  Honest framing in prose: on symmetric substrates (the 4-cycle
  rotation) the integration criterion gates pair decompositions before
  any temporal variant is built, while asymmetric substrates make
  temporal units win outright; a seeded random search found such wins in
  roughly one run in five.

### 4. Complexity page: `docs/theory/computational-complexity.md`

New `## The cost of the grain search` section between "The cost of
IIT 4.0" and "Reducing the cost":

- **The candidate axes** — decompositions (set-partition growth, capped
  by `max_constituents`) × mappings (2^(2^(τ′·|V|)−1) − 1 surjective
  tables; the paper's Fig. 3E illustration: 32,727 mappings for 4 micro
  constituents at τ′ = 1 — verify against the PDF before citing) ×
  update grains (composing multiplicatively) × assemblies (Eq. 18
  disjoint unit sets). The criteria factor mappings and grains out of
  the judgment (one verdict per decomposition).
- **The two cost drivers** — macro-TPM construction Θ(τ·4ⁿ) per distinct
  (footprint, grain) key; the SIA partition sweep growing with macro
  unit count m (`DIRECTED_SET_PARTITION`: m = 3 → 22, 4 → 150,
  5 → 1,061, 6 → 7,896 — live-verifiable).
- **Measured anatomy** — the exploration's verified n = 4 numbers
  (default sweep: 80 systems, ≈ 0.85 s, ≈ 92% of time in partition
  sweeps at that size; construction's share grows exponentially with n).
  Re-verify order-of-magnitude on the current code before citing wall
  times.
- **The pre-flight** — `SearchBounds.estimate(substrate)` with a live
  executed cell; what the worst-case bound means (all-pass assumption).

## Testing / verification

- Every page builds under `-W` with cells executed:
  `env -u VIRTUAL_ENV uv run --all-extras --group docs sphinx-build -W
  --keep-going -b html docs docs/_build/html` (note: a cold build
  currently fails on the pre-existing `pyphi.mcp.content.topics`
  autosummary warning unrelated to this work — verify page-level success
  by the page's own rendering and absence of NEW warnings).
- Prose numbers match executed cell output; every citation
  (equation/figure numbers) verified against `papers/` — never from
  memory.
- Tutorial `.ipynb` output-free and jupytext-synced; Colab badge remains
  correct.
- Full `uv run pytest` (no path argument) gate at the end — the pages
  are docs-only, but the doctest sweep guards the touched tutorials.

## Out of scope

- Exploration levers and conjectures (ii-gate, SLEM cap, construction
  cache) — the construction-cache plan is a separate work item.
- API reference changes (autosummary already picks up the new symbols).
- The IIT 3.0 story for macro units (the drivers reject IIT 3.0).
