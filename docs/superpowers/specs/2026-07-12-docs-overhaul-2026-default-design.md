# Documentation overhaul for the IIT 4.0 (2026) default

**Date:** 2026-07-12
**Status:** Design, approved for planning
**Scope:** User-facing documentation. Follows the default-formalism flip to IIT 4.0 (2026).

## Problem

The default formalism is now IIT 4.0 (2026), whose system φ is capped by the
intrinsic information `ii(s)` (Eq. 23). A confirmed property of this measure:
**deterministic systems have φ_s = 0** — a deterministic system has zero
intrinsic differentiation, so the cap drives φ_s to 0. This is correct theory,
not a bug (confirmed: the 2026 golden fixtures for `basic`/`xor` already store
φ = 0).

The consequence for the docs is large. Almost the entire classic IIT teaching
set is deterministic — `xor`, `basic`, `rule110`, `rule154`, `fig4`, `fig5a/b`,
`pqr`, `residue`, `macro` — so under the default they all compute **φ_s = 0**.
Every user-facing doc that introduces IIT on one of these examples now shows a
newcomer φ = 0, which reads as "broken." Measured under the 2026 default, the
only example systems with nonzero φ are the IIT 4.0 paper logistic networks
(`fig6c` 1.75, `fig7` 1.11, `fig6d` 1.05 — all 6-unit/slow) and `grid3` (0.025).

The decision (with the maintainer): **keep 2026 as the default** — it is the
current formalism — and **overhaul the teaching so it genuinely teaches the 2026
measure**, including turning the determinism⇒0 property from a hidden gotcha
into an explicit lesson.

## Keystone example

Promote the IIT 4.0 **Fig 1A logistic network** to `pyphi.examples` as a public
`iit4_fig1a_substrate()` / `iit4_fig1a_system()` (final names TBD in the plan).
It is:

- The example the IIT 4.0 (2023) paper *introduces the theory with* (Figs 1, 2,
  4), so it carries paper provenance and published φ values.
- 3 units, fast.
- **Probabilistic** (logistic activation, k=4), so the cap does not bind:
  verified φ is identical under 2023 and 2026 and matches the paper —
  `a = 0.0396`, `aB = 0.1719`, `aBC = 0.1339`; `aB` is a complex (the φ-max of
  the three), a built-in exclusion lesson.

The builder currently lives only in `test/integration/test_paper_reproduction.py`
(`_fig1_substrate`, `_FIG1_WEIGHTS`, k=4, state `(0,1,1)`). Move it into
`pyphi.examples`; the test imports it back (removing the duplication). Confirm
the promoted example reproduces the pinned paper values in the existing N1
tests.

## New teaching structure

This is a re-think, not a 1:1 migration of the existing pages. The teaching
spine, 2026-first:

1. **Getting started** — a first computation on Fig 1A: build the substrate,
   compute `φ_s`, see a nonzero value and the `aB` complex (the "aha"). No
   deterministic example in the newcomer's first result.
2. **One worked arc following the paper's Fig 1 → 2 → 4** — system integration
   (`φ_s`, the complex) → cause-effect structure (distinctions, Fig 2) →
   relations (Fig 4), all on Fig 1A, reproducing published numbers under the
   default. This **replaces the separate `cause-effect-structure` and
   `worked-example` tutorials** with one coherent narrative (exact split — one
   long page vs. a few linked pages — decided in the plan).
3. **The intrinsic-information measure (new conceptual centerpiece)** — what the
   2026 cap is (`φ_s = min{φ_c, φ_e, ii(s)}`), and **why deterministic systems
   have φ_s = 0**, using `xor`/`basic` as the illustration. The classic
   deterministic examples live *here*, repurposed from headline example to the
   specimen that teaches this property. Not every classic needs to appear — one
   or two clean illustrations suffice.
4. **Formalism versions** — 2026 (default, capped) vs 2023 (uncapped) vs 3.0:
   what each computes and when to choose it, with the determinism contrast made
   concrete (`xor`: φ_s = 0 under 2026, 1.5 under 2023). Expands the existing
   `theory/formalism-versions.md`.
5. **Supporting material** — macro, actual causation, and the how-tos, migrated
   under the per-doc principle below (not preserved 1:1; pages that no longer
   earn their place can be merged or dropped).

## Per-doc migration principle

Every user-facing doc that computes under the default and shows a φ value must
either (a) use Fig 1A or another probabilistic example, or (b) explicitly pin
the formalism it means to demonstrate (`with config.override(**presets.iit4_2023)`
etc.) with a one-line note. **Invariant: no doc silently shows a φ value that is
wrong under the shipping default.** Executable MyST pages regenerate at build
time, so the risk is committed notebook outputs and hardcoded φ numbers in prose.

## Affected docs (inventory to finalize in the plan)

Computing under the default (candidates for rework):

- `docs/getting-started/first-computation.{md,ipynb}`
- `docs/tutorials/cause-effect-structure.{md,ipynb}`, `docs/tutorials/worked-example.{md,ipynb}`
- `docs/tutorials/actual-causation.{md,ipynb}` (AC is unaffected by the cap — verify, likely only a note)
- `docs/examples/IIT_4.0_demo.ipynb` (committed outputs)
- `docs/theory/{overview,phi-structure,system-integration,distinctions-and-relations,macro-units}.md`
- `docs/howto/{landscape,save-load,export,tie-breaking,parallel,cache}.md`
- `docs/whats-new-in-2.0.md`

Already safe (pin a formalism): `docs/theory/formalism-versions.md` (2023),
`docs/theory/iit-3.0.md` (3.0), `docs/tutorials/macro.{md,ipynb}` (2023),
`docs/howto/grain-search.md` (2023). The plan re-verifies each.

The plan will enumerate, per doc: current example, whether the default changes
its numbers, and the target (new example / explicit pin / determinism lesson /
merge / drop).

## Non-goals

- The analytical-relations default (`c511e8bc` — separate tracked follow-up).
- The ii-gate build.
- `docs/superpowers/` specs and plans (historical records, not user docs).
- Re-deriving or changing any φ value or formalism behavior. This is a
  documentation and one-example-promotion change; no `pyphi/` computation logic
  changes except adding the Fig 1A example function.

## Verification

- The promoted `iit4_fig1a_*` example reproduces the pinned N1 paper values
  (existing tests, re-pointed at the public example).
- `uv run pytest` green (the example promotion is the only code change; docs are
  build-time executed, not in the pytest path).
- A docs build (`just docs`) succeeds with the executable pages regenerating
  under the 2026 default; spot-check that getting-started and the worked arc show
  nonzero Fig 1A numbers and that the determinism lesson shows φ = 0 for `xor`.
- No committed doc shows a stale default-computed φ (grep pass for hardcoded
  φ numbers in the reworked prose).
