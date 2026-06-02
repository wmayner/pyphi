# k-ary Repertoire Reference Oracle — Design

Date: 2026-06-02

## Problem

The k-ary (multi-valued) repertoire path carried a serious latent bug: system
partitions were silent no-ops on k-ary dimensions, under-reporting Φ to zero
(fixed in `7a8efe62`). The existing multivalued golden fixtures were both too
few and — for the two that existed — locked to the *buggy* values, so they did
not catch it. Nothing in the suite cross-checks repertoire *values* against an
independent computation; the current k-ary property test asserts only shape and
normalization.

During that fix an independent brute-force repertoire reference was written and
validated (it matched pyphi to ~2e-16 across ~1,186 cause/effect ×
mechanism × purview × cut comparisons, and crashed the buggy code). That
reference is the natural value-level oracle for the whole bug class — but it
lives uncommitted in a scratch file.

This sub-project promotes that reference to a first-class, reusable test
fixture and builds a value oracle around it. (A separate, later sub-project
would add an independent SIA/Φ oracle on top of this reference.)

## Goal

Catch the class of k-ary repertoire bug found in `7a8efe62` — and analogous
latent bugs across topologies, alphabets, and cuts — by asserting pyphi's
cause/effect repertoires match an independent from-scratch computation.

## Non-goals

- The SIA/Φ value oracle (a separate sub-project; it will reuse this
  reference).
- The 2026 II-cap; supporting reduced-dimension factors.
- IIT 3.0 multi-valued (unsupported: its EMD measure raises on k>2).

## Design

### Component A — The reference module

`test/reference/repertoire.py` (new package `test/reference/`): a documented,
standalone NumPy implementation independent of pyphi's `Node` /
`repertoire_algebra`. Two functions, each taking raw per-node forward/backward
factors, the per-node alphabet sizes, a (possibly cut) connectivity matrix, the
mechanism, the mechanism state, the purview, and `n`:

- `ref_effect(eff_factors, alph, cut_cm, mechanism, mstate, purview, n)`: for
  each purview node `z`, let `inputs_z = {j : cut_cm[j, z] == 1}` and
  `cond = mechanism ∩ inputs_z`. Build a weight over substrate previous-states
  that is a delta at `mstate[j]` on each conditioned axis `j ∈ cond` and uniform
  (`1/alph[j]`) on every other axis; contract it against `eff_factors[z]` to get
  the per-`z` distribution; place each at its canonical axis. The joint effect
  repertoire is the product (already normalized).
- `ref_cause(cau_factors, alph, cut_cm, mechanism, mstate, purview, n)`: for
  each mechanism node `m`, let `cut_inputs_m = {j : cut_cm[j, m] == 1}`. Take
  `cau_factors[m][..., mstate[m]]`; average (mean) over every axis **except**
  those in `purview ∩ cut_inputs_m`, keeping the kept axes at full alphabet. The
  joint is the product over mechanism nodes, normalized to sum 1.

The cut is applied **inside** the reference via `cut_cm` — for cause this means
averaging over severed inputs even when they are in the purview (the subtlety
that, if omitted, makes the reference silently ignore the cut: it caused a false
mismatch during the fix and must be encoded and commented).

`cau_factors` / `eff_factors` are obtained from a pyphi `System`'s
`cause_tpm.factor(i)` / `effect_tpm.factor(i)` on the **uncut** substrate; the
cut enters only through `cut_cm`. The reference deliberately does not reuse any
pyphi repertoire/Node code so it is a genuine cross-check.

### Component B — The value oracle test

`test/test_repertoire_reference.py`, two layers:

1. **Deterministic enumerated sweep (decisive anchor).** Over a matrix of
   seeded random substrates — `n ∈ {2, 3}`; alphabets drawn from `{2, 3, 4}`
   (including heterogeneous mixes); connectivity ∈ {dense (all-ones), sparse
   chain `i→i+1`, directed cycle} — assert, for **every** non-empty
   mechanism × purview subset, **every** cut (the base cm, each single-edge
   severance, and one multi-edge severance), and **both** directions, that
   `System(substrate=Substrate(marginals, state_space, cm=cut_cm)).repertoire(
   direction, mechanism, purview)` matches the reference to `atol=1e-12`. The
   cut is realized by constructing a `Substrate` whose `cm` is the cut matrix
   under the default `NullCut` (equivalent to a partition's `apply_cut`, since
   per-node TPMs depend only on the resulting `cm`).

2. **Hypothesis property test.** Upgrade
   `test/test_repertoire_kary_properties.py`'s existing test (currently shape +
   normalization) to *also* assert value-equality against the reference on
   random substrates and a random direction, keeping the existing shape and
   normalization assertions.

### Component C — The `apply_cut` bridge test

One test in `test/test_repertoire_reference.py`: build a `System` over the
uncut substrate, apply an actual `DirectedBipartition` that severs a chosen
edge, and assert its repertoire equals the repertoire from the cut-cm substrate
used by the sweep. This closes the one gap the sweep does not exercise — that
`partition.apply_cut` produces the expected cm.

### Invariants (cheap bonus)

Inside the sweep, additionally assert each pyphi repertoire sums to 1
(`atol=1e-12`), is non-negative, and has the canonical `repertoire_shape`.

## Testing / verification

This sub-project adds **only tests and a test-reference module — no library
source changes** — so all existing goldens are trivially unaffected. Verify:

- The new sweep + Hypothesis + bridge tests pass.
- `uv run pytest` with no path argument (full suite incl. doctests) stays
  green.
- The reference genuinely fails against known-buggy behavior: confirm by a
  one-off local check that reverting the `7a8efe62` node fix makes the sweep
  fail/raise (documented in the plan, not committed).

## Risks

- **Reference not truly independent.** If it mirrors pyphi's algebra it proves
  nothing. Mitigation: it works only from raw factors + `cut_cm` with plain
  NumPy, never importing repertoire/Node helpers.
- **Cause cut-handling subtlety.** `ref_cause` must average over severed inputs
  even in the purview (encode + comment); omitting it silently ignores the cut.
- **Sweep runtime.** Enumerating all subsets × cuts × directions over the
  matrix must stay fast. Keep `n ≤ 3` and the substrate matrix small; the
  validated `/tmp` run completed in well under a second per fixture.

## Success criteria

- An independent repertoire reference is committed under `test/reference/`.
- pyphi cause/effect repertoires match it to `atol=1e-12` across the full
  enumerated sweep and the Hypothesis property test.
- The `apply_cut` bridge test passes; repertoire invariants hold across the
  sweep.
- Full suite incl. doctests green; no golden changes (no source edits).
