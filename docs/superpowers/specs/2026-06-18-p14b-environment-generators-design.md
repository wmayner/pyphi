# P14b — Environment generators for matching: Design

**Status:** approved
**Date:** 2026-06-18
**Wave:** 2 (pre-freeze, surface-affecting)
**Part of:** P14b tail (env-generation). The analytical-projection half and the perception-maximized
projection remain separate follow-ups; the `stationary_distribution` / Metropolis-Ising port is
explicitly **out of scope** (see §3).

---

## 1. Motivation

`pyphi.matching.MatchingAnalysis(perceptions, world_distribution)` requires the caller to supply
`world_distribution` — a `Mapping[tuple[int, ...], float]` over sensory-interface stimuli — by hand
(today's tests hard-code e.g. `{(0,): 0.75, (1,): 0.25}`). The matching manuscript, however, defines
its environments through **stimulus generators**: a "segment" generator activates a run of
contiguous sensory units at a uniformly random location with some probability, a "point" generator
activates a single unit, and a "noise" background activates each unit independently; an environment
(E1, E2, E1b, E3) is a composition of these over the sensory interface. These generators exist only
as ad-hoc notebook code in the matching repository, not as reusable library functions.

This item ports the paper's environment generators into PyPhi as library functions producing a
`world_distribution`, so `MatchingAnalysis` runs end-to-end on paper-faithful environments without a
hand-coded distribution. It also fixes a reproducibility defect in the reference sampler, which
draws from the global NumPy RNG.

**Why not `substrate_generator`.** `substrate_generator/` builds *substrates* — TPMs / dynamics. An
environment generator builds a *probability distribution over sensory-interface stimuli*, a
different kind of object with a different consumer (`MatchingAnalysis`), and a concept specific to
the matching/perception framework. It therefore lives in the `matching` package. (A future
`stationary_distribution`/Ising substrate-construction helper, if ported, would go in
`substrate_generator` — which is exactly why the two concerns are kept apart.)

**Why not derive the world distribution from a substrate stationary distribution.** The manuscript
never does this. The world distribution over stimuli comes from environment generators; a separate
*intrinsic uniform* prior `Pr(stimulus) = 1/|Ω_S|` is used for the connectedness/triggering
coefficient. The earlier roadmap phrasing ("port stationary_distribution + Metropolis Ising")
conflated substrate-construction tools with world-distribution generators; this design follows the
paper.

## 2. Goals

- A `pyphi/matching/environment.py` module of pure functions that build a `world_distribution`
  (`Mapping[tuple[int, ...], float]`) over an `n`-unit sensory interface.
- Primitive generators: `segment`, `point`, `noise`.
- General composition: `superpose` (independent activation combined by OR) and `mixture`
  (weighted choice), both operating on distributions so any generators compose arbitrarily.
- Seeded sampling utilities (`sample`, `world_sample`, `noise_sample`) using an isolated
  `np.random.default_rng(seed)` — never the global RNG.
- Reproduce the manuscript's environments (E1, E2, E1b, E3) to exact per-stimulus probabilities.

## 3. Non-goals

- The `stationary_distribution` / Metropolis-Ising port (substrate/dynamics construction — separate
  concern, deferred; would belong in `substrate_generator`).
- The analytical (φ-maximized) differentiation projection (separate P14b follow-up).
- The perception-maximized projection (open research).
- Temporal/sequential stimulus models beyond i.i.d. sampling from a distribution (the manuscript
  notes the environmental sequence "is generally non-stationary", but matching as implemented
  samples i.i.d. from the world distribution; sequence models stay out of scope).

## 4. Design

### 4.1 Core representation

A generator *is* a `world_distribution`: a `Mapping[tuple[int, ...], float]` whose keys are sensory
states (tuples of `0`/`1` of length `n`, the number of sensory-interface units) and whose values are
probabilities summing to 1. This is exactly `MatchingAnalysis.world_distribution`'s type, so any
generator output drops straight in. Distributions are computed **exactly over all `2^n` states**
(feasible for the paper's small interfaces); sampling is a separate, seeded layer (§4.4).

A small internal helper normalizes/validates a distribution (non-negative, sums to 1 within
`PRECISION`) and is applied at every generator's output.

### 4.2 Primitive generators

All return a distribution over the `n`-unit interface (states are length-`n` tuples; unit order is
the sensory-interface order the caller will align with `PerceptualSystem`'s `sensory_indices`).

- `segment(n, length, p) -> dict[tuple[int, ...], float]`: with probability `p`, activate `length`
  contiguous units at a position chosen uniformly among the `n - length + 1` valid start positions
  (each segment-present outcome has probability `p / (n - length + 1)`); with probability `1 - p`,
  the all-off state. Raises `ValueError` if `length > n` or `p` ∉ [0, 1].
- `point(n, p) -> ...`: `segment(n, 1, p)` — a single unit at a uniformly random location with
  probability `p`. Provided as a named generator for readability.
- `noise(n, p) -> ...`: each unit independently on with probability `p` — the product Bernoulli
  distribution (`Pr(state) = Π p^{s_i} (1-p)^{1-s_i}`). `p = 0.5` yields the uniform "structureless
  world" used as the matching noise baseline.

### 4.3 Composition

Both take distributions (primitive or already-composed) and return a distribution, so composition
nests arbitrarily.

- `superpose(*distributions) -> dict[...]`: independent activation combined by logical OR. Each
  input distribution is drawn independently; the resulting state is the elementwise OR (a unit is on
  iff any generator turns it on). Computed exactly: for every tuple of input states (one per
  distribution), accumulate the product of their probabilities onto the OR-combined state. This is
  the manuscript's environment construction — segments/points overlaid on a noise background — and
  is general for any generators. All inputs must share the same `n`.
- `mixture(distributions, weights=None) -> dict[...]`: a weighted convex combination (pick generator
  `i` with probability `weights[i]`, emit its state). `weights` defaults to uniform; must be
  non-negative and is normalized to sum to 1. All inputs must share the same `n`.

### 4.4 Sampling (seeded)

- `sample(distribution, size, *, seed) -> list[tuple[int, ...]]`: draw `size` i.i.d. states from a
  distribution using `rng = np.random.default_rng(seed)` and `rng.choice` over the support. Returns
  a list of state tuples (the form `MatchingAnalysis.matching` consumes as a sample sequence).
- `world_sample(world_distribution, size, *, seed)` and `noise_sample(n, size, *, seed)` convenience
  wrappers; `noise_sample` samples from `noise(n, 0.5)` (the uniform structureless world). The
  `seed` is required (keyword-only) so a sample is always reproducible; callers save the seed
  alongside results.

### 4.5 Paper environments (validation fixtures, not shipped API)

The manuscript's environments are expressed as compositions, used in tests:

- **E3** (pure noise): `noise(n, 0.5)` — uniform.
- **E1** (segment): `superpose(segment(n, 3, 0.6), segment(n, 2, 0.9), noise(n, 0.05))`.
- **E2** (centered odd): `superpose(segment(n, 3, 0.6), point(n, 0.9), noise(n, 0.05))`.
- **E1b** (2-segment only): `superpose(segment(n, 2, 0.9), noise(n, 0.05))` — apparent 3-segments
  arise at chance from overlapping 2-segments.

(These exact probabilities are the paper's; the tests assert them, the module ships only the
generic generators.)

## 5. Testing

`test/test_environment.py`:

- **Primitive correctness (hand-computed):** `segment(n, length, p)` puts `p/(n-length+1)` on each
  contiguous-run state and `1-p` on all-off (small `n`); `noise(n, p)` matches the product Bernoulli
  on every state; `noise(n, 0.5)` is uniform; `point` equals `segment(n, 1, p)`.
- **Composition:** `superpose` of two known generators reproduces a hand-computed OR-distribution on
  a tiny interface; `superpose` with the all-off distribution is the identity; `mixture` weights
  combine correctly; both preserve normalization.
- **Invariants (property-based, larger `n`):** every generator/composition output is non-negative,
  sums to 1 within `PRECISION`, and has keys of length `n`.
- **Paper environments:** E1/E2/E1b/E3 built via the compositions in §4.5 produce normalized
  distributions with the expected support; pinned per-stimulus probabilities for a small interface.
- **Seeded sampling:** `sample(dist, size, seed=k)` is deterministic given `k`, reproduces across
  calls, differs across seeds, and its empirical frequencies converge to the distribution; uses no
  global RNG (a global `np.random.seed` does not change its output).
- **End-to-end:** a `MatchingAnalysis` built with a generator-produced `world_distribution` runs
  `matching(...)` without a hand-coded distribution.

Verification runs `uv run pytest` **with no path argument** (public surface; doctest sweep).

## 6. Risks and mitigations

- **Exact `2^n` enumeration cost.** Bounded — the sensory interface in the paper is small (≤ ~8
  units); `superpose` over `k` distributions costs `Π |support_i|`, dominated by the noise
  background's `2^n`. Acceptable for the matching regime; documented. (Sampling, not enumeration, is
  the path for large interfaces — but matching itself is `2^n`-bound elsewhere.)
- **Reproducibility.** All sampling takes a required `seed` and an isolated `default_rng`; no global
  RNG use anywhere in the module.
- **State-order alignment.** Distribution keys are tuples over the sensory interface; the caller
  aligns them with `PerceptualSystem.sensory_indices`. Documented in the function docstrings.

## 7. Acceptance criteria

- `pyphi/matching/environment.py` ships `segment`, `point`, `noise`, `superpose`, `mixture`,
  `sample`, `world_sample`, `noise_sample`, exported from `pyphi.matching`.
- E1/E2/E1b/E3 reproduce the paper's per-stimulus probabilities (pinned tests).
- A `MatchingAnalysis` runs end-to-end on a generator-produced world distribution.
- Sampling is seeded and global-RNG-free (test asserts it).
- `uv run pytest` (no path argument) green, including doctests.
- ROADMAP dashboard updated (P14b env-generation ✅); changelog fragment present.
