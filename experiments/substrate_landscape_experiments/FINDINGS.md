# Substrate parameter landscapes: n=5 roughness gate and the n=3 experiment record

**Verdict (2026-07-07): the black-box optimizer driver is unblocked. At n=5,
signed normalized φₛ stays empirically smooth across MIP switches — the primary
sweep crosses 31 distinct selection regimes with zero discontinuous jumps in
the objective — so finite-difference gradient ascent is viable and the driver
ships population-first. The one caveat: the smoothness is of the objective
value, not of the selection identity, which changes roughly thirty times across
the sweep; and a single ascent chain is start-dependent, so a population-based
driver (multiple starts) is required rather than one local ascent.**

This is the roughness replication the exploration synthesis (Wave 2, then
ROADMAP "Wave 7 — Exploration builds") gated the optimizer on: the six n=3
experiments below established that every IIT 4.0 quantity is piecewise-analytic
over selection regimes and that `signed_normalized_phi` is the right
optimization objective, but tested only n=3. exp5 extends the two load-bearing
experiments (E1 sweep roughness, E6/E7 ascent viability) to n=5.

## Method (exp5)

`exp5_n5_roughness.py`, seed 20260707, temperature 0.25, ring builder: a
5-node extension of the Fig-1A Ising-sigmoid construction — reciprocal +0.7
couplings tiled around a ring, −0.2 self-connections — whose all-OFF state has
φₛ = 1.404 (positive, unclamped), giving the sweep a positive-φ region to
cross as the n=3 E1 sweep had around Fig 1A.

- **Primary sweep**: weight `w[0,1]` (A→B) over 120 grid points in [0.02, 1.40],
  recording per-point selection identities (MIP partition, specified
  cause/effect states).
- **Secondary sweep**: weight `w[2,1]` (C→B) over 40 points in [−1.0, 0.4] as a
  cross-axis check.
- **Ascent**: finite-difference gradient ascent on signed normalized φₛ
  (the exp4 procedure) from two seeded starts — the ring base and a seeded
  random start — each capped at a 200 SIA-evaluation budget.
- **Baseline**: seeded random search over the same total evaluation budget
  (320 samples).

Reproduce: `uv run python exp5_n5_roughness.py` (add `--smoke` for the
10-point validation run). Raw output:
`exp5_n5_roughness_raw_seed20260707_p120_s40_b200_ring.json` (full run, 483 s);
`*_smoke.json` (two smoke runs). Console log: `exp5_full_run.log`.

## Results (exp5)

- **Objective smooth across selection switches.** Primary sweep: 31 selection
  regimes (segments), **0 objective jumps**, dead fraction 0.0. Secondary
  sweep: 17 regimes, **0 objective jumps**, dead fraction 0.35 (a clamped
  zero-φ region, as expected off the positive-φ axis). Signed normalized φₛ is
  continuous across every MIP switch — the property the n=3 landscape
  exploration found and this run confirms at n=5.
- **Gradient ascent works from a good start.** FD ascent from the ring base
  climbed monotonically 0.117 → 0.161 in 157 evaluations, exceeding the
  random-search best of 0.087.
- **But it is start-dependent.** FD ascent from the seeded random start climbed
  only −0.0218 → −0.0021 in 163 evaluations, stranded in a near-zero basin.
  Random search over 320 samples had best 0.087, mean −0.033.
- **Reading.** The objective is smooth enough for gradient information to be
  usable (no discontinuities to trip a local method), but the landscape has
  multiple basins and a single ascent chain depends heavily on its start. The
  driver therefore ships population-first — multiple seeded starts / a
  population method — not a single local ascent.
- **Caveat for the driver.** Selection identity is *not* stable: the primary
  sweep's 31 regimes over 120 points means the MIP partition and specified
  states flip at roughly single-grid-point frequency even where the objective
  is smooth. A driver must not depend on a stable selection identity between
  nearby points.

## The n=3 experiments (record)

The six n=3 experiments this replication builds on. Full detail is in the
exploration spec `docs/superpowers/specs/2026-07-07-substrate-parameter-landscapes.md`
§9; the raw JSON for each ships beside its script.

- **exp1** (`exp1_sweep.py` → `exp1_sweep_raw.json`): fine single-weight sweep
  recording every discrete selection identity, establishing the
  piecewise-analytic structure and locating kinks at selection boundaries.
- **exp2** (`exp2_derivatives.py` → `exp2_derivatives_raw.json`): central-
  difference derivative of φₛ within a single selection regime across step
  sizes h — the derivative is well-defined inside a regime.
- **exp3** (`exp3_ces_sweep.py` → `exp3_ces_sweep_raw.json`): per-distinction
  φ_d and structural identity across the A→B weight.
- **exp4** (`exp4_saturation_ascent.py` → `exp4_saturation_ascent_raw.json`):
  the E7 finite-difference ascent-vs-random-search procedure at n=3, over all
  9 weights, that exp5 extends to n=5.
