# Temporal-grain prospecting

Question: is there a small substrate where a unit at update grain τ=2
passes the intrinsic-unit criteria AND wins the bounded grain search
end-to-end (i.e. appears in a `pyphi.macro.complexes` winner)?

Motivation: the grain-discovery exploration's promised demo (deterministic
4-cycle, τ=2 at φₛ=2.0) does NOT reproduce through the real pipeline — the
integration criterion (Eq. 15) rejects every pair decomposition on the
4-cycle (`NOT_INTEGRATED`), so no temporal variant is ever built; the
exploration measured raw mapped variants bypassing the criteria. Symmetric
permutation substrates (swaps, rotations, independent NOT-oscillators)
either fail the criteria, lose to the micro incumbent, or tie everything
into exclusion failure.

Answer: **yes, and τ-wins are common in asymmetric substrates.**
`prospect.py --seed 42 --n 3 --trials 60 --eps 0.0 0.1` → 23/120 runs have
a winner containing a grain-2 unit (`prospect_n3_seed42_trials60_mc2_sweep1.json`;
raw per-candidate records included). Two deterministic specimens verified
end-to-end via `pyphi.analyze(substrate, history, grains=SearchBounds(max_update_grain=2))`
under `presets.iit4_2023`:

- `rand:22:eps=0.0` — fn_table `[2, 3, 4, 3, 0, 0, 5, 0]` (state index →
  next state index, little-endian bit order A=bit0), history
  `[(0,0,1), (0,0,0)]`. The substrate condenses into TWO temporal
  complexes and nothing else: `{A}` at τ=2, mapping `(0,0,1,1)` (macro
  state = A at the window end, i.e. A sampled every second step),
  φₛ=0.508305, margin 0.300787 — it excludes the full micro universe
  (φₛ=0.2075); and `{B,C}` at τ=2 blackboxed through B, φₛ=0.462996,
  margin 0.199962.
- `rand:13:eps=0.0` — fn_table `[3, 1, 1, 7, 1, 2, 2, 7]`, history
  `[(1,0,0), (1,0,0)]`. The whole-universe winner is a MIXED-GRAIN
  system: A and C at grain 1, B at grain 2, φₛ=0.314099 vs the all-micro
  universe's 0.276692; margin 0.037407.

Used by the grain-search documentation (tutorial temporal-grain section).
