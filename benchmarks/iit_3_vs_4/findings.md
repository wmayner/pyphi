# P17 — performance characterization findings

Companion to `README.md` (which holds Findings 1–4 from the initial
cross-temporal run). This file holds the P17 deep-dive results: the honest 2026
cap cost, the mechanism attribution for the IIT 4.0 speedup, the extended-size
thresholds, and the config-behavior sweep.

All numbers are medians over the per-trial raw JSON in `results/`. Runs are
sequential and in-process (`parallel=False`), single machine (Apple Silicon),
unless noted.

## Finding 5 — the 2026 cap is computationally free; the "cheap 2026" readings were the short-circuit

The harness README's Finding 4 noted that `iit4_sia_2026` returned φ=0 on every
standard network and warned its 8–156 ms wall time was "fixed overhead, not the
algorithm's true cost on non-trivial outputs." That gap is now closed using
`logistic3_k8` — the B4 cap-biting network (3-node fully-connected logistic
substrate, k=8, weights 0.3, state (0,0,0)), the one network where the 2026
ii(s) cap binds at a non-trivial intermediate value rather than collapsing to 0.

Medians over 5 trials:

| measurement | φ | wall (median) | `evaluate_partition` | `intrinsic_information` |
| --- | --- | --- | --- | --- |
| `iit3_sia` | 0.7259 | ~0.66 s | — | — |
| `iit4_sia_2023` | 0.03662 | **0.052 s** | 39.9 ms | 4.87 ms |
| `iit4_sia_2026` | **0.00323** | **0.052 s** | 40.2 ms | 4.86 ms |

Two results:

1. **The 2026 variant produces a real non-zero φ here** (0.0032 < 0.0366 = the
   2023 φ): the cap binds, so this is a genuine measurement of the full 2026
   path, not a short-circuit.
2. **The 2026 path costs the same as 2023** — identical at every phase
   (`sia` 48.5 vs 48.8 ms, `evaluate_partition` 39.9 vs 40.2 ms,
   `intrinsic_information` 4.87 vs 4.86 ms). The Eq-23 cap is a post-hoc
   `min{φ_c, φ_e, ii(s)}` applied to terms the 2023 partition search already
   computes; it adds no measurable search cost.

So the earlier "2026 is suspiciously cheap (8–156 ms)" readings were entirely
the short-circuit-to-zero on networks where the cap collapses φ to 0 — not a
cheaper algorithm. On a network where the 2026 result is non-trivial, the cap
variant is no cheaper and no more expensive than 2023. The honest cost of the
2026 formalism is "2023 plus a free min."

*(Raw: `results/post/logistic3_k8_iit4_sia_{2023,2026}_seed0_trial*.json`.)*

## Finding 6 — the cross-temporal "IIT 4.0 speedup" was a scope artifact; the real, de-confounded speedup is ~18–20× per SIA partition and ~2× for the CES

The harness README's Findings 1 and 3 report a "2.5×–43× IIT 4.0 speedup,
scaling with size." That comparison is **invalid as stated**: it maps
pre-refactor `new_big_phi.phi_structure` against post-refactor `formalism.sia`,
which are **different computations**. `phi_structure` computes the whole
phi-structure (SIA + all distinctions + relations); `sia` computes only system
irreducibility (Φ_s). The profiles make this stark:

| network | pre `phi_structure` | of which CES | of which SIA | post `sia` (CES frames) |
| --- | --- | --- | --- | --- |
| macro (4n) | 1.38 s | 1.35 s (15 distinctions) | 0.026 s | 0.09 s (0) |
| rule154 (5n) | 81 s | 80 s (31 distinctions, 600k mech-MIP evals) | 1.05 s | 2.55 s (0) |

So ~97% of the rule154 "speedup" is simply that `sia` does not compute the CES
that `phi_structure` does. It is not an algorithmic improvement to the same
computation.

**De-confounded controls** (`controls.py`; pre numbers read from the
`phi_structure` profiles' `sia`/`ces` frames, post measured on this checkout):

| control | macro | rule154 |
| --- | --- | --- |
| **CES-only** (post `System.ces()` vs pre `ces`) | 1.35 → 0.69 s (**1.9×**) | 80.1 → 38.9 s (**2.1×**) |
| **SIA per-partition**, matched `DIRECTED_BIPARTITION` scheme | 26.0 → 1.28 ms (**~20×**) | 34.9 → 1.89 ms (**~18×**) |

Two real, like-for-like speedups: the **SIA inner loop is ~18–20× faster per
partition**, and the **CES is ~2× faster**. The per-partition figure is the
robust one (on rule154 the matched scheme evaluates the same 30 partitions in
both generations; on macro the `DIRECTED_BI`/`DIRECTED_BIPARTITION` enumerations
differ slightly — 1 vs 14 partitions — so the per-partition rate is the fair
comparison there).

**Why the wall-time picture looked the way it did, and why it inverts.** Under
the default 2.0 preset the system scheme is the paper-faithful
`DIRECTED_SET_PARTITION`, which evaluates far more partitions than the pre era's
`DIRECTED_BI` (rule154: 1061 vs 30). Each partition is ~18× cheaper, but there
are ~35× more of them, so a default-config 2.0 user's *SIA wall time* on
rule154 is ~2.5× **higher** than pre (2.55 s vs 1.05 s) — the opposite of a
speedup — even though the per-operation cost fell sharply. The headline "43×"
came entirely from the `phi_structure`-vs-`sia` scope mismatch.

**Larissa's puzzle, corrected.** The per-operation IIT 4.0 speedup from the 2.0
refactor is real and large (~18× on the SIA partition evaluation, ~2× on the
CES), driven primarily by the repertoire-algebra kernel rewrite
(`core/repertoire_algebra.py` replacing the old `subsystem.py`/`repertoire.py`
methods — the post SIA's cost is almost entirely repertoire algebra). It is
masked at the wall-time level by (a) the original harness comparing different
scopes and (b) the paper-faithful partition scheme evaluating many more
partitions. The config-layering and parallel-engine changes are not the driver:
the comparison is sequential throughout (so parallelization contributes nothing
to this gap), and the ~18× per-partition gain is far too large to be config
attribute-access overhead.

*(Raw: `results/{pre,post}/{macro,rule154}_iit4_*_seed0_trial*.{json,pstats}`;
controls reproducible via `uv run python -m benchmarks.iit_3_vs_4.controls`.)*
