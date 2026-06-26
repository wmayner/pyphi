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
