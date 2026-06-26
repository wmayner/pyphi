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

## Finding 7 — the hot-path config flags behave as documented; both pre-2.0 config bugs are fixed

The Part 4 config-behavior sweep (`config_sweep.py`) audits the configuration
flags the IIT 4.0 hot paths read, asserting that documented behavior equals
observed behavior. Every audited flag matches its documentation, and the two
config bugs that existed before the 2.0 refactor are both closed.

**The global parallel switch now gates the per-level flags (pre-2.0 bug
fixed).** Before the refactor, `PARALLEL=False` did not reliably disable
parallelism: several call sites passed a truthy per-level config dict as the
`parallel` keyword, bypassing the old `MapReduce` class's `if self.parallel`
guard, so subprocesses spawned even with the global flag off. In 2.0 the
per-site kwargs are built by `conf.parallel_kwargs()`, which forces
`parallel=False` whenever `config.infrastructure.parallel` is off. Spying on
the two branches of `pyphi.parallel.map_reduce` (the in-process
`_map_sequential` versus the `default_scheduler` that owns subprocess dispatch)
across a `macro` SIA confirms the gate: the scheduler is entered **only** when
both the global flag and the per-level `parallel_partition_evaluation` flag are
True. With the global flag off, the in-process sequential branch runs even when
the per-level dict requests parallel.

| global `parallel` | per-level `parallel` | subprocess scheduler entered |
|---|---|---|
| False | False | no |
| False | True | **no** (global gate forces sequential) |
| True | True | yes |
| True | False | no |

**Incompatible config combinations raise a clean error, not a raw crash
(pre-2.0 bug fixed).** Before the refactor, pairing `IIT_3_0` with
`GENERALIZED_INTRINSIC_DIFFERENCE` raised a raw `AttributeError` deep in the
compute path. The sweep runs every (version × measure × system-scheme)
combination on `basic_system`: of the 18 combinations, 9 compute a valid φ and
9 are cleanly rejected at override time with a `ConfigurationError` that names
the conflicting fields and a fix; none raise a raw exception. The eager check
(`validate_config`, on by default) is what converts a deep compute-time failure
into a config-time error: with `validate_config=False` the `IIT_3_0` + GID
combination is accepted at override time but raises a *typed*
`MeasureNotCompatibleError` at compute time (itself an improvement over the
pre-2.0 raw `AttributeError`); with the default eager check it is rejected
before any computation runs.

**`shortcircuit_sia` is a φ-preserving optimization.** The flag toggles an
early null-SIA return for systems with no specified cause or effect. φ is
identical with the flag on and off on every strongly-connected standard example
(`basic`, `xor`, `grid3`, `macro`). A pure-noise 2-node substrate (every node
outputs 0.5) exercises the live short-circuit path: it returns reasons
`NO_CAUSE`/`NO_EFFECT`, and φ is the same (0.0) with the flag on or off.

**The cache flags never change the result.** `cache_repertoires` and
`cache_potential_purviews` are performance policy only: `basic_system` φ is
identical (0.415037) with both caches on and both off.

**No `pyphi/` change was warranted.** The audit found no documented-versus-actual
mismatch, so no source fix and no golden revalidation were needed.

*(Reproducible via `uv run python -m benchmarks.iit_3_vs_4.config_sweep`.)*

## Finding 8 — 6-7 node sizing: the SIA cost is bimodal (reducible short-circuits vs. full search), and n=7 is batch-only

Part 3 extends the harness to seeded 6-7 node networks
(`harness._synth_system`): an Ising substrate at the all-off state with an
Erdős–Rényi connectivity mask (a `sparse` and a `dense` density per size), edge
weights drawn from a fixed-seed `default_rng`, and the TPM built by the Ising
unit function at `temperature=0.25`. The four networks (`synth_n6_sparse`,
`synth_n6_dense`, `synth_n7_sparse`, `synth_n7_dense`) and their generative
inputs are committed under `results/synth_fixtures/`.

**The SIA cost is bimodal, and the generator must target the expensive mode.**
With mean-zero coupling weights the all-off state is almost always *reducible*:
the partition search finds a zero-φ partition early and `map_reduce`'s
`is_falsy` short-circuit stops it, so the SIA returns in well under a second and
never touches most of the partition set. Out of 20 mean-zero seeds at n=6, none
produced an integrated system. This makes mean-zero networks useless for
characterizing the full-search cost. A positive (ferromagnetic) coupling mean
fixes it: at mean 1.0, 11 of 12 n=6 dense seeds integrate and run the full
`DIRECTED_SET_PARTITION` search (~11 s bare, ~24 s under the harness's
`cProfile` wrapper); the generator therefore draws weights with mean 1.0. The φ
values are small (n=6 sparse φ = 5.96e-4, n=6 dense φ = 2.9e-6) but the
*computation* is the full search, which is the quantity the matrix sizes.

**Partition count grows ~7.8× per node** under the default
`DIRECTED_SET_PARTITION` scheme:

| n | DIRECTED_SET_PARTITION partitions | ratio to n−1 |
|---|---|---|
| 3 | 22 | — |
| 4 | 150 | 6.8× |
| 5 | 1061 | 7.1× |
| 6 | 7896 | 7.4× |
| 7 | 61888 | 7.8× |

**Measured n=6 sizing (3 trials each, harness `cProfile` mode):** sparse
median 25.8 s (25.3 / 27.1 / 25.8), dense median 23.1 s (26.4 / 22.8 / 23.1); φ
identical across trials (the construction is deterministic in the seed). Bare
(un-profiled) cost is roughly half that, ~11 s.

**n=7 is batch-only — it belongs on the server (Wave B).** Scaling the n=6
profiled cost by the 7.8× partition growth alone projects ~3 min/trial at n=7,
and the true factor is larger because per-partition repertoire algebra also
grows with n; a single n=7 trial is realistically 4-8 min. The full Wave B 4.0
matrix (sparse + dense × {2023, 2026} × 3 trials at n=7, plus n=6) is on the
order of an hour of compute — past the interactive budget, which is why the n=7
runs are deferred to the lab server. n=7 was deliberately **not** run locally.

*(Fixtures: `results/synth_fixtures/{name}.json` — seed, weights, CM, TPM;
regenerate with `uv run python -m benchmarks.iit_3_vs_4.synth_fixtures`. n=6
raw: `results/post/synth_n6_*_iit4_sia_2023_seed0_trial*.json`.)*
