# IIT 3.0 vs 4.0 cross-temporal performance harness

A standalone harness that measures, at **two points in the codebase history**,
where IIT 3.0 and IIT 4.0 spend their time on the same networks. It is separate
from the ASV suite in `benchmarks/benchmarks/` so it can iterate independently.

The harness runs the same script against two pyphi generations — a pre-refactor
anchor and the current 2.0 layout — and writes per-trial JSON and cProfile
output to `results/{generation}/`. A single `analyze.py` reads both result sets.

## Anchor commits

- **pre** — `b3aaa3e5`. Old layout: `pyphi.compute.sia`,
  `pyphi.new_big_phi.phi_structure`, flat `IIT_VERSION` config, `Subsystem`
  objects, no `formalism/` package.
- **post** — current HEAD on the `2.0` branch. New layout:
  `pyphi.formalism.iit3`/`iit4`, layered `config.formalism.iit.version`,
  `System` objects, formalism presets in `pyphi.conf.presets`.

The script auto-detects which generation it is running against by trying
imports and dispatches to the right entry points.

## Entry points

| label                | pre-refactor                                            | post-refactor                                                       |
| -------------------- | ------------------------------------------------------- | ------------------------------------------------------------------- |
| `iit3_sia`           | `compute.sia(subsystem)` + `IIT_VERSION=3.0`            | `formalism.sia(system)` + `pyphi.conf.presets.iit3`                 |
| `iit4_phi_structure` | `new_big_phi.phi_structure(subsystem)` + `IIT_VERSION=4.0` | (n/a)                                                            |
| `iit4_sia_2023`      | (n/a)                                                   | `formalism.sia(system)` + `pyphi.conf.presets.iit4_2023`           |
| `iit4_sia_2026`      | (n/a)                                                   | `formalism.sia(system)` + `pyphi.conf.presets.iit4_2026`           |

Pre-refactor IIT 4.0 has only the whole-`phi_structure` entry point (SIA + all
distinctions + relations); post-refactor `sia` computes system irreducibility
(Φ_s) alone. **The two are different computations** — a raw pre-vs-post
wall-time ratio between `phi_structure` and `sia` is not a like-for-like
speedup. The de-confounded comparison is in "Results" below.

## Results

All numbers are medians over the per-trial raw JSON in `results/`. Runs are
sequential and in-process (`parallel=False`) on a single machine (Apple
Silicon) unless noted.

### Per-operation speedup from the 2.0 refactor (de-confounded)

`controls.py` isolates the refactor from the entry-point scope and the
system-partition scheme by timing matched operations: the post-side CES against
the pre-side CES, and the post-side SIA under the pre era's
`DIRECTED_BIPARTITION` scheme (so both evaluate the same partitions). Pre
numbers are read from the `sia`/`ces` frames inside the pre `phi_structure`
profiles.

| control | macro (4n) | rule154 (5n) |
| --- | --- | --- |
| CES-only (post `System.ces()` vs pre `ces`) | 1.35 → 0.69 s (**1.9×**) | 80.1 → 38.9 s (**2.1×**) |
| SIA per-partition, matched `DIRECTED_BIPARTITION` | 26.0 → 1.28 ms (**~20×**) | 34.9 → 1.89 ms (**~18×**) |

The SIA inner loop is **~18–20× faster per partition** and the CES is **~2×
faster**, driven by the `core/repertoire_algebra.py` kernel rewrite. The
per-partition figure is the robust one: on rule154 the matched scheme evaluates
the same 30 partitions in both generations.

This per-operation gain does **not** always show up as lower wall time. The
default 2.0 system scheme is the paper-faithful `DIRECTED_SET_PARTITION`, which
evaluates far more partitions than the pre era's `DIRECTED_BI` (rule154: 1061 vs
30). Each partition is ~18× cheaper but there are ~35× more of them, so a
default-config SIA wall time on rule154 is ~2.5× *higher* post-refactor (2.55 s
vs 1.05 s). IIT 4.0 φ also stops matching pre vs post past 3 nodes, because the
two schemes select different system MIPs (rule154: 1.0 → 2.0).

Reproduce: `uv run python -m benchmarks.iit_3_vs_4.controls`.

### The IIT 4.0 (2026) Eq. 23 cap is computationally free

On the standard `pyphi.examples` networks the 2026 variant returns φ=0 by
short-circuit, so its wall time there reflects fixed overhead, not the cap's
cost. `logistic3_k8` (3-node fully-connected logistic substrate, k=8, weights
0.3, state (0,0,0)) is the network where the cap binds at a non-trivial value.
Medians over 5 trials:

| measurement | φ | wall (median) |
| --- | --- | --- |
| `iit4_sia_2023` | 0.03662 | 0.052 s |
| `iit4_sia_2026` | 0.00323 | 0.052 s |

The 2026 path costs the same as 2023 at every phase. The Eq. 23 cap is a
post-hoc `min{φ_c, φ_e, ii(s)}` over terms the 2023 partition search already
computes; it adds no measurable search cost.

### Hot-path config flags behave as documented

`config_sweep.py` audits the configuration flags the IIT 4.0 hot paths read.
Every flag matches its documentation, and two config bugs that existed before
the 2.0 refactor are confirmed closed:

- **The global parallel switch gates the per-level flags.** Pre-refactor,
  `PARALLEL=False` did not reliably disable parallelism (several call sites
  passed a truthy per-level dict as the `parallel` keyword, bypassing the old
  `MapReduce` bool check). In 2.0 `conf.parallel_kwargs()` forces
  `parallel=False` whenever `config.infrastructure.parallel` is off; the
  subprocess scheduler is entered only when both the global and per-level flags
  are True.
- **Incompatible config combinations raise a clean error.** Pre-refactor,
  `IIT_3_0` with `GENERALIZED_INTRINSIC_DIFFERENCE` raised a raw `AttributeError`
  deep in the compute path. In 2.0 the eager `validate_config` check rejects it
  at override time with a `ConfigurationError` naming the conflict and a fix;
  with the check off it raises a typed `MeasureNotCompatibleError` at compute
  time. Across an 18-combination (version × measure × system scheme) sweep, 9
  compute and 9 are cleanly rejected; none raise a raw exception.

`shortcircuit_sia` and the cache flags are φ-invariant. Reproduce:
`uv run python -m benchmarks.iit_3_vs_4.config_sweep`.

### 6–7 node sizing

`harness._synth_system` builds seeded Ising substrates at the all-off state for
n ∈ {6, 7} at sparse and dense connectivity; the four networks and their
generative inputs are committed under `results/synth_fixtures/`.

The SIA cost is bimodal. With mean-zero coupling the all-off state is almost
always reducible: the partition search finds a zero-φ partition early and
`map_reduce`'s `is_falsy` short-circuit stops it, so the SIA returns in under a
second without touching most partitions (0 of 20 mean-zero seeds at n=6
integrated). The generator therefore draws **ferromagnetic** weights (mean 1.0),
which make the system integrate and run the full `DIRECTED_SET_PARTITION` search
— the cost the matrix is meant to size.

Partition count under `DIRECTED_SET_PARTITION` grows ~7.8× per node:

| n | partitions | ratio to n−1 |
|---|---|---|
| 3 | 22 | — |
| 4 | 150 | 6.8× |
| 5 | 1061 | 7.1× |
| 6 | 7896 | 7.4× |
| 7 | 61888 | 7.8× |

Measured n=6 (3 trials, harness `cProfile` mode): sparse median 25.8 s, dense
median 23.1 s; φ identical across trials (the construction is deterministic in
the seed). Bare (un-profiled) cost is roughly half that. Scaling by the 7.8×
partition growth — and the per-partition cost also grows with n — puts a single
n=7 trial at several minutes and the full n=7 matrix at roughly an hour, so the
n=7 runs are batch-only and not run here.

Fixtures: `results/synth_fixtures/{name}.json` (seed, weights, CM, TPM);
regenerate with `uv run python -m benchmarks.iit_3_vs_4.synth_fixtures`.

## How to run

The harness lives in `benchmarks/iit_3_vs_4/` of the main pyphi checkout and in
a git worktree at the pre-refactor commit. Same script, same JSON output format.

```sh
# Main checkout (post-refactor): harness already present.
uv pip install pyemd  # IIT 3.0 EMD distance

# Pre-refactor worktree:
git worktree add ../pyphi-pre-refactor b3aaa3e5
cd ../pyphi-pre-refactor
uv venv --python 3.12
GIT_CONFIG_GLOBAL=/dev/null uv pip install -e ".[emd]"   # bypass graphillion git+ssh
cp -r ../pyphi/benchmarks/iit_3_vs_4 benchmarks/
```

```sh
# Post-refactor (main checkout):
uv run python -m benchmarks.iit_3_vs_4.run --networks basic,fig4,xor --trials 5

# Pre-refactor (worktree venv):
VIRTUAL_ENV=../pyphi-pre-refactor/.venv \
  uv run --directory ../pyphi-pre-refactor \
  python -m benchmarks.iit_3_vs_4.run --networks basic,fig4,xor --trials 5

# Copy pre-results into the main checkout for unified analysis, then analyze.
# Copy only the specific new result files, not the whole results/pre directory.
uv run python -m benchmarks.iit_3_vs_4.analyze --top 15
```

## Output format

Per trial, two files in `results/{generation}/`:

- `{network}_{measurement}_seed{S}_trial{T}.json` — wall time, per-phase
  cumulative time, φ, config snapshot, generation, pyphi version.
- `{network}_{measurement}_seed{S}_trial{T}.pstats` — cProfile output, readable
  via `python -m pstats <file>` or snakeviz (gitignored).

## Caveats

1. **IIT 3.0 φ differs pre vs post.** The 2.0 IIT 3.0 implementation was
   overhauled for paper fidelity (tie resolution, partition scheme, MIP
   selection), so pre and post are not the same computation. The wall-time
   comparison is still meaningful as "what an IIT 3.0 user saw in each era."
2. **IIT 4.0 φ matches pre vs post only at n ≤ 3.** On larger networks the
   default `DIRECTED_SET_PARTITION` scheme selects a different system MIP than
   the pre era's `DIRECTED_BI`, so φ differs (see Results).
3. **The pre vs post IIT 4.0 entry points have different scope.** Pre
   `phi_structure` computes the whole phi-structure; post `sia` computes Φ_s
   only. Use the `controls.py` matched comparison for a like-for-like number,
   not the raw entry-point wall times.
