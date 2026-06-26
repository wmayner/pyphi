# IIT 3.0 vs 4.0 Cross-Temporal Performance Investigation

Standalone harness that measures, at **two points in the codebase history**,
where IIT 3.0 and IIT 4.0 spend their time on the same networks. Separate from
the ASV harness in `benchmarks/benchmarks/` so it can iterate independently
of ASV modernization (P3.3 main scope).

## Motivation

Larissa Albantakis raised the puzzle that IIT 4.0 was not empirically faster
than IIT 3.0 in pyphi, despite the formalism advantages (effect-dominated
computation, specified states instead of full repertoires). She made the
observation against pre-2.0 pyphi.

This harness runs both at the pre-refactor anchor (the snapshot of pyphi
she was likely seeing) and at current 2.0 HEAD (post the substantial
restructure landed on 2026-05-09), so the comparison is:

- **Did 4.0 used to be slow compared to 3.0?** (answer: depends on network,
  but yes on some)
- **Did the 2.0 refactor close the gap?** (answer: yes, dramatically)

## Anchor commits

- **pre**: `b3aaa3e5` — *Add ROADMAP.md with path-dependency graph* (2026-05-04).
  Parent of `446c334e` (P0 of the 2.0 work). Has the old layout
  (`pyphi.compute.sia`, `pyphi.new_big_phi.phi_structure`, flat `IIT_VERSION`
  config, `Subsystem` objects, no `formalism/` directory).
- **post**: current HEAD on the `2.0` branch (2026-05-22). New layout
  (`pyphi.formalism.iit3`/`iit4`, layered `config.formalism.iit.version`,
  `System` objects, formalism presets in `pyphi.conf.presets`).

## Entry points used (the right ones — see "Lessons learned" below)

| label                | pre-refactor                                            | post-refactor                                                                  |
| -------------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `iit3_sia`           | `compute.sia(subsystem)` + `IIT_VERSION=3.0`            | `formalism.sia(system)` + `pyphi.conf.presets.iit3` applied via override       |
| `iit4_phi_structure` | `new_big_phi.phi_structure(subsystem)` + `IIT_VERSION=4.0` | (n/a)                                                                       |
| `iit4_sia_2023`      | (n/a)                                                   | `formalism.sia(system)` + `pyphi.conf.presets.iit4_2023` applied via override  |
| `iit4_sia_2026`      | (n/a)                                                   | `formalism.sia(system)` + `pyphi.conf.presets.iit4_2026` applied via override  |

## Findings

Run: 5 trials per (network, measurement) for the 3-node networks; 3 trials for
the larger ones. Phi values were stable across trials within each cell.

### Wall-time medians

| network | n | gen  | iit3_sia | iit4 (pre=`phi_structure`, post=2023) | iit4_sia_2026 |
| ------- | - | ---- | -------- | ------------------------------------- | ------------- |
| basic   | 3 | pre  | 139 ms   | 106 ms                                | n/a           |
| basic   | 3 | post | 146 ms   | **42 ms**                             | 8.5 ms        |
| fig4    | 3 | pre  | 199 ms   | 118 ms                                | n/a           |
| fig4    | 3 | post | 264 ms   | **23 ms**                             | 10 ms         |
| xor     | 3 | pre  | 136 ms   | 190 ms                                | n/a           |
| xor     | 3 | post | 178 ms   | **71 ms**                             | 9.6 ms        |
| macro   | 4 | pre  | 3.98 s   | 1.50 s                                | n/a           |
| macro   | 4 | post | 5.01 s   | **79 ms**                             | 77 ms         |
| rule154 | 5 | pre  | 538.8 s  | 96.8 s                                | n/a           |
| rule154 | 5 | post | 58.0 s   | **2.28 s**                            | 156 ms        |

> **Correction (P17, see `findings.md` Finding 6).** Findings 1 and 3 below
> compare pre-refactor `phi_structure` (full structure: SIA + distinctions +
> relations) against post-refactor `sia` (system-φ only) — **different
> computations**, so the "2.5×–43× speedup" is largely a scope artifact (on
> rule154, ~97% of the gap is the CES work that `sia` simply does not do).
> The de-confounded, like-for-like result: the SIA inner loop is **~18–20×
> faster per partition** and the CES **~2× faster**, but under the default
> paper-faithful `DIRECTED_SET_PARTITION` scheme the SIA evaluates ~35× more
> partitions, so a default-config user's SIA *wall time* can be higher than
> pre. Read Findings 1 and 3 with that correction in mind.

### Finding 1 — The 2.0 refactor delivered massive IIT 4.0 speedups, scaling with network size

`iit4_sia_2023` (post-refactor) is faster than `iit4_phi_structure`
(pre-refactor) on every network, with the gap widening dramatically at scale:

| network | n | pre 4.0  | post 4.0 | speedup |
| ------- | - | -------- | -------- | ------- |
| basic   | 3 | 106 ms   | 42 ms    | 2.5×    |
| fig4    | 3 | 118 ms   | 23 ms    | 5.1×    |
| xor     | 3 | 190 ms   | 71 ms    | 2.7×    |
| macro   | 4 | 1.50 s   | 79 ms    | **19×** |
| rule154 | 5 | 96.8 s   | 2.28 s   | **43×** |

This isn't a constant-factor improvement — it grows with system size,
suggesting an algorithmic change, not just better Python overhead. The phi
values match (or, where they differ, the pre-refactor value is clearly the
buggier one — e.g., the 2026 paper cap was being skipped).

### Finding 2 — IIT 3.0 went *much* faster in 2.0 on rule154, slower on the small networks

The IIT 3.0 picture is mixed and network-dependent:

| network | n | pre 3.0  | post 3.0 | post/pre |
| ------- | - | -------- | -------- | -------- |
| basic   | 3 | 139 ms   | 146 ms   | 1.05× SLOWER |
| fig4    | 3 | 199 ms   | 264 ms   | 1.33× SLOWER |
| xor     | 3 | 136 ms   | 178 ms   | 1.31× SLOWER |
| macro   | 4 | 3.98 s   | 5.01 s   | 1.26× SLOWER |
| rule154 | 5 | 538.8 s  | 58.0 s   | **9.3× FASTER** |

The 3-node and macro slowdowns are 5–33% — modest. The rule154 speedup is
nearly an order of magnitude. Phi values also changed across all 3.0 cells
(e.g., rule154: 4.875 pre → 10.71875 post), consistent with the IIT 3.0
restoration work on the 2.0 branch (tie resolution, partition scheme, MIP
selection). So we are *not* measuring the same computation — the 2.0 IIT 3.0
is paper-faithful where pre-refactor wasn't, with very different scaling
characteristics on a non-trivial network.

### Finding 3 — IIT 4.0 was already faster than IIT 3.0 on bigger networks, even pre-refactor

The original puzzle was framed against small networks. With larger ones in
view, the pre-refactor world had this 4.0/3.0 ratio:

| network | n | pre iit3_sia | pre iit4_phi_structure | 4.0/3.0 ratio |
| ------- | - | ------------ | ---------------------- | ------------- |
| basic   | 3 | 139 ms       | 106 ms                 | 0.76× (4.0 faster) |
| fig4    | 3 | 199 ms       | 118 ms                 | 0.59× (4.0 faster) |
| xor     | 3 | 136 ms       | 190 ms                 | **1.40× (4.0 SLOWER)** |
| macro   | 4 | 3.98 s       | 1.50 s                 | 0.38× (4.0 faster) |
| rule154 | 5 | 538.8 s      | 96.8 s                 | 0.18× (4.0 5.6× faster) |

So even pre-refactor, IIT 4.0 *was* faster than 3.0 on the meatier
networks. The "4.0 not faster than 3.0" puzzle was specifically a
small-network phenomenon — and on xor pre-refactor, 4.0 was actively
slower. Once a network gets large enough for the partition cost to
dominate, 4.0's specified-state machinery starts paying off.

Post-refactor, IIT 4.0 dominates IIT 3.0 across the board:

| network | n | post iit3_sia | post iit4_sia_2023 | 4.0/3.0 ratio |
| ------- | - | ------------- | ------------------ | ------------- |
| basic   | 3 | 146 ms        | 42 ms              | 0.29× (4.0 3.5× faster)  |
| fig4    | 3 | 264 ms        | 23 ms              | 0.09× (4.0 11× faster)   |
| xor     | 3 | 178 ms        | 71 ms              | 0.40× (4.0 2.5× faster)  |
| macro   | 4 | 5.01 s        | 79 ms              | 0.016× (4.0 **63× faster**) |
| rule154 | 5 | 58.0 s        | 2.28 s             | 0.039× (4.0 **25× faster**) |

### Finding 4 — IIT 4.0 2026 (Eq. 23 cap) returns phi=0 on every test network

`iit4_sia_2026` returns phi=0.0 on basic, fig4, xor, macro, rule154 — every
network we ran. Wall times (8–156 ms) reflect early short-circuit, not the
algorithm's true cost on non-trivial outputs. To honestly cost the 2026
variant we'd need a network where it produces non-zero phi. None of the
standard `pyphi.examples` 3–5-node systems do here.

## Larissa's original puzzle, answered (revised)

The puzzle as stated — "4.0 should be faster than 3.0, but isn't" — was
true specifically on small (3-node) networks pre-refactor, and starkly true
on xor (4.0 was 40% slower). Once you go to 4+ nodes, 4.0 was already
faster than 3.0 pre-refactor (1.6× on macro, 5.6× on rule154), just by less
than the formalism would predict.

Post-refactor, the picture is unambiguous: **IIT 4.0 in 2.0 is 2.5×–63×
faster than IIT 3.0**, scaling with network size. And **post-refactor IIT
4.0 is 2.5×–43× faster than pre-refactor IIT 4.0**, with the gap also
widening with network size. The 2.0 work didn't just renumber modules — it
made the 4.0 path substantially faster, exactly where the asymptotic
advantage was expected to live.

## How to run

The harness lives in `benchmarks/iit_3_vs_4/` of the main pyphi checkout
**and** in a git worktree at the pre-refactor commit. Same script, same
JSON output format. The script auto-detects which pyphi generation it's
running against and dispatches to the right entry points.

### Setting up

```sh
# In the main pyphi checkout: harness already present
cd /path/to/pyphi
uv pip install pyemd  # needed for IIT 3.0 EMD distance

# Pre-refactor worktree:
git worktree add /path/to/pyphi-pre-refactor b3aaa3e5
cd /path/to/pyphi-pre-refactor
uv venv --python 3.12
# graphillion has a git+ssh issue with global git config; bypass with:
GIT_CONFIG_GLOBAL=/dev/null uv pip install -e ".[emd]"
cp -r /path/to/pyphi/benchmarks/iit_3_vs_4 benchmarks/
```

### Running

```sh
# Post-refactor (in main checkout):
uv run python -m benchmarks.iit_3_vs_4.run --networks basic,fig4,xor --trials 5

# Pre-refactor (in worktree):
VIRTUAL_ENV=/path/to/pyphi-pre-refactor/.venv \
  uv run --directory /path/to/pyphi-pre-refactor \
  python -m benchmarks.iit_3_vs_4.run --networks basic,fig4,xor --trials 5

# Then copy pre-results back to main checkout for unified analysis:
cp -r /path/to/pyphi-pre-refactor/benchmarks/iit_3_vs_4/results/pre \
      /path/to/pyphi/benchmarks/iit_3_vs_4/results/

# Cross-temporal analysis (from main checkout):
uv run python -m benchmarks.iit_3_vs_4.analyze
uv run python -m benchmarks.iit_3_vs_4.analyze --top 15  # verbose
```

## Outputs

Per trial, two files in `results/{generation}/`:

- `{network}_{measurement}_seed{S}_trial{T}.json` — wall time, phase
  cumulative time, phi, config snapshot, generation, pyphi version
- `{network}_{measurement}_seed{S}_trial{T}.pstats` — cProfile output,
  readable via `python -m pstats <file>` or snakeviz

`dual_analysis.txt` in `results/` is a captured snapshot of the analyzer
output for reference.

## Methodological notes / caveats

1. **`PARALLEL=False` doesn't actually disable subprocess evaluation in
   `_sia_map_reduce`** on the pre-refactor side
   (`pyphi/compute/subsystem.py:222,232` passes the full
   `PARALLEL_*_EVALUATION` dict as the `parallel` keyword to
   `MapReduce(...)`; the truthy dict bypasses MapReduce's bool check). The
   harness works around this with a `force_sequential_mapreduce()` context
   manager. Whether the same hazard exists in the 2.0 code wasn't separately
   verified — the workaround is harmless either way.

2. **IIT 3.0 phi values differ pre vs post.** Pre: 1.25, 2.166, 1.5 on
   basic, fig4, xor. Post: 2.312, 1.817, 1.875. The 2.0 IIT 3.0
   implementation has been overhauled for paper-fidelity (see the recent
   tie-resolution commits on the `2.0` branch). The wall-time comparison
   is still meaningful as "what an IIT 3.0 user would see in each era," but
   strictly we are not measuring the same computation across generations
   for IIT 3.0.

3. **IIT 4.0 phi values match exactly pre vs post.** `iit4_sia_2023` in
   2.0 computes the same Φ as `phi_structure` did pre-refactor on these
   networks. So the 4.0 comparison is apples-to-apples.

4. **Network set is tiny.** Only 3-node networks here. The performance
   ratios likely scale differently on 4+ node systems. A natural follow-up
   is to add `residue` (5 nodes) and `rule154` (5 nodes), which the harness
   already lists as available but doesn't run by default.

5. **`iit4_sia_2026` returns phi=0.0 on all three networks** (likely cap +
   short-circuit; not investigated here). The 8–10 ms wall time on these
   small systems is largely fixed overhead, not the actual cost of the 2026
   algorithm on non-trivial outputs.

## Lessons learned (from getting it wrong the first time)

1. **`compute.sia()` was never the IIT 4.0 entry point.** Pre-refactor,
   IIT 4.0 went through `new_big_phi.phi_structure()`. An earlier
   iteration of this harness mistakenly toggled `IIT_VERSION=4.0` while
   still calling `compute.sia()`, producing meaningless measurements. Per
   formalism, use:
   - 3.0 pre: `compute.sia`
   - 4.0 pre: `new_big_phi.phi_structure`
   - 3.0/4.0 post: `formalism.sia` with the right preset

2. **Stale `.pyc` files masquerade as live code.** The first run accidentally
   imported pre-refactor pyphi modules from `pyphi/compute/__pycache__/*.pyc`
   even though the source files had been deleted weeks earlier. Verify the
   actual git state before trusting any benchmark result.

3. **`pyphi.conf.presets`** has canonical formalism configurations. Use
   them via `config.override(**iit3)`, `config.override(**iit4_2023)`,
   `config.override(**iit4_2026)` rather than hand-rolling overrides. The
   presets bundle partition scheme, tie resolution, measures, etc. — getting
   any of these wrong causes errors deep inside SIA computation.
