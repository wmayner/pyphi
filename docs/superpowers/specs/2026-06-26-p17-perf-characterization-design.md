# P17 — cross-formalism performance characterization — design

Characterize *why* the 2.0 refactor made IIT 4.0 substantially faster, extend
the cross-temporal benchmark past 5 nodes to find the interactive/batch size
thresholds, honestly cost the 2026 cap variant on a non-zero output, sweep the
2.0 hot-path config flags for documented-vs-actual behavior mismatches, and —
only if the deep-dive surfaces a clear bounded win — land one targeted
optimization. Internal-only (no public-surface change).

## Background — what already exists

- **The harness.** `benchmarks/iit_3_vs_4/` measures IIT 3.0 and 4.0 wall time +
  phase timings + cProfile (`.pstats`) at two points in history: the
  pre-refactor anchor `b3aaa3e5` (live local worktree at
  `/Users/will/projects/pyphi-pre-refactor`, its own `.venv`) and current `main`.
  It auto-detects the generation and dispatches to the correct entry points
  (3.0: `compute.sia` pre / `formalism.sia`+`presets.iit3` post; 4.0:
  `new_big_phi.phi_structure` pre / `formalism.sia`+`presets.iit4_2023|2026`
  post). A `NETWORKS` registry (`harness.py`) holds the fixtures.
- **Findings 1–3 (done), in `README.md`.** 2.0 made IIT 4.0 **2.5×–43×** faster
  than pre-refactor, the gap widening with size (an algorithmic change, not
  constant overhead); post-refactor 4.0 beats 3.0 by **2.5×–63×**; Larissa
  Albantakis's "4.0 isn't faster than 3.0" puzzle was a small-network-only
  effect (4.0 was already faster at ≥4 nodes pre-refactor).
- **Finding 4 (open).** `iit4_sia_2026` returns φ=0 on every network run so far
  (cap + short-circuit), so its measured 8–156 ms is fixed overhead, not real
  cost. **The network to close this now exists:** `logistic3_k8` (the B4
  cap-biting net, `test/integration/test_eq23_cap_oracle.py::_logistic3_k8`),
  which yields `0 < φ_2026 < φ_2023`.
- **Coverage today.** `basic`/`fig4`/`xor` (3n), `macro` (4n), `rule154` (5n).

## Goal

Convert the open questions of P17 into measured, reproducible results and an
internal performance-architecture note, without changing any public surface.

## Design

The work splits into five parts, sequenced cheapest-and-highest-value first.
Parts 1, 2, 4 and the *construction* of Part 3 are **server-independent** (run
on the Mac). Only the Part 3 6–7 node measurement *runs* are **server-gated**
(they would tie up the laptop for hours; they run on the `tononi-2`/`tononi-1`
lab server over SSH once access is restored).

### Part 1 — Close the 2026-cap cost (server-independent)

Add `logistic3_k8` to the harness `NETWORKS` registry (reuse the B4 substrate
construction; the cap-binding state is `(0,0,0)`). Run it through the
post-refactor measurements (`iit3_sia`, `iit4_sia_2023`, `iit4_sia_2026`).

*Deliverable.* Finding 4 rewritten from "2026 returns φ=0 everywhere" to an
honest 2026-vs-2023 wall-time and phase comparison on a non-zero φ output. State
plainly what fraction of the 2023→2026 delta is the cap arithmetic vs the
short-circuit.

### Part 2 — Mechanism deep-dive (server-independent; the scientific core)

The pre/post boundary is the entire 2.0 refactor (hundreds of commits), so clean
single-change ablation is impossible. Instead **turn each hypothesis into a
measured quantity** on `macro` (4n) and `rule154` (5n) — the two largest
speedups — using the harness's existing `.pstats`:

- **Function-level profile comparison.** Aggregate cProfile cumulative time by
  function, pre vs post, and identify which functions account for the wall-time
  gap. This is the map; the hypotheses below are the specific tests.
- **Hypothesis (a): per-cut unpartitioned-CES recompute.** Count how many times
  the whole-system repertoires / unpartitioned CES are computed per SIA in each
  generation (instrument or read from the profile call counts). Pre-refactor
  `compute._sia` is hypothesized to recompute per cut; post-refactor
  `formalism.iit4.sia` is hypothesized to compute once. A count ratio that
  tracks the partition count is the signature.
- **Hypothesis (b): layered-config `__getattr__` removal.** Count config
  attribute accesses in the hot loop pre (flat `config.__getattribute__`) vs
  post (layered dataclass). Magnitude tells whether this is a real contributor
  or a rounding error against (a).
- **Hypothesis (c): P11 parallelization.** Re-measure with parallelism forced
  off in both generations (the harness already has
  `force_sequential_mapreduce`); the residual speedup with parallelism removed
  isolates the algorithmic contribution from the parallel-engine contribution.

*Deliverable.* `benchmarks/iit_3_vs_4/findings.md` (new) — the attribution
backed by counts and the sequential-vs-parallel split, not prose alone. Each
hypothesis gets a verdict: confirmed / refuted / negligible, with the measured
quantity. Negative results are reported as negative (a hypothesis that turns out
negligible is a finding).

### Part 3 — Extended coverage to 6–7 nodes (construction local; runs server-gated)

- **Synthesized fixtures.** Generate `n ∈ {6, 7}` substrates with
  `substrate_generator.ising.probability` at `temperature = 1/k` (the N1 recipe)
  at two connectivity densities — sparse and dense (density is also the P18
  lever and changes both cost and φ structure). Each fixture is built from an
  explicit integer **seed** saved in the fixture metadata; the generated TPM is
  serialized to disk alongside results so a run is reproducible without
  regenerating. Fixtures are added to the `NETWORKS` registry.
- **Coverage matrix.** IIT 4.0 (2023 + 2026) at n ≤ 7; IIT 3.0 capped at n ≤ 5
  (skip above — it is the cost bottleneck, ~minutes/trial at 5n already). n=7 is
  best-effort (relations enumeration may dominate; if a single `sia()` exceeds a
  wall-time budget it is recorded as "exceeds budget" rather than blocking).
- **Threshold detection.** From the 4.0 curve, report the network size at which
  `iit4_sia_2023` crosses ~10 s/trial (interactive ceiling) and ~minutes/trial
  (batch-only). A short local sanity run at n=6 (one trial) estimates per-trial
  cost before committing the server to the full matrix.

*Deliverable.* The synthesized fixtures + their seeds committed; the
size-threshold table in `findings.md`. The heavy multi-trial runs execute on the
lab server; results JSON/pstats are rsynced back to the Mac checkout for
analysis here.

### Part 4 — Config-behavior sweep (server-independent)

Enumerate the config flags exercised in the 2.0 hot paths — the parallel gates
(`parallel`, the per-level `parallel_*_evaluation` dicts), the SIA short-circuit
(`shortcircuit_sia`), the cache flags, and the measure/scheme combinations — and
for each assert **documented behavior == actual behavior**. The harness
construction already surfaced one real instance pre-refactor (`PARALLEL=False`
not disabling subprocess evaluation because a truthy dict was passed as the
bool); this part looks for analogous live issues in 2.0:

- A config flag whose docstring promises behavior the code doesn't deliver
  (e.g. a disable switch that doesn't fully disable).
- A config *combination* that raises an exception rather than producing a clean
  result or a clear configuration error.

Method: targeted assertions / small experiments per flag, run under both
sequential and parallel settings. This is a correctness audit, not a
benchmark.

*Deliverable.* A findings list in `findings.md` (each item: flag, documented
behavior, actual behavior, severity) plus a fix for any genuine bug — each fix
gated behind the existing golden suite (no φ/α value may change).

### Part 5 — Optional targeted optimization (decision gate, not a commitment)

Only enter if Part 2 identifies a clear, bounded (~1 week) win (e.g. a redundant
repertoire recomputation, a parallelism gate that fails to fire). Any landed
change must be **byte-identical on every existing golden** — perf fixes in hot
paths have a history of silent correctness regressions, so the dense path / the
current goldens are the oracle and any difference is a bug, not an improvement.
If no clear win appears, record the negative result and stop. This part does not
pre-commit to landing anything.

## Reproducibility (standing project rules)

- Every synthesized network is built from an explicit integer seed, saved with
  its output (not just printed); the generated TPM is serialized alongside.
- Per-trial raw data (wall time, phase timings, φ, config snapshot, generation,
  `.pstats`) is written per trial — the harness already does this. Summary
  tables in `findings.md` are derived from, and never replace, the raw JSON.
- No clobbering: results use per-network/per-seed/per-trial filenames; a re-run
  with new parameters gets new files.

## Files

- `benchmarks/iit_3_vs_4/harness.py` — extend `NETWORKS` (logistic3_k8 +
  synthesized 6–7n fixtures); add the seeded synthesized-network generator.
- `benchmarks/iit_3_vs_4/run.py` — accept the new networks; n-gated 3.0 skip.
- `benchmarks/iit_3_vs_4/analyze.py` — threshold detection + the
  pre/post function-level profile diff for Part 2.
- `benchmarks/iit_3_vs_4/findings.md` (new) — mechanism write-up, threshold
  table, cap-cost result, config-sweep findings.
- `benchmarks/iit_3_vs_4/results/` — raw per-trial JSON/pstats (synthesized TPMs
  + seeds saved here).
- Possibly small fixes in `pyphi/formalism/` or `pyphi/conf/` if Part 4 or
  Part 5 lands a change (each behind the goldens).

## Execution waves (given the server gate)

1. **Wave A (local, now):** Part 1, Part 2, Part 4, and Part 3 *construction*
   (generator + fixtures + local n=6 sanity timing).
2. **Wave B (server, when access returns):** Part 3 full 6–7 node matrix on the
   lab server over SSH; rsync results back; threshold table.
3. **Wave C (local, conditional):** Part 5, only if Part 2 warrants it.

## Risk

- Low for the measurement parts. Medium for any landed optimization (Part 5) or
  config fix (Part 4) — gated behind the full golden suite, run with
  `uv run --all-extras pytest` (no path argument, doctest sweep included).
- The pre-refactor worktree depends on a separate `.venv` and the `emd` extra;
  if the venv drifts, Part 2's pre-side needs a reinstall (documented in the
  harness README).
- Server access is currently blocked on a lab AD credential issue (IT). Wave A
  proceeds independently; only Wave B waits.

## Non-goals

- No public-surface change (internal characterization only).
- No approximation work (that is P16; P17 informs it but does not build it).
- No sparse-inversion rewrite (that is P18; P17's threshold data feeds its
  go/no-go but P17 does not implement it).
- IIT 3.0 past 5 nodes (intractable; deliberately capped).
