# P11.8 Tier 2 — Benchmark suite rewrite + ASV-in-CI

**Status:** approved design (2026-06-22)
**Roadmap item:** P11.8 Tier 2 (Wave 3 — regression gate). Tier 1 (the inline
wall-time perf budget, `test/test_perf_budget.py`) is already landed.

## Problem

The `benchmarks/` suite predates the 2.0 architecture by years. Every module
imports symbols that no longer exist (`pyphi.Subsystem`, `pyphi.compute`,
`examples.basic_subsystem`) and uses pre-`distinction` vocabulary
(`BenchmarkConstellation`). It cannot run. PyPhi therefore has no perf
regression signal beyond Tier 1's catastrophic-only wall-time budget — the
class of gate that would have caught the shipped 60–300× slowdown (five nested
defensive `config.override` calls per partition interacting with a latent YAML
write) only because it was catastrophic. A 1.3–1.8× redundant-work creep would
pass Tier 1 silently, and the downstream performance work (P17 characterization,
P18 sparse inference) needs a trustworthy harness to prove a speedup is real and
to catch regressions.

## Goals

1. Replace the broken suite with a 2.0-vocabulary benchmark suite that covers
   every hot path.
2. Two regression signals with distinct roles:
   - **Deterministic call-count gate** that *blocks PRs* (exact, reproducible).
   - **Wall-time trend** that *advises* (nightly, never blocks).
3. Keep benchmarks in lockstep with the correctness goldens — one fixture set,
   no second museum to rot.
4. Add no overhead to shipped (non-benchmark) computation.

## Non-goals

- Hosting an ASV HTML dashboard (the `asv continuous` comparison and the trend
  database are kept; publishing the HTML report is a later option).
- A third-party benchmarking service (e.g. CodSpeed). Its deterministic
  CPU-instruction signal is valuable, but it requires an external account and
  sends computation traces off-machine. The cProfile call-count gate below is
  the in-repo equivalent: a deterministic count instead of wall-time, no
  external dependency.
- PR-time wall-time gating. CI wall-time swings 2–3× across runners; gating PRs
  on it produces flaky failures that get bypassed. Wall-time advises only.

## Design

### Inputs — reuse the golden zoo

The benchmark layer draws substrates, states, and config contexts from
`test/golden/zoo.py` (`ALL_FIXTURES`: 24 fixtures named `<substrate>_<formalism>`
across `iit3_emd` / `iit4_2023` / `iit4_2026`, including the `multivalued_*`
k-ary fixtures and the `logistic3_k8` cap-activation pair). The benchmark
operations call the existing layer functions in `test/golden/compute.py`
(`_compute_repertoires`, `_compute_mechanism_mips`, `_compute_sia`) with a
**no-op stash** (`lambda _arr: ""`), so the timed/profiled code path is
byte-identical to what the correctness goldens exercise — every golden fixture
gets a perf number, and the two suites cannot drift.

Each layer function reaches the real 2.0 hot paths:
`System.cause_repertoire` / `System.effect_repertoire` (repertoires),
`System.find_mip` (mechanism MIPs), and `System.sia()` /
`pyphi.formalism.iit3.sia()` (SIA).

Actual Causation is the one hot path outside the IIT golden zoo, so it gets one
bespoke fixture: the canonical `examples.actual_causation_substrate` transition,
driving `account()` (`pyphi/formalism/actual_causation/compute.py`).

### Signal 1 — wall-time (ASV nightly, advisory)

ASV (airspeed velocity) is the ecosystem standard for Python library perf
tracking; it is already a dev dependency. Its value here is the historical trend
database across the branch's commits plus the two-commit `asv continuous`
comparison.

`benchmarks/benchmarks/` is rewritten. The five museum modules (`compute.py`,
`subsystem.py`, `emd.py`, `tpm.py`, `utils.py`) are deleted and replaced with:

- `_fixtures.py` — adapter bridging the golden zoo into ASV: builds a `System`
  from a `GoldenFixture` inside its config context, exposes the no-op-stash
  layer callables.
- `layers.py` — the layered grid: **all 24 golden fixtures × 4 grains**.
  The grains are `repertoires`, `mechanism_mips`, `phi_structure`, and `sia`.
  `phi_structure` is a new grain (distinctions + relations enumeration, 4.0
  fixtures only) that times the relations work currently buried inside the `sia`
  layer, so a relations-enumeration regression localizes to its own number.
  ASV parameterization: `params = [fixture_names, grains]`, with grains skipped
  where `GoldenFixture.skip_layers` or formalism applies (3.0 fixtures have no
  `phi_structure`).
- `edges.py` — concerns the layered grid cannot see: parallel-vs-sequential SIA
  on a mid-size substrate (wall-time only — call counts are meaningless across
  worker processes), and cold-vs-warm repertoire cache on a repertoire-heavy
  substrate.
- `micro.py` — the EMD distance kernel (`ot.emd2` via the POT backend), the
  IIT 3.0 inner loop that recently changed backends.
- `actual_causation.py` — the AC `account()` benchmark on the AC fixture.

- `counts.py` — the full-zoo call-count coverage: ASV `track_*` benchmarks
  that return the exact cProfile `ncalls` of the hot frames (below) for **all
  24 fixtures × the applicable grains**. Because a call count is deterministic,
  ASV step-detection over these series is exact — any change is a real change,
  with no false positives — so the full zoo gets count-regression coverage with
  no in-repo pins to maintain. These are the nightly counterpart to the bounded
  pinned PR gate in Signal 2.

`benchmarks/asv.conf.json`: `branches` → `["2.0"]` (was `["develop"]`); pythons
left as configured.

`.github/workflows/benchmark.yml`: a new scheduled (nightly) workflow that
**accumulates results into a persisted ASV database** (committed to a dedicated
`benchmark-results` branch, or stored as a CI artifact restored at the start of
each run) and appends the night's run with `asv run`. Regressions are found by
ASV's built-in **step detection over the full accumulated history** — this
catches both sudden jumps and slow multi-week creep, where a single pairwise
comparison would miss the creep. On a detected step beyond ASV's configured
factor, the workflow opens or updates a tracking GitHub issue. The wall-time
regression factor is set generously because GitHub-hosted runners have variable
CPUs; the deterministic count series (from `counts.py`) tolerate a tight factor.
The workflow never runs on PRs and never blocks. *(A self-hosted/dedicated
runner would quiet the wall-time noise without any other change; not required to
ship.)*

### Signal 2 — call counts (cProfile, deterministic, blocks PRs)

`test/test_perf_counters.py` is the blocking gate. For a focused pinned subset —
each formalism × each layer, plus a k-ary fixture, a relations-heavy 4.0
fixture, and the AC fixture — it runs the operation under `cProfile`, then reads
the exact `ncalls` of a handful of hot frames from the profiler stats and
asserts them against pinned values:

- the SIA system-partition evaluation,
- the repertoire kernel (`core/repertoire_algebra`),
- `System.find_mip` / MICE search,
- relations enumeration (the P6b pure-Python DFS),
- **config attribute access** — the sentinel frame; the `config.override`-per-
  partition disaster would have shown as this count exploding.

The counts are exact and reproducible per fixture (the call structure is
deterministic given inputs); profiler overhead inflates wall-time only, never
the counts, so this runs in the normal PR test job with zero production cost and
no source instrumentation.

The subset is **bounded on purpose** — the blocking PR gate must stay fast so it
is never routed around, and profiling all 24 fixtures × 4 layers under cProfile
(2–5× overhead) would add minutes to every PR. The bounded subset still covers
every formalism, every layer, k-ary, the relations path, and AC. Full 24-fixture
count coverage is provided **nightly, not on PRs**, by the `counts.py` ASV
`track_*` metrics (Signal 1): deterministic counts make step-detection exact, so
the whole zoo is watched for count regressions with no in-repo pins to maintain.
The division of labor: in-repo exact pins block PRs on a fast subset; the
nightly ASV count series watch the full zoo.

**Threshold policy — exact pins, golden-style.** Expected counts live in
`test/data/perf/call_counts.json`, regenerated by `scripts/gen_perf_counts.py`
and reviewed in the diff exactly like a φ golden. A legitimate algorithm change
that alters the call structure regenerates the pins as a deliberate, reviewed
step. This gives maximum sensitivity (catches even a 1.2× redundant-work creep)
and matches the repo's existing "lock baselines deliberately" culture.

### Files

| File | Change |
|---|---|
| `benchmarks/benchmarks/compute.py`, `subsystem.py`, `emd.py`, `tpm.py`, `utils.py` | delete (museum) |
| `benchmarks/benchmarks/_fixtures.py` | new — golden-zoo → ASV adapter |
| `benchmarks/benchmarks/layers.py` | new — 24 fixtures × 4 grains |
| `benchmarks/benchmarks/edges.py` | new — parallel/sequential, cold/warm cache |
| `benchmarks/benchmarks/micro.py` | new — EMD distance |
| `benchmarks/benchmarks/actual_causation.py` | new — AC account |
| `benchmarks/benchmarks/counts.py` | new — full-zoo call-count `track_*` metrics |
| `benchmarks/asv.conf.json` | `branches` → `["2.0"]` |
| `test/test_perf_counters.py` | new — deterministic call-count gate (bounded subset) |
| `test/data/perf/call_counts.json` | new — pinned counts (bounded subset) |
| `scripts/gen_perf_counts.py` | new — regeneration script |
| `.github/workflows/benchmark.yml` | new — nightly accumulate-`asv run` + step-detection + issue alert |
| `ROADMAP.md` | mark P11.8 Tier 2 landed; mark P15 "Layer D" superseded |
| `changelog.d/<name>.misc.md` | dev-tooling change fragment |

## Testing

- The call-count gate *is* a test; it runs in `uv run pytest` (full, no-path).
- ASV benchmarks are smoke-verified during implementation with
  `asv run --quick` (one iteration, correctness of discovery/params, not timing).
- `scripts/gen_perf_counts.py` is verified by round-trip: regenerate, confirm
  the committed `call_counts.json` is unchanged.
- The nightly workflow is verified by a manual `workflow_dispatch` run.

## Risks

- **ASV benchmark discovery vs the test package.** The benchmark modules import
  from `test/golden/`, which is a test package, not shipped. ASV builds the
  project into an isolated env; the adapter must import the golden harness in a
  way that works under ASV's build (e.g. add the repo root / `test` to the path
  in `_fixtures.py`, or vendor the minimal fixture-build logic). Resolved in the
  plan; flagged here because it is the main integration unknown.
- **cProfile frame identification.** Frames are matched by
  `(filename, lineno, function name)`; a refactor that moves a function changes
  its identity. The regeneration script addresses intended moves; the pin diff
  surfaces unintended ones.
- **Persisted results database.** The nightly accumulate-and-step-detect design
  requires the ASV results database to survive between runs (a `benchmark-results`
  branch or a restored CI artifact). If it is lost, history resets and creep
  detection restarts from the next run — degraded, not broken. The plan picks one
  persistence mechanism and documents recovery.
- **Ephemeral-runner wall-time noise.** GitHub-hosted runners vary in CPU, so
  absolute wall-times in the accumulated database are noisy; step-detection on
  wall-time uses a generous factor and the alert is advisory (human-triaged). The
  deterministic count series (`counts.py`) are immune to this and carry a tight
  factor. A self-hosted runner removes the noise without other changes.
