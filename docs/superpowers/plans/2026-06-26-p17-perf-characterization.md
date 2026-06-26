# P17 — cross-formalism performance characterization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Characterize why the 2.0 refactor made IIT 4.0 faster, honestly cost the 2026 cap variant, extend coverage to 6–7 nodes, sweep hot-path config flags for behavior mismatches, and conditionally land one targeted optimization — all internal, no public-surface change.

**Architecture:** Extend the existing `benchmarks/iit_3_vs_4/` cross-temporal harness (post-refactor `main` checkout + pre-refactor worktree at `/Users/will/projects/pyphi-pre-refactor`, `b3aaa3e5`). Measurement is exploratory, not asserted behavior — the "verification" of a benchmark task is that it produces sensible, reproducible numbers written to raw per-trial files, summarized in `findings.md`. Only Part 3's heavy runs are server-gated.

**Tech Stack:** Python 3.12+, the harness (`cProfile`/`pstats`), `pyphi.substrate_generator.ising`, uv.

## Global Constraints

- **No public-surface change.** Code changes are confined to `benchmarks/iit_3_vs_4/` except a Part 4/Part 5 fix, which lands in `pyphi/` only behind the full golden suite (no φ/α value may change).
- **Reproducibility:** every synthesized network is built from an explicit integer `seed`, saved in the result JSON and as a serialized TPM next to results; per-trial raw data (`.json` + `.pstats`) is written per trial (harness already does this via `write_record`/`unique_path`); summaries derive from raw, never replace it.
- **No clobbering:** the harness's `unique_path` already appends `_v2`, `_v3`, … — keep using it; never overwrite a result file.
- **Commit trailer** on every commit:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_012dtSzF2YgDjGpFC9mA47ve
  ```
- **Full verification for any `pyphi/` change** = `uv run --all-extras pytest` (no path argument). Never `--no-verify`. Ask before any `git push`.
- **Pre-refactor runs** use the worktree's own venv:
  ```
  VIRTUAL_ENV=/Users/will/projects/pyphi-pre-refactor/.venv \
    uv run --directory /Users/will/projects/pyphi-pre-refactor \
    python -m benchmarks.iit_3_vs_4.run ...
  ```

---

## Wave A — local (runnable now, no server)

### Task 1: Part 1 — close the 2026-cap cost (`logistic3_k8`)

**Files:**
- Modify: `benchmarks/iit_3_vs_4/harness.py` (add the post-gen builder + registry entry)
- Output: `benchmarks/iit_3_vs_4/results/post/logistic3_k8_*` (raw, not committed individually; the harness writes them)

**Interfaces:**
- Produces: a `logistic3_k8` entry in the post `NETWORKS` registry building a 3-node `System` at state `(0,0,0)` with `0 < φ_2026 < φ_2023`.

- [ ] **Step 1: Add the builder to `harness.py`**

After the `NetworkFixture` dataclass (line ~100), add a self-contained builder (replicates the B4 construction so the harness imports nothing from `test/`):

```python
def _logistic3_k8_system() -> Any:
    """3-node fully-connected logistic substrate (k=8, weights 0.3) at (0,0,0).

    The cap-biting network from the Eq-23 oracle: phi_2023 ~ 0.037,
    phi_2026 ~ 0.003, so the 2026 ii(s) cap binds at a non-trivial value
    (unlike the standard examples, where 2026 short-circuits to 0).
    """
    import itertools
    import numpy as np
    from pyphi import Substrate, System

    k = 8.0
    weights = np.full((3, 3), 0.3)
    cm = np.ones((3, 3), dtype=int)
    tpm = np.zeros((8, 3))
    for i, s in enumerate(itertools.product([-1, 1], repeat=3)):
        for j in range(3):
            inp = sum(weights[ki, j] * s[ki] for ki in range(3))
            tpm[i, j] = 1.0 / (1.0 + np.exp(-k * inp))
    return System(Substrate(tpm, cm), (0, 0, 0))
```

- [ ] **Step 2: Register it (post generation only)**

In the `else:` (post) branch of the `NETWORKS` dict (line ~113), add:

```python
        "logistic3_k8": NetworkFixture("logistic3_k8", _logistic3_k8_system, 3),
```

(Leave the `pre` branch untouched — the 2026 variant doesn't exist pre-refactor.)

- [ ] **Step 3: Run it (post, all three measurements, 5 trials)**

Run:
```bash
cd /Users/will/projects/pyphi
uv run python -m benchmarks.iit_3_vs_4.run --networks logistic3_k8 --trials 5 -v
```
Expected: 15 trials complete; `iit4_sia_2026` reports a **non-zero** `phi` (≈0.003), unlike the φ=0 it returns on the standard nets. Eyeball that `phi_2026 < phi_2023` and both are non-zero.

- [ ] **Step 4: Record the cap-cost finding**

Inspect the wall/phase times across the three measurements (read the written JSONs or `analyze.py`). Write the result into `findings.md` (created in Task 2) as the rewrite of the harness README's Finding 4: the honest `iit4_sia_2026` cost on a non-zero output vs `iit4_sia_2023`, and a one-line statement of whether the 2026 path is cheaper because the cap arithmetic is light or because it still short-circuits less work.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/iit_3_vs_4/harness.py
git commit   # "Add logistic3_k8 cap-biting net to the cross-temporal harness"
```

---

### Task 2: Part 2 — mechanism deep-dive

**Files:**
- Create: `benchmarks/iit_3_vs_4/findings.md`
- Possibly modify: `benchmarks/iit_3_vs_4/analyze.py` (add a pre/post function-level cumulative-time diff if not already present)

**Interfaces:**
- Consumes: per-trial `.pstats` from `macro` (4n) and `rule154` (5n) in both generations.
- Produces: a verdict (confirmed / refuted / negligible, with the measured quantity) for each of the three hypotheses.

- [ ] **Step 1: Generate the profile corpus (post, then pre)**

Post (main checkout):
```bash
uv run python -m benchmarks.iit_3_vs_4.run --networks macro,rule154 \
  --measurements iit4_sia_2023 --trials 3 -v
```
Pre (worktree venv — `rule154` IIT 4.0 was ~97 s/trial pre, so this is the slow one; run in background):
```bash
VIRTUAL_ENV=/Users/will/projects/pyphi-pre-refactor/.venv \
  uv run --directory /Users/will/projects/pyphi-pre-refactor \
  python -m benchmarks.iit_3_vs_4.run --networks macro,rule154 \
  --measurements iit4_phi_structure --trials 3 -v
```
Then copy pre results into the main checkout for unified analysis:
```bash
cp -r /Users/will/projects/pyphi-pre-refactor/benchmarks/iit_3_vs_4/results/pre \
      /Users/will/projects/pyphi/benchmarks/iit_3_vs_4/results/
```

- [ ] **Step 2: Function-level cumulative-time diff**

Run `uv run python -m benchmarks.iit_3_vs_4.analyze --top 15` and, if it doesn't already emit a pre-vs-post per-function cumulative-time table, add one to `analyze.py`: for each generation, load the `macro`/`rule154` `.pstats`, rank functions by cumulative time, and print the top contributors side by side. This is the map: which functions hold the wall-time gap.

- [ ] **Step 3: Hypothesis (a) — per-cut unpartitioned-CES recompute (a direct count)**

From the `.pstats` `ncalls` (cProfile call counts), extract how many times the whole-system CES / unpartitioned repertoires are computed per SIA in each generation:
- pre: count calls to `pyphi/compute/subsystem.py::_ces` (and `ces`) per `_sia`.
- post: count calls to the IIT 4.0 `ces` per `sia`.
Compare the ratio to the partition count for `macro`/`rule154`. **Signature of (a):** pre's count scales with the partition count, post's is ~1. Record the actual counts.

- [ ] **Step 4: Hypothesis (b) — config attribute-access overhead**

Count config attribute accesses in the hot loop. Easiest measured proxy: in each generation, run one `macro` trial under cProfile and read the `ncalls` for the config attribute path (`config.__getattribute__` / the layered dataclass accessor). Record the per-SIA access count pre vs post and the cumulative time attributed. **Verdict:** real contributor only if its cumulative time is a non-trivial fraction of the gap from Step 2; otherwise "negligible."

- [ ] **Step 5: Hypothesis (c) — parallelization contribution**

The harness already runs under `force_sequential_mapreduce`, so the existing numbers are the sequential-only comparison. Confirm by reading a couple of `.pstats` that no subprocess time is hidden (no `MapReduce` parallel frames). State explicitly that the measured speedup is the **algorithmic** (sequential) speedup, and that the parallel engine is a separate, additional lever (not the source of the cross-temporal gap). If desired, one parallel-on `macro` trial per generation gives the parallel delta for context.

- [ ] **Step 6: Write `findings.md`**

Create `benchmarks/iit_3_vs_4/findings.md` with: the function-level diff table (Step 2), and the three hypotheses each with a verdict + the measured quantity that supports it. Fold in the Task 1 cap-cost result. Plain prose; negative results stated as negative.

- [ ] **Step 7: Commit**

```bash
git add benchmarks/iit_3_vs_4/findings.md benchmarks/iit_3_vs_4/analyze.py
git commit   # "Characterize the IIT 4.0 speedup mechanism (P17 deep-dive)"
```

---

### Task 3: Part 4 — config-behavior sweep

**Files:**
- Create: `benchmarks/iit_3_vs_4/config_sweep.py` (a small audit script, not a pytest)
- Modify: `benchmarks/iit_3_vs_4/findings.md` (append the sweep results)
- Possibly modify: `pyphi/` (a fix for any genuine bug — behind the goldens)

**Interfaces:**
- Produces: a list of `(flag, documented_behavior, actual_behavior, severity)` rows.

- [ ] **Step 1: Enumerate the hot-path flags to audit**

Target the flags the hot paths actually read: `infrastructure.parallel` + the per-level `parallel_*_evaluation` dicts; `formalism.iit.shortcircuit_sia`; `infrastructure.cache_repertoires` / `cache_potential_purviews`; the measure/scheme combinations in `formalism.iit`. List them in `config_sweep.py` with each flag's documented behavior (from `pyphi/conf/` docstrings).

- [ ] **Step 2: Write the behavior assertions**

For each flag, a small experiment asserting documented == actual. The known template (the pre-refactor `PARALLEL=False`-still-spawns bug): with `parallel=False`, run a `macro` SIA and assert no subprocess is spawned (e.g. patch the scheduler/`MapReduce` to record whether parallel mode was entered). For `shortcircuit_sia`, assert a reducible system short-circuits when true and does the full search when false. For cache flags, assert the cache is/ isn't populated. Run each under both sequential and parallel global settings.

- [ ] **Step 3: Probe raising config combinations**

Iterate a small grid of `(version, mechanism_phi_measure, system_phi_measure, partition_scheme)` combinations on `basic`; record any combination that raises an exception rather than returning a result or a clean `ConfigurationError`. (The IIT 3.0 + `GENERALIZED_INTRINSIC_DIFFERENCE` AttributeError was the pre-refactor instance; check whether an analogue exists in 2.0.)

- [ ] **Step 4: Record findings; fix any genuine bug**

Append the `(flag, documented, actual, severity)` table to `findings.md`. For any real mismatch, write a failing test first (in the appropriate `test/` module), fix it in `pyphi/`, and verify with `uv run --all-extras pytest` (no path argument). A config fix that changes no φ/α keeps every golden green.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/iit_3_vs_4/config_sweep.py benchmarks/iit_3_vs_4/findings.md
# plus any pyphi/ + test/ files if a fix landed
git commit   # "Audit 2.0 hot-path config flags for behavior mismatches (P17)"
```

---

### Task 4: Part 3 construction — synthesized 6–7 node generator (local; runs deferred to Wave B)

**Files:**
- Modify: `benchmarks/iit_3_vs_4/harness.py` (seeded synthesized-network builders + registry entries)
- Output: serialized synthesized TPMs + seeds under `benchmarks/iit_3_vs_4/results/` (committed as fixtures)

**Interfaces:**
- Produces: `synth_n6_sparse`, `synth_n6_dense`, `synth_n7_sparse`, `synth_n7_dense` registry entries, each built from a fixed seed.

- [ ] **Step 1: Confirm the generator API**

Run:
```bash
uv run python -c "from pyphi.substrate_generator import ising; help(ising.probability)"
```
Confirm the signature for building a substrate TPM from Ising weights at `temperature=1/k`. (If the exact entry point differs, adjust to the available `substrate_generator` API — the N1 recipe used `substrate_generator.ising.probability` at `temperature=1/k`, k=4.)

- [ ] **Step 2: Add a seeded builder to `harness.py`**

```python
def _synth_system(n: int, density: float, seed: int) -> Any:
    """A seeded n-node logistic/Ising substrate at the all-off state.

    Weights and connectivity are drawn from an isolated RNG (seed saved in
    the result JSON); the TPM is built with the N1 ising recipe at
    temperature=1/k so the dynamics match the paper-faithful 4.0 examples.
    """
    import numpy as np
    from pyphi import Substrate, System
    from pyphi.substrate_generator import ising

    rng = np.random.default_rng(seed)
    cm = (rng.random((n, n)) < density).astype(int)
    np.fill_diagonal(cm, 1)
    weights = cm * rng.normal(0.0, 0.5, size=(n, n))
    tpm = ising.probability(weights, temperature=0.25)  # k=4 -> T=1/k
    return System(Substrate(tpm, cm), (0,) * n)
```
(Adjust the `ising.probability` call to the confirmed Step-1 signature.)

- [ ] **Step 3: Register the four synthesized nets (post only), with fixed seeds**

```python
        "synth_n6_sparse": NetworkFixture("synth_n6_sparse", lambda: _synth_system(6, 0.35, 6001), 6),
        "synth_n6_dense":  NetworkFixture("synth_n6_dense",  lambda: _synth_system(6, 0.85, 6002), 6),
        "synth_n7_sparse": NetworkFixture("synth_n7_sparse", lambda: _synth_system(7, 0.30, 7001), 7),
        "synth_n7_dense":  NetworkFixture("synth_n7_dense",  lambda: _synth_system(7, 0.80, 7002), 7),
```

- [ ] **Step 4: Serialize the generated TPMs + seeds as committed fixtures**

Write a one-shot that builds each net and saves its TPM + seed + cm to `benchmarks/iit_3_vs_4/results/synth_fixtures/{name}.json` (so a run is reproducible without regenerating, and the seed lives with the data, per the reproducibility rule). Commit these.

- [ ] **Step 5: Local n=6 sanity timing (one trial) to estimate server cost**

```bash
uv run python -m benchmarks.iit_3_vs_4.run --networks synth_n6_sparse \
  --measurements iit4_sia_2023 --trials 1 -v
```
Record the single-trial wall time. If n=6 sparse is already minutes, n=7 will likely exceed the interactive budget — note that estimate in `findings.md` so the Wave B server matrix is sized realistically. **Do not** run n=7 locally (that's Wave B).

- [ ] **Step 6: Commit**

```bash
git add benchmarks/iit_3_vs_4/harness.py benchmarks/iit_3_vs_4/results/synth_fixtures/
git commit   # "Add seeded 6-7 node synthesized fixtures to the P17 harness"
```

---

## Wave B — server (deferred until lab SSH access is restored)

### Task 5: Part 3 runs — 6–7 node matrix on the lab server

**Blocked on:** lab AD credential fix (IT). Not startable until `ssh tononi-1`/`tononi-2` authenticates.

- [ ] **Step 1: Server setup** (per the spec's setup outline): `git clone`/pull pyphi @ `main` + `git worktree add … b3aaa3e5`, `uv venv --python 3.12`, install the `emd` extra, copy the harness into the worktree. (The pre-refactor side is only needed if extending the cross-temporal comparison to the synthesized nets; for 4.0-only scaling, the post checkout suffices.)
- [ ] **Step 2: Run the 4.0 matrix.** `iit4_sia_2023` + `iit4_sia_2026` on `synth_n6_{sparse,dense}` and `synth_n7_{sparse,dense}`, 3 trials; IIT 3.0 only on the ≤5-node existing nets. n=7 trials carry a wall-time budget — record "exceeds budget" rather than blocking.
- [ ] **Step 3: rsync results back** to `/Users/will/projects/pyphi/benchmarks/iit_3_vs_4/results/` and analyze here.
- [ ] **Step 4: Threshold table** in `findings.md`: the `n` where `iit4_sia_2023` crosses ~10 s (interactive) and ~minutes (batch-only).
- [ ] **Step 5: Commit** the extended `findings.md` + result summaries.

---

## Wave C — conditional (only if Task 2 warrants)

### Task 6: Part 5 — one targeted optimization

**Gate:** enter only if Task 2's deep-dive identifies a clear, bounded (~1 week) win.

- [ ] **Step 1:** Write a failing/measuring test that pins the redundant work (e.g. the per-cut recompute count).
- [ ] **Step 2:** Implement the fix in `pyphi/`.
- [ ] **Step 3:** Verify **byte-identical** results on every golden — `uv run --all-extras pytest` (no path argument) green, same φ/α everywhere. Any difference is a bug, not an improvement.
- [ ] **Step 4:** Re-run the P17 harness; record the achieved speedup in `findings.md`.
- [ ] **Step 5:** Changelog fragment (`changelog.d/<name>.optimization.md`) + commit.

If no clear win appears, record the negative result in `findings.md` and close P17 without a Wave C change.

---

## Notes for the implementer

- **The harness already does the reproducibility heavy lifting** — `run_trial` writes per-trial JSON + `.pstats`, `unique_path` prevents clobbering, `force_sequential_mapreduce` isolates the sequential cost. Don't reinvent these.
- **Pre-refactor runs need the worktree venv** (the `VIRTUAL_ENV=… uv run --directory …` form). The post checkout and the worktree share the same harness source.
- **`rule154` pre-refactor IIT 4.0 is the slow trial (~97 s)** — run it in the background (`run_in_background: true`) and keep working.
- **ROADMAP:** on completion, flip the P17 dashboard row (line ~51) from `⬜ open` to its landed state and record the findings location.
