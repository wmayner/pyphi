# P11.8 Tier 2 — Benchmark suite + ASV-in-CI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the broken pre-2.0 `benchmarks/` suite with a 2.0-vocabulary ASV benchmark suite driven off the golden harness, plus a nightly wall-time trend (advisory) and a deterministic cProfile call-count gate that blocks PRs.

**Architecture:** Benchmarks draw inputs from `test/golden/zoo.py` (`ALL_FIXTURES`) and call the existing `test/golden/compute.py` layer functions with a no-op stash, so the measured path is byte-identical to the correctness goldens. Wall-time is tracked by ASV `time_*` benchmarks over all 24 fixtures × 4 grains plus edge/micro/AC benchmarks; the full zoo's deterministic call counts are tracked by ASV `track_*` metrics nightly. A bounded, exact-pinned subset of those counts runs as a normal pytest that fails CI on any change.

**Tech Stack:** Python 3.12+, `asv` (airspeed velocity, already a dev dep), `cProfile`/`pstats` (stdlib), pytest, GitHub Actions.

## Global Constraints

- Python 3.12+ only; no backward-compat shims.
- Run all dev commands via `uv run` (e.g. `uv run asv ...`, `uv run pytest`).
- Commit trailer on every commit:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_012dtSzF2YgDjGpFC9mA47ve
  ```
- Pre-commit (ruff + pyright) must pass; never use `--no-verify`. ruff may reformat/strip imports and abort the commit — re-`git add` the named files and re-commit.
- Branch `2.0` is shared with concurrent instances; stage only the files this plan names, never `git add -A`. Do not commit `graphify-out/`, `.claude/settings.json`, or untracked scratch.
- No planning-artifact markers (`P11.8`, "Wave N", `TODO(Px)`) in `pyphi/` source/docstrings/changelog. They are fine in this plan and in `ROADMAP.md`.
- The full verification gate is `uv run pytest` with **no path argument** (collects the `pyphi/` doctest sweep).
- `benchmarks/` is dev-only (already `--ignore=./benchmarks` in pytest, `T201` print allowed, excluded from pyright per `pyproject.toml`). It is never imported by `pyphi`.

## Key facts from the codebase (read before starting)

- `test/golden/zoo.py` exports `ALL_FIXTURES: list[GoldenFixture]` — 24 fixtures named `<substrate>_<formalism>` across `iit3_emd`/`iit4_2023`/`iit4_2026`.
- `GoldenFixture` (`test/golden/fixture.py`): fields `name`, `config_overrides`, `substrate_factory`, `state`, `node_indices` (`None` = full), `skip_layers: frozenset[str]`, `slow: bool`; methods `config_context()` → `config.override(**config_overrides)`, `build_substrate()`.
- `test/golden/compute.py` layer functions, each `(system, stash) -> ...`:
  - `_compute_repertoires(system, stash)` — calls `system.cause_repertoire` / `system.effect_repertoire` over every (mechanism, purview).
  - `_compute_mechanism_mips(system, stash)` — calls `system.find_mip(direction, mechanism, purview)`.
  - `_compute_sia(system, stash, iit_version)` — `system.sia()` (4.0) or `pyphi.formalism.iit3.sia(system)` (3.0).
  - `stash` is only used to serialize arrays; a no-op `lambda _arr: ""` removes serialization from the measurement.
- System build (mirror `compute_all_layers` lines 49–51):
  ```python
  substrate = fixture.build_substrate()
  nodes = fixture.node_indices or substrate.node_indices
  system = System(substrate, fixture.state, nodes)
  ```
- `System.ces()` (`pyphi/system.py:675`) returns the 4.0 `CauseEffectStructure` (distinctions + relations) — the `phi_structure` grain. 3.0 fixtures skip it (they carry `skip_layers={"phi_structure"}`).
- AC: `pyphi.actual.Transition(substrate, before_state, after_state, cause_indices, effect_indices)` then `pyphi.actual.account(transition)`. Canonical fixture: `pyphi.examples.actual_causation_substrate()`, transition `(1,0) -> (1,0)` over indices `(0,1)/(0,1)` under the `iit3` preset (`pyphi.conf.presets.iit3`).
- `asv.conf.json` currently sets `branches: ["develop"]`, `environment_type: "virtualenv"`, `pythons: ["3.12","3.13"]`, `benchmark_dir: "benchmarks"`.
- Existing CI workflows: `.github/workflows/build.yml`, `lint.yml`, `test.yml`.

## File Structure

| File | Responsibility |
|---|---|
| `benchmarks/benchmarks/compute.py`, `subsystem.py`, `emd.py`, `tpm.py`, `utils.py` | **deleted** (pre-2.0 museum) |
| `benchmarks/benchmarks/_fixtures.py` | Adapter: golden zoo → `(name, System, no-op layer callables)`; `sys.path` shim so `test.golden` imports under ASV's project checkout |
| `benchmarks/benchmarks/layers.py` | ASV `time_*` over all 24 fixtures × {repertoires, mechanism_mips, phi_structure, sia} |
| `benchmarks/benchmarks/edges.py` | ASV `time_*`: parallel-vs-sequential SIA, cold-vs-warm repertoire cache |
| `benchmarks/benchmarks/micro.py` | ASV `time_*`: EMD distance kernel (`ot.emd2` via POT) |
| `benchmarks/benchmarks/actual_causation.py` | ASV `time_*`: AC `account()` |
| `benchmarks/benchmarks/counts.py` | ASV `track_*`: full-zoo deterministic call counts |
| `benchmarks/asv.conf.json` | `branches` → `["2.0"]` |
| `test/golden/perf.py` | **Single shared perf harness**: `count_calls`, `FRAMES`, `FIXTURES_BY_NAME`, `GRAINS`, `build_system`, `applies`, `run_grain` — imported by the gate test, the regen script, and (via the shim) `_fixtures.py`. One source of truth so `FRAMES`/grain-dispatch cannot drift across consumers. |
| `scripts/gen_perf_counts.py` | Regenerate `test/data/perf/call_counts.json` |
| `test/data/perf/call_counts.json` | Pinned exact counts (bounded subset) |
| `test/test_perf_counters.py` | Blocking PR gate asserting pinned counts |
| `.github/workflows/benchmark.yml` | Nightly: accumulate ASV results + step detection + issue alert |
| `ROADMAP.md` | Mark P11.8 Tier 2 landed; P15 "Layer D" superseded |
| `changelog.d/benchmark-suite-asv.misc.md` | Dev-tooling changelog fragment |

---

### Task 1: Shared perf harness (`test/golden/perf.py`)

The single source of truth every consumer (the PR gate test, the regen script, and the ASV benchmark modules) imports: the cProfile counting primitive, the canonical `FRAMES` list, and the fixture/grain helpers. Lives in `test/golden/` so the test and script import it directly and the benchmark suite imports it via the `sys.path` shim (Task 2). Putting it here is what makes it impossible for `FRAMES` or the grain dispatch to drift between consumers.

**Files:**
- Create: `test/golden/perf.py`
- Test: `test/test_perf_counters.py` (created here with one helper test; the gate proper is Task 8)

**Interfaces:**
- Produces:
  - `count_calls(thunk: Callable[[], object], frames: list[tuple[str, str]]) -> dict[str, int]` — runs `thunk()` under `cProfile`, returns `{f"{sub}:{func}": total_calls}` for each `(file_substring, funcname)`, summing the total-call count (`nc`) of every profiled frame whose filename contains `file_substring` and whose name equals `funcname` (0 if none match).
  - `FRAMES: list[tuple[str, str]]` — the canonical hot-frame patterns (tuned in Task 7).
  - `FIXTURES_BY_NAME: dict[str, GoldenFixture]`, `GRAINS: tuple[str, ...]`.
  - `build_system(fixture) -> System`, `applies(fixture, grain) -> bool`, `run_grain(fixture, grain) -> None` (enters the config context, runs the grain with a no-op stash).

- [ ] **Step 1: Write the failing test**

```python
# test/test_perf_counters.py
"""Deterministic call-count regression gate (cProfile-based)."""

from __future__ import annotations


def _helper(i: int) -> int:
    return i * 2


def test_count_calls_counts_a_known_frame():
    from test.golden.perf import count_calls

    def thunk():
        return [_helper(i) for i in range(5)]

    counts = count_calls(thunk, [("test_perf_counters", "_helper")])
    assert counts["test_perf_counters:_helper"] == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_perf_counters.py::test_count_calls_counts_a_known_frame -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'test.golden.perf'`.

- [ ] **Step 3: Write the implementation**

```python
# test/golden/perf.py
"""Shared performance-harness helpers.

The single source of truth imported by the perf-counter regression gate
(``test/test_perf_counters.py``), its regeneration script
(``scripts/gen_perf_counts.py``), and the ASV benchmark suite (via the
``benchmarks/benchmarks/_fixtures.py`` path shim). Keeping the FRAMES list and
the grain dispatch here means they cannot drift between consumers.

Call counts are exact and reproducible for a given computation; profiler
overhead inflates wall time only, never the counts.
"""

from __future__ import annotations

import cProfile
import pstats
from collections.abc import Callable

from pyphi import System

from .compute import (
    _compute_mechanism_mips,
    _compute_repertoires,
    _compute_sia,
)
from .zoo import ALL_FIXTURES

FIXTURES_BY_NAME = {f.name: f for f in ALL_FIXTURES}
GRAINS = ("repertoires", "mechanism_mips", "phi_structure", "sia")

# Hot frames as (file_substring, funcname). Tuned against live cProfile stats in
# Task 7; this is the canonical list every consumer imports.
FRAMES: list[tuple[str, str]] = [
    ("system.py", "find_mip"),
    ("system.py", "cause_repertoire"),
    ("system.py", "effect_repertoire"),
    ("relations.py", "relations"),
    ("conf/", "override"),
]


def count_calls(
    thunk: Callable[[], object],
    frames: list[tuple[str, str]],
) -> dict[str, int]:
    """Run ``thunk`` under cProfile; return total call counts for ``frames``."""
    profiler = cProfile.Profile()
    profiler.enable()
    thunk()
    profiler.disable()
    stats = pstats.Stats(profiler)
    counts = {f"{sub}:{func}": 0 for sub, func in frames}
    # stats.stats maps (filename, lineno, funcname) -> (cc, nc, tt, ct, callers).
    for (filename, _lineno, funcname), (_cc, nc, *_rest) in stats.stats.items():  # type: ignore[attr-defined]
        for sub, func in frames:
            if func == funcname and sub in filename:
                counts[f"{sub}:{func}"] += nc
    return counts


def _is_iit3(fixture) -> bool:
    iit = fixture.config_overrides.get("iit")
    if iit is not None and hasattr(iit, "version"):
        return iit.version == "IIT_3_0"
    return fixture.config_overrides.get("FORMALISM") == "IIT_3_0"


def build_system(fixture) -> System:
    substrate = fixture.build_substrate()
    nodes = fixture.node_indices or substrate.node_indices
    return System(substrate, fixture.state, nodes)


def applies(fixture, grain: str) -> bool:
    if grain in fixture.skip_layers:
        return False
    if grain == "phi_structure" and _is_iit3(fixture):
        return False
    return True


def run_grain(fixture, grain: str) -> None:
    """Run one grain of one fixture inside its config context (no-op stash)."""
    with fixture.config_context():
        system = build_system(fixture)
        if grain == "repertoires":
            _compute_repertoires(system, lambda _a: "")
        elif grain == "mechanism_mips":
            _compute_mechanism_mips(system, lambda _a: "")
        elif grain == "phi_structure":
            system.ces()
        elif grain == "sia":
            _compute_sia(system, lambda _a: "", 3.0 if _is_iit3(fixture) else 4.0)
        else:
            raise ValueError(f"unknown grain {grain!r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest test/test_perf_counters.py::test_count_calls_counts_a_known_frame -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add test/golden/perf.py test/test_perf_counters.py
git commit -m "Add shared performance harness (count_calls, FRAMES, grain dispatch)

test/golden/perf.py is the single source the perf-counter gate, its regen
script, and the ASV benchmark suite all import — the cProfile counting
primitive, the canonical FRAMES patterns, and the fixture/grain helpers — so
those cannot drift between consumers.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012dtSzF2YgDjGpFC9mA47ve"
```

---

### Task 2: Benchmark fixture adapter + delete the museum

Establish the 2.0 benchmark surface: delete the broken modules and add the adapter that turns a `GoldenFixture` into a built `System` plus no-op-stash layer callables. This is the foundation every ASV module imports.

**Files:**
- Delete: `benchmarks/benchmarks/compute.py`, `benchmarks/benchmarks/subsystem.py`, `benchmarks/benchmarks/emd.py`, `benchmarks/benchmarks/tpm.py`, `benchmarks/benchmarks/utils.py`
- Create: `benchmarks/benchmarks/_fixtures.py`

**Interfaces:**
- Consumes: `test.golden.perf` (`FIXTURES_BY_NAME`, `GRAINS`, `build_system`, `applies`, `run_grain`, `count_calls`, `FRAMES`).
- Produces: the same names, re-exported under `_fixtures` so ASV modules can `from _fixtures import ...` without knowing about the path shim.

- [ ] **Step 1: Delete the museum modules**

```bash
git rm benchmarks/benchmarks/compute.py benchmarks/benchmarks/subsystem.py \
       benchmarks/benchmarks/emd.py benchmarks/benchmarks/tpm.py benchmarks/benchmarks/utils.py
```

- [ ] **Step 2: Write the adapter (thin shim + re-export)**

```python
# benchmarks/benchmarks/_fixtures.py
"""Bridge the shared perf harness into ASV benchmarks.

ASV checks the project out into its env's ``project/`` dir and runs benchmarks
from ``benchmarks/benchmarks/``. The perf harness lives at the repo root under
``test/golden/``, which is not an installed package, so we put the repo root on
``sys.path`` before importing it, then re-export its names.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from test.golden.perf import (  # noqa: E402
    FIXTURES_BY_NAME,
    FRAMES,
    GRAINS,
    applies,
    build_system,
    count_calls,
    run_grain,
)

__all__ = [
    "FIXTURES_BY_NAME",
    "FRAMES",
    "GRAINS",
    "applies",
    "build_system",
    "count_calls",
    "run_grain",
]
```

- [ ] **Step 3: Verify the adapter imports and runs a grain**

Run:
```bash
uv run python -c "
import sys; sys.path.insert(0, 'benchmarks/benchmarks')
import _fixtures as f
print('fixtures:', len(f.FIXTURES_BY_NAME), 'grains:', f.GRAINS)
fx = f.FIXTURES_BY_NAME['basic_iit4_2023']
f.run_grain(fx, 'sia'); print('sia grain ran')
fx3 = f.FIXTURES_BY_NAME['basic_iit3_emd']
print('phi_structure applies to 3.0?', f.applies(fx3, 'phi_structure'))
"
```
Expected: `fixtures: 24 grains: ('repertoires', 'mechanism_mips', 'phi_structure', 'sia')`, `sia grain ran`, `phi_structure applies to 3.0? False`.

- [ ] **Step 4: Commit**

```bash
git add -A benchmarks/benchmarks/
git commit -m "Replace pre-2.0 benchmark museum with golden-harness adapter

The benchmarks/ modules predated 2.0 and imported removed symbols
(pyphi.Subsystem, pyphi.compute, BenchmarkConstellation). Delete them and
add _fixtures.py, which builds a System from each golden fixture and runs a
named grain (repertoires/mechanism_mips/phi_structure/sia) with a no-op
stash, so benchmarks measure byte-identical paths to the goldens.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012dtSzF2YgDjGpFC9mA47ve"
```

Note: `git add -A benchmarks/benchmarks/` is scoped to the benchmark dir (the deletions + the new file), not the whole repo.

---

### Task 3: Layered wall-time benchmarks

The core grid: every golden fixture × every applicable grain, as ASV `time_*` benchmarks.

**Files:**
- Create: `benchmarks/benchmarks/layers.py`

**Interfaces:**
- Consumes: `_fixtures.FIXTURES_BY_NAME`, `GRAINS`, `applies`, `run_grain`.
- Produces: a `Layers` ASV benchmark class parameterized by `(fixture_name, grain)`.

- [ ] **Step 1: Write the benchmark module**

```python
# benchmarks/benchmarks/layers.py
"""Wall-time benchmarks: every golden fixture x every applicable grain."""

from __future__ import annotations

from _fixtures import FIXTURES_BY_NAME, GRAINS, applies, run_grain


class Layers:
    params = (sorted(FIXTURES_BY_NAME), list(GRAINS))
    param_names = ("fixture", "grain")
    # SIA on the larger fixtures is slow; give ASV room and few repeats.
    timeout = 600.0
    number = 1
    repeat = (1, 3, 30.0)  # (min_repeat, max_repeat, max_seconds)

    def setup(self, fixture_name: str, grain: str) -> None:
        fixture = FIXTURES_BY_NAME[fixture_name]
        if not applies(fixture, grain):
            # ASV skips the (param) combo when setup raises NotImplementedError.
            raise NotImplementedError
        self.fixture = fixture

    def time_grain(self, fixture_name: str, grain: str) -> None:
        run_grain(self.fixture, grain)
```

- [ ] **Step 2: Smoke-run a single benchmark via ASV (quick mode)**

Run: `cd benchmarks && uv run asv run --quick --bench 'Layers.time_grain.*basic_iit4_2023.*sia' --python=same`
Expected: ASV discovers the benchmark, runs one iteration, reports a time (no traceback). `--python=same` reuses the current env so no project rebuild is needed.

If ASV reports `0 benchmarks` or an import error, fix the `sys.path` shim in `_fixtures.py` before continuing (this is the main integration risk).

- [ ] **Step 3: Commit**

```bash
git add benchmarks/benchmarks/layers.py
git commit -m "Add layered wall-time benchmarks over the golden zoo

Layers benchmarks every golden fixture x {repertoires, mechanism_mips,
phi_structure, sia}, skipping grains that do not apply (3.0 has no
phi_structure; fixture skip_layers honored).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012dtSzF2YgDjGpFC9mA47ve"
```

---

### Task 4: Edge, micro, and AC benchmarks

The concerns the layered grid cannot see: parallel speedup, cache sensitivity, the EMD kernel, and Actual Causation.

**Files:**
- Create: `benchmarks/benchmarks/edges.py`, `benchmarks/benchmarks/micro.py`, `benchmarks/benchmarks/actual_causation.py`

**Interfaces:**
- Consumes: `_fixtures.FIXTURES_BY_NAME`, `run_grain`; `pyphi.config`, `pyphi.System`, `pyphi.examples`, `pyphi.actual`, `pyphi.conf.presets`.

- [ ] **Step 1: Write `edges.py` (parallel/sequential + cold/warm cache)**

```python
# benchmarks/benchmarks/edges.py
"""Edge benchmarks: parallelism and cache sensitivity (wall-time only)."""

from __future__ import annotations

from _fixtures import FIXTURES_BY_NAME, build_system

from pyphi import config

# A mid-size fixture: large enough that parallel/cache effects show, small
# enough to stay under the nightly timeout.
_MID = "rule110_iit4_2023"


class ParallelSia:
    params = [True, False]
    param_names = ("parallel",)
    timeout = 600.0
    number = 1

    def setup(self, parallel: bool) -> None:
        self.fixture = FIXTURES_BY_NAME[_MID]

    def time_sia(self, parallel: bool) -> None:
        with self.fixture.config_context(), config.override(parallel=parallel):
            build_system(self.fixture).sia()


class RepertoireCache:
    params = [True, False]
    param_names = ("warm",)
    timeout = 600.0
    number = 1

    def setup(self, warm: bool) -> None:
        self.fixture = FIXTURES_BY_NAME[_MID]

    def time_repertoires(self, warm: bool) -> None:
        with self.fixture.config_context(), config.override(
            cache_repertoires=True
        ):
            system = build_system(self.fixture)
            if warm:
                system.sia()  # prime the repertoire cache
            for mechanism in system.node_indices:
                for purview in system.node_indices:
                    system.cause_repertoire((mechanism,), (purview,))
                    system.effect_repertoire((mechanism,), (purview,))
```

- [ ] **Step 2: Write `micro.py` (EMD distance kernel)**

```python
# benchmarks/benchmarks/micro.py
"""Micro-benchmark: the EMD distance kernel (POT backend)."""

from __future__ import annotations

import numpy as np

from pyphi.measures.distribution import hamming_emd


class Emd:
    params = [4, 8, 16]
    param_names = ("n_states",)

    def setup(self, n_states: int) -> None:
        rng = np.random.default_rng(2026)
        self.p = rng.dirichlet(np.ones(n_states))
        self.q = rng.dirichlet(np.ones(n_states))

    def time_emd(self, n_states: int) -> None:
        hamming_emd(self.p, self.q)
```

Note: the EMD kernels live in `pyphi.measures.distribution` (`emd`, `hamming_emd`, `effect_emd`). `hamming_emd(p, q)` is the cause-direction IIT 3.0 inner loop. Confirm its arity during the Step 4 smoke run; if `hamming_emd` needs reshaped inputs, switch to `emd(p, q)` (the general distance) — verify with `uv run python -c "from pyphi.measures.distribution import hamming_emd, emd; help(hamming_emd)"`.

- [ ] **Step 3: Write `actual_causation.py` (AC account)**

```python
# benchmarks/benchmarks/actual_causation.py
"""Actual Causation benchmark: the account of a canonical transition."""

from __future__ import annotations

from pyphi import actual, config, examples
from pyphi.conf import presets


class Account:
    timeout = 600.0
    number = 1

    def setup(self) -> None:
        self.ctx = config.override(**presets.iit3)
        self.ctx.__enter__()
        self.transition = actual.Transition(
            examples.actual_causation_substrate(),
            (1, 0),
            (1, 0),
            (0, 1),
            (0, 1),
        )

    def teardown(self) -> None:
        self.ctx.__exit__(None, None, None)

    def time_account(self) -> None:
        actual.account(self.transition)
```

- [ ] **Step 4: Smoke-run one benchmark from each new module**

Run:
```bash
cd benchmarks && \
uv run asv run --quick --python=same --bench 'ParallelSia' --bench 'Emd' --bench 'Account'
```
Expected: each runs without traceback and reports a time. Fix the EMD symbol (Step 2 note) if `Emd` import-errors.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/benchmarks/edges.py benchmarks/benchmarks/micro.py benchmarks/benchmarks/actual_causation.py
git commit -m "Add edge, EMD micro, and Actual Causation benchmarks

ParallelSia (parallel vs sequential) and RepertoireCache (cold vs warm)
cover concerns the layered grid cannot see; Emd benchmarks the POT distance
kernel; Account covers the AC formalism the IIT golden zoo omits.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012dtSzF2YgDjGpFC9mA47ve"
```

---

### Task 5: Full-zoo deterministic count metrics (ASV `track_*`)

Track the exact call counts of hot frames for every fixture × grain as ASV metrics. Deterministic, so step-detection over history is exact — full-zoo count coverage with no in-repo pins.

**Files:**
- Create: `benchmarks/benchmarks/counts.py`

**Interfaces:**
- Consumes: `_fixtures` (re-exports `FIXTURES_BY_NAME`, `GRAINS`, `applies`, `run_grain`, `count_calls`, `FRAMES` from `test.golden.perf`).
- Produces: a `Counts` ASV class with one `track_*` per hot frame, `unit = "calls"`.

- [ ] **Step 1: Write the module**

```python
# benchmarks/benchmarks/counts.py
"""Deterministic call-count metrics over the full zoo (ASV track_*).

Counts are exact, so ASV step-detection flags any change without false
positives — full-zoo count-regression coverage with no in-repo pins.
"""

from __future__ import annotations

from functools import partial

from _fixtures import (
    FIXTURES_BY_NAME,
    FRAMES,
    GRAINS,
    applies,
    count_calls,
    run_grain,
)


class Counts:
    params = (sorted(FIXTURES_BY_NAME), list(GRAINS))
    param_names = ("fixture", "grain")
    timeout = 600.0

    def setup(self, fixture_name: str, grain: str) -> None:
        fixture = FIXTURES_BY_NAME[fixture_name]
        if not applies(fixture, grain):
            raise NotImplementedError
        self.counts = count_calls(partial(run_grain, fixture, grain), FRAMES)

    def track_find_mip(self, fixture_name: str, grain: str) -> int:
        return self.counts["system.py:find_mip"]

    track_find_mip.unit = "calls"  # type: ignore[attr-defined]

    def track_relations(self, fixture_name: str, grain: str) -> int:
        return self.counts["relations.py:relations"]

    track_relations.unit = "calls"  # type: ignore[attr-defined]

    def track_config_override(self, fixture_name: str, grain: str) -> int:
        return self.counts["conf/:override"]

    track_config_override.unit = "calls"  # type: ignore[attr-defined]
```

Note: the dict keys (`"system.py:find_mip"`, etc.) are `f"{sub}:{func}"` from the canonical `FRAMES` in `test/golden/perf.py`. If Task 7's tuning changes a `FRAMES` pair, update the matching `track_*` key here.

- [ ] **Step 2: Smoke-run one count metric**

Run: `cd benchmarks && uv run asv run --quick --python=same --bench 'Counts.track_find_mip.*basic_iit4_2023.*mechanism_mips'`
Expected: runs, reports an integer call count > 0 (no traceback).

- [ ] **Step 3: Commit**

```bash
git add benchmarks/benchmarks/counts.py
git commit -m "Track deterministic hot-frame call counts over the full zoo

Counts runs every fixture x grain under cProfile and exposes exact call
counts (find_mip, relations, config override) as ASV track_* metrics.
Deterministic counts make nightly step-detection exact, giving full-zoo
count-regression coverage with no in-repo pins.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012dtSzF2YgDjGpFC9mA47ve"
```

---

### Task 6: Point ASV at the 2.0 branch

**Files:**
- Modify: `benchmarks/asv.conf.json` (the `branches` line)

- [ ] **Step 1: Change the branch**

In `benchmarks/asv.conf.json`, change:
```json
    "branches": ["develop"], // for git
```
to:
```json
    "branches": ["2.0"], // for git
```

- [ ] **Step 2: Verify ASV reads the config**

Run: `cd benchmarks && uv run asv check --python=same`
Expected: ASV reports the benchmark suite is importable with no errors (`Imported benchmark suite` / no tracebacks).

- [ ] **Step 3: Commit**

```bash
git add benchmarks/asv.conf.json
git commit -m "Point ASV benchmark config at the 2.0 branch

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012dtSzF2YgDjGpFC9mA47ve"
```

---

### Task 7: Count regeneration script + tune frames + pin the subset

Discover the real hot-frame identities against live stats, finalize `FRAMES`, and write the pinned counts for the bounded gate subset.

**Files:**
- Create: `scripts/gen_perf_counts.py`
- Create: `test/data/perf/call_counts.json`
- Modify (if frame names need correcting): `benchmarks/benchmarks/counts.py` (`FRAMES`)

**Interfaces:**
- Consumes: `test.golden.perf` (`FIXTURES_BY_NAME`, `FRAMES`, `count_calls`, `run_grain`).
- Produces: `test/data/perf/call_counts.json` mapping `"<fixture>::<grain>"` → `{frame_key: count}`.

The **gate subset** (bounded for speed) is:
```
basic_iit3_emd        :: sia
basic_iit4_2023       :: sia
basic_iit4_2026       :: sia
basic_iit4_2023       :: mechanism_mips
basic_iit4_2023       :: repertoires
basic_iit4_2023       :: phi_structure
multivalued_k3_tiny_iit4_2023 :: sia   (k-ary)
rule110_iit4_2023     :: phi_structure (relations-heavy)
```
AC is pinned separately (Task 8 includes the AC entry).

- [ ] **Step 1: Discover real frame names**

Run:
```bash
uv run python -c "
import cProfile, pstats, io
from test.golden.zoo import ALL_FIXTURES
from test.golden.compute import _compute_sia
from pyphi import System
fx = {f.name: f for f in ALL_FIXTURES}['basic_iit4_2023']
with fx.config_context():
    sub = fx.build_substrate(); system = System(sub, fx.state, fx.node_indices or sub.node_indices)
    pr = cProfile.Profile(); pr.enable(); _compute_sia(system, lambda a: '', 4.0); pr.disable()
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('ncalls').print_stats(25); print(s.getvalue())
"
```
Expected: a table of the 25 most-called frames. Read off the real `(filename, funcname)` for: the MIP search, cause/effect repertoire, relations enumeration, and config access (look for `override`, `__getattribute__`, or a `conf/` frame with a high count). Update `FRAMES` in both `benchmarks/benchmarks/counts.py` and the script below to the **verified** `(file_substring, funcname)` pairs. Use a `file_substring` specific enough to avoid accidental matches (e.g. `"conf/"` for config, `"relations.py"` for relations).

- [ ] **Step 2: Write the regeneration script**

```python
# scripts/gen_perf_counts.py
"""Regenerate test/data/perf/call_counts.json (the perf-counter gate pins).

Run after a deliberate algorithm change that alters call structure:

    uv run python scripts/gen_perf_counts.py

Review the JSON diff exactly like a phi golden.
"""

from __future__ import annotations

import json
from pathlib import Path

from test.golden.perf import FIXTURES_BY_NAME, FRAMES, count_calls, run_grain

# (fixture_name, grain) — the bounded, fast gate subset.
GATE_SUBSET = [
    ("basic_iit3_emd", "sia"),
    ("basic_iit4_2023", "sia"),
    ("basic_iit4_2026", "sia"),
    ("basic_iit4_2023", "mechanism_mips"),
    ("basic_iit4_2023", "repertoires"),
    ("basic_iit4_2023", "phi_structure"),
    ("multivalued_k3_tiny_iit4_2023", "sia"),
    ("rule110_iit4_2023", "phi_structure"),
]

_OUT = Path(__file__).resolve().parents[1] / "test" / "data" / "perf" / "call_counts.json"


def main() -> None:
    pins: dict[str, dict[str, int]] = {}
    for name, grain in GATE_SUBSET:
        fixture = FIXTURES_BY_NAME[name]
        counts = count_calls(lambda: run_grain(fixture, grain), FRAMES)  # noqa: B023
        pins[f"{name}::{grain}"] = counts
        print(f"{name}::{grain}: {counts}")
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(pins, indent=2, sort_keys=True) + "\n")
    print(f"wrote {_OUT}")


if __name__ == "__main__":
    main()
```

Note: `run_grain` already enters the fixture config context, so the script does not wrap it again. The `# noqa: B023` silences ruff's loop-variable-closure warning — the lambda is called immediately inside the loop iteration, so the late-binding footgun does not apply.

- [ ] **Step 3: Generate the pins**

Run: `uv run python scripts/gen_perf_counts.py`
Expected: prints one line per subset entry with non-zero counts, then `wrote .../call_counts.json`. Inspect the JSON — every entry should have plausible counts (find_mip > 0 for sia/mechanism_mips grains; relations > 0 for phi_structure/sia on 4.0; config override counts small and bounded, **not** thousands — a huge override count is exactly the regression class this gates).

- [ ] **Step 4: Commit**

```bash
git add scripts/gen_perf_counts.py test/data/perf/call_counts.json benchmarks/benchmarks/counts.py
git commit -m "Add perf-count regeneration script and pinned gate counts

gen_perf_counts.py profiles a bounded, fast subset (each formalism x layer
+ k-ary + relations-heavy) and writes exact call-count pins to
test/data/perf/call_counts.json, regenerated deliberately like a phi
golden. FRAMES tuned against live cProfile stats.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012dtSzF2YgDjGpFC9mA47ve"
```

---

### Task 8: The blocking PR gate

Assert the pinned counts (plus an AC entry) as a normal pytest that fails CI on any change.

**Files:**
- Modify: `test/test_perf_counters.py` (add the gate, keep the Task 1 helper test)
- Modify (regen): `test/data/perf/call_counts.json` (add the AC entry)

**Interfaces:**
- Consumes: `test.golden.profiling.count_calls`, `test/data/perf/call_counts.json`.

- [ ] **Step 1: Add the AC pin to the regen script and regenerate**

In `scripts/gen_perf_counts.py`, after the `GATE_SUBSET` loop in `main()`, add an AC entry:
```python
    # Actual Causation gate entry (outside the golden zoo).
    from pyphi import actual, config, examples
    from pyphi.conf import presets

    def _run_ac() -> None:
        transition = actual.Transition(
            examples.actual_causation_substrate(), (1, 0), (1, 0), (0, 1), (0, 1)
        )
        actual.account(transition)

    with config.override(**presets.iit3):
        pins["actual_causation::account"] = count_calls(_run_ac, FRAMES)
        print(f"actual_causation::account: {pins['actual_causation::account']}")
```
Run: `uv run python scripts/gen_perf_counts.py` and confirm the AC entry appears in the JSON.

- [ ] **Step 2: Write the failing gate test**

Add the gate to `test/test_perf_counters.py`. **Move all imports to the top of the file** (ruff E402 rejects imports below the first function); the final file is: module docstring → `from __future__ import annotations` → the imports below → `_helper` + helper test (Task 1) → the gate tests below.
```python
import json
from pathlib import Path

import pytest

from pyphi import actual, config, examples
from pyphi.conf import presets
from test.golden.perf import FIXTURES_BY_NAME, FRAMES, count_calls, run_grain

_PINS = json.loads(
    (Path(__file__).parent / "data" / "perf" / "call_counts.json").read_text()
)
_GOLDEN_KEYS = [k for k in _PINS if k != "actual_causation::account"]


@pytest.mark.parametrize("key", _GOLDEN_KEYS)
def test_call_counts_pinned(key: str) -> None:
    name, grain = key.split("::")
    fixture = FIXTURES_BY_NAME[name]
    counts = count_calls(lambda: run_grain(fixture, grain), FRAMES)
    assert counts == _PINS[key], (
        f"{key} call counts changed from the pins. If this is a deliberate "
        f"algorithm change, regenerate: uv run python scripts/gen_perf_counts.py"
    )


def test_call_counts_pinned_actual_causation() -> None:
    with config.override(**presets.iit3):
        transition = actual.Transition(
            examples.actual_causation_substrate(), (1, 0), (1, 0), (0, 1), (0, 1)
        )
        counts = count_calls(lambda: actual.account(transition), FRAMES)
    assert counts == _PINS["actual_causation::account"], (
        "AC account call counts changed from the pins. If deliberate, "
        "regenerate: uv run python scripts/gen_perf_counts.py"
    )
```

- [ ] **Step 3: Run the gate — expect PASS (pins were just generated)**

Run: `uv run pytest test/test_perf_counters.py -v`
Expected: all parametrized cases + AC + the helper test PASS (the pins match because they were generated from the same code).

To prove the gate actually catches a regression, temporarily edit one pinned value in `test/data/perf/call_counts.json`, rerun — expect that one case to FAIL with the regenerate hint — then restore the value.

- [ ] **Step 4: Mark the gate's slow cases**

If `rule110_iit4_2023::phi_structure` or `multivalued_k3_tiny_iit4_2023::sia` push the gate's wall time over ~30s total, add `@pytest.mark.slow` selectively by splitting the parametrization (fast subset unmarked, slow keys marked) so the default PR lane stays fast. Check with:
Run: `uv run pytest test/test_perf_counters.py --durations=0 -q`
Expected: total under ~30s; if not, split as described.

- [ ] **Step 5: Commit**

```bash
git add test/test_perf_counters.py test/data/perf/call_counts.json scripts/gen_perf_counts.py
git commit -m "Add blocking perf-counter regression gate

test_perf_counters asserts exact pinned call counts for a bounded subset
(each formalism x layer + k-ary + relations-heavy + AC). Deterministic and
fast; fails CI on any call-count change with a regenerate hint, catching the
redundant-work class of regression (e.g. config.override per partition).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012dtSzF2YgDjGpFC9mA47ve"
```

---

### Task 9: Nightly ASV workflow

A scheduled GitHub Actions workflow that accumulates ASV results across runs and alerts on a detected step.

**Files:**
- Create: `.github/workflows/benchmark.yml`

**Interfaces:**
- Consumes: the benchmark suite; the `2.0` branch; persisted results via a `benchmark-results` orphan branch.

- [ ] **Step 1: Read an existing workflow for house style**

Run: `cat .github/workflows/test.yml`
Note the checkout action version, the uv setup step, and the Python version pin so this workflow matches.

- [ ] **Step 2: Write the workflow**

```yaml
# .github/workflows/benchmark.yml
name: benchmark

on:
  schedule:
    - cron: "0 7 * * *"  # nightly, 07:00 UTC
  workflow_dispatch: {}    # manual trigger for verification

permissions:
  contents: write   # push accumulated results to the benchmark-results branch
  issues: write     # open/update the regression-alert issue

jobs:
  asv:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0   # asv needs history to benchmark commits

      - name: Install uv
        uses: astral-sh/setup-uv@v5

      - name: Set up Python
        run: uv python install 3.13

      - name: Install asv
        run: uv tool install asv

      - name: Restore accumulated results
        run: |
          git fetch origin benchmark-results || true
          if git rev-parse --verify origin/benchmark-results >/dev/null 2>&1; then
            git worktree add benchmarks/.results origin/benchmark-results
            cp -r benchmarks/.results/results/* benchmarks/results/ 2>/dev/null || true
          fi

      - name: Configure asv machine
        working-directory: benchmarks
        run: uv tool run asv machine --yes

      - name: Run benchmarks for new commits
        working-directory: benchmarks
        run: uv tool run asv run NEW --skip-existing --show-stderr || true

      - name: Detect regressions
        id: detect
        working-directory: benchmarks
        run: |
          uv tool run asv publish 2>/dev/null || true
          # `asv compare` against the previous benchmarked commit; non-zero exit
          # on a regression beyond the configured factor.
          if uv tool run asv continuous --factor 1.5 HEAD~1 HEAD --show-stderr; then
            echo "regressed=false" >> "$GITHUB_OUTPUT"
          else
            echo "regressed=true" >> "$GITHUB_OUTPUT"
          fi

      - name: Persist accumulated results
        working-directory: benchmarks
        run: |
          git config user.name "pyphi-bench"
          git config user.email "noreply@github.com"
          git add results
          git stash || true
          git switch --orphan benchmark-results 2>/dev/null || git switch benchmark-results
          git stash pop || true
          git add results && git commit -m "asv results $(date -u +%F)" || true
          git push origin benchmark-results || true

      - name: Open/update regression issue
        if: steps.detect.outputs.regressed == 'true'
        uses: actions/github-script@v7
        with:
          script: |
            const title = "Nightly benchmark regression detected";
            const body = "ASV step-detection flagged a wall-time regression on `2.0`. See the workflow run: " + context.serverUrl + "/" + context.repo.owner + "/" + context.repo.repo + "/actions/runs/" + context.runId;
            const issues = await github.rest.issues.listForRepo({owner: context.repo.owner, repo: context.repo.repo, state: "open", labels: "benchmark-regression"});
            if (issues.data.length) {
              await github.rest.issues.createComment({owner: context.repo.owner, repo: context.repo.repo, issue_number: issues.data[0].number, body});
            } else {
              await github.rest.issues.create({owner: context.repo.owner, repo: context.repo.repo, title, body, labels: ["benchmark-regression"]});
            }
```

Note: the persisted-results plumbing (orphan `benchmark-results` branch) is the pragmatic "accumulate over runs" mechanism from the spec. If the orphan-branch dance proves fragile in review, the documented fallback is to store `benchmarks/results/` as an `actions/upload-artifact`/`download-artifact` pair instead — same intent, simpler git handling. Either way the workflow is advisory and never blocks PRs.

- [ ] **Step 3: Lint the workflow YAML**

Run: `uv run python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/benchmark.yml')); print('yaml ok')"`
Expected: `yaml ok`.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/benchmark.yml
git commit -m "Add nightly ASV benchmark workflow with regression alerting

Scheduled workflow accumulates ASV results across runs (benchmark-results
branch), runs new commits, and opens/updates a tracking issue on a detected
wall-time step. Advisory only; never runs on PRs. Verify via workflow_dispatch.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012dtSzF2YgDjGpFC9mA47ve"
```

Note: the workflow cannot be fully verified until pushed (it needs the GitHub scheduler / `workflow_dispatch`). The plan's pre-push verification is the YAML lint (Step 3) plus a manual `workflow_dispatch` run after the branch is pushed — flagged for the human at execution-finish, not gated here.

---

### Task 10: Bookkeeping — ROADMAP + changelog

**Files:**
- Modify: `ROADMAP.md` (dashboard row 42; P15 "Layer D" note)
- Create: `changelog.d/benchmark-suite-asv.misc.md`

- [ ] **Step 1: Update the dashboard row**

In `ROADMAP.md`, change the P11.8 Tier 2 dashboard row (currently `| P11.8 Tier 2 | ⬜ open | 3 | Rewrite benchmark suite + ASV-in-CI (regression gate *before* the perf work) |`) status to `✅ landed` and append a one-line summary of what shipped (golden-harness-driven ASV suite, nightly accumulate+step-detect wall-time trend, blocking cProfile call-count gate, full-zoo count metrics). Also locate the P15 "Layer D" reference (archive search `grep -n "Layer D" ROADMAP.md`) and mark it superseded by P11.8 Tier 2.

- [ ] **Step 2: Move the P11.8 Tier 2 detail to landed prose**

In the "Remaining 2.0 Work" Wave 3 section, update the P11.8 Tier 2 bullet to past-tense "landed" with the same summary, mirroring how other landed items read.

- [ ] **Step 3: Write the changelog fragment**

```bash
cat > changelog.d/benchmark-suite-asv.misc.md <<'EOF'
Rebuilt the developer benchmark suite for the 2.0 architecture. ASV
benchmarks now run off the golden-regression fixtures (every fixture across
the three formalisms, layered as repertoires / mechanism MIPs / phi-structure
/ SIA, plus parallel, cache, EMD, and Actual Causation benchmarks). A nightly
workflow accumulates results and alerts on wall-time regressions, and a
deterministic cProfile call-count gate (`test/test_perf_counters.py`) fails CI
on any change to hot-path call counts.
EOF
```

- [ ] **Step 4: Commit**

```bash
git add ROADMAP.md changelog.d/benchmark-suite-asv.misc.md
git commit -m "Record P11.8 Tier 2 benchmark suite + ASV-in-CI as landed

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012dtSzF2YgDjGpFC9mA47ve"
```

---

### Task 11: Full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full gate (no path argument)**

Run: `uv run pytest -q`
Expected: green (the new `test/test_perf_counters.py` included; doctest sweep runs). Note the pass/skip counts.

- [ ] **Step 2: Refresh the graphify structural layer**

Run: `graphify update . 2>&1 | tail -3`
Expected: "Code graph updated." Do **not** commit `graphify-out/` (shared artifact; concurrent instances).

- [ ] **Step 3: Final ASV discovery check**

Run: `cd benchmarks && uv run asv check --python=same`
Expected: all benchmark modules import cleanly.

- [ ] **Step 4: Report**

Summarize: commits landed, full-suite tally, and the two items requiring a human after push — (a) a `workflow_dispatch` run of `benchmark.yml` to verify the nightly end-to-end, (b) confirming the `benchmark-results` branch is created on first run. These cannot be verified pre-push.

---

## Notes for the executor

- **Ask before pushing.** This branch is shared; pushing (and the first `workflow_dispatch`) needs explicit user consent per repo rules.
- **ASV + uv friction.** `asv check`/`asv run --python=same` reuse the current environment and avoid ASV building its own venv — use them for all local smoke checks. Full `asv run` (building isolated envs) is exercised only in the nightly workflow.
- **If `test.golden` won't import under ASV** (Task 3 Step 2): confirm `_REPO_ROOT = parents[2]` resolves to the repo root from `benchmarks/benchmarks/_fixtures.py`, and that the ASV checkout includes `test/`. This is the single biggest integration risk.
- **Frame tuning (Task 7 Step 1) is load-bearing.** The placeholder `FRAMES` names must be replaced with verified `(file_substring, funcname)` pairs, kept identical across `counts.py`, `gen_perf_counts.py`, and `test_perf_counters.py`. A frame that matches nothing silently pins `0` — check every pinned frame is non-zero where it should be.
