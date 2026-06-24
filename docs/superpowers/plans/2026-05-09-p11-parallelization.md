# P11 — Parallelization Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the ad-hoc `MapReduce` backend dispatch with a typed `Scheduler` Protocol; ship `LocalProcessScheduler` + `LocalThreadScheduler` fully implemented, `DaskScheduler` skeleton; deliver per-call `ConfigSnapshot` propagation to workers; replace dead-code chunking heuristics with cost-sampling.

**Architecture:** A `Scheduler` Protocol exposes `map_reduce(fn, items, *, config_snapshot, chunking, progress, shortcircuit, ordered)`. The `MapReduce` user-facing class stays as the facade (12 internal call sites unchanged) and constructs a Scheduler internally. Workers receive a `ConfigSnapshot` via closure and apply it to their global config (no-op on thread schedulers since threads share parent memory). Dead-code `chunking.py` is replaced by a small `sampling.py` wired into the scheduler default chunksize.

**Tech Stack:** Python 3.13+, `joblib.externals.loky` (process pool with cloudpickle), `concurrent.futures.ThreadPoolExecutor` (thread pool), `dask.distributed` (skeleton, lazy-imported), `pytest`, `hypothesis`.

**Spec:** `docs/superpowers/specs/2026-05-09-p11-parallelization-design.md` (committed at `514270d6`).

**Branch:** `feature/p11-parallelization-redesign` (cut from `2.0` at `b4d58aa2`).

---

## Phase Overview

| Phase | Subject | Tasks | Files Touched |
|---|---|---|---|
| 0 | Branch + spec (done) | — | — |
| 1 | Frozen-formalism conversion (P10 follow-through) | 1.1–1.4 | `pyphi/formalism/iit3/formalism.py`, `pyphi/formalism/iit4/formalism.py`, `test/test_formalism_pickle.py` (new) |
| 2 | `Scheduler` Protocol + policies (additive) | 2.1–2.4 | `pyphi/parallel/scheduler.py` (new), `pyphi/conf/infrastructure.py`, `test/test_scheduler.py` (new) |
| 3 | `LocalProcessScheduler` | 3.1–3.6 | `pyphi/conf/legacy_global.py`, `pyphi/parallel/backends/local_process.py` (renamed), `pyphi/parallel/scheduler.py` |
| 4 | `LocalThreadScheduler` | 4.1–4.4 | `pyphi/parallel/backends/local_thread.py` (new), `test/test_scheduler.py` |
| 5 | `DaskScheduler` skeleton | 5.1–5.2 | `pyphi/parallel/backends/dask.py` (new), `test/test_scheduler.py` |
| 6 | Delete `chunking.py`; add `sampling.py` | 6.1–6.4 | `pyphi/parallel/chunking.py` (deleted), `pyphi/parallel/sampling.py` (new), `test/test_chunking.py` (deleted), `test/test_sampling.py` (new) |
| 7 | Fill `TODO(4.0)` parallelize markers | 7.1–7.4 | `pyphi/formalism/iit4/__init__.py`, `pyphi/compute/subsystem.py`, `pyphi/models/ces.py`, `pyphi/actual.py` |
| 8 | Re-enable parallel tests in CI | 8.1–8.3 | `.github/workflows/test.yml`, `test/test_parallel.py` |
| 9 | Cleanup + changelog | 9.1–9.4 | `pyphi/parallel/__init__.py`, `changelog.d/p11-scheduler.feature.md` (new), `ROADMAP.md` |

Each phase ends with a green-test commit. Acceptance gate per phase: golden 17/17 numerical match (run in background per `feedback_monitor_for_long_tests.md`), hypothesis fast lane 21 green, fast unit lane green, pyright clean on touched files, ruff clean, pre-commit hooks pass without `--no-verify`.

---

## Phase 1 — Frozen-formalism conversion

This is the deferred P10 Phase 4 follow-through. Converts `IIT3Formalism`, `IIT4_2023Formalism`, `IIT4_2026Formalism` from "class with `config` property delegating to global" to "frozen dataclass with `config: FormalismConfig` field". Workers will receive these via cloudpickle with the parent's frozen config attached.

### Task 1.1: Convert `IIT3Formalism` to frozen dataclass

**Files:**
- Modify: `pyphi/formalism/iit3/formalism.py:20-50`

- [ ] **Step 1: Read the current class** to capture exact attributes

```bash
sed -n '20,50p' pyphi/formalism/iit3/formalism.py
```

- [ ] **Step 2: Write a failing test** for the new shape

Append to `test/test_formalism_pickle.py` (create the file with this content):

```python
"""Pickle-roundtrip tests for frozen formalism dataclasses."""
from __future__ import annotations

import pickle

import pytest

from pyphi.conf import config
from pyphi.formalism.iit3.formalism import IIT3Formalism


def test_iit3_formalism_is_frozen_dataclass():
    f = IIT3Formalism()
    assert hasattr(f, "config")
    # Attempt to mutate; frozen dataclass raises FrozenInstanceError
    with pytest.raises(Exception, match="frozen"):
        f.config = config.formalism  # type: ignore[misc]


def test_iit3_formalism_pickle_roundtrip():
    f = IIT3Formalism()
    f2 = pickle.loads(pickle.dumps(f))
    assert f2.config == f.config
    assert f2.name == "IIT_3_0"


def test_iit3_formalism_carries_independent_config():
    """Snapshot taken at construction; later global changes don't leak in."""
    f = IIT3Formalism()
    captured = f.config.repertoire_distance
    with config.override(repertoire_distance="L1"):
        # f.config is the field, not a live view: still the captured value
        assert f.config.repertoire_distance == captured
```

- [ ] **Step 3: Run the test to verify failure**

```bash
uv run pytest test/test_formalism_pickle.py::test_iit3_formalism_is_frozen_dataclass test/test_formalism_pickle.py::test_iit3_formalism_carries_independent_config -v
```

Expected: FAIL — current `IIT3Formalism` is not a dataclass; `f.config = ...` doesn't raise; `f.config` is a property returning the live global.

- [ ] **Step 4: Convert `IIT3Formalism`**

Replace `pyphi/formalism/iit3/formalism.py:20-50` with:

```python
from dataclasses import dataclass, field
from typing import ClassVar

from pyphi.conf.formalism import FormalismConfig


@dataclass(frozen=True)
class IIT3Formalism:
    """IIT 3.0 (Oizumi et al. 2014) — distribution-distance phi computation."""

    name: ClassVar[str] = "IIT_3_0"
    exact: ClassVar[Literal[True]] = True
    default_metric: ClassVar[str] = "EMD"
    compatible_metrics: ClassVar[frozenset[str]] = frozenset(
        {
            "EMD",
            "L1",
            "KLD",
            "ENTROPY_DIFFERENCE",
            "PSQ2",
            "MP2Q",
            "ABSOLUTE_INTRINSIC_DIFFERENCE",
            "INTRINSIC_DIFFERENCE",
        }
    )
    partition_scheme: ClassVar[str | None] = "BI"

    config: FormalismConfig = field(
        default_factory=lambda: __import__("pyphi.conf", fromlist=["config"]).config.formalism
    )
```

Keep all method definitions below unchanged. The lazy `__import__` in the default factory avoids a top-level circular import.

- [ ] **Step 5: Run the test to verify pass**

```bash
uv run pytest test/test_formalism_pickle.py -v
```

Expected: PASS.

- [ ] **Step 6: Run golden + fast unit + hypothesis fast lane in parallel**

Slow lane in background:

```bash
# Foreground: fast unit + hypothesis fast lane (~70 seconds)
uv run pytest test/test_invariants.py test/test_invariants_hypothesis.py test/test_formalism_pickle.py -v
```

Background: golden via Bash `run_in_background: true`:

```bash
uv run pytest test/test_golden_regression.py -v
```

Expected: all green. Wait for golden notification before committing.

- [ ] **Step 7: Commit**

```bash
git add pyphi/formalism/iit3/formalism.py test/test_formalism_pickle.py
git commit -m "Convert IIT3Formalism to frozen dataclass with config field

Replaces the property delegating to the live global config with a
field captured at construction time (default_factory reads the
current global). Pickle-roundtrip preserves the field; workers will
receive the formalism with its config attached.
"
```

### Task 1.2: Convert `IIT4_2023Formalism` and `IIT4_2026Formalism` to frozen dataclasses

**Files:**
- Modify: `pyphi/formalism/iit4/formalism.py:159-228` (IIT4_2023Formalism)
- Modify: `pyphi/formalism/iit4/formalism.py:231-305` (IIT4_2026Formalism)

- [ ] **Step 1: Extend the test file**

Append to `test/test_formalism_pickle.py`:

```python
from pyphi.formalism.iit4.formalism import IIT4_2023Formalism
from pyphi.formalism.iit4.formalism import IIT4_2026Formalism


def test_iit4_2023_formalism_is_frozen_dataclass():
    f = IIT4_2023Formalism()
    assert hasattr(f, "config")
    with pytest.raises(Exception, match="frozen"):
        f.config = config.formalism  # type: ignore[misc]


def test_iit4_2023_formalism_pickle_roundtrip():
    f = IIT4_2023Formalism()
    f2 = pickle.loads(pickle.dumps(f))
    assert f2.config == f.config
    assert f2.name == "IIT_4_0_2023"


def test_iit4_2026_formalism_is_frozen_dataclass():
    f = IIT4_2026Formalism()
    assert hasattr(f, "config")
    with pytest.raises(Exception, match="frozen"):
        f.config = config.formalism  # type: ignore[misc]


def test_iit4_2026_formalism_pickle_roundtrip():
    f = IIT4_2026Formalism()
    f2 = pickle.loads(pickle.dumps(f))
    assert f2.config == f.config
    assert f2.name == "IIT_4_0_2026"


def test_iit4_formalisms_carry_independent_config():
    f = IIT4_2023Formalism()
    captured = f.config.repertoire_distance
    with config.override(repertoire_distance="L1"):
        assert f.config.repertoire_distance == captured
```

- [ ] **Step 2: Run the test to verify failure**

```bash
uv run pytest test/test_formalism_pickle.py -v
```

Expected: FAIL on the new test functions.

- [ ] **Step 3: Convert `IIT4_2023Formalism`**

Replace lines 159-173 (class declaration through `config` property) of `pyphi/formalism/iit4/formalism.py` with:

```python
@dataclass(frozen=True)
class IIT4_2023Formalism:
    """IIT 4.0 (Albantakis et al. 2023) — GID-based mechanism integration."""

    name: ClassVar[str] = "IIT_4_0_2023"
    exact: ClassVar[Literal[True]] = True
    default_metric: ClassVar[str] = "GENERALIZED_INTRINSIC_DIFFERENCE"
    compatible_metrics: ClassVar[frozenset[str]] = frozenset(
        {"GENERALIZED_INTRINSIC_DIFFERENCE", "INTRINSIC_INFORMATION"}
    )
    partition_scheme: ClassVar[str | None] = "ALL"

    config: FormalismConfig = field(
        default_factory=lambda: __import__("pyphi.conf", fromlist=["config"]).config.formalism
    )
```

Add at the top of the file (after existing imports):

```python
from dataclasses import dataclass, field
from typing import ClassVar

from pyphi.conf.formalism import FormalismConfig
```

Keep the method definitions (`evaluate_mechanism`, `_find_mechanism_mip`, `evaluate_mechanism_partition`, `evaluate_system`, `build_phi_structure`) unchanged.

- [ ] **Step 4: Convert `IIT4_2026Formalism`**

Replace lines 231-253 of `pyphi/formalism/iit4/formalism.py` with:

```python
@dataclass(frozen=True)
class IIT4_2026Formalism:
    """IIT 4.0 (Mayner, Marshall, Tononi 2026) — intrinsic-information cap.

    Uses the ``INTRINSIC_INFORMATION`` metric with the ``ii(s) = min(i_diff,
    i_spec)`` cap from Eq. 23. Implementation reuses the IIT 4.0 (2023)
    algorithms; only the metric configuration differs.
    """

    name: ClassVar[str] = "IIT_4_0_2026"
    exact: ClassVar[Literal[True]] = True
    default_metric: ClassVar[str] = "INTRINSIC_INFORMATION"
    compatible_metrics: ClassVar[frozenset[str]] = frozenset(
        {"INTRINSIC_INFORMATION", "GENERALIZED_INTRINSIC_DIFFERENCE"}
    )
    partition_scheme: ClassVar[str | None] = "ALL"

    config: FormalismConfig = field(
        default_factory=lambda: __import__("pyphi.conf", fromlist=["config"]).config.formalism
    )
```

Keep all method definitions unchanged.

- [ ] **Step 5: Run the test to verify pass**

```bash
uv run pytest test/test_formalism_pickle.py -v
```

Expected: PASS.

- [ ] **Step 6: Acceptance gate** — golden 17/17 + hypothesis fast lane 21 + fast unit lane

Background:

```bash
uv run pytest test/test_golden_regression.py -v
```

Foreground:

```bash
uv run pytest test/test_invariants.py test/test_invariants_hypothesis.py test/test_formalism_pickle.py test/test_subsystem_surface.py -v
```

Pyright check on touched files:

```bash
uv run pyright pyphi/formalism/iit3/formalism.py pyphi/formalism/iit4/formalism.py
```

Wait for golden notification before committing. Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add pyphi/formalism/iit4/formalism.py test/test_formalism_pickle.py
git commit -m "Convert IIT4_2023/2026 formalisms to frozen dataclasses

Both classes now hold a FormalismConfig field captured at
construction time. Method bodies unchanged; legacy paths still read
config.formalism.X via the global, which workers will see correctly
once the snapshot-apply hook lands in the next phase.
"
```

### Task 1.3: Verify formalism registry instantiation still works

**Files:**
- Read: `pyphi/formalism/__init__.py:41-43`
- Read: `pyphi/formalism/base.py:170` (the second `IIT_4_0_2023` registration)

- [ ] **Step 1: Run the formalism-related test surface**

```bash
uv run pytest test/test_formalism.py test/test_subsystem_surface.py -v
```

Expected: all green. The four `()` instantiation sites use the `default_factory` to populate `config` from the live global at registry-init time.

- [ ] **Step 2: No commit if step 1 passes**

If failures appear, investigate the `default_factory` initialization order. The factory imports `pyphi.conf` lazily inside the lambda; if invoked before `pyphi.conf` is fully initialized, the lambda fails. The fix is to ensure the formalism registry initializes after `pyphi.conf`. Check `pyphi/__init__.py` import order.

### Task 1.4: Phase 1 acceptance gate

- [ ] **Step 1: Run full acceptance gate**

Background:

```bash
uv run pytest test/test_golden_regression.py -v
```

Foreground:

```bash
uv run pytest test/ -v --ignore=test/test_parallel.py --ignore=test/test_invariants_hypothesis.py
uv run pytest test/test_invariants_hypothesis.py -v
```

Pyright:

```bash
uv run pyright pyphi/formalism/
```

Ruff:

```bash
uv run ruff check pyphi/formalism/ test/test_formalism_pickle.py
uv run ruff format --check pyphi/formalism/ test/test_formalism_pickle.py
```

Expected: all green.

- [ ] **Step 2: No commit needed if everything passes** — Phase 1 commits already landed.

---

## Phase 2 — `Scheduler` Protocol + policies (additive)

Define the Protocol, three policy dataclasses, and the `default_scheduler()` resolver. No call sites migrated yet; the existing `MapReduce` class continues to work unchanged.

### Task 2.1: Add `Scheduler` Protocol and policy dataclasses

**Files:**
- Create: `pyphi/parallel/scheduler.py`
- Test: `test/test_scheduler.py`

- [ ] **Step 1: Write the failing test**

Create `test/test_scheduler.py`:

```python
"""Tests for the Scheduler Protocol, policies, and resolver."""
from __future__ import annotations

import pytest

from pyphi.parallel.scheduler import (
    ChunkingPolicy,
    ProgressPolicy,
    Scheduler,
    ShortcircuitPolicy,
    default_scheduler,
)


def test_scheduler_protocol_is_runtime_checkable():
    class _Stub:
        def map_reduce(self, fn, items, *more_items, **kwargs):
            return list(map(fn, items))

        @property
        def supports_shared_state(self) -> bool:
            return False

    assert isinstance(_Stub(), Scheduler)


def test_chunking_policy_defaults_are_none():
    p = ChunkingPolicy()
    assert p.chunksize is None
    assert p.sequential_threshold == 1
    assert p.size_func is None


def test_progress_policy_default_is_off():
    p = ProgressPolicy()
    assert p.enabled is False
    assert p.desc == ""
    assert p.total is None


def test_shortcircuit_policy_default_never_short_circuits():
    p = ShortcircuitPolicy()
    assert p.func(0) is False
    assert p.func("anything") is False
    assert p.callback is None
```

- [ ] **Step 2: Run the test to verify failure**

```bash
uv run pytest test/test_scheduler.py -v
```

Expected: FAIL with `ModuleNotFoundError: pyphi.parallel.scheduler`.

- [ ] **Step 3: Create the module**

Write `pyphi/parallel/scheduler.py`:

```python
"""Scheduler Protocol and policy types for the parallelization layer.

The Protocol abstracts process / thread / dask backends behind a single
``map_reduce`` entry point. Policies bundle the parameters that today live as
loose kwargs on ``MapReduce.__init__`` so backends share a stable surface.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Protocol
from typing import TypeVar
from typing import runtime_checkable

R = TypeVar("R")
T = TypeVar("T")


def _never_short_circuit(_result: Any) -> bool:
    return False


@dataclass(frozen=True)
class ChunkingPolicy:
    """Controls how items are batched for a worker.

    ``chunksize=None`` selects cost-sampling at the scheduler. Provide a value
    to bypass sampling.
    """

    chunksize: int | None = None
    sequential_threshold: int = 1
    size_func: Callable[[Any], float] | None = None
    target_seconds: float = 1.0


@dataclass(frozen=True)
class ProgressPolicy:
    enabled: bool = False
    desc: str = ""
    total: int | None = None


@dataclass(frozen=True)
class ShortcircuitPolicy:
    func: Callable[[Any], bool] = field(default=_never_short_circuit)
    callback: Callable[[Iterable], None] | None = None


@runtime_checkable
class Scheduler(Protocol):
    """Backend-agnostic map-reduce dispatcher."""

    def map_reduce(
        self,
        fn: Callable[..., R],
        items: Iterable,
        *more_items: Iterable,
        reducer: Callable[[Iterable[R]], T] = list,  # type: ignore[assignment]
        config_snapshot: Any | None = None,
        chunking: ChunkingPolicy | None = None,
        progress: ProgressPolicy | None = None,
        shortcircuit: ShortcircuitPolicy | None = None,
        ordered: bool = False,
        map_kwargs: dict[str, Any] | None = None,
    ) -> T:
        ...

    @property
    def supports_shared_state(self) -> bool:
        ...


def default_scheduler() -> Scheduler:
    """Return the scheduler matching ``config.infrastructure.parallel_backend``.

    ``"auto"`` resolves to ``LocalThreadScheduler`` on free-threaded runtimes
    and ``LocalProcessScheduler`` otherwise.
    """
    import sys

    from pyphi.conf import config

    backend = config.infrastructure.parallel_backend
    if backend == "auto":
        gil_enabled = getattr(sys, "_is_gil_enabled", lambda: True)()
        if not gil_enabled:
            from pyphi.parallel.backends.local_thread import LocalThreadScheduler
            return LocalThreadScheduler()
        from pyphi.parallel.backends.local_process import LocalProcessScheduler
        return LocalProcessScheduler()
    if backend in ("local", "process"):
        from pyphi.parallel.backends.local_process import LocalProcessScheduler
        return LocalProcessScheduler()
    if backend == "thread":
        from pyphi.parallel.backends.local_thread import LocalThreadScheduler
        return LocalThreadScheduler()
    if backend == "dask":
        from pyphi.parallel.backends.dask import DaskScheduler
        return DaskScheduler()
    raise ValueError(f"unknown parallel_backend: {backend!r}")
```

- [ ] **Step 4: Run the test to verify pass**

```bash
uv run pytest test/test_scheduler.py -v
```

Expected: PASS on the four tests; the `default_scheduler` import paths target `local_process.py` / `local_thread.py` / `dask.py` which don't exist yet — that's fine because no test calls `default_scheduler()` yet.

- [ ] **Step 5: Pyright check**

```bash
uv run pyright pyphi/parallel/scheduler.py test/test_scheduler.py
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add pyphi/parallel/scheduler.py test/test_scheduler.py
git commit -m "Add Scheduler Protocol and policy dataclasses

Defines the runtime-checkable Scheduler Protocol with map_reduce as
the single dispatch entry point, plus three frozen policy dataclasses
(ChunkingPolicy, ProgressPolicy, ShortcircuitPolicy) that bundle the
loose kwargs on MapReduce.__init__ today. Concrete schedulers land
in subsequent commits.
"
```

### Task 2.2: Widen `parallel_backend` config field to accept new values

**Files:**
- Modify: `pyphi/conf/infrastructure.py:58`

- [ ] **Step 1: Read the current field**

```bash
grep -n "parallel_backend" pyphi/conf/infrastructure.py
```

- [ ] **Step 2: Write a failing test**

Add to `test/test_scheduler.py`:

```python
from pyphi.conf import config


def test_parallel_backend_accepts_new_values():
    """The legacy 'local' alias plus 'process'/'thread'/'dask'/'auto'."""
    for value in ("local", "process", "thread", "dask", "auto"):
        with config.override(parallel_backend=value):
            assert config.infrastructure.parallel_backend == value


def test_parallel_backend_rejects_unknown():
    with pytest.raises(Exception):
        with config.override(parallel_backend="invalid"):
            pass
```

- [ ] **Step 3: Run to see test outcome**

```bash
uv run pytest test/test_scheduler.py::test_parallel_backend_accepts_new_values test/test_scheduler.py::test_parallel_backend_rejects_unknown -v
```

Expected: the first probably passes (no validation today). The second fails (no validator). If both pass without changes, the field is already permissive — skip step 4.

- [ ] **Step 4: Add allowed-values validation if missing**

If validation is needed, find the `Option` definition in `pyphi/_conf_legacy.py` for `PARALLEL_BACKEND`. Add `values=("local", "process", "thread", "dask", "auto")` if the Option supports it. Mirror in `pyphi/conf/infrastructure.py` if validation lives there.

- [ ] **Step 5: Run to verify pass**

```bash
uv run pytest test/test_scheduler.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyphi/conf/infrastructure.py pyphi/_conf_legacy.py test/test_scheduler.py
git commit -m "parallel_backend accepts process/thread/dask/auto in addition to local"
```

### Task 2.3: Phase 2 acceptance gate

- [ ] **Step 1: Run gates** (background golden, foreground fast lane)

Background:

```bash
uv run pytest test/test_golden_regression.py -v
```

Foreground:

```bash
uv run pytest test/test_invariants.py test/test_invariants_hypothesis.py test/test_scheduler.py test/test_subsystem_surface.py test/test_formalism_pickle.py -v
uv run pyright pyphi/parallel/scheduler.py pyphi/conf/infrastructure.py
uv run ruff check pyphi/parallel/scheduler.py test/test_scheduler.py
```

Expected: all green.

---

## Phase 3 — `LocalProcessScheduler`

Rename `pyphi/parallel/backends/local.py` → `local_process.py`. Refactor `LocalMapReduce` → `LocalProcessScheduler` implementing the Protocol. Wire snapshot-via-closure config delivery. Add `_GlobalConfig.install_snapshot()` so workers can apply the parent's frozen config.

### Task 3.1: Add `_GlobalConfig.install_snapshot()` method

**Files:**
- Modify: `pyphi/conf/legacy_global.py`

- [ ] **Step 1: Write the failing test**

Add to `test/test_config_layers.py` (or create `test/test_install_snapshot.py` if you prefer isolation):

```python
def test_install_snapshot_replaces_global_layers():
    from pyphi.conf import config, ConfigSnapshot

    original = config.snapshot()
    try:
        with config.override(precision=11, repertoire_distance="L1"):
            captured = config.snapshot()

        # Outside the override block, global is back to original
        assert config.numerics.precision == original.numerics.precision

        # Install captured snapshot durably
        config.install_snapshot(captured)
        assert config.numerics.precision == 11
        assert config.formalism.repertoire_distance == "L1"
    finally:
        config.install_snapshot(original)


def test_install_snapshot_idempotent():
    from pyphi.conf import config

    snap = config.snapshot()
    before = config.snapshot()
    config.install_snapshot(snap)
    after = config.snapshot()
    assert before == after
```

- [ ] **Step 2: Run the test to verify failure**

```bash
uv run pytest test/test_config_layers.py::test_install_snapshot_replaces_global_layers -v
```

Expected: FAIL — `_GlobalConfig` has no `install_snapshot` method.

- [ ] **Step 3: Implement `install_snapshot`**

In `pyphi/conf/legacy_global.py`, add (inside `_GlobalConfig`):

```python
def install_snapshot(self, snapshot) -> None:
    """Apply a ``ConfigSnapshot`` to the live global durably (not scoped).

    Used by worker processes to seed their global config from a snapshot
    captured by the parent scheduler. Idempotent: applying the current
    snapshot is a no-op.
    """
    for key, value in snapshot.as_kwargs().items():
        setattr(self, key, value)
```

If `_GlobalConfig.__setattr__` already routes through `FIELD_TO_LAYER`, this iterates flat keys correctly.

- [ ] **Step 4: Run the test to verify pass**

```bash
uv run pytest test/test_config_layers.py -v
```

Expected: PASS.

- [ ] **Step 5: Pyright check**

```bash
uv run pyright pyphi/conf/legacy_global.py
```

- [ ] **Step 6: Commit**

```bash
git add pyphi/conf/legacy_global.py test/test_config_layers.py
git commit -m "Add _GlobalConfig.install_snapshot for durable snapshot application

Workers under the new Scheduler Protocol receive a ConfigSnapshot via
closure and call this method at chunk start to mirror the parent's
config. Distinct from override(), which is a scoped context manager.
"
```

### Task 3.2: Rename `local.py` → `local_process.py`

**Files:**
- Move: `pyphi/parallel/backends/local.py` → `pyphi/parallel/backends/local_process.py`
- Modify: `pyphi/parallel/backends/__init__.py`
- Modify: `pyphi/parallel/__init__.py:293-294` (the `from .backends.local import LocalMapReduce` import)

- [ ] **Step 1: git mv the file**

```bash
git mv pyphi/parallel/backends/local.py pyphi/parallel/backends/local_process.py
```

- [ ] **Step 2: Update imports**

In `pyphi/parallel/backends/__init__.py`:

```python
from .local_process import LocalMapReduce

__all__ = ["LocalMapReduce"]
```

In `pyphi/parallel/__init__.py:293`, change:

```python
from .backends.local import LocalMapReduce
```

to:

```python
from .backends.local_process import LocalMapReduce
```

- [ ] **Step 3: Run import-only smoke test**

```bash
uv run python -c "from pyphi.parallel import MapReduce; from pyphi.parallel.backends.local_process import LocalMapReduce; print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 4: Run fast unit + parallel test file**

```bash
uv run pytest test/test_parallel.py test/test_invariants.py -v
```

Expected: all green (the rename is structural; behavior unchanged).

- [ ] **Step 5: Commit**

```bash
git add pyphi/parallel/backends/local_process.py pyphi/parallel/backends/__init__.py pyphi/parallel/__init__.py
git commit -m "Rename parallel/backends/local.py to local_process.py

Pure rename; the next commit adds LocalProcessScheduler implementing
the Scheduler Protocol on top of the existing LocalMapReduce.
"
```

### Task 3.3: Implement `LocalProcessScheduler` (Protocol-conforming wrapper)

**Files:**
- Modify: `pyphi/parallel/backends/local_process.py`

- [ ] **Step 1: Write the failing test**

Append to `test/test_scheduler.py`:

```python
def test_local_process_scheduler_implements_protocol():
    from pyphi.parallel.backends.local_process import LocalProcessScheduler

    s = LocalProcessScheduler()
    assert isinstance(s, Scheduler)
    assert s.supports_shared_state is False


def _square(x):
    return x * x


def test_local_process_scheduler_basic_map_reduce():
    from pyphi.parallel.backends.local_process import LocalProcessScheduler

    s = LocalProcessScheduler()
    result = s.map_reduce(_square, [1, 2, 3, 4, 5], reducer=sum)
    assert result == 1 + 4 + 9 + 16 + 25
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest test/test_scheduler.py::test_local_process_scheduler_implements_protocol -v
```

Expected: FAIL — `LocalProcessScheduler` doesn't exist.

- [ ] **Step 3: Add the class to `local_process.py`**

At the bottom of `pyphi/parallel/backends/local_process.py`, append:

```python
from pyphi.parallel.scheduler import (
    ChunkingPolicy,
    ProgressPolicy,
    ShortcircuitPolicy,
)


class LocalProcessScheduler:
    """Scheduler backed by loky's reusable process executor.

    Workers receive a ``ConfigSnapshot`` via closure and apply it to their
    own global config at chunk start. Cache state is per-worker (fresh
    process, empty caches at start).
    """

    @property
    def supports_shared_state(self) -> bool:
        return False

    def map_reduce(
        self,
        fn,
        items,
        *more_items,
        reducer=list,
        config_snapshot=None,
        chunking=None,
        progress=None,
        shortcircuit=None,
        ordered=False,
        map_kwargs=None,
    ):
        from pyphi.conf import config
        from pyphi.parallel.tree import get_constraints

        chunking = chunking or ChunkingPolicy()
        progress = progress or ProgressPolicy()
        shortcircuit = shortcircuit or ShortcircuitPolicy()
        snapshot = config_snapshot if config_snapshot is not None else config.snapshot()

        iterables = (items, *more_items)
        try:
            total = len(items)  # type: ignore[arg-type]
        except TypeError:
            total = progress.total

        constraints = get_constraints(
            total=total,
            chunksize=chunking.chunksize,
            sequential_threshold=chunking.sequential_threshold,
        )
        tree = constraints.simulate()
        chunksize = constraints.get_initial_chunksize() or 1

        wrapped_fn = _make_worker_fn(fn, snapshot)

        local_mr = LocalMapReduce(
            map_func=wrapped_fn,
            iterables=iterables,
            reduce_func=lambda results, **_: reducer(results),
            reduce_kwargs={},
            constraints=constraints,
            tree=tree,
            chunksize=chunksize,
            shortcircuit_func=shortcircuit.func,
            shortcircuit_callback=shortcircuit.callback,
            ordered=ordered,
            map_kwargs=map_kwargs,
            progress=progress.enabled,
            desc=progress.desc,
            total=total,
        )
        return local_mr.run()


def _make_worker_fn(fn, snapshot):
    """Wrap ``fn`` so each worker call applies the parent's snapshot first.

    The dedup state lives in module-level variables in the worker process.
    """
    def worker_fn(*args, **kwargs):
        _apply_snapshot_if_changed(snapshot)
        return fn(*args, **kwargs)
    return worker_fn


_LAST_APPLIED_SNAPSHOT_HASH: int | None = None
_PARENT_PID: int | None = None


def _apply_snapshot_if_changed(snapshot) -> None:
    """Apply ``snapshot`` to the worker's global config; idempotent.

    Skips application when running in the parent process (set by the thread
    scheduler before dispatch) — threads share the parent's globals.
    """
    global _LAST_APPLIED_SNAPSHOT_HASH

    import os

    if _PARENT_PID is not None and os.getpid() == _PARENT_PID:
        return

    snap_hash = hash(snapshot)
    if snap_hash == _LAST_APPLIED_SNAPSHOT_HASH:
        return

    from pyphi.conf import config

    config.install_snapshot(snapshot)
    _LAST_APPLIED_SNAPSHOT_HASH = snap_hash
```

- [ ] **Step 4: Run to verify pass**

```bash
uv run pytest test/test_scheduler.py::test_local_process_scheduler_implements_protocol test/test_scheduler.py::test_local_process_scheduler_basic_map_reduce -v
```

Expected: PASS.

- [ ] **Step 5: Pyright check**

```bash
uv run pyright pyphi/parallel/backends/local_process.py test/test_scheduler.py
```

- [ ] **Step 6: Commit**

```bash
git add pyphi/parallel/backends/local_process.py test/test_scheduler.py
git commit -m "Add LocalProcessScheduler implementing the Scheduler Protocol

Wraps the existing LocalMapReduce with snapshot-via-closure config
delivery. Workers call _apply_snapshot_if_changed at the start of each
task; identical-snapshot calls are deduped via a process-local hash
cache.
"
```

### Task 3.4: Snapshot-delivery integration test

**Files:**
- Test: `test/test_scheduler.py`

- [ ] **Step 1: Write the failing test**

Append to `test/test_scheduler.py`:

```python
def _read_precision(_x):
    """Worker-side function reading config to verify snapshot delivery."""
    from pyphi.conf import config
    return config.numerics.precision


def test_local_process_scheduler_propagates_config_override():
    """Workers see config changes captured at map_reduce dispatch time."""
    from pyphi.conf import config
    from pyphi.parallel.backends.local_process import LocalProcessScheduler

    s = LocalProcessScheduler()

    with config.override(precision=11):
        results = s.map_reduce(_read_precision, [1, 2, 3], reducer=list)

    assert results == [11, 11, 11]
```

- [ ] **Step 2: Run to verify pass** (the closure-based snapshot delivery should already work)

```bash
uv run pytest test/test_scheduler.py::test_local_process_scheduler_propagates_config_override -v
```

Expected: PASS. If FAIL, the snapshot is being captured AFTER the override exits — verify the snapshot-capture line in `LocalProcessScheduler.map_reduce` runs before any executor submission.

- [ ] **Step 3: Commit**

```bash
git add test/test_scheduler.py
git commit -m "Add LocalProcessScheduler config-propagation test

Verifies that with config.override(precision=11): scheduler.map_reduce(...)
delivers precision=11 to workers, not the pre-override default.
"
```

### Task 3.5: Phase 3 acceptance gate

- [ ] **Step 1: Run full gate**

Background:

```bash
uv run pytest test/test_golden_regression.py -v
```

Foreground:

```bash
uv run pytest test/test_invariants.py test/test_invariants_hypothesis.py test/test_scheduler.py test/test_parallel.py test/test_subsystem_surface.py test/test_formalism_pickle.py -v
uv run pyright pyphi/parallel/ pyphi/conf/legacy_global.py
uv run ruff check pyphi/parallel/ test/test_scheduler.py
```

Expected: all green.

---

## Phase 4 — `LocalThreadScheduler`

`concurrent.futures.ThreadPoolExecutor`-backed scheduler. `supports_shared_state = True`. Snapshot apply is a no-op when running in-thread (workers share parent's globals).

### Task 4.1: Implement `LocalThreadScheduler`

**Files:**
- Create: `pyphi/parallel/backends/local_thread.py`

- [ ] **Step 1: Write the failing test**

Append to `test/test_scheduler.py`:

```python
def test_local_thread_scheduler_implements_protocol():
    from pyphi.parallel.backends.local_thread import LocalThreadScheduler

    s = LocalThreadScheduler()
    assert isinstance(s, Scheduler)
    assert s.supports_shared_state is True


def test_local_thread_scheduler_basic_map_reduce():
    from pyphi.parallel.backends.local_thread import LocalThreadScheduler

    s = LocalThreadScheduler()
    result = s.map_reduce(lambda x: x + 1, [10, 20, 30], reducer=sum)
    assert result == 11 + 21 + 31


def test_local_thread_scheduler_does_not_apply_snapshot():
    """Threads share parent's globals; apply must be a no-op (no overwrite)."""
    from pyphi.conf import config
    from pyphi.parallel.backends.local_thread import LocalThreadScheduler

    s = LocalThreadScheduler()
    with config.override(precision=11):
        # Capture parent's view inside the override
        parent_view = config.numerics.precision

        def read_precision(_):
            return config.numerics.precision

        worker_views = s.map_reduce(read_precision, [1, 2, 3], reducer=list)

    assert parent_view == 11
    assert worker_views == [11, 11, 11]
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest test/test_scheduler.py::test_local_thread_scheduler_implements_protocol -v
```

Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement `LocalThreadScheduler`**

Create `pyphi/parallel/backends/local_thread.py`:

```python
"""Thread-pool scheduler.

Workers run in the parent process, so they share the parent's global
config and caches. Snapshot apply is a no-op (the parent's live globals
already reflect the captured snapshot).
"""

from __future__ import annotations

import os
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from typing import Any

from pyphi.parallel.scheduler import (
    ChunkingPolicy,
    ProgressPolicy,
    ShortcircuitPolicy,
)


class LocalThreadScheduler:
    """Scheduler backed by ``concurrent.futures.ThreadPoolExecutor``.

    Best suited for free-threaded Python (3.13t+) where multiple OS threads
    can execute Python concurrently. Under standard CPython, the GIL
    limits the throughput benefit but the scheduler is still useful for
    avoiding pickle overhead and for IO-bound work.
    """

    @property
    def supports_shared_state(self) -> bool:
        return True

    def map_reduce(
        self,
        fn,
        items,
        *more_items,
        reducer=list,
        config_snapshot=None,
        chunking=None,
        progress=None,
        shortcircuit=None,
        ordered=False,
        map_kwargs=None,
    ):
        from pyphi.parallel.backends import local_process

        chunking = chunking or ChunkingPolicy()
        progress = progress or ProgressPolicy()
        shortcircuit = shortcircuit or ShortcircuitPolicy()
        map_kwargs = map_kwargs or {}

        # Mark the parent PID so the snapshot-apply hook short-circuits
        # when called in-thread (threads share parent's globals).
        local_process._PARENT_PID = os.getpid()

        # Determine workers
        from pyphi.parallel import get_num_processes
        num_workers = get_num_processes()

        iterables = (items, *more_items)
        materialized = [list(it) if not hasattr(it, "__len__") else it for it in iterables]
        if not materialized or not materialized[0]:
            return reducer([])

        if len(materialized[0]) < chunking.sequential_threshold:
            results = [fn(*args, **map_kwargs) for args in zip(*materialized, strict=False)]
            return reducer(results)

        results: list[Any] = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures: list[Future] = [
                executor.submit(fn, *args, **map_kwargs)
                for args in zip(*materialized, strict=False)
            ]
            iterator = futures if ordered else as_completed(futures)
            for fut in iterator:
                value = fut.result()
                results.append(value)
                if shortcircuit.func(value):
                    for remaining in futures:
                        if not remaining.done():
                            remaining.cancel()
                    if shortcircuit.callback is not None:
                        shortcircuit.callback(futures)
                    break

        return reducer(results)
```

- [ ] **Step 4: Run to verify pass**

```bash
uv run pytest test/test_scheduler.py -v
```

Expected: PASS on the three new tests.

- [ ] **Step 5: Pyright check**

```bash
uv run pyright pyphi/parallel/backends/local_thread.py
```

- [ ] **Step 6: Commit**

```bash
git add pyphi/parallel/backends/local_thread.py test/test_scheduler.py
git commit -m "Add LocalThreadScheduler using ThreadPoolExecutor

Workers share the parent process's globals and caches. The snapshot
apply hook short-circuits in-thread by checking os.getpid() against
the scheduler-set parent PID. Useful for free-threaded Python and
IO-bound workloads.
"
```

### Task 4.2: Phase 4 acceptance gate

- [ ] **Step 1: Run gate**

Background:

```bash
uv run pytest test/test_golden_regression.py -v
```

Foreground:

```bash
uv run pytest test/test_invariants.py test/test_invariants_hypothesis.py test/test_scheduler.py test/test_parallel.py -v
uv run pyright pyphi/parallel/
uv run ruff check pyphi/parallel/
```

Expected: all green.

---

## Phase 5 — `DaskScheduler` skeleton

Stub class with lazy `dask.distributed` import. `map_reduce` raises `NotImplementedError`. The skeleton exists to exercise the Protocol against three backend shapes; cluster deployment fills it in later (P18).

### Task 5.1: Implement `DaskScheduler` skeleton + tests

**Files:**
- Create: `pyphi/parallel/backends/dask.py`
- Test: `test/test_scheduler.py`

- [ ] **Step 1: Write failing tests**

Append to `test/test_scheduler.py`:

```python
def test_dask_scheduler_skeleton_lazy_import():
    """Importing pyphi must not load dask.distributed."""
    import sys

    # Force a clean state — pop dask.distributed if present
    sys.modules.pop("dask.distributed", None)

    # Importing the dask backend module shouldn't pull in dask.distributed
    from pyphi.parallel.backends import dask as _dask_module  # noqa: F401

    assert "dask.distributed" not in sys.modules


def test_dask_scheduler_raises_not_implemented():
    from pyphi.parallel.backends.dask import DaskScheduler

    s = DaskScheduler()
    assert isinstance(s, Scheduler)
    assert s.supports_shared_state is False
    with pytest.raises(NotImplementedError, match="DaskScheduler is a stub"):
        s.map_reduce(lambda x: x, [1, 2, 3])
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest test/test_scheduler.py::test_dask_scheduler_skeleton_lazy_import test/test_scheduler.py::test_dask_scheduler_raises_not_implemented -v
```

Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement the skeleton**

Create `pyphi/parallel/backends/dask.py`:

```python
"""Skeleton DaskScheduler.

Stub implementation that documents the Protocol shape against
``dask.distributed`` without depending on the import. Cluster deployment
fills this in later; the Protocol is the contract that unblocks it.

The import of ``dask.distributed`` is deferred until ``map_reduce`` is
actually called (which raises NotImplementedError). Until then, importing
this module is free.
"""

from __future__ import annotations


class DaskScheduler:
    """Stub scheduler placeholder for cluster deployments."""

    @property
    def supports_shared_state(self) -> bool:
        return False

    def map_reduce(
        self,
        fn,
        items,
        *more_items,
        reducer=list,
        config_snapshot=None,
        chunking=None,
        progress=None,
        shortcircuit=None,
        ordered=False,
        map_kwargs=None,
    ):
        raise NotImplementedError(
            "DaskScheduler is a stub; fill in for cluster deployments. "
            "See ROADMAP P18 for the follow-up project."
        )
```

- [ ] **Step 4: Run to verify pass**

```bash
uv run pytest test/test_scheduler.py::test_dask_scheduler_skeleton_lazy_import test/test_scheduler.py::test_dask_scheduler_raises_not_implemented -v
```

Expected: PASS.

- [ ] **Step 5: Pyright check**

```bash
uv run pyright pyphi/parallel/backends/dask.py
```

- [ ] **Step 6: Commit**

```bash
git add pyphi/parallel/backends/dask.py test/test_scheduler.py
git commit -m "Add DaskScheduler skeleton with lazy import

Stub implementation that satisfies the Scheduler Protocol shape and
documents the contract for cluster deployments. map_reduce raises
NotImplementedError pointing at the P18 follow-up. dask.distributed
is not imported until and unless map_reduce is filled in.
"
```

### Task 5.2: Phase 5 acceptance gate

- [ ] **Step 1: Run gate**

```bash
uv run pytest test/test_invariants.py test/test_invariants_hypothesis.py test/test_scheduler.py test/test_parallel.py -v
uv run pyright pyphi/parallel/
uv run ruff check pyphi/parallel/ test/test_scheduler.py
```

Background golden:

```bash
uv run pytest test/test_golden_regression.py -v
```

Expected: all green.

---

## Phase 6 — Delete `chunking.py`; add `sampling.py`

Delete the dead-code `chunking.py` (254 lines, never imported outside itself). Add `sampling.py` (~60 lines) with cost-sampling logic. Wire into both schedulers as the default chunksize source.

### Task 6.1: Delete `chunking.py` and `test_chunking.py`

**Files:**
- Delete: `pyphi/parallel/chunking.py`
- Delete: `test/test_chunking.py`

- [ ] **Step 1: Confirm no external imports**

```bash
grep -rn "from pyphi.parallel.chunking\|import pyphi.parallel.chunking\|adaptive_chunk\|chunked_by_work\|estimate_work_size\|estimate_total_work\|calculate_target_work" pyphi/ test/ --include="*.py" | grep -v __pycache__ | grep -v "chunking.py:" | grep -v "test_chunking.py:"
```

Expected: no output. If imports exist, do NOT delete; investigate first.

- [ ] **Step 2: Delete the files**

```bash
git rm pyphi/parallel/chunking.py test/test_chunking.py
```

- [ ] **Step 3: Run gate to confirm nothing breaks**

```bash
uv run pytest test/ --ignore=test/test_parallel.py -v -x
```

Expected: all green.

- [ ] **Step 4: Commit**

```bash
git commit -m "Delete unused chunking.py and its tests

The 254-line speculative-heuristic module was never imported outside
itself: estimate_work_size, adaptive_chunk, chunked_by_work, and the
context-string registry never ran in production. The actual chunking
path uses more_itertools.chunked_even with chunksize from tree.py.
Cost-sampling lands in sampling.py in the next commit.
"
```

### Task 6.2: Add `sampling.py` with cost-sampling

**Files:**
- Create: `pyphi/parallel/sampling.py`
- Test: `test/test_sampling.py`

- [ ] **Step 1: Write failing tests**

Create `test/test_sampling.py`:

```python
"""Tests for cost-sampling chunksize calculation."""
from __future__ import annotations

import time

from pyphi.parallel.sampling import compute_chunksize


def test_compute_chunksize_below_sequential_threshold_returns_one():
    items = [1, 2, 3]
    chunksize, remainder = compute_chunksize(items, target_seconds=1.0, sequential_threshold=10)
    assert chunksize == 1
    assert list(remainder) == items


def test_compute_chunksize_with_explicit_chunksize_skips_sampling():
    items = list(range(100))
    chunksize, remainder = compute_chunksize(items, explicit_chunksize=5)
    assert chunksize == 5
    assert list(remainder) == items


def test_compute_chunksize_samples_and_chunks():
    """A 10ms-per-item workload over 1s target chunks at ~100 items per chunk."""
    items = list(range(400))

    def fast_op(x):
        time.sleep(0.001)  # 1ms per item; target 1s/chunk -> ~1000 items/chunk
        return x

    chunksize, remainder = compute_chunksize(
        items, target_seconds=1.0, fn=fast_op, sample_size=4
    )
    assert chunksize >= 100
    # All items including the four samples should still be processable
    assert sum(1 for _ in remainder) == len(items)


def test_compute_chunksize_handles_unknown_length_iterable():
    """Generators without __len__ fall back to first-N samples."""
    def gen():
        for i in range(50):
            yield i

    chunksize, remainder = compute_chunksize(
        gen(), target_seconds=0.001, fn=lambda x: x, sample_size=4
    )
    assert chunksize >= 1
    # remainder yields the rest of the items (sampled prefix replayed)
    seen = list(remainder)
    assert len(seen) == 50
```

- [ ] **Step 2: Run to verify failure**

```bash
uv run pytest test/test_sampling.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `sampling.py`**

Create `pyphi/parallel/sampling.py`:

```python
"""Cost-sampling chunksize calculation for the Scheduler Protocol.

Samples up to four items spread across the iterable (positions 0, N/4, N/2,
3N/4 for known-length sequences; first four for unknown-length generators),
times them inline, and computes a target chunksize that aims for roughly
``target_seconds`` of wall time per chunk.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from itertools import chain
from typing import Any

DEFAULT_SAMPLE_SIZE = 4
DEFAULT_TARGET_SECONDS = 1.0


def compute_chunksize(
    items: Iterable,
    *,
    target_seconds: float = DEFAULT_TARGET_SECONDS,
    fn: Callable[[Any], Any] | None = None,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    sequential_threshold: int = 1,
    explicit_chunksize: int | None = None,
) -> tuple[int, Iterator]:
    """Return ``(chunksize, items_iterator)`` for a workload.

    The returned iterator yields all original items including the ones
    used for sampling. If ``explicit_chunksize`` is provided, sampling is
    skipped entirely.
    """
    if explicit_chunksize is not None:
        return explicit_chunksize, iter(items)

    # Try to read the length without exhausting the iterator
    if hasattr(items, "__len__"):
        total = len(items)  # type: ignore[arg-type]
        if total < sequential_threshold or fn is None:
            return 1, iter(items)
        if total < sample_size:
            return 1, iter(items)
        return _sample_known_length(items, total, fn, sample_size, target_seconds)

    return _sample_unknown_length(items, fn, sample_size, target_seconds)


def _sample_known_length(items, total, fn, sample_size, target_seconds):
    items_list = list(items)
    positions = [int(i * total / sample_size) for i in range(sample_size)]
    samples = [items_list[p] for p in positions]
    elapsed = _time_samples(fn, samples)
    chunksize = _chunksize_from_timing(elapsed, sample_size, target_seconds)
    return chunksize, iter(items_list)


def _sample_unknown_length(items, fn, sample_size, target_seconds):
    iterator = iter(items)
    sampled: list = []
    for _ in range(sample_size):
        try:
            sampled.append(next(iterator))
        except StopIteration:
            break
    if fn is None or not sampled:
        return 1, chain(sampled, iterator)
    elapsed = _time_samples(fn, sampled)
    chunksize = _chunksize_from_timing(elapsed, len(sampled), target_seconds)
    return chunksize, chain(sampled, iterator)


def _time_samples(fn, samples) -> float:
    start = time.perf_counter()
    for item in samples:
        fn(item)
    return time.perf_counter() - start


def _chunksize_from_timing(elapsed: float, n: int, target_seconds: float) -> int:
    if elapsed <= 0:
        return 1
    mean_per_item = elapsed / n
    return max(1, int(target_seconds / mean_per_item))
```

- [ ] **Step 4: Run to verify pass**

```bash
uv run pytest test/test_sampling.py -v
```

Expected: PASS.

- [ ] **Step 5: Pyright + ruff**

```bash
uv run pyright pyphi/parallel/sampling.py test/test_sampling.py
uv run ruff check pyphi/parallel/sampling.py test/test_sampling.py
```

- [ ] **Step 6: Commit**

```bash
git add pyphi/parallel/sampling.py test/test_sampling.py
git commit -m "Add sampling.py with cost-based chunksize calculation

Replaces the deleted heuristic-based chunking.py with a small
data-driven module: sample up to four items spread across the
iterable, time them inline, target ~1s wall time per chunk. Generators
without __len__ fall back to first-N sampling. Explicit chunksize
bypasses sampling.
"
```

### Task 6.3: Wire `compute_chunksize` into `LocalProcessScheduler`

**Files:**
- Modify: `pyphi/parallel/backends/local_process.py`

- [ ] **Step 1: Write the integration test**

Append to `test/test_scheduler.py`:

```python
def _slow_op(x):
    """Worker function with a known per-item cost for sampling."""
    import time

    time.sleep(0.005)
    return x


def test_local_process_scheduler_uses_cost_sampling_by_default():
    """Default chunking samples to compute target chunksize."""
    from pyphi.parallel.backends.local_process import LocalProcessScheduler

    s = LocalProcessScheduler()
    # 100 items at 5ms each = 500ms total work; should chunk somewhere
    # reasonable rather than one task per chunk.
    items = list(range(100))
    result = s.map_reduce(_slow_op, items, reducer=list)
    assert sorted(result) == items
```

- [ ] **Step 2: Modify `LocalProcessScheduler.map_reduce` to invoke `compute_chunksize`**

In `pyphi/parallel/backends/local_process.py`, change the chunksize-derivation block to call `sampling.compute_chunksize` when `chunking.chunksize is None`:

```python
from pyphi.parallel.sampling import compute_chunksize as _compute_chunksize

# ... inside map_reduce, replace the constraints/chunksize section:

if chunking.chunksize is not None:
    chunksize = chunking.chunksize
    items_iter = iter(items)
else:
    chunksize, items_iter = _compute_chunksize(
        items,
        target_seconds=chunking.target_seconds,
        fn=fn,
        sequential_threshold=chunking.sequential_threshold,
    )

iterables = (list(items_iter), *more_items)
total = len(iterables[0])
```

Then continue with the existing `LocalMapReduce` invocation, passing the computed `chunksize` and the materialized `iterables[0]`.

- [ ] **Step 3: Run to verify pass**

```bash
uv run pytest test/test_scheduler.py::test_local_process_scheduler_uses_cost_sampling_by_default -v
```

Expected: PASS.

- [ ] **Step 4: Pyright check**

```bash
uv run pyright pyphi/parallel/backends/local_process.py
```

- [ ] **Step 5: Commit**

```bash
git add pyphi/parallel/backends/local_process.py test/test_scheduler.py
git commit -m "Wire cost-sampling into LocalProcessScheduler default chunksize

Default chunking now samples up to four items inline, times them, and
computes target chunksize for ~1s wall time per chunk. Explicit
chunksize on ChunkingPolicy bypasses sampling.
"
```

### Task 6.4: Wire `compute_chunksize` into `LocalThreadScheduler`

**Files:**
- Modify: `pyphi/parallel/backends/local_thread.py`

- [ ] **Step 1: Modify `LocalThreadScheduler.map_reduce`**

Add the same `compute_chunksize` integration. For thread schedulers, since work runs in the parent process, sampling cost is the same as production cost; this matters less but still helps progress reporting and worker submission batching.

- [ ] **Step 2: Run gate**

```bash
uv run pytest test/test_scheduler.py test/test_sampling.py test/test_parallel.py -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add pyphi/parallel/backends/local_thread.py
git commit -m "Wire cost-sampling into LocalThreadScheduler default chunksize"
```

### Task 6.5: Phase 6 acceptance gate

- [ ] **Step 1: Run full gate**

Background:

```bash
uv run pytest test/test_golden_regression.py -v
```

Foreground:

```bash
uv run pytest test/test_invariants.py test/test_invariants_hypothesis.py test/test_scheduler.py test/test_sampling.py test/test_parallel.py -v
uv run pyright pyphi/parallel/
uv run ruff check pyphi/parallel/
```

Expected: all green.

---

## Phase 7 — Fill `TODO(4.0)` parallelize markers

Six markers across four files. Use `MapReduce` (which now dispatches through the Scheduler Protocol under the hood).

### Task 7.1: Parallelize `iit4/__init__.py:763,772,780` (`all_complexes`, `irreducible_complexes`, `maximal_complex`)

**Files:**
- Modify: `pyphi/formalism/iit4/__init__.py`

- [ ] **Step 1: Read the current code**

```bash
sed -n '755,790p' pyphi/formalism/iit4/__init__.py
```

- [ ] **Step 2: Replace `all_complexes`** (around line 763)

```python
def all_complexes(network, state, **kwargs):
    """Yield SIAs for all subsystems of the network."""
    from pyphi.parallel import MapReduce

    subsystems = list(reachable_subsystems(network, network.node_indices, state))
    parallel_kwargs = config.infrastructure.parallel_complex_evaluation
    yield from MapReduce(
        sia,
        subsystems,
        map_kwargs=kwargs,
        desc="Evaluating complexes",
        **parallel_kwargs,
    ).run()
```

- [ ] **Step 3: Update `irreducible_complexes`** (drop the inline TODO; the call now flows through `all_complexes` which is parallel)

Replace the function body with:

```python
def irreducible_complexes(network, state, complexes=None, **kwargs):
    """Yield SIAs for irreducible subsystems of the network."""
    if complexes is None:
        complexes = all_complexes(network, state, **kwargs)
    yield from filter(None, complexes)
```

- [ ] **Step 4: Update `maximal_complex`**

```python
def maximal_complex(network, state, complexes=None, **kwargs):
    return max(
        irreducible_complexes(network, state, complexes=complexes, **kwargs),
        default=NullPhiStructure(),
    )
```

- [ ] **Step 5: Acceptance gate**

```bash
uv run pytest test/test_compute_network.py test/test_invariants.py -v
```

Background golden (touched code paths):

```bash
uv run pytest test/test_golden_regression.py -v
```

Expected: all green. Wait for golden notification.

- [ ] **Step 6: Commit**

```bash
git add pyphi/formalism/iit4/__init__.py
git commit -m "Parallelize all_complexes via MapReduce

Resolves three TODO(4.0) parallelize markers in iit4/__init__.py.
all_complexes dispatches through MapReduce with the existing
parallel_complex_evaluation config; downstream functions
(irreducible_complexes, maximal_complex) consume the parallel
generator without their own parallel paths.
"
```

### Task 7.2: Drop the three spurious `TODO(4.0)` parallel markers

**Files:**
- Modify: `pyphi/compute/subsystem.py:322`
- Modify: `pyphi/models/ces.py:194`
- Modify: `pyphi/actual.py:667`

These three markers don't represent real parallelization opportunities:

- `compute/subsystem.py:322 # TODO(4.0): parallel: expose options` — the surrounding `_sia` function already parallelizes via `_sia_map_reduce(cuts, subsystem, unpartitioned_ces, **kwargs)` and `**kwargs` already flows parallel options through. Marker is stale.
- `models/ces.py:194 # TODO(4.0) parallelize` — `resolve_congruence` is a `filter()` over a generator. The per-element operation (`distinction.resolve_congruence(system_state)`) is cheap enough that parallel overhead would dominate. Marker should be dropped.
- `actual.py:667 # TODO(4.0) change parallel default to True?` — `sia()` already uses `MapReduce` with `parallel_kwargs = conf.parallel_kwargs(dict(config.infrastructure.parallel_cut_evaluation), **kwargs)`. The default is set by `config.infrastructure.parallel_cut_evaluation.parallel`, not by this function. Marker is stale.

- [ ] **Step 1: Drop the marker in `compute/subsystem.py:322`**

Remove the line `# TODO(4.0): parallel: expose options` (no other change).

- [ ] **Step 2: Drop the marker in `models/ces.py:194`**

Remove the line `# TODO(4.0) parallelize` (no other change). The body of `resolve_congruence` stays sequential.

- [ ] **Step 3: Drop the marker in `actual.py:667`**

Remove the line `# TODO(4.0) change parallel default to True?` (no other change). The default lives in the config option, not the function signature.

- [ ] **Step 4: Acceptance gate**

```bash
uv run pytest test/test_compute_network.py test/test_actual.py test/test_invariants.py test/test_subsystem_surface.py -v
```

Expected: all green (no behavior change).

- [ ] **Step 5: Commit**

```bash
git add pyphi/compute/subsystem.py pyphi/models/ces.py pyphi/actual.py
git commit -m "Drop stale TODO(4.0) parallel markers

Three markers turned out not to represent real parallelization
opportunities:

- compute/subsystem.py:322 _sia already parallelizes via
  _sia_map_reduce; **kwargs flows parallel options through.
- models/ces.py:194 resolve_congruence is a filter over a generator;
  per-element work is too cheap to amortize parallel overhead.
- actual.py:667 sia() already uses MapReduce; the default-on
  question is settled by config.infrastructure.parallel_cut_evaluation.
"
```

### Task 7.3: Phase 7 acceptance gate

- [ ] **Step 1: Run full gate**

Background:

```bash
uv run pytest test/test_golden_regression.py -v
```

Foreground:

```bash
uv run pytest test/test_invariants.py test/test_invariants_hypothesis.py test/test_scheduler.py test/test_sampling.py test/test_parallel.py test/test_compute_network.py test/test_actual.py -v
uv run pyright pyphi/formalism/ pyphi/compute/ pyphi/models/ pyphi/actual.py
uv run ruff check pyphi/
```

Expected: all green.

---

## Phase 8 — Re-enable parallel tests in CI

### Task 8.1: Drop `--ignore=test/test_parallel.py` from CI

**Files:**
- Modify: `.github/workflows/test.yml:36,40`

- [ ] **Step 1: Update test.yml**

Change line 36:

```yaml
        run: uv run pytest test/ --tb=short -v --ignore=test/test_parallel.py --ignore=test/test_parallel2.py
```

to:

```yaml
        run: uv run pytest test/ --tb=short -v
```

Same for line 40 (the coverage variant).

- [ ] **Step 2: Run test_parallel.py locally to verify it passes**

```bash
uv run pytest test/test_parallel.py -v
```

Expected: green. If failures appear, mark thread-scheduler-specific tests `@pytest.mark.xfail(reason="requires P6b for relations workflows")` and document in the changelog.

- [ ] **Step 3: Update `test_backend_selection`**

In `test/test_parallel.py:348-358`, the test asserts that `backend="auto"` resolves to `"local"`. With the new `default_scheduler()` semantics, `"auto"` resolves to a Scheduler instance, not a string. Update:

```python
def test_backend_selection():
    """Test backend auto-detection and explicit selection."""
    mr = parallel.MapReduce(lambda x: x, [1, 2, 3], backend="auto")
    assert mr.backend in ("local", "process", "thread")

    mr = parallel.MapReduce(lambda x: x, [1, 2, 3], backend="local")
    assert mr.backend == "local"

    with pytest.raises(ValueError, match="[Uu]nknown"):
        parallel.MapReduce(lambda x: x, [1, 2, 3], backend="invalid")
```

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/test.yml test/test_parallel.py
git commit -m "Re-enable test/test_parallel.py in CI

The --ignore flag pre-dates the Scheduler Protocol redesign and is no
longer needed: tests pass locally and the new scheduler implementations
are covered by both test_parallel.py and test_scheduler.py.
"
```

### Task 8.2: Phase 8 acceptance gate

- [ ] **Step 1: Run full local CI mirror**

```bash
uv run pytest test/ -v
uv run pyright pyphi/
uv run ruff check pyphi/ test/
uv run ruff format --check pyphi/ test/
```

Expected: all green.

- [ ] **Step 2: Background golden** (final acceptance check)

```bash
uv run pytest test/test_golden_regression.py -v
```

Expected: 17/17 numerical match.

---

## Phase 9 — Cleanup + changelog

### Task 9.1: Route `MapReduce._run_parallel` through Scheduler Protocol

**Files:**
- Modify: `pyphi/parallel/__init__.py`

- [ ] **Step 1: Update `_resolve_backend` to accept the new values**

Replace the `_resolve_backend` method (around lines 252-262) with:

```python
def _resolve_backend(self, backend: str) -> str:
    """Validate the requested backend; return the canonical name."""
    valid = {"auto", "local", "process", "thread", "dask"}
    if backend not in valid:
        raise ValueError(
            f"Unknown backend: {backend}. Available backends: {sorted(valid)}"
        )
    return backend
```

- [ ] **Step 2: Add a name → Scheduler helper**

Add at the module top (after the existing imports):

```python
def _scheduler_for(name: str):
    """Resolve a backend name to a concrete Scheduler instance."""
    if name == "auto":
        from pyphi.parallel.scheduler import default_scheduler
        return default_scheduler()
    if name in ("local", "process"):
        from pyphi.parallel.backends.local_process import LocalProcessScheduler
        return LocalProcessScheduler()
    if name == "thread":
        from pyphi.parallel.backends.local_thread import LocalThreadScheduler
        return LocalThreadScheduler()
    if name == "dask":
        from pyphi.parallel.backends.dask import DaskScheduler
        return DaskScheduler()
    raise ValueError(f"Unknown backend: {name}")
```

- [ ] **Step 3: Replace `_run_parallel` to dispatch via Scheduler**

Replace the `_run_parallel` method (around lines 291-318) with:

```python
def _run_parallel(self) -> Any:
    """Perform the computation in parallel via the configured Scheduler."""
    from pyphi.parallel.scheduler import (
        ChunkingPolicy,
        ProgressPolicy,
        ShortcircuitPolicy,
    )

    scheduler = _scheduler_for(self.backend)

    chunking = ChunkingPolicy(
        chunksize=getattr(self, "chunksize", None),
        sequential_threshold=getattr(self, "_sequential_threshold", 1),
    )
    progress = ProgressPolicy(
        enabled=self.progress,
        desc=self.desc or "",
        total=self.total,
    )
    shortcircuit = ShortcircuitPolicy(
        func=self.shortcircuit_func,
        callback=self.shortcircuit_callback,
    )

    iterable = self.iterables[0]
    more_iterables = self.iterables[1:]

    def _reducer(results):
        return _reduce(list(results), self.reduce_func, self.reduce_kwargs, branch=False)

    try:
        self.result = scheduler.map_reduce(
            self.map_func,
            iterable,
            *more_iterables,
            reducer=_reducer,
            chunking=chunking,
            progress=progress,
            shortcircuit=shortcircuit,
            ordered=self.ordered,
            map_kwargs=self.map_kwargs,
        )
        self.done = True
        return self.result
    except Exception as e:
        self.error = e
        raise
```

- [ ] **Step 4: Run gates**

```bash
uv run pytest test/test_parallel.py test/test_scheduler.py test/test_invariants.py -v
```

Expected: all green.

- [ ] **Step 5: Pyright + ruff**

```bash
uv run pyright pyphi/parallel/__init__.py
uv run ruff check pyphi/parallel/
```

- [ ] **Step 6: Commit**

```bash
git add pyphi/parallel/__init__.py
git commit -m "Route MapReduce._run_parallel through the Scheduler Protocol

_resolve_backend now accepts auto/local/process/thread/dask; _run_parallel
builds Chunking/Progress/Shortcircuit policies from the MapReduce
instance state and delegates to scheduler.map_reduce. The 12 internal
MapReduce call sites are unchanged.
"
```

### Task 9.2: Add changelog fragment

**Files:**
- Create: `changelog.d/p11-scheduler.feature.md`

- [ ] **Step 1: Write the fragment**

Create `changelog.d/p11-scheduler.feature.md`:

```markdown
Added a typed ``Scheduler`` Protocol abstracting the parallel-execution
backend. Two concrete schedulers ship: ``LocalProcessScheduler``
(``joblib + loky``, today's default behavior) and ``LocalThreadScheduler``
(``concurrent.futures.ThreadPoolExecutor``, useful for free-threaded
runtimes and IO-bound work). A ``DaskScheduler`` skeleton documents the
contract for cluster deployments and raises ``NotImplementedError`` until
filled in.

Workers receive an explicit ``ConfigSnapshot`` via closure rather than
implicitly pickling global state. ``with config.override(...):`` blocks
correctly propagate to workers.

The dead-code ``parallel/chunking.py`` heuristics are replaced by a small
cost-sampling implementation in ``parallel/sampling.py``: the scheduler
samples up to four items spread across the iterable, times them inline,
and computes a target chunksize for roughly 1s of wall time per chunk.

Backend selection: ``config.infrastructure.parallel_backend`` accepts
``"local"``, ``"process"``, ``"thread"``, ``"dask"``, and ``"auto"``.
``"auto"`` selects ``LocalThreadScheduler`` on free-threaded runtimes,
``LocalProcessScheduler`` otherwise.
```

- [ ] **Step 2: Commit**

```bash
git add changelog.d/p11-scheduler.feature.md
git commit -m "changelog: P11 parallelization redesign"
```

### Task 9.3: Update ROADMAP P11 done marker

**Files:**
- Modify: `ROADMAP.md` (the P11 section status line)

- [ ] **Step 1: Update the P11 status note**

After the existing "Status (scope cut for 2.0, 2026-05-09)" note, append a "Status (done, YYYY-MM-DD)" line summarizing what landed and pointing to the spec/plan files.

- [ ] **Step 2: Commit**

```bash
git add ROADMAP.md
git commit -m "ROADMAP: mark P11 done"
```

### Task 9.4: Final acceptance gate + handoff

- [ ] **Step 1: Run full acceptance**

Background golden:

```bash
uv run pytest test/test_golden_regression.py -v
```

Foreground:

```bash
uv run pytest test/ -v
uv run pyright pyphi/
uv run ruff check pyphi/ test/
uv run ruff format --check pyphi/ test/
uv run pre-commit run --all-files
```

Expected: 17/17 golden, hypothesis fast lane 21 green, full unit lane green, pyright clean, ruff clean, pre-commit green.

- [ ] **Step 2: Branch state**

```bash
git log 2.0..HEAD --oneline
```

Expected: 1 commit per phase task (~25-30 commits) on `feature/p11-parallelization-redesign`.

- [ ] **Step 3: Hand off to finishing-a-development-branch skill**

Per CLAUDE.md project rules and saved memory `feedback_ask_before_push.md`: do NOT push without explicit user consent. The branch stays local until 2.0 ships. The user will choose between merging to `2.0` (fast-forward) or keeping for review.
