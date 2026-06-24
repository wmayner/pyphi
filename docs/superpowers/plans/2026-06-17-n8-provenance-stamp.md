# N8 Provenance Stamp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Attach a self-contained provenance record (version, git sha + dirty flag, timestamp, wall-time, seed, Python/numpy/scipy versions, platform) to every top-level result, so a saved result can be audited and reproduced.

**Architecture:** A new frozen `Provenance` dataclass in `pyphi/provenance.py` is auto-stamped into a sibling `provenance` field on the four result types that already carry a `ConfigSnapshot`. Wall-time is filled at the public compute entry points. Provenance renders only at a new top display verbosity tier (`PROVENANCE = 4`); equality stays φ-based so provenance never pollutes comparisons or goldens.

**Tech Stack:** Python 3.12+, stdlib (`dataclasses`, `importlib.metadata`, `datetime`, `platform`, `subprocess`, `functools`), numpy/scipy (already deps), pytest.

## Global Constraints

- Python 3.12+ only; no backwards-compat shims.
- Use `uv run` for all Python commands (`uv run python`, `uv run pytest`).
- Numerical correctness is paramount; this feature must not change any φ/α value.
- Result equality stays φ-based (`cmp.OrderableByPhi`); do not add provenance to equality.
- Final verification runs `uv run pytest` with **no path argument** (so the `pyphi/` doctest sweep runs).
- Do not bypass pre-commit hooks (`--no-verify` forbidden). Fix ruff/pyright legitimately.
- No planning-artifact markers (`N8`, `P`-numbers, "Wave 2") in `pyphi/` source, docstrings, or changelog.

---

### Task 1: The `Provenance` value type

**Files:**
- Create: `pyphi/provenance.py`
- Modify: `pyphi/jsonify.py:122-176` (add `Provenance` to `_loadable_models`)
- Test: `test/test_provenance.py`

**Interfaces:**
- Produces:
  - `Provenance` — frozen dataclass with fields `pyphi_version: str`, `git_sha: str | None`, `git_dirty: bool | None`, `timestamp: str`, `python_version: str`, `numpy_version: str`, `scipy_version: str`, `platform: str`, `wall_time: float | None = None`, `seed: int | None = None`.
  - `Provenance.capture(*, wall_time: float | None = None, seed: int | None = None) -> Provenance` (classmethod).
  - `Provenance.to_json(self) -> dict` and `Provenance.from_json(cls, dct) -> Provenance`.
  - `Provenance.display_rows(self) -> list[tuple[str, str]]` — pure (label, value) pairs for the display layer (no display imports).

- [ ] **Step 1: Write the failing test**

```python
# test/test_provenance.py
from __future__ import annotations

import importlib.metadata
from unittest import mock

from pyphi.provenance import Provenance


def test_capture_populates_fields():
    prov = Provenance.capture()
    assert prov.pyphi_version == importlib.metadata.version("pyphi")
    assert isinstance(prov.timestamp, str) and prov.timestamp.endswith("+00:00")
    assert prov.python_version.count(".") == 2
    assert isinstance(prov.numpy_version, str) and prov.numpy_version
    assert isinstance(prov.scipy_version, str) and prov.scipy_version
    assert "/" in prov.platform
    assert prov.wall_time is None
    assert prov.seed is None
    # git fields are either both populated or both None
    assert (prov.git_sha is None) == (prov.git_dirty is None)


def test_capture_passes_through_wall_time_and_seed():
    prov = Provenance.capture(wall_time=1.5, seed=42)
    assert prov.wall_time == 1.5
    assert prov.seed == 42


def test_git_info_fallback_when_not_a_repo():
    from pyphi import provenance

    provenance._git_info.cache_clear()
    with mock.patch(
        "pyphi.provenance.subprocess.run",
        side_effect=FileNotFoundError("git not found"),
    ):
        sha, dirty = provenance._git_info()
    assert sha is None
    assert dirty is None
    provenance._git_info.cache_clear()


def test_jsonify_round_trip():
    from pyphi.jsonify import jsonify, loads

    prov = Provenance.capture(wall_time=2.0, seed=7)
    restored = loads(jsonify(prov))
    assert isinstance(restored, Provenance)
    assert restored == prov
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_provenance.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyphi.provenance'`.

- [ ] **Step 3: Write the implementation**

```python
# pyphi/provenance.py
"""Provenance record attached to top-level result objects.

A :class:`Provenance` captures how, when, and by what code a result was
computed: the pyphi version and source revision, a timestamp and wall-clock
duration, the RNG seed when one was used, and the Python / numpy / scipy
versions and platform. It is a sibling to :class:`pyphi.conf.ConfigSnapshot`
(which records the configuration), so a saved result is self-describing.
"""

from __future__ import annotations

import functools
import importlib.metadata
import platform as _platform
import subprocess
from dataclasses import dataclass
from dataclasses import replace
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy

_PACKAGE_ROOT = Path(__file__).resolve().parent


@functools.cache
def _git_info() -> tuple[str | None, bool | None]:
    """Return ``(commit_sha, is_dirty)`` for the package's working tree.

    Returns ``(None, None)`` when git is unavailable or the package is not
    inside a working tree (e.g. an installed wheel). Cached: the subprocess
    runs at most once per process.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_PACKAGE_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=_PACKAGE_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None, None
    return sha, bool(status.strip())


@dataclass(frozen=True)
class Provenance:
    """Immutable record of how, when, and by what code a result was computed."""

    pyphi_version: str
    git_sha: str | None
    git_dirty: bool | None
    timestamp: str
    python_version: str
    numpy_version: str
    scipy_version: str
    platform: str
    wall_time: float | None = None
    seed: int | None = None

    @classmethod
    def capture(
        cls, *, wall_time: float | None = None, seed: int | None = None
    ) -> Provenance:
        """Capture the current environment into a :class:`Provenance`.

        ``wall_time`` (seconds) is supplied by the compute entry point; ``seed``
        is supplied only by code paths that consumed an RNG. Both default to
        ``None`` for deterministic, directly-constructed results.
        """
        sha, dirty = _git_info()
        return cls(
            pyphi_version=importlib.metadata.version("pyphi"),
            git_sha=sha,
            git_dirty=dirty,
            timestamp=datetime.now(UTC).isoformat(),
            python_version=_platform.python_version(),
            numpy_version=np.__version__,
            scipy_version=scipy.__version__,
            platform=f"{_platform.system()}/{_platform.machine()}",
            wall_time=wall_time,
            seed=seed,
        )

    def with_wall_time(self, wall_time: float) -> Provenance:
        """Return a copy with ``wall_time`` set (the record is frozen)."""
        return replace(self, wall_time=wall_time)

    def display_rows(self) -> list[tuple[str, str]]:
        """Return ``(label, value)`` pairs for the display layer."""
        git = "n/a"
        if self.git_sha is not None:
            git = self.git_sha[:12] + (" (dirty)" if self.git_dirty else "")
        rows = [
            ("pyphi", self.pyphi_version),
            ("git", git),
            ("Computed", self.timestamp),
            ("Wall time", "n/a" if self.wall_time is None else f"{self.wall_time:.3g} s"),
            ("Python", self.python_version),
            ("numpy", self.numpy_version),
            ("scipy", self.scipy_version),
            ("Platform", self.platform),
        ]
        if self.seed is not None:
            rows.append(("Seed", str(self.seed)))
        return rows

    def to_json(self) -> dict[str, Any]:
        return {
            "pyphi_version": self.pyphi_version,
            "git_sha": self.git_sha,
            "git_dirty": self.git_dirty,
            "timestamp": self.timestamp,
            "python_version": self.python_version,
            "numpy_version": self.numpy_version,
            "scipy_version": self.scipy_version,
            "platform": self.platform,
            "wall_time": self.wall_time,
            "seed": self.seed,
        }

    @classmethod
    def from_json(cls, dct: dict[str, Any]) -> Provenance:
        return cls(**dct)


__all__ = ["Provenance"]
```

Then register `Provenance` as a loadable model. In `pyphi/jsonify.py`, inside `_loadable_models()`, add to the `classes` list (alphabetically near the top, after the `import pyphi` references are available):

```python
        pyphi.provenance.Provenance,  # pyright: ignore[reportAttributeAccessIssue]
```

Verify `pyphi/jsonify.py` imports `pyphi` (it does at module top via the lazy system). If `pyphi.provenance` is not auto-imported, add `import pyphi.provenance  # noqa: F401` near the other `import pyphi.*` lines at the top of `_loadable_models` is unnecessary — the attribute access `pyphi.provenance.Provenance` triggers the lazy import. Confirm with the round-trip test.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/test_provenance.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add pyphi/provenance.py pyphi/jsonify.py test/test_provenance.py
git commit -m "Add Provenance value type with environment capture and jsonify support"
```

---

### Task 2: Attach provenance to the four result types

**Files:**
- Modify: `pyphi/formalism/iit4/__init__.py:179-185` (IIT 4.0 SIA dataclass field + `__post_init__`)
- Modify: `pyphi/models/sia.py:68-103` (IIT 3.0 SIA `__init__`)
- Modify: `pyphi/models/ces.py:70-77` (CES frozen dataclass field + `__post_init__`)
- Modify: `pyphi/models/actual_causation.py:650-687` (AcSIA `__init__`)
- Test: `test/test_provenance.py` (extend)

**Interfaces:**
- Consumes: `Provenance.capture()` from Task 1.
- Produces: each of the four result types exposes a `provenance: Provenance` attribute, auto-stamped at construction when not supplied.

- [ ] **Step 1: Write the failing test**

```python
# append to test/test_provenance.py
import pytest

from pyphi import examples
from pyphi.provenance import Provenance


def _basic_iit4_sia():
    from pyphi import config
    sub = examples.basic_substrate()
    sys = sub.system((1, 0, 0))
    return sys.sia()


def test_iit4_sia_carries_provenance():
    sia = _basic_iit4_sia()
    assert isinstance(sia.provenance, Provenance)


def test_provenance_does_not_pollute_equality():
    # Two independent runs differ in timestamp but must stay equal.
    a = _basic_iit4_sia()
    b = _basic_iit4_sia()
    assert a.provenance.timestamp != b.provenance.timestamp or True  # timestamps may tie
    assert a == b


def test_provenance_excluded_from_diff():
    a = _basic_iit4_sia()
    b = _basic_iit4_sia()
    d = a.diff(b)
    assert d.delta_phi == 0
    assert d.config_diff == {}
```

(If `sub.system(...)` / `.sia()` are not the exact entry points, use the
project's standard way to compute a basic IIT 4.0 SIA — see
`test/test_paper_reproduction.py` for the canonical call. The assertion that
matters is `isinstance(result.provenance, Provenance)`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_provenance.py -k provenance -v`
Expected: FAIL with `AttributeError: 'SystemIrreducibilityAnalysis' object has no attribute 'provenance'`.

- [ ] **Step 3: Implement — IIT 4.0 SIA** (`pyphi/formalism/iit4/__init__.py`)

Add the import near the other model imports at the top of the file:

```python
from pyphi.provenance import Provenance
```

Add the field after `config: ConfigSnapshot | None = None` (line ~179):

```python
    config: ConfigSnapshot | None = None
    provenance: Provenance | None = None
```

In `__post_init__`, after the existing config-snapshot block:

```python
        if self.config is None:
            from pyphi.conf import config as _global

            self.config = _global.snapshot()
        if self.provenance is None:
            self.provenance = Provenance.capture()
```

- [ ] **Step 4: Implement — IIT 3.0 SIA** (`pyphi/models/sia.py`)

Add `provenance=None` to the `__init__` signature (after `config=None`):

```python
        config=None,
        provenance=None,
        reasons=None,
        runner_up=None,
    ):
```

After the existing `self.config = config` block:

```python
        if config is None:
            from pyphi.conf import config as _global

            config = _global.snapshot()
        self.config = config
        if provenance is None:
            from pyphi.provenance import Provenance

            provenance = Provenance.capture()
        self.provenance = provenance
```

- [ ] **Step 5: Implement — CES** (`pyphi/models/ces.py`)

Add the field after `config: Any = None`:

```python
    config: Any = None  # ConfigSnapshot from pyphi.conf.snapshot
    provenance: Any = None  # Provenance from pyphi.provenance
```

In `__post_init__`, after the existing config block:

```python
        if self.config is None:
            from pyphi.conf import config as _global

            object.__setattr__(self, "config", _global.snapshot())
        if self.provenance is None:
            from pyphi.provenance import Provenance

            object.__setattr__(self, "provenance", Provenance.capture())
```

- [ ] **Step 6: Implement — AcSIA** (`pyphi/models/actual_causation.py`)

Add `provenance=None` to the `__init__` signature (after `config=None`, line ~664), then after the existing config block (line ~687):

```python
        if config is None:
            from pyphi.conf import config as _global

            config = _global.snapshot()
        self.config = config
        if provenance is None:
            from pyphi.provenance import Provenance

            provenance = Provenance.capture()
        self.provenance = provenance
```

- [ ] **Step 7: Write the coverage-invariant test**

```python
# append to test/test_provenance.py
from pyphi.conf.snapshot import ConfigSnapshot


def _all_top_level_results():
    """One instance of each result type that carries a ConfigSnapshot."""
    results = []
    # IIT 4.0 SIA + CES
    sub = examples.basic_substrate()
    sys = sub.system((1, 0, 0))
    sia = sys.sia()
    results.append(sia)
    results.append(sys.ces())
    # AcSIA
    ac_sub = examples.actual_causation_substrate()
    results.append(ac_sub.transition((1, 0), (1, 0)).account_sia())  # adjust to real API
    return results


@pytest.mark.parametrize("result", _all_top_level_results())
def test_every_config_carrying_result_carries_provenance(result):
    assert isinstance(result.config, ConfigSnapshot)
    assert isinstance(result.provenance, Provenance)
```

(Use the project's real construction calls for CES / AcSIA / the IIT 3.0 SIA;
mirror how `test/test_result_diff.py` or `test/test_explanation.py` enumerate
the result types for their coverage invariants — those tests are the template
for this one. The required assertion is that every result with a `.config`
also has a `Provenance` on `.provenance`.)

- [ ] **Step 8: Run tests**

Run: `uv run pytest test/test_provenance.py -v`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add pyphi/formalism/iit4/__init__.py pyphi/models/sia.py pyphi/models/ces.py pyphi/models/actual_causation.py test/test_provenance.py
git commit -m "Auto-stamp Provenance onto the four top-level result types"
```

---

### Task 3: Wall-time instrumentation at the compute entry points

**Files:**
- Modify: `pyphi/formalism/queries.py:367` (`sia` dispatch entry point)
- Modify: `pyphi/formalism/iit4/__init__.py:1122` (`ces` entry point) and `pyphi/formalism/iit3/__init__.py:167` (`ces` entry point)
- Test: `test/test_provenance.py` (extend)

**Interfaces:**
- Consumes: `Provenance.with_wall_time(seconds)` from Task 1; the `provenance` field from Task 2.
- Produces: results returned from the public `sia()` / `ces()` entry points have `provenance.wall_time` set to a non-negative float.

- [ ] **Step 1: Write the failing test**

```python
# append to test/test_provenance.py
def test_entry_point_sets_wall_time():
    sia = _basic_iit4_sia()
    assert sia.provenance.wall_time is not None
    assert sia.provenance.wall_time >= 0.0


def test_direct_construction_has_no_wall_time():
    # A result built without going through the entry point keeps wall_time=None.
    from pyphi.provenance import Provenance
    assert Provenance.capture().wall_time is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_provenance.py -k wall_time -v`
Expected: FAIL — `sia.provenance.wall_time is None` (entry point not yet timing).

- [ ] **Step 3: Implement a timing helper**

Add to `pyphi/provenance.py`:

```python
import time
from contextlib import contextmanager


def stamp_wall_time(result: Any, elapsed: float) -> Any:
    """Set ``elapsed`` seconds on ``result.provenance`` if it has one.

    Returns ``result`` (mutated in place for mutable results, or with the
    frozen ``provenance`` replaced). No-op when the result has no provenance.
    """
    prov = getattr(result, "provenance", None)
    if prov is None:
        return result
    stamped = prov.with_wall_time(elapsed)
    try:
        result.provenance = stamped
    except (AttributeError, TypeError):
        object.__setattr__(result, "provenance", stamped)
    return result
```

Add `"stamp_wall_time"` to `__all__`.

- [ ] **Step 4: Wire timing into the `sia` dispatch entry point** (`pyphi/formalism/queries.py`)

Wrap the body of `def sia(cs, **kwargs)` so it times the dispatch and stamps the result before returning:

```python
def sia(cs: System, **kwargs: Any) -> Any:
    import time

    from pyphi.provenance import stamp_wall_time

    start = time.perf_counter()
    result = _dispatch_sia(cs, **kwargs)   # existing dispatch body
    return stamp_wall_time(result, time.perf_counter() - start)
```

(Rename the existing function body to `_dispatch_sia` or inline the timer
around the existing `return` — whichever keeps the diff smallest. The single
requirement: the value returned to the caller has been passed through
`stamp_wall_time`.)

Apply the same pattern to the IIT 4.0 `ces` (`pyphi/formalism/iit4/__init__.py:1122`) and IIT 3.0 `ces` (`pyphi/formalism/iit3/__init__.py:167`) entry points.

- [ ] **Step 5: Run tests**

Run: `uv run pytest test/test_provenance.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyphi/provenance.py pyphi/formalism/queries.py pyphi/formalism/iit4/__init__.py pyphi/formalism/iit3/__init__.py test/test_provenance.py
git commit -m "Fill provenance wall-time at the sia/ces compute entry points"
```

---

### Task 4: Display — `PROVENANCE = 4` verbosity tier

**Files:**
- Modify: `pyphi/display/mixin.py:9-34` (add `PROVENANCE = 4`, update `FULL` docstring)
- Modify: `pyphi/display/__init__.py:9-20` (export `PROVENANCE`)
- Modify: `pyphi/conf/infrastructure.py:21` (`_VALID_REPR_VERBOSITY`)
- Create: `pyphi/display/provenance.py` (the `provenance_section` helper)
- Modify: the four `_describe` methods (iit4 SIA `:307`, models/sia.py, models/ces.py, models/actual_causation.py) to append the provenance section at `verbosity >= PROVENANCE`
- Test: `test/test_provenance.py` and `test/test_display.py` (extend)

**Interfaces:**
- Consumes: `Provenance.display_rows()` from Task 1; `Section`/`Row`/`Description` from `pyphi.display.description`.
- Produces: `pyphi.display.PROVENANCE = 4`; `pyphi.display.provenance.provenance_section(prov: Provenance) -> Section`.

- [ ] **Step 1: Write the failing tests**

```python
# append to test/test_provenance.py
from pyphi import config


def test_provenance_shown_only_at_level_4():
    sia = _basic_iit4_sia()
    with config.override(repr_verbosity=3):
        assert "Provenance" not in repr(sia)
    with config.override(repr_verbosity=4):
        text = repr(sia)
        assert "Provenance" in text
        assert sia.provenance.pyphi_version in text


def test_repr_verbosity_4_is_valid_and_5_is_rejected():
    with config.override(repr_verbosity=4):
        pass  # must not raise
    import pytest
    with pytest.raises(Exception):
        with config.override(repr_verbosity=5):
            pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest test/test_provenance.py -k "level_4 or verbosity" -v`
Expected: FAIL — `repr_verbosity=4` rejected by the validator and/or "Provenance" absent.

- [ ] **Step 3: Add the level and widen the validator**

In `pyphi/display/mixin.py`:

```python
LOW = 0
MEDIUM = 1
HIGH = 2
FULL = 3  # everything HIGH shows, plus all mathematical content (cut grids, repertoires)
PROVENANCE = 4  # FULL plus the provenance metadata section
```

Update the policy docstring in the same file to list `PROVENANCE (4)`.

In `pyphi/display/__init__.py`, add the import and `__all__` entry:

```python
from pyphi.display.mixin import PROVENANCE
...
    "PROVENANCE",
```

In `pyphi/conf/infrastructure.py:21`:

```python
_VALID_REPR_VERBOSITY = frozenset({0, 1, 2, 3, 4})
```

- [ ] **Step 4: Create the section helper**

```python
# pyphi/display/provenance.py
"""Render a Provenance record as a display Section."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyphi.display.description import Row
from pyphi.display.description import Section

if TYPE_CHECKING:
    from pyphi.provenance import Provenance


def provenance_section(prov: Provenance) -> Section:
    """A 'Provenance' Section with one Row per recorded field."""
    return Section(
        label="Provenance",
        rows=tuple(Row(label, value) for label, value in prov.display_rows()),
    )
```

- [ ] **Step 5: Wire into the four `_describe` methods**

In each of the four result types' `_describe(self, verbosity)`, immediately before the `return Description(...)`, append the section (and remove the `# noqa: ARG002` on the IIT 4.0 SIA `_describe`, since `verbosity` is now used):

```python
        from pyphi.display.mixin import PROVENANCE
        if verbosity >= PROVENANCE and self.provenance is not None:
            from pyphi.display.provenance import provenance_section
            sections.append(provenance_section(self.provenance))
```

For result types whose `_describe` builds a tuple of sections rather than a
mutable list, accumulate into a list first (mirror the IIT 4.0 SIA `_describe`,
which already uses a `sections` list).

- [ ] **Step 6: Run tests**

Run: `uv run pytest test/test_provenance.py test/test_display.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add pyphi/display/mixin.py pyphi/display/__init__.py pyphi/display/provenance.py pyphi/conf/infrastructure.py pyphi/formalism/iit4/__init__.py pyphi/models/sia.py pyphi/models/ces.py pyphi/models/actual_causation.py test/test_provenance.py test/test_display.py
git commit -m "Render provenance at a new PROVENANCE=4 display verbosity tier"
```

---

### Task 5: Full verification, roadmap, and changelog

**Files:**
- Modify: `ROADMAP.md` (N8 dashboard row + Wave 2 prose)
- Create: `changelog.d/n8-provenance.feature.md`

- [ ] **Step 1: Run the full suite (with the doctest sweep)**

Run: `uv run pytest`
Expected: PASS (no path argument, so the `pyphi/` doctest sweep runs — this catches any repr/doctest drift from the display change). If any doctest asserts on a result repr that now differs, confirm the difference is only at `repr_verbosity=4` (default is `HIGH=2`, unaffected) and fix the doctest deliberately.

- [ ] **Step 2: Run ruff + pyright directly**

Run: `uv run ruff check pyphi test && uv run pyright pyphi`
Expected: clean. Fix any issues legitimately (no `--no-verify`).

- [ ] **Step 3: Write the changelog fragment**

```bash
cat > changelog.d/n8-provenance.feature.md <<'EOF'
Top-level results (`SystemIrreducibilityAnalysis`, `CauseEffectStructure`, `AcSIA`) now
carry a `provenance` record (pyphi version, source revision, timestamp, wall-time, and
Python/numpy/scipy/platform versions) alongside the existing `config` snapshot, so a saved
result is self-describing. A new `repr_verbosity` level (`4`) displays it.
EOF
```

- [ ] **Step 4: Update ROADMAP dashboard**

Change the N8 row in the Status Dashboard from `⬜ open` to `✅ landed` and update the Wave 2 prose bullet for N8 to past tense describing what shipped (fields, sibling field, `PROVENANCE=4` tier). Do not introduce planning-artifact markers into `pyphi/`.

- [ ] **Step 5: Commit**

```bash
git add ROADMAP.md changelog.d/n8-provenance.feature.md
git commit -m "Mark N8 provenance stamp landed; add changelog fragment"
```

---

## Self-Review

**Spec coverage:**
- Provenance value type + fields + capture + seed semantics → Task 1. ✓
- git sha + dirty via cached subprocess with fallback → Task 1. ✓
- Sibling `provenance` field on the four result types, auto-stamped → Task 2. ✓
- Wall-time at the dispatch entry points; direct construction → None → Task 3. ✓
- `PROVENANCE = 4` display tier + validator widening + four `_describe` hooks → Task 4. ✓
- Non-pollution (equality φ-based, diff excludes provenance, goldens unaffected) → Task 2 tests + the default-verbosity doctest sweep in Task 5. ✓
- jsonify round-trip → Task 1. ✓
- Testing (capture, git fallback, coverage invariant, round-trip, wall-time, display, non-pollution) → Tasks 1–4. ✓
- Out of scope (seed threading, hostname, N4 cache) → not implemented, correctly. ✓

**Placeholder scan:** The CES/AcSIA construction calls in the coverage test are flagged to mirror the existing `test_result_diff.py` / `test_explanation.py` coverage tests rather than spelled out, because the exact public constructors for those result types must match the project's current API — the implementer confirms against those template tests. Every code step that introduces new behavior shows complete code.

**Type consistency:** `Provenance.capture(*, wall_time, seed)`, `Provenance.with_wall_time(wall_time)`, `Provenance.display_rows() -> list[tuple[str, str]]`, `stamp_wall_time(result, elapsed)`, `provenance_section(prov) -> Section`, `PROVENANCE = 4` — names and signatures are consistent across Tasks 1–4.
