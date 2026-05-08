# P10 — Config Split with Result-Object Snapshotting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the flat `pyphi/conf.py` (1112 lines, 43 options on `PyphiConfig`) with three frozen layered dataclasses (`FormalismConfig`, `InfrastructureConfig`, `NumericsConfig`) wrapped in a `ConfigSnapshot`. Attach a `.config: ConfigSnapshot` to every result object so reproducibility is self-contained. Restructure `pyphi_config.yml` to mirror the layered shape.

**Architecture:** Three frozen dataclasses + a `_GlobalConfig` facade (`pyphi.config`). Reads go through layers (`config.numerics.precision`); persistent writes go through the top-level (`config.precision = 6`, routed via `_FIELD_TO_LAYER` map built at import). Scoped writes via `config.override(precision=6, parallel=True)` with build-time field-name collision detection. `PhiFormalism` instances own their `FormalismConfig` via composition; the active formalism is rebuilt from the registry factory when its config changes. Every result object (`SIA`, `RIA`, `MICE`, `Distinction`, `Concept`, `CES`, `PhiStructure`) carries one `ConfigSnapshot` field set at construction time.

**Tech Stack:** Python 3.12+ frozen dataclasses; existing `pyphi/conf.py` (replaced); existing `pyphi/formalism/` package; existing model classes in `pyphi/models/`. No new third-party deps.

**Spec:** `docs/superpowers/specs/2026-05-08-p10-config-split-design.md`.

**Branch:** `feature/p10-config-split` (already cut from `feature/p9-unified-cache` tip `0c62db4c` in worktree `/Users/will/projects/pyphi-p7-kernel-rewrite`). Spec already committed at `da5b58a`.

---

## File Structure

```
pyphi/conf/                            # NEW package, replaces pyphi/conf.py
├── __init__.py                        # CREATE: facade _GlobalConfig + module-level `config` instance
├── _io.py                             # CREATE: YAML load/save helpers (moved from old conf.py)
├── _validate.py                       # CREATE: combination-validation helpers (moved from old conf.py)
├── formalism.py                       # CREATE: FormalismConfig frozen dataclass
├── infrastructure.py                  # CREATE: InfrastructureConfig frozen dataclass
├── numerics.py                        # CREATE: NumericsConfig frozen dataclass
├── snapshot.py                        # CREATE: ConfigSnapshot frozen dataclass
└── legacy_global.py                   # CREATE: _GlobalConfig class (the facade)

pyphi/conf.py                          # DELETE in Phase 6 (shimmed during Phase 2)
pyphi/conf.pyi                         # DELETE in Phase 6; pyphi/conf/__init__.pyi created instead

pyphi/formalism/base.py                # MODIFY: add config: FormalismConfig to PhiFormalism Protocol
pyphi/formalism/iit3/formalism.py      # MODIFY: become @dataclass(frozen=True) with FormalismConfig
pyphi/formalism/iit4/formalism.py      # MODIFY: same for IIT4_2023Formalism + IIT4_2026Formalism
pyphi/formalism/__init__.py            # MODIFY: factory pattern; registry returns formalism with config

pyphi/models/subsystem.py              # MODIFY: add config: ConfigSnapshot to SIA, MICE, etc.
pyphi/models/mechanism.py              # MODIFY: add config: ConfigSnapshot to RIA, Concept, Distinction
pyphi/models/ces.py                    # MODIFY: add config: ConfigSnapshot to CauseEffectStructure
pyphi/models/phi_structure.py          # MODIFY: add config: ConfigSnapshot to PhiStructure
pyphi/jsonify.py                       # MODIFY: register serializers for the four frozen types

pyphi_config.yml                       # MODIFY: convert flat keys to nested layer structure
pyphi_config_3.0.yml                   # MODIFY: same

test/
├── test_config_layers.py              # CREATE: layer dataclass + collision check
├── test_config_override.py            # CREATE: top-level + per-layer override semantics
├── test_config_yaml.py                # CREATE: nested round-trip + flat-format error
├── test_config_jsonify.py             # CREATE: snapshot round-trips through jsonify
└── test_result_config_snapshot.py     # CREATE: end-to-end result has snapshot

changelog.d/p10-config-split.refactor.md  # CREATE: rename map + structural narrative

ROADMAP.md                             # MODIFY in Phase 6: mark P10 complete
```

---

## Phase 0: Baseline + audit + rename map

### Task 0.1: Confirm worktree + branch state

**Files:** none (verification only)

- [ ] **Step 1: Confirm worktree path and branch**

```bash
cd /Users/will/projects/pyphi-p7-kernel-rewrite
git rev-parse --abbrev-ref HEAD
git log --oneline -3
```

Expected: branch `feature/p10-config-split`; HEAD is `da5b58a P10 Phase 0: spec ...` on top of `0c62db4c P9 follow-up: drop Network.purview_cache name registration`.

- [ ] **Step 2: Confirm baseline tests pass**

```bash
uv run pytest test/test_golden_regression.py test/test_invariants.py -q
```

Expected: 17 golden + ~21 invariants pass.

### Task 0.2: Inventory all `config.*` call sites

**Files:** Create `docs/superpowers/plans/p10-callsite-inventory.md` (working scratch — deleted at end of Phase 6)

- [ ] **Step 1: Generate the inventory**

```bash
git grep -n 'config\.' pyphi/ test/ | grep -v '^Binary' > /tmp/p10-config-sites.txt
wc -l /tmp/p10-config-sites.txt
```

Expected: ~150–250 lines.

- [ ] **Step 2: Bucket by access pattern**

```bash
grep -E 'config\.[A-Z_]+\s*=' /tmp/p10-config-sites.txt | wc -l   # writes
grep -E 'config\.override' /tmp/p10-config-sites.txt | wc -l       # overrides
grep -E 'config\.[A-Z_]+\b' /tmp/p10-config-sites.txt | grep -v '=' | grep -v override | wc -l  # reads
```

Record the three counts in `docs/superpowers/plans/p10-callsite-inventory.md`. They are the work units for Phases 2 and 3.

### Task 0.3: Build the rename-map table

**Files:** Modify `docs/superpowers/specs/2026-05-08-p10-config-split-design.md` (replace Appendix A placeholder)

- [ ] **Step 1: Replace Appendix A in the spec with the full table**

```markdown
## Appendix A: Rename-map table

| Old (1.x flat) | New (2.0 layered read) | Layer | YAML key |
|---|---|---|---|
| `FORMALISM` | `config.formalism.formalism` | formalism | `formalism.formalism` |
| `ASSUME_CUTS_CANNOT_CREATE_NEW_CONCEPTS` | `config.formalism.assume_cuts_cannot_create_new_concepts` | formalism | `formalism.assume_cuts_cannot_create_new_concepts` |
| `REPERTOIRE_DISTANCE` | `config.formalism.repertoire_distance` | formalism | `formalism.repertoire_distance` |
| `REPERTOIRE_DISTANCE_DIFFERENTIATION` | `config.formalism.repertoire_distance_differentiation` | formalism | `formalism.repertoire_distance_differentiation` |
| `REPERTOIRE_DISTANCE_SPECIFICATION` | `config.formalism.repertoire_distance_specification` | formalism | `formalism.repertoire_distance_specification` |
| `CES_DISTANCE` | `config.formalism.ces_distance` | formalism | `formalism.ces_distance` |
| `ACTUAL_CAUSATION_MEASURE` | `config.formalism.actual_causation_measure` | formalism | `formalism.actual_causation_measure` |
| `PARTITION_TYPE` | `config.formalism.partition_type` | formalism | `formalism.partition_type` |
| `SYSTEM_PARTITION_TYPE` | `config.formalism.system_partition_type` | formalism | `formalism.system_partition_type` |
| `SYSTEM_PARTITION_INCLUDE_COMPLETE` | `config.formalism.system_partition_include_complete` | formalism | `formalism.system_partition_include_complete` |
| `SYSTEM_CUTS` | `config.formalism.system_cuts` | formalism | `formalism.system_cuts` |
| `DISTINCTION_PHI_NORMALIZATION` | `config.formalism.distinction_phi_normalization` | formalism | `formalism.distinction_phi_normalization` |
| `RELATION_COMPUTATION` | `config.formalism.relation_computation` | formalism | `formalism.relation_computation` |
| `STATE_TIE_RESOLUTION` | `config.formalism.state_tie_resolution` | formalism | `formalism.state_tie_resolution` |
| `MIP_TIE_RESOLUTION` | `config.formalism.mip_tie_resolution` | formalism | `formalism.mip_tie_resolution` |
| `PURVIEW_TIE_RESOLUTION` | `config.formalism.purview_tie_resolution` | formalism | `formalism.purview_tie_resolution` |
| `SHORTCIRCUIT_SIA` | `config.formalism.shortcircuit_sia` | formalism | `formalism.shortcircuit_sia` |
| `SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI` | `config.formalism.single_micro_nodes_with_selfloops_have_phi` | formalism | `formalism.single_micro_nodes_with_selfloops_have_phi` |
| `PARALLEL` | `config.infrastructure.parallel` | infrastructure | `infrastructure.parallel` |
| `PARALLEL_COMPLEX_EVALUATION` | `config.infrastructure.parallel_complex_evaluation` | infrastructure | `infrastructure.parallel_complex_evaluation` |
| `PARALLEL_CUT_EVALUATION` | `config.infrastructure.parallel_cut_evaluation` | infrastructure | `infrastructure.parallel_cut_evaluation` |
| `PARALLEL_CONCEPT_EVALUATION` | `config.infrastructure.parallel_concept_evaluation` | infrastructure | `infrastructure.parallel_concept_evaluation` |
| `PARALLEL_PURVIEW_EVALUATION` | `config.infrastructure.parallel_purview_evaluation` | infrastructure | `infrastructure.parallel_purview_evaluation` |
| `PARALLEL_MECHANISM_PARTITION_EVALUATION` | `config.infrastructure.parallel_mechanism_partition_evaluation` | infrastructure | `infrastructure.parallel_mechanism_partition_evaluation` |
| `PARALLEL_RELATION_EVALUATION` | `config.infrastructure.parallel_relation_evaluation` | infrastructure | `infrastructure.parallel_relation_evaluation` |
| `PARALLEL_WORKERS` | `config.infrastructure.parallel_workers` | infrastructure | `infrastructure.parallel_workers` |
| `PARALLEL_BACKEND` | `config.infrastructure.parallel_backend` | infrastructure | `infrastructure.parallel_backend` |
| `MAXIMUM_CACHE_MEMORY_PERCENTAGE` | `config.infrastructure.maximum_cache_memory_percentage` | infrastructure | `infrastructure.maximum_cache_memory_percentage` |
| `CACHE_REPERTOIRES` | `config.infrastructure.cache_repertoires` | infrastructure | `infrastructure.cache_repertoires` |
| `CACHE_POTENTIAL_PURVIEWS` | `config.infrastructure.cache_potential_purviews` | infrastructure | `infrastructure.cache_potential_purviews` |
| `CLEAR_SUBSYSTEM_CACHES_AFTER_COMPUTING_SIA` | `config.infrastructure.clear_subsystem_caches_after_computing_sia` | infrastructure | `infrastructure.clear_subsystem_caches_after_computing_sia` |
| `LOG_FILE` | `config.infrastructure.log_file` | infrastructure | `infrastructure.log_file` |
| `LOG_FILE_LEVEL` | `config.infrastructure.log_file_level` | infrastructure | `infrastructure.log_file_level` |
| `LOG_STDOUT_LEVEL` | `config.infrastructure.log_stdout_level` | infrastructure | `infrastructure.log_stdout_level` |
| `PROGRESS_BARS` | `config.infrastructure.progress_bars` | infrastructure | `infrastructure.progress_bars` |
| `REPR_VERBOSITY` | `config.infrastructure.repr_verbosity` | infrastructure | `infrastructure.repr_verbosity` |
| `PRINT_FRACTIONS` | `config.infrastructure.print_fractions` | infrastructure | `infrastructure.print_fractions` |
| `LABEL_SEPARATOR` | `config.infrastructure.label_separator` | infrastructure | `infrastructure.label_separator` |
| `WELCOME_OFF` | `config.infrastructure.welcome_off` | infrastructure | `infrastructure.welcome_off` |
| `VALIDATE_SUBSYSTEM_STATES` | `config.infrastructure.validate_subsystem_states` | infrastructure | `infrastructure.validate_subsystem_states` |
| `VALIDATE_CONDITIONAL_INDEPENDENCE` | `config.infrastructure.validate_conditional_independence` | infrastructure | `infrastructure.validate_conditional_independence` |
| `VALIDATE_JSON_VERSION` | `config.infrastructure.validate_json_version` | infrastructure | `infrastructure.validate_json_version` |
| `PRECISION` | `config.numerics.precision` | numerics | `numerics.precision` |

Persistent-write form: `config.<lowercase_name> = value` (e.g. `config.precision = 6`). Scoped: `with config.override(precision=6, parallel=True): ...`.
```

- [ ] **Step 2: Commit Phase 0 — audit + rename map**

```bash
cd /Users/will/projects/pyphi-p7-kernel-rewrite
git add docs/superpowers/specs/2026-05-08-p10-config-split-design.md docs/superpowers/plans/2026-05-08-p10-config-split.md docs/superpowers/plans/p10-callsite-inventory.md
git commit -m "$(cat <<'EOF'
P10 Phase 0: implementation plan + rename map + callsite inventory

Adds the full implementation plan for the config split, populates the
rename-map table in the design spec (43 options × 4 columns), and snapshots
the current call-site count by access pattern (read/write/override) into a
working-scratch file that gets deleted at end of Phase 6.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 1: Three frozen dataclass layers + `ConfigSnapshot` (additive only)

The old `pyphi/conf.py` stays untouched; everything new lives in `pyphi/conf/` (new package). New code is wired in at the end of this phase via a temporary read-shim on `_GlobalConfig` that delegates to the old `PyphiConfig` until Phase 5.

### Task 1.1: Create `pyphi/conf/numerics.py`

**Files:**
- Create: `pyphi/conf/__init__.py` (empty for now; populated in Task 1.6)
- Create: `pyphi/conf/numerics.py`
- Test: `test/test_config_layers.py`

- [ ] **Step 1: Write failing test**

`test/test_config_layers.py`:
```python
"""Tests for the layered config dataclasses introduced in P10."""

from __future__ import annotations

import pytest
from dataclasses import FrozenInstanceError

from pyphi.conf.numerics import NumericsConfig


class TestNumericsConfig:
    def test_default_precision_is_13(self):
        cfg = NumericsConfig()
        assert cfg.precision == 13

    def test_explicit_precision(self):
        cfg = NumericsConfig(precision=6)
        assert cfg.precision == 6

    def test_is_frozen(self):
        cfg = NumericsConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.precision = 6  # type: ignore[misc]

    def test_equality_by_value(self):
        assert NumericsConfig(precision=13) == NumericsConfig(precision=13)
        assert NumericsConfig(precision=13) != NumericsConfig(precision=6)

    def test_hashable(self):
        # Frozen dataclasses are hashable by default; needed for caching
        # snapshot dicts keyed by config.
        assert hash(NumericsConfig(precision=13)) == hash(NumericsConfig(precision=13))
```

- [ ] **Step 2: Run test to verify failure**

```bash
uv run pytest test/test_config_layers.py::TestNumericsConfig -v
```

Expected: ImportError (`pyphi.conf.numerics` doesn't exist).

- [ ] **Step 3: Create `pyphi/conf/__init__.py`**

```python
"""Layered configuration system for PyPhi 2.0.

Three frozen dataclasses (`FormalismConfig`, `InfrastructureConfig`,
`NumericsConfig`) wrapped in a `ConfigSnapshot` value type, accessed
through the `config` singleton.
"""
```

- [ ] **Step 4: Create `pyphi/conf/numerics.py`**

```python
"""Numerics layer of the PyPhi config.

Holds knobs that govern numerical comparison (precision, future tolerances).
Frozen dataclass — replace via ``dataclasses.replace`` or top-level write.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NumericsConfig:
    """Numerical-comparison settings.

    Attributes:
        precision: Decimal places of agreement required when comparing phi
            values via ``pyphi.utils.eq`` and friends. Values smaller than
            ``10**-precision`` are treated as zero. ``PyPhiFloat`` snapshots
            this at construction so its hash is stable across config writes.
    """

    precision: int = 13
```

- [ ] **Step 5: Run test to verify pass**

```bash
uv run pytest test/test_config_layers.py::TestNumericsConfig -v
```

Expected: 5 passed.

### Task 1.2: Create `pyphi/conf/formalism.py`

**Files:**
- Create: `pyphi/conf/formalism.py`
- Modify: `test/test_config_layers.py` (add `TestFormalismConfig`)

- [ ] **Step 1: Write failing test (append to test/test_config_layers.py)**

```python
from pyphi.conf.formalism import FormalismConfig


class TestFormalismConfig:
    def test_defaults_match_legacy(self):
        cfg = FormalismConfig()
        assert cfg.formalism == "IIT_4_0_2023"
        assert cfg.repertoire_distance == "GENERALIZED_INTRINSIC_DIFFERENCE"
        assert cfg.partition_type == "ALL"
        assert cfg.system_partition_type == "SET_UNI/BI"
        assert cfg.shortcircuit_sia is True

    def test_is_frozen(self):
        cfg = FormalismConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.formalism = "IIT_3_0"  # type: ignore[misc]

    def test_replace_returns_new_instance(self):
        from dataclasses import replace
        a = FormalismConfig()
        b = replace(a, repertoire_distance="EMD")
        assert a.repertoire_distance == "GENERALIZED_INTRINSIC_DIFFERENCE"
        assert b.repertoire_distance == "EMD"

    def test_mip_tie_resolution_default_is_list(self):
        cfg = FormalismConfig()
        assert cfg.mip_tie_resolution == ["NORMALIZED_PHI", "NEGATIVE_PHI"]

    def test_mip_tie_resolution_each_instance_independent(self):
        # default_factory protects against shared-list bugs
        a = FormalismConfig()
        b = FormalismConfig()
        assert a.mip_tie_resolution is not b.mip_tie_resolution
```

- [ ] **Step 2: Run tests to verify failure**

```bash
uv run pytest test/test_config_layers.py::TestFormalismConfig -v
```

Expected: ImportError.

- [ ] **Step 3: Create `pyphi/conf/formalism.py`**

```python
"""Formalism layer of the PyPhi config.

Holds knobs that define the IIT mathematical formalism: which metric, which
partition scheme, which tie-resolution policy. Bundled into the
``PhiFormalism`` instance via composition (see ``pyphi/formalism/base.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FormalismConfig:
    """Formalism-scoped configuration.

    These knobs collectively define what mathematical object PyPhi computes.
    They travel with each :class:`~pyphi.formalism.base.PhiFormalism` instance
    and are snapshotted onto every result object so reproducibility doesn't
    depend on the live global config.
    """

    formalism: str = "IIT_4_0_2023"
    assume_cuts_cannot_create_new_concepts: bool = False
    repertoire_distance: str = "GENERALIZED_INTRINSIC_DIFFERENCE"
    repertoire_distance_specification: str = "GENERALIZED_INTRINSIC_DIFFERENCE"
    repertoire_distance_differentiation: str = "GENERALIZED_INTRINSIC_DIFFERENCE"
    ces_distance: str = "SUM_SMALL_PHI"
    actual_causation_measure: str = "PMI"
    partition_type: str = "ALL"
    system_partition_type: str = "SET_UNI/BI"
    system_partition_include_complete: bool = False
    system_cuts: str = "3.0_STYLE"
    distinction_phi_normalization: str = "NUM_CONNECTIONS_CUT"
    relation_computation: str = "CONCRETE"
    state_tie_resolution: str = "PHI"
    mip_tie_resolution: list[str] = field(
        default_factory=lambda: ["NORMALIZED_PHI", "NEGATIVE_PHI"]
    )
    purview_tie_resolution: str = "PHI"
    shortcircuit_sia: bool = True
    single_micro_nodes_with_selfloops_have_phi: bool = True
```

- [ ] **Step 4: Run tests to verify pass**

```bash
uv run pytest test/test_config_layers.py::TestFormalismConfig -v
```

Expected: 5 passed.

### Task 1.3: Create `pyphi/conf/infrastructure.py`

**Files:**
- Create: `pyphi/conf/infrastructure.py`
- Modify: `test/test_config_layers.py` (add `TestInfrastructureConfig`)

- [ ] **Step 1: Write failing test (append to test/test_config_layers.py)**

```python
from pyphi.conf.infrastructure import InfrastructureConfig


class TestInfrastructureConfig:
    def test_defaults_match_legacy(self):
        cfg = InfrastructureConfig()
        assert cfg.parallel is False
        assert cfg.parallel_workers == -1
        assert cfg.parallel_backend == "local"
        assert cfg.cache_repertoires is True
        assert cfg.cache_potential_purviews is True
        assert cfg.clear_subsystem_caches_after_computing_sia is False
        assert cfg.maximum_cache_memory_percentage == 50
        assert cfg.log_file_level == "INFO"
        assert cfg.log_stdout_level == "WARNING"
        assert cfg.progress_bars is True
        assert cfg.repr_verbosity == 2
        assert cfg.print_fractions is True
        assert cfg.label_separator == ""
        assert cfg.welcome_off is False
        assert cfg.validate_subsystem_states is True
        assert cfg.validate_conditional_independence is True
        assert cfg.validate_json_version is True

    def test_parallel_evaluation_dict_has_expected_keys(self):
        cfg = InfrastructureConfig()
        assert set(cfg.parallel_complex_evaluation.keys()) == {
            "parallel", "sequential_threshold", "chunksize", "progress",
        }

    def test_is_frozen(self):
        cfg = InfrastructureConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.parallel = True  # type: ignore[misc]
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest test/test_config_layers.py::TestInfrastructureConfig -v`
Expected: ImportError.

- [ ] **Step 3: Create `pyphi/conf/infrastructure.py`**

```python
"""Infrastructure layer of the PyPhi config.

Holds knobs that govern how PyPhi runs (parallelism, caching, logging,
display, validation) but not what it computes. Snapshotted onto every result
object alongside the formalism config.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _default_parallel_dict(sequential_threshold: int, chunksize: int, progress: bool = True) -> dict[str, Any]:
    return {
        "parallel": False,
        "sequential_threshold": sequential_threshold,
        "chunksize": chunksize,
        "progress": progress,
    }


@dataclass(frozen=True)
class InfrastructureConfig:
    """Infrastructure-scoped configuration.

    Knobs in this layer don't change PyPhi's mathematical output — they
    affect performance, caching policy, logging, presentation, and
    validation. Frozen dataclass; replace via ``dataclasses.replace`` or
    top-level write on the global config.
    """

    # Parallelism
    parallel: bool = False
    parallel_complex_evaluation: Mapping[str, Any] = field(
        default_factory=lambda: _default_parallel_dict(2**4, 2**6, progress=True)
    )
    parallel_cut_evaluation: Mapping[str, Any] = field(
        default_factory=lambda: _default_parallel_dict(2**10, 2**12, progress=False)
    )
    parallel_concept_evaluation: Mapping[str, Any] = field(
        default_factory=lambda: _default_parallel_dict(2**6, 2**8, progress=True)
    )
    parallel_purview_evaluation: Mapping[str, Any] = field(
        default_factory=lambda: _default_parallel_dict(2**6, 2**8, progress=True)
    )
    parallel_mechanism_partition_evaluation: Mapping[str, Any] = field(
        default_factory=lambda: _default_parallel_dict(2**10, 2**12, progress=True)
    )
    parallel_relation_evaluation: Mapping[str, Any] = field(
        default_factory=lambda: _default_parallel_dict(2**10, 2**12, progress=True)
    )
    parallel_workers: int = -1
    parallel_backend: str = "local"

    # Cache
    maximum_cache_memory_percentage: int = 50
    cache_repertoires: bool = True
    cache_potential_purviews: bool = True
    clear_subsystem_caches_after_computing_sia: bool = False

    # Logging
    log_file: str | Path = "pyphi.log"
    log_file_level: str | None = "INFO"
    log_stdout_level: str | None = "WARNING"

    # Display / UX
    progress_bars: bool = True
    repr_verbosity: int = 2
    print_fractions: bool = True
    label_separator: str = ""
    welcome_off: bool = False

    # Validation
    validate_subsystem_states: bool = True
    validate_conditional_independence: bool = True
    validate_json_version: bool = True
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest test/test_config_layers.py::TestInfrastructureConfig -v`
Expected: 3 passed.

### Task 1.4: Create `pyphi/conf/snapshot.py` with `ConfigSnapshot`

**Files:**
- Create: `pyphi/conf/snapshot.py`
- Modify: `test/test_config_layers.py`

- [ ] **Step 1: Write failing test (append)**

```python
from pyphi.conf.snapshot import ConfigSnapshot


class TestConfigSnapshot:
    def test_construction(self):
        snap = ConfigSnapshot(
            formalism=FormalismConfig(),
            infrastructure=InfrastructureConfig(),
            numerics=NumericsConfig(),
        )
        assert snap.numerics.precision == 13
        assert snap.formalism.formalism == "IIT_4_0_2023"

    def test_is_frozen(self):
        snap = ConfigSnapshot(
            formalism=FormalismConfig(),
            infrastructure=InfrastructureConfig(),
            numerics=NumericsConfig(),
        )
        with pytest.raises(FrozenInstanceError):
            snap.numerics = NumericsConfig(precision=6)  # type: ignore[misc]

    def test_diff_finds_differences(self):
        a = ConfigSnapshot(
            formalism=FormalismConfig(),
            infrastructure=InfrastructureConfig(),
            numerics=NumericsConfig(precision=13),
        )
        b = ConfigSnapshot(
            formalism=FormalismConfig(repertoire_distance="EMD"),
            infrastructure=InfrastructureConfig(),
            numerics=NumericsConfig(precision=6),
        )
        diff = a.diff(b)
        assert diff == {
            "formalism.repertoire_distance": (
                "GENERALIZED_INTRINSIC_DIFFERENCE", "EMD",
            ),
            "numerics.precision": (13, 6),
        }

    def test_diff_empty_when_equal(self):
        a = ConfigSnapshot(
            formalism=FormalismConfig(),
            infrastructure=InfrastructureConfig(),
            numerics=NumericsConfig(),
        )
        b = ConfigSnapshot(
            formalism=FormalismConfig(),
            infrastructure=InfrastructureConfig(),
            numerics=NumericsConfig(),
        )
        assert a.diff(b) == {}

    def test_as_kwargs_returns_flat_dict(self):
        snap = ConfigSnapshot(
            formalism=FormalismConfig(repertoire_distance="EMD"),
            infrastructure=InfrastructureConfig(parallel=True),
            numerics=NumericsConfig(precision=6),
        )
        kw = snap.as_kwargs()
        assert kw["repertoire_distance"] == "EMD"
        assert kw["parallel"] is True
        assert kw["precision"] == 6
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest test/test_config_layers.py::TestConfigSnapshot -v`
Expected: ImportError.

- [ ] **Step 3: Create `pyphi/conf/snapshot.py`**

```python
"""Frozen snapshot of the three config layers, attached to result objects.

A ``ConfigSnapshot`` mirrors the live ``pyphi.config`` shape but is
immutable: once a result object carries a snapshot, mutating the live
global doesn't change the snapshot's view of what produced the result.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from pyphi.conf.formalism import FormalismConfig
from pyphi.conf.infrastructure import InfrastructureConfig
from pyphi.conf.numerics import NumericsConfig


@dataclass(frozen=True)
class ConfigSnapshot:
    """Immutable snapshot of the three config layers at construction time.

    Result objects carry one of these so reproducibility is self-contained:
    rerunning a saved result is ``pyphi.config.override(**snap.as_kwargs())``.
    """

    formalism: FormalismConfig
    infrastructure: InfrastructureConfig
    numerics: NumericsConfig

    def diff(self, other: ConfigSnapshot) -> dict[str, tuple[Any, Any]]:
        """Return a dict of dotted-path → (self_value, other_value) for fields that differ.

        Used to summarize what changed between two result objects' snapshots.
        """
        result: dict[str, tuple[Any, Any]] = {}
        for layer_name in ("formalism", "infrastructure", "numerics"):
            self_layer = getattr(self, layer_name)
            other_layer = getattr(other, layer_name)
            for f in fields(self_layer):
                self_val = getattr(self_layer, f.name)
                other_val = getattr(other_layer, f.name)
                if self_val != other_val:
                    result[f"{layer_name}.{f.name}"] = (self_val, other_val)
        return result

    def as_kwargs(self) -> dict[str, Any]:
        """Return a flat dict suitable for ``pyphi.config.override(**snap.as_kwargs())``.

        Field names are unique across all three layers (enforced by the
        build-time collision check), so flattening is unambiguous.
        """
        result: dict[str, Any] = {}
        for layer_name in ("formalism", "infrastructure", "numerics"):
            layer = getattr(self, layer_name)
            for f in fields(layer):
                result[f.name] = getattr(layer, f.name)
        return result
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest test/test_config_layers.py::TestConfigSnapshot -v`
Expected: 5 passed.

### Task 1.5: Build the `_FIELD_TO_LAYER` collision-checking map

**Files:**
- Create: `pyphi/conf/_field_routing.py`
- Modify: `test/test_config_layers.py`

- [ ] **Step 1: Write failing test (append)**

```python
from pyphi.conf._field_routing import FIELD_TO_LAYER, ConfigurationError


class TestFieldRouting:
    def test_all_layer_fields_present(self):
        # Every field of every layer should appear in the map.
        from dataclasses import fields
        for layer_name, layer_cls in [
            ("formalism", FormalismConfig),
            ("infrastructure", InfrastructureConfig),
            ("numerics", NumericsConfig),
        ]:
            for f in fields(layer_cls):
                assert FIELD_TO_LAYER[f.name] == layer_name

    def test_no_collisions_in_current_layers(self):
        # If this fails, a layer field name was added that overlaps another
        # layer. The build-time check should have raised at import.
        all_fields: list[str] = []
        from dataclasses import fields
        for layer_cls in (FormalismConfig, InfrastructureConfig, NumericsConfig):
            all_fields.extend(f.name for f in fields(layer_cls))
        assert len(all_fields) == len(set(all_fields)), \
            f"Collision in field names: {sorted(all_fields)}"

    def test_collision_raises(self):
        # Synthesize a colliding pair via _build_field_map and verify it raises.
        from dataclasses import dataclass
        from pyphi.conf._field_routing import _build_field_map

        @dataclass(frozen=True)
        class _LayerA:
            x: int = 0

        @dataclass(frozen=True)
        class _LayerB:
            x: int = 0  # collides

        with pytest.raises(ConfigurationError, match="Config field name collision"):
            _build_field_map([("a", _LayerA), ("b", _LayerB)])
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest test/test_config_layers.py::TestFieldRouting -v`
Expected: ImportError on `pyphi.conf._field_routing`.

- [ ] **Step 3: Create `pyphi/conf/_field_routing.py`**

```python
"""Build-time map from flat field name to owning layer name.

Used by ``_GlobalConfig.__setattr__`` to route ``config.precision = 6``
to the correct frozen layer, and by ``override(**kwargs)`` to dispatch
kwargs across layers. Raises at module import time if any field name
appears in two layers — fail-fast prevents silent misdispatch.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Type

from pyphi.conf.formalism import FormalismConfig
from pyphi.conf.infrastructure import InfrastructureConfig
from pyphi.conf.numerics import NumericsConfig


class ConfigurationError(ValueError):
    """Raised on config schema problems (collisions, unknown options, etc.)."""


def _build_field_map(layers: list[tuple[str, Type]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for layer_name, layer_cls in layers:
        for f in fields(layer_cls):
            if f.name in out:
                raise ConfigurationError(
                    f"Config field name collision: {f.name!r} appears in both "
                    f"{out[f.name]!r} and {layer_name!r}. Rename one."
                )
            out[f.name] = layer_name
    return out


FIELD_TO_LAYER: dict[str, str] = _build_field_map([
    ("formalism", FormalismConfig),
    ("infrastructure", InfrastructureConfig),
    ("numerics", NumericsConfig),
])
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest test/test_config_layers.py::TestFieldRouting -v`
Expected: 3 passed.

### Task 1.6: Create `pyphi/conf/legacy_global.py` (the `_GlobalConfig` facade) — read-only shim phase

The facade is wired up in two halves: this task adds it side-by-side with the old `PyphiConfig` (read-shim form). Phase 3 cuts over the write path and `override()`. Phase 6 deletes the shim.

**Files:**
- Create: `pyphi/conf/legacy_global.py`
- Modify: `pyphi/conf/__init__.py` (export `config` singleton)

- [ ] **Step 1: Write a passing test for the facade's attribute access**

`test/test_config_layers.py` (append):
```python
class TestGlobalConfigFacade:
    def test_layered_reads_work(self):
        from pyphi.conf import config as new_config
        # Layered reads route to frozen layer dataclasses.
        assert new_config.numerics.precision == 13
        assert new_config.formalism.formalism == "IIT_4_0_2023"
        assert new_config.infrastructure.parallel is False

    def test_snapshot_returns_config_snapshot(self):
        from pyphi.conf import config as new_config
        snap = new_config.snapshot()
        assert snap.numerics.precision == 13

    def test_layered_writes_replace_layer(self):
        from pyphi.conf import config as new_config
        original = new_config.numerics
        new_config.numerics = NumericsConfig(precision=7)
        try:
            assert new_config.numerics.precision == 7
        finally:
            new_config.numerics = original
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest test/test_config_layers.py::TestGlobalConfigFacade -v`
Expected: ImportError on `pyphi.conf.config`.

- [ ] **Step 3: Create `pyphi/conf/legacy_global.py`**

```python
"""Top-level config facade. The ``pyphi.config`` singleton is an instance.

In Phase 1 (this commit), only layered reads work; persistent flat writes
and the ``override()`` context manager are wired in Phase 3 (Task 3.1+).
"""

from __future__ import annotations

from pyphi.conf.formalism import FormalismConfig
from pyphi.conf.infrastructure import InfrastructureConfig
from pyphi.conf.numerics import NumericsConfig
from pyphi.conf.snapshot import ConfigSnapshot


class _GlobalConfig:
    """Holds the three frozen layers; mutable references — assigning
    a new layer instance replaces it wholesale.

    Flat writes (``config.precision = 6``) and ``override()`` land in
    Phase 3.
    """

    formalism: FormalismConfig
    infrastructure: InfrastructureConfig
    numerics: NumericsConfig

    def __init__(self) -> None:
        # Use ``object.__setattr__`` to bypass the routing logic added later;
        # this keeps Phase 1 minimal.
        object.__setattr__(self, "formalism", FormalismConfig())
        object.__setattr__(self, "infrastructure", InfrastructureConfig())
        object.__setattr__(self, "numerics", NumericsConfig())

    def snapshot(self) -> ConfigSnapshot:
        return ConfigSnapshot(
            formalism=self.formalism,
            infrastructure=self.infrastructure,
            numerics=self.numerics,
        )
```

- [ ] **Step 4: Update `pyphi/conf/__init__.py` to export the singleton**

```python
"""Layered configuration system for PyPhi 2.0.

Three frozen dataclasses (`FormalismConfig`, `InfrastructureConfig`,
`NumericsConfig`) wrapped in a `ConfigSnapshot` value type, accessed
through the `config` singleton.
"""

from pyphi.conf.formalism import FormalismConfig
from pyphi.conf.infrastructure import InfrastructureConfig
from pyphi.conf.legacy_global import _GlobalConfig
from pyphi.conf.numerics import NumericsConfig
from pyphi.conf.snapshot import ConfigSnapshot

config = _GlobalConfig()

__all__ = [
    "ConfigSnapshot",
    "FormalismConfig",
    "InfrastructureConfig",
    "NumericsConfig",
    "config",
]
```

- [ ] **Step 5: Run tests to verify pass**

Run: `uv run pytest test/test_config_layers.py -v`
Expected: all of `TestGlobalConfigFacade` pass; full file ~16+ tests green.

### Task 1.7: Verify nothing in PyPhi imports the new `pyphi.conf` package yet

The new package must be entirely additive in Phase 1. Old `pyphi/conf.py` remains the source of truth.

- [ ] **Step 1: Confirm no current callers of `pyphi.conf` (the package)**

```bash
git grep -n 'from pyphi.conf import\|from pyphi.conf\.' pyphi/ test/ | grep -v 'docs/superpowers\|\.pyi$' | grep -v 'test_config_layers'
```

Expected: empty output (only the new test file references it).

- [ ] **Step 2: Confirm full suite still green**

```bash
uv run pytest test/test_golden_regression.py test/test_invariants.py test/test_config_layers.py -q
```

Expected: 17 golden + ~21 invariants + ~16 layer tests pass.

### Task 1.8: Commit Phase 1

- [ ] **Step 1: Stage and commit**

```bash
cd /Users/will/projects/pyphi-p7-kernel-rewrite
git add pyphi/conf/ test/test_config_layers.py
git commit -m "$(cat <<'EOF'
P10 Phase 1: introduce three layered configs + ConfigSnapshot (additive)

Adds the new pyphi/conf/ package with three frozen dataclass layers
(FormalismConfig, InfrastructureConfig, NumericsConfig) wrapped in a
ConfigSnapshot value type. Builds a _FIELD_TO_LAYER map at module import
that raises on field-name collision across layers (currently zero
collisions across all 43 options). Adds the _GlobalConfig facade with
layered reads only — persistent flat writes and override() routing land
in Phase 3.

Old pyphi/conf.py PyphiConfig stays untouched; nothing in PyPhi imports
the new package yet. Full suite (17 golden + 21 invariants + 16 new
layer tests) passes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 2: Cut over read sites, module by module

The new `pyphi.conf.config` is wired in alongside the old `pyphi.config`. Each module's reads are flipped from old → new in turn, with tests after each module to catch regressions immediately.

**Strategy:** A temporary read-shim on `_GlobalConfig` delegates unknown attribute reads to the old `PyphiConfig`. This means `config.PRECISION` (uppercase, old style) still works when read from `pyphi.conf.config` — letting modules switch to the new singleton without simultaneously needing to switch to lowercase layered names. Layered reads (`config.numerics.precision`) also work. Phase 3 removes the shim's read path; Phase 6 deletes the shim entirely.

### Task 2.1: Add the read-shim to `_GlobalConfig`

**Files:**
- Modify: `pyphi/conf/legacy_global.py`
- Modify: `test/test_config_layers.py`

- [ ] **Step 1: Write failing test (append to `TestGlobalConfigFacade`)**

```python
    def test_legacy_uppercase_read_during_phase_2(self):
        """Phase 2 read-shim: config.PRECISION delegates to the old global.

        This shim is removed in Phase 3 (Task 3.1) once all reads have been
        migrated to layered access. Asserting it works confirms cutover-in-progress
        modules don't break when they read config.X (uppercase) on the new singleton.
        """
        from pyphi.conf import config as new_config
        # Uppercase legacy access reads through the shim.
        assert new_config.PRECISION == 13
        assert new_config.PARALLEL is False
        assert new_config.FORMALISM == "IIT_4_0_2023"
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest test/test_config_layers.py::TestGlobalConfigFacade::test_legacy_uppercase_read_during_phase_2 -v`
Expected: AttributeError on `PRECISION`.

- [ ] **Step 3: Add the read-shim to `_GlobalConfig.__getattr__`**

In `pyphi/conf/legacy_global.py`, add:

```python
    def __getattr__(self, name: str):
        """Phase 2 read-shim: delegate uppercase legacy names to old PyphiConfig.

        Removed in Phase 3 once all reads use layered access. The shim emits
        a DEBUG log per access so any straggler call sites are discoverable
        in test logs.
        """
        if name.isupper():
            # Lazy import to avoid circularity; the old conf module imports
            # cleanly even with this package present.
            from pyphi import conf as _old_conf_module
            import logging as _logging
            _logging.getLogger("pyphi.conf").debug(
                "P10 read-shim hit: legacy uppercase access to %r", name
            )
            return getattr(_old_conf_module.config, name)
        raise AttributeError(name)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest test/test_config_layers.py -v`
Expected: all 17 layer tests pass; the new shim test passes.

### Task 2.2: Re-export the new `config` from the top-level `pyphi`

**Files:**
- Modify: `pyphi/__init__.py`

- [ ] **Step 1: Read the current top-level `pyphi/__init__.py` config import line**

```bash
grep -n 'from pyphi.conf\|from pyphi import conf\|pyphi.config\|^config' pyphi/__init__.py
```

- [ ] **Step 2: Replace `from pyphi.conf import config` (or equivalent) with the new package**

Find the existing line that exposes `config` (likely `from pyphi.conf import config` referring to the old module-level `config = PyphiConfig()`). After Phase 1's package, `pyphi.conf` resolves to the new package. To keep the old module accessible during Phase 2, rename the old file's import:

In `pyphi/__init__.py`, change:
```python
from pyphi.conf import config  # OLD: refers to PyphiConfig() in old conf.py
```
to:
```python
from pyphi.conf import config  # NEW: refers to _GlobalConfig() in pyphi/conf/__init__.py
```

The import path is the same string but resolves to the new package. The old file (`pyphi/conf.py`) is shadowed by the new package directory; Python prefers the package. Confirm this is happening:

- [ ] **Step 3: Confirm the new package shadows the old file**

```bash
uv run python -c "import pyphi.conf; print(pyphi.conf.__file__)"
```

Expected: `.../pyphi/conf/__init__.py` (NOT `.../pyphi/conf.py`).

- [ ] **Step 4: Make the old file importable as `pyphi._conf_legacy`**

Rename so the read-shim can still reach it:

```bash
git mv pyphi/conf.py pyphi/_conf_legacy.py
```

Update `pyphi/conf/legacy_global.py`'s shim to import from the new path:

```python
            from pyphi import _conf_legacy as _old_conf_module
```

Update the test to expect this:

In `test/test_config_layers.py`, the shim test passes unchanged because it uses the public path; nothing else should reference `pyphi.conf` as a module file directly. Verify:

```bash
git grep -n 'from pyphi.conf import\|from pyphi import conf\|pyphi\.conf\.' pyphi/ test/ | grep -v 'docs/' | grep -v 'pyphi/conf/'
```

For each hit, confirm it resolves correctly with the new package (most lines like `from pyphi.conf import config` work because the new package re-exports `config`). Lines that import private names (`from pyphi.conf import PyphiConfig`, `from pyphi.conf import Option`) need to update:

```python
# Before:
from pyphi.conf import PyphiConfig
# After:
from pyphi._conf_legacy import PyphiConfig
```

- [ ] **Step 5: Run full suite**

```bash
uv run pytest test/test_golden_regression.py test/test_invariants.py test/test_config_layers.py -q
```

Expected: 17 + 21 + 17 pass.

- [ ] **Step 6: Commit**

```bash
git add pyphi/__init__.py pyphi/conf/legacy_global.py pyphi/_conf_legacy.py test/test_config_layers.py
git rm pyphi/conf.py 2>/dev/null || true   # already moved by git mv
git commit -m "$(cat <<'EOF'
P10 Phase 2 (start): wire new config package + read-shim to legacy

Renames pyphi/conf.py to pyphi/_conf_legacy.py so the new pyphi/conf/
package can take over the public import path. _GlobalConfig.__getattr__
delegates uppercase legacy names (config.PRECISION, etc.) to the legacy
PyphiConfig singleton during Phase 2 of the cutover. Layered reads
(config.numerics.precision) also work via the new frozen-dataclass
layers. Persistent flat writes and override() routing land in Phase 3.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 2.3 — 2.N: Migrate read sites module by module

**Strategy.** Process modules in dependency order (leaves first, top-level entry points last), so each module's tests can run standalone after migration.

**Migration template** (apply once per module):

For module `pyphi/<modpath>.py`:

- [ ] **Step 1: Identify all `config.X` reads in the module**

```bash
git grep -n 'config\.[A-Z_]\+' pyphi/<modpath>.py | grep -v '=' | grep -v override
```

- [ ] **Step 2: Migrate each read to layered access using the rename map**

For each line, replace `config.<UPPER>` → `config.<layer>.<lower>` per Appendix A in the spec. Example diff:

```python
# Before:
if config.PRECISION > 0:
    ...
# After:
if config.numerics.precision > 0:
    ...
```

**Caveat:** Reads inside a `with config.override(X=...): ...` block — the override key stays uppercase here in Phase 2; it's converted in Phase 3 (Task 3.2). The kwarg name is on the override, the read inside the block is on `config.X`.

- [ ] **Step 3: Run that module's tests**

```bash
uv run pytest test/test_<modname>.py -q
```

Expected: all green. Any failure is a missed read or a typo in the layer prefix.

- [ ] **Step 4: Commit per module**

```bash
git add pyphi/<modpath>.py
git commit -m "P10 Phase 2: migrate <modname> reads to layered config access"
```

**Module migration order** (dependency-leaf first):

| Order | Module | Read count (approximate; confirm in Phase 0 inventory) |
|---|---|---|
| 1 | `pyphi/utils.py` | small |
| 2 | `pyphi/combinatorics.py` | small |
| 3 | `pyphi/data_structures/pyphi_float.py` | reads `PRECISION` only |
| 4 | `pyphi/distribution.py` | medium |
| 5 | `pyphi/metrics/distribution.py` | medium |
| 6 | `pyphi/metrics/ces.py` | small |
| 7 | `pyphi/partition.py` | medium |
| 8 | `pyphi/repertoire.py` | small |
| 9 | `pyphi/cache/__init__.py` | small (CACHE_REPERTOIRES, CACHE_POTENTIAL_PURVIEWS) |
| 10 | `pyphi/cache/cache_utils.py` | small (MAXIMUM_CACHE_MEMORY_PERCENTAGE) |
| 11 | `pyphi/parallel/tree.py` + `pyphi/parallel/progress.py` | medium |
| 12 | `pyphi/network.py` | small |
| 13 | `pyphi/jsonify.py` | small |
| 14 | `pyphi/models/*.py` | medium |
| 15 | `pyphi/formalism/iit3/*.py` + `pyphi/formalism/iit4/*.py` | medium |
| 16 | `pyphi/formalism/queries.py` + `pyphi/formalism/__init__.py` | small |
| 17 | `pyphi/core/*.py` | small |
| 18 | `pyphi/compute/*.py` | medium |
| 19 | `pyphi/actual.py` | medium |
| 20 | `pyphi/relations.py` | medium |
| 21 | `pyphi/visualize/*.py` | medium |

**Per-module tasks 2.3 through 2.23** follow the migration template above. At the end of each, the suite must be green for that module.

### Task 2.24: Migrate test files' reads

Once all production code uses layered access, test files do the same.

- [ ] **Step 1: Find all uppercase config reads in tests**

```bash
git grep -n 'config\.[A-Z_]\+' test/ | grep -v override | grep -v '='
```

- [ ] **Step 2: Apply rename-map to each line**

Same pattern as production migrations. `config.PRECISION` → `config.numerics.precision`.

- [ ] **Step 3: Run full suite**

```bash
uv run pytest -q
```

Expected: full suite green; the read-shim should now have zero hits in DEBUG logs.

- [ ] **Step 4: Confirm shim is unused**

```bash
uv run pytest test/test_golden_regression.py -q --log-cli-level=DEBUG 2>&1 | grep "P10 read-shim hit" | head
```

Expected: zero shim-hit log lines for the golden run.

- [ ] **Step 5: Commit**

```bash
git add test/
git commit -m "P10 Phase 2 end: migrate test reads to layered config access"
```

---

## Phase 3: Cut over write sites + `override()` compat

### Task 3.1: Implement `_GlobalConfig.__setattr__` for flat writes + remove read-shim

**Files:**
- Modify: `pyphi/conf/legacy_global.py`
- Modify: `test/test_config_layers.py` (drop the shim test, add write tests)

- [ ] **Step 1: Drop the read-shim test (no longer required)**

Delete `test_legacy_uppercase_read_during_phase_2` from `test/test_config_layers.py`.

- [ ] **Step 2: Write failing tests for flat-write routing**

Append to `test/test_config_layers.py`:
```python
class TestGlobalConfigFlatWrites:
    def test_flat_write_routes_to_layer(self):
        from pyphi.conf import config as new_config
        original_precision = new_config.numerics.precision
        new_config.precision = 7
        try:
            assert new_config.numerics.precision == 7
            # The layer instance is replaced wholesale; old layer unchanged.
            assert original_precision != 7
        finally:
            new_config.precision = original_precision

    def test_flat_write_unknown_name_raises(self):
        from pyphi.conf import config as new_config
        from pyphi.conf._field_routing import ConfigurationError
        with pytest.raises(ConfigurationError, match="Unknown config option"):
            new_config.nonexistent_field = 0  # type: ignore[attr-defined]

    def test_flat_write_uppercase_legacy_name_raises(self):
        # Phase 3 removes the shim; legacy uppercase access raises.
        from pyphi.conf import config as new_config
        with pytest.raises(AttributeError):
            new_config.PRECISION = 6  # type: ignore[attr-defined]
```

- [ ] **Step 3: Run tests to verify failure**

Run: `uv run pytest test/test_config_layers.py::TestGlobalConfigFlatWrites -v`
Expected: 3 failures (write isn't routed yet; old shim still allows uppercase).

- [ ] **Step 4: Replace `_GlobalConfig.__setattr__` and remove read-shim**

Edit `pyphi/conf/legacy_global.py`:

```python
"""Top-level config facade. The ``pyphi.config`` singleton is an instance.

After Phase 3:
- Layered reads: ``config.numerics.precision``
- Flat persistent writes: ``config.precision = 6`` (routes via _FIELD_TO_LAYER)
- Scoped writes: ``with config.override(precision=6): ...``

Frozen layers can't be mutated in place; writes rebuild the relevant layer
via ``dataclasses.replace`` and assign it back.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from pyphi.conf._field_routing import FIELD_TO_LAYER, ConfigurationError
from pyphi.conf.formalism import FormalismConfig
from pyphi.conf.infrastructure import InfrastructureConfig
from pyphi.conf.numerics import NumericsConfig
from pyphi.conf.snapshot import ConfigSnapshot

_LAYER_ATTRS = frozenset({"formalism", "infrastructure", "numerics"})


class _GlobalConfig:
    """Holds the three frozen layers; routes flat writes by field name."""

    formalism: FormalismConfig
    infrastructure: InfrastructureConfig
    numerics: NumericsConfig

    def __init__(self) -> None:
        object.__setattr__(self, "formalism", FormalismConfig())
        object.__setattr__(self, "infrastructure", InfrastructureConfig())
        object.__setattr__(self, "numerics", NumericsConfig())

    def __setattr__(self, name: str, value: Any) -> None:
        # Wholesale layer replacement (rare; mostly for tests).
        if name in _LAYER_ATTRS:
            object.__setattr__(self, name, value)
            return
        # Flat-name route via FIELD_TO_LAYER.
        layer_name = FIELD_TO_LAYER.get(name)
        if layer_name is None:
            raise ConfigurationError(
                f"Unknown config option: {name!r}. "
                f"See changelog.d/p10-config-split.refactor.md for the rename map."
            )
        current_layer = getattr(self, layer_name)
        new_layer = replace(current_layer, **{name: value})
        object.__setattr__(self, layer_name, new_layer)

    def snapshot(self) -> ConfigSnapshot:
        return ConfigSnapshot(
            formalism=self.formalism,
            infrastructure=self.infrastructure,
            numerics=self.numerics,
        )
```

Note: the old `__getattr__` shim is removed; uppercase reads now raise `AttributeError`. This is fine because Phase 2 migrated all reads.

- [ ] **Step 5: Run tests to verify pass**

Run: `uv run pytest test/test_config_layers.py -v`
Expected: all green; 3 new flat-write tests pass.

- [ ] **Step 6: Run full suite to surface stragglers**

```bash
uv run pytest -q
```

Expected: green. Any failure means a Phase 2 read missed a site that's now an `AttributeError`. Fix the missed site, commit it as a Phase 2 follow-up.

### Task 3.2: Implement `_GlobalConfig.override()` with kwarg routing

**Files:**
- Modify: `pyphi/conf/legacy_global.py`
- Create: `test/test_config_override.py`

- [ ] **Step 1: Write failing tests**

`test/test_config_override.py`:
```python
"""Tests for the layered ``config.override()`` context manager."""

from __future__ import annotations

import pytest

from pyphi.conf import config
from pyphi.conf._field_routing import ConfigurationError


class TestOverrideTopLevel:
    def test_single_layer_override(self):
        original = config.numerics.precision
        with config.override(precision=7):
            assert config.numerics.precision == 7
        assert config.numerics.precision == original

    def test_multi_layer_override(self):
        with config.override(precision=6, parallel=True, repertoire_distance="EMD"):
            assert config.numerics.precision == 6
            assert config.infrastructure.parallel is True
            assert config.formalism.repertoire_distance == "EMD"
        # Restored
        assert config.numerics.precision == 13
        assert config.infrastructure.parallel is False

    def test_override_unknown_name_raises(self):
        with pytest.raises(ConfigurationError, match="Unknown config option"):
            with config.override(nonexistent_field=0):
                pass

    def test_override_restores_on_exception(self):
        original = config.numerics.precision
        with pytest.raises(RuntimeError):
            with config.override(precision=99):
                raise RuntimeError("boom")
        assert config.numerics.precision == original


class TestOverridePerLayer:
    def test_per_layer_override(self):
        original = config.numerics.precision
        with config.numerics.override(precision=7):
            assert config.numerics.precision == 7
        assert config.numerics.precision == original

    def test_per_layer_override_only_affects_that_layer(self):
        with config.numerics.override(precision=7):
            assert config.formalism.formalism == "IIT_4_0_2023"  # untouched
            assert config.infrastructure.parallel is False  # untouched

    def test_per_layer_override_field_not_in_layer_raises(self):
        # config.numerics has no `parallel` field.
        with pytest.raises(ConfigurationError):
            with config.numerics.override(parallel=True):
                pass
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest test/test_config_override.py -v`
Expected: AttributeError on `config.override`.

- [ ] **Step 3: Add `override()` to `_GlobalConfig` and per-layer override methods**

In `pyphi/conf/legacy_global.py`, append:

```python
import contextlib
from collections import defaultdict


class _LayeredOverride(contextlib.AbstractContextManager):
    def __init__(self, target: "_GlobalConfig", by_layer: dict[str, dict[str, Any]]) -> None:
        self._target = target
        self._by_layer = by_layer
        self._saved: dict[str, Any] = {}

    def __enter__(self) -> "_LayeredOverride":
        for layer_name, kwargs in self._by_layer.items():
            current = getattr(self._target, layer_name)
            self._saved[layer_name] = current
            object.__setattr__(self._target, layer_name, replace(current, **kwargs))
        return self

    def __exit__(self, *exc: Any) -> None:
        for layer_name, saved in self._saved.items():
            object.__setattr__(self._target, layer_name, saved)


def _override_method(target: "_GlobalConfig", **kwargs: Any) -> _LayeredOverride:
    by_layer: dict[str, dict[str, Any]] = defaultdict(dict)
    for name, value in kwargs.items():
        layer_name = FIELD_TO_LAYER.get(name)
        if layer_name is None:
            raise ConfigurationError(f"Unknown config option: {name!r}")
        by_layer[layer_name][name] = value
    return _LayeredOverride(target, by_layer)


# Bind override() onto the class:
_GlobalConfig.override = _override_method  # type: ignore[attr-defined]
```

For per-layer override (`config.numerics.override(precision=6)`), add a free function attached to each frozen-dataclass type via a wrapper class. Simplest: a sibling helper that the user uses by calling `config.numerics.override(...)` — but frozen dataclasses can't take new methods. Use a thin wrapper:

In `pyphi/conf/legacy_global.py`, replace direct field access pattern by making the layer attributes return a wrapper. Actually simpler: install `override` on each layer type via a monkey-patch in `pyphi/conf/__init__.py`:

```python
# In pyphi/conf/__init__.py, after the imports:

def _make_layer_override(layer_cls):
    def _override(self, **kwargs):
        # This is called on the *instance* layer; we need to know which
        # global slot this layer occupies. Look it up by type.
        from pyphi.conf import config as _cfg
        for slot in ("formalism", "infrastructure", "numerics"):
            if isinstance(getattr(_cfg, slot), layer_cls):
                return _LayeredOverride(_cfg, {slot: kwargs})
        raise RuntimeError(f"No global slot for layer {layer_cls.__name__}")
    return _override


# Frozen dataclasses won't accept new attributes via assignment, but methods
# bound at class-creation time work fine. Use a metaclass-free approach by
# attaching the method directly:
FormalismConfig.override = _make_layer_override(FormalismConfig)  # type: ignore[attr-defined]
InfrastructureConfig.override = _make_layer_override(InfrastructureConfig)  # type: ignore[attr-defined]
NumericsConfig.override = _make_layer_override(NumericsConfig)  # type: ignore[attr-defined]
```

Note: `from pyphi.conf import config as _cfg` inside `_override` is a forward reference that works because by the time `_override` is *called*, `config` exists. To break the import-time circular dependency between the layer modules and `legacy_global.py`, the `_LayeredOverride` import in `_make_layer_override` must be done at call time too:

```python
def _make_layer_override(layer_cls):
    def _override(self, **kwargs):
        from pyphi.conf.legacy_global import _LayeredOverride
        from pyphi.conf import config as _cfg
        for slot in ("formalism", "infrastructure", "numerics"):
            if isinstance(getattr(_cfg, slot), layer_cls):
                return _LayeredOverride(_cfg, {slot: kwargs})
        raise RuntimeError(f"No global slot for layer {layer_cls.__name__}")
    return _override
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest test/test_config_override.py -v`
Expected: 7 tests pass.

### Task 3.3: Migrate all `config.override(UPPER=...)` call sites to lowercase

**Files:** ~30+ test and production files (find via grep).

- [ ] **Step 1: Find all override call sites with uppercase kwargs**

```bash
git grep -n 'config\.override(' pyphi/ test/ | grep -E '[A-Z_]+\s*='
```

- [ ] **Step 2: Apply lowercase rename per Appendix A**

Each `config.override(PRECISION=N)` → `config.override(precision=N)`. Each `config.override(REPERTOIRE_DISTANCE="...")` → `config.override(repertoire_distance="...")`. Same for all 43 options.

- [ ] **Step 3: Run full suite**

```bash
uv run pytest -q
```

Expected: green. Any failure is a missed override site or a typo in the lowercase name.

- [ ] **Step 4: Commit Phase 3**

```bash
git add pyphi/conf/legacy_global.py pyphi/conf/__init__.py test/test_config_override.py test/test_config_layers.py pyphi/ test/
git commit -m "$(cat <<'EOF'
P10 Phase 3: cut over write sites + override() routing

_GlobalConfig.__setattr__ routes flat writes (config.precision = 6) via
the FIELD_TO_LAYER map, rebuilding the affected layer via
dataclasses.replace. config.override(precision=6, parallel=True) accepts
kwargs across layers and dispatches each to the right layer; per-layer
config.numerics.override(precision=6) also works. Unknown names raise
ConfigurationError. The Phase 2 read-shim is removed — all reads now
use layered access.

Migrates all config.override(UPPER=...) call sites to lowercase across
pyphi/ and test/.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 4: PhiFormalism wiring

The active `PhiFormalism` instance gains a `FormalismConfig` field. Concrete formalisms become frozen dataclasses; the registry stores factories that build a formalism with a given config.

### Task 4.1: Extend the `PhiFormalism` Protocol

**Files:**
- Modify: `pyphi/formalism/base.py`
- Modify: `test/test_formalism.py` (or create `test/test_formalism_config.py`)

- [ ] **Step 1: Write failing test (create `test/test_formalism_config.py`)**

```python
"""Tests for PhiFormalism + FormalismConfig integration (P10)."""

from __future__ import annotations

import pytest

from pyphi.conf.formalism import FormalismConfig
from pyphi.formalism import FORMALISM_REGISTRY
from pyphi.formalism.base import PhiFormalism


class TestPhiFormalismHoldsConfig:
    def test_formalism_has_config_field(self):
        cfg = FormalismConfig(formalism="IIT_4_0_2023")
        formalism = FORMALISM_REGISTRY.build("IIT_4_0_2023", cfg)
        assert isinstance(formalism, PhiFormalism)
        assert formalism.config is cfg

    def test_changing_config_field_yields_new_formalism(self):
        cfg_a = FormalismConfig(repertoire_distance="GENERALIZED_INTRINSIC_DIFFERENCE")
        cfg_b = FormalismConfig(repertoire_distance="EMD")
        a = FORMALISM_REGISTRY.build("IIT_3_0", cfg_a)
        b = FORMALISM_REGISTRY.build("IIT_3_0", cfg_b)
        assert a.config.repertoire_distance != b.config.repertoire_distance

    def test_formalism_is_frozen(self):
        from dataclasses import FrozenInstanceError
        cfg = FormalismConfig()
        formalism = FORMALISM_REGISTRY.build("IIT_4_0_2023", cfg)
        with pytest.raises(FrozenInstanceError):
            formalism.config = FormalismConfig()  # type: ignore[misc]
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest test/test_formalism_config.py -v`
Expected: AttributeError or registry method missing.

- [ ] **Step 3: Add `config` field to `PhiFormalism` Protocol**

In `pyphi/formalism/base.py`:

```python
from pyphi.conf.formalism import FormalismConfig


@runtime_checkable
class PhiFormalism(Protocol):
    """The minimum shape every formalism satisfies."""

    name: str
    default_metric: str
    compatible_metrics: frozenset[str]
    partition_scheme: str | None
    config: FormalismConfig    # NEW

    def evaluate_mechanism(self, ...): ...
    # ... existing methods ...
```

### Task 4.2: Convert concrete formalisms to frozen dataclasses

**Files:**
- Modify: `pyphi/formalism/iit3/formalism.py`
- Modify: `pyphi/formalism/iit4/formalism.py`

- [ ] **Step 1: Inspect existing IIT 3.0 formalism class**

```bash
sed -n '1,50p' pyphi/formalism/iit3/formalism.py
```

- [ ] **Step 2: Convert `IIT3Formalism` to frozen dataclass with `config` field**

In `pyphi/formalism/iit3/formalism.py`:
```python
from dataclasses import dataclass

from pyphi.conf.formalism import FormalismConfig


@dataclass(frozen=True)
class IIT3Formalism:
    """IIT 3.0 formalism (Oizumi et al. 2014)."""

    config: FormalismConfig
    name: str = "IIT_3_0"
    default_metric: str = "EMD"
    compatible_metrics: frozenset[str] = frozenset({
        "EMD", "KLD", "L1", "PSI",
    })
    partition_scheme: str | None = "ALL"

    def evaluate_mechanism(self, subsystem, direction, mechanism, purview, **kwargs):
        # ... existing logic, reading from self.config instead of pyphi.config ...
        ...

    def evaluate_mechanism_partition(self, subsystem, direction, mechanism, purview, partition, **kwargs):
        ...

    def evaluate_system(self, subsystem, **kwargs):
        ...

    def build_phi_structure(self, subsystem, **kwargs):
        ...
```

Apply the same pattern to `IIT4_2023Formalism` and `IIT4_2026Formalism` in `pyphi/formalism/iit4/formalism.py`.

- [ ] **Step 3: Update method bodies to read `self.config.X` instead of `pyphi.config.formalism.X`**

For each `evaluate_*` method, replace any read of the live global with a read of `self.config`. Example:

```python
# Before:
def evaluate_mechanism(self, subsystem, direction, mechanism, purview, **kwargs):
    metric = pyphi.config.formalism.repertoire_distance
    ...

# After:
def evaluate_mechanism(self, subsystem, direction, mechanism, purview, **kwargs):
    metric = self.config.repertoire_distance
    ...
```

This is the architectural payoff: methods take their parameters from the bundled config, not the live global. Workers under P11 will receive the formalism instance with its config attached.

### Task 4.3: Add `FORMALISM_REGISTRY.build(name, config)` factory method

**Files:**
- Modify: `pyphi/formalism/__init__.py`

- [ ] **Step 1: Add the build method to `FormalismRegistry`**

In `pyphi/formalism/__init__.py`:
```python
from pyphi.conf.formalism import FormalismConfig
from pyphi.formalism.iit3.formalism import IIT3Formalism
from pyphi.formalism.iit4.formalism import IIT4_2023Formalism, IIT4_2026Formalism


_FACTORY = {
    "IIT_3_0": IIT3Formalism,
    "IIT_4_0_2023": IIT4_2023Formalism,
    "IIT_4_0_2026": IIT4_2026Formalism,
}


class FormalismRegistry:
    def build(self, name: str, config: FormalismConfig) -> PhiFormalism:
        cls = _FACTORY.get(name)
        if cls is None:
            raise KeyError(f"Unknown formalism: {name!r}; known: {sorted(_FACTORY)}")
        return cls(config=config)


FORMALISM_REGISTRY = FormalismRegistry()
```

### Task 4.4: Wire `_GlobalConfig` to rebuild active formalism on change

**Files:**
- Modify: `pyphi/conf/legacy_global.py`

- [ ] **Step 1: Add `active_formalism` cached property + invalidation**

In `pyphi/conf/legacy_global.py`:
```python
class _GlobalConfig:
    # ... existing ...

    @property
    def active_formalism(self):
        """The current PhiFormalism, built from FORMALISM_REGISTRY using
        the current FormalismConfig.

        Rebuilt lazily when the formalism layer changes — _last_formalism_id
        tracks the FormalismConfig instance identity to detect changes.
        """
        # Lazy import to avoid circularity at module-load time.
        from pyphi.formalism import FORMALISM_REGISTRY
        cached = getattr(self, "_active_formalism_cache", None)
        if cached is None or cached[0] is not self.formalism:
            built = FORMALISM_REGISTRY.build(self.formalism.formalism, self.formalism)
            object.__setattr__(self, "_active_formalism_cache", (self.formalism, built))
            return built
        return cached[1]
```

- [ ] **Step 2: Run tests**

```bash
uv run pytest test/test_formalism_config.py test/test_golden_regression.py -q
```

Expected: green; golden 17/17 unchanged (no semantic change in formalism behavior).

- [ ] **Step 3: Migrate all `pyphi.formalism.get_active_formalism()` (or equivalent) call sites to use `config.active_formalism`**

```bash
git grep -n 'get_active_formalism\|active_formalism' pyphi/ test/
```

For each, ensure the source-of-truth path is `config.active_formalism`.

- [ ] **Step 4: Commit Phase 4**

```bash
git add pyphi/conf/legacy_global.py pyphi/formalism/ test/test_formalism_config.py
git commit -m "$(cat <<'EOF'
P10 Phase 4: PhiFormalism owns FormalismConfig via composition

Concrete formalisms (IIT3Formalism, IIT4_2023Formalism, IIT4_2026Formalism)
become frozen dataclasses with a FormalismConfig field. FORMALISM_REGISTRY
gains a build(name, config) factory. _GlobalConfig.active_formalism is a
cached property that rebuilds the formalism instance whenever the formalism
layer changes (detected by config dataclass identity comparison).

Method bodies in formalisms now read self.config.X instead of
pyphi.config.formalism.X — the dispatch site stays stable, but the formalism
instance carries its own parameter bundle. Sets up workers under P11 to
receive the formalism with its config attached, eliminating implicit
global-via-pickle.

Golden 17/17 unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 5: Result-object snapshot wiring

Every result object gains a `.config: ConfigSnapshot` field. Construction sites snapshot the live global once and thread the snapshot down. jsonify learns to serialize the four frozen types.

### Task 5.1: Add `config: ConfigSnapshot` field to `SIA`

**Files:**
- Modify: `pyphi/models/subsystem.py` (or wherever `SystemIrreducibilityAnalysis` lives post-P8)
- Create: `test/test_result_config_snapshot.py`

- [ ] **Step 1: Find SIA's current definition**

```bash
git grep -n 'class SystemIrreducibilityAnalysis\|class SIA' pyphi/
```

- [ ] **Step 2: Write failing test**

`test/test_result_config_snapshot.py`:
```python
"""Tests that every result object carries a frozen ConfigSnapshot (P10)."""

from __future__ import annotations

import pytest

from pyphi import config, examples
from pyphi.conf.snapshot import ConfigSnapshot
from pyphi.formalism.queries import sia
from pyphi.core.candidate_system import CandidateSystem
from pyphi.core.causal_model import CausalModel


@pytest.fixture
def cs():
    network = examples.basic_network()
    return CandidateSystem(
        causal_model=CausalModel.from_network(network),
        state=(1, 0, 0),
        node_indices=(0, 1, 2),
    )


class TestSIASnapshot:
    def test_sia_has_config_snapshot(self, cs):
        result = sia(cs)
        assert isinstance(result.config, ConfigSnapshot)

    def test_snapshot_records_precision_at_construction(self, cs):
        with config.override(precision=7):
            result = sia(cs)
        assert result.config.numerics.precision == 7

    def test_mutating_global_after_construction_doesnt_change_snapshot(self, cs):
        result = sia(cs)
        original = result.config.numerics.precision
        try:
            config.precision = 99
            assert result.config.numerics.precision == original
        finally:
            config.precision = 13
```

- [ ] **Step 3: Run tests to verify failure**

Run: `uv run pytest test/test_result_config_snapshot.py::TestSIASnapshot -v`
Expected: AttributeError on `result.config`.

- [ ] **Step 4: Add `config` field to SIA**

```python
# In pyphi/models/subsystem.py (or current location)
from pyphi.conf.snapshot import ConfigSnapshot

@dataclass(frozen=True)
class SystemIrreducibilityAnalysis:
    phi: float
    signed_phi: float
    cut: SystemPartition
    # ... existing fields ...
    config: ConfigSnapshot
```

- [ ] **Step 5: Update SIA construction sites to thread snapshot**

```bash
git grep -n 'SystemIrreducibilityAnalysis(\|SIA(' pyphi/
```

For each construction site, take the snapshot once at the entry point (typically `pyphi.formalism.queries.sia` or the formalism's `evaluate_system` method), then thread the snapshot down. Example:

```python
# Before:
def sia(cs):
    formalism = config.active_formalism
    return formalism.evaluate_system(cs)

# After:
def sia(cs):
    formalism = config.active_formalism
    snap = config.snapshot()
    return formalism.evaluate_system(cs, _config_snapshot=snap)
```

The formalism's `evaluate_system` accepts the snapshot via kwargs and passes it to `SIA(...)`. Same pattern for every result type.

- [ ] **Step 6: Run tests**

Run: `uv run pytest test/test_result_config_snapshot.py::TestSIASnapshot -v`
Expected: 3 pass.

### Task 5.2: Apply same pattern to `RepertoireIrreducibilityAnalysis`, `MaximallyIrreducibleCauseOrEffect`, `Distinction`, `Concept`, `CauseEffectStructure`, `PhiStructure`

**Files:**
- Modify: `pyphi/models/mechanism.py`
- Modify: `pyphi/models/ces.py`
- Modify: `pyphi/models/phi_structure.py`
- Modify: `test/test_result_config_snapshot.py`

For each result type, repeat the Task 5.1 pattern:

- [ ] **Step 1: Add failing tests for that type's snapshot**

For each type, append a `Test<Type>Snapshot` class to `test/test_result_config_snapshot.py` with the same three tests (has snapshot; records at construction; immune to later global writes).

- [ ] **Step 2: Add `config: ConfigSnapshot` field to the dataclass**

- [ ] **Step 3: Update construction sites to thread snapshot**

For inner-loop constructions (`Distinction`, `Concept`, `MICE`), the snapshot is taken once at the outer scope and shared by reference across all inner constructions — no per-inner-call cost.

- [ ] **Step 4: Run tests after each type**

```bash
uv run pytest test/test_result_config_snapshot.py -v
```

### Task 5.3: jsonify serializers for the four frozen config types

**Files:**
- Modify: `pyphi/jsonify.py`
- Create: `test/test_config_jsonify.py`

- [ ] **Step 1: Write failing test**

`test/test_config_jsonify.py`:
```python
"""Tests that ConfigSnapshot round-trips through jsonify (P10)."""

from __future__ import annotations

import json

from pyphi import jsonify
from pyphi.conf.formalism import FormalismConfig
from pyphi.conf.infrastructure import InfrastructureConfig
from pyphi.conf.numerics import NumericsConfig
from pyphi.conf.snapshot import ConfigSnapshot


class TestConfigJsonify:
    def test_numerics_round_trip(self):
        original = NumericsConfig(precision=7)
        text = jsonify.dumps(original)
        recovered = jsonify.loads(text)
        assert recovered == original

    def test_formalism_round_trip(self):
        original = FormalismConfig(repertoire_distance="EMD")
        text = jsonify.dumps(original)
        recovered = jsonify.loads(text)
        assert recovered == original

    def test_infrastructure_round_trip(self):
        original = InfrastructureConfig(parallel=True, parallel_workers=8)
        text = jsonify.dumps(original)
        recovered = jsonify.loads(text)
        assert recovered == original

    def test_snapshot_round_trip(self):
        original = ConfigSnapshot(
            formalism=FormalismConfig(repertoire_distance="EMD"),
            infrastructure=InfrastructureConfig(parallel=True),
            numerics=NumericsConfig(precision=7),
        )
        text = jsonify.dumps(original)
        recovered = jsonify.loads(text)
        assert recovered == original
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest test/test_config_jsonify.py -v`
Expected: jsonify doesn't know how to serialize the frozen dataclasses.

- [ ] **Step 3: Register serializers in `pyphi/jsonify.py`**

The existing `jsonify` infrastructure handles dataclasses via `dataclass.asdict` + a class registry. Add:

```python
# In pyphi/jsonify.py, near the existing class registry:
from pyphi.conf.formalism import FormalismConfig
from pyphi.conf.infrastructure import InfrastructureConfig
from pyphi.conf.numerics import NumericsConfig
from pyphi.conf.snapshot import ConfigSnapshot

# Register each frozen config type with the existing jsonify class registry.
# (Pattern follows other dataclass registrations already present in this file.)
register_class(NumericsConfig)
register_class(FormalismConfig)
register_class(InfrastructureConfig)
register_class(ConfigSnapshot)
```

(The actual function name `register_class` may differ; mirror the pattern used elsewhere in `jsonify.py`.)

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest test/test_config_jsonify.py -v`
Expected: 4 pass.

### Task 5.4: Phase 5 acceptance + commit

- [ ] **Step 1: Run full result-snapshot tests**

```bash
uv run pytest test/test_result_config_snapshot.py test/test_config_jsonify.py -v
```

Expected: green.

- [ ] **Step 2: Run golden + invariants**

```bash
uv run pytest test/test_golden_regression.py test/test_invariants.py -q
```

Expected: 17 + 21 pass.

- [ ] **Step 3: Commit Phase 5**

```bash
git add pyphi/models/ pyphi/formalism/ pyphi/jsonify.py test/test_result_config_snapshot.py test/test_config_jsonify.py
git commit -m "$(cat <<'EOF'
P10 Phase 5: attach ConfigSnapshot to every result object

SIA, RepertoireIrreducibilityAnalysis, MaximallyIrreducibleCauseOrEffect,
Distinction, Concept, CauseEffectStructure, and PhiStructure all gain a
config: ConfigSnapshot field set at construction time. Construction sites
snapshot the live global once at the top of the call chain and thread the
snapshot down — inner-loop constructions share the snapshot by reference,
so per-distinction memory cost is one pointer (~8 bytes).

jsonify learns to serialize the four frozen config types via existing
class-registration pattern, so result objects round-trip cleanly to
disk and back.

Golden 17/17 + invariants 21 unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 6: Delete old `PyphiConfig` + nested-only YAML + acceptance

### Task 6.1: Restructure `pyphi_config.yml` and `pyphi_config_3.0.yml`

**Files:**
- Modify: `pyphi_config.yml`
- Modify: `pyphi_config_3.0.yml`

- [ ] **Step 1: Write the new nested `pyphi_config.yml`**

```yaml
# PyPhi 2.0 configuration
# ~~~~~~~~~~~~~~~~~~~~~~~
# See ``pyphi.conf`` documentation for layer descriptions.
---
formalism:
  formalism: IIT_4_0_2023
  assume_cuts_cannot_create_new_concepts: false
  repertoire_distance: GENERALIZED_INTRINSIC_DIFFERENCE
  repertoire_distance_specification: GENERALIZED_INTRINSIC_DIFFERENCE
  repertoire_distance_differentiation: GENERALIZED_INTRINSIC_DIFFERENCE
  ces_distance: SUM_SMALL_PHI
  actual_causation_measure: PMI
  partition_type: ALL
  system_partition_type: SET_UNI/BI
  system_partition_include_complete: false
  system_cuts: 3.0_STYLE
  distinction_phi_normalization: NUM_CONNECTIONS_CUT
  relation_computation: CONCRETE
  state_tie_resolution: PHI
  mip_tie_resolution: [NORMALIZED_PHI, NEGATIVE_PHI]
  purview_tie_resolution: PHI
  shortcircuit_sia: true
  single_micro_nodes_with_selfloops_have_phi: true

infrastructure:
  parallel: false
  parallel_workers: -1
  parallel_backend: local
  parallel_complex_evaluation:
    parallel: false
    sequential_threshold: 16
    chunksize: 64
    progress: true
  parallel_cut_evaluation:
    parallel: false
    sequential_threshold: 1024
    chunksize: 4096
    progress: false
  parallel_concept_evaluation:
    parallel: false
    sequential_threshold: 64
    chunksize: 256
    progress: true
  parallel_purview_evaluation:
    parallel: false
    sequential_threshold: 64
    chunksize: 256
    progress: true
  parallel_mechanism_partition_evaluation:
    parallel: false
    sequential_threshold: 1024
    chunksize: 4096
    progress: true
  parallel_relation_evaluation:
    parallel: false
    sequential_threshold: 1024
    chunksize: 4096
    progress: true
  maximum_cache_memory_percentage: 50
  cache_repertoires: true
  cache_potential_purviews: true
  clear_subsystem_caches_after_computing_sia: false
  log_file: pyphi.log
  log_file_level: INFO
  log_stdout_level: WARNING
  progress_bars: true
  repr_verbosity: 2
  print_fractions: true
  label_separator: ""
  welcome_off: false
  validate_subsystem_states: true
  validate_conditional_independence: true
  validate_json_version: true

numerics:
  precision: 13
```

- [ ] **Step 2: Apply the IIT 3.0 overrides to `pyphi_config_3.0.yml`**

Inspect the existing flat `pyphi_config_3.0.yml`, identify which keys differ from the default, and produce the nested form mirroring those differences.

### Task 6.2: Implement nested YAML loader with friendly old-format error

**Files:**
- Create: `pyphi/conf/_io.py`
- Modify: `pyphi/conf/legacy_global.py`
- Create: `test/test_config_yaml.py`

- [ ] **Step 1: Write failing tests**

`test/test_config_yaml.py`:
```python
"""Tests for the layered YAML loader (P10)."""

from __future__ import annotations

import textwrap

import pytest

from pyphi.conf import config
from pyphi.conf._field_routing import ConfigurationError


class TestNestedYAMLLoader:
    def test_load_nested(self, tmp_path):
        path = tmp_path / "config.yml"
        path.write_text(textwrap.dedent("""\
        ---
        formalism:
          repertoire_distance: EMD
        numerics:
          precision: 7
        """))
        config.load_yaml(str(path))
        try:
            assert config.formalism.repertoire_distance == "EMD"
            assert config.numerics.precision == 7
        finally:
            config.formalism = type(config.formalism)()
            config.numerics = type(config.numerics)()

    def test_old_flat_format_raises_with_rename_map(self, tmp_path):
        path = tmp_path / "config.yml"
        path.write_text(textwrap.dedent("""\
        ---
        PRECISION: 13
        PARALLEL: false
        """))
        with pytest.raises(ConfigurationError, match="rename map"):
            config.load_yaml(str(path))

    def test_unknown_top_level_key_raises(self, tmp_path):
        path = tmp_path / "config.yml"
        path.write_text("nonexistent: 1\n")
        with pytest.raises(ConfigurationError, match="Unknown top-level YAML key"):
            config.load_yaml(str(path))
```

- [ ] **Step 2: Run tests to verify failure**

Run: `uv run pytest test/test_config_yaml.py -v`
Expected: AttributeError on `config.load_yaml`.

- [ ] **Step 3: Create `pyphi/conf/_io.py` with the loader**

```python
"""YAML I/O for the layered config system."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from pyphi.conf._field_routing import ConfigurationError


def load_yaml(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load a nested-format YAML config file.

    Raises ConfigurationError if the file uses the old 1.x flat format
    (any uppercase top-level keys), with a rename-map pointer in the
    error message.
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigurationError(f"Config file {path!r} doesn't parse to a dict.")
    upper_keys = [k for k in data if isinstance(k, str) and k.isupper()]
    if upper_keys:
        raise ConfigurationError(
            f"Config file {path!r} uses the 1.x flat format (e.g., {upper_keys[0]!r}). "
            f"In 2.0, options are grouped by layer (formalism / infrastructure / numerics). "
            f"See changelog.d/p10-config-split.refactor.md for the rename map."
        )
    known = {"formalism", "infrastructure", "numerics"}
    unknown = set(data) - known
    if unknown:
        raise ConfigurationError(
            f"Unknown top-level YAML key(s): {sorted(unknown)}. "
            f"Expected one of: {sorted(known)}."
        )
    return data
```

- [ ] **Step 4: Add `load_yaml` and `to_yaml` to `_GlobalConfig`**

In `pyphi/conf/legacy_global.py`:
```python
    def load_yaml(self, path: str | Path) -> None:
        from dataclasses import replace as _replace
        from pyphi.conf._io import load_yaml as _load
        data = _load(path)
        for layer_name, fields_dict in data.items():
            current = getattr(self, layer_name)
            object.__setattr__(self, layer_name, _replace(current, **fields_dict))

    def to_yaml(self, path: str | Path) -> None:
        from dataclasses import asdict as _asdict
        import yaml as _yaml
        data = {
            "formalism": _asdict(self.formalism),
            "infrastructure": _asdict(self.infrastructure),
            "numerics": _asdict(self.numerics),
        }
        with open(path, "w") as f:
            _yaml.safe_dump(data, f)
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest test/test_config_yaml.py -v`
Expected: 3 pass.

### Task 6.3: Wire YAML auto-load on `pyphi` import

PyPhi 1.x auto-loaded `pyphi_config.yml` from CWD on import. Preserve this.

**Files:**
- Modify: `pyphi/__init__.py` (or `pyphi/conf/__init__.py`)

- [ ] **Step 1: Find the existing 1.x auto-load**

```bash
git grep -n 'pyphi_config\.yml\|PYPHI_USER_CONFIG_PATH\|load_file' pyphi/_conf_legacy.py pyphi/__init__.py | head
```

- [ ] **Step 2: Replace it with `config.load_yaml`**

In whichever module currently auto-loads (likely `pyphi/__init__.py`):
```python
import os
from pathlib import Path

from pyphi.conf import config

_USER_CONFIG = Path("pyphi_config.yml")
if _USER_CONFIG.exists():
    config.load_yaml(str(_USER_CONFIG))
```

- [ ] **Step 3: Run smoke test**

```bash
cd /Users/will/projects/pyphi-p7-kernel-rewrite
uv run python -c "import pyphi; print(pyphi.config.formalism.formalism)"
```

Expected: `IIT_4_0_2023` (or whatever the YAML configures).

### Task 6.4: Delete `pyphi/_conf_legacy.py` + `pyphi/conf.pyi`

**Files:**
- Delete: `pyphi/_conf_legacy.py`
- Delete: `pyphi/conf.pyi`
- Create: `pyphi/conf/__init__.pyi`

- [ ] **Step 1: Verify no live readers of `_conf_legacy`**

```bash
git grep -n '_conf_legacy' pyphi/ test/
```

Expected: zero hits (Phase 3 removed the read-shim that was the only reader).

- [ ] **Step 2: Delete the legacy module + old stub**

```bash
git rm pyphi/_conf_legacy.py pyphi/conf.pyi
```

- [ ] **Step 3: Create new stub `pyphi/conf/__init__.pyi`**

```python
"""Type stub for pyphi.conf package."""

from collections.abc import Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Self

from pyphi.conf.formalism import FormalismConfig
from pyphi.conf.infrastructure import InfrastructureConfig
from pyphi.conf.numerics import NumericsConfig
from pyphi.conf.snapshot import ConfigSnapshot

class _GlobalConfig:
    formalism: FormalismConfig
    infrastructure: InfrastructureConfig
    numerics: NumericsConfig

    def __setattr__(self, name: str, value: Any) -> None: ...
    def __getattr__(self, name: str) -> Any: ...
    def snapshot(self) -> ConfigSnapshot: ...
    def override(self, **kwargs: Any) -> AbstractContextManager[Any]: ...
    def load_yaml(self, path: str | Path) -> None: ...
    def to_yaml(self, path: str | Path) -> None: ...

config: _GlobalConfig
```

### Task 6.5: Delete the working-scratch callsite inventory file

- [ ] **Step 1: Delete the scratch file from Phase 0**

```bash
git rm docs/superpowers/plans/p10-callsite-inventory.md
```

### Task 6.6: Update CLAUDE.md, ROADMAP.md, and CACHING.rst

**Files:**
- Modify: `ROADMAP.md` (mark P10 done; carry P11 forward)

- [ ] **Step 1: Update ROADMAP P10 section to "done"**

In `ROADMAP.md`, under the **Status (as of YYYY-MM-DD)** block, add:
```
- **P10** (Config split with result-object snapshotting) — done. Three frozen
  layered dataclasses (FormalismConfig, InfrastructureConfig, NumericsConfig)
  wrapped in ConfigSnapshot; every result object carries a snapshot taken at
  construction. Hard break on flat config.X access; layered reads / top-level
  writes / scoped override(). Restructured nested YAML.
```

### Task 6.7: Write the changelog fragment

**Files:**
- Create: `changelog.d/p10-config-split.refactor.md`

- [ ] **Step 1: Write the fragment**

```markdown
**Breaking — Config layered into three frozen dataclasses.**

The flat ``pyphi.config`` singleton is replaced by a layered facade with
three frozen dataclass layers: ``config.formalism``, ``config.infrastructure``,
``config.numerics``. Reads use layered access (``config.numerics.precision``);
persistent writes use the top-level facade (``config.precision = 6`` routes
to the right layer); scoped writes use ``config.override(precision=6,
parallel=True, repertoire_distance="EMD")`` with build-time field-name
collision detection. Per-layer ``config.numerics.override(...)`` also works.

Every result object (``SIA``, ``RepertoireIrreducibilityAnalysis``,
``MaximallyIrreducibleCauseOrEffect``, ``Distinction``, ``Concept``,
``CauseEffectStructure``, ``PhiStructure``) carries a ``.config:
ConfigSnapshot`` field set at construction time, so reproducibility is
self-contained: ``pyphi.config.override(**result.config.as_kwargs())``
reruns the exact computation.

``pyphi_config.yml`` is restructured to mirror the layered shape (top-level
``formalism:``, ``infrastructure:``, ``numerics:``). Old 1.x flat YAML files
raise on load with a pointer to the rename map.

| Old (1.x flat) | New (2.0 layered read) | Layer |
|---|---|---|
| ``FORMALISM`` | ``config.formalism.formalism`` | formalism |
| ``REPERTOIRE_DISTANCE`` | ``config.formalism.repertoire_distance`` | formalism |
| ``PARTITION_TYPE`` | ``config.formalism.partition_type`` | formalism |
| ``SHORTCIRCUIT_SIA`` | ``config.formalism.shortcircuit_sia`` | formalism |
| ``PARALLEL`` | ``config.infrastructure.parallel`` | infrastructure |
| ``CACHE_REPERTOIRES`` | ``config.infrastructure.cache_repertoires`` | infrastructure |
| ``LOG_FILE_LEVEL`` | ``config.infrastructure.log_file_level`` | infrastructure |
| ``REPR_VERBOSITY`` | ``config.infrastructure.repr_verbosity`` | infrastructure |
| ``PRECISION`` | ``config.numerics.precision`` | numerics |
| *(all 43 options)* | *(see full table in P10 spec)* | |

Removed: the ``RedisCache`` config keys (already gone in P9), ``IIT_VERSION``
(gone in P4). The ``@deprecated`` Option marker is gone — 2.0 is where
deprecations become removals.
```

### Task 6.8: Final acceptance + commit

- [ ] **Step 1: Run pyright on the new conf package**

```bash
uv run pyright pyphi/conf/
```

Expected: zero errors.

- [ ] **Step 2: Run ruff check + format**

```bash
uv run ruff check pyphi/conf/ test/test_config_layers.py test/test_config_override.py test/test_config_yaml.py test/test_config_jsonify.py test/test_result_config_snapshot.py
uv run ruff format --check pyphi/conf/ test/
```

Expected: clean.

- [ ] **Step 3: Run golden + hypothesis fast lane (foreground) and full suite (background)**

```bash
# Foreground: fast acceptance.
uv run pytest test/test_golden_regression.py test/test_invariants.py test/test_config_layers.py test/test_config_override.py test/test_config_yaml.py test/test_config_jsonify.py test/test_result_config_snapshot.py -q
```

Expected: golden 17/17 + invariants 21 + new layer/override/yaml/jsonify/snapshot tests all green.

```bash
# Foreground: full unit lane.
uv run pytest -q --ignore=test/test_invariants_hypothesis.py
```

Expected: full unit suite green.

```bash
# Background: hypothesis property tests (5–10 min).
uv run pytest test/test_invariants_hypothesis.py -q
```

Expected (when complete): green.

- [ ] **Step 4: Commit Phase 6**

```bash
git add -A
git commit -m "$(cat <<'EOF'
P10 Phase 6: delete _conf_legacy, restructure YAML, ship changelog

Deletes the legacy PyphiConfig module (renamed to _conf_legacy.py during
Phase 2) and the old conf.pyi stub. Adds nested-format YAML loader with
friendly error for old 1.x flat-format files. Adds new
pyphi/conf/__init__.pyi stub. Restructures pyphi_config.yml and
pyphi_config_3.0.yml to mirror the layered shape. Marks P10 done in
ROADMAP. Adds changelog fragment with the rename-map table.

Acceptance: golden 17/17 + invariants 21 + hypothesis fast lane + full
unit suite all green; pyright clean on pyphi/conf/; ruff clean.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-review checklist (run before declaring done)

- [ ] Every section of the spec maps to a phase or task in this plan.
- [ ] No "TBD"/"TODO"/"add appropriate error handling" without concrete code.
- [ ] Type names, method names, and field names are consistent across phases:
  - `_FIELD_TO_LAYER` (Task 1.5) matches `FIELD_TO_LAYER` import in Task 3.1 ✓
  - `_LayeredOverride` defined in Task 3.2 ✓
  - `ConfigSnapshot.diff()` and `.as_kwargs()` match between spec D6 and Task 1.4 ✓
- [ ] Each phase ends with a green-test commit; the commit message names the phase.
- [ ] CLAUDE.md project rules: no `P#` markers leak into source/comments/docstrings (only in plan/spec/commit-subjects).
- [ ] No `--no-verify` or `--no-gpg-sign` anywhere in the plan.
- [ ] Pre-commit hooks (ruff + pyright) gate every commit.
