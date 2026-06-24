# P10: Config split with result-object snapshotting

**Date:** 2026-05-08
**Branch:** `feature/p10-config-split` (cut from `feature/p9-unified-cache` tip `0c62db4c`)
**Predecessors:** P4 (formalism split) revealed which keys are formalism-scoped; P7/P8/P9 stabilized the call graph that this split lands into.
**Status:** Design approved, spec written.

## Goal

Split `pyphi/conf.py` (1112 lines, ~43 options on a flat `PyphiConfig`) into three frozen, layered config dataclasses; attach a `ConfigSnapshot` to every result object so reproducibility no longer depends on the live global; restructure `pyphi_config.yml` to mirror the layered shape.

## Why

Three forces converge here:

1. **P4 revealed the formalism layer.** With 18 of the 43 options now belonging conceptually to `PhiFormalism` (which metric, which partition, which tie-breaker), bundling them into the formalism object lets the formalism own its parameters instead of reaching into a global on every dispatch.
2. **Reproducibility is broken at result granularity.** Today, an `SIA` object on disk doesn't tell you which `PRECISION` produced it. Reloading a one-year-old result and rerunning is a guessing game. Attaching a frozen snapshot at construction makes results self-describing.
3. **P11 (parallelization) needs config snapshotting.** Workers today read `pyphi.config.*` globals that pickle implicitly via cloudpickle. Post-P10 they receive an explicit `ConfigSnapshot` bundle, eliminating the global-state-via-pickle smell and letting the no-GIL `LocalThreadScheduler` share configs safely.

## Decisions

### D1. Three layers, one snapshot wrapper

Three frozen dataclasses, grouped under one `ConfigSnapshot` value type that mirrors the live config's shape:

```python
@dataclass(frozen=True)
class NumericsConfig:
    precision: int = 13
    # Future seam: phi_tolerance, etc.

@dataclass(frozen=True)
class FormalismConfig:
    formalism: str = "IIT_4_0_2023"
    repertoire_distance: str = "GENERALIZED_INTRINSIC_DIFFERENCE"
    repertoire_distance_specification: str = ...
    repertoire_distance_differentiation: str = ...
    ces_distance: str = "SUM_SMALL_PHI"
    actual_causation_measure: str = ...
    partition_type: str = "BI"
    system_partition_type: str = "SET_UNI/BI"
    system_partition_include_complete: bool = False
    system_cuts: str = "3.0_STYLE"
    distinction_phi_normalization: str = "NONE"
    relation_computation: str = "CONCRETE"
    state_tie_resolution: str = ...
    mip_tie_resolution: list[str] = field(default_factory=list)
    purview_tie_resolution: str = ...
    assume_cuts_cannot_create_new_concepts: bool = False
    shortcircuit_sia: bool = True
    single_micro_nodes_with_selfloops_have_phi: bool = False

@dataclass(frozen=True)
class InfrastructureConfig:
    # Parallelism (9)
    parallel: bool = False
    parallel_complex_evaluation: Mapping[str, Any] = ...
    parallel_cut_evaluation: Mapping[str, Any] = ...
    parallel_concept_evaluation: Mapping[str, Any] = ...
    parallel_purview_evaluation: Mapping[str, Any] = ...
    parallel_mechanism_partition_evaluation: Mapping[str, Any] = ...
    parallel_relation_evaluation: Mapping[str, Any] = ...
    parallel_workers: int = -1
    parallel_backend: str = ...
    # Cache (4)
    maximum_cache_memory_percentage: int = 50
    cache_repertoires: bool = True
    cache_potential_purviews: bool = True
    clear_subsystem_caches_after_computing_sia: bool = True
    # Logging (3)
    log_file: str | Path = "pyphi.log"
    log_file_level: str | None = "INFO"
    log_stdout_level: str | None = "WARNING"
    # Display/UX (5)
    progress_bars: bool = True
    repr_verbosity: int = 2
    print_fractions: bool = True
    label_separator: str = ","
    welcome_off: bool = False
    # Validation (3)
    validate_subsystem_states: bool = True
    validate_conditional_independence: bool = False
    validate_json_version: bool = True

@dataclass(frozen=True)
class ConfigSnapshot:
    formalism: FormalismConfig
    infrastructure: InfrastructureConfig
    numerics: NumericsConfig

    def diff(self, other: ConfigSnapshot) -> dict[str, tuple[Any, Any]]: ...
    def as_kwargs(self) -> dict[str, Any]: ...
```

**Edge-case categorization rationale:**

| Option | Layer | Rationale |
|---|---|---|
| `SHORTCIRCUIT_SIA` | Formalism | Algorithmic shortcut shaped by IIT semantics, not a generic perf knob. |
| `SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI` | Formalism | Pure IIT 3.0 semantic edge case. |
| `*_TIE_RESOLUTION`, `DISTINCTION_PHI_NORMALIZATION`, `SYSTEM_PARTITION_INCLUDE_COMPLETE` | Formalism | Change which canonical answer wins. |
| Display/UX (`REPR_VERBOSITY`, `PRINT_FRACTIONS`, `LABEL_SEPARATOR`, `PROGRESS_BARS`, `WELCOME_OFF`) | Infrastructure | Avoids a fourth `DisplayConfig` layer for 5 options. They're orchestration of presentation, not formalism. |
| `VALIDATE_*` | Infrastructure | "Should we run the check," not "what does the check verify." |

`PRECISION` is the only Numerics option today; the layer stays as a future seam. P5/P6a noted that `PyPhiFloat` snapshots `PRECISION` at hash time — that snapshot logic already exists; P10 just hands it a structured home.

### D2. PhiFormalism owns its FormalismConfig (composition, not identity)

```python
@runtime_checkable
class PhiFormalism(Protocol):
    name: str
    default_metric: str
    compatible_metrics: frozenset[str]
    partition_scheme: str | None
    config: FormalismConfig    # NEW

    def evaluate_mechanism(self, ...): ...
    # ...

@dataclass(frozen=True)
class IIT4_2023Formalism(PhiFormalism):
    config: FormalismConfig
    # ... behavior methods ...

class FormalismRegistry:
    # Registry stores factories that construct PhiFormalism with a given config.
    def get(name: str, config: FormalismConfig) -> PhiFormalism: ...
```

Mutating `config.formalism.repertoire_distance = "EMD"` triggers a rebuild of the active formalism via the registry factory. The cost is one frozen-dataclass construction per config write — noise.

### D3. Hard break on flat `config.X` access; layered reads, top-level writes

The layers themselves are frozen, so the read/write story is asymmetric:

```python
# Reads — go through layers (encourages learning the structure):
pyphi.config.numerics.precision         # 13
pyphi.config.formalism.repertoire_distance
pyphi.config.infrastructure.parallel

# Persistent writes — go through the top-level (frozen layers can't be mutated in place):
pyphi.config.precision = 6              # _GlobalConfig.__setattr__ routes via _FIELD_TO_LAYER
pyphi.config.parallel = True            # rebuilds the relevant layer with replace()
pyphi.config.repertoire_distance = "EMD"

# Scoped writes — context manager:
with pyphi.config.override(precision=6, parallel=True, repertoire_distance="EMD"):
    ...

# Wholesale layer replacement (rare; mostly for tests):
pyphi.config.numerics = NumericsConfig(precision=6)

# Writing through a layer raises FrozenInstanceError with a helpful message:
pyphi.config.numerics.precision = 6
# FrozenInstanceError: NumericsConfig is frozen.
# Use `pyphi.config.precision = 6` or `pyphi.config.override(precision=6)`.
```

**Why this asymmetry:** layered reads make the structure visible at every call site (`config.numerics.precision` self-documents which layer owns `precision`). Frozen layers prevent silent shared-state bugs (a `ConfigSnapshot` attached to an SIA can't be mutated by a later config write). Top-level writes preserve the 1.x ergonomic of `config.PRECISION = 6` — just lowercase and without the layer suffix the user would otherwise have to type.

The 1.x uppercase flat access (`pyphi.config.PRECISION`) raises `AttributeError`. No `__getattr__` shim post-P10 (a temporary one exists during Phase 2 of execution to keep tests green during the read-site cutover, removed at end of Phase 5). 2.0 is a major version bump, consistent with the no-back-compat-shims project directive.

### D4. `config.override(...)` accepts kwargs across layers, with build-time collision check

```python
# Top-level (works across layers, like 1.x in shape)
with config.override(precision=6, parallel=True, repertoire_distance="EMD"):
    ...

# Per-layer (also works)
with config.numerics.override(precision=6):
    ...
```

Build-time scan walks every layer's fields and builds a `_FIELD_TO_LAYER` map; raises at import time if any name appears in two layers (currently zero collisions across all 43 options). Routing logic:

```python
def override(self, **kwargs):
    by_layer: dict[str, dict[str, Any]] = defaultdict(dict)
    for name, value in kwargs.items():
        layer = _FIELD_TO_LAYER.get(name)
        if layer is None:
            raise ConfigurationError(f"Unknown config option: {name!r}")
        by_layer[layer][name] = value
    return _LayeredOverride(self, by_layer)
```

`_LayeredOverride` is a context manager that snapshots the three layers, replaces each affected layer with `dataclasses.replace(...)`, yields, and restores on exit. Per-layer override is a thin wrapper scoped to one layer.

### D5. Restructured nested YAML

```yaml
# pyphi_config.yml (2.0)
formalism:
  formalism: IIT_4_0_2023
  repertoire_distance: GENERALIZED_INTRINSIC_DIFFERENCE
  partition_type: BI
infrastructure:
  parallel: false
  cache_repertoires: true
  log_file_level: INFO
numerics:
  precision: 13
```

YAML loader walks each top-level key (`formalism`, `infrastructure`, `numerics`), constructs a frozen layer dataclass from the nested dict, and assigns it to the global. Unknown top-level keys raise. **Old flat YAML detection:** if the YAML root has any uppercase keys (`PRECISION:`, `PARALLEL:`), raise with the rename map embedded in the error message. Hard break with a soft landing.

### D6. Every result object gets one `ConfigSnapshot` field

```python
@dataclass(frozen=True)
class SIA:
    phi: float
    signed_phi: float
    cut: SystemPartition
    # ... existing fields ...
    config: ConfigSnapshot
```

Same wiring for `RepertoireIrreducibilityAnalysis`, `MaximallyIrreducibleCauseOrEffect`, `Distinction`, `Concept`, `CauseEffectStructure`, `PhiStructure`. The snapshot is taken at construction time; once a result exists, its config is immutable even if `pyphi.config` changes underneath.

**Construction discipline:** result objects don't read `pyphi.config` themselves. The construction site (the formalism's `evaluate_*` method, the SIA computation entry point, etc.) reads the global once and passes the configs down. This makes results parallel-safe under P11 — a worker that receives an existing result via cloudpickle gets a self-contained reproducibility record without any global state dependency.

**Memory cost:** all distinctions in one `PhiStructure` share the same three frozen dataclass instances by reference (Python is reference-based; frozen dataclasses are safely shared). One pointer per result per layer = ~24 bytes/result. For a million distinctions: 24MB. Acceptable.

**`pyphi.jsonify` integration:** the configs are frozen dataclasses; jsonify gets them via 4 small serializer registrations (one per dataclass). Loading old `.json` results without snapshot fields is a hard break — old results don't reload in 2.0.

## Implementation phases

Each phase keeps tests green and golden 17/17 passing. Cadence mirrors P7/P8/P9: additive first, cut over piece by piece, delete old at the end.

### Phase 0 — Branch + audit

- Cut `feature/p10-config-split` from `feature/p9-unified-cache` tip (`0c62db4c`).
- Inventory every `config.X` read/write site (~200 expected). `git grep -n 'config\.' pyphi/`.
- Write the rename-map table; check it into this spec.

### Phase 1 — Three layers + `ConfigSnapshot` (additive only)

- New module: `pyphi/conf/__init__.py`, `pyphi/conf/formalism.py`, `pyphi/conf/infrastructure.py`, `pyphi/conf/numerics.py`, `pyphi/conf/snapshot.py`, `pyphi/conf/legacy_global.py`.
- The old `pyphi/conf.py` `PyphiConfig` stays untouched, side by side. New layers don't yet plug in anywhere.
- Build-time collision check runs at import.
- Tests: unit tests for each frozen dataclass, the collision check, and `ConfigSnapshot.diff()` / `as_kwargs()`.

### Phase 2 — Cut over read sites, module by module

- Module-by-module: `config.X` → `config.layer.x`. Start with leaf modules (`combinatorics`, `distribution`, `partition`).
- After each module, run `make test` for that module's unit tests.
- The old `PyphiConfig` is still the source of truth — both flat and layered access work *during* this phase via a temporary read-only `__getattr__` shim on `_GlobalConfig` that delegates to `PyphiConfig`. Removed at end of Phase 5.

### Phase 3 — Cut over write sites + `override()` compat

- Replace every `config.X = value` with `config.layer.x = value` (the `_GlobalConfig.__setattr__` routes via `_FIELD_TO_LAYER`).
- Replace every `config.override(X=v)` call with the lowercase layered name.
- Tests using `pyphi.config.override(...)` with old uppercase names rewritten to lowercase layered names.

### Phase 4 — PhiFormalism wiring

- `PhiFormalism` Protocol gains `config: FormalismConfig`.
- Concrete formalisms become `@dataclass(frozen=True)` with `FormalismConfig` field; registry stores factories.
- `_GlobalConfig.formalism = ...` triggers active formalism rebuild via the factory.
- Tests: golden 17/17 must match (no semantic change).

### Phase 5 — Result-object snapshot wiring

- Add `config: ConfigSnapshot` field to `SIA`, `RepertoireIrreducibilityAnalysis`, `MaximallyIrreducibleCauseOrEffect`, `Distinction`, `Concept`, `CauseEffectStructure`, `PhiStructure`.
- Construction sites snapshot the global once and thread through.
- jsonify serializers for the four frozen types.
- Tests: every result type has a non-None `config` snapshot post-construction; round-trip through jsonify preserves it; mutating global afterwards doesn't change the snapshot.

### Phase 6 — Delete old `PyphiConfig` + nested-only YAML + acceptance

- Delete `pyphi/conf.py`. Move remaining helpers (`atomic_write_yaml`, etc.) into `pyphi/conf/_io.py`.
- Remove the temporary `__getattr__` shim from `_GlobalConfig`.
- YAML loader: nested-only with friendly error for old-format files.
- Update `pyphi_config.yml` and `pyphi_config_3.0.yml` to nested format.
- Update `pyphi/conf.pyi` (or replace with `pyphi/conf/__init__.pyi`).
- Run full acceptance: golden 17/17, hypothesis fast lane 21, fast unit lane 137, pyright clean, ruff clean.
- Changelog fragment with rename map and structural narrative.

## Acceptance gates

Per CLAUDE.md project rules: golden 17/17 + hypothesis fast lane 21 are the numerical-correctness gates. Add for P10:

- `test/test_config_layers.py`: each layer dataclass constructs with defaults; frozen-ness enforced; collision check raises on injected duplicate.
- `test/test_config_override.py`: `config.override(precision=6, parallel=True, repertoire_distance="EMD")` snapshots, mutates, restores. Per-layer `config.numerics.override(precision=6)` ditto. Unknown name raises.
- `test/test_result_config_snapshot.py`: end-to-end SIA on `basic_network()` produces a SIA whose `config.numerics.precision` matches the global at construction time; mutating global afterwards doesn't change the snapshot.
- `test/test_config_yaml.py`: nested YAML round-trips. Old flat YAML raises with the rename map in the error.
- `test/test_jsonify_config.py`: `pyphi.jsonify(sia)` and `pyphi.loads(jsonify(sia))` preserve the `ConfigSnapshot`.

Plus the standard P-level gates: pyright clean on `pyphi/conf/`, ruff clean.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Hidden read of `config.X` at module import time (e.g. log handler setup, parallel decorator default lookup) breaks silently if cutover misses it | Phase 2 cuts module by module with tests after each; the temporary `__getattr__` shim catches stragglers and logs them at DEBUG level so they're discoverable. |
| Frozen dataclass + mutation pattern (assigning `config.numerics.precision = 6` requires rebuilding the whole layer) is slower than the current attribute write | Profile before optimizing; expected overhead is sub-microsecond per write and writes are rare relative to reads. |
| Result-object snapshot adds memory overhead at scale | All distinctions in one PhiStructure share frozen-dataclass references; per-result cost is 3 pointers (~24 bytes). Negligible relative to repertoire data. |
| PhiFormalism rebuild on every formalism config write churns the registry | Cache the active formalism instance; rebuild only when a field actually changes (compare frozen dataclass equality before swapping). |
| YAML hard break breaks user setups silently | Friendly error in the loader names the rename map; CHANGELOG fragment carries the full table. |
| Mid-cutover state where some sites use `config.X` (via shim) and others use `config.layer.x` is confusing | Phase boundaries are atomic; every commit leaves the suite green. The shim is documented as temporary at the top of `legacy_global.py`. |

## What does NOT happen in P10 (deferred)

- **`REPERTOIRE_DISTANCE_DIFFERENTIATION` / `_SPECIFICATION` config-key cleanup** — named in the cross-cutting deferred registry under "P10," but it's a metric-API tidiness concern that doesn't affect the layered split. Land in current form, revisit naming separately.
- **Removing `IIT_VERSION`** — already removed in P4.
- **Validation semantics changes** — just reclassifying.
- **`@deprecated` markers for the few remaining options** — they all become hard removals; if users hit them, the rename-map error in YAML / `AttributeError` from the global is the migration prompt.
- **Automated migration tool** — the rename is mechanical enough that grep+sed is faster than building a script.
- **DisplayConfig as a fourth layer** — folded into Infrastructure for 5 options; revisit only if Display grows past ~10 options or develops independent invariants.
- **Snapshotting the active `PhiFormalism` registry contents** — only the `FormalismConfig` is snapshotted; the registered formalism factory is identified by name string and looked up at deserialize time. If a future PyPhi version renames a formalism, old result objects need an explicit migration — accepted cost.

## Open questions

None at design time. Anything that surfaces during execution gets a follow-up commit on this branch with a note in the changelog.

## Appendix A: Rename-map table

| Old (1.x flat) | New (2.0 layered read) | Layer |
|---|---|---|
| `FORMALISM` | `config.formalism.formalism` | formalism |
| `ASSUME_CUTS_CANNOT_CREATE_NEW_CONCEPTS` | `config.formalism.assume_cuts_cannot_create_new_concepts` | formalism |
| `REPERTOIRE_DISTANCE` | `config.formalism.repertoire_distance` | formalism |
| `REPERTOIRE_DISTANCE_DIFFERENTIATION` | `config.formalism.repertoire_distance_differentiation` | formalism |
| `REPERTOIRE_DISTANCE_SPECIFICATION` | `config.formalism.repertoire_distance_specification` | formalism |
| `CES_DISTANCE` | `config.formalism.ces_distance` | formalism |
| `ACTUAL_CAUSATION_MEASURE` | `config.formalism.actual_causation_measure` | formalism |
| `PARTITION_TYPE` | `config.formalism.partition_type` | formalism |
| `SYSTEM_PARTITION_TYPE` | `config.formalism.system_partition_type` | formalism |
| `SYSTEM_PARTITION_INCLUDE_COMPLETE` | `config.formalism.system_partition_include_complete` | formalism |
| `SYSTEM_CUTS` | `config.formalism.system_cuts` | formalism |
| `DISTINCTION_PHI_NORMALIZATION` | `config.formalism.distinction_phi_normalization` | formalism |
| `RELATION_COMPUTATION` | `config.formalism.relation_computation` | formalism |
| `STATE_TIE_RESOLUTION` | `config.formalism.state_tie_resolution` | formalism |
| `MIP_TIE_RESOLUTION` | `config.formalism.mip_tie_resolution` | formalism |
| `PURVIEW_TIE_RESOLUTION` | `config.formalism.purview_tie_resolution` | formalism |
| `SHORTCIRCUIT_SIA` | `config.formalism.shortcircuit_sia` | formalism |
| `SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI` | `config.formalism.single_micro_nodes_with_selfloops_have_phi` | formalism |
| `PARALLEL` | `config.infrastructure.parallel` | infrastructure |
| `PARALLEL_COMPLEX_EVALUATION` | `config.infrastructure.parallel_complex_evaluation` | infrastructure |
| `PARALLEL_CUT_EVALUATION` | `config.infrastructure.parallel_cut_evaluation` | infrastructure |
| `PARALLEL_CONCEPT_EVALUATION` | `config.infrastructure.parallel_concept_evaluation` | infrastructure |
| `PARALLEL_PURVIEW_EVALUATION` | `config.infrastructure.parallel_purview_evaluation` | infrastructure |
| `PARALLEL_MECHANISM_PARTITION_EVALUATION` | `config.infrastructure.parallel_mechanism_partition_evaluation` | infrastructure |
| `PARALLEL_RELATION_EVALUATION` | `config.infrastructure.parallel_relation_evaluation` | infrastructure |
| `PARALLEL_WORKERS` | `config.infrastructure.parallel_workers` | infrastructure |
| `PARALLEL_BACKEND` | `config.infrastructure.parallel_backend` | infrastructure |
| `MAXIMUM_CACHE_MEMORY_PERCENTAGE` | `config.infrastructure.maximum_cache_memory_percentage` | infrastructure |
| `CACHE_REPERTOIRES` | `config.infrastructure.cache_repertoires` | infrastructure |
| `CACHE_POTENTIAL_PURVIEWS` | `config.infrastructure.cache_potential_purviews` | infrastructure |
| `CLEAR_SUBSYSTEM_CACHES_AFTER_COMPUTING_SIA` | `config.infrastructure.clear_subsystem_caches_after_computing_sia` | infrastructure |
| `LOG_FILE` | `config.infrastructure.log_file` | infrastructure |
| `LOG_FILE_LEVEL` | `config.infrastructure.log_file_level` | infrastructure |
| `LOG_STDOUT_LEVEL` | `config.infrastructure.log_stdout_level` | infrastructure |
| `PROGRESS_BARS` | `config.infrastructure.progress_bars` | infrastructure |
| `REPR_VERBOSITY` | `config.infrastructure.repr_verbosity` | infrastructure |
| `PRINT_FRACTIONS` | `config.infrastructure.print_fractions` | infrastructure |
| `LABEL_SEPARATOR` | `config.infrastructure.label_separator` | infrastructure |
| `WELCOME_OFF` | `config.infrastructure.welcome_off` | infrastructure |
| `VALIDATE_SUBSYSTEM_STATES` | `config.infrastructure.validate_subsystem_states` | infrastructure |
| `VALIDATE_CONDITIONAL_INDEPENDENCE` | `config.infrastructure.validate_conditional_independence` | infrastructure |
| `VALIDATE_JSON_VERSION` | `config.infrastructure.validate_json_version` | infrastructure |
| `PRECISION` | `config.numerics.precision` | numerics |

Persistent-write form: `config.<lowercase_name> = value` (e.g. `config.precision = 6`). Scoped: `with config.override(precision=6, parallel=True): ...`.
