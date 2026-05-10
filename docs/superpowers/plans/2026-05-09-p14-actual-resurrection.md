# P14 — `actual.Transition` Resurrection + Formalism Config Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring `pyphi.actual.Transition` back online against the frozen `System` value type while auditing the formalism config namespace, and retire the orphaned concept-style cuts machinery.

**Architecture:** Three architectural seams: (1) `pyphi.actual.TransitionSystem` (new frozen dataclass, parametric in `Direction`, satisfying `pyphi.protocols.SystemPublicInterface`); (2) nested config namespaces `formalism.iit.*` and `formalism.actual_causation.*` replacing today's flat `formalism` keys; (3) AC-specific registries (`partitioned_repertoire_schemes`, `background_strategies`, `alpha_aggregations`) decouple AC math from IIT formalism config.

**Tech Stack:** Python 3.12+, `@dataclass(frozen=True)`, `cached_property`, `pyphi.registry.Registry`, pytest, pyright (standard mode), ruff.

**Spec:** `docs/superpowers/specs/2026-05-09-p14-actual-transition-resurrection-design.md` at commit `aa0518e4`.

**Branch:** `feature/p14-actual-resurrection` (cut from `2.0` head `b5dfb8a5`).

---

## Acceptance gates (every commit)

- Golden 17/17 numerical match — `uv run pytest test/test_golden_regression.py -q` green.
- Hypothesis fast lane 21 green — `uv run pytest test/test_invariants_hypothesis.py -q` green.
- Fast unit lane green — `uv run pytest test/test_invariants.py test/test_subsystem_surface.py test/test_formalism_pickle.py test/test_parallel.py test/test_scheduler.py test/test_sampling.py test/test_install_snapshot.py -q`.
- Pyright clean on touched files: `uv run pyright pyphi/`.
- Ruff clean: `uv run ruff check pyphi/ test/`.
- Pre-commit hooks pass on every commit; **never** use `--no-verify`.
- Per saved memory `feedback_ask_before_push.md`: do **not** push without explicit per-action consent.

**Long-running tests** (per saved memory `feedback_monitor_for_long_tests.md`): use `Bash` with `run_in_background=true` for slow runs. Golden suite is ~1 min currently (post-Phase-2 fix); if slower, monitor in background.

---

# Phase 1: Delete concept-style cuts machinery

**Why first:** Smallest, most self-contained change. Deletion before rename means less code to rename later. Concept-style cuts have unclear provenance, rotted integration tests, no current users (per maintainer).

**Files affected:**
- Modify: `pyphi/formalism/iit3/__init__.py:389-557`
- Modify: `pyphi/conf/formalism.py:17, 40, 72-75`
- Modify: `pyphi/__init__.py` (if `ConceptStyleSystem` etc. are exported)
- Delete: `test/test_concept_style_cuts.py`
- Modify: `test/test_big_phi.py` (delete `test_system_cut_styles`)
- Modify: `test/test_config.py:35`
- Modify: `pyphi_config.yml`

## Task 1.1: Survey concept-style export surface

**Files:**
- Read: `pyphi/__init__.py`, `pyphi/formalism/__init__.py`, `pyphi/formalism/iit3/__init__.py`

- [ ] **Step 1: Confirm what's exported**

```bash
grep -n "ConceptStyleSystem\|sia_concept_style\|concept_cuts\|directional_sia\|SystemIrreducibilityAnalysisConceptStyle" pyphi/__init__.py pyphi/formalism/__init__.py pyphi/formalism/iit3/__init__.py
```

Expected: zero hits in `pyphi/__init__.py` (top-level not exported); the names appear only in `pyphi/formalism/iit3/__init__.py`. If the grep shows hits in `pyphi/formalism/__init__.py`, note them for cleanup in Task 1.2.

## Task 1.2: Delete concept-style code from `pyphi/formalism/iit3/__init__.py`

**Files:**
- Modify: `pyphi/formalism/iit3/__init__.py:389-557`

- [ ] **Step 1: Read current sia() definition (line 389-395)**

```python
@functools.wraps(_sia)
def sia(
    system: System, **kwargs: Any
) -> SystemIrreducibilityAnalysis | SystemIrreducibilityAnalysisConceptStyle:
    if config.formalism.system_cuts == "CONCEPT_STYLE":
        return sia_concept_style(system, **kwargs)
    return _sia(system, **kwargs)
```

- [ ] **Step 2: Replace `sia()` with direct alias**

Replace the `sia()` function and everything from the `class ConceptStyleSystem:` line through the end of `def sia_concept_style(...)` with:

```python
sia = _sia
```

This drops:
- The `if config.formalism.system_cuts == "CONCEPT_STYLE":` branch
- `class ConceptStyleSystem:`
- `def concept_cuts(...)`
- `def directional_sia(...)`
- `class SystemIrreducibilityAnalysisConceptStyle(cmp.Orderable):`
- `def sia_concept_style(...)`
- The `# TODO: cache` comment

Total deletion ≈ 165 lines (lines 389–557 minus the new alias line).

- [ ] **Step 3: Remove now-unused imports**

After the deletion, run:

```bash
uv run ruff check pyphi/formalism/iit3/__init__.py
```

Expected: report unused imports (likely `KCut`, `cmp`, `mip_partitions`, `functools` if not used elsewhere). Remove each unused import at the top of the file.

- [ ] **Step 4: Verify the file still compiles**

```bash
uv run python -c "from pyphi.formalism import iit3; print(iit3.sia)"
```

Expected: prints `<function _sia at 0x...>` (or similar) — the alias resolves.

## Task 1.3: Delete `system_cuts` from formalism config

**Files:**
- Modify: `pyphi/conf/formalism.py:15, 17, 40, 72-75`

- [ ] **Step 1: Remove `_VALID_SYSTEM_CUTS` and the field**

In `pyphi/conf/formalism.py`:

Delete line 17 (`_VALID_SYSTEM_CUTS = frozenset({"3.0_STYLE", "CONCEPT_STYLE"})`).

Delete line 40 (`system_cuts: str = "3.0_STYLE"`).

Delete lines 72-75:

```python
        if self.system_cuts not in _VALID_SYSTEM_CUTS:
            raise ValueError(
                f"system_cuts={self.system_cuts!r} not in {sorted(_VALID_SYSTEM_CUTS)}"
            )
```

- [ ] **Step 2: Verify import works**

```bash
uv run python -c "from pyphi.conf.formalism import FormalismConfig; print(FormalismConfig())"
```

Expected: prints the FormalismConfig with all current defaults except `system_cuts`.

## Task 1.4: Delete `test/test_concept_style_cuts.py` and the matching test in `test_big_phi.py`

**Files:**
- Delete: `test/test_concept_style_cuts.py`
- Modify: `test/test_big_phi.py:272-280`

- [ ] **Step 1: Delete the file**

```bash
git rm test/test_concept_style_cuts.py
```

- [ ] **Step 2: Read current `test_system_cut_styles` (test_big_phi.py:272-280)**

```python
@pytest.mark.outdated
@pytest.mark.slow
@config.override(system_partition_type="DIRECTED_BI")
def test_system_cut_styles(s):
    with config.override(system_cuts="3.0_STYLE"):
        assert iit3.phi(s) == 0.5  # 2.3125

    with config.override(system_cuts="CONCEPT_STYLE"):
        assert iit3.phi(s) == 0.6875
```

- [ ] **Step 3: Delete `test_system_cut_styles` function**

Remove lines 272-280 from `test/test_big_phi.py`. If preceded by an empty line or a section divider that was specific to this test, remove that too.

## Task 1.5: Update `test/test_config.py` and `pyphi_config.yml`

**Files:**
- Modify: `test/test_config.py:35`
- Modify: `pyphi_config.yml`

- [ ] **Step 1: Remove SYSTEM_CUTS validation entry**

In `test/test_config.py`, find and delete line 35 (or whatever line the entry is on):

```python
        ("SYSTEM_CUTS", ["3.0_STYLE", "CONCEPT_STYLE"], ["OTHER"]),
```

- [ ] **Step 2: Check `pyphi_config.yml` for system_cuts entry**

```bash
grep -n "system_cuts\|SYSTEM_CUTS" pyphi_config.yml
```

If a line exists, delete it.

- [ ] **Step 3: Run config tests to verify**

```bash
uv run pytest test/test_config.py -v
```

Expected: all tests pass (the SYSTEM_CUTS row is no longer a parameter).

## Task 1.6: Run full acceptance gates and commit

- [ ] **Step 1: Pyright clean**

```bash
uv run pyright pyphi/
```

Expected: zero new errors. If there were pre-existing errors unrelated to this phase, they are tolerated as a baseline.

- [ ] **Step 2: Ruff clean**

```bash
uv run ruff check pyphi/ test/
```

Expected: clean.

- [ ] **Step 3: Fast unit lane green**

```bash
uv run pytest test/test_invariants.py test/test_subsystem_surface.py test/test_formalism_pickle.py test/test_parallel.py test/test_scheduler.py test/test_sampling.py test/test_install_snapshot.py test/test_config.py -q
```

Expected: all green.

- [ ] **Step 4: Hypothesis fast lane green**

```bash
uv run pytest test/test_invariants_hypothesis.py -q
```

Expected: 21 properties pass.

- [ ] **Step 5: Golden 17/17 numerical match (background, monitor)**

```bash
uv run pytest test/test_golden_regression.py -q
```

If runtime > 5 minutes, run in background: `Bash(command, run_in_background=true)` and continue with the commit; verify completion before next phase.

Expected: 17/17 pass.

- [ ] **Step 6: Stage and commit**

```bash
git add -A
git status  # confirm only Phase 1 files changed
git commit -m "$(cat <<'EOF'
Delete concept-style cuts machinery

The concept-style SIA variant in pyphi.formalism.iit3 has unclear
provenance — it is not described in the published 2014 IIT 3.0 paper main
text or figures, its integration tests rotted in the IIT 4.0 transition
(noted in test/test_concept_style_cuts.py:149), and no current workflows
depend on it.

Removed: ConceptStyleSystem, concept_cuts, directional_sia,
SystemIrreducibilityAnalysisConceptStyle, sia_concept_style from
pyphi.formalism.iit3; the system_cuts config field and validation;
test/test_concept_style_cuts.py; test_system_cut_styles in
test/test_big_phi.py; the SYSTEM_CUTS entry in test/test_config.py.

iit3.sia is now a direct alias for _sia.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

**Acceptance:** All gates green; commit lands.

---

# Phase 2: Config audit and rename

**Why second:** Restructure config before adding new code that would consume it. The rename touches ~30 call sites; doing it after Phase 1's deletion means less code to rename.

**Strategy:** Restructure `FormalismConfig` as a thin holder of two nested frozen dataclasses (`IITConfig` and `ActualCausationConfig`). The flat-routing layer (`FIELD_TO_LAYER`) stays for `infrastructure`/`numerics`; the formalism layer routes through nested-aware logic.

**Files affected:**
- Modify: `pyphi/conf/formalism.py` (full rewrite)
- Modify: `pyphi/conf/_field_routing.py` (handle nested formalism)
- Modify: `pyphi/conf/_global.py:155-186` (`__setattr__` nested routing)
- Modify: `pyphi/conf/_io.py` (load yaml with nested format)
- Modify: `pyphi/conf/snapshot.py` (snapshot includes nested layers)
- Modify: ~30 call sites: `pyphi/system.py`, `pyphi/partition.py`, `pyphi/relations.py`, `pyphi/resolve_ties.py`, `pyphi/actual.py`, `pyphi/metrics/distribution.py`, `pyphi/metrics/ces.py`, `pyphi/core/repertoire_algebra.py`, `pyphi/models/state_specification.py`, `pyphi/visualize/distribution.py`, `pyphi/formalism/__init__.py`, `pyphi/formalism/base.py`, `pyphi/formalism/iit3/formalism.py`, `pyphi/formalism/queries.py`, `pyphi/formalism/iit4/formalism.py`, `pyphi/formalism/iit4/__init__.py`
- Modify: `pyphi_config.yml`
- Modify: `test/test_config.py`, `test/test_config_layers.py`
- Modify: any test using `config.override(repertoire_distance=...)` etc.

## Task 2.1: Define `IITConfig` and `ActualCausationConfig` frozen dataclasses

**Files:**
- Create new module structure inside: `pyphi/conf/formalism.py`

- [ ] **Step 1: Read current `pyphi/conf/formalism.py`**

The file currently defines a flat `FormalismConfig` with all fields directly. After Phase 1 deletion, `system_cuts` is gone.

- [ ] **Step 2: Write the failing test**

In `test/test_config_layers.py`, add:

```python
def test_formalism_iit_subnamespace_exists():
    from pyphi.conf.formalism import FormalismConfig, IITConfig

    fc = FormalismConfig()
    assert isinstance(fc.iit, IITConfig)
    assert fc.iit.version == "IIT_4_0_2023"
    assert fc.iit.repertoire_measure == "GENERALIZED_INTRINSIC_DIFFERENCE"
    assert fc.iit.mechanism_partition_scheme == "ALL"
    assert fc.iit.system_partition_scheme == "SET_UNI/BI"


def test_formalism_actual_causation_subnamespace_exists():
    from pyphi.conf.formalism import ActualCausationConfig, FormalismConfig

    fc = FormalismConfig()
    assert isinstance(fc.actual_causation, ActualCausationConfig)
    assert fc.actual_causation.measure == "PMI"
    assert fc.actual_causation.mechanism_partition_scheme == "ALL"
    assert fc.actual_causation.partitioned_repertoire_scheme == "PRODUCT"
    assert fc.actual_causation.background_strategy == "UNIFORM"
    assert fc.actual_causation.alpha_aggregation == "SUBTRACTIVE"


def test_formalism_iit_post_init_validators():
    import pytest

    from pyphi.conf.formalism import IITConfig

    with pytest.raises(ValueError, match="distinction_phi_normalization"):
        IITConfig(distinction_phi_normalization="BOGUS")
    with pytest.raises(ValueError, match="relation_computation"):
        IITConfig(relation_computation="BOGUS")


def test_formalism_actual_causation_post_init_validators():
    import pytest

    from pyphi.conf.formalism import ActualCausationConfig

    # measure must be a registered key (validated at construction time)
    with pytest.raises(ValueError, match="partitioned_repertoire_scheme"):
        ActualCausationConfig(partitioned_repertoire_scheme="BOGUS")
    with pytest.raises(ValueError, match="background_strategy"):
        ActualCausationConfig(background_strategy="BOGUS")
    with pytest.raises(ValueError, match="alpha_aggregation"):
        ActualCausationConfig(alpha_aggregation="BOGUS")
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run pytest test/test_config_layers.py::test_formalism_iit_subnamespace_exists -v
```

Expected: ImportError or AttributeError — `IITConfig` and `ActualCausationConfig` don't exist yet.

- [ ] **Step 4: Rewrite `pyphi/conf/formalism.py`**

Replace the entire file content:

```python
"""Formalism layer of the PyPhi config.

Holds knobs that define the mathematical formalism — split into two
nested sub-namespaces:

- :class:`IITConfig` for IIT-formalism dispatch and IIT-specific knobs
  (which IIT version, which repertoire measure, which partition scheme,
  tie-resolution policy, etc.).
- :class:`ActualCausationConfig` for the actual-causation framework
  (which information measure, which partitioned-repertoire scheme,
  which background strategy, which α aggregation).

Bundled into the :class:`~pyphi.formalism.base.PhiFormalism` instance via
composition; the active formalism is rebuilt from the registry factory
whenever the IIT sub-config changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

_VALID_DISTINCTION_PHI_NORMALIZATION = frozenset({"NONE", "NUM_CONNECTIONS_CUT"})
_VALID_RELATION_COMPUTATION = frozenset({"CONCRETE", "ANALYTICAL"})

_VALID_PARTITIONED_REPERTOIRE_SCHEMES = frozenset({"PRODUCT", "FORWARD_PROBABILITY"})
_VALID_BACKGROUND_STRATEGIES = frozenset({"UNIFORM", "STATIONARY", "OBSERVED"})
_VALID_ALPHA_AGGREGATIONS = frozenset({"SUBTRACTIVE", "RATIO"})


@dataclass(frozen=True)
class IITConfig:
    """IIT-formalism configuration sub-namespace.

    Knobs that define how IIT phi is computed: which formalism version
    dispatches, which measure quantifies repertoire distance, which
    partition scheme enumerates candidate partitions, which tie-resolution
    policies fire when multiple options have the same phi.
    """

    version: str = "IIT_4_0_2023"
    repertoire_measure: str = "GENERALIZED_INTRINSIC_DIFFERENCE"
    repertoire_measure_specification: str = "GENERALIZED_INTRINSIC_DIFFERENCE"
    repertoire_measure_differentiation: str = "GENERALIZED_INTRINSIC_DIFFERENCE"
    ces_measure: str = "SUM_SMALL_PHI"
    mechanism_partition_scheme: str = "ALL"
    system_partition_scheme: str = "SET_UNI/BI"
    system_partition_include_complete: bool = False
    distinction_phi_normalization: str = "NUM_CONNECTIONS_CUT"
    relation_computation: str = "CONCRETE"
    assume_partitions_cannot_create_new_concepts: bool = False
    shortcircuit_sia: bool = True
    single_micro_nodes_with_selfloops_have_phi: bool = True
    state_tie_resolution: str = "PHI"
    mip_tie_resolution: list[str] = field(
        default_factory=lambda: ["NORMALIZED_PHI", "NEGATIVE_PHI"]
    )
    purview_tie_resolution: str = "PHI"

    def __post_init__(self) -> None:
        if not isinstance(self.assume_partitions_cannot_create_new_concepts, bool):
            raise ValueError(
                "assume_partitions_cannot_create_new_concepts must be bool; "
                f"got {type(self.assume_partitions_cannot_create_new_concepts).__name__}"
            )
        if not isinstance(self.system_partition_include_complete, bool):
            raise ValueError(
                "system_partition_include_complete must be bool; "
                f"got {type(self.system_partition_include_complete).__name__}"
            )
        if not isinstance(self.shortcircuit_sia, bool):
            raise ValueError(
                "shortcircuit_sia must be bool; got "
                f"{type(self.shortcircuit_sia).__name__}"
            )
        if not isinstance(self.single_micro_nodes_with_selfloops_have_phi, bool):
            raise ValueError(
                "single_micro_nodes_with_selfloops_have_phi must be bool; "
                f"got {type(self.single_micro_nodes_with_selfloops_have_phi).__name__}"
            )
        if (
            self.distinction_phi_normalization
            not in _VALID_DISTINCTION_PHI_NORMALIZATION
        ):
            raise ValueError(
                f"distinction_phi_normalization={self.distinction_phi_normalization!r} "
                f"not in {sorted(_VALID_DISTINCTION_PHI_NORMALIZATION)}"
            )
        if self.relation_computation not in _VALID_RELATION_COMPUTATION:
            raise ValueError(
                f"relation_computation={self.relation_computation!r} "
                f"not in {sorted(_VALID_RELATION_COMPUTATION)}"
            )


@dataclass(frozen=True)
class ActualCausationConfig:
    """Actual-causation configuration sub-namespace.

    Decomposes the 2019 Albantakis et al. AC framework into its
    parameterized choices. Defaults match the published formalism;
    alternative registered values let users investigate variants.
    """

    measure: str = "PMI"
    mechanism_partition_scheme: str = "ALL"
    partitioned_repertoire_scheme: str = "PRODUCT"
    background_strategy: str = "UNIFORM"
    alpha_aggregation: str = "SUBTRACTIVE"

    def __post_init__(self) -> None:
        if self.partitioned_repertoire_scheme not in _VALID_PARTITIONED_REPERTOIRE_SCHEMES:
            raise ValueError(
                f"partitioned_repertoire_scheme={self.partitioned_repertoire_scheme!r} "
                f"not in {sorted(_VALID_PARTITIONED_REPERTOIRE_SCHEMES)}"
            )
        if self.background_strategy not in _VALID_BACKGROUND_STRATEGIES:
            raise ValueError(
                f"background_strategy={self.background_strategy!r} "
                f"not in {sorted(_VALID_BACKGROUND_STRATEGIES)}"
            )
        if self.alpha_aggregation not in _VALID_ALPHA_AGGREGATIONS:
            raise ValueError(
                f"alpha_aggregation={self.alpha_aggregation!r} "
                f"not in {sorted(_VALID_ALPHA_AGGREGATIONS)}"
            )


@dataclass(frozen=True)
class FormalismConfig:
    """Formalism-scoped configuration.

    A thin holder of two nested frozen dataclasses: :class:`IITConfig`
    for IIT-formalism dispatch and IIT-specific knobs, and
    :class:`ActualCausationConfig` for the actual-causation framework.
    Both travel with each :class:`~pyphi.formalism.base.PhiFormalism`
    instance and are snapshotted onto every result object.
    """

    iit: IITConfig = field(default_factory=IITConfig)
    actual_causation: ActualCausationConfig = field(default_factory=ActualCausationConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.iit, IITConfig):
            raise ValueError(
                f"iit must be an IITConfig; got {type(self.iit).__name__}"
            )
        if not isinstance(self.actual_causation, ActualCausationConfig):
            raise ValueError(
                f"actual_causation must be an ActualCausationConfig; "
                f"got {type(self.actual_causation).__name__}"
            )
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest test/test_config_layers.py::test_formalism_iit_subnamespace_exists test/test_config_layers.py::test_formalism_actual_causation_subnamespace_exists test/test_config_layers.py::test_formalism_iit_post_init_validators test/test_config_layers.py::test_formalism_actual_causation_post_init_validators -v
```

Expected: 4 passed.

## Task 2.2: Update `_field_routing.py` to handle nested formalism

**Files:**
- Modify: `pyphi/conf/_field_routing.py`

- [ ] **Step 1: Write the failing test**

Add to `test/test_config_layers.py`:

```python
def test_field_routing_nested_formalism_unique_fields():
    from pyphi.conf._field_routing import FIELD_TO_LAYER

    # Unique IIT fields route to ("formalism", "iit")
    assert FIELD_TO_LAYER["repertoire_measure"] == ("formalism", "iit")
    assert FIELD_TO_LAYER["version"] == ("formalism", "iit")
    # Unique AC fields route to ("formalism", "actual_causation")
    assert FIELD_TO_LAYER["measure"] == ("formalism", "actual_causation")
    assert FIELD_TO_LAYER["partitioned_repertoire_scheme"] == ("formalism", "actual_causation")


def test_field_routing_nested_formalism_collision_raises():
    """`mechanism_partition_scheme` exists in both IIT and AC namespaces;
    flat routing must raise to prevent silent misdispatch."""
    import pytest

    from pyphi.conf._field_routing import FIELD_TO_LAYER

    # The colliding key should NOT be flat-routable.
    assert "mechanism_partition_scheme" not in FIELD_TO_LAYER
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest test/test_config_layers.py::test_field_routing_nested_formalism_unique_fields test/test_config_layers.py::test_field_routing_nested_formalism_collision_raises -v
```

Expected: failures — current `FIELD_TO_LAYER` is flat to a single layer name (string), not a tuple.

- [ ] **Step 3: Rewrite `pyphi/conf/_field_routing.py`**

```python
"""Build-time map from flat field name to owning (layer, sub-namespace) path.

Used by :class:`pyphi.conf._global._GlobalConfig` ``__setattr__`` to route
``config.precision = 6`` to the correct frozen layer (and sub-namespace),
and by ``override(**kwargs)`` to dispatch kwargs.

For non-formalism layers (``infrastructure``, ``numerics``), the routing
target is ``(layer_name, None)``. For formalism, the target is
``("formalism", "iit")`` or ``("formalism", "actual_causation")``.

Field names that exist in both IIT and AC sub-namespaces are excluded from
the flat map — they require nested writes to disambiguate. Raises at
module import time if non-formalism layers have a name collision —
fail-fast prevents silent misdispatch.
"""

from __future__ import annotations

from dataclasses import fields

from pyphi.conf.formalism import ActualCausationConfig
from pyphi.conf.formalism import FormalismConfig
from pyphi.conf.formalism import IITConfig
from pyphi.conf.infrastructure import InfrastructureConfig
from pyphi.conf.numerics import NumericsConfig


class ConfigurationError(ValueError):
    """Raised on config schema problems (collisions, unknown options, etc.)."""


def _build_field_map() -> dict[str, tuple[str, str | None]]:
    out: dict[str, tuple[str, str | None]] = {}
    flat_layers: list[tuple[str, type]] = [
        ("infrastructure", InfrastructureConfig),
        ("numerics", NumericsConfig),
    ]
    for layer_name, layer_cls in flat_layers:
        for f in fields(layer_cls):
            if f.name in out:
                raise ConfigurationError(
                    f"Config field name collision: {f.name!r} appears in both "
                    f"{out[f.name]!r} and ({layer_name!r}, None). Rename one."
                )
            out[f.name] = (layer_name, None)

    iit_field_names = {f.name for f in fields(IITConfig)}
    ac_field_names = {f.name for f in fields(ActualCausationConfig)}

    # Top-level FormalismConfig fields (iit, actual_causation) themselves
    for f in fields(FormalismConfig):
        if f.name in out:
            raise ConfigurationError(
                f"Config field name collision: {f.name!r} appears in both "
                f"{out[f.name]!r} and ('formalism', None)."
            )
        out[f.name] = ("formalism", None)

    # IIT sub-namespace fields, except those colliding with AC
    for name in iit_field_names - ac_field_names:
        if name in out:
            raise ConfigurationError(
                f"Config field name collision: IIT field {name!r} also appears in "
                f"{out[name]!r}."
            )
        out[name] = ("formalism", "iit")

    # AC sub-namespace fields, except those colliding with IIT
    for name in ac_field_names - iit_field_names:
        if name in out:
            raise ConfigurationError(
                f"Config field name collision: AC field {name!r} also appears in "
                f"{out[name]!r}."
            )
        out[name] = ("formalism", "actual_causation")

    return out


FIELD_TO_LAYER: dict[str, tuple[str, str | None]] = _build_field_map()


def colliding_formalism_fields() -> set[str]:
    """Field names that exist in both IIT and AC sub-namespaces.

    These are excluded from the flat ``FIELD_TO_LAYER`` map — flat writes
    to them must raise :class:`ConfigurationError` directing the user to
    the nested form.
    """
    iit_field_names = {f.name for f in fields(IITConfig)}
    ac_field_names = {f.name for f in fields(ActualCausationConfig)}
    return iit_field_names & ac_field_names
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest test/test_config_layers.py::test_field_routing_nested_formalism_unique_fields test/test_config_layers.py::test_field_routing_nested_formalism_collision_raises -v
```

Expected: 2 passed.

## Task 2.3: Update `_GlobalConfig.__setattr__` for nested-formalism routing

**Files:**
- Modify: `pyphi/conf/_global.py:155-186`

- [ ] **Step 1: Write the failing test**

Add to `test/test_config_layers.py`:

```python
def test_global_config_flat_write_unique_iit_field():
    from pyphi.conf import config

    with config.override(repertoire_measure="EMD"):
        assert config.formalism.iit.repertoire_measure == "EMD"
    assert config.formalism.iit.repertoire_measure == "GENERALIZED_INTRINSIC_DIFFERENCE"


def test_global_config_flat_write_unique_ac_field():
    from pyphi.conf import config

    with config.override(measure="KLD"):  # AC's measure
        assert config.formalism.actual_causation.measure == "KLD"
    assert config.formalism.actual_causation.measure == "PMI"


def test_global_config_flat_write_colliding_field_raises():
    import pytest

    from pyphi.conf import ConfigurationError, config

    with pytest.raises(ConfigurationError, match="ambiguous"):
        config.mechanism_partition_scheme = "BI"


def test_global_config_nested_formalism_replacement():
    from pyphi.conf import config
    from pyphi.conf.formalism import IITConfig

    new_iit = IITConfig(repertoire_measure="EMD")
    with config.override(iit=new_iit):
        assert config.formalism.iit.repertoire_measure == "EMD"
    assert config.formalism.iit.repertoire_measure == "GENERALIZED_INTRINSIC_DIFFERENCE"


def test_global_config_uppercase_iit_alias():
    from pyphi.conf import config

    # `config.REPERTOIRE_MEASURE` should map to formalism.iit.repertoire_measure
    assert config.REPERTOIRE_MEASURE == config.formalism.iit.repertoire_measure
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest test/test_config_layers.py::test_global_config_flat_write_unique_iit_field test/test_config_layers.py::test_global_config_flat_write_unique_ac_field test/test_config_layers.py::test_global_config_flat_write_colliding_field_raises test/test_config_layers.py::test_global_config_nested_formalism_replacement test/test_config_layers.py::test_global_config_uppercase_iit_alias -v
```

Expected: failures — current `__setattr__` doesn't handle nested routing.

- [ ] **Step 3: Update `_GlobalConfig.__setattr__`**

In `pyphi/conf/_global.py`, replace `__setattr__` (lines 155-186) and `__getattr__` (lines 147-153) with:

```python
    def __getattr__(self, name: str) -> Any:
        if name.isupper():
            field_name = name.lower()
            if field_name in FIELD_TO_LAYER:
                target = FIELD_TO_LAYER[field_name]
                return _read_via_target(self, target, field_name)
        if name in colliding_formalism_fields():
            raise AttributeError(
                f"{name!r} is ambiguous (exists in both formalism.iit and "
                "formalism.actual_causation). Use the qualified path: "
                f"config.formalism.iit.{name} or "
                f"config.formalism.actual_causation.{name}."
            )
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return

        field_name = name.lower() if name.isupper() else name

        # Wholesale layer replacement: `config.numerics = NumericsConfig(...)`,
        # `config.formalism = FormalismConfig(...)`.
        if field_name in _LAYER_NAMES and isinstance(value, _LAYER_TYPES[field_name]):
            old_layer = getattr(self, "_" + field_name)
            object.__setattr__(self, "_" + field_name, value)
            self._fire_layer_replacement_callbacks(old_layer, value)
            return

        # Wholesale formalism sub-namespace replacement: `config.iit = IITConfig(...)`,
        # `config.actual_causation = ActualCausationConfig(...)`.
        if field_name == "iit" and isinstance(value, IITConfig):
            old_formalism = self._formalism
            new_formalism = replace(old_formalism, iit=value)
            object.__setattr__(self, "_formalism", new_formalism)
            self._fire_layer_replacement_callbacks(old_formalism, new_formalism)
            return
        if field_name == "actual_causation" and isinstance(value, ActualCausationConfig):
            old_formalism = self._formalism
            new_formalism = replace(old_formalism, actual_causation=value)
            object.__setattr__(self, "_formalism", new_formalism)
            self._fire_layer_replacement_callbacks(old_formalism, new_formalism)
            return

        if field_name in colliding_formalism_fields():
            raise ConfigurationError(
                f"Field {field_name!r} is ambiguous (exists in both formalism.iit "
                "and formalism.actual_causation). Use the qualified path: "
                f"config.formalism.iit.{field_name} = {value!r} (via nested replace) "
                "or config.iit = replace(config.formalism.iit, ...)."
            )

        if field_name in FIELD_TO_LAYER:
            target = FIELD_TO_LAYER[field_name]
            _write_via_target(self, target, field_name, value)
            self._fire_field_callback(field_name)
            return

        if field_name in _LAYER_NAMES:
            expected = _LAYER_TYPES[field_name]
            raise ConfigurationError(
                f"Cannot replace layer {field_name!r} with "
                f"{type(value).__name__}; expected {expected.__name__}."
            )
        raise ConfigurationError(
            f"Unknown config option: {name!r}. "
            "See changelog.d/p10-config-split.refactor.md for the rename map."
        )
```

Add at the top of the file (after the existing imports):

```python
from pyphi.conf._field_routing import colliding_formalism_fields
from pyphi.conf.formalism import ActualCausationConfig
from pyphi.conf.formalism import IITConfig
```

Add helper functions just below the `_LOG_FIELDS` constant:

```python
def _read_via_target(
    cfg: _GlobalConfig, target: tuple[str, str | None], field_name: str
) -> Any:
    layer_name, sub_namespace = target
    layer = getattr(cfg, "_" + layer_name)
    if sub_namespace is None:
        return getattr(layer, field_name)
    return getattr(getattr(layer, sub_namespace), field_name)


def _write_via_target(
    cfg: _GlobalConfig, target: tuple[str, str | None], field_name: str, value: Any
) -> None:
    layer_name, sub_namespace = target
    layer_attr = "_" + layer_name
    current_layer = getattr(cfg, layer_attr)
    if sub_namespace is None:
        new_layer = replace(current_layer, **{field_name: value})
    else:
        current_sub = getattr(current_layer, sub_namespace)
        new_sub = replace(current_sub, **{field_name: value})
        new_layer = replace(current_layer, **{sub_namespace: new_sub})
    object.__setattr__(cfg, layer_attr, new_layer)
```

- [ ] **Step 4: Update `_GlobalConfig.__init__` to also configure log on construction**

Verify line 55-59 still works (no changes needed there).

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest test/test_config_layers.py -v
```

Expected: all green, including the new tests.

## Task 2.4: Update YAML loader for nested format

**Files:**
- Modify: `pyphi/conf/_io.py`
- Modify: `pyphi/conf/_global.py:102-114` (`load_yaml`)

- [ ] **Step 1: Read current `pyphi/conf/_io.py`**

Open and read the existing implementation to understand the YAML loading flow.

- [ ] **Step 2: Write the failing test**

Add to `test/test_config_layers.py`:

```python
def test_load_yaml_nested_formalism(tmp_path):
    from pyphi.conf._global import _GlobalConfig

    yaml_text = """
formalism:
  iit:
    version: IIT_3_0
    repertoire_measure: EMD
  actual_causation:
    measure: KLD
    partitioned_repertoire_scheme: PRODUCT
infrastructure:
  precision_unused_below_this_line: ignored
numerics:
  precision: 6
"""
    yaml_path = tmp_path / "pyphi_config.yml"
    yaml_path.write_text(yaml_text.replace("precision_unused_below_this_line", "log_file_level"))
    cfg = _GlobalConfig()
    cfg.load_yaml(yaml_path)
    assert cfg.formalism.iit.version == "IIT_3_0"
    assert cfg.formalism.iit.repertoire_measure == "EMD"
    assert cfg.formalism.actual_causation.measure == "KLD"
    assert cfg.numerics.precision == 6
```

- [ ] **Step 3: Run test to verify it fails**

Expected: failure — current `load_yaml` iterates `data.values()` flat, doesn't handle the nested formalism structure.

- [ ] **Step 4: Update `_GlobalConfig.load_yaml`**

In `pyphi/conf/_global.py`, replace the `load_yaml` method body (lines 102-114) with:

```python
    def load_yaml(self, path: str | Path) -> None:
        """Load a 2.0 nested-format YAML config file.

        Each layer's section is applied via per-field writes. Formalism's
        nested ``iit`` and ``actual_causation`` sections route through the
        sub-namespace replacement path. Raises :class:`ConfigurationError`
        on unrecognized keys or 1.x flat format.
        """
        from pyphi.conf._io import load_yaml as _load

        data = _load(path)
        formalism_data = data.pop("formalism", {})
        for fields_dict in data.values():
            for field_name, value in fields_dict.items():
                setattr(self, field_name, value)
        # Formalism: handle nested iit / actual_causation explicitly.
        for sub_name in ("iit", "actual_causation"):
            sub_data = formalism_data.get(sub_name, {})
            for field_name, value in sub_data.items():
                setattr(self, field_name, value)
        # Top-level formalism fields (currently none beyond iit/actual_causation)
        # are tolerated as no-ops.
```

- [ ] **Step 5: Run test to verify it passes**

```bash
uv run pytest test/test_config_layers.py::test_load_yaml_nested_formalism -v
```

Expected: pass.

## Task 2.5: Update snapshot to handle nested formalism

**Files:**
- Modify: `pyphi/conf/snapshot.py`

- [ ] **Step 1: Read current `pyphi/conf/snapshot.py`**

```bash
cat pyphi/conf/snapshot.py
```

- [ ] **Step 2: Write the failing test**

Add to `test/test_config_layers.py`:

```python
def test_snapshot_includes_nested_formalism():
    from pyphi.conf import config
    from pyphi.conf.formalism import ActualCausationConfig, IITConfig

    snap = config.snapshot()
    assert isinstance(snap.formalism, type(config.formalism))
    assert isinstance(snap.formalism.iit, IITConfig)
    assert isinstance(snap.formalism.actual_causation, ActualCausationConfig)


def test_snapshot_as_kwargs_flattens_unique_fields():
    """Snapshot.as_kwargs returns all unique-named fields as flat kwargs.
    Colliding fields require nested replacement and are NOT in as_kwargs."""
    from pyphi.conf import config

    snap = config.snapshot()
    kw = snap.as_kwargs()
    # Unique IIT fields:
    assert "repertoire_measure" in kw
    assert kw["repertoire_measure"] == "GENERALIZED_INTRINSIC_DIFFERENCE"
    # Unique AC fields:
    assert "measure" in kw
    assert kw["measure"] == "PMI"
    # Colliding name not in kwargs (must be set via nested replacement):
    assert "mechanism_partition_scheme" not in kw
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run pytest test/test_config_layers.py::test_snapshot_includes_nested_formalism test/test_config_layers.py::test_snapshot_as_kwargs_flattens_unique_fields -v
```

Expected: failures.

- [ ] **Step 4: Update `pyphi/conf/snapshot.py`**

Read the current `as_kwargs` and update its implementation to handle nested formalism. The exact body depends on what's there — replace the body that walks formalism's fields with one that walks IIT and AC sub-namespaces, EXCLUDING fields that collide between them.

```python
def as_kwargs(self) -> dict[str, Any]:
    from dataclasses import asdict, fields
    from pyphi.conf._field_routing import colliding_formalism_fields

    out: dict[str, Any] = {}
    # infrastructure + numerics: all fields are flat
    for layer in (self.infrastructure, self.numerics):
        for f in fields(layer):
            out[f.name] = getattr(layer, f.name)
    # formalism: walk iit + actual_causation, excluding colliding names
    excluded = colliding_formalism_fields()
    for sub_name in ("iit", "actual_causation"):
        sub_layer = getattr(self.formalism, sub_name)
        for f in fields(sub_layer):
            if f.name in excluded:
                continue
            out[f.name] = getattr(sub_layer, f.name)
    return out
```

(If `as_kwargs` already takes a different shape, adapt; the contract is: return all unique-named fields as flat kwargs.)

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest test/test_config_layers.py::test_snapshot_includes_nested_formalism test/test_config_layers.py::test_snapshot_as_kwargs_flattens_unique_fields -v
```

Expected: pass.

## Task 2.6: Migrate `pyphi_config.yml`

**Files:**
- Modify: `pyphi_config.yml`

- [ ] **Step 1: Read current `pyphi_config.yml`**

- [ ] **Step 2: Restructure to nested format**

Rewrite to:

```yaml
formalism:
  iit:
    version: IIT_4_0_2023
    repertoire_measure: GENERALIZED_INTRINSIC_DIFFERENCE
    repertoire_measure_specification: GENERALIZED_INTRINSIC_DIFFERENCE
    repertoire_measure_differentiation: GENERALIZED_INTRINSIC_DIFFERENCE
    ces_measure: SUM_SMALL_PHI
    mechanism_partition_scheme: ALL
    system_partition_scheme: SET_UNI/BI
    system_partition_include_complete: false
    distinction_phi_normalization: NUM_CONNECTIONS_CUT
    relation_computation: CONCRETE
    assume_partitions_cannot_create_new_concepts: false
    shortcircuit_sia: true
    single_micro_nodes_with_selfloops_have_phi: true
    state_tie_resolution: PHI
    mip_tie_resolution:
      - NORMALIZED_PHI
      - NEGATIVE_PHI
    purview_tie_resolution: PHI
  actual_causation:
    measure: PMI
    mechanism_partition_scheme: ALL
    partitioned_repertoire_scheme: PRODUCT
    background_strategy: UNIFORM
    alpha_aggregation: SUBTRACTIVE

infrastructure:
  # ... existing infrastructure fields preserved as-is ...

numerics:
  # ... existing numerics fields preserved as-is ...
```

Preserve all `infrastructure` and `numerics` field values from the current file unchanged.

- [ ] **Step 3: Verify config loads**

```bash
uv run python -c "from pyphi.conf import config; print(config.formalism.iit.version, config.formalism.actual_causation.measure)"
```

Expected: `IIT_4_0_2023 PMI`.

## Task 2.7: Update all call sites

**Files:**
- Modify: `pyphi/system.py:478`
- Modify: `pyphi/partition.py:432, 862, 897`
- Modify: `pyphi/relations.py:389`
- Modify: `pyphi/resolve_ties.py:123, 135, 147`
- Modify: `pyphi/actual.py:372, 588, 596, 602` (still keeps line 372 reading the IIT key for now; Phase 4 decouples it)
- Modify: `pyphi/metrics/distribution.py:938, 939, 1054`
- Modify: `pyphi/metrics/ces.py:83, 85, 246`
- Modify: `pyphi/core/repertoire_algebra.py:297, 530`
- Modify: `pyphi/models/state_specification.py:175`
- Modify: `pyphi/visualize/distribution.py:154`
- Modify: `pyphi/formalism/__init__.py:9, 39`
- Modify: `pyphi/formalism/base.py:46, 173, 199`
- Modify: `pyphi/formalism/iit3/formalism.py:86, 123, 125, 158`
- Modify: `pyphi/formalism/iit4/formalism.py:67, 207, 221, 228, 233`
- Modify: `pyphi/formalism/queries.py:69, 173, 348`
- Modify: `pyphi/formalism/iit3/__init__.py:380` (`config.infrastructure.clear_system_caches_after_computing_sia` is in infrastructure — no rename needed — but the surrounding partition_type / system_partition_type calls are)

- [ ] **Step 1: Run a survey of all call sites**

```bash
grep -rn "config\.formalism\.\(formalism\|repertoire_distance\|repertoire_distance_specification\|repertoire_distance_differentiation\|ces_distance\|partition_type\|system_partition_type\|system_partition_include_complete\|distinction_phi_normalization\|relation_computation\|assume_cuts_cannot_create_new_concepts\|shortcircuit_sia\|single_micro_nodes_with_selfloops_have_phi\|state_tie_resolution\|mip_tie_resolution\|purview_tie_resolution\|actual_causation_measure\)" pyphi/ test/ | tee /tmp/p14_rename_sites.txt
```

This produces an exhaustive list of every line needing rename.

- [ ] **Step 2: Apply the rename mechanically**

For each entry in `/tmp/p14_rename_sites.txt`, edit the file and apply the rename map:

| Old | New |
|---|---|
| `config.formalism.formalism` | `config.formalism.iit.version` |
| `config.formalism.repertoire_distance` | `config.formalism.iit.repertoire_measure` |
| `config.formalism.repertoire_distance_specification` | `config.formalism.iit.repertoire_measure_specification` |
| `config.formalism.repertoire_distance_differentiation` | `config.formalism.iit.repertoire_measure_differentiation` |
| `config.formalism.ces_distance` | `config.formalism.iit.ces_measure` |
| `config.formalism.partition_type` | `config.formalism.iit.mechanism_partition_scheme` |
| `config.formalism.system_partition_type` | `config.formalism.iit.system_partition_scheme` |
| `config.formalism.system_partition_include_complete` | `config.formalism.iit.system_partition_include_complete` |
| `config.formalism.distinction_phi_normalization` | `config.formalism.iit.distinction_phi_normalization` |
| `config.formalism.relation_computation` | `config.formalism.iit.relation_computation` |
| `config.formalism.assume_cuts_cannot_create_new_concepts` | `config.formalism.iit.assume_partitions_cannot_create_new_concepts` |
| `config.formalism.shortcircuit_sia` | `config.formalism.iit.shortcircuit_sia` |
| `config.formalism.single_micro_nodes_with_selfloops_have_phi` | `config.formalism.iit.single_micro_nodes_with_selfloops_have_phi` |
| `config.formalism.state_tie_resolution` | `config.formalism.iit.state_tie_resolution` |
| `config.formalism.mip_tie_resolution` | `config.formalism.iit.mip_tie_resolution` |
| `config.formalism.purview_tie_resolution` | `config.formalism.iit.purview_tie_resolution` |
| `config.formalism.actual_causation_measure` | `config.formalism.actual_causation.measure` |

Use the `Edit` tool with `replace_all=true` per file for files with multiple occurrences of the same old key.

- [ ] **Step 3: Update test files using `config.override(<old_key>=...)`**

```bash
grep -rn "config\.override(\(repertoire_distance\|partition_type\|system_partition_type\|distinction_phi_normalization\|relation_computation\|state_tie_resolution\|mip_tie_resolution\|purview_tie_resolution\|shortcircuit_sia\|actual_causation_measure\)" test/
```

For each, rename the kwarg per the map. The flat-write routing in `_GlobalConfig.__setattr__` (Task 2.3) handles unique fields automatically; tests that override `mechanism_partition_scheme` flat will fail with `ConfigurationError` — those need to be rewritten to use nested replacement:

```python
# Old:
@config.override(partition_type="DIRECTED_BI")
# New:
@config.override(mechanism_partition_scheme="DIRECTED_BI")  # disambiguated by being a unique IIT name? NO — collides with AC.
```

Wait — `mechanism_partition_scheme` collides between IIT and AC. For tests that previously used `partition_type` (now renamed to `mechanism_partition_scheme` on IIT side), the test must use the nested form:

```python
from pyphi.conf.formalism import IITConfig
@config.override(iit=IITConfig(mechanism_partition_scheme="DIRECTED_BI"))
```

OR the test framework can use the qualified-path API once that exists. For Phase 2, the explicit `IITConfig(...)` form is the safe path.

- [ ] **Step 4: Verify config tests still pass**

```bash
uv run pytest test/test_config.py test/test_config_layers.py -v
```

Expected: all green.

## Task 2.8: Run full acceptance gates and commit

- [ ] **Step 1: Pyright**

```bash
uv run pyright pyphi/
```

Expected: zero new errors.

- [ ] **Step 2: Ruff**

```bash
uv run ruff check pyphi/ test/
```

Expected: clean.

- [ ] **Step 3: Fast unit lane**

```bash
uv run pytest test/test_invariants.py test/test_subsystem_surface.py test/test_formalism_pickle.py test/test_parallel.py test/test_scheduler.py test/test_sampling.py test/test_install_snapshot.py test/test_config.py test/test_config_layers.py -q
```

Expected: green.

- [ ] **Step 4: Hypothesis fast lane**

```bash
uv run pytest test/test_invariants_hypothesis.py -q
```

Expected: 21 properties pass.

- [ ] **Step 5: Golden 17/17 (background)**

```bash
uv run pytest test/test_golden_regression.py -q
```

Expected: 17/17. **Note**: The serialized config blobs in golden fixtures will be in old flat form. Two outcomes are acceptable:

1. The fixture loader handles old-format gracefully (legacy compat for fixture loading) → fixtures pass unchanged.
2. The fixture loader rejects old format → regenerate fixtures with `uv run pytest test/test_golden_regression.py --regenerate-golden -q` and verify the regenerated fixtures' `phi`, `signed_phi`, `partition`, `cause`, `effect` values match the originals to 1e-12 (a manual diff before committing).

If regeneration is required: regenerate, diff numerically against the original `phi` values, commit the regenerated fixtures alongside the rename in the same commit.

- [ ] **Step 6: Stage and commit**

```bash
git add -A
git status
git commit -m "$(cat <<'EOF'
Restructure formalism config: nested iit / actual_causation namespaces

FormalismConfig becomes a thin holder of two nested frozen dataclasses,
IITConfig and ActualCausationConfig. Field-routing infrastructure handles
nested formalism via path tuples (layer, sub-namespace); unique-named
fields remain flat-routable; colliding names (mechanism_partition_scheme
in both IIT and AC) require nested replacement.

Rename map applied across ~30 call sites:
  formalism.formalism → formalism.iit.version
  formalism.repertoire_distance → formalism.iit.repertoire_measure (+ specification, differentiation)
  formalism.ces_distance → formalism.iit.ces_measure
  formalism.partition_type → formalism.iit.mechanism_partition_scheme
  formalism.system_partition_type → formalism.iit.system_partition_scheme
  formalism.assume_cuts_cannot_create_new_concepts →
    formalism.iit.assume_partitions_cannot_create_new_concepts
  formalism.actual_causation_measure → formalism.actual_causation.measure

Naming principles:
  - "measure" matches the metrics.distribution.measures registry
  - "scheme" reads naturally for partition-generator registries
  - "partition" replaces "cut" where the concept is type/operation-level
    (the runtime-state verb/noun "cut" survives in apply_cut, is_cut, etc.)

Adds AC-specific knobs (mechanism_partition_scheme,
partitioned_repertoire_scheme, background_strategy, alpha_aggregation)
with paper-faithful defaults; not yet wired into computation (the
computation-level decoupling lands in a subsequent commit).

pyphi_config.yml migrated to nested format.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

**Acceptance:** All gates green; commit lands.

---

# Phase 3: Add `TransitionSystem` class + AC registries

**Why third:** Implements the new type without touching `Transition`. Existing tests stay skipped; new `TransitionSystem` unit tests pass in isolation.

**Files affected:**
- Modify: `pyphi/actual.py` (add `TransitionSystem`, three new registries, paper-faithful default schemes/strategies/aggregations)
- Modify: `pyphi/__init__.py` (export `TransitionSystem`)
- Modify: `test/test_actual.py` (add `test_transition_system_*` tests under a per-test skip lift)

## Task 3.1: Define new AC registries with paper-faithful defaults

**Files:**
- Modify: `pyphi/actual.py` (add at module level, after the imports)

- [ ] **Step 1: Write the failing test**

Add to `test/test_actual.py` (above the `pytestmark = pytest.mark.skip(...)` line, so it runs):

```python
# These tests run independent of the module-level skip; they don't touch Transition.
import pytest as _pytest

_pytest.importorskip("pyphi")


def test_actual_partitioned_repertoire_schemes_registry():
    from pyphi import actual

    assert "PRODUCT" in actual.partitioned_repertoire_schemes
    # FORWARD_PROBABILITY is not registered in this commit (defaults only).
    assert "FORWARD_PROBABILITY" not in actual.partitioned_repertoire_schemes


def test_actual_background_strategies_registry():
    from pyphi import actual

    assert "UNIFORM" in actual.background_strategies
    assert "STATIONARY" not in actual.background_strategies
    assert "OBSERVED" not in actual.background_strategies


def test_actual_alpha_aggregations_registry():
    from pyphi import actual

    assert "SUBTRACTIVE" in actual.alpha_aggregations
    assert "RATIO" not in actual.alpha_aggregations
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest test/test_actual.py::test_actual_partitioned_repertoire_schemes_registry test/test_actual.py::test_actual_background_strategies_registry test/test_actual.py::test_actual_alpha_aggregations_registry -v
```

Expected: fail (registries don't exist).

- [ ] **Step 3: Add registries and default-only registrations to `pyphi/actual.py`**

Near the top of `pyphi/actual.py` (after the existing imports and before `class Transition`), add:

```python
from pyphi.registry import Registry


class PartitionedRepertoireSchemeRegistry(Registry):
    """Registry of partitioned-repertoire computation schemes for AC.

    Schemes consume ``(transition_system, direction, partition)`` and
    return the partitioned repertoire as a probability distribution
    consistent with the parent System's TPM shape.
    """
    desc = "partitioned-repertoire schemes"


class BackgroundStrategyRegistry(Registry):
    """Registry of background-conditioning strategies for AC.

    Strategies consume ``(substrate, before_state, external_indices)``
    and return a probability weight per external state used during
    causal marginalization (paper Eq 2).
    """
    desc = "background-conditioning strategies"


class AlphaAggregationRegistry(Registry):
    """Registry of α-aggregation rules for AC.

    Aggregators consume ``(rho, rho_partitioned)`` and return α — the
    integrated information of an actual cause/effect link (paper Eq 15).
    """
    desc = "α-aggregation rules"


partitioned_repertoire_schemes = PartitionedRepertoireSchemeRegistry()
background_strategies = BackgroundStrategyRegistry()
alpha_aggregations = AlphaAggregationRegistry()


@partitioned_repertoire_schemes.register("PRODUCT")
def _partitioned_repertoire_product(
    transition_system: Any,
    direction: Direction,
    partition: Any,
) -> Any:
    """2019 Albantakis et al. Eq 8: product of per-part repertoires.

    The partitioned repertoire is the product of the per-part cause/effect
    repertoires, multiplied by the unconstrained repertoire over any
    nodes not assigned to a part.
    """
    # Delegate to the underlying System's partitioned_repertoire (which
    # routes through pyphi.core.repertoire_algebra).
    return transition_system.partitioned_repertoire(direction, partition)


@background_strategies.register("UNIFORM")
def _background_uniform(
    substrate: Any,
    before_state: Any,
    external_indices: Any,
) -> Any:
    """2019 Albantakis et al. Eq 2: uniform causal marginalization.

    Each external state receives equal weight |Ω_W|^(-1) during the
    causal marginalization. The legacy substrate-level marginalization
    machinery in pyphi.core.tpm is the operational implementation; this
    registration documents the choice.
    """
    return None  # Sentinel: caller takes the uniform-weight branch.


@alpha_aggregations.register("SUBTRACTIVE")
def _alpha_subtractive(rho: float, rho_partitioned: float) -> float:
    """2019 Albantakis et al. Eq 15: α = ρ − ρ_partition.

    Used by Transition.find_mip when minimizing α over partitions.
    """
    return rho - rho_partitioned
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest test/test_actual.py::test_actual_partitioned_repertoire_schemes_registry test/test_actual.py::test_actual_background_strategies_registry test/test_actual.py::test_actual_alpha_aggregations_registry -v
```

Expected: 3 passed.

## Task 3.2: Define `TransitionSystem` class

**Files:**
- Modify: `pyphi/actual.py` (add after the registries from Task 3.1)

- [ ] **Step 1: Write the failing tests**

Add to `test/test_actual.py` (above the module-level skip):

```python
def test_transition_system_is_frozen():
    """TransitionSystem is a frozen dataclass — direct mutation must fail."""
    import dataclasses

    import numpy as np

    from pyphi import Direction, Substrate
    from pyphi.actual import TransitionSystem

    tpm = np.array(
        [
            [0, 0.5, 0.5],
            [0, 0.5, 0.5],
            [1, 0.5, 0.5],
            [1, 0.5, 0.5],
            [1, 0.5, 0.5],
            [1, 0.5, 0.5],
            [1, 0.5, 0.5],
            [1, 0.5, 0.5],
        ]
    )
    cm = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]])
    substrate = Substrate(tpm, cm)
    ts = TransitionSystem(
        substrate=substrate,
        before_state=(0, 1, 1),
        after_state=(1, 0, 0),
        cause_indices=(1, 2),
        effect_indices=(0,),
        direction=Direction.CAUSE,
    )
    import pytest as _pt

    with _pt.raises(dataclasses.FrozenInstanceError):
        ts.before_state = (1, 1, 1)


def test_transition_system_satisfies_protocol():
    """TransitionSystem satisfies SystemPublicInterface via runtime_checkable."""
    import numpy as np

    from pyphi import Direction, Substrate
    from pyphi.actual import TransitionSystem
    from pyphi.protocols import SystemPublicInterface

    tpm = np.array(
        [[0, 0.5, 0.5]] * 1 + [[0, 0.5, 0.5]] * 1 + [[1, 0.5, 0.5]] * 6
    )
    cm = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]])
    substrate = Substrate(tpm, cm)
    ts = TransitionSystem(
        substrate=substrate,
        before_state=(0, 1, 1),
        after_state=(1, 0, 0),
        cause_indices=(1, 2),
        effect_indices=(0,),
        direction=Direction.CAUSE,
    )
    assert isinstance(ts, SystemPublicInterface)


def test_transition_system_cause_uses_after_state():
    """CAUSE-direction TransitionSystem.state == after_state."""
    import numpy as np

    from pyphi import Direction, Substrate
    from pyphi.actual import TransitionSystem

    tpm = np.array([[0, 0.5, 0.5]] * 8)
    cm = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]])
    substrate = Substrate(tpm, cm)
    ts = TransitionSystem(
        substrate=substrate,
        before_state=(0, 1, 1),
        after_state=(1, 0, 0),
        cause_indices=(1, 2),
        effect_indices=(0,),
        direction=Direction.CAUSE,
    )
    assert ts.state == (1, 0, 0)


def test_transition_system_effect_uses_before_state():
    """EFFECT-direction TransitionSystem.state == before_state."""
    import numpy as np

    from pyphi import Direction, Substrate
    from pyphi.actual import TransitionSystem

    tpm = np.array([[0, 0.5, 0.5]] * 8)
    cm = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]])
    substrate = Substrate(tpm, cm)
    ts = TransitionSystem(
        substrate=substrate,
        before_state=(0, 1, 1),
        after_state=(1, 0, 0),
        cause_indices=(1, 2),
        effect_indices=(0,),
        direction=Direction.EFFECT,
    )
    assert ts.state == (0, 1, 1)


def test_transition_system_external_indices_excludes_cause_indices():
    """external_indices = substrate.node_indices - cause_indices (paper-faithful)."""
    import numpy as np

    from pyphi import Direction, Substrate
    from pyphi.actual import TransitionSystem

    tpm = np.array([[0, 0.5, 0.5]] * 8)
    cm = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]])
    substrate = Substrate(tpm, cm)
    ts = TransitionSystem(
        substrate=substrate,
        before_state=(0, 1, 1),
        after_state=(1, 0, 0),
        cause_indices=(1, 2),
        effect_indices=(0,),
        direction=Direction.CAUSE,
    )
    # cause_indices is (1, 2), so external = (0,)
    assert ts.external_indices == (0,)


def test_transition_system_apply_cut_returns_new_instance():
    """apply_cut returns a new TransitionSystem with the new cut; original unchanged."""
    import numpy as np

    from pyphi import Direction, Substrate
    from pyphi.actual import TransitionSystem
    from pyphi.models.cuts import SystemPartition

    tpm = np.array([[0, 0.5, 0.5]] * 8)
    cm = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]])
    substrate = Substrate(tpm, cm)
    ts = TransitionSystem(
        substrate=substrate,
        before_state=(0, 1, 1),
        after_state=(1, 0, 0),
        cause_indices=(1, 2),
        effect_indices=(0,),
        direction=Direction.CAUSE,
    )
    new_cut = SystemPartition(
        Direction.CAUSE, (0, 1), (2,), substrate.node_labels
    )
    ts2 = ts.apply_cut(new_cut)
    assert ts2 is not ts
    assert ts2.cut == new_cut
    # Original cut unchanged (NullCut in this case)
    from pyphi.models.cuts import NullCut

    assert isinstance(ts.cut, NullCut)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest test/test_actual.py::test_transition_system_is_frozen -v
```

Expected: ImportError or AttributeError — `TransitionSystem` doesn't exist.

- [ ] **Step 3: Implement `TransitionSystem` in `pyphi/actual.py`**

Add to `pyphi/actual.py` (after the registries from Task 3.1, before the existing `class Transition:`):

```python
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from functools import cached_property
from typing import Any

from pyphi import utils
from pyphi import validate
from pyphi.conf import config
from pyphi.direction import Direction
from pyphi.models.cuts import NullCut
from pyphi.models.cuts import SystemPartition
from pyphi.substrate import Substrate

# Import lazily inside methods to avoid circular imports if needed.


@dataclass(frozen=True, eq=False)
class TransitionSystem:
    """A directional view of a state transition.

    Implements :class:`pyphi.protocols.SystemPublicInterface` via the
    standard System surface (cause_tpm, effect_tpm, cm, node_indices,
    state, repertoire methods, etc.).

    The TPMs are conditioned on ``before_state`` for every substrate
    index outside ``cause_indices`` (the asymmetric background-conditioning
    rule from the 2019 Albantakis et al. formalism). The mechanism-
    evaluation ``state`` is ``after_state`` for the CAUSE direction and
    ``before_state`` for the EFFECT direction. Two TransitionSystem
    instances live inside each :class:`Transition`, one per direction.
    """

    substrate: Substrate
    before_state: tuple[int, ...]
    after_state: tuple[int, ...]
    cause_indices: tuple[int, ...]
    effect_indices: tuple[int, ...]
    direction: Direction
    cut: SystemPartition = field(default=None)  # type: ignore[assignment]
    noise_background: bool = False

    def __post_init__(self) -> None:
        validate.state_length(self.before_state, self.substrate.size)
        validate.state_length(self.after_state, self.substrate.size)
        validate.node_states(self.before_state)
        validate.node_states(self.after_state)
        coerce = self.substrate.node_labels.coerce_to_indices
        object.__setattr__(self, "cause_indices", coerce(self.cause_indices))
        object.__setattr__(self, "effect_indices", coerce(self.effect_indices))
        if self.cut is None:
            object.__setattr__(
                self, "cut", NullCut(self.node_indices, self.substrate.node_labels)
            )
        if (
            self.direction == Direction.CAUSE
            and config.infrastructure.validate_system_states
        ):
            # Cause-side state is `after_state`; it must be reachable.
            # Build a temporary System view in the after_state for validation.
            from pyphi.system import System

            temp_system = System(
                substrate=self.substrate,
                state=self.after_state,
                node_indices=self.node_indices,
                cut=self.cut,
            )
            validate.state_reachable(temp_system)

    @cached_property
    def node_indices(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.cause_indices) | set(self.effect_indices)))

    @cached_property
    def state(self) -> tuple[int, ...]:
        return self.after_state if self.direction == Direction.CAUSE else self.before_state

    @cached_property
    def external_indices(self) -> tuple[int, ...]:
        if self.noise_background:
            return ()
        all_indices = set(self.substrate.node_indices)
        return tuple(sorted(all_indices - set(self.cause_indices)))

    @cached_property
    def node_labels(self) -> Any:
        return self.substrate.node_labels

    @cached_property
    def proper_state(self) -> Any:
        return utils.state_of(self.node_indices, self.state)

    # The cause_tpm / effect_tpm cached_properties build TPMs conditioned on
    # before_state for external indices. Reuse the System machinery by
    # constructing a System with the appropriate state and external_indices.
    @cached_property
    def _underlying_system(self) -> Any:
        """A System instance used as the TPM-derivation engine.

        The System is constructed in ``before_state`` (so its TPM
        marginalization conditions on before_state for all external
        indices), with node_indices = cause_indices for the cause-side
        TPM derivation. Used internally; not part of the Protocol surface.
        """
        from pyphi.system import System

        # Use cause_indices as the system nodes — paper-faithful: external
        # = substrate - cause_indices regardless of direction.
        with config.override(validate_system_states=False):
            return System(
                substrate=self.substrate,
                state=self.before_state,
                node_indices=self.cause_indices,
                cut=self.cut,
            )

    @cached_property
    def cause_tpm(self) -> Any:
        return self._underlying_system.cause_tpm

    @cached_property
    def effect_tpm(self) -> Any:
        return self._underlying_system.effect_tpm

    @cached_property
    def cm(self) -> Any:
        return self._underlying_system.cm

    @cached_property
    def proper_cause_tpm(self) -> Any:
        return self._underlying_system.proper_cause_tpm

    @cached_property
    def proper_effect_tpm(self) -> Any:
        return self._underlying_system.proper_effect_tpm

    @cached_property
    def proper_cm(self) -> Any:
        return self._underlying_system.proper_cm

    @cached_property
    def connectivity_matrix(self) -> Any:
        return self.cm

    @cached_property
    def cut_indices(self) -> tuple[int, ...]:
        return self.node_indices

    @cached_property
    def cut_node_labels(self) -> Any:
        return self.node_labels

    @cached_property
    def is_cut(self) -> bool:
        return not isinstance(self.cut, NullCut)

    @cached_property
    def size(self) -> int:
        return len(self.node_indices)

    @cached_property
    def tpm_size(self) -> int:
        return self.substrate.size

    @cached_property
    def nodes(self) -> Any:
        from pyphi.node import generate_nodes

        return generate_nodes(
            self.cause_tpm,
            self.effect_tpm,
            self.cm,
            self.state,
            self.node_indices,
            self.node_labels,
        )

    @cached_property
    def cut_mechanisms(self) -> Any:
        return list(self.cut.all_cut_mechanisms())

    @cached_property
    def null_distinction(self) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.null_distinction(self)

    @cached_property
    def null_concept(self) -> Any:
        return self.null_distinction

    def apply_cut(self, cut: SystemPartition) -> "TransitionSystem":
        return replace(self, cut=cut)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TransitionSystem):
            return NotImplemented
        return (
            self.substrate == other.substrate
            and self.before_state == other.before_state
            and self.after_state == other.after_state
            and self.cause_indices == other.cause_indices
            and self.effect_indices == other.effect_indices
            and self.direction == other.direction
            and self.cut == other.cut
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.substrate,
                self.before_state,
                self.after_state,
                self.cause_indices,
                self.effect_indices,
                self.direction,
                self.cut,
            )
        )

    def __len__(self) -> int:
        return len(self.node_indices)

    def __str__(self) -> str:
        labels = self.node_labels.coerce_to_labels(self.node_indices)
        return f"TransitionSystem({self.direction}, {', '.join(str(l) for l in labels)})"

    # Repertoire-algebra surface: delegate to pyphi.core.repertoire_algebra.

    def cause_repertoire(self, mechanism: Any, purview: Any, **kw: Any) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.cause_repertoire(self, mechanism, purview, **kw)

    def effect_repertoire(self, mechanism: Any, purview: Any, **kw: Any) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.effect_repertoire(self, mechanism, purview, **kw)

    def repertoire(
        self, direction: Direction, mechanism: Any, purview: Any, **kw: Any
    ) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.repertoire(self, direction, mechanism, purview, **kw)

    def unconstrained_cause_repertoire(self, purview: Any) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.unconstrained_cause_repertoire(self, purview)

    def unconstrained_effect_repertoire(self, purview: Any) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.unconstrained_effect_repertoire(self, purview)

    def unconstrained_repertoire(self, direction: Direction, purview: Any) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.unconstrained_repertoire(self, direction, purview)

    def partitioned_repertoire(
        self, direction: Direction, partition: Any, **kw: Any
    ) -> Any:
        # AC's partitioned_repertoire is paper-faithful by default — dispatches
        # via partitioned_repertoire_schemes registry. The PRODUCT scheme
        # delegates back to repertoire_algebra (which honors AC's call shape).
        scheme_name = config.formalism.actual_causation.partitioned_repertoire_scheme
        scheme = partitioned_repertoire_schemes[scheme_name]
        return scheme(self, direction, partition, **kw)

    def expand_cause_repertoire(
        self, repertoire_array: Any, *, new_purview: Any | None = None
    ) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.expand_cause_repertoire(self, repertoire_array, new_purview=new_purview)

    def expand_effect_repertoire(
        self, repertoire_array: Any, *, new_purview: Any | None = None
    ) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.expand_effect_repertoire(self, repertoire_array, new_purview=new_purview)

    def expand_repertoire(
        self, direction: Direction, repertoire_array: Any, new_purview: Any | None = None
    ) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.expand_repertoire(
            self, direction, repertoire_array, new_purview=new_purview
        )

    def forward_cause_repertoire(
        self, mechanism: Any, purview: Any, purview_state: Any | None = None
    ) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.forward_cause_repertoire(self, mechanism, purview, purview_state)

    def forward_effect_repertoire(
        self, mechanism: Any, purview: Any, **kw: Any
    ) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.forward_effect_repertoire(self, mechanism, purview, **kw)

    def forward_repertoire(
        self,
        direction: Direction,
        mechanism: Any,
        purview: Any,
        purview_state: Any | None = None,
        **kw: Any,
    ) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.forward_repertoire(
            self, direction, mechanism, purview, purview_state, **kw
        )

    def unconstrained_forward_cause_repertoire(
        self, mechanism: Any, purview: Any
    ) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.unconstrained_forward_cause_repertoire(self, mechanism, purview)

    def unconstrained_forward_effect_repertoire(
        self, mechanism: Any, purview: Any
    ) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.unconstrained_forward_effect_repertoire(self, mechanism, purview)

    def unconstrained_forward_repertoire(
        self, direction: Direction, mechanism: Any, purview: Any
    ) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.unconstrained_forward_repertoire(self, direction, mechanism, purview)

    def forward_cause_probability(
        self,
        mechanism: Any,
        purview: Any,
        purview_state: Any,
        mechanism_state: Any | None = None,
    ) -> float:
        from pyphi.core import repertoire_algebra as ra

        return ra.forward_cause_probability(
            self, mechanism, purview, purview_state, mechanism_state
        )

    def forward_effect_probability(
        self, mechanism: Any, purview: Any, purview_state: Any
    ) -> float:
        from pyphi.core import repertoire_algebra as ra

        return ra.forward_effect_probability(self, mechanism, purview, purview_state)

    def forward_probability(
        self,
        direction: Direction,
        mechanism: Any,
        purview: Any,
        purview_state: Any,
        **kw: Any,
    ) -> float:
        from pyphi.core import repertoire_algebra as ra

        return ra.forward_probability(
            self, direction, mechanism, purview, purview_state, **kw
        )

    def cause_info(self, mechanism: Any, purview: Any, **kw: Any) -> float:
        from pyphi.core import repertoire_algebra as ra

        return ra.cause_info(self, mechanism, purview, **kw)

    def effect_info(self, mechanism: Any, purview: Any, **kw: Any) -> float:
        from pyphi.core import repertoire_algebra as ra

        return ra.effect_info(self, mechanism, purview, **kw)

    def cause_effect_info(self, mechanism: Any, purview: Any, **kw: Any) -> float:
        from pyphi.core import repertoire_algebra as ra

        return ra.cause_effect_info(self, mechanism, purview, **kw)

    def intrinsic_information(
        self, direction: Direction, mechanism: Any, purview: Any, **kw: Any
    ) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.intrinsic_information(self, direction, mechanism, purview, **kw)

    def potential_purviews(
        self, direction: Direction, mechanism: Any, **kw: Any
    ) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.potential_purviews(self, direction, mechanism, **kw)

    def indices2nodes(self, indices: Any) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.indices2nodes(self, indices)

    # Cache surface
    def cache_info(self) -> dict[str, Any]:
        from pyphi.core import repertoire_algebra as ra

        return ra.cache_info()

    def clear_caches(self) -> None:
        from pyphi.core import repertoire_algebra as ra

        ra.clear_caches(self)

    # IIT-formalism dispatchers — category errors for AC.

    def sia(self, **kw: Any) -> Any:
        raise NotImplementedError(
            "TransitionSystem does not support IIT formalism dispatch. "
            "Use pyphi.actual.sia(transition) for actual-causation analysis."
        )

    def phi_structure(self, **kw: Any) -> Any:
        raise NotImplementedError(
            "TransitionSystem does not support IIT phi_structure. "
            "Use pyphi.actual.account(transition, direction) instead."
        )

    def ces(self, **kw: Any) -> Any:
        raise NotImplementedError(
            "TransitionSystem does not support IIT ces. "
            "Use pyphi.actual.account(transition, direction) instead."
        )

    def find_mip(
        self, direction: Direction, mechanism: Any, purview: Any, **kw: Any
    ) -> Any:
        raise NotImplementedError(
            "TransitionSystem does not expose IIT mechanism MIP search. "
            "Use Transition.find_mip(direction, mechanism, purview) instead."
        )

    def cause_mip(self, mechanism: Any, purview: Any, **kw: Any) -> Any:
        raise NotImplementedError("Use Transition.find_mip instead.")

    def effect_mip(self, mechanism: Any, purview: Any, **kw: Any) -> Any:
        raise NotImplementedError("Use Transition.find_mip instead.")

    def phi_cause_mip(self, mechanism: Any, purview: Any, **kw: Any) -> float:
        raise NotImplementedError("Use Transition.find_mip instead.")

    def phi_effect_mip(self, mechanism: Any, purview: Any, **kw: Any) -> float:
        raise NotImplementedError("Use Transition.find_mip instead.")

    def phi(self, mechanism: Any, purview: Any, **kw: Any) -> float:
        raise NotImplementedError("AC has no IIT-style phi. See pyphi.actual.")

    def find_mice(self, direction: Direction, mechanism: Any, **kw: Any) -> Any:
        raise NotImplementedError(
            "Use Transition.find_causal_link(direction, mechanism) instead."
        )

    def mic(self, mechanism: Any, **kw: Any) -> Any:
        raise NotImplementedError("Use Transition.find_actual_cause instead.")

    def mie(self, mechanism: Any, **kw: Any) -> Any:
        raise NotImplementedError("Use Transition.find_actual_effect instead.")

    def phi_max(self, mechanism: Any) -> float:
        raise NotImplementedError("AC has no IIT-style phi_max.")

    def distinction(self, mechanism: Any) -> Any:
        raise NotImplementedError("AC has no IIT distinctions.")

    def all_distinctions(self, **kw: Any) -> Any:
        raise NotImplementedError("AC has no IIT distinctions.")

    def evaluate_partition(
        self,
        direction: Direction,
        mechanism: Any,
        purview: Any,
        partition: Any,
        **kw: Any,
    ) -> Any:
        raise NotImplementedError("Use Transition.find_mip / Transition.repertoire.")

    @classmethod
    def from_substrate(
        cls,
        substrate: Substrate,
        before_state: Any,
        after_state: Any,
        cause_indices: Any,
        effect_indices: Any,
        direction: Direction,
        cut: SystemPartition | None = None,
        **kwargs: Any,
    ) -> "TransitionSystem":
        return cls(
            substrate=substrate,
            before_state=tuple(before_state),
            after_state=tuple(after_state),
            cause_indices=tuple(cause_indices),
            effect_indices=tuple(effect_indices),
            direction=direction,
            cut=cut,  # type: ignore[arg-type]
            **kwargs,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "substrate": self.substrate,
            "before_state": list(self.before_state),
            "after_state": list(self.after_state),
            "cause_indices": list(self.cause_indices),
            "effect_indices": list(self.effect_indices),
            "direction": self.direction,
            "cut": self.cut,
            "noise_background": self.noise_background,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest test/test_actual.py::test_transition_system_is_frozen test/test_actual.py::test_transition_system_satisfies_protocol test/test_actual.py::test_transition_system_cause_uses_after_state test/test_actual.py::test_transition_system_effect_uses_before_state test/test_actual.py::test_transition_system_external_indices_excludes_cause_indices test/test_actual.py::test_transition_system_apply_cut_returns_new_instance -v
```

Expected: 6 passed.

## Task 3.3: Export `TransitionSystem` from `pyphi/__init__.py`

**Files:**
- Modify: `pyphi/__init__.py`

- [ ] **Step 1: Write the failing test**

Add to `test/test_actual.py`:

```python
def test_transition_system_top_level_export():
    import pyphi

    assert hasattr(pyphi, "TransitionSystem")
    from pyphi.actual import TransitionSystem

    assert pyphi.TransitionSystem is TransitionSystem
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest test/test_actual.py::test_transition_system_top_level_export -v
```

Expected: AttributeError.

- [ ] **Step 3: Add export**

In `pyphi/__init__.py`, find the section where `actual` items are imported (or where similar top-level types like `System` are exported) and add:

```python
from pyphi.actual import TransitionSystem
```

If there's a `__all__` list, add `"TransitionSystem"` to it.

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest test/test_actual.py::test_transition_system_top_level_export -v
```

Expected: pass.

## Task 3.4: Run full acceptance gates and commit

- [ ] **Step 1: Pyright**

```bash
uv run pyright pyphi/actual.py
```

Expected: clean.

- [ ] **Step 2: Ruff**

```bash
uv run ruff check pyphi/actual.py test/test_actual.py
```

Expected: clean.

- [ ] **Step 3: Fast unit lane + new TransitionSystem tests**

```bash
uv run pytest test/test_invariants.py test/test_subsystem_surface.py test/test_formalism_pickle.py test/test_parallel.py test/test_scheduler.py test/test_sampling.py test/test_install_snapshot.py test/test_config.py test/test_config_layers.py test/test_actual.py -q
```

Expected: green. The 826 lines of original test_actual.py are still skipped via `pytestmark`; only the new tests above the skip run.

- [ ] **Step 4: Hypothesis fast lane**

Expected: 21 properties pass.

- [ ] **Step 5: Golden 17/17**

Expected: 17/17 unchanged.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
Add TransitionSystem and AC registries

Adds pyphi.actual.TransitionSystem — a frozen dataclass parametric in
Direction, satisfying SystemPublicInterface. Holds substrate +
before_state + after_state + cause_indices + effect_indices + direction +
optional cut. Cause-side state == after_state, effect-side state ==
before_state (the asymmetric two-state requirement of the 2019
Albantakis et al. AC formalism). external_indices = substrate − cause_indices,
the paper-faithful background-conditioning rule.

IIT-formalism dispatchers (sia, phi_structure, ces, find_mip, etc.) raise
NotImplementedError pointing at the appropriate pyphi.actual free
functions — calling them on a TransitionSystem is a category error.

Adds three new registries with paper-faithful default-only registrations:
  partitioned_repertoire_schemes: PRODUCT
  background_strategies: UNIFORM
  alpha_aggregations: SUBTRACTIVE

Existing Transition class untouched in this commit; it still uses the
legacy mutate-after-construct pattern. The next commit rewrites it.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

**Acceptance:** All gates green; commit lands.

---

# Phase 4: Rewrite `Transition`; remove skips; paper-fixture tests

**Why fourth:** With TransitionSystem in place, Transition can compose two of them. The 826 lines of dark tests come back online.

**Files affected:**
- Modify: `pyphi/actual.py` — rewrite `Transition` class
- Modify: `test/test_actual.py` — remove `pytestmark = pytest.mark.skip(...)`; add paper-fixture tests
- Modify: `test/conftest.py:381-384` — remove the `pytest.skip(...)` in the `transition` fixture

## Task 4.1: Rewrite `Transition` as frozen wrapper

**Files:**
- Modify: `pyphi/actual.py:53-538` (the `Transition` class definition)

- [ ] **Step 1: Read the current `Transition` class and methods**

Already surveyed in spec. Note all instance methods that must be preserved:

```
__init__, __repr__, __str__, __eq__, __hash__, __len__, __bool__, node_labels (property), to_json, apply_cut, cause_repertoire, effect_repertoire, unconstrained_cause_repertoire, unconstrained_effect_repertoire, repertoire, state_probability, probability, unconstrained_probability, purview_state, mechanism_state, mechanism_indices, purview_indices, _ratio, cause_ratio, effect_ratio, partitioned_repertoire, partitioned_probability, find_mip, potential_purviews, find_causal_link, find_actual_cause, find_actual_effect, find_mice
```

- [ ] **Step 2: Replace the entire `Transition` class definition**

Replace lines 53-538 (or whatever range covers `class Transition:` start through last method end) with:

```python
@dataclass(frozen=True, eq=False)
class Transition:
    """A state transition over a substrate, holding two TransitionSystem views.

    Implements the 2019 Albantakis et al. actual-causation framework. The
    cause-side and effect-side analyses live in :class:`TransitionSystem`
    instances accessed via :attr:`cause_system` and :attr:`effect_system`,
    keyed by Direction in :attr:`system`.
    """

    substrate: Substrate
    before_state: tuple[int, ...]
    after_state: tuple[int, ...]
    cause_indices: tuple[int, ...]
    effect_indices: tuple[int, ...]
    cut: SystemPartition = field(default=None)  # type: ignore[assignment]
    noise_background: bool = False

    def __post_init__(self) -> None:
        coerce = self.substrate.node_labels.coerce_to_indices
        object.__setattr__(self, "cause_indices", coerce(self.cause_indices))
        object.__setattr__(self, "effect_indices", coerce(self.effect_indices))
        if self.cut is None:
            object.__setattr__(
                self, "cut", NullCut(self.node_indices, self.substrate.node_labels)
            )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Transition):
            return NotImplemented
        return (
            self.substrate == other.substrate
            and self.before_state == other.before_state
            and self.after_state == other.after_state
            and self.cause_indices == other.cause_indices
            and self.effect_indices == other.effect_indices
            and self.cut == other.cut
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.substrate,
                self.before_state,
                self.after_state,
                self.cause_indices,
                self.effect_indices,
                self.cut,
            )
        )

    def __len__(self) -> int:
        return len(self.node_indices)

    def __bool__(self) -> bool:
        return len(self) > 0

    def __repr__(self) -> str:
        return fmt.fmt_transition(self)

    def __str__(self) -> str:
        return repr(self)

    @cached_property
    def node_indices(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.cause_indices) | set(self.effect_indices)))

    @property
    def node_labels(self) -> Any:
        return self.substrate.node_labels

    @cached_property
    def cause_system(self) -> TransitionSystem:
        return TransitionSystem(
            substrate=self.substrate,
            before_state=self.before_state,
            after_state=self.after_state,
            cause_indices=self.cause_indices,
            effect_indices=self.effect_indices,
            direction=Direction.CAUSE,
            cut=self.cut,
            noise_background=self.noise_background,
        )

    @cached_property
    def effect_system(self) -> TransitionSystem:
        return TransitionSystem(
            substrate=self.substrate,
            before_state=self.before_state,
            after_state=self.after_state,
            cause_indices=self.cause_indices,
            effect_indices=self.effect_indices,
            direction=Direction.EFFECT,
            cut=self.cut,
            noise_background=self.noise_background,
        )

    @cached_property
    def system(self) -> Mapping[Direction, TransitionSystem]:
        from types import MappingProxyType

        return MappingProxyType(
            {Direction.CAUSE: self.cause_system, Direction.EFFECT: self.effect_system}
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "substrate": self.substrate,
            "before_state": list(self.before_state),
            "after_state": list(self.after_state),
            "cause_indices": list(self.cause_indices),
            "effect_indices": list(self.effect_indices),
            "cut": self.cut,
        }

    def apply_cut(self, cut: SystemPartition) -> "Transition":
        return replace(self, cut=cut)

    def cause_repertoire(self, mechanism: Any, purview: Any) -> Any:
        return self.repertoire(Direction.CAUSE, mechanism, purview)

    def effect_repertoire(self, mechanism: Any, purview: Any) -> Any:
        return self.repertoire(Direction.EFFECT, mechanism, purview)

    def unconstrained_cause_repertoire(self, purview: Any) -> Any:
        return self.cause_repertoire((), purview)

    def unconstrained_effect_repertoire(self, purview: Any) -> Any:
        return self.effect_repertoire((), purview)

    def repertoire(self, direction: Direction, mechanism: Any, purview: Any) -> Any:
        system = self.system[direction]
        node_labels = system.node_labels
        if not set(purview).issubset(self.purview_indices(direction)):
            raise ValueError(
                f"{fmt.fmt_mechanism(purview, node_labels)} is not a "
                f"{direction} purview in {self}"
            )
        if not set(mechanism).issubset(self.mechanism_indices(direction)):
            raise ValueError(
                f"{fmt.fmt_mechanism(mechanism, node_labels)} is not a "
                f"{direction} mechanism in {self}"
            )
        return system.repertoire(direction, mechanism, purview)

    def state_probability(
        self, direction: Direction, repertoire: Any, purview: Any
    ) -> float:
        purview_state = self.purview_state(direction)
        system = self.system[direction]
        if repertoire.ndim == system.substrate.size:
            node_indices = system.substrate.node_indices
        else:
            node_indices = system.node_indices
        index = tuple(
            purview_state[node] if node in purview else 0 for node in node_indices
        )
        return repertoire[index]

    def probability(self, direction: Direction, mechanism: Any, purview: Any) -> float:
        repertoire = self.repertoire(direction, mechanism, purview)
        return self.state_probability(direction, repertoire, purview)

    def unconstrained_probability(self, direction: Direction, purview: Any) -> float:
        return self.probability(direction, (), purview)

    def purview_state(self, direction: Direction) -> tuple[int, ...]:
        return {
            Direction.CAUSE: self.before_state,
            Direction.EFFECT: self.after_state,
        }[direction]

    def mechanism_state(self, direction: Direction) -> tuple[int, ...]:
        return self.system[direction].state

    def mechanism_indices(self, direction: Direction) -> tuple[int, ...]:
        return {
            Direction.CAUSE: self.effect_indices,
            Direction.EFFECT: self.cause_indices,
        }[direction]

    def purview_indices(self, direction: Direction) -> tuple[int, ...]:
        return {
            Direction.CAUSE: self.cause_indices,
            Direction.EFFECT: self.effect_indices,
        }[direction]

    def _ratio(self, direction: Direction, mechanism: Any, purview: Any) -> float:
        return probability_distance(
            self.probability(direction, mechanism, purview),
            self.unconstrained_probability(direction, purview),
            measure="PMI",
        )

    def cause_ratio(self, mechanism: Any, purview: Any) -> float:
        return self._ratio(Direction.CAUSE, mechanism, purview)

    def effect_ratio(self, mechanism: Any, purview: Any) -> float:
        return self._ratio(Direction.EFFECT, mechanism, purview)

    def partitioned_repertoire(self, direction: Direction, partition: Any) -> Any:
        # Paper-faithful: dispatches via partitioned_repertoire_schemes registry
        # (default PRODUCT implements 2019 Eq 8). Independent of IIT formalism config.
        return self.system[direction].partitioned_repertoire(direction, partition)

    def partitioned_probability(self, direction: Direction, partition: Any) -> float:
        repertoire = self.partitioned_repertoire(direction, partition)
        return self.state_probability(direction, repertoire, partition.purview)

    def find_mip(
        self,
        direction: Direction,
        mechanism: Any,
        purview: Any,
        allow_neg: bool = False,
    ) -> Any:
        if not purview:
            return _null_ac_ria(
                self.mechanism_state(direction), direction, mechanism, purview
            )
        alpha_min = float("inf")
        probability = self.probability(direction, mechanism, purview)
        acria = None
        for partition in mip_partitions(mechanism, purview, self.node_labels):
            partitioned_probability = self.partitioned_probability(direction, partition)
            alpha = probability_distance(probability, partitioned_probability)
            if utils.eq(alpha, 0) or (alpha < 0 and not allow_neg):
                return _null_ac_ria(
                    self.mechanism_state(direction),
                    direction,
                    mechanism,
                    purview,
                    partition,
                )
            if (abs(alpha_min) - abs(alpha)) > 10 ** (-config.numerics.precision):
                alpha_min = alpha
                acria = AcRepertoireIrreducibilityAnalysis(
                    state=self.mechanism_state(direction),
                    direction=direction,
                    mechanism=mechanism,
                    purview=purview,
                    partition=partition,
                    probability=probability,
                    partitioned_probability=partitioned_probability,
                    node_labels=self.node_labels,
                    alpha=alpha_min,
                )
        return acria

    def potential_purviews(
        self,
        direction: Direction,
        mechanism: Any,
        purviews: Any | None = None,
    ) -> Any:
        system = self.system[direction]
        return [
            purview
            for purview in system.potential_purviews(direction, mechanism, purviews)
            if set(purview).issubset(self.purview_indices(direction))
        ]

    def find_causal_link(
        self,
        direction: Direction,
        mechanism: Any,
        purviews: Any | None = None,
        allow_neg: bool = False,
    ) -> Any:
        purviews = self.potential_purviews(direction, mechanism, purviews)
        if not purviews:
            max_ria = _null_ac_ria(
                self.mechanism_state(direction), direction, mechanism, None
            )
            return CausalLink(max_ria)
        all_ria = [
            self.find_mip(direction, mechanism, purview, allow_neg=allow_neg)
            for purview in purviews
        ]
        valid_ria = [ria for ria in all_ria if ria is not None]
        if not valid_ria:
            return []
        max_ria = max(valid_ria)
        purviews = [ria.purview for ria in valid_ria if ria.alpha == max_ria.alpha]

        def is_not_superset(purview: Any) -> bool:
            return all(
                (not set(purview).issuperset(set(other_purview)))
                or (set(purview) == set(other_purview))
                for other_purview in purviews
            )

        extended_purview = filter(is_not_superset, purviews)
        return CausalLink(max_ria, tuple(extended_purview))

    def find_actual_cause(self, mechanism: Any, purviews: Any | None = None) -> Any:
        return self.find_causal_link(Direction.CAUSE, mechanism, purviews)

    def find_actual_effect(self, mechanism: Any, purviews: Any | None = None) -> Any:
        return self.find_causal_link(Direction.EFFECT, mechanism, purviews)

    def find_mice(self, *args: Any, **kwargs: Any) -> Any:
        return self.find_causal_link(*args, **kwargs)
```

Add the necessary imports at the top:

```python
from collections.abc import Mapping
from dataclasses import replace
from functools import cached_property
```

(`@dataclass`, `field`, `Direction`, etc. should already be imported from earlier additions; verify with ruff.)

- [ ] **Step 3: Verify import works**

```bash
uv run python -c "from pyphi.actual import Transition; print(Transition)"
```

Expected: prints `<class 'pyphi.actual.Transition'>` (no errors).

## Task 4.2: Remove module-level skip in `test/test_actual.py` and `test/conftest.py` transition skip

**Files:**
- Modify: `test/test_actual.py:16-19` (remove `pytestmark = pytest.mark.skip(...)`)
- Modify: `test/conftest.py:381-384` (remove the `pytest.skip(...)` inside `transition` fixture)

- [ ] **Step 1: Remove the module-level skip**

In `test/test_actual.py`, delete lines 16-19:

```python
pytestmark = pytest.mark.skip(
    reason="actual.Transition pending refactor for frozen System "
    "(uses _external_indices override + state mutation)"
)
```

- [ ] **Step 2: Remove the conftest fixture skip**

In `test/conftest.py:381-384`, delete:

```python
    pytest.skip(
        "actual.Transition pending refactor for frozen System "
        "(uses _external_indices override + state mutation)"
    )
```

The `transition` fixture body should now run from the `tpm = np.array(...)` line onwards normally.

- [ ] **Step 3: Run the full `test_actual.py` suite**

```bash
uv run pytest test/test_actual.py -v
```

Expected: 826 lines of tests run; all green. If any fail, address them — most likely culprits are subtle differences in TPM marginalization or state validation.

If failures cluster around `validate.state_reachable`, the `_underlying_system` construction in `TransitionSystem` may need its `validate_system_states=False` override expanded to cover more validation paths.

## Task 4.3: Add paper-fixture acceptance tests

**Files:**
- Modify: `test/test_actual.py` (add a new section at the end with paper-fixture tests)

- [ ] **Step 1: Add fixtures and tests for paper-faithful α values**

Append to `test/test_actual.py`:

```python
# Paper-fixture acceptance tests: pin α values from
# Albantakis, Marshall, Hoel, Tononi 2019 ("What Caused What?", Entropy 21:459).
# These tests anchor correctness on the published 2019 formalism rather than
# on the legacy implementation's behavior.


@pytest.fixture
def or_and_substrate():
    """Two-element substrate from 2019 Figs 1–6: OR gate with self-loop, AND gate."""
    # OR(B, A) = B OR A; AND(B, A) = B AND A; both update simultaneously
    # State-by-node TPM for (OR_t, AND_t) given (OR_{t-1}, AND_{t-1}).
    tpm = np.array(
        [
            [0, 0],  # state (0,0): OR=0, AND=0
            [1, 0],  # state (1,0): OR=1, AND=0
            [1, 0],  # state (0,1): OR=1, AND=0
            [1, 1],  # state (1,1): OR=1, AND=1
        ]
    )
    cm = np.array([[1, 1], [1, 1]])
    return Substrate(tpm, cm, node_labels=("OR", "AND"))


def test_paper_fig5_6_or_and_first_order_alpha(or_and_substrate):
    """2019 Fig 5/6: first-order links {OR_{t-1}=1}↔{OR_t=1} have α≈0.415 bits."""
    transition = actual.Transition(
        or_and_substrate,
        before_state=(1, 0),  # (OR=1, AND=0)
        after_state=(1, 0),
        cause_indices=(0, 1),
        effect_indices=(0, 1),
    )
    cause = transition.find_actual_cause((0,))  # mechanism is OR_t = 1
    assert cause.alpha == pytest.approx(0.415, abs=1e-3)


def test_paper_fig6_or_and_second_order_alpha(or_and_substrate):
    """2019 Fig 6: second-order cause link {(OR,AND)_{t-1}=10}←{(OR,AND)_t=10} has α≈0.170."""
    transition = actual.Transition(
        or_and_substrate,
        before_state=(1, 0),
        after_state=(1, 0),
        cause_indices=(0, 1),
        effect_indices=(0, 1),
    )
    cause = transition.find_actual_cause((0, 1))  # mechanism is {(OR, AND) = 10}
    assert cause.alpha == pytest.approx(0.170, abs=1e-3)


@pytest.fixture
def conjunction_substrate():
    """3-node substrate from 2019 Fig 7B: A, B independent inputs to AND-gate D."""
    # A, B feed an AND gate D; A and B are independent.
    # State indexing: (A, B, D) where D = A AND B at next step.
    # 8 states; A, B fixed (independent), D = AB.
    # Use TPM where A_t = A_{t-1} (copy), B_t = B_{t-1} (copy), D_t = A_{t-1} AND B_{t-1}.
    tpm = np.array(
        [
            [0, 0, 0],  # (0,0,0) → (0,0,0)
            [0, 0, 0],  # (1,0,0) → (0,0,0); A copies wait — fix
            # ...
        ]
    )
    # Simpler: use the example from the paper Fig 7B directly.
    # AB → D where D = A AND B. A and B are inputs (no recurrent dynamics for them).
    # Use a 3-node feed-forward TPM.
    raise pytest.skip("Conjunction fixture: TODO build paper-faithful 3-node TPM")


def test_paper_fig7b_conjunction_alpha(conjunction_substrate):
    """2019 Fig 7B: {AB=11}←{D=1} has α_c^max = 2.0 bits."""
    transition = actual.Transition(
        conjunction_substrate,
        before_state=(1, 1, 0),
        after_state=(1, 1, 1),
        cause_indices=(0, 1),
        effect_indices=(2,),
    )
    cause = transition.find_actual_cause((2,))
    assert cause.alpha == pytest.approx(2.0, abs=1e-3)
```

**Important note**: Building each paper-faithful fixture (OR/AND, conjunction, bi-conditional, majority gate, voting) requires a careful TPM construction matching the paper's gate logic. The TPM and CM for each example need to be derived from the gate equations in the paper. This is non-trivial bookkeeping; the executor should consult Figs 5, 7A, 7B, 7C, 8A, 11, 12 directly when constructing each fixture.

For Phase 4's commit, **the minimum bar is**: at least the OR/AND first-order (`test_paper_fig5_6_or_and_first_order_alpha`) and second-order (`test_paper_fig6_or_and_second_order_alpha`) tests pass against actual computed values. Additional paper-fixture tests (conjunction, bi-conditional, majority, voting) should be added but may be marked `@pytest.mark.slow` or `@pytest.mark.todo_paper_fixture` initially and developed iteratively.

- [ ] **Step 2: Verify the OR/AND paper-fixture tests pass**

```bash
uv run pytest test/test_actual.py::test_paper_fig5_6_or_and_first_order_alpha test/test_actual.py::test_paper_fig6_or_and_second_order_alpha -v
```

Expected: pass. If they fail, the asymmetric `external_indices = substrate − cause_indices` rule may need adjustment in `TransitionSystem._underlying_system` to match the paper's causal-marginalization step.

- [ ] **Step 3: Add stub paper-fixture tests for the remaining figures**

Add to `test/test_actual.py`:

```python
@pytest.mark.skip(reason="paper-fixture TPM construction TODO; tracked for follow-up")
def test_paper_fig7c_biconditional_alpha():
    pass


@pytest.mark.skip(reason="paper-fixture TPM construction TODO; tracked for follow-up")
def test_paper_fig8a_majority_alpha():
    pass


@pytest.mark.skip(reason="paper-fixture TPM construction TODO; tracked for follow-up")
def test_paper_fig11_three_candidate_alpha():
    pass


@pytest.mark.skip(reason="paper-fixture TPM construction TODO; tracked for follow-up")
def test_paper_fig12_probabilistic_alpha():
    pass
```

These are placeholders so the test surface advertises future work without holding up Phase 4.

## Task 4.4: Run full acceptance gates and commit

- [ ] **Step 1: Pyright + ruff**

```bash
uv run pyright pyphi/actual.py
uv run ruff check pyphi/actual.py test/test_actual.py
```

Expected: clean.

- [ ] **Step 2: Full test_actual.py suite**

```bash
uv run pytest test/test_actual.py -v
```

Expected: 826 lines of legacy tests + new paper-fixture tests, all passing (or skipped per `pytest.mark.skip` for the placeholder paper-fixtures).

- [ ] **Step 3: Fast unit lane + hypothesis fast lane + golden 17/17**

Per usual gates.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
Rewrite Transition as frozen wrapper; resurrect test_actual.py

Transition becomes a @dataclass(frozen=True, eq=False) wrapping two
TransitionSystem instances (one per Direction) accessed via
cached_property. apply_cut returns a new Transition via
dataclasses.replace. Explicit __eq__ / __hash__ preserve the legacy
comparison set (substrate, before_state, after_state, cause_indices,
effect_indices, cut — `noise_background` excluded from equality).

The legacy mutate-after-construction block (cause_system.state =
after_state + per-node mutation + _external_indices override) is
deleted. The asymmetric two-state requirement is satisfied by
TransitionSystem's direction-parametric state.

Transition.partitioned_repertoire now dispatches via
partitioned_repertoire_schemes registry (paper-faithful PRODUCT default)
rather than branching on config.formalism.iit.repertoire_measure. AC
behavior is now independent of IIT formalism configuration.

Removed module-level skip in test/test_actual.py and the conftest.py
transition fixture skip. The 826 lines of dark tests come back online.

Adds initial paper-fixture acceptance tests for the OR/AND example
(Figs 5/6, first-order and second-order α values from 2019 Albantakis
et al.). Placeholder tests for Figs 7B/7C/8A/11/12 are added with
pytest.mark.skip pending TPM construction.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

**Acceptance:** All gates green; `test_actual.py` runs ≥826 tests (legacy) plus the 2 paper-fixture tests for OR/AND.

---

# Phase 5: ROADMAP + changelog

**Why last:** Documentation update closes the project.

## Task 5.1: Update ROADMAP

**Files:**
- Modify: `ROADMAP.md`

- [ ] **Step 1: Find and read the current P14 entry**

```bash
grep -n "P14\|macro.py.*actual.py" ROADMAP.md | head
```

- [ ] **Step 2: Update the "Updated 2.0 ordering" schedule**

Mark P14 (actual portion) as complete. Update wording to reflect actual-only scope; drop any "third PhiFormalism implementation" language.

Insert a new project entry for the deferred macro / Marshall-2024-intrinsic-units work, citing:

> Marshall W, Findlay G, Albantakis L, Tononi G (2024). Intrinsic Units: Identifying a system's causal grain. (See `papers/2024__marshall-et-al__intrinsic-units.pdf`.)

The new project's scope: paper-faithful macro framework with hierarchical meso constituents, sliding-window state mappings $g_J$, explicit background apportionment $W^J$, and intrinsic-unit search via $\varphi_s$ optimization. Replaces legacy `pyphi/macro.py` outright.

- [ ] **Step 3: Verify ROADMAP renders**

```bash
git diff ROADMAP.md | head -100
```

Spot-check formatting.

## Task 5.2: Add changelog fragment

**Files:**
- Create: `changelog.d/p14-actual-resurrection.fix.md`

- [ ] **Step 1: Read existing changelog conventions**

```bash
ls changelog.d/ | head -20
cat changelog.d/README.md 2>/dev/null | head -30
```

- [ ] **Step 2: Create the changelog fragment**

```bash
cat > changelog.d/p14-actual-resurrection.fix.md <<'EOF'
**Resurrected ``pyphi.actual.Transition`` against the frozen ``System``
value type, with full formalism config audit.**

``pyphi.actual.TransitionSystem`` is a new frozen dataclass parametric in
``Direction`` that satisfies ``pyphi.protocols.SystemPublicInterface``.
``Transition`` becomes a frozen wrapper holding two TransitionSystem
instances (one per direction); IIT-formalism dispatchers raise
``NotImplementedError`` when called on a TransitionSystem (category
errors). The 826 lines of currently-skipped ``test/test_actual.py`` come
back online.

Configuration restructured: ``config.formalism`` now holds two nested
frozen dataclasses, ``IITConfig`` and ``ActualCausationConfig``. Field
naming aligned uniformly:

- ``repertoire_distance`` family → ``repertoire_measure`` family
  (matches the ``measures`` registry name)
- ``ces_distance`` → ``ces_measure``
- ``partition_type`` → ``mechanism_partition_scheme``
- ``system_partition_type`` → ``system_partition_scheme``
- ``assume_cuts_cannot_create_new_concepts`` →
  ``assume_partitions_cannot_create_new_concepts``
- ``actual_causation_measure`` → ``actual_causation.measure``

New AC-specific knobs (under ``formalism.actual_causation``):
``mechanism_partition_scheme``, ``partitioned_repertoire_scheme``,
``background_strategy``, ``alpha_aggregation`` — paper-faithful defaults
match the 2019 Albantakis et al. formalism. AC's ``partitioned_repertoire``
no longer branches on ``config.formalism.iit.repertoire_measure``;
behavior is now independent of IIT-formalism configuration.

Removed the orphaned concept-style cuts machinery
(``ConceptStyleSystem``, ``concept_cuts``, ``directional_sia``,
``SystemIrreducibilityAnalysisConceptStyle``, ``sia_concept_style``) and
the ``system_cuts`` config field. The variant has unclear provenance,
its integration tests rotted in the IIT 4.0 transition, and no current
workflows depend on it.

``pyphi/macro.py`` resurrection is deferred to a separate paper-faithful
project tracking Marshall et al. 2024 (intrinsic units).
EOF
```

- [ ] **Step 3: Verify the fragment**

```bash
cat changelog.d/p14-actual-resurrection.fix.md
```

## Task 5.3: Run final acceptance gates and commit

- [ ] **Step 1: Full gate run**

Same as previous phases (pyright + ruff + fast unit + hypothesis + golden).

- [ ] **Step 2: Commit**

```bash
git add ROADMAP.md changelog.d/p14-actual-resurrection.fix.md
git commit -m "$(cat <<'EOF'
ROADMAP + changelog: actual.Transition resurrection + formalism config audit

Marks the actual-causation portion of the macro/actual resurrection
project complete. Inserts a new project entry for the deferred macro /
Marshall-2024 intrinsic-units work, which will replace legacy macro.py
outright with a paper-faithful framework.

Changelog fragment summarizes: TransitionSystem + frozen Transition +
paper-fixture tests + nested formalism.iit / formalism.actual_causation
namespaces + full rename map + AC's new config knobs + concept-style
cuts deletion.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3: Surface to user with finishing-a-development-branch options**

Per saved memory `feedback_ask_before_push.md`: do not push without explicit consent.

Surface the four standard finishing-a-development-branch options:

```
Implementation complete. What would you like to do?

1. Merge back to 2.0 locally (fast-forward 2.0 to feature/p14-actual-resurrection
   tip, delete the feature branch — same pattern as P10/P11)
2. Push and create a Pull Request
3. Keep the branch as-is (I'll handle it later)
4. Discard this work

Which option?
```

**Acceptance:** Project complete; user chooses how to land it.

---

## Self-Review

**1. Spec coverage check:**

- ✓ Theoretical foundation + 2019 paper as oracle: Phase 4 paper-fixture tests
- ✓ Macro deferred: Phase 5 ROADMAP entry
- ✓ System Protocol architecture: Phase 3 TransitionSystem + Protocol satisfaction test
- ✓ Configuration audit + nested namespaces: Phase 2 (entire phase)
- ✓ AC's 4 new config knobs: Phase 3 registries + Phase 2 ActualCausationConfig dataclass
- ✓ Concept-style cuts deletion: Phase 1 (entire phase)
- ✓ TransitionSystem class design: Phase 3
- ✓ Transition wrapper redesign: Phase 4
- ✓ Q2 decoupling: Phase 4 (Transition.partitioned_repertoire dispatches via registry)
- ✓ Paper-fixture acceptance tests: Phase 4 Task 4.3
- ✓ ROADMAP + changelog: Phase 5

**2. Placeholder scan:**

- The conjunction fixture in Task 4.3 has a `raise pytest.skip("Conjunction fixture: TODO build paper-faithful 3-node TPM")` — flagged as future work but the executor knows the minimum bar (OR/AND fixtures are required; conjunction is optional).
- Several tests in Task 4.3 are marked `@pytest.mark.skip(reason="paper-fixture TPM construction TODO")` — this is intentional placeholder structure, not a plan failure. Documents expected future work.

**3. Type consistency:**

- `TransitionSystem` field names match between Phase 3 dataclass definition and Phase 4 Transition's calls to `TransitionSystem(...)`.
- Registry names (`partitioned_repertoire_schemes`, `background_strategies`, `alpha_aggregations`) consistent across Phase 3 and Phase 4.
- Config key names match between Phase 2 dataclass definitions and Phase 3/4 reads.
- `Direction.CAUSE` / `Direction.EFFECT` used consistently.

---

## Plan summary

| Phase | Goal | Key deliverable |
|---|---|---|
| 1 | Delete concept-style cuts | ~165 lines deleted; iit3.sia simplified |
| 2 | Config audit + rename | Nested formalism.iit + formalism.actual_causation; ~30 call sites updated |
| 3 | Add TransitionSystem + AC registries | New frozen class + 3 registries with paper-faithful defaults |
| 4 | Rewrite Transition; lift skips; paper-fixtures | 826 dark tests resurrected; ≥2 paper-fixture acceptance tests |
| 5 | ROADMAP + changelog | Project closed; macro deferred to next project |

Five commits total.
