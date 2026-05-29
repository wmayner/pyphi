# AC Formalism Object — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Actual Causation a registered formalism object (`AC2019Formalism`) that owns evaluation dispatch and measure/scheme resolution, mirroring the IIT formalism objects, in a self-contained `pyphi/formalism/actual_causation/` package.

**Architecture:** Add an `ActualCausationFormalism` Protocol + `ACTUAL_CAUSATION_FORMALISM_REGISTRY` beside the IIT equivalents in `pyphi/formalism/base.py`. Move the AC compute algorithms out of `pyphi/actual.py` into a new `pyphi/formalism/actual_causation/` package. The `AC2019Formalism` object resolves the configured `alpha_measure` (checked against `compatible_measures`) plus the three AC schemes, then delegates to the moved algorithm functions — exactly as `IIT4_2023Formalism.evaluate_system` resolves measures then calls `_sia`. `actual.py` keeps `Transition`/`TransitionSystem` (data) and exposes its public functions as thin dispatchers through the registry. Pure refactor: zero numeric change to any AC result.

**Tech Stack:** Python 3.12+, frozen dataclasses, `pyphi.registry.Registry`, `runtime_checkable` Protocols, pytest. Run everything with `uv run`.

**Reference spec:** `docs/superpowers/specs/2026-05-28-ac-formalism-object-design.md`.

**Settled design (do not relitigate):**
- Spell out infrastructure names: Protocol `ActualCausationFormalism`, registry class `ActualCausationFormalismRegistry`, constant `ACTUAL_CAUSATION_FORMALISM_REGISTRY`. Concrete class `AC2019Formalism`. Version key `"AC_2019"`.
- Validation policy mirrors IIT: `version`/`alpha_measure` are **not** validated in `__post_init__` (deferred to resolve-time `check_measure_compatible` + registry lookup). The three scheme fields keep their existing frozenset checks.
- The three AC registries (`partitioned_repertoire_schemes`, `background_strategies`, `alpha_aggregations`), `probability_distance`, and `account_distance` move into the package. No back-compat re-export shim; in-repo references are updated and the move is noted in the changelog.
- Out of scope: IIT-4.0-style AC measures (a future second formalism; only the registry seam is built here).

---

## File Structure

**New files:**
- `pyphi/formalism/actual_causation/__init__.py` — exports `AC2019Formalism`; registers it in `ACTUAL_CAUSATION_FORMALISM_REGISTRY`; re-exports the algorithm functions + registries used by `actual.py`'s dispatchers.
- `pyphi/formalism/actual_causation/compute.py` — the moved algorithm functions (`_account`, `_directed_account`, `_find_mip`, `_find_causal_link`, `_sia`, `_evaluate_partition`, `_get_partitions`), the AC compute utilities (`probability_distance`, `account_distance`), and the three AC registries with their registered scheme functions.
- `pyphi/formalism/actual_causation/formalism.py` — the `AC2019Formalism` frozen dataclass and the `_resolve_ac_measures` helper.
- `test/test_ac_formalism.py` — new tests for the formalism object, registry, and compatibility gate.

**Modified files:**
- `pyphi/formalism/base.py` — add `ActualCausationFormalism` Protocol, `ActualCausationFormalismRegistry`, `ACTUAL_CAUSATION_FORMALISM_REGISTRY`.
- `pyphi/conf/formalism.py` — add `version: str = "AC_2019"` to `ActualCausationConfig`.
- `pyphi/formalism/__init__.py` — import the AC package so registration runs at formalism-package import.
- `pyphi/actual.py` — remove the moved functions; `account`/`directed_account`/`sia` and `Transition.find_mip`/`find_causal_link`/`find_actual_cause`/`find_actual_effect` become thin dispatchers; keep `Transition`/`TransitionSystem`.
- `test/test_actual.py` — update the three registry references (`actual.partitioned_repertoire_schemes` → new path).
- `changelog.d/ac-formalism-object.feature.md` — new fragment.
- `ROADMAP.md` — mark the AC formalism / measure-config item.

**Circular-import discipline:** `compute.py` operates on `transition` objects passed in; it imports result types from `pyphi.models` (not from `pyphi.actual`) and uses a `TYPE_CHECKING`-only import for the `Transition`/`TransitionSystem` types. `actual.py`'s dispatchers import `ACTUAL_CAUSATION_FORMALISM_REGISTRY` **lazily inside the function bodies** (as `_resolve_ac_kwargs` already imports `resolve_actual_causation_measure` lazily), exactly mirroring how `pyphi/formalism/queries.py` looks up `FORMALISM_REGISTRY`.

---

## Task 0: Worktree

**Files:** none (environment setup).

- [ ] **Step 1:** Create an isolated worktree off `2.0` HEAD via `superpowers:using-git-worktrees`. Use the project's sibling-directory convention: `git worktree add /Users/will/projects/pyphi-ac-formalism -b feature/ac-formalism-object 2.0`.
- [ ] **Step 2:** Set up deps in the worktree: `uv sync --all-extras --all-groups`.
- [ ] **Step 3:** Confirm baseline green: `uv run pytest test/test_actual.py test/test_golden_regression.py -q --no-header`. Expected: all pass.

---

## Task 1: `ActualCausationFormalism` Protocol + registry

**Files:**
- Modify: `pyphi/formalism/base.py`
- Test: `test/test_ac_formalism.py` (create)

- [ ] **Step 1: Write the failing test.** Create `test/test_ac_formalism.py`:

```python
"""Tests for the Actual Causation formalism object and registry."""

from __future__ import annotations

import pytest


def test_ac_formalism_registry_exists_and_is_typed():
    from pyphi.formalism.base import (
        ACTUAL_CAUSATION_FORMALISM_REGISTRY,
        ActualCausationFormalismRegistry,
    )

    assert isinstance(
        ACTUAL_CAUSATION_FORMALISM_REGISTRY, ActualCausationFormalismRegistry
    )


def test_ac_formalism_registry_rejects_wrong_shape():
    from pyphi.formalism.base import ActualCausationFormalismRegistry

    registry = ActualCausationFormalismRegistry()
    with pytest.raises(TypeError):
        registry.register("BOGUS", object())
```

- [ ] **Step 2: Run it to verify it fails.**

Run: `uv run pytest test/test_ac_formalism.py -q --no-header`
Expected: FAIL with `ImportError` (`ACTUAL_CAUSATION_FORMALISM_REGISTRY` not defined).

- [ ] **Step 3: Add the Protocol + registry to `pyphi/formalism/base.py`.** Append after the `FORMALISM_REGISTRY` definition (after line 191). Add the four new names to `__all__` (the list at lines 28-37).

```python
@runtime_checkable
class ActualCausationFormalism(Protocol):
    """The minimum shape every actual-causation formalism satisfies.

    The AC analog of :class:`PhiFormalism`. AC operates on transitions
    (before/after state pairs) rather than systems-in-a-state, so its
    evaluation surface differs: ``evaluate_account`` /
    ``evaluate_system`` / ``evaluate_mechanism`` / ``evaluate_causal_link``.

    Concrete formalisms also declare:

    - ``name``: stable identifier held in
      ``config.formalism.actual_causation.version`` and registered in
      :data:`ACTUAL_CAUSATION_FORMALISM_REGISTRY`.
    - ``compatible_measures``: frozenset of α-measure names accepted.
    - ``config``: the :class:`FormalismConfig` snapshot operated against.

    Signatures are intentionally permissive (``Any``), matching
    :class:`PhiFormalism`.
    """

    name: ClassVar[str]
    compatible_measures: ClassVar[frozenset[str]]

    @property
    def config(self) -> FormalismConfig: ...

    def evaluate_account(
        self, transition: Any, direction: Any = ..., **kwargs: Any
    ) -> Any: ...

    def evaluate_system(
        self, transition: Any, direction: Any = ..., **kwargs: Any
    ) -> Any: ...

    def evaluate_mechanism(
        self, transition: Any, direction: Any, mechanism: Any, purview: Any, **kwargs: Any
    ) -> Any: ...

    def evaluate_causal_link(
        self, transition: Any, direction: Any, mechanism: Any, **kwargs: Any
    ) -> Any: ...


class ActualCausationFormalismRegistry(Registry[ActualCausationFormalism]):
    """Storage for actual-causation formalisms.

    Validates registrations against :class:`ActualCausationFormalism` so
    wrong-shape registrations fail at import. Lookup uses the string held in
    ``config.formalism.actual_causation.version``. Parallel to
    :class:`FormalismRegistry` / :class:`ActualCausationMeasureRegistry`.
    """

    desc = "actual-causation formalisms"

    def register(  # type: ignore[override]
        self, name: str, formalism: object
    ) -> ActualCausationFormalism:
        if not isinstance(formalism, ActualCausationFormalism):
            raise TypeError(
                f"Cannot register {formalism!r} as AC formalism {name!r}: "
                "object does not satisfy the ActualCausationFormalism Protocol."
            )
        self.store[name] = formalism  # type: ignore[assignment]
        return formalism


ACTUAL_CAUSATION_FORMALISM_REGISTRY: ActualCausationFormalismRegistry = (
    ActualCausationFormalismRegistry()
)
"""Global registry of actual-causation formalisms. Looked up by the string
held in ``config.formalism.actual_causation.version``."""
```

Add to `__all__`: `"ACTUAL_CAUSATION_FORMALISM_REGISTRY"`, `"ActualCausationFormalism"`, `"ActualCausationFormalismRegistry"`.

- [ ] **Step 4: Run the test to verify it passes.**

Run: `uv run pytest test/test_ac_formalism.py -q --no-header`
Expected: PASS (2 tests).

- [ ] **Step 5: Lint + commit.**

```bash
uv run ruff check pyphi/formalism/base.py test/test_ac_formalism.py
uv run ruff format pyphi/formalism/base.py test/test_ac_formalism.py
git add pyphi/formalism/base.py test/test_ac_formalism.py
git -c commit.gpgsign=false commit -m "Add ActualCausationFormalism Protocol and registry"
```

---

## Task 2: `version` field on `ActualCausationConfig`

**Files:**
- Modify: `pyphi/conf/formalism.py:97`
- Test: `test/test_ac_formalism.py`

- [ ] **Step 1: Write the failing test.** Append to `test/test_ac_formalism.py`:

```python
def test_ac_config_version_default_and_override():
    from pyphi import config

    assert config.formalism.actual_causation.version == "AC_2019"
    with config.override(version="AC_2019"):
        assert config.formalism.actual_causation.version == "AC_2019"
```

- [ ] **Step 2: Run it to verify it fails.**

Run: `uv run pytest test/test_ac_formalism.py::test_ac_config_version_default_and_override -q --no-header`
Expected: FAIL with `AttributeError` (no `version` attribute).

- [ ] **Step 3: Add the field.** In `pyphi/conf/formalism.py`, add as the first field of `ActualCausationConfig` (before `alpha_measure` at line 97), mirroring `IITConfig.version`:

```python
    version: str = "AC_2019"
```

Do **not** add any `__post_init__` validation for `version` or `alpha_measure` (resolve-time validation, per spec). Leave the existing three scheme checks unchanged.

- [ ] **Step 4: Run the test to verify it passes.**

Run: `uv run pytest test/test_ac_formalism.py -q --no-header`
Expected: PASS (3 tests).

- [ ] **Step 5: Confirm config still loads + `version` routes.**

Run: `uv run python -c "from pyphi import config; print(config.formalism.actual_causation.version)"`
Expected: prints `AC_2019`.

- [ ] **Step 6: Lint + commit.**

```bash
uv run ruff check pyphi/conf/formalism.py
git add pyphi/conf/formalism.py test/test_ac_formalism.py
git -c commit.gpgsign=false commit -m "Add version field to ActualCausationConfig"
```

---

## Task 3: Move AC compute algorithms into the package

This task **relocates** code with no behavioral change. The guard is the existing `test/test_actual.py` staying green and byte-identical. After this task, `actual.py`'s public functions call the moved `compute.py` functions directly (the formalism object does not exist yet).

**Files:**
- Create: `pyphi/formalism/actual_causation/__init__.py`, `pyphi/formalism/actual_causation/compute.py`
- Modify: `pyphi/actual.py`, `test/test_actual.py:19-37`

- [ ] **Step 1: Create the package skeleton.** Create `pyphi/formalism/actual_causation/__init__.py`:

```python
"""Actual Causation formalism (Albantakis et al. 2019)."""

from __future__ import annotations
```

- [ ] **Step 2: Create `compute.py` and move the algorithm functions verbatim.** Create `pyphi/formalism/actual_causation/compute.py`. Move the following from `pyphi/actual.py` **verbatim** (cut, do not rewrite the bodies):
  - The three registry classes + instances + registered functions: `PartitionedRepertoireSchemeRegistry`, `BackgroundStrategyRegistry`, `AlphaAggregationRegistry`, their instances `partitioned_repertoire_schemes` / `background_strategies` / `alpha_aggregations`, and `_partitioned_repertoire_product` / `_background_uniform` / `_alpha_subtractive` (`actual.py:64-130`).
  - `probability_distance` (`actual.py:1149-1192`).
  - `account_distance` (`actual.py:1200-1212`).
  - `_get_partitions` (`actual.py:1252-1276`).
  - `_evaluate_partition` (`actual.py:1214-1248`).
  - `directed_account` (`actual.py:1047-1092`) → rename to `_directed_account`.
  - `account` (`actual.py:1095-1146`) → rename to `_account`.
  - `sia` (`actual.py:1278-1345`) → rename to `_sia`.
  - The bodies of `Transition.find_mip` (`actual.py:836-927`) and `Transition.find_causal_link` (`actual.py:953-1027`) → extract into module functions `_find_mip(transition, direction, mechanism, purview, allow_neg=False, *, alpha_measure=None, partitioned_repertoire_scheme=None)` and `_find_causal_link(transition, direction, mechanism, purviews=None, allow_neg=False, *, alpha_measure=None, partitioned_repertoire_scheme=None)`. Each body's `self` becomes the `transition` parameter.

  Add the necessary imports at the top of `compute.py` (port the relevant subset of `actual.py`'s imports — `numpy`, `pyphi.conf.config`, `Direction`, `utils`, `validate`, `connectivity`, `resolve_ties`, `MapReduce`, `mechanism_partitions`, `actual_causation_measures as measures`, `resolve_actual_causation_measure`, and from `pyphi.models`: `Account`, `AcRepertoireIrreducibilityAnalysis`, `AcSystemIrreducibilityAnalysis`, `CausalLink`, `DirectedAccount`, `_null_ac_ria`, `_null_ac_sia`). Add a `TYPE_CHECKING` import of `Transition`/`TransitionSystem` from `pyphi.actual` for annotations only.

- [ ] **Step 3: Rewire internal calls inside `compute.py`.** Within the moved code, replace method-style calls with the module functions so `compute.py` is self-contained and never calls back into `actual.py`:
  - In `_directed_account`: `transition.find_causal_link(...)` → `_find_causal_link(transition, ...)`.
  - In `_account`: calls to `directed_account(...)` → `_directed_account(...)`.
  - In `_find_causal_link`: `self.find_mip(...)` → `_find_mip(transition, ...)`.
  - In `_sia` and `_evaluate_partition`: `account(...)` → `_account(...)`; `_get_partitions(...)` and `account_distance(...)` are local now.
  - Keep `_resolve_ac_kwargs` **in `actual.py` for now** (Task 5 replaces it); the moved `_find_mip` / `_directed_account` / `_account` / `_sia` already accept `alpha_measure` / `partitioned_repertoire_scheme` as kwargs, so they do not call `_resolve_ac_kwargs` themselves — their callers pass the resolved values. Verify each moved function resolves kwargs the same way it did as a method (the `if alpha_measure is None: resolved = _resolve_ac_kwargs()` blocks must still work — move a copy of `_resolve_ac_kwargs` into `compute.py` and have the moved functions call the local copy).

- [ ] **Step 4: Export from the package `__init__.py`.** Add to `pyphi/formalism/actual_causation/__init__.py`:

```python
from .compute import (
    account_distance,
    alpha_aggregations,
    background_strategies,
    partitioned_repertoire_schemes,
    probability_distance,
    _account,
    _directed_account,
    _evaluate_partition,
    _find_causal_link,
    _find_mip,
    _get_partitions,
    _sia,
)
```

- [ ] **Step 5: Turn `actual.py`'s removed functions into thin wrappers.** In `pyphi/actual.py`:
  - Delete the moved definitions (registries, scheme functions, `probability_distance`, `account_distance`, `_evaluate_partition`, `_get_partitions`, and the bodies of `directed_account`/`account`/`sia`).
  - Re-add module-level `account` / `directed_account` / `sia` that import from and call the package functions, preserving their exact public signatures:

```python
def directed_account(transition, direction, mechanisms=None, purviews=None,
                     allow_neg=False, *, alpha_measure=None,
                     partitioned_repertoire_scheme=None):
    from pyphi.formalism.actual_causation.compute import _directed_account
    return _directed_account(transition, direction, mechanisms, purviews,
                             allow_neg, alpha_measure=alpha_measure,
                             partitioned_repertoire_scheme=partitioned_repertoire_scheme)


def account(transition, direction=Direction.BIDIRECTIONAL, *, alpha_measure=None,
            partitioned_repertoire_scheme=None):
    from pyphi.formalism.actual_causation.compute import _account
    return _account(transition, direction, alpha_measure=alpha_measure,
                    partitioned_repertoire_scheme=partitioned_repertoire_scheme)


def sia(transition, direction=Direction.BIDIRECTIONAL, **kwargs):
    from pyphi.formalism.actual_causation.compute import _sia
    return _sia(transition, direction, **kwargs)
```

  - `Transition.find_mip` / `Transition.find_causal_link` bodies become one-line delegations:

```python
    def find_mip(self, direction, mechanism, purview, allow_neg=False, *,
                 alpha_measure=None, partitioned_repertoire_scheme=None):
        from pyphi.formalism.actual_causation.compute import _find_mip
        return _find_mip(self, direction, mechanism, purview, allow_neg,
                         alpha_measure=alpha_measure,
                         partitioned_repertoire_scheme=partitioned_repertoire_scheme)

    def find_causal_link(self, direction, mechanism, purviews=None, allow_neg=False, *,
                         alpha_measure=None, partitioned_repertoire_scheme=None):
        from pyphi.formalism.actual_causation.compute import _find_causal_link
        return _find_causal_link(self, direction, mechanism, purviews, allow_neg,
                                 alpha_measure=alpha_measure,
                                 partitioned_repertoire_scheme=partitioned_repertoire_scheme)
```

  - `find_actual_cause` / `find_actual_effect` (`actual.py:1029-1035`) are unchanged (they already call `self.find_causal_link`).
  - For backward references inside `actual.py` to the moved registries (e.g. `_resolve_ac_kwargs` reads `partitioned_repertoire_schemes` / `background_strategies` / `alpha_aggregations`), import them from the package: `from pyphi.formalism.actual_causation.compute import partitioned_repertoire_schemes, background_strategies, alpha_aggregations` (at the point of use, lazily, to avoid an import cycle).
  - Remove now-unused imports from `actual.py` (e.g. `MapReduce`, `mechanism_partitions`, `actual_causation_measures`) **only if** nothing else in `actual.py` uses them — verify with grep before deleting each.

- [ ] **Step 6: Update the three registry references in `test/test_actual.py`.** Lines 19-37 reference `actual.partitioned_repertoire_schemes` / `actual.background_strategies` / `actual.alpha_aggregations`. Repoint to the new path:

```python
def test_actual_partitioned_repertoire_schemes_registry():
    from pyphi.formalism.actual_causation.compute import partitioned_repertoire_schemes

    assert "PRODUCT" in partitioned_repertoire_schemes
    assert "FORWARD_PROBABILITY" not in partitioned_repertoire_schemes
```

Apply the analogous change to `test_actual_background_strategies_registry` and `test_actual_alpha_aggregations_registry`.

- [ ] **Step 7: Verify import + no cycle.**

Run: `uv run python -c "import pyphi; import pyphi.actual; import pyphi.formalism.actual_causation.compute"`
Expected: no `ImportError` / no circular-import error.

- [ ] **Step 8: Run the AC suite + goldens — must be byte-identical.**

Run: `uv run pytest test/test_actual.py test/test_golden_regression.py test/test_substrate_multivalued.py -q --no-header`
Expected: same pass/skip counts as the Task 0 baseline; no failures.

- [ ] **Step 9: Lint + commit.**

```bash
uv run ruff check pyphi/actual.py pyphi/formalism/actual_causation/ test/test_actual.py
uv run ruff format pyphi/actual.py pyphi/formalism/actual_causation/ test/test_actual.py
uv run pyright pyphi/actual.py pyphi/formalism/actual_causation/
git add pyphi/actual.py pyphi/formalism/actual_causation/ test/test_actual.py
git -c commit.gpgsign=false commit -m "Move AC compute algorithms into pyphi/formalism/actual_causation"
```

---

## Task 4: `AC2019Formalism` object + `_resolve_ac_measures`

Adds the formalism object and registers it. Additive — nothing dispatches through it yet (Task 5 does that).

**Files:**
- Create: `pyphi/formalism/actual_causation/formalism.py`
- Modify: `pyphi/formalism/actual_causation/__init__.py`, `pyphi/formalism/__init__.py`
- Test: `test/test_ac_formalism.py`

- [ ] **Step 1: Write the failing tests.** Append to `test/test_ac_formalism.py`:

```python
def test_ac2019_formalism_registered_and_satisfies_protocol():
    from pyphi.formalism.base import (
        ACTUAL_CAUSATION_FORMALISM_REGISTRY,
        ActualCausationFormalism,
    )

    formalism = ACTUAL_CAUSATION_FORMALISM_REGISTRY["AC_2019"]
    assert isinstance(formalism, ActualCausationFormalism)
    assert formalism.name == "AC_2019"
    assert "PMI" in formalism.compatible_measures
    assert "WPMI" in formalism.compatible_measures


def test_ac_formalism_rejects_incompatible_measure():
    from pyphi.formalism.actual_causation.formalism import _resolve_ac_measures
    from pyphi.formalism.base import (
        ACTUAL_CAUSATION_FORMALISM_REGISTRY,
        MeasureNotCompatibleError,
    )

    formalism = ACTUAL_CAUSATION_FORMALISM_REGISTRY["AC_2019"]
    # An IIT measure name is not in AC's compatible_measures.
    with pytest.raises(MeasureNotCompatibleError):
        _resolve_ac_measures(formalism, alpha_measure_name="GENERALIZED_INTRINSIC_DIFFERENCE")


def test_ac_formalism_unknown_version_raises():
    from pyphi.formalism.base import ACTUAL_CAUSATION_FORMALISM_REGISTRY

    with pytest.raises(KeyError):
        ACTUAL_CAUSATION_FORMALISM_REGISTRY["NOPE"]
```

- [ ] **Step 2: Run them to verify they fail.**

Run: `uv run pytest test/test_ac_formalism.py -k "ac2019 or incompatible or unknown_version" -q --no-header`
Expected: FAIL (`KeyError`/`ImportError`: nothing registered, `_resolve_ac_measures` absent).

- [ ] **Step 3: Create `formalism.py`.** Create `pyphi/formalism/actual_causation/formalism.py`:

```python
"""The AC_2019 actual-causation formalism object."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import ClassVar

from pyphi.conf import config
from pyphi.conf.formalism import FormalismConfig
from pyphi.formalism.base import check_measure_compatible

from . import compute


def _default_formalism_config() -> FormalismConfig:
    from pyphi.conf import config as _global

    return _global.formalism


def _resolve_ac_measures(
    formalism: Any,
    *,
    alpha_measure_name: str | None = None,
    partitioned_repertoire_scheme_name: str | None = None,
    background_scheme_name: str | None = None,
    alpha_aggregation_name: str | None = None,
) -> dict[str, Any]:
    """Resolve AC measure/scheme config into callables, checking compatibility.

    Mirrors IIT's ``_resolve_system_measures``: an explicitly-passed name
    overrides config; the chosen ``alpha_measure`` name is checked against
    ``formalism.compatible_measures`` before resolution. The three schemes
    resolve from their registries by name.
    """
    from pyphi.measures.distribution import resolve_actual_causation_measure

    ac = formalism.config.actual_causation
    alpha_name = alpha_measure_name if alpha_measure_name is not None else ac.alpha_measure
    check_measure_compatible(formalism, alpha_name)

    pr_name = (
        partitioned_repertoire_scheme_name
        if partitioned_repertoire_scheme_name is not None
        else ac.partitioned_repertoire_scheme
    )
    bg_name = background_scheme_name if background_scheme_name is not None else ac.background_scheme
    agg_name = alpha_aggregation_name if alpha_aggregation_name is not None else ac.alpha_aggregation

    return {
        "alpha_measure": resolve_actual_causation_measure(alpha_name),
        "partitioned_repertoire_scheme": compute.partitioned_repertoire_schemes[pr_name],
        "background_scheme": compute.background_strategies[bg_name],
        "alpha_aggregation": compute.alpha_aggregations[agg_name],
    }


@dataclass(frozen=True)
class AC2019Formalism:
    """Actual Causation formalism (Albantakis et al. 2019, "What Caused What?")."""

    name: ClassVar[str] = "AC_2019"
    compatible_measures: ClassVar[frozenset[str]] = frozenset({"PMI", "WPMI"})

    config: FormalismConfig = field(default_factory=_default_formalism_config)

    def evaluate_account(self, transition: Any, direction: Any = None, **kwargs: Any) -> Any:
        from pyphi.direction import Direction

        if direction is None:
            direction = Direction.BIDIRECTIONAL
        resolved = _resolve_ac_measures(self)
        return compute._account(
            transition,
            direction,
            alpha_measure=resolved["alpha_measure"],
            partitioned_repertoire_scheme=resolved["partitioned_repertoire_scheme"],
        )

    def evaluate_system(self, transition: Any, direction: Any = None, **kwargs: Any) -> Any:
        from pyphi.direction import Direction

        if direction is None:
            direction = Direction.BIDIRECTIONAL
        return compute._sia(transition, direction, **kwargs)

    def evaluate_mechanism(
        self, transition: Any, direction: Any, mechanism: Any, purview: Any, **kwargs: Any
    ) -> Any:
        resolved = _resolve_ac_measures(self)
        return compute._find_mip(
            transition,
            direction,
            mechanism,
            purview,
            kwargs.get("allow_neg", False),
            alpha_measure=resolved["alpha_measure"],
            partitioned_repertoire_scheme=resolved["partitioned_repertoire_scheme"],
        )

    def evaluate_causal_link(
        self, transition: Any, direction: Any, mechanism: Any, **kwargs: Any
    ) -> Any:
        resolved = _resolve_ac_measures(self)
        return compute._find_causal_link(
            transition,
            direction,
            mechanism,
            kwargs.get("purviews"),
            kwargs.get("allow_neg", False),
            alpha_measure=resolved["alpha_measure"],
            partitioned_repertoire_scheme=resolved["partitioned_repertoire_scheme"],
        )
```

(Note: `_sia` resolves its own kwargs internally today via the local `_resolve_ac_kwargs` copy; `evaluate_system` therefore forwards `**kwargs` and lets `_sia` resolve. Task 5 tightens this so `_sia` receives the resolved values from `_resolve_ac_measures`.)

- [ ] **Step 4: Register the formalism.** Append to `pyphi/formalism/actual_causation/__init__.py`:

```python
from pyphi.formalism.base import ACTUAL_CAUSATION_FORMALISM_REGISTRY

from .formalism import AC2019Formalism

ACTUAL_CAUSATION_FORMALISM_REGISTRY.register("AC_2019", AC2019Formalism())

__all__ = ["AC2019Formalism"]
```

- [ ] **Step 5: Ensure registration runs at formalism-package import.** In `pyphi/formalism/__init__.py`, add an import of the AC package alongside the IIT formalism imports (find where `IIT4_2023Formalism` etc. are imported/registered, around line 40):

```python
from pyphi.formalism import actual_causation  # noqa: F401  (registers AC_2019)
```

- [ ] **Step 6: Run the new tests to verify they pass.**

Run: `uv run pytest test/test_ac_formalism.py -q --no-header`
Expected: PASS (all tests).

- [ ] **Step 7: Verify import + full AC suite still green.**

Run: `uv run python -c "import pyphi"` then `uv run pytest test/test_actual.py -q --no-header`
Expected: import clean; AC suite unchanged.

- [ ] **Step 8: Lint + commit.**

```bash
uv run ruff check pyphi/formalism/actual_causation/ pyphi/formalism/__init__.py test/test_ac_formalism.py
uv run ruff format pyphi/formalism/actual_causation/ pyphi/formalism/__init__.py test/test_ac_formalism.py
uv run pyright pyphi/formalism/actual_causation/
git add pyphi/formalism/actual_causation/ pyphi/formalism/__init__.py test/test_ac_formalism.py
git -c commit.gpgsign=false commit -m "Add AC2019Formalism object and register it"
```

---

## Task 5: Dispatch the public AC API through the formalism

Repoints `actual.py`'s public functions to go through `ACTUAL_CAUSATION_FORMALISM_REGISTRY[config.formalism.actual_causation.version]`, so the formalism object is the dispatch boundary (mirroring `queries.sia`). Behavior unchanged.

**Files:**
- Modify: `pyphi/actual.py`, `pyphi/formalism/actual_causation/formalism.py`
- Test: `test/test_ac_formalism.py`

- [ ] **Step 1: Write the failing test.** Append to `test/test_ac_formalism.py`:

```python
def test_account_dispatches_through_active_formalism(monkeypatch):
    """pyphi.actual.account routes through the registered AC formalism."""
    import pyphi.actual as actual
    from pyphi.formalism.base import ACTUAL_CAUSATION_FORMALISM_REGISTRY

    called = {}
    formalism = ACTUAL_CAUSATION_FORMALISM_REGISTRY["AC_2019"]
    original = formalism.evaluate_account

    def spy(transition, direction=None, **kwargs):
        called["hit"] = True
        return original(transition, direction, **kwargs)

    monkeypatch.setattr(formalism, "evaluate_account", spy, raising=False)
    transition = __import__("pyphi").examples.actual_causation()  # paper fixture
    actual.account(transition)
    assert called["hit"]
```

(If `pyphi.examples.actual_causation()` is not the exact fixture name, use the one `test/test_actual.py` uses to build a `Transition`; confirm by grep `def .*actual` in `pyphi/examples.py`.)

- [ ] **Step 2: Run it to verify it fails.**

Run: `uv run pytest test/test_ac_formalism.py::test_account_dispatches_through_active_formalism -q --no-header`
Expected: FAIL (`account` calls `compute._account` directly, not the formalism).

- [ ] **Step 3: Repoint the public dispatchers in `actual.py`.** Replace the Task-3 wrappers so they look up the active formalism (lazy import to avoid a cycle), mirroring `queries.sia`:

```python
def account(transition, direction=Direction.BIDIRECTIONAL, **kwargs):
    from pyphi.conf import config
    from pyphi.formalism.base import ACTUAL_CAUSATION_FORMALISM_REGISTRY

    formalism = ACTUAL_CAUSATION_FORMALISM_REGISTRY[
        config.formalism.actual_causation.version
    ]
    return formalism.evaluate_account(transition, direction, **kwargs)


def directed_account(transition, direction, **kwargs):
    from pyphi.conf import config
    from pyphi.formalism.base import ACTUAL_CAUSATION_FORMALISM_REGISTRY

    formalism = ACTUAL_CAUSATION_FORMALISM_REGISTRY[
        config.formalism.actual_causation.version
    ]
    return formalism.evaluate_account(transition, direction, **kwargs)


def sia(transition, direction=Direction.BIDIRECTIONAL, **kwargs):
    from pyphi.conf import config
    from pyphi.formalism.base import ACTUAL_CAUSATION_FORMALISM_REGISTRY

    formalism = ACTUAL_CAUSATION_FORMALISM_REGISTRY[
        config.formalism.actual_causation.version
    ]
    return formalism.evaluate_system(transition, direction, **kwargs)
```

  - `Transition.find_mip` / `find_causal_link` repoint to `formalism.evaluate_mechanism` / `evaluate_causal_link`:

```python
    def find_mip(self, direction, mechanism, purview, allow_neg=False, **kwargs):
        from pyphi.conf import config
        from pyphi.formalism.base import ACTUAL_CAUSATION_FORMALISM_REGISTRY

        formalism = ACTUAL_CAUSATION_FORMALISM_REGISTRY[
            config.formalism.actual_causation.version
        ]
        return formalism.evaluate_mechanism(
            self, direction, mechanism, purview, allow_neg=allow_neg, **kwargs
        )

    def find_causal_link(self, direction, mechanism, purviews=None, allow_neg=False, **kwargs):
        from pyphi.conf import config
        from pyphi.formalism.base import ACTUAL_CAUSATION_FORMALISM_REGISTRY

        formalism = ACTUAL_CAUSATION_FORMALISM_REGISTRY[
            config.formalism.actual_causation.version
        ]
        return formalism.evaluate_causal_link(
            self, direction, mechanism, purviews=purviews, allow_neg=allow_neg, **kwargs
        )
```

  **Caution:** `directed_account` historically takes a required `direction` plus optional `mechanisms`/`purviews`. Preserve its full signature and thread `mechanisms`/`purviews` through `evaluate_account` via `**kwargs` → `compute._directed_account`. If `evaluate_account` currently ignores `mechanisms`/`purviews`, extend `evaluate_account` to accept and forward them. Verify against the original `directed_account` signature (`actual.py:1047`).

- [ ] **Step 4: Collapse `_resolve_ac_kwargs` into `_resolve_ac_measures`.** Now that all entry points resolve via the formalism, remove the `_resolve_ac_kwargs` copy from `compute.py` and have the moved `_account`/`_directed_account`/`_find_mip`/`_find_causal_link`/`_sia` require their `alpha_measure`/`partitioned_repertoire_scheme` kwargs (raise/assert if `None`), since the formalism always supplies them. Update `evaluate_system` to call `_resolve_ac_measures(self)` and pass the resolved values into `_sia` (rather than `_sia` resolving internally). Grep to confirm no remaining `_resolve_ac_kwargs` references.

- [ ] **Step 5: Run the dispatch test + full AC suite.**

Run: `uv run pytest test/test_ac_formalism.py test/test_actual.py test/test_golden_regression.py test/test_substrate_multivalued.py -q --no-header`
Expected: all pass; AC golden/paper results byte-identical.

- [ ] **Step 6: Full doctest sweep.**

Run: `uv run pytest -q --no-header` (no path argument)
Expected: green (same counts as baseline plus the new `test_ac_formalism.py` tests).

- [ ] **Step 7: Lint + commit.**

```bash
uv run ruff check pyphi/actual.py pyphi/formalism/actual_causation/
uv run ruff format pyphi/actual.py pyphi/formalism/actual_causation/
uv run pyright pyphi/actual.py pyphi/formalism/actual_causation/
git add pyphi/actual.py pyphi/formalism/actual_causation/ test/test_ac_formalism.py
git -c commit.gpgsign=false commit -m "Dispatch public AC API through AC2019Formalism"
```

---

## Task 6: Changelog + ROADMAP

**Files:**
- Create: `changelog.d/ac-formalism-object.feature.md`
- Modify: `ROADMAP.md`

- [ ] **Step 1: Create the changelog fragment.**

```
Actual Causation now has a registered formalism object, ``AC2019Formalism``, mirroring the IIT formalism objects. It is selected by the new ``config.formalism.actual_causation.version`` setting (default ``"AC_2019"``) and validates the configured ``alpha_measure`` against its ``compatible_measures`` set at resolve time. The AC compute algorithms, the ``probability_distance`` / ``account_distance`` utilities, and the ``partitioned_repertoire_schemes`` / ``background_strategies`` / ``alpha_aggregations`` registries moved from ``pyphi.actual`` into the new ``pyphi.formalism.actual_causation`` package; import them from there (custom-scheme registration now uses ``pyphi.formalism.actual_causation.compute``). The public ``pyphi.actual.account`` / ``directed_account`` / ``sia`` API and ``Transition`` methods are unchanged. No change to AC results.
```

Write it: `echo "<text>" > changelog.d/ac-formalism-object.feature.md` (or use `uv run towncrier create`).

- [ ] **Step 2: Update ROADMAP.** Find the AC measure-config / formalism item in `ROADMAP.md` and either remove it (if fully delivered) or annotate it as landed, noting the deferred IIT-4.0-style AC formalism is the next AC step. Grep: `grep -ni "actual.causation\|alpha_measure\|AC formalism" ROADMAP.md`.

- [ ] **Step 3: Verify towncrier renders + full suite.**

Run: `uv run towncrier build --draft --version 2.0` (confirm the fragment appears) and `uv run pytest -q --no-header`
Expected: fragment listed; suite green.

- [ ] **Step 4: Commit.**

```bash
git add changelog.d/ac-formalism-object.feature.md ROADMAP.md
git -c commit.gpgsign=false commit -m "Changelog and ROADMAP for the AC formalism object"
```

---

## Acceptance gate

- `ACTUAL_CAUSATION_FORMALISM_REGISTRY["AC_2019"]` returns an `AC2019Formalism` satisfying the `ActualCausationFormalism` Protocol.
- `config.formalism.actual_causation.version` selects the formalism; an unknown version raises `KeyError`; an incompatible `alpha_measure` raises `MeasureNotCompatibleError` at resolve time.
- AC compute lives in `pyphi/formalism/actual_causation/`; `actual.py` retains only `Transition`/`TransitionSystem` + thin dispatchers.
- Public API (`pyphi.actual.account`/`directed_account`/`sia`, `Transition.find_*`) unchanged; `examples.py` and all existing AC tests pass byte-identical.
- Full `uv run pytest` (no path) green; pyright clean; ruff clean.
- Changelog fragment added; ROADMAP updated.

## Self-review notes

- **Spec coverage:** Protocol+registry (Task 1) ✓; version field + deferred validation (Task 2) ✓; package + moved algorithms/registries/utils (Task 3) ✓; `AC2019Formalism` + `compatible_measures` resolve-time check + `_resolve_ac_measures` (Task 4) ✓; stable public API via dispatchers (Task 5) ✓; changelog/ROADMAP + moved-path note (Task 6) ✓; zero-numeric-change contract (Tasks 3/5 byte-identical gate) ✓.
- **Naming consistency:** `ActualCausationFormalism` (Protocol), `ActualCausationFormalismRegistry`, `ACTUAL_CAUSATION_FORMALISM_REGISTRY`, `AC2019Formalism`, `_resolve_ac_measures` used identically across Tasks 1, 4, 5.
- **Open verification points flagged inline:** the exact `pyphi.examples` AC-fixture name (Task 5 Step 1) and `directed_account`'s `mechanisms`/`purviews` threading (Task 5 Step 3) must be confirmed against the live code during execution.
