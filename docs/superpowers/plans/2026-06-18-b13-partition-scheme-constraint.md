# B13 — Partition-scheme × version constraint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an eager config constraint rejecting `IIT_3_0` paired with a `system_partition_scheme` outside its compatible set, sharing a single source of truth with the existing reactive raise; close the EMD-precision and IIT-4.0 deferrals in the docs.

**Architecture:** A `compatible_system_partition_schemes` class attribute on the IIT formalisms (3.0 restricted, 4.0 `None` = open) becomes the single source of truth. The reactive raise in `iit3`'s `sia_partitions()` reads it; a new constraint in `conf/constraints.py` reads it via the formalism registry and rejects out-of-set combinations eagerly, exactly as the landed measure constraint mirrors `check_measure_compatible`.

**Tech Stack:** Python 3.12+, pytest.

**Spec:** `docs/superpowers/specs/2026-06-18-b13-partition-scheme-constraint-design.md`

## Global Constraints

- Python 3.12+ only; no backward-compatibility shims; no new dependency.
- Validation only — **no computed value changes**. The accepted set is exactly what already computes (confirmed by `b13_experiments/FINDINGS.md`).
- IIT 4.0 stays unconstrained on the system scheme (encoded as `None`, not a branch).
- Single source of truth: the reactive `sia_partitions()` raise and the eager constraint must read the same `compatible_system_partition_schemes` attribute.
- Constraint message names both conflicting fields (`formalism.iit.system_partition_scheme`, `formalism.iit.version`) and a concrete fix, like `_measure_compatible_with_version`.
- Use `uv run` for all Python commands. Final verification runs `uv run pytest` **with no path argument** (config surface; doctest sweep).
- Do not bypass pre-commit hooks. Stage only the files each task names (the tree has unrelated untracked work; never `git add -A`).

---

### Task 1: Single source of truth on the formalisms + reactive site reads it

**Files:**
- Modify: `pyphi/formalism/base.py` (declare the attribute on the `PhiFormalism` Protocol)
- Modify: `pyphi/formalism/iit3/formalism.py` (set the restricted set on `IIT3Formalism`)
- Modify: `pyphi/formalism/iit4/formalism.py` (set `None` on both 4.0 formalisms)
- Modify: `pyphi/formalism/iit3/__init__.py` (`sia_partitions()` reads the attribute)
- Test: `test/test_config_constraints.py` (new class `TestSystemSchemeSingleSourceOfTruth`)

**Interfaces:**
- Produces: `PhiFormalism.compatible_system_partition_schemes: ClassVar[frozenset[str] | None]`
- Produces: `IIT3Formalism.compatible_system_partition_schemes == frozenset({"DIRECTED_BIPARTITION", "DIRECTED_BIPARTITION_CUT_ONE"})`
- Produces: `IIT4_2023Formalism` / `IIT4_2026Formalism` `.compatible_system_partition_schemes is None`

- [ ] **Step 1: Write the failing test**

First add `from pyphi.partition import system_partition_types` to the **top-level import block** of `test/test_config_constraints.py` (with the other `from pyphi...` imports, so ruff does not flag E402). Then add this class to the end of the file:

```python
class TestSystemSchemeSingleSourceOfTruth:
    def test_iit3_formalism_declares_restricted_scheme_set(self) -> None:
        formalism = FORMALISM_REGISTRY["IIT_3_0"]
        assert formalism.compatible_system_partition_schemes == frozenset(
            {"DIRECTED_BIPARTITION", "DIRECTED_BIPARTITION_CUT_ONE"}
        )

    @pytest.mark.parametrize("version", _FOUR_OH)
    def test_iit4_formalism_leaves_scheme_open(self, version: str) -> None:
        assert FORMALISM_REGISTRY[version].compatible_system_partition_schemes is None

    def test_reactive_raise_reads_the_attribute(self) -> None:
        """sia_partitions() rejects an out-of-set scheme under IIT 3.0, using the
        same set the formalism declares (validation off, so the eager constraint
        is not what fires)."""
        from pyphi import examples

        out_of_set = "DIRECTED_SET_PARTITION"
        assert out_of_set not in FORMALISM_REGISTRY[
            "IIT_3_0"
        ].compatible_system_partition_schemes
        with config.override(
            **presets.iit3,
            validate_config=False,
            **{"iit.system_partition_scheme": out_of_set},
        ):
            with pytest.raises(ValueError, match="system partition scheme"):
                examples.basic_system().sia()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest test/test_config_constraints.py::TestSystemSchemeSingleSourceOfTruth -q`
Expected: FAIL (`compatible_system_partition_schemes` attribute does not exist).

- [ ] **Step 3: Declare the attribute on the Protocol base**

In `pyphi/formalism/base.py`, in the `PhiFormalism` class body, add the declaration next to `compatible_measures` / `partition_scheme` (after `partition_scheme: ClassVar[str | None]`):

```python
    compatible_system_partition_schemes: ClassVar[frozenset[str] | None]
```

And add a bullet to the class docstring's "Concrete formalisms also declare" list, after the `partition_scheme` bullet:

```python
    - ``compatible_system_partition_schemes``: frozenset of system partition
      scheme names this formalism accepts, or ``None`` if it accepts any
      registered scheme.
```

- [ ] **Step 4: Set the restricted set on IIT 3.0**

In `pyphi/formalism/iit3/formalism.py`, in the `IIT3Formalism` class body, add after the `partition_scheme` ClassVar (line ~45):

```python
    compatible_system_partition_schemes: ClassVar[frozenset[str] | None] = frozenset(
        {"DIRECTED_BIPARTITION", "DIRECTED_BIPARTITION_CUT_ONE"}
    )
```

- [ ] **Step 5: Set `None` on both IIT 4.0 formalisms**

In `pyphi/formalism/iit4/formalism.py`, in **both** `IIT4_2023Formalism` and `IIT4_2026Formalism` class bodies, add after each `partition_scheme` ClassVar (lines ~279 and ~440):

```python
    compatible_system_partition_schemes: ClassVar[frozenset[str] | None] = None
```

- [ ] **Step 6: Make the reactive raise read the attribute**

In `pyphi/formalism/iit3/__init__.py`, replace the body of `sia_partitions()` that hardcodes `valid` (the block at lines ~298-304):

```python
    scheme = config.formalism.iit.system_partition_scheme
    valid = ["DIRECTED_BIPARTITION", "DIRECTED_BIPARTITION_CUT_ONE"]
    if scheme not in valid:
        raise ValueError(
            "IIT 3.0 calculations must use one of the following system "
            f"partition schemes: {valid}; got {scheme}"
        )
```

with a version that reads the single source of truth (function-level import avoids any module-load cycle):

```python
    scheme = config.formalism.iit.system_partition_scheme
    from pyphi.formalism.iit3.formalism import IIT3Formalism

    valid = IIT3Formalism.compatible_system_partition_schemes
    if scheme not in valid:
        raise ValueError(
            "IIT 3.0 calculations must use one of the following system "
            f"partition schemes: {sorted(valid)}; got {scheme}"
        )
```

(Leave the preceding `# TODO(4.0 consolidate 3.0 and 4.0 cuts)` comment and the `return system_partition_types[...]` line unchanged.)

- [ ] **Step 7: Run the test to verify it passes**

Run: `uv run pytest test/test_config_constraints.py::TestSystemSchemeSingleSourceOfTruth -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add pyphi/formalism/base.py pyphi/formalism/iit3/formalism.py pyphi/formalism/iit4/formalism.py pyphi/formalism/iit3/__init__.py test/test_config_constraints.py
git commit -m "Add compatible_system_partition_schemes as single source of truth

IIT 3.0 declares its restricted system-scheme set; IIT 4.0 leaves it open
(None). The reactive sia_partitions() raise reads the attribute instead of a
hardcoded list, so the eager constraint and the reactive raise cannot drift."
```

---

### Task 2: Eager constraint

**Files:**
- Modify: `pyphi/conf/constraints.py` (factor the formalism-resolution helper; add the constraint)
- Test: `test/test_config_constraints.py` (new class `TestSystemSchemeConstraint`)

**Interfaces:**
- Consumes: `compatible_system_partition_schemes` (Task 1); existing `_FORMALISM_UNAVAILABLE`, `register_constraint`.
- Produces: registered constraint `"system_partition_scheme_compatible_with_version"`.
- Produces (internal): `_active_formalism(version) -> object | None | <PhiFormalism>` — returns the formalism instance, `None` (unregistered), or `_FORMALISM_UNAVAILABLE` (bootstrap window).

- [ ] **Step 1: Write the failing tests**

Add to `test/test_config_constraints.py`:

```python
class TestSystemSchemeConstraint:
    def test_iit3_with_set_partition_scheme_rejected(self) -> None:
        with pytest.raises(ConfigurationError) as exc, config.override(
            **presets.iit3,
            **{"iit.system_partition_scheme": "DIRECTED_SET_PARTITION"},
        ):
            pass
        message = str(exc.value)
        assert "system_partition_scheme" in message
        assert "version" in message
        assert "DIRECTED_SET_PARTITION" in message
        assert "Fix" in message

    @pytest.mark.parametrize(
        "scheme", ["DIRECTED_BIPARTITION", "DIRECTED_BIPARTITION_CUT_ONE"]
    )
    def test_iit3_with_valid_scheme_passes(self, scheme: str) -> None:
        with config.override(
            **presets.iit3, **{"iit.system_partition_scheme": scheme}
        ):
            assert config.formalism.iit.system_partition_scheme == scheme

    @pytest.mark.parametrize("version", _FOUR_OH)
    def test_iit4_accepts_every_registered_scheme(self, version: str) -> None:
        for scheme in system_partition_types.store:
            with config.override(
                **{"iit.version": version, "iit.system_partition_scheme": scheme}
            ):
                assert config.formalism.iit.system_partition_scheme == scheme

    def test_validate_config_false_bypasses_scheme_constraint(self) -> None:
        with config.override(
            **presets.iit3,
            validate_config=False,
            **{"iit.system_partition_scheme": "DIRECTED_SET_PARTITION"},
        ):
            assert (
                config.formalism.iit.system_partition_scheme
                == "DIRECTED_SET_PARTITION"
            )

    def test_rejected_scheme_override_restores_state(self) -> None:
        with config.override(**presets.iit3):
            before = config.formalism.iit.system_partition_scheme
            with pytest.raises(ConfigurationError), config.override(
                **{"iit.system_partition_scheme": "DIRECTED_SET_PARTITION"}
            ):
                pass
            assert config.formalism.iit.system_partition_scheme == before


class TestSystemSchemeEnumerationConsistency:
    """The eager constraint's accept/reject for IIT 3.0 matches whether a real
    SIA computes vs raises, for every registered system scheme."""

    def test_iit3_classification_matches_sia_behavior(self) -> None:
        from pyphi import examples

        for scheme in sorted(system_partition_types.store):
            eager_rejected = False
            try:
                with config.override(
                    **presets.iit3, **{"iit.system_partition_scheme": scheme}
                ):
                    pass
            except ConfigurationError:
                eager_rejected = True

            sia_raised = False
            try:
                with config.override(
                    **presets.iit3,
                    validate_config=False,
                    **{"iit.system_partition_scheme": scheme},
                ):
                    examples.basic_system().sia()
            except ValueError:
                sia_raised = True

            assert eager_rejected == sia_raised, scheme
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest test/test_config_constraints.py::TestSystemSchemeConstraint test/test_config_constraints.py::TestSystemSchemeEnumerationConsistency -q`
Expected: FAIL (the IIT 3.0 + `DIRECTED_SET_PARTITION` override does not raise `ConfigurationError` yet).

- [ ] **Step 3: Factor the formalism-resolution helper**

In `pyphi/conf/constraints.py`, add a helper above `_compatible_measures` and refactor `_compatible_measures` to use it (behavior identical):

```python
def _active_formalism(version: str) -> Any:
    """Return the formalism instance for ``version``.

    Returns ``None`` if ``version`` is unregistered, or
    :data:`_FORMALISM_UNAVAILABLE` if the formalism registry can't be imported
    yet (the bootstrap window; see :data:`_FORMALISM_UNAVAILABLE`). Imported
    lazily: ``pyphi.formalism`` depends on ``pyphi.conf``.
    """
    try:
        from pyphi.formalism.base import FORMALISM_REGISTRY
    except ImportError:
        return _FORMALISM_UNAVAILABLE
    try:
        return FORMALISM_REGISTRY[version]
    except KeyError:
        return None
```

Replace the body of `_compatible_measures` with:

```python
def _compatible_measures(version: str) -> frozenset[str] | None | object:
    """Return the active formalism's ``compatible_measures`` (or the
    ``None`` / :data:`_FORMALISM_UNAVAILABLE` sentinels from
    :func:`_active_formalism`)."""
    formalism = _active_formalism(version)
    if formalism is None or formalism is _FORMALISM_UNAVAILABLE:
        return formalism
    return frozenset(formalism.compatible_measures)
```

- [ ] **Step 4: Add the constraint**

Append to `pyphi/conf/constraints.py`:

```python
@register_constraint("system_partition_scheme_compatible_with_version")
def _system_partition_scheme_compatible_with_version(config: Any) -> str | None:
    """The system partition scheme must be one the active formalism accepts.

    IIT 3.0 only supports ``DIRECTED_BIPARTITION`` /
    ``DIRECTED_BIPARTITION_CUT_ONE`` system schemes (its ``sia_partitions``
    raises otherwise); pairing it with any other scheme computes nothing usable.
    Formalisms that accept any registered scheme declare
    ``compatible_system_partition_schemes = None`` and are not constrained.
    """
    iit = config.formalism.iit
    version = iit.version
    formalism = _active_formalism(version)
    if formalism is None or formalism is _FORMALISM_UNAVAILABLE:
        return None  # bootstrap window, or unregistered (the measure constraint reports it)
    compatible = getattr(formalism, "compatible_system_partition_schemes", None)
    if compatible is None:
        return None  # unconstrained (e.g. IIT 4.0)
    scheme = iit.system_partition_scheme
    if scheme not in compatible:
        return (
            f"formalism.iit.system_partition_scheme={scheme!r} is not compatible "
            f"with formalism.iit.version={version!r}. Compatible system partition "
            f"schemes for this version: {sorted(compatible)}. Fix: set "
            f"formalism.iit.system_partition_scheme to one of those, or change "
            f"formalism.iit.version to one whose formalism accepts {scheme!r}."
        )
    return None
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest test/test_config_constraints.py -q`
Expected: PASS (the whole file, including the existing measure-constraint tests unaffected by the `_compatible_measures` refactor).

- [ ] **Step 6: Commit**

```bash
git add pyphi/conf/constraints.py test/test_config_constraints.py
git commit -m "Add eager system_partition_scheme x version constraint

Rejects IIT 3.0 paired with a system partition scheme outside its compatible
set at configuration time, mirroring the reactive sia_partitions() raise. IIT
4.0 (compatible set None) accepts any scheme. Factors a shared
_active_formalism() helper out of _compatible_measures."
```

---

### Task 3: Docs closeout + full verification

**Files:**
- Modify: `pyphi/conf/constraints.py` (docstring: close both deferrals)
- Create: `changelog.d/b13-partition-scheme-constraint.config.md`
- Modify: `ROADMAP.md` (B13 dashboard row 🟡 → ✅; Wave 2 archive note)

- [ ] **Step 1: Update the constraints.py docstring**

In `pyphi/conf/constraints.py`, the module docstring's "Design notes" mention deferred work implicitly via "add only ones backed by a confirmation experiment." Update the final paragraph to record the two now-closed deferrals. Replace:

```python
Registering more constraints is a matter of appending a
:class:`ConfigConstraint` to :data:`CONFIG_CONSTRAINTS` (or using
:func:`register_constraint`); add only ones backed by a confirmation
experiment.
```

with:

```python
Registering more constraints is a matter of appending a
:class:`ConfigConstraint` to :data:`CONFIG_CONSTRAINTS` (or using
:func:`register_constraint`); add only ones backed by a confirmation
experiment.

Two previously-deferred constraints were resolved by confirmation experiment
(seed 20260618):

- **EMD precision** — no constraint. Under the POT backend (``ot.emd2``, an exact
  network-simplex LP) the EMD noise floor is machine epsilon, and IIT 3.0 phi is
  stable across precision 6–13 with an identical MIP. The ``precision: 6`` pin is
  a goldens-calibration choice, not a correctness requirement.
- **System partition scheme × version** — IIT 3.0 only accepts
  ``DIRECTED_BIPARTITION`` / ``DIRECTED_BIPARTITION_CUT_ONE`` (the
  ``system_partition_scheme_compatible_with_version`` constraint, mirroring its
  reactive ``sia_partitions`` raise). IIT 4.0 accepts any registered scheme (a
  non-default scheme computes a well-defined per-scheme phi), so it is left
  unconstrained (``compatible_system_partition_schemes = None``).
```

- [ ] **Step 2: Write the changelog fragment**

Create `changelog.d/b13-partition-scheme-constraint.config.md`:

```markdown
The eager config validator now rejects `IIT_3_0` paired with a
`system_partition_scheme` other than `DIRECTED_BIPARTITION` or
`DIRECTED_BIPARTITION_CUT_ONE` at configuration time (previously this failed
only reactively, deep in the compute path). The error names both conflicting
fields and a fix. Configurable off via `validate_config=False`. The IIT 4.0
family accepts any registered system scheme and is unaffected.
```

- [ ] **Step 3: Update the ROADMAP B13 dashboard row**

In `ROADMAP.md`, change the `B13 config validator` row status from `🟡 partial` to `✅ landed` and update its one-line to record the resolution:

```markdown
| B13 config validator | ✅ landed | 2 | Eager rejection of cross-field config conflicts (`pyphi/conf/constraints.py`): measure↔version combos, and now `IIT_3_0` + a `system_partition_scheme` outside `{DIRECTED_BIPARTITION, DIRECTED_BIPARTITION_CUT_ONE}` (mirrors the reactive `sia_partitions` raise eagerly via a shared `compatible_system_partition_schemes` source of truth). Two-field error + fix; `validate_config` opt-out; failed apply restores state. **Both deferrals closed by confirmation experiment (seed 20260618):** EMD under POT is numerically clean (noise floor machine-epsilon; IIT 3.0 phi stable across precision 6–13, identical MIP) → no precision constraint; IIT 4.0 computes a well-defined per-scheme phi for any system scheme → left open. |
```

- [ ] **Step 4: Update the Wave 2 archive note for B13**

In `ROADMAP.md`, find the Wave 2 archive bullet describing B13 (search `B13 — eager config-combination validator`) and append a sentence recording that the two deferrals are now resolved: the scheme constraint landed (IIT 3.0 only, mirroring the reactive raise), the EMD-precision deferral was closed with no constraint (POT is clean), and IIT 4.0 was left open by confirmation. In the `### ✅ Landed` prose line near the top, append `· B13 scheme-constraint`.

- [ ] **Step 5: Run the full verification gate**

Run: `uv run pytest`
Expected: PASS with **no path argument** (collects `pyphi/` + `test/` doctests and the config-constraint tests). If it errors at collection on matplotlib, run `uv sync --all-extras` first.

- [ ] **Step 6: Commit**

```bash
git add pyphi/conf/constraints.py changelog.d/b13-partition-scheme-constraint.config.md ROADMAP.md
git commit -m "Close B13 deferrals: document EMD-precision (none) + scheme constraint

Record in the constraints docstring, changelog, and ROADMAP that the EMD
precision floor needs no constraint (POT is numerically clean) and the system
partition scheme x version constraint landed for IIT 3.0; IIT 4.0 left open."
```

---

## Self-Review

**Spec coverage:**
- Single source of truth attribute (spec §4.1) → Task 1 (Steps 3–5).
- Reactive site reads the attribute (spec §4.2) → Task 1 (Step 6).
- Eager constraint + shared helper (spec §4.3) → Task 2.
- Evaluation path is automatic (spec §4.4) → no task needed (constraint registers via decorator).
- Testing matrix (spec §5): enumeration/behavior agreement, eager rejection fires, IIT 3.0 valid passes, IIT 4.0 unconstrained, opt-out, state restore, single source of truth, presets pass → Tasks 1–2 (presets pass is already covered by the existing `TestPresetsPass`, unaffected).
- Docs closeout (spec §7) → Task 3.

**Placeholder scan:** none — every code step shows complete code. Task 3 Step 4 describes a localized prose edit to an existing ROADMAP bullet (its exact current text is long and version-specific); the editor reads the bullet and appends the specified sentence.

**Type consistency:** `compatible_system_partition_schemes` is `ClassVar[frozenset[str] | None]` everywhere (base Protocol, IIT 3.0 frozenset, IIT 4.0 None). The constraint reads it with `getattr(..., None)` and treats `None` as open. `_active_formalism` returns the formalism instance / `None` / `_FORMALISM_UNAVAILABLE`, and both `_compatible_measures` and the new constraint branch on the same two sentinels. The reactive raise reads `IIT3Formalism.compatible_system_partition_schemes` (the same frozenset the registry instance exposes).
