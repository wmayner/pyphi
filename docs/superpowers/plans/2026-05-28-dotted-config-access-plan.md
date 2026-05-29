# Dotted Config Access — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `config.override(...)` accept dotted-path keys and let the dotted-path grammar accept an `iit` / `actual_causation` sub-namespace shorthand, so colliding formalism fields gain an ergonomic scoped-override form.

**Architecture:** A path-normalization step prepends `formalism.` when a dotted path is rooted at a formalism sub-namespace (`iit.version` → `formalism.iit.version`), reused by the existing `__getitem__` / `__setitem__`. `override` gains a positional-only `_paths` mapping merged with `**kwargs`; `_OverrideContext` routes keys containing `.` through `__setitem__` (dotted) and other keys through `setattr` (flat, unchanged). Colliding-field error messages advertise the dotted forms. Additive: flat-colliding writes still raise; routing semantics are untouched.

**Tech Stack:** Python 3.12+, frozen dataclasses, pytest. Run everything with `uv run`.

**Reference spec:** `docs/superpowers/specs/2026-05-28-dotted-config-access-design.md`.

**Settled design (do not relitigate):**
- Grammar: accept BOTH full path (`formalism.iit.version`) and sub-namespace shorthand (`iit.version` / `actual_causation.alpha_measure`).
- `override(self, _paths: Mapping[str, Any] | None = None, /, **kwargs)`; dotted keys route via `__setitem__`, flat keys via `setattr`.
- Flat bare colliding names still RAISE (dotted is the explicit disambiguator).
- No change to `FIELD_TO_LAYER` / `colliding_formalism_fields` semantics.
- Migrate the flat-`version` override sites (`conftest.py`, `test_system_cause_effect_info.py`) to dotted form now (forward-looking; `version` is still IIT-unique today, so this is a behavior-preserving rewrite).

---

## File Structure

**Modified:**
- `pyphi/conf/_global.py` — add `_FORMALISM_SUBNAMESPACES` constant + `_normalize_config_path` helper; apply it in `__getitem__` / `__setitem__`; widen `override` signature; route dotted keys in `_OverrideContext.__enter__`; extend the colliding-field error messages in `__setattr__` / `__getattr__`.
- `test/conftest.py` — migrate the `IIT_4_CONFIG` flat `version=` override to dotted.
- `test/test_system_cause_effect_info.py` — migrate 3 flat `version=` override decorators to dotted.
- `test/test_config_layers.py` — new tests for dotted override + shorthand subscript + error messages.
- `changelog.d/dotted-config-access.feature.md` — new fragment.

All imports needed (`fields`, `FormalismConfig`, `_LAYER_NAMES`, `replace`) already exist in `_global.py` (lines 31, 45, 51, 33).

---

## Task 0: Worktree

**Files:** none.

- [ ] **Step 1:** Create a worktree off `2.0` HEAD via `superpowers:using-git-worktrees`, sibling convention: `git worktree add /Users/will/projects/pyphi-dotted-config -b feature/dotted-config-access 2.0`.
- [ ] **Step 2:** `uv sync --all-extras --all-groups`.
- [ ] **Step 3:** Baseline green: `uv run pytest test/test_config_layers.py -q --no-header`. Expected: all pass.

---

## Task 1: Sub-namespace shorthand in dotted subscript

**Files:**
- Modify: `pyphi/conf/_global.py` (`__getitem__` ≈220, `__setitem__` ≈249; add helper near `_LAYER_NAMES` ≈51)
- Test: `test/test_config_layers.py`

- [ ] **Step 1: Write the failing test.** Append to `test/test_config_layers.py`:

```python
def test_dotted_subscript_subnamespace_shorthand_read():
    from pyphi import config

    assert config["iit.version"] == config["formalism.iit.version"]
    assert (
        config["actual_causation.alpha_measure"]
        == config["formalism.actual_causation.alpha_measure"]
    )


def test_dotted_subscript_subnamespace_shorthand_write():
    from pyphi import config

    original = config["formalism.iit.mechanism_phi_measure"]
    try:
        config["iit.mechanism_phi_measure"] = "EMD"
        assert config["formalism.iit.mechanism_phi_measure"] == "EMD"
    finally:
        config["formalism.iit.mechanism_phi_measure"] = original
```

- [ ] **Step 2: Run it to verify it fails.**

Run: `uv run pytest test/test_config_layers.py -k "shorthand" -q --no-header`
Expected: FAIL — `config["iit.version"]` raises `KeyError` (today `iit` is not a top-level layer, so the multi-part walk does `getattr(config, "iit")` → AttributeError → KeyError).

- [ ] **Step 3: Add the constant + normalization helper.** In `pyphi/conf/_global.py`, after the `_LAYER_TYPES` definition (just before `def _rebuild_nested` at ≈line 60), add:

```python
_FORMALISM_SUBNAMESPACES: frozenset[str] = frozenset(
    f.name for f in fields(FormalismConfig)
)
"""Names of the formalism sub-namespaces ('iit', 'actual_causation') —
recognized as dotted-path roots that imply a leading 'formalism.'."""


def _normalize_config_path(path: str) -> str:
    """Expand a sub-namespace-rooted dotted path to its full layer path.

    ``iit.version`` -> ``formalism.iit.version``;
    ``actual_causation.alpha_measure`` ->
    ``formalism.actual_causation.alpha_measure``. Paths already rooted at a
    top-level layer (or bare leaf names) are returned unchanged.
    """
    head, sep, _rest = path.partition(".")
    if sep and head in _FORMALISM_SUBNAMESPACES:
        return "formalism." + path
    return path
```

- [ ] **Step 4: Apply normalization in `__getitem__`.** In `__getitem__` (≈line 220), make the first line of the body normalize the path:

```python
    def __getitem__(self, path: str) -> Any:
        """..."""  # keep existing docstring
        path = _normalize_config_path(path)
        parts = path.split(".")
        # ... rest unchanged
```

- [ ] **Step 5: Apply normalization in `__setitem__`.** In `__setitem__` (≈line 249), normalize first:

```python
    def __setitem__(self, path: str, value: Any) -> None:
        """..."""  # keep existing docstring
        path = _normalize_config_path(path)
        parts = path.split(".")
        # ... rest unchanged
```

(`__contains__` delegates to `__getitem__`, so it inherits the shorthand automatically — no separate change.)

- [ ] **Step 6: Run the tests to verify they pass.**

Run: `uv run pytest test/test_config_layers.py -k "shorthand" -q --no-header`
Expected: PASS (2 tests).

- [ ] **Step 7: Lint + pyright + commit.**

```bash
uv run ruff check pyphi/conf/_global.py test/test_config_layers.py
uv run ruff format pyphi/conf/_global.py test/test_config_layers.py
uv run pyright pyphi/conf/_global.py
git add pyphi/conf/_global.py test/test_config_layers.py
git -c commit.gpgsign=false commit -m "Recognize formalism sub-namespace roots in dotted config paths"
```

---

## Task 2: `override` accepts dotted keys

**Files:**
- Modify: `pyphi/conf/_global.py` (`override` ≈176, `_OverrideContext.__enter__` ≈451)
- Test: `test/test_config_layers.py`

- [ ] **Step 1: Write the failing test.** Append to `test/test_config_layers.py`:

```python
def test_override_dotted_positional_dict():
    from pyphi import config

    before = config["formalism.iit.version"]
    with config.override({"iit.version": "IIT_3_0"}):
        assert config["formalism.iit.version"] == "IIT_3_0"
    assert config["formalism.iit.version"] == before


def test_override_dotted_via_kwargs():
    from pyphi import config

    before = config["formalism.iit.version"]
    with config.override(**{"iit.version": "IIT_3_0"}):
        assert config["formalism.iit.version"] == "IIT_3_0"
    assert config["formalism.iit.version"] == before


def test_override_mixed_dotted_and_flat():
    from pyphi import config

    before_v = config["formalism.iit.version"]
    before_p = config["numerics.precision"]
    with config.override({"iit.version": "IIT_3_0"}, precision=6):
        assert config["formalism.iit.version"] == "IIT_3_0"
        assert config["numerics.precision"] == 6
    assert config["formalism.iit.version"] == before_v
    assert config["numerics.precision"] == before_p
```

- [ ] **Step 2: Run them to verify they fail.**

Run: `uv run pytest test/test_config_layers.py -k "override_dotted or override_mixed" -q --no-header`
Expected: FAIL — `override` has no positional param (`TypeError`), and dotted kwargs route through `setattr` → `ConfigurationError`/`AttributeError` on `iit.version`.

- [ ] **Step 3: Widen the `override` signature.** Replace `override` (≈line 176):

```python
    def override(
        self, _paths: Mapping[str, Any] | None = None, /, **kwargs: Any
    ) -> _OverrideContext:
        """Scoped override of one or more config fields.

        Returns a :class:`contextlib.ContextDecorator` — usable as
        ``with config.override(...):`` or ``@config.override(...)``.

        Accepts flat layered names (``precision=6``), legacy uppercase names
        (``PRECISION=6``), and dotted paths via the positional mapping or
        kwargs (``override({"iit.version": "IIT_3_0"})`` or
        ``override(**{"iit.version": "IIT_3_0"})``). Dotted paths accept the
        sub-namespace shorthand (``iit.version``) or the full path
        (``formalism.iit.version``). Unknown names raise
        :class:`ConfigurationError`.
        """
        merged: dict[str, Any] = dict(_paths) if _paths else {}
        merged.update(kwargs)
        return _OverrideContext(self, merged)
```

Add `from collections.abc import Mapping` to the imports if not present (the module imports `Iterator` from `collections.abc` at line 29 — add `Mapping` alongside).

- [ ] **Step 4: Route dotted keys in `_OverrideContext.__enter__`.** Replace the loop in `__enter__` (≈line 453):

```python
    def __enter__(self) -> _OverrideContext:
        self._saved = self._config.snapshot()
        for name, value in self._new_values.items():
            if "." in name:
                self._config[name] = value
            else:
                setattr(self._config, name, value)
        return self
```

- [ ] **Step 5: Run the tests to verify they pass.**

Run: `uv run pytest test/test_config_layers.py -k "override_dotted or override_mixed" -q --no-header`
Expected: PASS (3 tests).

- [ ] **Step 6: Lint + pyright + commit.**

```bash
uv run ruff check pyphi/conf/_global.py test/test_config_layers.py
uv run ruff format pyphi/conf/_global.py test/test_config_layers.py
uv run pyright pyphi/conf/_global.py
git add pyphi/conf/_global.py test/test_config_layers.py
git -c commit.gpgsign=false commit -m "Accept dotted-path keys in config.override"
```

---

## Task 3: Colliding-field error messages mention the dotted forms

**Files:**
- Modify: `pyphi/conf/_global.py` (`__setattr__` collision branch ≈382, `__getattr__` collision branch ≈345)
- Test: `test/test_config_layers.py`

- [ ] **Step 1: Write the failing test.** `mechanism_partition_scheme` is the existing colliding field. Append:

```python
def test_colliding_setattr_error_mentions_dotted_form():
    import pytest

    from pyphi import config
    from pyphi.conf._field_routing import ConfigurationError

    with pytest.raises(ConfigurationError) as exc:
        config.mechanism_partition_scheme = "JOINT_BIPARTITION"
    msg = str(exc.value)
    assert "override(" in msg
    assert "iit.mechanism_partition_scheme" in msg


def test_colliding_getattr_error_mentions_dotted_form():
    import pytest

    from pyphi import config

    with pytest.raises(AttributeError) as exc:
        _ = config.mechanism_partition_scheme
    assert "iit.mechanism_partition_scheme" in str(exc.value)
```

- [ ] **Step 2: Run them to verify they fail.**

Run: `uv run pytest test/test_config_layers.py -k "error_mentions_dotted" -q --no-header`
Expected: FAIL — current messages mention `replace(...)` / `config.formalism.iit.X` but not `override(` or `config["iit.X"]`.

- [ ] **Step 3: Extend the `__setattr__` collision message.** Replace the `ConfigurationError` raised in the colliding branch (≈line 382):

```python
        if field_name in colliding_formalism_fields():
            raise ConfigurationError(
                f"Field {field_name!r} is ambiguous (exists in both "
                "formalism.iit and formalism.actual_causation). Qualify it: "
                f'config["iit.{field_name}"] = ... (or '
                f'"actual_causation.{field_name}"), '
                f'config.override({{"iit.{field_name}": ...}}), or set the '
                f"sub-namespace wholesale via config.iit = "
                f"replace(config.formalism.iit, {field_name}=...)."
            )
```

- [ ] **Step 4: Extend the `__getattr__` collision message.** Replace the `AttributeError` raised in the colliding branch (≈line 345):

```python
        if name in colliding_formalism_fields():
            raise AttributeError(
                f"{name!r} is ambiguous (exists in both formalism.iit and "
                "formalism.actual_causation). Use a qualified form: "
                f"config.formalism.iit.{name}, config[\"iit.{name}\"], or "
                f"config[\"actual_causation.{name}\"]."
            )
```

- [ ] **Step 5: Run the tests to verify they pass.**

Run: `uv run pytest test/test_config_layers.py -k "error_mentions_dotted" -q --no-header`
Expected: PASS (2 tests).

- [ ] **Step 6: Run the full `test_config_layers.py` (no regressions in existing collision tests).**

Run: `uv run pytest test/test_config_layers.py -q --no-header`
Expected: all pass.

- [ ] **Step 7: Lint + commit.**

```bash
uv run ruff check pyphi/conf/_global.py test/test_config_layers.py
uv run ruff format pyphi/conf/_global.py test/test_config_layers.py
git add pyphi/conf/_global.py test/test_config_layers.py
git -c commit.gpgsign=false commit -m "Point colliding-field config errors at the dotted forms"
```

---

## Task 4: Migrate flat-`version` override sites to dotted

**Files:**
- Modify: `test/conftest.py` (`IIT_4_CONFIG` ≈46), `test/test_system_cause_effect_info.py` (3 decorators ≈11, 24, 37)

- [ ] **Step 1: Migrate `conftest.py`.** Change `IIT_4_CONFIG` (≈line 46) so only `version` moves to the dotted positional mapping; the IIT-unique fields stay flat:

```python
IIT_4_CONFIG = config.override(
    {"iit.version": "IIT_4_0_2023"},
    mechanism_phi_measure="GENERALIZED_INTRINSIC_DIFFERENCE",
    system_phi_measure="GENERALIZED_INTRINSIC_DIFFERENCE",
    system_partition_scheme="DIRECTED_SET_PARTITION",
)
```

- [ ] **Step 2: Migrate `test_system_cause_effect_info.py`.** Change each of the 3 decorators (≈lines 11, 24, 37) from `@config.override(version="IIT_3_0", mechanism_phi_measure="EMD")` to:

```python
@config.override({"iit.version": "IIT_3_0"}, mechanism_phi_measure="EMD")
```

- [ ] **Step 3: Run the affected suites.**

Run: `uv run pytest test/test_system_cause_effect_info.py -q --no-header`
Expected: same pass/skip counts as baseline (behavior-preserving — `version` is still IIT-unique today, so flat and dotted resolve identically).

- [ ] **Step 4: Confirm conftest still imports (whole suite collects).**

Run: `uv run pytest test/ -q --no-header --co > /dev/null && echo COLLECT_OK`
Expected: prints `COLLECT_OK` (collection succeeds; the module-load `IIT_4_CONFIG` override didn't raise).

- [ ] **Step 5: Lint + commit.**

```bash
uv run ruff check test/conftest.py test/test_system_cause_effect_info.py
uv run ruff format test/conftest.py test/test_system_cause_effect_info.py
git add test/conftest.py test/test_system_cause_effect_info.py
git -c commit.gpgsign=false commit -m "Migrate flat version= config overrides to dotted form"
```

---

## Task 5: Changelog + full verification

**Files:**
- Create: `changelog.d/dotted-config-access.feature.md`

- [ ] **Step 1: Create the changelog fragment.**

```
``config.override(...)`` now accepts dotted-path keys, and dotted config paths accept an ``iit`` / ``actual_causation`` sub-namespace shorthand: ``config.override({"iit.version": "IIT_3_0"})`` and ``config["iit.version"]`` both resolve to ``config.formalism.iit.version``. This gives colliding formalism fields (names present in both the IIT and actual-causation sub-namespaces) an ergonomic scoped-override form. Flat writes to a colliding bare name still raise, now pointing at the dotted forms.
```

Write it: `printf '%s\n' "<text>" > changelog.d/dotted-config-access.feature.md` (or `uv run towncrier create`).

- [ ] **Step 2: Verify towncrier renders.**

Run: `uv run towncrier build --draft --version 2.0 2>&1 | grep -i "dotted-path keys"`
Expected: the fragment line appears.

- [ ] **Step 3: Full `uv run pytest` (no path — doctests).**

Run: `uv run pytest -q --no-header`
Expected: green (baseline counts plus the new `test_config_layers.py` tests).

- [ ] **Step 4: Commit.**

```bash
git add changelog.d/dotted-config-access.feature.md
git -c commit.gpgsign=false commit -m "Changelog for dotted config access"
```

---

## Acceptance gate

- `config.override({"iit.version": "IIT_3_0"})` and `config.override(**{"iit.version": ...})` set `config.formalism.iit.version` within scope and restore on exit; mixed dotted + flat works.
- `config["iit.version"]` / `config["actual_causation.alpha_measure"]` read and write equivalently to the full paths.
- Colliding-field flat write/read still raise, with messages naming the dotted forms.
- `conftest.py` + `test_system_cause_effect_info.py` migrated; whole suite collects and passes.
- Full `uv run pytest` (no path) green; pyright clean; ruff clean.
- Changelog fragment added.

## Self-review notes

- **Spec coverage:** sub-namespace roots (Task 1) ✓; `override` dotted keys + signature (Task 2) ✓; error messages (Task 3) ✓; migration (Task 4) ✓; changelog (Task 5) ✓; non-goals respected (flat colliding still raises — Task 3 tests assert the raise; routing semantics untouched) ✓.
- **Naming/signature consistency:** `_normalize_config_path`, `_FORMALISM_SUBNAMESPACES`, and `override(_paths, /, **kwargs)` used identically across Tasks 1–2.
- **Verification point flagged:** confirm `Mapping` is added to the `collections.abc` import in Task 2 Step 3 (the module imports `Iterator` from there at line 29).
