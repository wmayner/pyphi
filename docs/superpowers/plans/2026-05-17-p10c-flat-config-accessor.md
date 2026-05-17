# Flat Config Accessor — Mapping Protocol Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the Mapping protocol on `pyphi.config` so callers can iterate, check containment, query, and round-trip the full config surface via standard dict idioms.

**Architecture:** Add seven Mapping-protocol methods plus a `_iter_leaf_paths` helper to `_GlobalConfig` in `pyphi/conf/_global.py`. Existing `__getitem__` is extended to accept bare leaf keys (`config["precision"]`) via the same `FIELD_TO_LAYER` routing already used by `__getattr__`. No ABC inheritance — duck-typed protocol only.

**Tech Stack:** PyPhi 2.0, Python 3.13+, frozen dataclasses (`numerics.NumericsConfig`, `formalism.FormalismConfig` with nested `IITConfig` + `ActualCausationConfig`, `infrastructure.InfrastructureConfig`).

**Spec:** `docs/superpowers/specs/2026-05-17-p10c-flat-config-accessor-design.md` (committed at `05f35a92`).

---

## Working constraints (apply at all times)

- Pre-commit hooks (ruff + ruff format + pyright + towncrier-check) gate the commit. NEVER bypass with `--no-verify` or `SKIP=*`. If a hook fails, run `uv run ruff check <file>` / `uv run pyright <file>` directly to diagnose and fix the root cause.
- gpgsign: use `git -c commit.gpgsign=false commit ...`. If it fails with "1Password: agent returned an error", surface to the controller — do not auto-bypass.
- Do NOT push.
- **Targeted `git add` only**: stage ONLY the 3 files this plan touches (`pyphi/conf/_global.py`, `test/test_config_layers.py`, `changelog.d/config-mapping-protocol.feature.md`). The repo has ~18 unrelated unstaged tracked-file mods and ~20 untracked items that must NOT be staged.
- **Before each commit**: run `git diff --cached --stat` and confirm only intended files are staged. **After each commit**: run `git show --stat HEAD` to confirm.
- After pre-commit auto-fix (ruff format may rewrite files), the staged version may be updated. If commit fails post-format, `git add -u pyphi/conf/_global.py test/test_config_layers.py changelog.d/config-mapping-protocol.feature.md` and retry.
- No P# markers / "Phase A" / "Cluster A" / `TODO(Px)` / "Task N" / ROADMAP IDs in source/comments/docstrings/changelog. Commit message MAY reference "the spec" (a real doc path).
- No design-narrative in docstrings. Keep them precise about what the methods do.
- No back-compat shims.

---

## File map

| File | Change |
|---|---|
| `pyphi/conf/_global.py` | Add 2 imports (`Iterator`, `is_dataclass`); add `_iter_leaf_paths` module-level helper; extend existing `__getitem__` to route bare leaf keys via `FIELD_TO_LAYER`; add 7 new methods on `_GlobalConfig` (`__iter__`, `__contains__`, `__len__`, `keys`, `values`, `items`, `get`). |
| `test/test_config_layers.py` | Add ~12 tests covering the new Mapping surface and the bare-leaf-key extension. |
| `changelog.d/config-mapping-protocol.feature.md` | New fragment summarizing the Mapping-protocol additions. |

---

## Task 1: Mapping protocol completion (single commit)

**Files:**
- Modify: `pyphi/conf/_global.py`
- Test: `test/test_config_layers.py`
- Create: `changelog.d/config-mapping-protocol.feature.md`

---

### Step 1: Write the failing tests

Add to the bottom of `test/test_config_layers.py` (the existing file has flat-accessor tests starting ~line 372; add a new test class or grouped functions at end). Use the existing import block for `config`; the new tests don't need fixtures beyond the live default config.

```python
# ---------------------------------------------------------------------------
# Mapping protocol surface
# ---------------------------------------------------------------------------


class TestConfigMappingProtocol:
    """Pin the Mapping protocol behavior on the global config facade."""

    def test_iter_yields_all_leaves(self):
        """iter(config) yields every leaf as a dotted path in declaration order."""
        keys = list(config)
        # Representative leaves from each layer must appear.
        assert "numerics.precision" in keys
        assert "formalism.iit.version" in keys
        assert "formalism.iit.mechanism_phi_measure" in keys
        assert "formalism.actual_causation.alpha_measure" in keys
        assert "infrastructure.parallel" in keys
        # No layer-level keys (only leaves).
        assert "numerics" not in keys
        assert "formalism" not in keys
        assert "formalism.iit" not in keys

    def test_len_matches_iter(self):
        """len(config) equals the number of paths yielded by iteration."""
        assert len(config) == len(list(iter(config)))
        assert len(config) > 0

    def test_contains_dotted_path(self):
        """'numerics.precision' in config is True."""
        assert "numerics.precision" in config
        assert "formalism.iit.mechanism_phi_measure" in config
        assert "infrastructure.parallel" in config

    def test_contains_bare_leaf(self):
        """A bare leaf name resolves via FIELD_TO_LAYER routing."""
        assert "precision" in config
        assert "mechanism_phi_measure" in config

    def test_contains_invalid_path(self):
        """An unknown path is not contained."""
        assert "foo.bar" not in config
        assert "numerics.nonexistent" not in config
        assert "nonexistent" not in config

    def test_contains_non_string(self):
        """Non-string keys are not contained (and do not raise)."""
        assert 42 not in config
        assert None not in config
        assert ("numerics", "precision") not in config

    def test_get_existing_dotted_path(self):
        """config.get('numerics.precision') matches attribute access."""
        assert config.get("numerics.precision") == config.numerics.precision

    def test_get_existing_bare_leaf(self):
        """config.get('precision') routes via FIELD_TO_LAYER."""
        assert config.get("precision") == config.numerics.precision

    def test_get_missing_returns_default(self):
        """config.get on a missing path returns the default."""
        assert config.get("nonexistent", 42) == 42
        assert config.get("foo.bar.baz") is None

    def test_keys_values_items_consistent(self):
        """keys, values, and items agree on order and content."""
        keys = config.keys()
        values = config.values()
        items = config.items()
        assert len(keys) == len(values) == len(items)
        for i, key in enumerate(keys):
            assert items[i] == (key, values[i])
            assert config[key] == values[i]

    def test_items_round_trip(self):
        """Capturing items, mutating, and restoring via __setitem__ recovers state."""
        original_precision = config.numerics.precision
        captured = dict(config.items())
        try:
            config["numerics.precision"] = 99
            assert config["numerics.precision"] == 99
            # Restore from captured items.
            for path, value in captured.items():
                config[path] = value
            assert config.numerics.precision == original_precision
        finally:
            # Belt-and-suspenders: ensure restored even if assertion above fired.
            config["numerics.precision"] = original_precision

    def test_getitem_bare_leaf(self):
        """config['precision'] returns config.numerics.precision."""
        assert config["precision"] == config.numerics.precision
        assert config["mechanism_phi_measure"] == config.formalism.iit.mechanism_phi_measure

    def test_getitem_unknown_bare_leaf(self):
        """An unknown bare leaf key raises KeyError."""
        import pytest

        with pytest.raises(KeyError, match="Unknown config path"):
            config["nonexistent"]
```

(`pytest` is imported once inside `test_getitem_unknown_bare_leaf`; if the test file already imports it at module level, move the import there.)

---

### Step 2: Run the failing tests

Run: `uv run pytest test/test_config_layers.py::TestConfigMappingProtocol -v`
Expected: all tests FAIL — methods not defined / bare leaf key returns KeyError where it shouldn't.

---

### Step 3: Add imports to `pyphi/conf/_global.py`

At the top of `pyphi/conf/_global.py`, after the existing `from __future__ import annotations` block, add:

```python
from collections.abc import Iterator
from dataclasses import is_dataclass
```

`fields` is already imported (verified at line 30).

Place `Iterator` after the existing `import contextlib` and before the `dataclasses` imports; place `is_dataclass` next to the existing `from dataclasses import fields` import. Concretely the import block should look like:

```python
import contextlib
from collections.abc import Iterator
from dataclasses import asdict
from dataclasses import fields
from dataclasses import is_dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any
```

---

### Step 4: Add `_iter_leaf_paths` helper

In `pyphi/conf/_global.py`, add this module-level helper after the existing `_read_via_target` helper (around line 67-95) and before the `class _GlobalConfig:` definition (around line 99):

```python
def _iter_leaf_paths(dc_instance: Any, prefix: str) -> Iterator[str]:
    """Yield dotted leaf paths under ``prefix`` for a dataclass instance.

    For a nested dataclass field (e.g. ``FormalismConfig.iit``), recurses
    one level into the sub-dataclass. For flat layers (numerics,
    infrastructure), yields one path per field directly.
    """
    for field in fields(dc_instance):
        attr = getattr(dc_instance, field.name)
        if is_dataclass(attr) and not isinstance(attr, type):
            sub_prefix = f"{prefix}.{field.name}"
            for sub_field in fields(attr):
                yield f"{sub_prefix}.{sub_field.name}"
        else:
            yield f"{prefix}.{field.name}"
```

The `is_dataclass(attr) and not isinstance(attr, type)` check distinguishes dataclass instances from dataclass classes (the latter would happen if a default is a class object — defensive guard).

---

### Step 5: Extend existing `__getitem__` to route bare leaf keys

Find the existing `__getitem__` method on `_GlobalConfig` (around line 201-217 in `pyphi/conf/_global.py`). Replace its body with:

```python
def __getitem__(self, path: str) -> Any:
    """Read a config field by dotted path or by bare leaf name.

    Dotted paths address layered fields:
    ``config["numerics.precision"]``,
    ``config["formalism.iit.mechanism_phi_measure"]``,
    ``config["infrastructure.parallel"]``.

    Bare leaf names route through ``FIELD_TO_LAYER`` to the owning layer:
    ``config["precision"]`` returns ``config.numerics.precision``.
    """
    parts = path.split(".")
    if not parts or not all(parts):
        raise KeyError(f"Invalid config path: {path!r}")

    if len(parts) == 1:
        leaf = parts[0]
        if leaf in FIELD_TO_LAYER:
            return _read_via_target(self, FIELD_TO_LAYER[leaf], leaf)
        raise KeyError(f"Unknown config path: {path!r}")

    obj: Any = self
    for p in parts:
        try:
            obj = getattr(obj, p)
        except AttributeError as exc:
            raise KeyError(f"Unknown config path: {path!r}") from exc
    return obj
```

The `__setitem__` method below it is unchanged (bare leaf writes go through `__setattr__`'s existing top-level routing, not through `__setitem__`).

---

### Step 6: Add the seven new Mapping-protocol methods on `_GlobalConfig`

Add these methods on `_GlobalConfig` after the existing `__setitem__` method (around line 240) and before the existing `__repr__` method. Order: `__iter__`, `__contains__`, `__len__`, `keys`, `values`, `items`, `get`.

```python
def __iter__(self) -> Iterator[str]:
    """Yield every leaf field as a dotted path in dataclass declaration order.

    Order: numerics fields, formalism fields (iit then actual_causation
    leaves, in declaration order), infrastructure fields.
    """
    yield from _iter_leaf_paths(self._numerics, "numerics")
    yield from _iter_leaf_paths(self._formalism, "formalism")
    yield from _iter_leaf_paths(self._infrastructure, "infrastructure")


def __contains__(self, path: object) -> bool:
    """Return True if ``path`` resolves to a leaf via ``__getitem__``."""
    if not isinstance(path, str):
        return False
    try:
        self[path]
    except KeyError:
        return False
    return True


def __len__(self) -> int:
    """Return the number of leaf fields across all layers."""
    return sum(1 for _ in self)


def keys(self) -> list[str]:
    """Return a list of dotted leaf paths in declaration order."""
    return list(self)


def values(self) -> list[Any]:
    """Return a list of leaf values in declaration order."""
    return [self[k] for k in self]


def items(self) -> list[tuple[str, Any]]:
    """Return a list of ``(path, value)`` pairs in declaration order."""
    return [(k, self[k]) for k in self]


def get(self, path: str, default: Any = None) -> Any:
    """Return ``self[path]`` or ``default`` if the path doesn't resolve."""
    try:
        return self[path]
    except KeyError:
        return default
```

---

### Step 7: Run the tests to verify they pass

Run: `uv run pytest test/test_config_layers.py::TestConfigMappingProtocol -v`
Expected: 12 passed.

If any test fails:
- `test_iter_yields_all_leaves`: verify `_iter_leaf_paths` recurses correctly into `FormalismConfig.iit` and `FormalismConfig.actual_causation` (both are sub-dataclasses).
- `test_contains_bare_leaf` / `test_getitem_bare_leaf`: verify `__getitem__` Step 5 update correctly routes single-part paths via `FIELD_TO_LAYER`.
- `test_items_round_trip`: verify `__setitem__` accepts the round-tripped values (some values like `ConfigSnapshot` or callables may not round-trip via setitem — restrict the round-trip test if needed).

---

### Step 8: Run pyright and ruff

Run: `uv run pyright pyphi/conf/_global.py test/test_config_layers.py`
Expected: 0 errors / 0 warnings on these files.

Run: `uv run ruff check pyphi/conf/_global.py test/test_config_layers.py`
Run: `uv run ruff format --check pyphi/conf/_global.py test/test_config_layers.py`
Expected: clean.

---

### Step 9: Run the full fast lane to confirm no regressions

Run: `uv run pytest test/ -m "not slow" -q --no-header 2>&1 | tail -3`
Expected: 1278+ passed / 0 failures / 3 xfailed.

The Mapping additions are non-invasive (new methods, extended `__getitem__` only for bare leaf keys); regressions are unlikely but the fast lane confirms.

---

### Step 10: Run the end-to-end smoke test

```bash
uv run python -c "
import pyphi
config = pyphi.config

# Enumeration
all_keys = list(config)
print(f'Found {len(config)} leaf settings')
assert 'numerics.precision' in all_keys
assert 'formalism.iit.mechanism_phi_measure' in all_keys
assert 'formalism.actual_causation.alpha_measure' in all_keys
assert 'infrastructure.parallel' in all_keys

# Mapping surface
assert 'numerics.precision' in config
assert 'precision' in config   # bare leaf shortcut
assert 'nonexistent' not in config
assert config.get('precision') == config.numerics.precision
assert config.get('nope', 42) == 42

# Round-trip
captured = dict(config.items())
assert len(captured) == len(config)
print('Mapping protocol round-trip OK')
"
```

Expected: prints "Found N leaf settings" with N ≥ ~40, then "Mapping protocol round-trip OK".

---

### Step 11: Add the changelog fragment

Create `changelog.d/config-mapping-protocol.feature.md`:

```markdown
The :data:`pyphi.config` facade now implements the Mapping protocol:
:func:`iter`, :func:`len`, ``in``, ``keys()``, ``values()``, ``items()``,
and ``get(path, default)``. Callers can enumerate all leaf settings as
dotted paths in dataclass declaration order::

    >>> for path in pyphi.config:
    ...     print(path, "=", pyphi.config[path])
    numerics.precision = 13
    formalism.iit.version = IIT_4_0_2023
    ...

The existing ``config[path]`` indexer now also accepts bare leaf keys
(``config["precision"]``) as a shortcut for the fully-qualified path
(``config["numerics.precision"]``), mirroring the routing that ``config.precision``
attribute access already used.
```

---

### Step 12: Stage and commit

```bash
git add pyphi/conf/_global.py test/test_config_layers.py changelog.d/config-mapping-protocol.feature.md
git diff --cached --stat   # confirm ONLY 3 files staged
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
Complete the Mapping protocol on pyphi.config

The dotted-path indexer (config["numerics.precision"]) and writer were
already in place; this adds __iter__, __contains__, __len__, keys,
values, items, and get(path, default) so callers can enumerate the
full config surface as standard Python dict idioms. The indexer is
extended to accept bare leaf keys (config["precision"]) via the
existing FIELD_TO_LAYER routing, mirroring __getattr__'s shortcut.

Iteration order is dataclass declaration order across the three
layers: numerics, formalism (iit then actual_causation leaves),
infrastructure.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
git show --stat HEAD
```

Expected: 3 files committed.

---

## Acceptance gates summary

After Task 1 lands:

```bash
uv run pytest test/test_config_layers.py::TestConfigMappingProtocol -v   # 12 passed
uv run pytest test/test_config_layers.py -x -q                            # ~25-30 passed
uv run pytest test/ -m "not slow" -q                                      # 1278+/0 failures
uv run pyright pyphi/                                                     # 0 errors / 1 baseline warning
uv run ruff check pyphi/ test/                                            # clean
uv run ruff format --check pyphi/ test/                                   # clean
```

End-to-end smoke test (Step 10) prints "Mapping protocol round-trip OK".

---

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| `_iter_leaf_paths` recursion misses a sub-dataclass level | Low | `FormalismConfig` is the only two-level case; `test_iter_yields_all_leaves` pins the four representative leaf paths covering all layers. |
| Bare leaf key collision (a future leaf shares a layer name) | Low | Existing `colliding_formalism_fields()` validator prevents leaf/layer name collisions. |
| `items()` round-trip via `__setitem__` fails on non-trivial values (e.g., `ConfigSnapshot`, callables) | Medium | `test_items_round_trip` only mutates `numerics.precision` (an int). If a config layer grows non-trivial fields in the future, narrow the test to skip those keys. |
| Pyright flags `__contains__(object)` signature mismatch with Mapping protocol | Low | Duck-typed Mapping — no inheritance enforces signatures. The `object` parameter type is conventional for `__contains__` (accept any input, narrow to str internally). |
| `keys()` / `values()` / `items()` returning lists instead of views breaks a caller expecting views | Very low | No current caller uses these methods; future callers can adapt. Returning lists is simpler and matches the spec. |
| Iteration order divergence from `__repr__` YAML dump | Low | Both use dataclass field declaration order via `fields()`. `test_iter_yields_all_leaves` pins representative paths in the expected layer order. |

---

## Sequencing

P10c is independent of all other open 2.0 work. Lands in a single
commit. ~2 hours of work end-to-end.
