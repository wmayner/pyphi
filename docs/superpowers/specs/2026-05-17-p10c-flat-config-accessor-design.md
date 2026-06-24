# Flat dotted-string config accessor — Mapping protocol completion

## Summary

Complete the dict-style flat dotted-string accessor on
:data:`pyphi.config` by adding the Mapping protocol methods that are
currently missing. The dotted-path reader / writer
(``config["numerics.precision"]`` / ``config["numerics.precision"] = 6``)
already exists; this spec adds iteration, containment, length, and a
``get(key, default)`` method so callers can enumerate, query, and
round-trip the full config surface programmatically. ``__getitem__`` is
also extended to accept bare leaf keys (``config["precision"]``) as a
shortcut for ``config["numerics.precision"]``, mirroring the existing
top-level read routing on ``__getattr__``.

## Background

### Today's surface

The :class:`pyphi.conf._global._GlobalConfig` facade already exposes:

- ``config.numerics.precision`` — attribute traversal of the layered
  dataclass.
- ``config.precision`` — top-level read via ``__getattr__``'s
  ``FIELD_TO_LAYER`` routing (unique leaf names route to their owning
  layer).
- ``config["numerics.precision"]`` — dotted-path read.
- ``config["formalism.iit.mechanism_phi_measure"]`` — three-layer
  dotted-path read.
- ``config["numerics.precision"] = 6`` — dotted-path write.
- ``config.snapshot()`` / ``config.install_snapshot()`` /
  ``config.override(**kwargs)`` — snapshot capture, restore, and scoped
  override.

Coverage: 18 tests under ``test/test_config_layers.py`` exercise the
dotted-path reader and writer including KeyError on invalid paths.

### What's missing

The Mapping protocol is incomplete. Today these all fail or behave
incorrectly:

- ``"numerics.precision" in config`` — ``__contains__`` falls through to
  the default ``object`` behavior, which (because ``_GlobalConfig``
  defines ``__getitem__``) is implemented but Python's fallback is
  inefficient and quirky.
- ``for path in config`` — no ``__iter__``; iterates nothing useful.
- ``config.keys()`` / ``.items()`` / ``.values()`` — ``AttributeError``.
- ``len(config)`` — ``TypeError``.
- ``config.get("precision", default)`` — ``AttributeError``.
- ``config["precision"]`` (bare leaf, no dot) — ``KeyError`` (the
  reader rejects non-dotted paths).

Programmatic enumeration of all config keys (one of the roadmap-stated
use cases) is not possible.

## Goals

- Complete the Mapping protocol on ``_GlobalConfig`` so the config can
  be iterated, queried, and round-tripped via standard dict idioms.
- Extend ``__getitem__`` to accept bare leaf keys, matching the
  ergonomic shortcut that ``__getattr__`` already provides.
- Preserve all existing behavior: attribute access, dotted-path read /
  write, top-level read / write via ``__setattr__``, override /
  snapshot / install_snapshot.
- Keep internal storage on the layered frozen dataclasses (type
  checking, validators, ``ConfigSnapshot``).

## Non-goals

- No ``collections.abc.Mapping`` inheritance. Duck-typed protocol only;
  no ABC overhead, no enforced method signatures.
- No new YAML-format support. The layered nested YAML format is
  unchanged; this spec doesn't add flat dotted keys to ``pyphi_config.yml``.
- No write-time validator changes. ``__setitem__`` already routes
  through ``_rebuild_nested`` which goes through the dataclass
  constructors; existing validators stay.
- No back-compat shims for the missing methods (they aren't currently
  callable anywhere in the codebase).

## Design

### Method surface added to ``_GlobalConfig``

All in ``pyphi/conf/_global.py``.

```python
def __iter__(self) -> Iterator[str]:
    """Yield every leaf field as a dotted path in dataclass declaration order.

    For the nested formalism layer (formalism.iit, formalism.actual_causation),
    yields one path per leaf inside each sub-layer.
    """
    yield from _iter_leaf_paths(self._numerics, "numerics")
    yield from _iter_leaf_paths(self._formalism, "formalism")
    yield from _iter_leaf_paths(self._infrastructure, "infrastructure")


def __contains__(self, path: object) -> bool:
    """Return True if path resolves to a leaf via __getitem__."""
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

### Helper function

Added at module scope in ``pyphi/conf/_global.py``:

```python
def _iter_leaf_paths(dc_instance: Any, prefix: str) -> Iterator[str]:
    """Yield dotted leaf paths under ``prefix`` for a dataclass instance.

    For ``FormalismConfig``, recurses one level into the nested ``iit`` /
    ``actual_causation`` sub-dataclasses. For flat layers (numerics,
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

The recursion is exactly one level — the layered config has at most
two levels of dataclass nesting (top-level ``FormalismConfig`` plus its
``iit`` and ``actual_causation`` sub-dataclasses). ``NumericsConfig``
and ``InfrastructureConfig`` have no sub-dataclass fields.

### ``__getitem__`` extension

Today ``__getitem__`` rejects non-dotted paths (single-part paths fall
through to ``KeyError("Unknown config path:")``). Update so a single-
part path looks up in ``FIELD_TO_LAYER`` and routes to the owning layer:

```python
def __getitem__(self, path: str) -> Any:
    parts = path.split(".")
    if not parts or not all(parts):
        raise KeyError(f"Invalid config path: {path!r}")

    # Bare leaf key: route via FIELD_TO_LAYER if it's a unique leaf name.
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

``__setitem__`` is unchanged — bare leaf write was already handled via
``__setattr__``'s top-level routing.

### Iteration order

Dataclass field declaration order. The three layers are yielded in the
order: ``numerics``, ``formalism`` (with ``iit`` before
``actual_causation`` per ``FormalismConfig``'s field order),
``infrastructure``. Within each (sub-)dataclass, fields are yielded in
declaration order.

Stable across runs. Matches the order used by ``_as_nested_dict`` and
``__repr__`` so the YAML dump and the flat enumeration agree.

### Imports

Add to the top of ``pyphi/conf/_global.py``:

```python
from collections.abc import Iterator
from dataclasses import fields
from dataclasses import is_dataclass
```

(``fields`` and ``is_dataclass`` may already be imported for other
helpers; check and don't duplicate.)

## Testing

Add to ``test/test_config_layers.py`` (or a new sibling test module if
the file is unwieldy; check size first). Each test exercises one Mapping-
protocol method:

- ``test_config_iter_yields_all_leaves`` — pin the set of paths, including
  representative numerics, formalism.iit, formalism.actual_causation,
  infrastructure leaves.
- ``test_config_len_matches_iter`` — ``len(config) == len(list(iter(config)))``.
- ``test_config_contains_dotted_path`` — ``"numerics.precision" in config``.
- ``test_config_contains_bare_leaf`` — ``"precision" in config``.
- ``test_config_contains_invalid_path`` — ``"foo.bar" not in config``.
- ``test_config_contains_non_string`` — ``42 not in config``.
- ``test_config_get_existing_path`` — ``config.get("precision") == config.numerics.precision``.
- ``test_config_get_missing_returns_default`` — ``config.get("nonexistent", 42) == 42``.
- ``test_config_keys_values_items_consistent`` — ``zip(config.keys(), config.values()) == config.items()``.
- ``test_config_items_round_trip`` — capture items, mutate global, restore via ``config[k] = v``, verify state.
- ``test_config_getitem_bare_leaf`` — ``config["precision"] == config.numerics.precision``.
- ``test_config_getitem_unknown_bare_leaf`` — ``config["nonexistent"]`` raises ``KeyError``.

All tests run under the live default config (no need for ``presets``
overrides — they exercise the Mapping surface, not formalism behavior).

## Acceptance gates

After implementation:

```bash
uv run pytest test/test_config_layers.py -x -q   # ~25-30 passed
uv run pytest test/ -m "not slow" -q             # full fast lane unchanged
uv run pyright pyphi/conf/_global.py test/test_config_layers.py
uv run ruff check pyphi/conf/_global.py test/test_config_layers.py
```

Expected: all green, pyright 0 errors / 1 baseline warning, ruff clean.

End-to-end smoke test:

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

## File map

| File | Change |
|---|---|
| ``pyphi/conf/_global.py`` | Add ``_iter_leaf_paths`` helper; add ``__iter__`` / ``__contains__`` / ``__len__`` / ``keys`` / ``values`` / ``items`` / ``get`` methods; extend ``__getitem__`` to route bare leaf keys via ``FIELD_TO_LAYER``. |
| ``test/test_config_layers.py`` | Add ~12 tests covering the new Mapping surface. |
| ``changelog.d/config-mapping-protocol.feature.md`` | Changelog fragment summarizing the new surface. |

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Iteration order divergence from YAML dump order | Low | Both use dataclass field order; pinned by the iter test. |
| Bare leaf key ambiguity (a future field name colliding with a layer name) | Low | Existing ``colliding_formalism_fields()`` validator already prevents this. |
| Performance: ``len(config)`` does full iteration on each call | Low | Config has ~50 leaves; iteration is microseconds. If hot, cache later. |
| ``__contains__`` swallowing legitimate non-KeyError exceptions during ``__getitem__`` | Low | Only ``KeyError`` is caught; other exceptions propagate. |
| Test additions inflate ``test_config_layers.py`` beyond comfortable | Low | File is currently ~410 lines; +12 small tests fit. If file grows past 600 lines, split later. |

## Effort

~2 hours: ~30 min implementation, ~45 min tests, ~15 min changelog, ~30
min verification + commit. Single commit.

## Sequencing

P10c is independent of all other open 2.0 work. Can land any time. The
flat accessor pairs well with P15 (jsonify retirement / CLI overrides)
since CLI tools benefit from dotted-key access, but doesn't block P15.

## Open questions

None — all design decisions settled during brainstorm:

- **Bare leaf keys in __getitem__**: yes, via ``FIELD_TO_LAYER`` routing.
- **Mapping key shape**: leaf dotted paths only.
- **ABC inheritance**: no (duck-typed protocol).
- **YAML flat-key support**: no (separate concern; out of scope).
