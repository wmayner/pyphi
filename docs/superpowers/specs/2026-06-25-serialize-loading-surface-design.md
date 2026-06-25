# Serialization loading surface + lazy registration — design

Follows the `jsonify` → `pyphi.serialize` rewrite. That work built the
serializer and made it the only one; this work gives it an ergonomic public
surface (the part P15 freezes) and removes a latent import-cycle hazard.

## Goal

Make saving and loading PyPhi results ergonomic and discoverable — path-based
module functions, top-level re-exports, and `.save()`/`.load()` methods on the
user-facing result types — while keeping all serialization *logic* centralized
in `pyphi/serialize/convert.py`. Also make `pyphi.serialize` safe to import from
anywhere by deferring its type registration to first use.

## Background

The new serializer's public surface today is `dumps`/`loads` (bytes) and
`dump`/`load` (file objects) — the `json`/`pickle` shape. Two gaps:

1. **No path-based convenience.** The common case ("save this result to a
   file", "load it back") forces the user to open a binary file handle and pass
   it. Scientific users expect the numpy/joblib/pandas convention
   (`np.save(path, arr)` / `np.load(path)`, `joblib.dump`/`load`), which is
   path-based.
2. **`substrate.from_json(filename)` is a redundant per-type loader.** The new
   format is self-describing (every document carries a type tag), so one generic
   loader reconstructs any type. The old `to_json`/`from_json` design needed
   per-type entry points; the new one does not. `substrate.from_json` also
   creates an upward dependency (a core module reaching up to the serializer),
   which forced a deferred-import workaround during the rewrite.

Separately, `pyphi/serialize/convert.py` registers every type's converter via
`_register_*()` calls that run **at module import**, each importing a domain
module. So `import pyphi.serialize` eagerly imports the whole domain tree, which
makes import order matter and caused a real cycle (`substrate` → `serialize` →
`convert` → `formalism.iit4` → `substrate`).

## The architectural rule (load-bearing)

`.save()`/`.load()` on result objects re-introduce a dependency from the domain
classes to the serializer. This is acceptable **only** because they are *thin
delegations* — one line each, no serialization logic on the class:

```python
def save(self, target, *, format=None):
    from pyphi import serialize
    serialize.save(self, target, format=format)
```

The conversion logic stays entirely in `serialize/convert.py`. The whole point
of retiring `to_json`/`from_json` was to remove *scattered serialization logic*
from ~40 classes; a thin facade that delegates to the central serializer is the
pandas pattern (`df.to_parquet()` delegates to the parquet engine) and does not
re-create that problem. **Rule: these methods may only delegate. No field
mapping, no reconstruction, no format branching on the class.** If logic ever
needs to live somewhere, it lives in `serialize`.

## Design

### 1. Lazy registration (`serialize/convert.py`)

Replace the bottom-of-module `_register_*()` calls with a single idempotent
`_ensure_registered()` guarded by a module-level flag. `to_schema` and
`from_schema` call `_ensure_registered()` on entry. Importing `pyphi.serialize`
then performs no domain imports (only `msgspec`, `numpy`, `pyphi.direction` —
all low-level); the domain tree is imported on the first `dumps`/`loads`, by
which point `pyphi` is fully loaded, so no cycle is possible. This stands on its
own merits (cheap, inert import) independent of the cycle.

### 2. Path-based module API (`serialize/__init__.py`)

Consolidate the file/path surface onto `save`/`load`; keep `dumps`/`loads` for
bytes:

- `dumps(obj, *, format="json") -> bytes`
- `loads(data, *, format="json") -> Any`
- `save(obj, target, *, format=None) -> None` — `target` is a path
  (`str`/`os.PathLike`) **or** an open binary file object.
- `load(target, *, format=None) -> Any` — same `target` flexibility.

`save`/`load` replace the previous `dump`/`load(fp)` pair (`load` still accepts a
file object, so existing `serialize.load(fp)` call-sites keep working; the only
`dump` call-site is the scratch demo notebook, updated to `save`).

**Format inference** (`_infer_format(target, format)`): if `format` is given,
use it; else if `target` is a path, infer from the suffix — `.json` → `"json"`,
`.msgpack`/`.mpk` → `"msgpack"`, anything else → `"json"`; else (a file object)
→ `"json"`. Paths are opened binary (`"wb"`/`"rb"`); passed file objects are
assumed binary.

### 3. Top-level re-exports (`pyphi/__init__.py`)

Lift `save` and `load` to the package top level so the most common operation is
the most discoverable:

```python
pyphi.save(result, "result.json")
result = pyphi.load("result.json")
```

These are added to the lifted-names block and `__all__`. They are part of the
surface P15 freezes.

### 4. `Serializable` mixin (`pyphi/serializable.py`, new)

A standalone, lightweight module (imports nothing heavy; **does not import
`pyphi.serialize` at module level** — that would re-create the load-time
coupling lazy registration removes). The mixin's methods import `serialize`
deferred, at call time:

```python
class Serializable:
    """Mixin adding ``save``/``load`` that delegate to :mod:`pyphi.serialize`."""

    def save(self, target, *, format=None) -> None:
        from pyphi import serialize
        serialize.save(self, target, format=format)

    @classmethod
    def load(cls, target, *, format=None):
        from pyphi import serialize
        obj = serialize.load(target, format=format)
        if not isinstance(obj, cls):
            raise TypeError(
                f"{target!r} contains a {type(obj).__name__}, not a {cls.__name__}"
            )
        return obj
```

`save` is an instance method (save self). `load` is a classmethod — it is a
constructor, not an instance operation — and type-checks the result, so
`CauseEffectStructure.load(path)` means "I expect a CES" and raises clearly
otherwise. The generic `serialize.load(path)` remains available for "load
whatever's in the file".

**Applied to the user-facing result types only:** `Substrate`, `System`,
`Transition`, `IIT3SystemIrreducibilityAnalysis`,
`SystemIrreducibilityAnalysis` (+ `NullSystemIrreducibilityAnalysis`),
`CauseEffectStructure` (+ `NullCauseEffectStructure`), `Complex`, `Account`
(+ `DirectedAccount`), `AcSystemIrreducibilityAnalysis`, `Distinctions`
(+ `Resolved`/`Unresolved`), and the `Relations` family. **Not** on internal
building blocks (`Part`, `RIA`, `MICE`, `RelationFace`, `StateSpecification`,
partitions) — those stay reachable through `serialize.dumps`/`load` if needed,
which keeps the mixin from sprawling back onto every class.

Verify no target type already defines its own `save`/`load` (none expected). If
one did, its own method would win by MRO and the mixin would be a silent no-op
there — so the plan checks each type before adding the mixin.

### 5. Remove `substrate.from_json`

Delete the module-level `from_json(filename)` loader in `pyphi/substrate.py` and
its deferred `serialize` import. Callers use `pyphi.load(path)` /
`serialize.load(path)` / `Substrate.load(path)`. This removes the upward edge
outright (better than the deferred-import band-aid). It is a public API removal,
acceptable in this format-break release.

## Out of scope / deferred

- The dead `validate_json_version` config option (a separate cleanup that also
  touches `test_config_layers.py`).
- Config-as-`msgspec.Struct` and generative-relations compression (already
  deferred).
- Async / streaming IO, compression-on-save (`.json.gz`): not now.

## Risks

- **Top-level surface growth.** `pyphi.save`/`load` are two more frozen names;
  justified by being the single most common user operation.
- **Mixin re-coupling.** Mitigated by the delegation-only rule (logic stays in
  `serialize`) and the standalone mixin module with deferred imports.
- **Format inference surprises.** An unrecognized extension silently defaults to
  JSON; documented, and `format=` always overrides.
- **`substrate.from_json` removal** breaks any code/tests calling it; grep
  confirms the test suite's only use is what we migrate, and it is a
  format-break release.

## Testing

- `save`/`load` round-trip to a temp path in both formats, and with an explicit
  file object, for a representative result.
- Format inference: `.json`/`.msgpack`/`.mpk`/no-extension resolve correctly;
  explicit `format=` overrides the suffix.
- `Type.load(path)` returns the right type; loading a file holding a different
  type raises `TypeError`.
- `obj.save(path)` then `pyphi.load(path) == obj` for one type per family.
- Lazy registration: `import pyphi.serialize` imports no domain module (assert
  via `sys.modules` in a subprocess), and the first `dumps` still works.
- `pyphi.save`/`pyphi.load` exist and round-trip.
- Full `uv run --all-extras pytest` (no path argument) green.
