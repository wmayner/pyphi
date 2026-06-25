# Serialization rewrite: `jsonify` → `msgspec` (`pyphi.serialize`) — design

Part of the surface-freeze bundle. Replaces the custom `pyphi/jsonify.py`
serialization layer with a modern, typed, performant `pyphi.serialize` package
built on `msgspec`, supporting both JSON and msgpack from one schema.

## Goal

Serialize and deserialize PyPhi result objects (SIA, CauseEffectStructure,
relations, distinctions, partitions, substrates, accounts, …) through a typed
`msgspec.Struct` schema layer that is faster, far more compact, validated on
decode, and decoupled from the domain classes — deleting the hand-maintained
`_loadable_models` registry, the `__class__`/`__version__`/`__id__` per-object
metadata, and the scattered `to_json`/`from_json` dispatch.

## Background: the current `jsonify`

`pyphi/jsonify.py` (471 lines) is a custom JSON layer:

- A recursive `jsonify()` walker plus `PyPhiJSONEncoder` / `PyPhiJSONDecoder`.
- A hand-maintained `_loadable_models()` list of ~45 classes (each line carries a
  pyright-ignore because the classes are reached through dynamic attribute
  access).
- Per-object metadata `__class__` / `__version__` / `__id__` injected by
  `_push_metadata`; `__id__` (`hash(obj)`) drives decode-side memoization in
  `_ObjectCache`.
- Enum-keyed-dict special handling, NumPy array/scalar conversion, and a
  version-compatibility gate (`_check_version`).
- 22 `to_json` and 16 `from_json` hooks scattered across the model modules.
- 35 reference golden fixtures (3.9 MB) round-trip through it for regression.

### Measured costs (real goldens)

| Golden | File | Objects | Per-object metadata | gzip ratio |
| --- | --- | --- | --- | --- |
| `rule154` (phi-structure) | 1277 KB | 8,795 | ~60% of file | 28.7× (97% redundant) |
| `big_subsys` (SIA) | 65 KB | 238 | ~41% of file | 25.2× (96% redundant) |

Two facts drive the design:

1. **~60% of a phi-structure file is per-object metadata** — every object
   carries `__class__` + `__version__` + `__id__`, and the repeated version
   string alone is 137 KB in `rule154`.
2. **The content is ~97% redundant.** A `Relation` is a `frozenset` of
   distinctions whose `to_json` embeds each member distinction *in full*; the
   same distinction belongs to many relations, so it is re-serialized many
   times. NumPy arrays, by contrast, are a small slice of the file (0.5–8%),
   and are tiny (median 1 element).

So the bloat is serialization overhead, not the domain model — removing the
metadata, encoding as msgpack, and normalizing the shared distinctions captures
the compaction without changing the result types.

## Design decisions (settled during brainstorming)

- **Invasiveness:** a decoupled `msgspec.Struct` schema layer (not making the
  domain classes themselves Structs — they are heterogeneous: dataclasses,
  `frozenset` subclasses, plain classes with mixin multiple inheritance, so a
  uniform Struct conversion is infeasible and unnecessary).
- **Wire formats:** JSON (default, human-readable structure) and msgpack
  (binary, compact/fast) from one schema.
- **NumPy arrays:** stored as their `.npy` bytes in a `bytes`-typed schema field;
  msgspec emits base64 in JSON and raw bytes in msgpack automatically. Loaded
  with `allow_pickle=False`. Chosen over a hand-rolled `{shape, dtype, data}`
  struct because the measured size difference is immaterial (arrays are <8% of
  any file) and `.npy` is numpy's own battle-tested format — no hand-rolling.
- **Reference-identity:** all result models are immutable value types with
  `__eq__`/`__hash__`, and round-trip tests compare with `==`, never `is`.
  Nothing relies on post-decode object identity, so collapsing shared
  sub-objects is behavior-neutral — reference-sharing is a pure size
  optimization.
- **Compaction via schema normalization:** the CES schema stores its
  distinctions once in a table; relations reference distinctions by integer
  index instead of embedding them.
- **Compatibility:** regenerate all 35 goldens in the new format; ship a
  standalone one-shot converter (old jsonify JSON → new format) for users' saved
  results. A clean format break is acceptable at the 2.0 major bump.
- **Module:** new `pyphi.serialize` package; `pyphi.jsonify` removed, all
  call-sites updated; no compat shim for the old module.

## Architecture: `pyphi/serialize/`

```
pyphi/serialize/
├── __init__.py     # public API: dumps/loads/dump/load (+ binary path)
├── schema.py       # msgspec.Struct types + the tagged union
├── convert.py      # to_schema()/from_schema() per domain type (centralized)
└── arrays.py       # numpy <-> .npy-bytes helpers
```

### Public API (`__init__.py`)

- `dumps(obj, *, format="json") -> bytes` and `loads(data, *, format="json")` —
  `format` is `"json"` or `"msgpack"`.
- `dump(obj, fp, *, format="json")` / `load(fp, *, format="json")` — file
  streams.
- A single top-level `format_version` constant is written into the encoded
  document (one field for the whole document, not per object) and checked on
  decode (configurable via the existing
  `config.infrastructure.validate_json_version`, renamed in spirit to the
  format-version check).

### Schema layer (`schema.py`)

- One `msgspec.Struct` per serializable type. Each Struct carries a `tag` (via
  `msgspec.Struct(tag=...)` or a tagged union), so the union of all schema types
  is a **tagged union** discriminated by a `type` field. msgspec validates the
  payload against the union and dispatches to the right Struct on decode — this
  is the typed replacement for `_loadable_models` + `__class__` dispatch.
- `Struct`s are `frozen=True`, `forbid_unknown_fields=True` for strict decode.
- NumPy-bearing fields are typed `bytes` and carry `.npy` payloads (see
  `arrays.py`).

### Converters (`convert.py`)

- A central registry mapping each domain type to a `to_schema(obj) -> Struct`
  and `from_schema(struct) -> obj`. This replaces the 22 `to_json` and 16
  `from_json` methods, which are removed from the model modules.
- The φ-value type distinction is preserved: a field records whether a φ was a
  plain `PyPhiFloat` or a `DistanceResult` (with its auxiliary data), so
  `RelationFace`/`MICE` φ values reconstruct to the correct type.
- Enum values (e.g. `Direction`) serialize by `.name`; enum-keyed dicts become a
  typed list of `(key, value)` pairs in the schema (no marker-dict hack).

### CES normalization (the compaction)

The `CauseEffectStructure` schema is normalized:

```
CESSchema(type="ces",
          sia: SIASchema,
          distinctions: list[DistinctionSchema],     # the table, once
          relations: list[RelationSchema])

RelationSchema(distinction_indices: list[int], phi: PhiSchema, faces: ...)
```

At encode, `convert` builds an identity map from each distinction object to its
index in the `distinctions` table (distinctions are identity-shared instances).
At decode, relations resolve their `distinction_indices` against the decoded
`distinctions` list. Correct because the models are value types and nothing
relies on post-decode identity.

### Arrays (`arrays.py`)

- `array_to_bytes(arr) -> bytes`: `np.save` the array into a `BytesIO`, return
  the buffer.
- `bytes_to_array(b) -> np.ndarray`: `np.load(BytesIO(b), allow_pickle=False)`.
- Used by any schema field that carries a repertoire / TPM / connectivity
  matrix. Guarded by a bit-identity round-trip property test.

### `to_dict()` mixin

Types that only need export (display, pandas) and never reload get a small
`ToDictMixin` providing `to_dict()` derived from the same converter description,
without a full round-trip schema. (Round-trip types — SIA, CES, relations,
distinctions, partitions, substrate, account family — get full schemas.)

## Migration & goldens

- Regenerate all 35 jsonify-format goldens (`test/data/**`) in the new format.
  The regeneration script reads each golden with the old `jsonify` decoder (kept
  available only inside the migration script, not in the package), converts to a
  domain object, and re-encodes with `pyphi.serialize`. Reviewed as a golden
  diff.
- `scripts/convert_jsonify.py`: a standalone one-shot converter taking an old
  jsonify JSON file → new-format file, for users with saved results. Stdlib +
  the migration reader only; `# noqa: T201` on prints.

## Validation & safety

- **Typed decode:** `forbid_unknown_fields=True` and the tagged union mean a
  malformed or wrong-typed payload raises at decode rather than silently
  mis-loading.
- **No code execution on load:** `allow_pickle=False` everywhere; the format
  carries only data, never pickles.
- **Array bit-identity property test** (Hypothesis): arbitrary
  shapes/dtypes/contiguity round-trip identically in both formats.
- **Round-trip equality tests** for every round-trip result type (`decoded ==
  original`), in both JSON and msgpack.
- **Golden regression** re-pointed at the regenerated fixtures (`conftest.py`).
- A **compaction check** asserting the new phi-structure goldens are
  substantially smaller than the old (sanity bound on the metadata/normalization
  win), and a re-measurement recorded for the deferred Lever-2 decision.

## Out of scope (deferred, noted)

- **Config storage → `msgspec.Struct`** — a separate `pyphi.conf` refactor; the
  ROADMAP lists it as an open subdecision, not a requirement here.
- **Domain-level "generative relations" compression (Lever 2)** — storing
  relations generatively / restructuring the model. It is a surface change to
  the result types and carries persist-vs-recompute research-semantics risk. The
  measurement shows the bloat is serialization overhead, not the data model, so
  this is deferred: re-measure post-rewrite, and only open it if the new files
  are still large.

## Risks

- **Coverage of the ~45 types.** Each needs a schema + converter; a missed type
  fails its round-trip test. Mitigated by porting the existing
  `test_result_protocols` / `test_complex_model` / golden suites first and
  driving the work until they pass in both formats.
- **φ-type fidelity.** `PyPhiFloat` vs `DistanceResult` must round-trip to the
  right type, including `DistanceResult` auxiliary data. Covered by per-type
  round-trip tests asserting type and value.
- **Embedded `config` and `Provenance` in the SIA** must serialize; `config`
  serialization reuses the existing config-to-dict path (config-as-Struct stays
  out of scope).
- **msgspec dependency** added to `pyproject.toml` core deps.
