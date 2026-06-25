# Serialization rewrite (`jsonify` → `msgspec`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `pyphi/jsonify.py` with a typed `pyphi.serialize` package built on `msgspec`, serializing every round-trip result type through a tagged-union `msgspec.Struct` schema, supporting JSON and msgpack, with a normalized CES schema and `.npy`-bytes arrays.

**Architecture:** A decoupled schema layer: one `msgspec.Struct` per serializable type (a tagged union discriminated by a `type` tag) lives in `pyphi/serialize/schema.py`; per-type `to_schema`/`from_schema` converters in `pyphi/serialize/convert.py`; numpy `.npy`-bytes helpers in `pyphi/serialize/arrays.py`; the public `dumps`/`loads`/`dump`/`load` API in `pyphi/serialize/__init__.py`. The domain model classes are untouched. The CES schema is normalized (distinctions stored once; relations reference them by index).

**Tech Stack:** Python 3.12+, `msgspec` (new core dependency), numpy `.npy` format, pytest + Hypothesis, ruff + pyright.

## Global Constraints

- **Python 3.12+ only.** No backward-compatibility shims for older Python.
- **No planning-artifact markers** (P-numbers, "Wave N", task numbers, `pre-PXX`) in `pyphi/` source, docstrings, or changelog fragments.
- **Branch `serialize-msgspec`**, off `main`; ships as one PR / local merge into `main` (CI-gated). **Ask before `git push`.**
- **Stage only the files named in each task**; never `git add -A` (concurrent instances share the trunk). Leave `.claude/`, `graphify-out/`, untracked scratch alone.
- **Pre-commit = ruff + pyright; never `--no-verify`.** Ruff reformats and aborts when it changes a file — re-`git add` and re-commit. One import per line.
- **Commit trailer** on every commit:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_012dtSzF2YgDjGpFC9mA47ve
  ```
- **Full verification = `uv run --all-extras pytest` with NO path argument** (runs the `pyphi/` doctest sweep).
- **Round-trip discipline:** every round-trip type has a test asserting `loads(dumps(obj)) == obj` in BOTH `format="json"` and `format="msgpack"`. These tests are the gate for each per-family task.
- **Fidelity:** `loads(dumps(obj)) == obj` must hold for φ values including the `PyPhiFloat` vs `DistanceResult` distinction; numpy arrays must round-trip bit-identically; `allow_pickle=False` on every array load.

---

## Type inventory (the ~45 round-trip types, grouped by task)

The round-trip types and their canonical field sets (from the current
`to_json`/`from_json` and constructors):

- **Simple values** (Task 3): `PyPhiFloat` (float), `Direction` (enum name),
  `NodeLabels` (labels tuple + node_indices), `DistanceResult` (value + aux
  data), `StateSpecification` / `SystemStateSpecification` / `UnitState`.
- **Partitions** (Task 4): `Part`, `NullCut`, `EdgeCut`, `CompleteEdgeCut`,
  `DirectedBipartition` (`direction`, `from_nodes`, `to_nodes`),
  `DirectedJointPartition`, `DirectedSetPartition`, `JointBipartition`,
  `JointPartition`, `JointTripartition`.
- **RIA / MICE** (Task 5): `RepertoireIrreducibilityAnalysis` (fields:
  `phi`, `direction`, `mechanism`, `mechanism_state`, `purview`,
  `purview_state`, `partition`, `repertoire`, `partitioned_repertoire`,
  `specified_state`, `node_labels`; **+ tie-peer graph**:
  `_partition_ties`, `_state_ties`), `MaximallyIrreducibleCause` /
  `Effect` / `CauseOrEffect`.
- **Distinctions** (Task 6): `Concept`/`Distinction`, `Distinctions`,
  `ResolvedDistinctions`, `UnresolvedDistinctions`.
- **SIA family** (Task 7): `IIT3SystemIrreducibilityAnalysis` (`phi`,
  `distinctions`, `partitioned_distinctions`, `partition`, `node_indices`,
  `node_labels`, `current_state`; **+ tie-peers**),
  `SystemIrreducibilityAnalysis` (IIT4, `__dict__` minus `_ties`/`runner_up`;
  **+ tie-peers**), `NullSystemIrreducibilityAnalysis`,
  `NullCauseEffectStructure`, `Complex`, `ExcludedCandidate`.
- **Relations + CES** (Task 8): `RelationFace` (`relata` + `phi`), `Relation`
  (`distinctions` + `phi`), `Relations`, `AnalyticalRelations`,
  `ConcreteRelations`, `NullRelations`, `CauseEffectStructure`
  (`sia`, `distinctions`, `relations`) — **normalized**.
- **Substrate / System / AC** (Task 9): `Substrate`, `System`, `Transition`,
  `Account`, `AcRepertoireIrreducibilityAnalysis`,
  `AcSystemIrreducibilityAnalysis`, `CausalLink`, `Provenance`.

Each per-family task enumerates its types' exact fields from the constructor and
current `to_json`, defines a `msgspec.Struct` per type with a unique `tag`, and a
`to_schema`/`from_schema` pair following the patterns established in Tasks 2–5.

---

## Task 1: Add `msgspec` and the array helpers

**Files:**
- Modify: `pyproject.toml` (add `msgspec` to core deps)
- Create: `pyphi/serialize/__init__.py` (empty package marker for now)
- Create: `pyphi/serialize/arrays.py`
- Create: `test/test_serialize_arrays.py`

**Interfaces:**
- Produces: `array_to_bytes(arr: np.ndarray) -> bytes`,
  `bytes_to_array(b: bytes) -> np.ndarray` in `pyphi.serialize.arrays`.

- [ ] **Step 1: Add the dependency**

In `pyproject.toml` core `dependencies`, add (keeping the list sorted):
```
    "msgspec>=0.18",
```
Run: `uv sync --all-extras` and confirm it installs.

- [ ] **Step 2: Write the failing bit-identity property test**

`test/test_serialize_arrays.py`:
```python
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from pyphi.serialize.arrays import array_to_bytes, bytes_to_array


@settings(max_examples=100)
@given(
    hnp.arrays(
        dtype=st.sampled_from([np.float64, np.float32, np.int64, np.bool_]),
        shape=hnp.array_shapes(min_dims=0, max_dims=4, max_side=6),
    )
)
def test_array_round_trip_is_bit_identical(arr):
    restored = bytes_to_array(array_to_bytes(arr))
    assert restored.dtype == arr.dtype
    assert restored.shape == arr.shape
    assert np.array_equal(restored, arr)


def test_non_contiguous_array_round_trips():
    arr = np.arange(24, dtype=np.float64).reshape(4, 6)[:, ::2]
    assert not arr.flags["C_CONTIGUOUS"]
    restored = bytes_to_array(array_to_bytes(arr))
    assert np.array_equal(restored, arr)
```

- [ ] **Step 3: Run — expect FAIL (module missing)**

Run: `uv run --all-extras pytest test/test_serialize_arrays.py -q`
Expected: import error / FAIL.

- [ ] **Step 4: Implement `arrays.py`**

```python
# serialize/arrays.py
"""Exact, compact serialization of numpy arrays as ``.npy`` bytes.

The ``.npy`` format records dtype, shape, byte order, and Fortran/C order, so an
array round-trips bit-identically and loads correctly across platforms. Stored in
a ``bytes`` schema field, msgspec emits it as base64 in JSON and as raw bytes in
msgpack. Loaded with ``allow_pickle=False`` so a serialized file can never
execute code on load.
"""

import io

import numpy as np


def array_to_bytes(arr: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(arr), allow_pickle=False)
    return buffer.getvalue()


def bytes_to_array(data: bytes) -> np.ndarray:
    return np.load(io.BytesIO(data), allow_pickle=False)
```

- [ ] **Step 5: Run — expect PASS**

Run: `uv run --all-extras pytest test/test_serialize_arrays.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock pyphi/serialize/__init__.py pyphi/serialize/arrays.py test/test_serialize_arrays.py
git commit  # "Add msgspec dependency and numpy .npy-bytes array helpers"
```

---

## Task 2: Core API + tagged-union plumbing (end-to-end for one type)

**Files:**
- Modify: `pyphi/serialize/__init__.py` (public API)
- Create: `pyphi/serialize/schema.py` (the tagged union + first Struct)
- Create: `pyphi/serialize/convert.py` (the converter registry + first converter)
- Create: `test/test_serialize_api.py`

**Interfaces:**
- Produces:
  - `dumps(obj, *, format="json") -> bytes`, `loads(data, *, format="json") -> Any`
  - `dump(obj, fp, *, format="json") -> None`, `load(fp, *, format="json") -> Any`
  - `FORMAT_VERSION: int`
  - `schema.Schema` = the tagged union type; `schema.DirectionSchema` (first member)
  - `convert.to_schema(obj) -> Schema`, `convert.from_schema(s) -> Any`,
    `convert.register(domain_type, encode, decode)`

- [ ] **Step 1: Write the failing API round-trip test (Direction)**

`test/test_serialize_api.py`:
```python
import pytest

from pyphi.direction import Direction
from pyphi import serialize


@pytest.mark.parametrize("fmt", ["json", "msgpack"])
def test_direction_round_trips(fmt):
    data = serialize.dumps(Direction.CAUSE, format=fmt)
    assert isinstance(data, bytes)
    assert serialize.loads(data, format=fmt) == Direction.CAUSE


def test_unknown_type_tag_raises():
    bad = serialize.dumps(Direction.CAUSE, format="json").replace(
        b'"direction"', b'"nonsense"'
    )
    with pytest.raises(Exception):
        serialize.loads(bad, format="json")
```

- [ ] **Step 2: Run — expect FAIL**

Run: `uv run --all-extras pytest test/test_serialize_api.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement `schema.py` (tagged union, first member)**

```python
# serialize/schema.py
"""msgspec schema types for serializing PyPhi results.

Each serializable type has one frozen ``msgspec.Struct`` carrying a unique
string ``tag``. ``Schema`` is the tagged union of all of them; msgspec uses the
tag to validate and dispatch on decode. Adding a type means adding its Struct
here and registering its converter in :mod:`pyphi.serialize.convert`.
"""

from typing import Union

import msgspec


class DirectionSchema(msgspec.Struct, frozen=True, tag="direction"):
    name: str


# The tagged union grows one member per serializable type.
Schema = Union[DirectionSchema,]
```

- [ ] **Step 4: Implement `convert.py` (registry + first converter)**

```python
# serialize/convert.py
"""Convert between PyPhi domain objects and their msgspec schema Structs.

Two registries map a domain type to its encoder and a schema Struct type to its
decoder. This replaces the per-class ``to_json`` / ``from_json`` methods. Each
serializable type adds one ``_register_<type>()`` populating both registries,
all called at import time.
"""

from collections.abc import Callable
from typing import Any

from pyphi.direction import Direction
from pyphi.serialize import schema

_ENCODERS: dict[type, Callable[[Any], Any]] = {}  # domain type   -> encode
_DECODERS: dict[type, Callable[[Any], Any]] = {}  # schema Struct  -> decode


def to_schema(obj: Any) -> Any:
    encode = _ENCODERS.get(type(obj))
    if encode is None:
        raise TypeError(f"No serializer registered for {type(obj).__name__}")
    return encode(obj)


def from_schema(struct: Any) -> Any:
    decode = _DECODERS.get(type(struct))
    if decode is None:
        raise TypeError(f"No deserializer registered for {type(struct).__name__}")
    return decode(struct)


def _register_direction() -> None:
    _ENCODERS[Direction] = lambda d: schema.DirectionSchema(name=d.name)
    _DECODERS[schema.DirectionSchema] = lambda s: Direction[s.name]


_register_direction()
```

Each later task adds one `_register_<type>()` per type, following this exact
shape, and a call to it at module import.

- [ ] **Step 5: Implement the public API in `__init__.py`**

```python
# serialize/__init__.py
"""Typed, compact (de)serialization of PyPhi results via msgspec.

Supports two wire formats from one schema: ``"json"`` (default, readable
structure) and ``"msgpack"`` (binary, compact). The document carries a single
top-level ``format_version``.
"""

from typing import Any

import msgspec

from pyphi.serialize import convert
from pyphi.serialize import schema

FORMAT_VERSION = 1


class _Document(msgspec.Struct, frozen=True):
    format_version: int
    payload: schema.Schema


def _encoder(fmt: str):
    if fmt == "json":
        return msgspec.json.encode
    if fmt == "msgpack":
        return msgspec.msgpack.encode
    raise ValueError(f"Unknown format: {fmt!r}")


def _decode(data: bytes, fmt: str) -> _Document:
    if fmt == "json":
        return msgspec.json.decode(data, type=_Document)
    if fmt == "msgpack":
        return msgspec.msgpack.decode(data, type=_Document)
    raise ValueError(f"Unknown format: {fmt!r}")


def dumps(obj: Any, *, format: str = "json") -> bytes:
    doc = _Document(format_version=FORMAT_VERSION, payload=convert.to_schema(obj))
    return _encoder(format)(doc)


def loads(data: bytes, *, format: str = "json") -> Any:
    doc = _decode(data, format)
    return convert.from_schema(doc.payload)


def dump(obj: Any, fp, *, format: str = "json") -> None:
    fp.write(dumps(obj, format=format))


def load(fp, *, format: str = "json") -> Any:
    return loads(fp.read(), format=format)
```

- [ ] **Step 6: Run — expect PASS**

Run: `uv run --all-extras pytest test/test_serialize_api.py -q`
Expected: PASS (both formats; unknown tag raises a msgspec `ValidationError`).

- [ ] **Step 7: Commit**

```bash
git add pyphi/serialize/__init__.py pyphi/serialize/schema.py pyphi/serialize/convert.py test/test_serialize_api.py
git commit  # "Add pyphi.serialize core API and tagged-union schema plumbing"
```

---

## Tasks 3–9: per-family schemas + converters

Each task follows the **same recipe**, gated by a round-trip test:

1. Write `test/test_serialize_<family>.py`: for each type in the family,
   construct a representative instance (reuse `test/example_networks.py` and the
   existing `test_result_protocols` / `test_complex_model` fixtures) and assert
   `serialize.loads(serialize.dumps(obj, format=fmt), format=fmt) == obj` for
   `fmt in ("json", "msgpack")`.
2. Run it — expect FAIL (`TypeError: No serializer registered`).
3. Add one `msgspec.Struct` per type to `schema.py` (unique `tag`; numpy fields
   typed `bytes` via `arrays.array_to_bytes`; nested domain objects typed as the
   relevant schema or `Schema`), add each member to the `Schema` union.
4. Add one `_register_<type>()` per type to `convert.py` populating
   `_ENCODERS`/`_DECODERS`, called at import.
5. Run — expect PASS.
6. Commit.

The three structural patterns the families need (full code below) are: a flat
value type (Task 3), an array-bearing type (Task 5), a **tie-peer graph** type
(Task 5/7), and the **normalized CES** (Task 8).

### Task 3: simple value types
Types: `PyPhiFloat`, `NodeLabels`, `DistanceResult`,
`StateSpecification`/`SystemStateSpecification`/`UnitState`.
Pattern (flat value) — e.g. `NodeLabels`:
```python
class NodeLabelsSchema(msgspec.Struct, frozen=True, tag="node_labels"):
    labels: tuple[str, ...]
    node_indices: tuple[int, ...]

def _register_node_labels() -> None:
    from pyphi.labels import NodeLabels
    _ENCODERS[NodeLabels] = lambda n: schema.NodeLabelsSchema(
        labels=tuple(n.labels), node_indices=tuple(n.node_indices)
    )
    _DECODERS[schema.NodeLabelsSchema] = lambda s: NodeLabels(s.labels, s.node_indices)
```
`DistanceResult` records its float value plus its public auxiliary data
(`_public_aux_data()`); the φ-type tag (`PyPhiFloat` vs `DistanceResult`) is the
Struct identity. Define the alias `PhiSchema = Union[PyPhiFloatSchema,
DistanceResultSchema]` in `schema.py` and use it for every φ field, so the
distinction round-trips.

**Union aliases:** as each family lands, define a union alias next to it in
`schema.py` for fields that accept any member of that family —
`PhiSchema` (Task 3), `StateSpecSchema` (Task 3), `PartitionSchema` (Task 4, the
union of the ~10 partition Structs), `ConceptSchema`/`DistinctionSchema`
(Task 6). Later tasks reference these aliases (e.g. `RIASchema.partition:
PartitionSchema`). A field that can hold an arbitrary nested result uses the
top-level `Schema` union.

### Task 4: partitions
Types: `Part`, `NullCut`, `EdgeCut`, `CompleteEdgeCut`, `DirectedBipartition`,
`DirectedJointPartition`, `DirectedSetPartition`, `JointBipartition`,
`JointPartition`, `JointTripartition`. Each has a small explicit field set
(e.g. `DirectedBipartition`: `direction`, `from_nodes`, `to_nodes`) — read each
type's current `to_json`/`from_json` for the exact fields; flat-value pattern.

### Task 5: RIA + MICE (array-bearing + tie-peer patterns)
**Array-bearing field** pattern:
```python
class RIASchema(msgspec.Struct, frozen=True, tag="ria"):
    phi: PhiSchema
    direction: DirectionSchema
    mechanism: tuple[int, ...]
    mechanism_state: tuple[int, ...] | None
    purview: tuple[int, ...]
    purview_state: tuple[int, ...] | None
    partition: PartitionSchema
    repertoire: bytes | None            # array_to_bytes(...)
    partitioned_repertoire: bytes | None
    specified_state: StateSpecSchema | None
    node_labels: NodeLabelsSchema | None
    partition_tie_peers: tuple["RIASchema", ...] = ()
    state_tie_peers: tuple["RIASchema", ...] = ()
```
**Tie-peer pattern** (mirrors the current `_SERIALIZING_AS_TIE_PEER`
contextvar): the encoder emits peers (excluding self) into
`partition_tie_peers` / `state_tie_peers` once (peers encode without their own
peers to break the cycle); the decoder reconstructs the mutual tie lists after
building all peers, exactly as the current `from_json` does
(`_partition_ties`/`_state_ties` set to `[instance, *peers]`). `repertoire` uses
`arrays.array_to_bytes` on encode and `arrays.bytes_to_array` on decode.

### Task 6: distinctions
Types: `Concept`/`Distinction`, `Distinctions`, `ResolvedDistinctions`,
`UnresolvedDistinctions`. A `Distinctions` schema holds `concepts:
tuple[ConceptSchema, ...]`; the subtype tag (`resolved`/`unresolved`) preserves
the resolution-status marker.

### Task 7: SIA family (tie-peer)
Types: `IIT3SystemIrreducibilityAnalysis`, `SystemIrreducibilityAnalysis`,
`NullSystemIrreducibilityAnalysis`, `NullCauseEffectStructure`, `Complex`,
`ExcludedCandidate`. Field sets from the current `to_json` (IIT3 SIA: `phi`,
`distinctions`, `partitioned_distinctions`, `partition`, `node_indices`,
`node_labels`, `current_state`; IIT4 SIA: the dataclass fields minus `_ties` and
`runner_up`, plus `system_state`, `cause`, `effect`, `intrinsic_differentiation`,
`config`). Tie-peers via the Task-5 pattern. `config` is serialized via the
existing config-to-dict path stored as a schema field (config-as-Struct stays
out of scope).

### Task 8: relations + normalized CES
`RelationFace` (`relata`, `phi`), `Relation` (`distinctions`, `phi`),
`Relations`/`AnalyticalRelations`/`ConcreteRelations`/`NullRelations`.
**Normalized CES** (full pattern):
```python
class CESSchema(msgspec.Struct, frozen=True, tag="ces"):
    sia: Schema
    distinctions: tuple[ConceptSchema, ...]      # the table, encoded once
    relations: tuple["RelationRefSchema", ...]

class RelationRefSchema(msgspec.Struct, frozen=True, tag="relation_ref"):
    distinction_indices: tuple[int, ...]
    phi: PhiSchema
```
Encoder: build `index = {id(d): i for i, d in enumerate(ces.distinctions)}`;
encode each relation as the indices of its member distinctions
(`tuple(index[id(d)] for d in relation)`). Decoder: decode the `distinctions`
table first, then rebuild each `Relation` from
`tuple(decoded_distinctions[i] for i in rel.distinction_indices)` with its phi.
Correct because distinctions are identity-shared and nothing relies on
post-decode identity.

### Task 9: substrate / system / AC
Types: `Substrate`, `System`, `Transition`, `Account`,
`AcRepertoireIrreducibilityAnalysis`, `AcSystemIrreducibilityAnalysis`,
`CausalLink`, `Provenance`. Substrate/System carry TPM arrays (`bytes` fields via
`arrays`). AC field sets from the four `to_json` sites in
`pyphi/models/actual_causation.py`.

Each of Tasks 3–9 ends with: `uv run --all-extras pytest
test/test_serialize_<family>.py -q` PASS, then commit
(`git add pyphi/serialize/schema.py pyphi/serialize/convert.py test/test_serialize_<family>.py`).

---

## Task 10: Wire call-sites, remove `jsonify` and the `to_json`/`from_json` hooks

**Files:**
- Modify: `pyphi/substrate.py:20,602`, `pyphi/models/sia.py:282`,
  `pyphi/models/state_specification.py:178`,
  `pyphi/models/actual_causation.py:216,438`, `pyphi/models/ria.py:563`,
  `pyphi/formalism/iit4/__init__.py:482` (replace `jsonify.*` /
  `from pyphi.jsonify import jsonify` with `pyphi.serialize` calls)
- Modify: every model module — remove the now-dead `to_json`/`from_json` methods
  (22 + 16) and the `_SERIALIZING_AS_TIE_PEER` contextvar machinery
- Delete: `pyphi/jsonify.py`
- Modify: `test/conftest.py` and the `jsonify`-importing tests
  (`test_provenance.py`, `test_cache_registry.py`, `test_complex_model.py`,
  `test_iit4.py`, `test_result_protocols.py`) to use `pyphi.serialize`

- [ ] **Step 1: Replace each source call-site**

`pyphi/substrate.py:602` `jsonify.load(f)` → `serialize.load(f)`; the
`from pyphi.jsonify import jsonify` sites that call `jsonify(obj)` become
`serialize.dumps(obj)` or are removed if they only built a dict for embedding
(those become direct `convert.to_schema` usage inside the relevant converter).
Update the imports to `from pyphi import serialize`.

- [ ] **Step 2: Remove the `to_json`/`from_json` methods and the tie-peer contextvar**

Delete every `def to_json` / `def from_json` from the model modules (their logic
now lives in `convert.py`) and the `_SERIALIZING_AS_TIE_PEER` contextvar in
`sia.py`/`ria.py`. Run `grep -rn "def to_json\|def from_json\|_SERIALIZING_AS_TIE_PEER" pyphi`
and confirm empty.

- [ ] **Step 3: Delete `jsonify.py`**

```bash
git rm pyphi/jsonify.py
```

- [ ] **Step 4: Update the tests that imported `jsonify`**

Replace `from pyphi import jsonify` / `jsonify.dumps`/`loads`/`load` with
`from pyphi import serialize` / `serialize.dumps`/`loads`/`load` in
`test/conftest.py` and the five test files above. (Golden loading in conftest is
updated in Task 11.)

- [ ] **Step 5: Run the serialize + protocol tests**

Run: `uv run --all-extras pytest test/test_serialize_api.py test/test_serialize_*.py test/test_result_protocols.py test/test_complex_model.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -- pyphi/ test/conftest.py test/test_provenance.py test/test_cache_registry.py test/test_complex_model.py test/test_iit4.py test/test_result_protocols.py
git commit  # "Route serialization through pyphi.serialize; remove jsonify and the to_json/from_json hooks"
```
(Use explicit paths, not `-A`.)

---

## Task 11: Regenerate goldens + compaction check

**Files:**
- Create: `scripts/regenerate_serialize_goldens.py`
- Modify: `test/conftest.py` (load goldens via `pyphi.serialize`)
- Modify: the 35 `test/data/**` jsonify-format fixtures (regenerated)
- Create: `test/test_serialize_compaction.py`

- [ ] **Step 1: Write the regeneration script**

`scripts/regenerate_serialize_goldens.py`: for each old jsonify-format golden
(those containing `__class__`), read it with a vendored copy of the *old*
jsonify decoder (kept only in this script, not in the package), convert to a
domain object, and re-encode with `pyphi.serialize.dumps(obj, format="json")`,
writing back to the same path. `# noqa: T201` on prints.

- [ ] **Step 2: Point conftest at `pyphi.serialize`**

Replace the seven `jsonify.load(f)` calls in `test/conftest.py` with
`serialize.load(f)`.

- [ ] **Step 3: Regenerate**

Run: `uv run python scripts/regenerate_serialize_goldens.py`
Review the diff: every regenerated golden is valid JSON, smaller, and loads back
to an equal object.

- [ ] **Step 4: Compaction check**

`test/test_serialize_compaction.py`: assert each regenerated phi-structure /
SIA golden is at least, say, 3× smaller than its byte size recorded from the old
format (pin the old sizes as constants), and record the new sizes in the test as
the re-measurement input for the deferred Lever-2 decision.

- [ ] **Step 5: Run the golden regression suite**

Run: `uv run --all-extras pytest test/ -k "golden or regression or result_protocols or relations or sia or phi_structure" -q`
Expected: PASS against the regenerated fixtures.

- [ ] **Step 6: Commit**

```bash
git add scripts/regenerate_serialize_goldens.py test/conftest.py test/data test/test_serialize_compaction.py
git commit  # "Regenerate goldens in the new serialize format; add compaction check"
```

---

## Task 12: Standalone old-format converter

**Files:**
- Create: `scripts/convert_jsonify.py`
- Create: `test/test_convert_jsonify.py`

- [ ] **Step 1: Write the converter test**

`test/test_convert_jsonify.py`: take a small fixture saved in the OLD jsonify
format (keep one copy under `test/data/legacy/`), run the converter, and assert
the output loads via `pyphi.serialize` to an object equal to loading the legacy
file via the vendored old reader.

- [ ] **Step 2: Implement the converter**

`scripts/convert_jsonify.py <old.json> <new.json>`: reads an old jsonify file
with the vendored old reader, converts to a domain object, writes
`pyphi.serialize.dumps(obj, format="json")`. Stdlib + the vendored reader only;
`# noqa: T201` on prints; argparse CLI.

- [ ] **Step 3: Run — expect PASS**

Run: `uv run --all-extras pytest test/test_convert_jsonify.py -q`

- [ ] **Step 4: Commit**

```bash
git add scripts/convert_jsonify.py test/test_convert_jsonify.py test/data/legacy
git commit  # "Add standalone converter for old jsonify-format files"
```

---

## Task 13: Full verification + ROADMAP

**Files:**
- Modify: `ROADMAP.md` (P15 detail: serialization rewrite landed)
- Create: `changelog.d/serialize-msgspec.change.md`

- [ ] **Step 1: Full suite, no path argument**

Run: `uv run --all-extras pytest`
Expected: PASS (doctest sweep included; any `pyphi/` doctest that referenced
`jsonify` must have been updated in Task 10).

- [ ] **Step 2: Pre-commit + pyright**

Run: `SKIP=pyright uv run pre-commit run --all-files && uv run pyright pyphi`
Expected: clean.

- [ ] **Step 3: Changelog**

```bash
echo 'Replaced `pyphi.jsonify` with `pyphi.serialize`: a typed msgspec-based serializer supporting JSON and msgpack, with far smaller output. Old-format files can be upgraded with `scripts/convert_jsonify.py`.' > changelog.d/serialize-msgspec.change.md
```

- [ ] **Step 4: Update ROADMAP**

In the Wave-5 detail, record the serialization rewrite as landed (new
`pyphi.serialize`, jsonify removed, goldens regenerated, measured compaction);
note the re-measurement result and the standing deferral of Lever-2
generative-relations. Update the P15 row detail accordingly (still 🟡 partial:
docstring sweep, test reorg, to_pandas, PR triage, changelog condense remain).

- [ ] **Step 5: Commit and finish the branch**

```bash
git add ROADMAP.md changelog.d/serialize-msgspec.change.md
git commit  # "Record serialization rewrite in ROADMAP"
```
Then use superpowers:finishing-a-development-branch. **Ask before pushing.**

---

## Notes for the implementer

- **Order matters for the union:** a type's Struct must be added to the `Schema`
  union before a type that references it is round-tripped. The family order
  (values → partitions → RIA/MICE → distinctions → SIA → relations/CES →
  substrate/AC) respects the containment order.
- **The round-trip test is the real gate** for Tasks 3–9; the per-family `-k`
  runs are fast feedback, but Task 13 Step 1 (`uv run --all-extras pytest`, no
  path) is the verification of record.
- **Tie-peers** (RIA, SIA): replicate the existing encode-peers-without-their-
  own-peers / rebuild-mutual-lists-on-decode logic exactly; the current
  `from_json` in `sia.py`/`ria.py` is the reference.
- **φ fidelity:** keep the `PyPhiFloat` vs `DistanceResult` distinction as
  separate Struct tags so a relation/MICE φ reconstructs to the right type.
- **Do not** widen scope to config-as-Struct or generative relations — both are
  explicitly deferred in the spec.
- **`to_dict()` for export-only types:** the codebase already provides
  `ToPandasMixin` and `ToDictFromExplicitAttrsMixin` (see `pyphi/models/`), which
  cover the display/pandas export need the spec's `to_dict` mixin described. Do
  not add a new mixin unless a concrete export gap surfaces during the work;
  if one does, extend the existing mixin rather than introducing a parallel one.
