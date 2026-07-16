# Serialization Round-Trip Completeness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the serialization layer round-trip every computed field the result objects carry, fix the null-AcSIA serialization crash, and reject files written by a future wire format.

**Architecture:** Replicate the complete `IIT4SIASchema` pattern (fields for signed φ, reasons, config, provenance, tie peers, plus the encode-peers-without-self recursion guard) onto the seven schemas that lack it, in `pyphi/serialize/schema.py` + `convert.py`. Two root-cause model fixes ride along: `_null_ac_sia` constructs real `Account` objects, and `Transition.__eq__`/`__hash__` include `noise_background`. All schema additions are optional-with-defaults, so existing files keep loading.

**Tech Stack:** Python 3.13, msgspec (tagged Structs, json + msgpack), pytest.

**Spec:** `docs/superpowers/specs/2026-07-16-serialization-round-trip-completeness-design.md`

## Global Constraints

- Work in the worktree `/Users/will/projects/pyphi/.claude/worktrees/serialize-round-trip` (branch `fix/serialize-round-trip`). All commands below run from that directory.
- Always `uv run` for python/pytest.
- `FORMAT_VERSION` stays `1`; `loads()` rejects only versions *greater* than the library's.
- New schema fields must be appended **after** all existing fields and must have defaults (msgspec requires defaulted fields last; defaults are what keep old files loading).
- Commit messages end with the two trailers:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01UB4jbV4bQ8Y7Eywt1TNgJe`.
- Never `git commit --no-verify`. If `git commit` output shows only hook lines and `git status` shows `MM` files, the ruff formatter modified files mid-hook and the commit did NOT land — re-stage and re-commit.
- Docstrings: NumPy style, final-state voice, no references to the review/plan.
- Final gate (Task 9): pathless `uv run pytest` redirected to a file; read the summary line, never trust pipeline exit codes.

---

### Task 1: Reject future `format_version` on load

**Files:**
- Modify: `pyphi/serialize/__init__.py` (function `loads`, lines 50–52)
- Test: `test/serialize/test_serialize_io.py`

**Interfaces:**
- Consumes: existing `_Document`, `FORMAT_VERSION = 1`.
- Produces: `loads()` raises `ValueError` when `doc.format_version > FORMAT_VERSION`. No signature changes.

- [ ] **Step 1: Write the failing tests**

Add to `test/serialize/test_serialize_io.py` (add `import json` at the top, after `import io`):

```python
def test_future_format_version_rejected():
    doc = json.loads(serialize.dumps(1.0, format="json"))
    doc["format_version"] = serialize.FORMAT_VERSION + 1
    with pytest.raises(ValueError, match="format_version"):
        serialize.loads(json.dumps(doc).encode(), format="json")


def test_current_format_version_loads():
    data = serialize.dumps(1.0, format="json")
    assert serialize.loads(data, format="json") == 1.0
```

- [ ] **Step 2: Run tests to verify the new one fails**

Run: `uv run pytest test/serialize/test_serialize_io.py -v`
Expected: `test_future_format_version_rejected` FAILS (`DID NOT RAISE`); `test_current_format_version_loads` passes.

- [ ] **Step 3: Implement the check**

In `pyphi/serialize/__init__.py` replace `loads`:

```python
def loads(data: bytes, *, format: str = "json") -> Any:
    doc = _decode(data, format)
    if doc.format_version > FORMAT_VERSION:
        raise ValueError(
            f"cannot load format_version {doc.format_version}: this version of "
            f"PyPhi reads format_version {FORMAT_VERSION} or lower"
        )
    return convert.from_schema(doc.payload)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/serialize/test_serialize_io.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add pyphi/serialize/__init__.py test/serialize/test_serialize_io.py
git commit -m "Reject serialized documents with a newer format_version on load"
```

---

### Task 2: RIA `signed_phi`, `selectivity`, `reasons`

**Files:**
- Modify: `pyphi/serialize/schema.py` (`RIASchema`, lines 138–152)
- Modify: `pyphi/serialize/convert.py` (`_encode_ria` ~line 285, `_decode_ria` ~line 305)
- Test: `test/serialize/test_serialize_ria_mice.py`

**Interfaces:**
- Consumes: `RepertoireIrreducibilityAnalysis.__init__(..., selectivity=None, reasons=None, signed_phi=None, ...)` (`pyphi/models/ria.py`); helpers `_enc_optional`/`_dec_optional`, `_enc_reasons`/`_dec_reasons` (in `convert.py`).
- Produces: `RIASchema` fields `signed_phi: PhiSchema | None = None`, `selectivity: float | None = None`, `reasons: tuple[str, ...] | None = None`, threaded through encode/decode. `signed_normalized_phi` is derived by the constructor from `signed_phi`, so it needs no field.

- [ ] **Step 1: Write the failing tests**

Add to `test/serialize/test_serialize_ria_mice.py` (add `import json` at the top; the `NullResultReason` import goes at the top too: `from pyphi.models.explanation import NullResultReason`):

```python
@pytest.mark.parametrize("fmt", FORMATS)
def test_ria_preserves_negative_signed_phi(fmt):
    obj = make_ria(phi=-0.25)
    assert float(obj.phi) == 0.0
    assert float(obj.signed_phi) == -0.25
    restored = round_trip(obj, fmt)
    assert float(restored.signed_phi) == -0.25
    assert float(restored.signed_normalized_phi) == float(obj.signed_normalized_phi)
    assert float(restored.phi) == 0.0


@pytest.mark.parametrize("fmt", FORMATS)
def test_ria_preserves_selectivity_and_reasons(fmt):
    obj = RepertoireIrreducibilityAnalysis(
        phi=0.3,
        direction=Direction.CAUSE,
        mechanism=(0,),
        purview=(1,),
        partition=JointPartition(Part((0,), (1,))),
        repertoire=np.array([0.4, 0.6]),
        partitioned_repertoire=np.array([0.5, 0.5]),
        mechanism_state=(1,),
        purview_state=(0,),
        selectivity=0.5,
        reasons=[NullResultReason.NO_PURVIEWS],
    )
    restored = round_trip(obj, fmt)
    assert restored.selectivity == 0.5
    assert restored.reasons == [NullResultReason.NO_PURVIEWS]


def test_ria_loads_without_new_fields():
    # Payloads written before these fields existed decode with the defaults.
    obj = make_ria()
    data = json.loads(serialize.dumps(obj, format="json"))

    def strip(o):
        if isinstance(o, dict):
            for key in ("signed_phi", "selectivity", "reasons"):
                o.pop(key, None)
            for v in o.values():
                strip(v)
        elif isinstance(o, list):
            for item in o:
                strip(item)

    strip(data)
    restored = serialize.loads(json.dumps(data).encode(), format="json")
    assert float(restored.signed_phi) == float(restored.phi)
    assert restored.selectivity is None
    assert restored.reasons is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/serialize/test_serialize_ria_mice.py -v`
Expected: the two new round-trip tests FAIL (`signed_phi` 0.0 ≠ −0.25; `selectivity` None). The strip test may pass trivially before implementation — that is fine; it exists to pin backward compatibility after.

- [ ] **Step 3: Implement**

In `pyphi/serialize/schema.py`, append to `RIASchema` (after `partition_margin`):

```python
    signed_phi: PhiSchema | None = None
    selectivity: float | None = None
    reasons: tuple[str, ...] | None = None
```

In `pyphi/serialize/convert.py`, `_encode_ria`: add to the `schema.RIASchema(...)` call, after `partition_margin=...`:

```python
        signed_phi=_enc_optional(ria.signed_phi),
        selectivity=ria.selectivity,
        reasons=_enc_reasons(ria.reasons),
```

In `_decode_ria`: add to the `RepertoireIrreducibilityAnalysis(...)` call, after `partition_margin=...`:

```python
        signed_phi=_dec_optional(struct.signed_phi),
        selectivity=struct.selectivity,
        reasons=_dec_reasons(struct.reasons),
```

(`_enc_reasons`/`_dec_reasons` are defined later in the module than `_encode_ria`; module-level name resolution at call time makes that fine.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/serialize/test_serialize_ria_mice.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add pyphi/serialize/schema.py pyphi/serialize/convert.py test/serialize/test_serialize_ria_mice.py
git commit -m "Serialize RIA signed_phi, selectivity, and reasons"
```

---

### Task 3: MICE purview ties (tri-state)

**Files:**
- Modify: `pyphi/serialize/schema.py` (the three MICE schemas, lines 155–167)
- Modify: `pyphi/serialize/convert.py` (`_decode_mice` ~line 346, `_register_mice` ~line 353)
- Test: `test/serialize/test_serialize_ria_mice.py`

**Interfaces:**
- Consumes: `MaximallyIrreducibleCauseOrEffect._purview_ties` (`None` = never computed; tuple including self otherwise), `num_purview_ties` (NaN when `None`, else `len − 1`); `from_schema` dispatches decoding by struct type.
- Produces: each MICE schema gains `purview_tie_peers: tuple["MICEAnySchema", ...] | None = None`; new `convert._encode_mice(mice, struct_cls, *, include_peers=True)` and `convert._mice_struct_cls(mice)`. Encoding: `None` ⇒ not computed; `()` ⇒ computed, no ties; else peers-without-self with their own tie fields suppressed. Decoding restores the shared tie tuple.

- [ ] **Step 1: Write the failing tests**

Add to `test/serialize/test_serialize_ria_mice.py`:

```python
@pytest.mark.parametrize("fmt", FORMATS)
def test_mice_preserves_purview_ties(fmt):
    a = MaximallyIrreducibleCause(make_ria(Direction.CAUSE))
    b = MaximallyIrreducibleCause(make_ria(Direction.CAUSE))
    tied = (a, b)
    a._purview_ties = tied
    b._purview_ties = tied
    assert a.num_purview_ties == 1
    restored = round_trip(a, fmt)
    assert restored.num_purview_ties == 1
    peers = [t for t in restored._purview_ties if t is not restored]
    assert len(peers) == 1
    assert peers[0]._purview_ties is restored._purview_ties


@pytest.mark.parametrize("fmt", FORMATS)
def test_mice_not_computed_ties_round_trip(fmt):
    obj = MaximallyIrreducibleCause(make_ria(Direction.CAUSE))
    assert obj._purview_ties is None
    assert np.isnan(obj.num_purview_ties)
    restored = round_trip(obj, fmt)
    assert restored._purview_ties is None
    assert np.isnan(restored.num_purview_ties)


@pytest.mark.parametrize("fmt", FORMATS)
def test_mice_computed_no_ties_round_trip(fmt):
    obj = MaximallyIrreducibleCause(make_ria(Direction.CAUSE))
    obj._purview_ties = (obj,)
    restored = round_trip(obj, fmt)
    assert restored._purview_ties == (restored,)
    assert restored.num_purview_ties == 0


def test_mice_loads_without_tie_field_as_not_computed():
    # A payload without the field decodes as "ties not computed", not as
    # a claim of zero ties.
    obj = MaximallyIrreducibleCause(make_ria(Direction.CAUSE))
    obj._purview_ties = (obj,)
    data = json.loads(serialize.dumps(obj, format="json"))

    def strip(o):
        if isinstance(o, dict):
            o.pop("purview_tie_peers", None)
            for v in o.values():
                strip(v)
        elif isinstance(o, list):
            for item in o:
                strip(item)

    strip(data)
    restored = serialize.loads(json.dumps(data).encode(), format="json")
    assert restored._purview_ties is None
    assert np.isnan(restored.num_purview_ties)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/serialize/test_serialize_ria_mice.py -v`
Expected: `test_mice_preserves_purview_ties` FAILS (`num_purview_ties` 0 ≠ 1); `test_mice_not_computed_ties_round_trip` FAILS (`_purview_ties` is `(restored,)`, not `None`). The other two pass only after implementation or trivially — run them anyway.

- [ ] **Step 3: Implement**

In `pyphi/serialize/schema.py`, append the same field to each of `MICESchema`, `MICECauseSchema`, and `MICEEffectSchema` (after `purview_margin`):

```python
    purview_tie_peers: tuple["MICEAnySchema", ...] | None = None
```

(`MICEAnySchema` is defined just below the three classes; the string forward reference resolves at first use, matching the existing `tuple["RIASchema", ...]` pattern.)

In `pyphi/serialize/convert.py`, replace `_decode_mice` and `_register_mice` with:

```python
def _mice_struct_cls(mice: Any) -> Any:
    from pyphi.models.mice import MaximallyIrreducibleCause
    from pyphi.models.mice import MaximallyIrreducibleEffect

    if isinstance(mice, MaximallyIrreducibleCause):
        return schema.MICECauseSchema
    if isinstance(mice, MaximallyIrreducibleEffect):
        return schema.MICEEffectSchema
    return schema.MICESchema


def _encode_mice(mice: Any, struct_cls: Any, *, include_peers: bool = True) -> Any:
    # Purview ties are tri-state: None = never computed; () = computed with
    # no ties; otherwise the tied peers excluding this MICE, each encoded
    # with its own tie field suppressed (the shared tie tuple contains this
    # MICE, so recursing into peers' ties would never terminate).
    peers: tuple | None = None
    if mice._purview_ties is not None:
        peers = (
            tuple(
                _encode_mice(t, _mice_struct_cls(t), include_peers=False)
                for t in mice._purview_ties
                if t is not mice
            )
            if include_peers
            else ()
        )
    return struct_cls(
        ria=to_schema(mice.ria),
        purview_margin=_enc_optional(mice.purview_margin),
        purview_tie_peers=peers,
    )


def _decode_mice(cls: type, struct: Any) -> Any:
    instance = cls(from_schema(struct.ria))
    if struct.purview_tie_peers is None:
        instance._purview_ties = None
    else:
        peers = tuple(from_schema(p) for p in struct.purview_tie_peers)
        tied = (instance, *peers)
        instance._purview_ties = tied
        for peer in peers:
            peer._purview_ties = tied
    instance.purview_margin = _dec_optional(struct.purview_margin)
    return instance


def _register_mice() -> None:
    from pyphi.models.mice import MaximallyIrreducibleCause
    from pyphi.models.mice import MaximallyIrreducibleCauseOrEffect
    from pyphi.models.mice import MaximallyIrreducibleEffect

    _ENCODERS[MaximallyIrreducibleCauseOrEffect] = lambda m: _encode_mice(
        m, schema.MICESchema
    )
    _ENCODERS[MaximallyIrreducibleCause] = lambda m: _encode_mice(
        m, schema.MICECauseSchema
    )
    _ENCODERS[MaximallyIrreducibleEffect] = lambda m: _encode_mice(
        m, schema.MICEEffectSchema
    )
    _DECODERS[schema.MICESchema] = lambda s: _decode_mice(
        MaximallyIrreducibleCauseOrEffect, s
    )
    _DECODERS[schema.MICECauseSchema] = lambda s: _decode_mice(
        MaximallyIrreducibleCause, s
    )
    _DECODERS[schema.MICEEffectSchema] = lambda s: _decode_mice(
        MaximallyIrreducibleEffect, s
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/serialize/test_serialize_ria_mice.py test/serialize/test_serialize_distinctions.py test/serialize/test_serialize_relations_ces.py -v`
Expected: all PASS (distinctions and CES embed MICEs, so run their suites too).

- [ ] **Step 5: Commit**

```bash
git add pyphi/serialize/schema.py pyphi/serialize/convert.py test/serialize/test_serialize_ria_mice.py
git commit -m "Serialize MICE purview ties, preserving the not-computed state"
```

---

### Task 4: `RunnerUpSchema`; IIT4 `runner_up`; IIT3 `runner_up`/`reasons`/`config`/`provenance`

**Files:**
- Modify: `pyphi/serialize/schema.py` (new `RunnerUpSchema` before `IIT3SIASchema`; fields on `IIT3SIASchema` lines 238–246 and `IIT4SIASchema` lines 253–270; add `RunnerUpSchema` to the `Schema` union at the bottom)
- Modify: `pyphi/serialize/convert.py` (`_encode_iit3_sia`/`_decode_iit3_sia` ~lines 490–522, `_encode_iit4_sia`/`_decode_iit4_sia` ~lines 583–642; new `_enc_runner_up`/`_dec_runner_up` next to `_enc_reasons`)
- Test: `test/serialize/test_serialize_sia.py`

**Interfaces:**
- Consumes: `RunnerUp` frozen dataclass with `partition`, `phi` (`pyphi/models/explanation.py`); `IIT3SystemIrreducibilityAnalysis.__init__(..., config=None, provenance=None, reasons=None, runner_up=None)`; IIT4 `SystemIrreducibilityAnalysis` dataclass field `runner_up: Any = None`; helpers `_enc_reasons`/`_dec_reasons`, `_enc_config`, `_enc_optional`/`_dec_optional`; `IIT_3_CONFIG` from `test/conftest.py`.
- Produces: `RunnerUpSchema(partition: PartitionSchema, phi: PhiSchema)`; `convert._enc_runner_up(runner_up) -> RunnerUpSchema | None` and `convert._dec_runner_up(struct) -> RunnerUp | None`; the four IIT3 fields and the IIT4 `runner_up` field round-trip.

- [ ] **Step 1: Write the failing tests**

Add to `test/serialize/test_serialize_sia.py`. New imports at the top: `from pyphi.models.explanation import NullResultReason`, `from pyphi.models.explanation import RunnerUp`, and `from ..conftest import IIT_3_CONFIG` (the same relative-conftest pattern `test/test_actual.py` uses; `test/serialize/__init__.py` exists).

```python
@pytest.mark.parametrize("fmt", FORMATS)
def test_iit3_sia_preserves_runner_up_and_reasons(fmt):
    obj = IIT3SystemIrreducibilityAnalysis(
        phi=0.5,
        partition=DirectedBipartition(Direction.CAUSE, (0,), (1,)),
        node_indices=(0, 1),
        current_state=(1, 0),
        reasons=[NullResultReason.NO_SYSTEM],
        runner_up=RunnerUp(
            partition=DirectedBipartition(Direction.CAUSE, (1,), (0,)), phi=0.75
        ),
    )
    restored = round_trip(obj, fmt)
    assert restored.runner_up == obj.runner_up
    assert restored.reasons == [NullResultReason.NO_SYSTEM]


@pytest.mark.parametrize("fmt", FORMATS)
def test_iit3_sia_preserves_config_and_provenance(fmt):
    with IIT_3_CONFIG:
        obj = IIT3SystemIrreducibilityAnalysis(
            phi=0.5,
            partition=DirectedBipartition(Direction.CAUSE, (0,), (1,)),
            node_indices=(0, 1),
            current_state=(1, 0),
        )
    restored = round_trip(obj, fmt)
    # config degrades to a plain dict, matching the IIT4 decoder.
    assert isinstance(restored.config, dict)
    assert restored.config["formalism"]["iit"]["version"] == "IIT_3_0"
    # provenance is the saved one, not freshly captured at load time.
    assert restored.provenance == obj.provenance


@pytest.mark.parametrize("fmt", FORMATS)
def test_iit4_sia_preserves_runner_up(fmt):
    obj = SystemIrreducibilityAnalysis(phi=0.5, partition=NullCut((0, 1)))
    obj.runner_up = RunnerUp(partition=NullCut((0, 1)), phi=0.75)
    restored = round_trip(obj, fmt)
    assert restored.runner_up == obj.runner_up


def test_iit3_sia_loads_without_new_fields():
    obj = IIT3SystemIrreducibilityAnalysis(
        phi=0.5,
        partition=DirectedBipartition(Direction.CAUSE, (0,), (1,)),
        node_indices=(0, 1),
        current_state=(1, 0),
    )
    data = json.loads(serialize.dumps(obj, format="json"))

    def strip(o):
        if isinstance(o, dict):
            for key in ("runner_up", "reasons", "config", "provenance"):
                o.pop(key, None)
            for v in o.values():
                strip(v)
        elif isinstance(o, list):
            for item in o:
                strip(item)

    strip(data)
    restored = serialize.loads(json.dumps(data).encode(), format="json")
    assert restored.runner_up is None
    assert restored.reasons == []
    # Nothing stored: the constructor still snapshots load-time context.
    assert restored.config is not None
    assert restored.provenance is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/serialize/test_serialize_sia.py -v`
Expected: the three new round-trip tests FAIL (`runner_up` is None; config records `IIT_4_0_2026`; provenance differs). The strip test passes only after implementation (it needs the JSON to contain the keys) — expect a pass either way; its job is pinning behavior after.

Note the config/provenance test FAILURE MODE before the fix: `restored.config` is a fabricated `ConfigSnapshot` (not a dict), so `isinstance(restored.config, dict)` fails — that is the bug being fixed.

- [ ] **Step 3: Implement**

In `pyphi/serialize/schema.py`, add before `IIT3SIASchema`:

```python
class RunnerUpSchema(msgspec.Struct, frozen=True, tag="runner_up"):
    partition: PartitionSchema
    phi: PhiSchema
```

Append to `IIT3SIASchema` (after `tie_peers`):

```python
    runner_up: RunnerUpSchema | None = None
    reasons: tuple[str, ...] | None = None
    config: dict[str, Any] | None = None
    provenance: ProvenanceSchema | None = None
```

Append to `IIT4SIASchema` (after `partition_margin`; `NullIIT4SIASchema` inherits it):

```python
    runner_up: RunnerUpSchema | None = None
```

Add `| RunnerUpSchema` to the `Schema` union at the bottom of the module (next to `ProvenanceSchema`).

In `pyphi/serialize/convert.py`, add next to `_enc_reasons`/`_dec_reasons`:

```python
def _enc_runner_up(runner_up: Any) -> Any:
    if runner_up is None:
        return None
    return schema.RunnerUpSchema(
        partition=to_schema(runner_up.partition),
        phi=to_schema(runner_up.phi),
    )


def _dec_runner_up(struct: Any) -> Any:
    if struct is None:
        return None
    from pyphi.models.explanation import RunnerUp

    return RunnerUp(
        partition=from_schema(struct.partition), phi=from_schema(struct.phi)
    )
```

In `_encode_iit3_sia`, add to the `schema.IIT3SIASchema(...)` call after `tie_peers=...`:

```python
        runner_up=_enc_runner_up(sia.runner_up),
        reasons=_enc_reasons(sia.reasons),
        config=_enc_config(sia.config),
        provenance=_enc_optional(sia.provenance),
```

In `_decode_iit3_sia`, add to the `IIT3SystemIrreducibilityAnalysis(...)` call after `current_state=...`:

```python
        runner_up=_dec_runner_up(struct.runner_up),
        reasons=_dec_reasons(struct.reasons),
        config=struct.config,
        provenance=_dec_optional(struct.provenance),
```

In `_encode_iit4_sia`, add after `partition_margin=...`:

```python
        runner_up=_enc_runner_up(sia.runner_up),
```

In `_decode_iit4_sia`, add to the `kwargs` dict after `"partition_margin": ...`:

```python
        "runner_up": _dec_runner_up(struct.runner_up),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/serialize/test_serialize_sia.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add pyphi/serialize/schema.py pyphi/serialize/convert.py test/serialize/test_serialize_sia.py
git commit -m "Serialize SIA runner_up and IIT3 SIA reasons, config, and provenance"
```

---

### Task 5: CES `config`/`provenance`

**Files:**
- Modify: `pyphi/serialize/schema.py` (`CESSchema`, line 340; `NullCESSchema` inherits)
- Modify: `pyphi/serialize/convert.py` (`_encode_ces`/`_decode_ces`, lines 744–762)
- Test: `test/serialize/test_serialize_relations_ces.py`

**Interfaces:**
- Consumes: `CauseEffectStructure` frozen dataclass with `config`/`provenance` fields and lazy `__post_init__` capture (`pyphi/models/ces.py`); the file's existing `make_ces()` helper; `pyphi.config.override`.
- Produces: `CESSchema` fields `config: dict[str, Any] | None = None`, `provenance: ProvenanceSchema | None = None`, threaded through encode/decode.

- [ ] **Step 1: Write the failing tests**

Add to `test/serialize/test_serialize_relations_ces.py` (add `import json` and `import pyphi` at the top):

```python
@pytest.mark.parametrize("fmt", FORMATS)
def test_ces_preserves_config_and_provenance(fmt):
    with pyphi.config.override(precision=7):
        obj = make_ces()
    restored = round_trip(obj, fmt)
    assert isinstance(restored.config, dict)
    assert restored.config["numerics"]["precision"] == 7
    assert restored.provenance == obj.provenance


def test_ces_loads_without_config_and_provenance():
    obj = make_ces()
    data = json.loads(serialize.dumps(obj, format="json"))

    def strip(o):
        if isinstance(o, dict):
            for key in ("config", "provenance"):
                o.pop(key, None)
            for v in o.values():
                strip(v)
        elif isinstance(o, list):
            for item in o:
                strip(item)

    strip(data)
    restored = serialize.loads(json.dumps(data).encode(), format="json")
    # Nothing stored: __post_init__ still snapshots load-time context.
    assert restored.config is not None
    assert restored.provenance is not None
```

Note the strip test removes `config`/`provenance` keys everywhere, including from the embedded IIT4 SIA — that is exactly the pre-existing-file shape, and the SIA constructor handles it the same way.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/serialize/test_serialize_relations_ces.py -v`
Expected: `test_ces_preserves_config_and_provenance` FAILS (`restored.config` is a fabricated `ConfigSnapshot`, so the isinstance check fails).

- [ ] **Step 3: Implement**

In `pyphi/serialize/schema.py`, append to `CESSchema`:

```python
    config: dict[str, Any] | None = None
    provenance: ProvenanceSchema | None = None
```

In `pyphi/serialize/convert.py`, `_encode_ces`: add to the `struct_cls(...)` call after `relations=...`:

```python
        config=_enc_config(ces.config),
        provenance=_enc_optional(ces.provenance),
```

In `_decode_ces`: add to the `domain_cls(...)` call after `relations=...`:

```python
        config=struct.config,
        provenance=_dec_optional(struct.provenance),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/serialize/test_serialize_relations_ces.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add pyphi/serialize/schema.py pyphi/serialize/convert.py test/serialize/test_serialize_relations_ces.py
git commit -m "Serialize CES config and provenance"
```

---

### Task 6: `Transition.noise_background` — schema field and equality contract

**Files:**
- Modify: `pyphi/actual.py` (`Transition.__eq__` and `__hash__`, ~lines 455–477)
- Modify: `pyphi/serialize/schema.py` (`TransitionSchema`, lines 375–381)
- Modify: `pyphi/serialize/convert.py` (`_register_transition`, lines 823–841)
- Test: `test/test_actual.py`, `test/serialize/test_serialize_substrate_ac.py`

**Interfaces:**
- Consumes: `Transition` dataclass field `noise_background: bool = False`; `Transition._ratio(direction, mechanism, purview)`.
- Produces: `TransitionSchema.noise_background: bool = False` round-tripped; `Transition.__eq__`/`__hash__` include `noise_background`.

- [ ] **Step 1: Write the failing tests**

Add to `test/test_actual.py` (near the existing noise_background tests, ~line 330):

```python
def test_transition_equality_includes_noise_background():
    substrate = examples.actual_causation_substrate()
    frozen = actual.Transition(substrate, (1, 1), (1, 1), (0,), (1,))
    noised = actual.Transition(
        substrate, (1, 1), (1, 1), (0,), (1,), noise_background=True
    )
    assert frozen != noised
    assert len({frozen, noised}) == 2
```

Add to `test/serialize/test_serialize_substrate_ac.py` (new imports at the top: `import numpy as np`, `from pyphi.direction import Direction`, `from pyphi.substrate import Substrate`):

```python
@pytest.mark.parametrize("fmt", FORMATS)
def test_transition_preserves_noise_background(fmt):
    # OR gate driven by a noised background unit: the EFFECT ratio is
    # nonzero only when noise_background survives the round-trip.
    substrate = Substrate(np.array([[0, 0], [1, 1], [1, 1], [1, 1]]))
    obj = Transition(substrate, (1, 1), (1, 1), (0,), (1,), noise_background=True)
    restored = round_trip(obj, fmt)
    assert restored.noise_background is True
    assert restored == obj
    original_ratio = obj._ratio(Direction.EFFECT, (0,), (1,))
    assert restored._ratio(Direction.EFFECT, (0,), (1,)) == original_ratio
    assert original_ratio != 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/test_actual.py::test_transition_equality_includes_noise_background "test/serialize/test_serialize_substrate_ac.py::test_transition_preserves_noise_background" -v`
Expected: equality test FAILS (`frozen == noised`); round-trip test FAILS (`restored.noise_background` is False).

- [ ] **Step 3: Implement**

In `pyphi/actual.py`, `Transition.__eq__`: add to the conjunction:

```python
            and self.noise_background == other.noise_background
```

In `Transition.__hash__`: add `self.noise_background,` to the hashed tuple (after `self.partition,`).

In `pyphi/serialize/schema.py`, append to `TransitionSchema`:

```python
    noise_background: bool = False
```

In `pyphi/serialize/convert.py`, `_register_transition`: add `noise_background=t.noise_background,` to **both** the encoder's `schema.TransitionSchema(...)` call and the decoder's `Transition(...)` call.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/test_actual.py test/serialize/test_serialize_substrate_ac.py -v`
Expected: all PASS (the full `test_actual.py` run checks the equality change against existing AC behavior).

- [ ] **Step 5: Commit**

```bash
git add pyphi/actual.py pyphi/serialize/schema.py pyphi/serialize/convert.py test/test_actual.py test/serialize/test_serialize_substrate_ac.py
git commit -m "Round-trip Transition noise_background and include it in equality"
```

---

### Task 7: Null AcSIA constructs real `Account` objects

**Files:**
- Modify: `pyphi/models/actual_causation.py` (`_null_ac_sia`, ~line 856)
- Modify: `test/test_actual.py` (`test_null_ac_sia`, ~lines 780–790)
- Test: `test/serialize/test_serialize_substrate_ac.py`

**Interfaces:**
- Consumes: `Account` (defined in the same module); `actual._null_ac_sia(transition, direction, alpha=0.0, reasons=None)`.
- Produces: `_null_ac_sia` passes `account=Account(())` and `partitioned_account=Account(())`; null AC SIAs serialize without error.

- [ ] **Step 1: Write the failing test**

Add to `test/serialize/test_serialize_substrate_ac.py`:

```python
@pytest.mark.parametrize("fmt", FORMATS)
def test_null_ac_sia_round_trips(fmt):
    sia = actual._null_ac_sia(make_transition(), Direction.CAUSE)
    restored = round_trip(sia, fmt)
    assert restored == sia
    assert len(restored.account) == 0
    assert len(restored.partitioned_account) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest "test/serialize/test_serialize_substrate_ac.py::test_null_ac_sia_round_trips" -v`
Expected: FAIL with `TypeError: No serializer registered for tuple`.

- [ ] **Step 3: Implement, and update the pinning test**

In `pyphi/models/actual_causation.py`, `_null_ac_sia`: replace `account=(),` and `partitioned_account=(),` with:

```python
        account=Account(()),
        partitioned_account=Account(()),
```

In `test/test_actual.py`, `test_null_ac_sia`: replace the two account assertions with:

```python
    assert sia.account == models.Account(())
    assert sia.partitioned_account == models.Account(())
    assert len(sia.account) == 0
```

(`models` is already imported in that file; `Account` is re-exported from `pyphi.models`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/test_actual.py test/serialize/test_serialize_substrate_ac.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add pyphi/models/actual_causation.py test/test_actual.py test/serialize/test_serialize_substrate_ac.py
git commit -m "Construct null AcSIA accounts as Account objects, fixing serialization"
```

---

### Task 8: AC completeness — AcRIA `node_labels`/`reasons`; AcSIA `reasons`/`config`/`provenance`/ties

**Files:**
- Modify: `pyphi/serialize/schema.py` (`AcRIASchema` lines 387–396, `AcSIASchema` lines 416–428)
- Modify: `pyphi/serialize/convert.py` (`_encode_ac_ria`/`_decode_ac_ria` ~lines 844–880, `_register_ac_sia` ~lines 941–971)
- Test: `test/serialize/test_serialize_substrate_ac.py`

**Interfaces:**
- Consumes: `AcRepertoireIrreducibilityAnalysis.__init__(..., node_labels=None, reasons=None)`; `AcSystemIrreducibilityAnalysis.__init__(..., config=None, provenance=None, reasons=None)` with `.ties` property and `._ties` attribute; helpers `_enc_optional_direction`, `_enc_reasons`/`_dec_reasons`, `_enc_config`, `_opt_tuple`.
- Produces: `AcRIASchema` gains `node_labels: NodeLabelsSchema | None = None` and `reasons: tuple[str, ...] | None = None`; `AcSIASchema` gains `reasons: tuple[str, ...] | None = None`, `config: dict[str, Any] | None = None`, `provenance: ProvenanceSchema | None = None`, `tie_peers: tuple["AcSIASchema", ...] = ()`; new `convert._encode_ac_sia(s, *, include_peers)` and `convert._decode_ac_sia(struct)`.

- [ ] **Step 1: Write the failing tests**

Add to `test/serialize/test_serialize_substrate_ac.py` (new imports at the top: `import json`, `from pyphi.models.explanation import NullResultReason`):

```python
@pytest.mark.parametrize("fmt", FORMATS)
def test_ac_ria_preserves_node_labels(fmt):
    s = actual.sia(make_transition())
    link = list(s.account)[0]
    assert link.ria.node_labels is not None
    restored = round_trip(s, fmt)
    rlink = list(restored.account)[0]
    assert rlink.ria.node_labels == link.ria.node_labels


@pytest.mark.parametrize("fmt", FORMATS)
def test_ac_sia_preserves_reasons_ties_config_provenance(fmt):
    t = make_transition()
    a = actual._null_ac_sia(t, Direction.CAUSE, reasons=[NullResultReason.NO_SYSTEM])
    b = actual._null_ac_sia(t, Direction.CAUSE, alpha=0.5)
    a.set_ties([a, b])
    restored = round_trip(a, fmt)
    assert restored.reasons == [NullResultReason.NO_SYSTEM]
    peers = [p for p in restored.ties if p is not restored]
    assert len(peers) == 1
    assert isinstance(restored.config, dict)
    assert restored.provenance == a.provenance


def test_ac_sia_loads_without_new_fields():
    sia = actual._null_ac_sia(
        make_transition(), Direction.CAUSE, reasons=[NullResultReason.NO_SYSTEM]
    )
    data = json.loads(serialize.dumps(sia, format="json"))

    def strip(o):
        if isinstance(o, dict):
            for key in ("reasons", "config", "provenance", "tie_peers"):
                o.pop(key, None)
            for v in o.values():
                strip(v)
        elif isinstance(o, list):
            for item in o:
                strip(item)

    strip(data)
    restored = serialize.loads(json.dumps(data).encode(), format="json")
    assert restored.reasons == []
    assert restored.ties == (restored,)
    # Nothing stored: the constructor still snapshots load-time context.
    assert restored.config is not None
    assert restored.provenance is not None
```

(The strip test must not remove `node_labels`: `AcSIASchema.node_labels` is a pre-existing required field.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/serialize/test_serialize_substrate_ac.py -v`
Expected: `test_ac_ria_preserves_node_labels` FAILS (labels come back None); `test_ac_sia_preserves_reasons_ties_config_provenance` FAILS (reasons come back `[]`).

- [ ] **Step 3: Implement**

In `pyphi/serialize/schema.py`, append to `AcRIASchema` (after `partition_tie_peers`):

```python
    node_labels: NodeLabelsSchema | None = None
    reasons: tuple[str, ...] | None = None
```

Append to `AcSIASchema` (after `node_labels`):

```python
    reasons: tuple[str, ...] | None = None
    config: dict[str, Any] | None = None
    provenance: ProvenanceSchema | None = None
    tie_peers: tuple["AcSIASchema", ...] = ()
```

In `pyphi/serialize/convert.py`:

`_encode_ac_ria`: add to the `schema.AcRIASchema(...)` call after `partition_tie_peers=...`:

```python
        node_labels=_enc_optional(ria.node_labels),
        reasons=_enc_reasons(ria.reasons),
```

`_decode_ac_ria`: add to the `AcRepertoireIrreducibilityAnalysis(...)` call after `partitioned_probability=...`:

```python
        node_labels=_dec_optional(struct.node_labels),
        reasons=_dec_reasons(struct.reasons),
```

Replace `_register_ac_sia` with:

```python
def _encode_ac_sia(s: Any, *, include_peers: bool) -> Any:
    peers = tuple(t for t in s.ties if t is not s) if include_peers else ()
    return schema.AcSIASchema(
        alpha=None if s.alpha is None else float(s.alpha),
        direction=_enc_optional_direction(s.direction),
        account=_enc_optional(s.account),
        partitioned_account=_enc_optional(s.partitioned_account),
        partition=_enc_optional(s.partition),
        before_state=_opt_tuple(s.before_state),
        after_state=_opt_tuple(s.after_state),
        size=s.size,
        node_indices=_opt_tuple(s.node_indices),
        cause_indices=_opt_tuple(s.cause_indices),
        effect_indices=_opt_tuple(s.effect_indices),
        node_labels=_enc_optional(s.node_labels),
        reasons=_enc_reasons(s.reasons),
        config=_enc_config(s.config),
        provenance=_enc_optional(s.provenance),
        tie_peers=tuple(_encode_ac_sia(p, include_peers=False) for p in peers),
    )


def _decode_ac_sia(struct: Any) -> Any:
    from pyphi.models.actual_causation import AcSystemIrreducibilityAnalysis

    instance = AcSystemIrreducibilityAnalysis(
        alpha=struct.alpha,
        direction=_dec_optional(struct.direction),
        account=_dec_optional(struct.account),
        partitioned_account=_dec_optional(struct.partitioned_account),
        partition=_dec_optional(struct.partition),
        before_state=_opt_tuple(struct.before_state),
        after_state=_opt_tuple(struct.after_state),
        size=struct.size,
        node_indices=_opt_tuple(struct.node_indices),
        cause_indices=_opt_tuple(struct.cause_indices),
        effect_indices=_opt_tuple(struct.effect_indices),
        node_labels=_dec_optional(struct.node_labels),
        reasons=_dec_reasons(struct.reasons),
        config=struct.config,
        provenance=_dec_optional(struct.provenance),
    )
    if struct.tie_peers:
        peers = tuple(_decode_ac_sia(p) for p in struct.tie_peers)
        tied = (instance, *peers)
        instance._ties = tied
        for peer in peers:
            peer._ties = tied
    return instance


def _register_ac_sia() -> None:
    from pyphi.models.actual_causation import AcSystemIrreducibilityAnalysis

    _ENCODERS[AcSystemIrreducibilityAnalysis] = lambda s: _encode_ac_sia(
        s, include_peers=True
    )
    _DECODERS[schema.AcSIASchema] = _decode_ac_sia
```

Note `_dec_reasons(None)` returns `None` and the constructor's `reasons or []` turns it into `[]` — old files land on the documented default.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/serialize/test_serialize_substrate_ac.py test/test_actual.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add pyphi/serialize/schema.py pyphi/serialize/convert.py test/serialize/test_serialize_substrate_ac.py
git commit -m "Serialize AC node_labels, reasons, config, provenance, and alpha ties"
```

---

### Task 9: Changelog fragment and full-suite gate

**Files:**
- Create: `changelog.d/serialize-round-trip-completeness.fix.md`

**Interfaces:**
- Consumes: everything above.
- Produces: a green pathless full suite and the user-facing changelog entry.

- [ ] **Step 1: Write the changelog fragment**

Create `changelog.d/serialize-round-trip-completeness.fix.md`:

```markdown
Serialization now round-trips every computed field the result objects carry.
Previously dropped silently: RIA `signed_phi` (negative preventative-cause
values were re-clamped to 0 on reload), `selectivity`, and `reasons`; MICE
purview ties (a never-computed tie state was also rewritten as "no ties");
IIT 3.0 and IIT 4.0 SIA `runner_up`; IIT 3.0 SIA `reasons`, `config`, and
`provenance`; CES `config` and `provenance`; actual-causation `node_labels`,
`reasons`, and alpha-tie sets; and `Transition.noise_background` (a noised
transition reloaded as frozen, changing α). Loaded IIT 3.0, AC, and CES
results no longer claim the loader's config and load time as their
provenance. Null actual-causation results (empty accounts) now serialize
instead of raising `TypeError`, `Transition` equality and hashing now include
`noise_background`, and `pyphi.load`/`pyphi.serialize.loads` reject files
written by a newer serialization format instead of silently dropping their
fields.
```

- [ ] **Step 2: Run the full suite (pathless, from the worktree)**

```bash
uv run pytest -q > /tmp/serialize_full.log 2>&1; tail -5 /tmp/serialize_full.log
```

Then **Read the end of `/tmp/serialize_full.log`** and confirm the summary line reports only passes/skips (baseline: 3660 passed, 286 skipped before this branch; expect the count to grow by the new tests). Investigate any failure before proceeding.

- [ ] **Step 3: Commit**

```bash
git add changelog.d/serialize-round-trip-completeness.fix.md
git commit -m "Add changelog fragment for serialization round-trip completeness"
```

---

## Self-review notes

- Spec coverage: finding 1–2 → Task 2; finding 3 → Task 3; finding 4 → Task 4; finding 5 → Tasks 4–5 (IIT3/CES) and 8 (AcSIA); finding 6 → Task 8; finding 7 → Task 6; finding 8 → Task 7; finding 9 → Task 1.
- Type consistency: `_enc_runner_up`/`_dec_runner_up` (Task 4) are used only within Task 4; `_encode_mice`/`_mice_struct_cls` (Task 3) only within Task 3; `_encode_ac_sia`/`_decode_ac_sia` (Task 8) only within Task 8. Field names in tests match the schema fields added in the same task.
- Ordering: Task 7 (null-AcSIA crash) precedes Task 8, whose tie/reasons tests round-trip null AcSIAs.
