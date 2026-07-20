# SweepResult / OptimizationResult serialization

**Date:** 2026-07-19
**Status:** Approved
**Roadmap:** Capabilities (small builds), item M12

## 1. Problem

Neither `SweepResult` (`pyphi/sweep.py`) nor `OptimizationResult`
(`pyphi/optimize.py`) can be loaded back from disk. `SweepResult` has no
persistence at all. `OptimizationResult.save` writes a bespoke summary JSON
that silently drops the winning substrate and SIA — the two objects that
justify the run — and there is no corresponding load path. Both classes are
batch-run outputs that are expensive to recompute, exactly the objects
`pyphi.serialize` exists for.

## 2. Decisions (settled during design review)

1. **DataFrames are embedded as parquet bytes via pyarrow.** Chosen for
   interoperability and to minimize bespoke codec maintenance, over an
   npy-bytes-per-column codec and over pandas `to_json(orient="table")`.
2. **pyarrow is a core dependency**, not an extra. No ImportError paths to
   maintain; parquet becomes the project-wide DataFrame-output convention.
3. **The bespoke `OptimizationResult.save` is deleted**, replaced by the
   `Serializable` mixin (full fidelity). No back-compat shim.
4. **Scope is these two classes only.** Converting other DataFrame disk
   outputs (benchmark CSVs, experiment scripts, the future provenance
   writer) to parquet is recorded as a ROADMAP wishlist entry, not built
   here.

## 3. Confirmed facts driving the design

- **msgspec JSON corrupts NaN**: `msgspec.json.encode` emits `null` for
  NaN/inf, and decoding `null` as `float` raises. The trajectory
  (`objective`, margins) and `best_objective` legitimately contain NaN, so
  naive float fields are unsound in the JSON wire format. Parquet preserves
  NaN bit-exactly.
- **Parquet round-trips the real table shapes exactly.** Verified with
  `pd.testing.assert_frame_equal` on all three shapes that occur:
  MultiIndex with tuple-valued levels (multi-axis sweep), single named
  index of tuples built with `tupleize_cols=False` (single-axis sweep), and
  RangeIndex (trajectory) — using reset-index-before-write plus tuple
  repair on read (see §4). Dtypes (int64, bool, float64), None cells, and
  NaN all survive.
- **msgspec resolves forward-referenced recursive tagged unions**: a Struct
  field annotated `tuple["Schema", ...]` where `Schema` is the union alias
  defined later in the module decodes correctly. This lets
  `SweepResultSchema.results` hold any registered payload type.

## 4. DataFrame codec (`pyphi/serialize/frames.py`)

A new module mirroring `arrays.py`, exposing two functions used only by
`convert.py`:

- `dataframe_to_schema(df) -> DataFrameSchema`:
  1. If the frame has any named index level, `reset_index()`; record the
     level names as `index_columns` (empty tuple for a RangeIndex).
  2. Record `tuple_columns`: object-dtype columns in which any non-null
     cell is a `tuple` (inspected at encode time, never guessed at decode).
  3. Write parquet bytes with `to_parquet(buffer, engine="pyarrow",
     index=False)`.
- `schema_to_dataframe(schema) -> pd.DataFrame`:
  1. `read_parquet` from the bytes.
  2. For each recorded tuple column, map each non-null cell (a list/array
     after parquet) back to a tuple of Python scalars (`.item()` for numpy
     scalars).
  3. If `index_columns` is non-empty, `set_index(list(index_columns))`; for
     a single level, rebuild as `pd.Index(..., tupleize_cols=False)` so
     tuple entries stay scalar index entries.

The schema Struct (in `schema.py`):

```python
class DataFrameSchema(msgspec.Struct, frozen=True, tag="dataframe"):
    parquet: bytes
    index_columns: tuple[str, ...] = ()
    tuple_columns: tuple[str, ...] = ()
```

`DataFrameSchema` joins the `Schema` union (every Struct in the module
does), but it is an internal representation: no domain type maps to it
directly.

## 5. Result schemas (`pyphi/serialize/schema.py`)

```python
class SweepResultSchema(msgspec.Struct, frozen=True, tag="sweep_result"):
    df: DataFrameSchema
    results: tuple["Schema | float", ...]
    skipped: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...]


class OptimizationResultSchema(
    msgspec.Struct, frozen=True, tag="optimization_result"
):
    best_params: bytes                     # .npy via arrays helper
    best_objective: float | None           # None encodes NaN (see below)
    best_substrate: SubstrateSchema
    best_sia: SIASchema | None
    trajectory: DataFrameSchema
    bounds: tuple[tuple[float, float], ...]
    seed: int
    direction: str
    objective_name: str
    settings: dict[str, Any]
    config_snapshot: dict[str, Any]
    n_evaluations: int
    n_unreachable: int
```

- `results` elements are the already-registered payload encodings of
  whatever the sweep computed (SIA and CES types for `compute="sia"/"ces"`;
  a callable compute may produce any registered type or a bare float). The
  exact forward-reference spelling (e.g. a `Schema | float` alias defined
  beside the union) is an implementation detail; the requirement is that
  elements decode back to tagged Structs, not dicts.
- `best_objective` is NaN exactly when every candidate was dynamically
  unreachable (`best_sia is None`); it is stored as `None` and restored to
  `math.nan` on decode. This is the one scalar float field that can be NaN,
  and mapping it through `None` keeps the JSON wire format honest.
- `best_params` never contains NaN but is stored as `.npy` bytes for exact
  float64 round-trip, consistent with every other array field.

Both Structs join the `Schema` union.

## 6. Converters (`pyphi/serialize/convert.py`)

`_register_sweep_result()` and `_register_optimization_result()`, following
the module's existing pattern (one `_register_<type>()` per type populating
`_ENCODERS` by domain type and `_DECODERS` by schema Struct):

- Sweep encode: `dataframe_to_schema(r.df)`, `_encode(obj)` per results
  element (floats pass through), `skipped` tuples coerced to the typed
  shape. Decode reverses each. An unregistered results element fails with
  the registry's existing `TypeError` naming the type.
- Optimization encode/decode: field-by-field, reusing `_encode`/`_decode`
  for the substrate and SIA, `array_to_bytes`/`bytes_to_array` for
  `best_params`, and the NaN↔None mapping for `best_objective`.

Node-labels frame handling needs no new logic: nested substrate/SIA
encodings already carry or inherit the document frame through the existing
`_encode`/`_decode` machinery.

## 7. Domain-class changes

- `SweepResult` and `OptimizationResult` gain the `Serializable` base
  (`pyphi/serializable.py`), acquiring `save(target, format=None)` and
  `load(target, format=None)` with wire-format inference (`.json`,
  `.msgpack`/`.mpk`, `.gz`) exactly as SIA/CES/Substrate have.
- The bespoke `OptimizationResult.save` method and its docstring are
  deleted. Its class docstring keeps the Attributes section unchanged.
- Both remain frozen dataclasses; inheriting the plain `Serializable` class
  is compatible. Note for tests: dataclass `==` is unusable on these
  classes (DataFrame fields raise on ambiguous truth); round-trip
  assertions compare field-by-field.

## 8. Dependency

`pyarrow>=25.0` is added to the core `dependencies` list in
`pyproject.toml` (25.0.0 is what currently resolves and what the design
was verified against). No extras change.

## 9. Tests (`test/serialize/test_serialize_results.py`)

1. **Sweep round-trips**: multi-axis sweep (two formalisms × subsets ×
   states on a small example substrate — MultiIndex with tuple levels),
   single-axis sweep (single named tuple index), and `compute="ces"`.
   Assert: `assert_frame_equal` on `df`, element-wise equality of
   `results` (existing domain equality), `skipped` equality, for both wire
   formats and a `.gz` path.
2. **Optimization round-trip**: a tiny seeded `optimize` run (1–2
   parameters, minimal popsize/maxiter). Assert every field: params
   (exact array equality), objective, substrate and SIA equality,
   `assert_frame_equal` on the trajectory (NaN rows included if present),
   bounds/seed/direction/objective_name/settings/config_snapshot/counts.
3. **All-unreachable NaN mapping**: a directly-constructed
   `OptimizationResult` with `best_objective=nan`, `best_sia=None`;
   round-trip restores NaN (`math.isnan`) and None.
4. **Mixin behavior**: `SweepResult.load` on a file holding a different
   type raises the mixin's `TypeError`.
5. **Formalism pinning**: any test computing φ pins its formalism via the
   preset-sourced context managers, per the suite convention.

## 10. Docs, changelog, ROADMAP

- Changelog fragment `changelog.d/result-serialization.feature.md`.
- Update the serialization docs page that enumerates supported types (and
  the MCP content surface if it lists them) to include both classes.
- ROADMAP: mark M12 landed; add a wishlist entry for the project-wide
  parquet convention for DataFrame disk outputs (benchmark aggregate
  outputs, experiment scripts, the provenance writer build), noting
  pyarrow is now a core dependency.

## 11. Out of scope

- Converting any existing CSV/JSON DataFrame outputs elsewhere in the
  repository.
- Standalone `.parquet` file outputs (the embedded document format is the
  deliverable here).
- Serialization of arbitrary unregistered callable-compute results (fails
  loudly, as today, with the registry's `TypeError`).
