# P14d — `to_pandas` consolidation (design)

## Problem

`to_pandas()` on PyPhi result objects is split between two incompatible
implementations:

- A **generic heuristic** `ToPandasMixin.to_pandas()` (`pyphi/models/pandas.py`)
  that introspects `to_json()`/`to_dict()` and runs `pd.json_normalize`. Column
  names are dotted JSON paths, there is no deliberate index, units appear as raw
  integer indices, and the return type (`Series` vs `DataFrame`) is guessed from
  whether the normalized frame has one row. Used by `RepertoireIrreducibilityAnalysis`
  (RIA), `MaximallyIrreducibleCauseOrEffect` (MICE), `Distinction`, `Distinctions`,
  `StateSpecification`, and `SystemStateSpecification`.
- A **bespoke labeled** `TriggeredTPM.to_pandas()` (`pyphi/matching/triggered_tpm.py`)
  with a deliberate `MultiIndex` (stimulus states × system states) named by node
  labels — carrying an in-code note that it is provisional and "subsumed by the
  unified to_pandas project".

The result is an inconsistent, unlabeled export surface. Because this surface is
public and is the substrate that `result.explain()`, `result.diff()`, and the
substrate↔networkx bridge will build on, it must be made consistent before the
public-surface freeze.

## Goal

One documented labeled-export **convention** — consistent per-category schema,
units always rendered as labels — applied to every current `ToPandasMixin` user,
with `TriggeredTPM` reconciled onto the same convention. Adding new types later
is a few lines against the convention.

### Out of scope

- Unifying the separate `_repr_html_` path (`SIA`, `CES`, `ActualCausation`, the
  IIT 4.0 SIA) — that display path is untouched here.
- Adding `to_pandas()` to top-level result types that lack it (`SIA`, `CES`,
  `ActualCausation`).
- Relation-face visualization.
- A `spread_state=True` opt-in (see "Distribution types" below) — designed-for but
  not built now.

## Convention

The shape is **determined by the object's category**. Four categories cover every
current user, each matched to a genuinely different mathematical object:

| Category | Objects | Return |
|---|---|---|
| Scalar record | `RIA`, `MICE`, `Distinction` | `Series` |
| Collection | `Distinctions` | key-indexed `DataFrame` |
| 1-D distribution (+metadata) | `StateSpecification`, `SystemStateSpecification` | long/tidy `DataFrame` |
| 2-D conditional | `TriggeredTPM` | wide labeled-matrix `DataFrame` |

Consistency lives in the **rules** (fixed schema within a category, units always
labeled), not in one universal shape. Two invariants hold across all categories:

1. **Units are always rendered as labels**, never raw integer indices, via the
   existing `NodeLabels.coerce_to_labels` / `NodeLabels.label_string`. Missing
   labels fall back to integers (already `coerce_to_labels`'s behavior).
2. The index/columns are **meaningful and labeled** — never a dotted-path
   `json_normalize` artifact, never a meaningless `RangeIndex` standing in for a
   real key.

## Architecture

Thin mixin + per-class hook + pure helper layer (`pyphi/models/pandas.py`).

- **`ToPandasMixin.to_pandas()`** — the single public entry point. Documents the
  contract and delegates to `self._to_pandas()`. The `json_normalize` heuristic and
  the `len(df) == 1` Series-guessing are **deleted**.
- **`ToPandasMixin._to_pandas()`** — protected per-class hook. Base implementation
  raises `NotImplementedError(type(self).__name__)`; no silent fallback.
- **Pure helper functions** (module-level in `models/pandas.py`), composing the
  existing `NodeLabels` utilities. They own all `Series`/`DataFrame`/`MultiIndex`
  construction so the schema lives in exactly one place:
  - `record_to_series(record: Mapping[str, Any], name: str | None) -> pd.Series`
  - `records_to_frame(rows: Iterable[Mapping[str, Any]], index: str | None) -> pd.DataFrame`
  - `distribution_rows(direction, kind, purview_labels, repertoire, state_space) -> list[dict]`
    — yields tidy `{direction, kind, purview, state, probability}` rows for one
    repertoire. **k-ary aware**: states are enumerated from the repertoire array
    shape / state space, never assumed binary.
  - `state_multiindex(node_labels, indices, state_space) -> pd.MultiIndex` — a
    `MultiIndex` over all states of the given units, level-named by label.
    **k-ary aware** (per-unit cardinality from the state space). Used by the 2-D
    conditional (`TriggeredTPM`) for both axes.

Each object builds a small ordered record (or a labeled distribution) of
*already-labeled* values, then a helper wraps it. No object touches pandas
internals directly.

## Per-type schema

### Scalar record → `Series`

`RIA` / `MICE` — `_to_pandas()` returns `record_to_series(record, name=type name)`
where the curated record is:

```
phi, direction, mechanism (labels), purview (labels),
mechanism_state (tuple), purview_state (tuple), specified_state
```

Repertoire **arrays are excluded** — they are distributions, available on the
object; the record is the scalar summary (deliberately a curated subset, *not*
`_dict_attrs`). `MICE` delegates to its wrapped `ria`'s record.

`Distinction` — `_to_pandas()` returns a `Series`:

```
phi, mechanism (labels), mechanism_state (tuple),
cause_purview (labels), effect_purview (labels)
```

`mechanism` is the labeled tuple (e.g. `('A','B')`); `mechanism_state` is the raw
state tuple (e.g. `(1,0)`). Together they reconstruct the labeled mechanism via
`label_string` if a caller wants it.

### Collection → key-indexed `DataFrame`

`Distinctions` — one row per concept (each concept's record), with the labeled
**`mechanism` as the index** (index name `"mechanism"`). Mechanisms are unique
within a CES, so the index is a valid unique key; this powers index-aligned
comparison (`result.diff`). `reset_index()` recovers a default range index for
callers who prefer it; this is documented on the method.

### 1-D distribution (+metadata) → long/tidy `DataFrame`

Fixed schema, identical across every distribution object and direction:

```
columns: [direction, kind, purview, state, probability]
```

- `direction` — `"CAUSE"` / `"EFFECT"` (a real column, robust to `concat`).
- `kind` — `"repertoire"` or `"unconstrained"`.
- `purview` — labeled tuple, e.g. `('B','C')`.
- `state` — raw state tuple, e.g. `(0,1)`; `dict(zip(purview, state))` gives the
  labeled mapping.
- `probability` — float.

One row per `(kind, state)`. The single packed `state` column (rather than one
column per purview unit) keeps the schema fixed across objects with different
purviews and makes the pair concat (below) clean; a future `spread_state=True`
flag can expand it to per-unit columns as a pure transform without disturbing the
default. Scalar metadata (`intrinsic_information`, specified state) stays on the
object — the frame is the distribution view.

`StateSpecification` — emits the rows above for its `repertoire` and
`unconstrained_repertoire`.

`SystemStateSpecification` — the cause/effect pair is simply:

```python
pd.concat([self.cause.to_pandas(), self.effect.to_pandas()], ignore_index=True)
```

The differing cause/effect purviews coexist with no ragged columns because each
row carries its own `purview`/`state`; `direction` distinguishes the halves. This
*reuses* `StateSpecification`'s implementation rather than duplicating field logic.

### 2-D conditional → wide labeled-matrix `DataFrame`

`TriggeredTPM` — keeps its current wide matrix (rows = stimulus states, columns =
system states, values = `Pr(s | x)`) and its output stays **byte-identical**, but
its index/columns are rebuilt through the shared `state_multiindex` helper so it
shares the convention. It remains a `matching/` dataclass conforming to the
`to_pandas()` contract (importing the helper from `models.pandas`). The provisional
note is removed.

## Error handling

- `_to_pandas()` unimplemented → `NotImplementedError(class name)`.
- Null/empty objects (a null `Distinction`, an empty `Distinctions`) → an **empty**
  `Series`/`DataFrame` with the correct index/column names, not an error.
- Missing node labels → integer fallback (already `coerce_to_labels`).

## Testing

- **Per-type unit tests** on `basic_network` (binary): assert exact index/column
  **labels**, the per-category shape, and dtypes for each of the six model users.
- **Shared contract test**, parametrized over every `ToPandasMixin` user: returns
  the declared type for its category; the index/columns are labeled (no dotted-path
  columns); every unit renders as a label string, never a raw int.
- **k-ary test**: `distribution_rows` / `state_multiindex` on the multivalued
  `gomez_p53_mdm2_substrate` (ternary `p53`) yields correct ternary states — guards
  against a binary assumption.
- **`TriggeredTPM` reconciliation guard**: the rebuilt frame equals the current
  output exactly (`pd.testing.assert_frame_equal`), locking the byte-identical claim.
- **Pair test**: `SystemStateSpecification.to_pandas()` equals the `concat` of its
  parts and has the fixed five-column schema.
- `to_json` / `jsonify` outputs unchanged (this touches only the pandas path).

## Migration

2.0 is a breaking release; the old `json_normalize` output shape changes with no
deprecation shim. A `changelog.d/*.change.md` fragment documents the new convention
and the `Series`-vs-`DataFrame`-by-category rule.
