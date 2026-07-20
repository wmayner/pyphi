# Provenance writers for experiment scripts

**Date:** 2026-07-20
**Status:** Approved design, pending implementation plan

## 1. Motivation

Experiment and benchmark scripts in this repository repeatedly re-implement the
same output-file mechanics: encode parameters (seed, trial counts, run label)
into a filename, avoid overwriting existing files by appending `_v2`/`_v3`
suffixes, and embed run metadata in the saved record. At least three
near-identical copies of a `unique_path` helper exist
(`benchmarks/iit_3_vs_4/harness.py`, `benchmarks/b18_dispatch_gate.py`,
`benchmarks/iit_3_vs_4/p18_inversion_share.py`), and none of the scripts record
the git SHA — even though `pyphi.provenance.Provenance.capture()` already
captures the pyphi version, git SHA and dirty flag, timestamp, interpreter and
library versions, platform, and seed.

This design adds script-facing writer functions to `pyphi.provenance` that
consolidate the pattern: parameters in the filename, no-clobber versioning, and
a full `Provenance` record embedded in every output file. The DataFrame writer
uses parquet, the project convention for DataFrame disk outputs.

## 2. Scope

- New public functions in `pyphi/provenance.py`: `format_stem`, `unique_path`,
  `save_json`, `save_npz`, `save_dataframe`, `read_metadata`.
- Migration of the three scripts above to the shared helpers (helpers only —
  their record formats are unchanged; see §7).
- Tests, a how-to section, a changelog fragment, and the ROADMAP row.

Out of scope: rewrapping existing scripts' record layouts in the new JSON
envelope; a generic loader beyond `read_metadata`; lifting the writers to the
top-level `pyphi` namespace.

## 3. API

All functions live in `pyphi/provenance.py` and are added to `__all__`.

```python
def format_stem(
    name: str,
    params: Mapping[str, Any] | None = None,
    run_label: str | None = None,
) -> str

def unique_path(directory: Path | str, stem: str, suffix: str) -> Path

def save_json(
    data: Any,
    directory: Path | str,
    name: str,
    *,
    params: Mapping[str, Any] | None = None,
    run_label: str | None = None,
    seed: int | None = None,
    note: str | None = None,
) -> Path

def save_npz(
    arrays: Mapping[str, np.ndarray],
    directory: Path | str,
    name: str,
    *,
    params: Mapping[str, Any] | None = None,
    run_label: str | None = None,
    seed: int | None = None,
    note: str | None = None,
) -> Path

def save_dataframe(
    df: pd.DataFrame,
    directory: Path | str,
    name: str,
    *,
    params: Mapping[str, Any] | None = None,
    run_label: str | None = None,
    seed: int | None = None,
    note: str | None = None,
) -> Path

def read_metadata(path: Path | str) -> dict[str, Any]
```

### Shared writer behavior

Every `save_*` function:

1. Builds the filename stem with `format_stem(name, params, run_label)`.
2. Resolves the output path with `unique_path(directory, stem, suffix)`
   (suffix `.json` / `.npz` / `.parquet`), creating the directory if needed
   and never overwriting an existing file.
3. Captures a `Provenance` via `Provenance.capture(seed=resolved_seed)`,
   where `resolved_seed` is the `seed` keyword argument if given, otherwise
   `int(params["seed"])` when `params` contains a `seed` key, otherwise
   `None`. When `note` is given it is set on the record
   (`dataclasses.replace`).
4. Embeds the provenance record and the params mapping in the output file
   (format-specific; §5).
5. Returns the written `Path`.

### `format_stem`

Joins `name`, one `_{key}{value}` segment per `params` entry (insertion
order), and `_{run_label}` when given. Values and the run label are formatted
with `str()`, then `.` is replaced with `p` (so `0.7` → `0p7` and the filename
keeps a single suffix), then any character outside `[A-Za-z0-9_+-]` is
replaced with `-`. `name` is used verbatim.

Example: `format_stem("p18_inversion_share", {"seed": 6001}, "post_reduction")`
→ `"p18_inversion_share_seed6001_post_reduction"`.

### `unique_path`

Signature and semantics match the existing script helpers exactly:
`directory/stem.suffix` if free, else `directory/stem_v2.suffix`,
`_v3`, ... Creates `directory` (with parents) if it does not exist.

## 4. JSON value coercion

`save_json` and the metadata embedding use a `json.dumps` `default` hook that
converts numpy scalars (`.item()`), numpy arrays (`.tolist()`), and
`pathlib.Path` (`str()`) to JSON-native values, and raises `TypeError`
otherwise. This removes the most common serialization failure in experiment
scripts (a numpy scalar in a record dict).

## 5. Per-format metadata embedding

Every output file is self-describing; there are no sidecar files.

- **JSON** (`save_json`): the file holds the envelope
  `{"provenance": {...}, "params": {...}, "data": <payload>}`, written with
  `indent=2`. `provenance` is `dataclasses.asdict` of the record.
- **NPZ** (`save_npz`): `np.savez_compressed` of the given arrays plus two
  reserved keys, `_provenance` and `_params`, each a 0-d unicode array
  holding a JSON string. Array names beginning with `_` raise `ValueError`
  (reserved namespace).
- **Parquet** (`save_dataframe`): the table is built with
  `pyarrow.Table.from_pandas(df, preserve_index=True)` and written with
  `pyphi_provenance` and `pyphi_params` entries (JSON strings) merged into
  the schema metadata. `pd.read_parquet` reads the data normally; the
  metadata rides along in the schema. DataFrame fidelity follows parquet
  semantics (this writer targets interoperable script outputs, not the exact
  tuple-preserving round-trip of `pyphi.serialize`).

## 6. `read_metadata`

Dispatches on the file suffix and returns
`{"provenance": dict, "params": dict}`:

- `.json`: load the envelope, return its `provenance` and `params` entries.
- `.npz`: `np.load`, `json.loads` of the `_provenance` and `_params` entries.
- `.parquet`: `pyarrow.parquet.read_schema(path).metadata`, decode the
  `pyphi_provenance` and `pyphi_params` keys.

A file without the expected metadata (not written by these writers) raises
`ValueError` naming the path and the missing key. An unrecognized suffix
raises `ValueError`.

## 7. Script migration (helpers only)

The three scripts drop their local copies of the mechanics and import the
shared helpers:

- `benchmarks/iit_3_vs_4/harness.py`: delete its `unique_path`, import
  `from pyphi.provenance import unique_path` (identical signature).
- `benchmarks/b18_dispatch_gate.py`: same replacement.
- `benchmarks/iit_3_vs_4/p18_inversion_share.py`: `_output_path` is
  rewritten in terms of `format_stem` + `unique_path`, producing identical
  filenames to today (verified by the existing example
  `p18_inversion_share_seed6001_post_reduction.json`).

Their record formats and output directories are **unchanged**:
`benchmarks/iit_3_vs_4/analyze.py` and the existing result files depend on
the current flat JSON layout, so the new envelope applies only to new
scripts. Migrated scripts are verified by import and by exercising the
path-construction code, not by re-running the studies.

## 8. Testing

New `test/test_provenance_writers.py`:

- `format_stem`: int/float/string params, float dot → `p`, sanitization of
  unsafe characters, run label, no params.
- `unique_path`: fresh path, `_v2`/`_v3` on collisions, directory creation.
- `save_json`: envelope structure, numpy scalar/array in payload, seed
  resolved from `params["seed"]`, explicit `seed=` overrides, `note` stored,
  provenance fields present (pyphi version, timestamp), returns the path it
  wrote.
- `save_npz`: arrays round-trip via `np.load`, reserved-key `ValueError`,
  metadata extraction.
- `save_dataframe`: `pd.read_parquet` round-trip (`assert_frame_equal`) for a
  long-format frame with a named index, metadata in schema.
- `read_metadata`: all three formats, `ValueError` on a plain JSON file
  without the envelope and on an unknown suffix.
- No-clobber across writers: second save with identical arguments lands in
  `_v2`.

Migration checks: the three scripts still compile/import and their path
helpers produce the same filenames as before (direct unit assertions on the
imported helpers where feasible).

## 9. Documentation and bookkeeping

- New section in `docs/howto/save-load.md` showing `save_json` and
  `save_dataframe` in a short experiment-script example (params in filename,
  `_v2` on re-run, `read_metadata` to recover seed and SHA).
- Changelog fragment `changelog.d/provenance-writers.feature.md`.
- ROADMAP M14 row marked landed in the same change.
