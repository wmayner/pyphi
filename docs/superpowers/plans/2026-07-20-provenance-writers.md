# Provenance Writers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Script-facing writer functions in `pyphi.provenance` — parameters in the filename, no-clobber versioning, embedded `Provenance` record — plus helper-only migration of the three scripts that hand-roll the pattern.

**Architecture:** All new functions live in `pyphi/provenance.py` beside the existing `Provenance` dataclass they wrap. `save_json`/`save_npz`/`save_dataframe` share a stem builder (`format_stem`), a no-clobber path resolver (`unique_path`), and a metadata-capture helper; each embeds the provenance record in its format's native metadata channel (JSON envelope, reserved NPZ keys, parquet schema metadata). `read_metadata` dispatches on suffix. Three benchmark scripts then import the shared helpers instead of their local copies.

**Tech Stack:** numpy, pyarrow (core dependency), pandas (type-checking only in `provenance.py`), pytest with `tmp_path`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-20-provenance-writers-design.md`.
- Stem value formatting: `str()`, then `.` → `p`, then any character outside `[A-Za-z0-9_+-]` → `-`. `name` is used verbatim.
- `unique_path(directory, stem, suffix)` signature must match the existing script helpers exactly (drop-in replacement).
- Seed resolution: `seed` kwarg wins; else `int(params["seed"])` when present; else `None`.
- NPZ keys starting with `_` are reserved and raise `ValueError`.
- Migrated scripts keep their record formats and output directories unchanged.
- Docstrings: NumPy style, final-state voice, Unicode symbols.
- Commit messages end with the `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01PEAxNzhDCaTrntX3o1JqMV` trailers. Never `--no-verify`; after every commit check `git log --oneline -1` (the ruff-format hook aborts silently).
- Work in `.claude/worktrees/provenance-writers` (branch `provenance-writers`); run tests with the worktree venv via `uv run`.

---

### Task 1: `format_stem`, `unique_path`, `save_json`

**Files:**
- Modify: `pyphi/provenance.py`
- Test: `test/test_provenance_writers.py` (create)

**Interfaces:**
- Consumes: `Provenance.capture(seed=...)`, `dataclasses.replace`, existing module imports.
- Produces: `format_stem(name, params=None, run_label=None) -> str`; `unique_path(directory, stem, suffix) -> Path`; `save_json(data, directory, name, *, params=None, run_label=None, seed=None, note=None) -> Path`; private helpers `_stem_value(value) -> str`, `_json_default(obj)`, `_capture_metadata(params, seed, note) -> Provenance` reused by Task 2.

- [ ] **Step 1: Write the failing tests**

Create `test/test_provenance_writers.py`:

```python
"""Tests for the script-facing provenance writers."""

import json

import numpy as np
import pytest

from pyphi import provenance


class TestFormatStem:
    def test_params_in_insertion_order(self):
        assert (
            provenance.format_stem("study", {"seed": 42, "trials": 60})
            == "study_seed42_trials60"
        )

    def test_float_dot_becomes_p(self):
        assert provenance.format_stem("study", {"noise": 0.7}) == "study_noise0p7"

    def test_unsafe_characters_sanitized(self):
        assert provenance.format_stem("study", {"tag": "a/b c"}) == "study_taga-b-c"

    def test_run_label_appended(self):
        assert (
            provenance.format_stem("study", {"seed": 1}, "post_reduction")
            == "study_seed1_post_reduction"
        )

    def test_no_params(self):
        assert provenance.format_stem("study") == "study"


class TestUniquePath:
    def test_fresh_path(self, tmp_path):
        assert provenance.unique_path(tmp_path, "run", ".json") == tmp_path / "run.json"

    def test_versions_on_collision(self, tmp_path):
        (tmp_path / "run.json").touch()
        assert (
            provenance.unique_path(tmp_path, "run", ".json") == tmp_path / "run_v2.json"
        )
        (tmp_path / "run_v2.json").touch()
        assert (
            provenance.unique_path(tmp_path, "run", ".json") == tmp_path / "run_v3.json"
        )

    def test_creates_directory(self, tmp_path):
        target = tmp_path / "a" / "b"
        provenance.unique_path(target, "run", ".json")
        assert target.is_dir()


class TestSaveJson:
    def test_envelope_and_numpy_values(self, tmp_path):
        path = provenance.save_json(
            {"phi": np.float64(0.5), "counts": np.arange(3)},
            tmp_path,
            "study",
            params={"seed": 42},
        )
        assert path == tmp_path / "study_seed42.json"
        document = json.loads(path.read_text())
        assert document["data"] == {"phi": 0.5, "counts": [0, 1, 2]}
        assert document["params"] == {"seed": 42}
        assert document["provenance"]["seed"] == 42
        assert document["provenance"]["pyphi_version"]
        assert document["provenance"]["timestamp"]

    def test_explicit_seed_overrides_params(self, tmp_path):
        path = provenance.save_json({}, tmp_path, "study", params={"seed": 42}, seed=7)
        assert json.loads(path.read_text())["provenance"]["seed"] == 7

    def test_note_stored(self, tmp_path):
        path = provenance.save_json({}, tmp_path, "study", note="pilot run")
        assert json.loads(path.read_text())["provenance"]["note"] == "pilot run"

    def test_unserializable_payload_raises(self, tmp_path):
        with pytest.raises(TypeError, match="not JSON serializable"):
            provenance.save_json({"bad": object()}, tmp_path, "study")

    def test_no_clobber(self, tmp_path):
        first = provenance.save_json({"x": 1}, tmp_path, "study", params={"seed": 1})
        second = provenance.save_json({"x": 2}, tmp_path, "study", params={"seed": 1})
        assert second == tmp_path / "study_seed1_v2.json"
        assert json.loads(first.read_text())["data"] == {"x": 1}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/test_provenance_writers.py -q`
Expected: FAIL/ERROR with `AttributeError: module 'pyphi.provenance' has no attribute 'format_stem'` (and similar).

- [ ] **Step 3: Implement in `pyphi/provenance.py`**

Add to the import block (alphabetical order within groups; the module already has `from __future__ import annotations`):

```python
import json
import re
from collections.abc import Mapping
from dataclasses import asdict
```

(`dataclass` and `replace` are already imported from `dataclasses`; keep them and add `asdict`.)

Add after the `Provenance` class (before `_set_provenance`):

```python
def _stem_value(value: Any) -> str:
    """Format a parameter value for use in a filename stem."""
    return re.sub(r"[^A-Za-z0-9_+-]", "-", str(value).replace(".", "p"))


def format_stem(
    name: str,
    params: Mapping[str, Any] | None = None,
    run_label: str | None = None,
) -> str:
    """Build a filename stem encoding a script's parameters.

    Joins ``name``, one ``{key}{value}`` segment per ``params`` entry (in
    insertion order), and ``run_label`` when given, with underscores.
    Values and the run label are formatted with ``str()``; ``.`` becomes
    ``p`` (so ``0.7`` → ``0p7`` and the filename keeps a single suffix)
    and any character outside ``[A-Za-z0-9_+-]`` becomes ``-``. ``name``
    is used verbatim.

    Examples
    --------
    >>> format_stem("study", {"seed": 42, "noise": 0.7}, "pilot")
    'study_seed42_noise0p7_pilot'
    """
    parts = [name]
    for key, value in (params or {}).items():
        parts.append(f"{key}{_stem_value(value)}")
    if run_label:
        parts.append(_stem_value(run_label))
    return "_".join(parts)


def unique_path(directory: Path | str, stem: str, suffix: str) -> Path:
    """Return a non-clobbering path: ``stem+suffix``, else ``stem_v2+suffix``, ...

    Creates ``directory`` (with parents) if it does not exist. Never
    returns a path that already exists, so earlier outputs are never
    overwritten.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem}{suffix}"
    version = 2
    while path.exists():
        path = directory / f"{stem}_v{version}{suffix}"
        version += 1
    return path


def _json_default(obj: Any) -> Any:
    """``json.dumps`` fallback for numpy values and paths."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _capture_metadata(
    params: Mapping[str, Any] | None,
    seed: int | None,
    note: str | None,
) -> Provenance:
    """Capture a :class:`Provenance`, resolving the seed from ``params``."""
    if seed is None and params is not None and "seed" in params:
        seed = int(params["seed"])
    prov = Provenance.capture(seed=seed)
    if note is not None:
        prov = replace(prov, note=note)
    return prov


def save_json(
    data: Any,
    directory: Path | str,
    name: str,
    *,
    params: Mapping[str, Any] | None = None,
    run_label: str | None = None,
    seed: int | None = None,
    note: str | None = None,
) -> Path:
    """Write ``data`` to a self-describing, non-clobbering JSON file.

    The file holds the envelope ``{"provenance": ..., "params": ...,
    "data": ...}``. The filename encodes ``params`` and ``run_label``
    (see :func:`format_stem`); an existing file is never overwritten (a
    ``_v2``/``_v3`` suffix is added instead). The provenance record
    stores the seed from ``seed`` or, when omitted, from
    ``params["seed"]``. numpy scalars and arrays in ``data`` are
    converted to JSON-native values.

    Returns the written path.
    """
    prov = _capture_metadata(params, seed, note)
    path = unique_path(directory, format_stem(name, params, run_label), ".json")
    envelope = {
        "provenance": asdict(prov),
        "params": dict(params or {}),
        "data": data,
    }
    path.write_text(json.dumps(envelope, indent=2, default=_json_default))
    return path
```

Update `__all__` at the bottom of the module:

```python
__all__ = [
    "HasProvenance",
    "Provenance",
    "format_stem",
    "save_json",
    "stamp_wall_time",
    "unique_path",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/test_provenance_writers.py test/test_provenance.py -q`
Expected: all PASS (the existing `test_provenance.py` guards against regressions in the module).

- [ ] **Step 5: Commit**

```bash
git add pyphi/provenance.py test/test_provenance_writers.py
git commit -m "Add format_stem, unique_path, and save_json provenance writers

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PEAxNzhDCaTrntX3o1JqMV"
git log --oneline -1
```

---

### Task 2: `save_npz`, `save_dataframe`, `read_metadata`

**Files:**
- Modify: `pyphi/provenance.py`
- Test: `test/test_provenance_writers.py`

**Interfaces:**
- Consumes: `format_stem`, `unique_path`, `_capture_metadata`, `_json_default` from Task 1.
- Produces: `save_npz(arrays, directory, name, *, params=None, run_label=None, seed=None, note=None) -> Path`; `save_dataframe(df, directory, name, *, params=None, run_label=None, seed=None, note=None) -> Path`; `read_metadata(path) -> dict[str, Any]` returning `{"provenance": dict, "params": dict}`.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_provenance_writers.py` (add `import pandas as pd` to the imports):

```python
class TestSaveNpz:
    def test_arrays_round_trip_with_metadata(self, tmp_path):
        arrays = {"phis": np.linspace(0, 1, 5), "states": np.eye(2)}
        path = provenance.save_npz(arrays, tmp_path, "study", params={"seed": 3})
        assert path == tmp_path / "study_seed3.npz"
        with np.load(path) as npz:
            np.testing.assert_array_equal(npz["phis"], arrays["phis"])
            np.testing.assert_array_equal(npz["states"], arrays["states"])
        metadata = provenance.read_metadata(path)
        assert metadata["params"] == {"seed": 3}
        assert metadata["provenance"]["seed"] == 3

    def test_reserved_names_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="reserved"):
            provenance.save_npz({"_provenance": np.zeros(1)}, tmp_path, "study")


class TestSaveDataframe:
    def test_round_trip_and_metadata(self, tmp_path):
        df = pd.DataFrame(
            {"phi": [0.1, 0.2], "n": [3, 4]},
            index=pd.Index(["a", "b"], name="system"),
        )
        path = provenance.save_dataframe(df, tmp_path, "study", params={"seed": 9})
        assert path == tmp_path / "study_seed9.parquet"
        pd.testing.assert_frame_equal(pd.read_parquet(path), df)
        metadata = provenance.read_metadata(path)
        assert metadata["provenance"]["seed"] == 9
        assert metadata["params"] == {"seed": 9}


class TestReadMetadata:
    def test_json_metadata(self, tmp_path):
        path = provenance.save_json({"x": 1}, tmp_path, "study", params={"seed": 5})
        metadata = provenance.read_metadata(path)
        assert metadata["params"] == {"seed": 5}
        assert metadata["provenance"]["seed"] == 5
        assert "pyphi_version" in metadata["provenance"]

    def test_plain_json_rejected(self, tmp_path):
        path = tmp_path / "plain.json"
        path.write_text('{"x": 1}')
        with pytest.raises(ValueError, match="no pyphi provenance"):
            provenance.read_metadata(path)

    def test_unknown_suffix_rejected(self, tmp_path):
        path = tmp_path / "file.csv"
        path.touch()
        with pytest.raises(ValueError, match="unrecognized suffix"):
            provenance.read_metadata(path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/test_provenance_writers.py -q`
Expected: new tests FAIL with `AttributeError: module 'pyphi.provenance' has no attribute 'save_npz'` (and similar); Task 1 tests still pass.

- [ ] **Step 3: Implement in `pyphi/provenance.py`**

Add to the import block:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
```

(Place the `TYPE_CHECKING` block after the third-party imports; it exists only for the `pd.DataFrame` annotation on `save_dataframe`.)

Add after `save_json`:

```python
def _metadata_json(
    prov: Provenance, params: Mapping[str, Any] | None
) -> tuple[str, str]:
    """Serialize the provenance record and params as JSON strings."""
    return (
        json.dumps(asdict(prov), default=_json_default),
        json.dumps(dict(params or {}), default=_json_default),
    )


def save_npz(
    arrays: Mapping[str, np.ndarray],
    directory: Path | str,
    name: str,
    *,
    params: Mapping[str, Any] | None = None,
    run_label: str | None = None,
    seed: int | None = None,
    note: str | None = None,
) -> Path:
    """Write ``arrays`` to a self-describing, non-clobbering ``.npz`` file.

    The arrays are stored with :func:`numpy.savez_compressed`, plus two
    reserved entries ``_provenance`` and ``_params`` holding JSON strings
    (read back with :func:`read_metadata`). Array names beginning with
    ``_`` raise :class:`ValueError`. Filename, versioning, and seed
    resolution follow :func:`save_json`.

    Returns the written path.
    """
    reserved = [key for key in arrays if key.startswith("_")]
    if reserved:
        raise ValueError(f"array names beginning with '_' are reserved: {reserved}")
    prov = _capture_metadata(params, seed, note)
    prov_json, params_json = _metadata_json(prov, params)
    path = unique_path(directory, format_stem(name, params, run_label), ".npz")
    np.savez_compressed(
        path,
        **arrays,
        _provenance=np.array(prov_json),
        _params=np.array(params_json),
    )
    return path


def save_dataframe(
    df: pd.DataFrame,
    directory: Path | str,
    name: str,
    *,
    params: Mapping[str, Any] | None = None,
    run_label: str | None = None,
    seed: int | None = None,
    note: str | None = None,
) -> Path:
    """Write ``df`` to a self-describing, non-clobbering parquet file.

    The frame is written with its index preserved; ``pyphi_provenance``
    and ``pyphi_params`` entries (JSON strings) are merged into the
    parquet schema metadata, so :func:`pandas.read_parquet` reads the
    data normally and :func:`read_metadata` recovers the metadata.
    Filename, versioning, and seed resolution follow :func:`save_json`.
    DataFrame fidelity follows parquet semantics.

    Returns the written path.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    prov = _capture_metadata(params, seed, note)
    prov_json, params_json = _metadata_json(prov, params)
    path = unique_path(directory, format_stem(name, params, run_label), ".parquet")
    table = pa.Table.from_pandas(df, preserve_index=True)
    metadata = dict(table.schema.metadata or {})
    metadata[b"pyphi_provenance"] = prov_json.encode()
    metadata[b"pyphi_params"] = params_json.encode()
    pq.write_table(table.replace_schema_metadata(metadata), path)
    return path


def read_metadata(path: Path | str) -> dict[str, Any]:
    """Read the provenance and params embedded in a writer's output file.

    Dispatches on the file suffix (``.json``, ``.npz``, or ``.parquet``)
    and returns ``{"provenance": dict, "params": dict}``. A file without
    the expected metadata (not produced by :func:`save_json`,
    :func:`save_npz`, or :func:`save_dataframe`) raises
    :class:`ValueError`, as does an unrecognized suffix.
    """
    path = Path(path)
    missing = ValueError(f"no pyphi provenance metadata in {path}")
    if path.suffix == ".json":
        document = json.loads(path.read_text())
        try:
            return {
                "provenance": document["provenance"],
                "params": document["params"],
            }
        except (KeyError, TypeError):
            raise missing from None
    if path.suffix == ".npz":
        with np.load(path) as npz:
            try:
                return {
                    "provenance": json.loads(str(npz["_provenance"][()])),
                    "params": json.loads(str(npz["_params"][()])),
                }
            except KeyError:
                raise missing from None
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        metadata = pq.read_schema(path).metadata or {}
        try:
            return {
                "provenance": json.loads(metadata[b"pyphi_provenance"]),
                "params": json.loads(metadata[b"pyphi_params"]),
            }
        except KeyError:
            raise missing from None
    raise ValueError(f"unrecognized suffix {path.suffix!r} for {path}")
```

Update `__all__`:

```python
__all__ = [
    "HasProvenance",
    "Provenance",
    "format_stem",
    "read_metadata",
    "save_dataframe",
    "save_json",
    "save_npz",
    "stamp_wall_time",
    "unique_path",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/test_provenance_writers.py test/test_provenance.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add pyphi/provenance.py test/test_provenance_writers.py
git commit -m "Add save_npz, save_dataframe, and read_metadata provenance writers

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PEAxNzhDCaTrntX3o1JqMV"
git log --oneline -1
```

---

### Task 3: Migrate the three scripts to the shared helpers

**Files:**
- Modify: `benchmarks/iit_3_vs_4/harness.py` (delete local `unique_path` at ~line 510; add import)
- Modify: `benchmarks/b18_dispatch_gate.py` (delete local `unique_path` at ~line 645; add import)
- Modify: `benchmarks/iit_3_vs_4/p18_inversion_share.py` (rewrite `_output_path` at ~line 113; add import)

**Interfaces:**
- Consumes: `pyphi.provenance.unique_path` and `pyphi.provenance.format_stem` (Task 1 signatures).
- Produces: nothing new — record formats and output directories are unchanged (spec §7).

- [ ] **Step 1: Migrate `harness.py`**

In `benchmarks/iit_3_vs_4/harness.py`, add to the pyphi import group (after `import pyphi`):

```python
from pyphi.provenance import unique_path
```

Delete the local definition (the function currently at lines 510–520):

```python
def unique_path(directory: Path, stem: str, suffix: str) -> Path:
    """Return a non-clobbering path: stem.suffix, then stem_v2.suffix, ..."""
    directory.mkdir(parents=True, exist_ok=True)
    base = directory / f"{stem}{suffix}"
    if not base.exists():
        return base
    n = 2
    while True:
        candidate = directory / f"{stem}_v{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1
```

Both call sites (`pstats_path = unique_path(...)` and `path = unique_path(RESULTS_DIR, stem, ".json")`) keep working — the shared function has the identical signature and also creates the directory.

- [ ] **Step 2: Migrate `b18_dispatch_gate.py`**

In `benchmarks/b18_dispatch_gate.py`, add after `import numpy as np`:

```python
from pyphi.provenance import unique_path
```

Delete the local definition (currently at lines 645–651):

```python
def unique_path(directory: Path, stem: str, suffix: str) -> Path:
    path = directory / f"{stem}{suffix}"
    version = 2
    while path.exists():
        path = directory / f"{stem}_v{version}{suffix}"
        version += 1
    return path
```

Leave the existing `RESULTS_DIR.mkdir(parents=True, exist_ok=True)` call in `main` — it is now redundant but harmless, and removing it would widen the diff.

- [ ] **Step 3: Migrate `p18_inversion_share.py`**

In `benchmarks/iit_3_vs_4/p18_inversion_share.py`, add after `import numpy as np`:

```python
from pyphi.provenance import format_stem
from pyphi.provenance import unique_path
```

Replace `_output_path` (currently at lines 113–124):

```python
def _output_path(seed: int, run_label: str | None) -> Path:
    results_dir = Path(__file__).parent / "results"
    return unique_path(
        results_dir,
        format_stem("p18_inversion_share", {"seed": seed}, run_label),
        ".json",
    )
```

Filename equivalence: the old code produced `p18_inversion_share_seed{seed}` plus `_{run_label}`; `format_stem` produces the same for integer seeds and underscore labels (e.g. the existing `p18_inversion_share_seed6001_post_reduction.json`).

- [ ] **Step 4: Verify the migrated scripts**

```bash
uv run python -m py_compile benchmarks/iit_3_vs_4/harness.py benchmarks/b18_dispatch_gate.py benchmarks/iit_3_vs_4/p18_inversion_share.py
uv run python - <<'EOF'
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "p18", "benchmarks/iit_3_vs_4/p18_inversion_share.py"
)
p18 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p18)
path = p18._output_path(6001, "post_reduction")
assert path.name == "p18_inversion_share_seed6001_post_reduction.json" or (
    path.stem.startswith("p18_inversion_share_seed6001_post_reduction_v")
), path
print("p18 filename equivalence OK:", path.name)
EOF
```

Expected: no output from `py_compile`; the p18 check prints the equivalent filename (a `_v2`-style name is correct when the base file already exists in `results/`).

Also confirm no other local copies remain:

```bash
grep -rn "def unique_path" benchmarks/ --include="*.py"
```

Expected: no matches.

- [ ] **Step 5: Run the writer tests (regression guard) and commit**

```bash
uv run pytest test/test_provenance_writers.py -q
git add benchmarks/iit_3_vs_4/harness.py benchmarks/b18_dispatch_gate.py benchmarks/iit_3_vs_4/p18_inversion_share.py
git commit -m "Use shared provenance path helpers in benchmark scripts

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PEAxNzhDCaTrntX3o1JqMV"
git log --oneline -1
```

---

### Task 4: How-to section, changelog, ROADMAP

**Files:**
- Modify: `docs/howto/save-load.md` (insert before the `## Compatibility note` section)
- Create: `changelog.d/provenance-writers.feature.md`
- Modify: `ROADMAP.md` (M14 row, ~line 633)

**Interfaces:**
- Consumes: `provenance.save_json`, `provenance.read_metadata` (Tasks 1–2). The doc reuses the `out` temp directory defined earlier in the file (`out = Path(mkdtemp())`).
- Produces: documentation only.

- [ ] **Step 1: Add the how-to section**

Insert into `docs/howto/save-load.md` immediately before `## Compatibility note`:

````markdown
## Experiment provenance writers

Experiment scripts need two things beyond `pyphi.save`: output files that
never overwrite earlier runs, and a record of how each file was produced.
The writers in `pyphi.provenance` provide both. Parameters are encoded
into the filename, a repeated save lands in a `_v2` file instead of
clobbering the first, and every file embeds a full provenance record —
pyphi version, git commit, timestamp, and seed.

```{code-cell} python
from pyphi import provenance

path = provenance.save_json(
    {"phi": 0.133873},
    out,
    "sweep_study",
    params={"seed": 42, "trials": 60},
)
path.name
```

```{code-cell} python
provenance.save_json(
    {"phi": 0.5}, out, "sweep_study", params={"seed": 42, "trials": 60}
).name
```

`save_npz` does the same for arrays of raw per-trial data, and
`save_dataframe` writes a DataFrame as parquet — the format used for
DataFrame outputs throughout PyPhi — with the metadata embedded in the
parquet schema. `read_metadata` recovers the provenance and parameters
from any of the three formats:

```{code-cell} python
metadata = provenance.read_metadata(path)
{key: metadata["provenance"][key] for key in ("seed", "pyphi_version")}
```
````

- [ ] **Step 2: Changelog fragment**

```bash
cat > changelog.d/provenance-writers.feature.md <<'EOF'
Added script-facing provenance writers: `pyphi.provenance.save_json`, `save_npz`, and `save_dataframe` (parquet) encode parameters in the filename, never overwrite existing files (`_v2`/`_v3` versioning), and embed a full `Provenance` record (git SHA, seed, versions) in every output; `read_metadata` reads it back from any of the three formats.
EOF
```

- [ ] **Step 3: Update the ROADMAP M14 row**

Replace:

```markdown
- **Script-facing provenance writer (M14).** `provenance.save_json`/`save_npz` with git SHA,
  parameters encoded in the filename, and no-clobber versioning — consolidating the pattern
  that experiment scripts repeatedly re-implement.
```

with:

```markdown
- **Script-facing provenance writer (M14).** *Landed 2026-07-20:*
  `provenance.save_json`/`save_npz`/`save_dataframe` (parquet, per the DataFrame-output
  convention) with `format_stem` filename encoding, `unique_path` no-clobber versioning, an
  embedded `Provenance` record, and `read_metadata` to read it back; the three benchmark
  scripts with hand-rolled copies now import the shared helpers. Spec:
  `docs/superpowers/specs/2026-07-20-provenance-writers-design.md`.
```

- [ ] **Step 4: Verify docs build and full suite**

```bash
rm -rf docs/reference/_autosummary
just docs > /tmp/provenance-docs-build.log 2>&1; echo "DOCS EXIT: $?"
uv run pytest -q > /tmp/provenance-full-suite.log 2>&1
tail -3 /tmp/provenance-full-suite.log
```

Expected: `DOCS EXIT: 0`; suite summary line shows no failures. (Read the log summary lines — do not trust pipeline exit codes.)

- [ ] **Step 5: Commit**

```bash
git add docs/howto/save-load.md changelog.d/provenance-writers.feature.md ROADMAP.md
git commit -m "Document provenance writers

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PEAxNzhDCaTrntX3o1JqMV"
git log --oneline -1
```

---

## Completion

After Task 4: finish with superpowers:finishing-a-development-branch (standing choice: merge to main locally with `--no-ff`, run the full pathless suite in the main tree, remove the worktree and branch from the main root, update the ledger and memory).
