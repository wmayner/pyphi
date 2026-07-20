# SweepResult / OptimizationResult Serialization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Register `SweepResult` and `OptimizationResult` in `pyphi.serialize` with full-fidelity save/load, embedding their DataFrames as parquet bytes via a new core pyarrow dependency.

**Architecture:** A new internal DataFrame codec (`pyphi/serialize/frames.py` + `DataFrameSchema`) writes each frame with its index reset to columns and records tuple-valued column names for exact restoration. Two new tagged Structs (`SweepResultSchema`, `OptimizationResultSchema`) join the `Schema` union with converters following the existing `_register_<type>()` pattern. Both domain classes gain the `Serializable` mixin; the bespoke lossy `OptimizationResult.save` is deleted.

**Tech Stack:** msgspec (existing), pandas (existing), pyarrow >= 25.0 (new core dependency).

**Spec:** `docs/superpowers/specs/2026-07-19-result-serialization-design.md`

## Global Constraints

- `pyarrow>=25.0` is added to the core `dependencies` list in `pyproject.toml` (spec §8). No extras change.
- All commands run with `uv run` from the worktree root. Python 3.13+ only.
- Commit messages end with the two trailers used throughout this branch (Co-Authored-By + Claude-Session); never `git commit --no-verify`. After every commit, check `git log --oneline -1` — hooks abort silently.
- Docstrings: NumPy style, final-state voice, no planning-artifact references (no "M12", no spec/plan mentions) in code, docstrings, or the changelog fragment.
- Tests that compute φ pin their formalism with a complete preset (`config.override(**presets.iit4_2023, ...)`), pinned inside fixtures.
- Never pipe pytest through tail/head; redirect to a file and read the summary line.
- Stage only the files this plan names.

## Verified facts (do not re-derive)

- msgspec JSON encodes NaN/inf as `null`, and decoding `null` as `float` raises — scalar float fields must not carry NaN in JSON.
- Parquet round-trips all frame shapes that occur here **exactly** (verified with `assert_frame_equal`): MultiIndex with tuple levels, single named tuple index (`tupleize_cols=False`), RangeIndex, all-None object columns, all-NaN float columns, int64/bool/float64 dtypes.
- msgspec resolves forward-referenced recursive tagged unions: a field annotated `tuple["Schema | float", ...]` where `Schema` is defined later in the module decodes correctly.
- The worktree venv has pyarrow 25.0.0 installed.

---

### Task 1: pyarrow dependency + DataFrame codec

**Files:**
- Modify: `pyproject.toml` (dependencies list, ~line 30)
- Modify: `pyphi/serialize/schema.py` (new section before the `Schema` union)
- Create: `pyphi/serialize/frames.py`
- Create: `test/serialize/test_serialize_results.py`

**Interfaces:**
- Consumes: `pyphi.serialize.schema` (Struct conventions), pandas, pyarrow.
- Produces: `schema.DataFrameSchema` (fields `parquet: bytes`, `index_columns: tuple[str, ...]`, `tuple_columns: tuple[str, ...]`); `frames.dataframe_to_schema(df) -> DataFrameSchema`; `frames.schema_to_dataframe(struct) -> pd.DataFrame`. Tasks 2 and 3 call both functions.

- [ ] **Step 1: Add the dependency**

In `pyproject.toml`, in the `dependencies` list, insert between `"psutil>=2.1.1",` and `"pyyaml>=3.13",`:

```toml
    "pyarrow>=25.0",
```

Run: `uv sync --all-extras 2>&1 | tail -1` (or `uv pip install pyarrow` if sync is slow) and verify `uv run python -c "import pyarrow; print(pyarrow.__version__)"` prints `25.0.0`.

- [ ] **Step 2: Add `DataFrameSchema` to `pyphi/serialize/schema.py`**

Insert a new section after the estimation-layer posteriors section (after `PhiPosteriorSchema`) and **before** the `# The tagged union grows one member per serializable type.` comment:

```python
# --- Batch-run results --------------------------------------------------------


class DataFrameSchema(msgspec.Struct, frozen=True, tag="dataframe"):
    """A pandas DataFrame as embedded parquet.

    ``index_columns`` names the index levels reset to columns before the
    parquet write; ``tuple_columns`` names the object columns whose non-null
    cells are restored as tuples on decode (parquet represents them as
    lists).
    """

    parquet: bytes
    index_columns: tuple[str, ...] = ()
    tuple_columns: tuple[str, ...] = ()
```

`DataFrameSchema` does **not** join the `Schema` union: it is an internal field type, following the `RelationRefSchema` precedent for Structs that never appear as a top-level payload. (This deviates from a one-line remark in spec §4; flag it at review.)

- [ ] **Step 3: Write the failing codec tests**

Create `test/serialize/test_serialize_results.py`:

```python
"""Round-trip serialization of SweepResult and OptimizationResult."""

import math

import pandas as pd
import pytest

from pyphi.serialize import frames


class TestDataFrameCodec:
    def _roundtrip(self, df):
        return frames.schema_to_dataframe(frames.dataframe_to_schema(df))

    def test_multiindex_tuple_levels(self):
        idx = pd.MultiIndex.from_arrays(
            [
                ["IIT_4_0_2026", "IIT_4_0_2026", "IIT_3_0"],
                [(0, 1), (0, 1, 2), (0, 1)],
                [(1, 0), (1, 1, 0), (0, 0)],
            ],
            names=["formalism", "subset", "state"],
        )
        df = pd.DataFrame(
            {
                "phi": [0.5, math.nan, 1.25],
                "is_irreducible": [True, False, True],
                "effectively_tied": [None, True, None],
                "n": [3, 4, 5],
            },
            index=idx,
        )
        pd.testing.assert_frame_equal(self._roundtrip(df), df)

    def test_single_named_tuple_index(self):
        df = pd.DataFrame(
            {"phi": [0.5, 0.7]},
            index=pd.Index([(0, 1), (1, 0)], name="state", tupleize_cols=False),
        )
        back = self._roundtrip(df)
        pd.testing.assert_frame_equal(back, df)
        # Tuple entries stay scalar index entries (not a MultiIndex).
        assert back.index[0] == (0, 1)

    def test_range_index_with_nan_and_tuples(self):
        df = pd.DataFrame(
            {
                "eval": [0, 1, 2],
                "objective": [0.1, math.nan, 0.3],
                "partition": ["a", None, "b"],
                "cause_state": [(1, 0), None, (0, 1)],
            }
        )
        back = self._roundtrip(df)
        pd.testing.assert_frame_equal(back, df)
        assert back["cause_state"][0] == (1, 0)
        assert back["cause_state"][1] is None

    def test_unnamed_nondefault_index_raises(self):
        df = pd.DataFrame({"x": [1, 2]}, index=pd.Index([10, 20]))
        with pytest.raises(ValueError, match="unnamed"):
            frames.dataframe_to_schema(df)
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `uv run pytest test/serialize/test_serialize_results.py -v 2>&1 | tail -5`
Expected: collection error — `ImportError` / `ModuleNotFoundError` for `pyphi.serialize.frames`.

- [ ] **Step 5: Create `pyphi/serialize/frames.py`**

```python
"""Exact serialization of pandas DataFrames as embedded parquet.

The frame is written with its named index levels reset to columns, so any
index shape that occurs on PyPhi result tables round-trips exactly —
including levels whose entries are tuples. Object columns holding tuples
are recorded by name at encode time; parquet represents them as lists, and
the recorded names make the restoration exact rather than heuristic. NaN,
None cells, and column dtypes survive bit-exactly via pyarrow.
"""

import io

import numpy as np
import pandas as pd

from . import schema


def dataframe_to_schema(df: pd.DataFrame) -> schema.DataFrameSchema:
    names = list(df.index.names)
    if names == [None]:
        if not isinstance(df.index, pd.RangeIndex):
            raise ValueError(
                "cannot serialize a DataFrame with an unnamed, non-default index"
            )
        reset = df
        index_columns: tuple[str, ...] = ()
    else:
        if any(name is None for name in names):
            raise ValueError(
                "cannot serialize a DataFrame with unnamed index levels"
            )
        reset = df.reset_index()
        index_columns = tuple(str(name) for name in names)
    tuple_columns = tuple(
        str(column)
        for column in reset.columns
        if reset[column].dtype == object
        and any(isinstance(value, tuple) for value in reset[column])
    )
    buffer = io.BytesIO()
    reset.to_parquet(buffer, engine="pyarrow", index=False)
    return schema.DataFrameSchema(
        parquet=buffer.getvalue(),
        index_columns=index_columns,
        tuple_columns=tuple_columns,
    )


def _as_tuple(value):
    if value is None:
        return None
    return tuple(x.item() if isinstance(x, np.generic) else x for x in value)


def schema_to_dataframe(struct: schema.DataFrameSchema) -> pd.DataFrame:
    df = pd.read_parquet(io.BytesIO(struct.parquet), engine="pyarrow")
    for column in struct.tuple_columns:
        df[column] = [_as_tuple(value) for value in df[column]]
    if struct.index_columns:
        df = df.set_index(list(struct.index_columns))
        if len(struct.index_columns) == 1:
            df.index = pd.Index(
                df.index, name=struct.index_columns[0], tupleize_cols=False
            )
    return df
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest test/serialize/test_serialize_results.py -v 2>&1 | tail -8`
Expected: 4 passed.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock pyphi/serialize/schema.py pyphi/serialize/frames.py test/serialize/test_serialize_results.py
git commit -m "Add parquet DataFrame codec to pyphi.serialize"
```

(If `uv sync` did not touch `uv.lock`, commit without it. Append the standard trailers.)

---

### Task 2: SweepResult serialization

**Files:**
- Modify: `pyphi/serialize/schema.py` (same new section; `Schema` union at bottom)
- Modify: `pyphi/serialize/convert.py` (imports; new `_register_sweep_result`; `_ensure_registered` list)
- Modify: `pyphi/sweep.py` (`SweepResult` gains `Serializable`)
- Test: `test/serialize/test_serialize_results.py`

**Interfaces:**
- Consumes: `frames.dataframe_to_schema` / `frames.schema_to_dataframe` (Task 1); existing `to_schema` / `from_schema` in convert.py.
- Produces: `schema.SweepResultSchema` (tag `"sweep_result"`); `SweepResult.save(target, format=None)` / `SweepResult.load(target, format=None)` via the mixin.

- [ ] **Step 1: Write the failing tests**

Append to `test/serialize/test_serialize_results.py`. Extend the module imports at the top of the file to:

```python
"""Round-trip serialization of SweepResult and OptimizationResult."""

import math

import pandas as pd
import pytest

from pyphi import examples
from pyphi import serialize
from pyphi.conf import config
from pyphi.conf import presets
from pyphi.direction import Direction
from pyphi.serialize import frames
from pyphi.sweep import SweepResult
from pyphi.sweep import sweep
```

Then append:

```python
@pytest.fixture(scope="module")
def multi_axis_sweep():
    # subsets x states vary -> MultiIndex with tuple-valued levels; states="all"
    # auto-enumerates, so unreachable cells populate .skipped.
    substrate = examples.basic_substrate()
    with config.override(**presets.iit4_2023, progress_bars=False):
        return sweep(
            substrate,
            states="all",
            subsets=[(0, 1, 2), (0, 1)],
            parallel=False,
        )


@pytest.fixture(scope="module")
def single_axis_sweep():
    # Only states vary -> single named index of tuples (tupleize_cols=False).
    substrate = examples.basic_substrate()
    with config.override(**presets.iit4_2023, progress_bars=False):
        return sweep(substrate, states="all", parallel=False)


def _assert_sweep_equal(a, b):
    assert isinstance(b, SweepResult)
    pd.testing.assert_frame_equal(a.df, b.df)
    assert a.skipped == b.skipped
    for x, y in zip(a.results, b.results, strict=True):
        assert x == y


class TestSweepResultRoundTrip:
    @pytest.mark.parametrize("fmt", ["json", "msgpack"])
    def test_multi_axis(self, multi_axis_sweep, fmt):
        assert multi_axis_sweep.skipped  # precondition: exercises skipped cells
        data = serialize.dumps(multi_axis_sweep, format=fmt)
        _assert_sweep_equal(multi_axis_sweep, serialize.loads(data, format=fmt))

    @pytest.mark.parametrize("fmt", ["json", "msgpack"])
    def test_single_axis(self, single_axis_sweep, fmt):
        data = serialize.dumps(single_axis_sweep, format=fmt)
        _assert_sweep_equal(single_axis_sweep, serialize.loads(data, format=fmt))

    def test_ces_compute(self):
        substrate = examples.basic_substrate()
        with config.override(**presets.iit4_2023, progress_bars=False):
            result = sweep(
                substrate, states=[(1, 0, 0)], compute="ces", parallel=False
            )
        _assert_sweep_equal(result, serialize.loads(serialize.dumps(result)))

    def test_float_results(self):
        substrate = examples.basic_substrate()
        with config.override(**presets.iit4_2023, progress_bars=False):
            result = sweep(
                substrate,
                states=[(1, 0, 0)],
                compute=lambda s: float(s.sia().phi),
                parallel=False,
            )
        back = serialize.loads(serialize.dumps(result))
        assert back.results == result.results
        assert isinstance(back.results[0], float)

    def test_save_load_gz(self, single_axis_sweep, tmp_path):
        path = tmp_path / "sweep.json.gz"
        single_axis_sweep.save(path)
        _assert_sweep_equal(single_axis_sweep, SweepResult.load(path))

    def test_load_wrong_type_raises(self, tmp_path):
        path = tmp_path / "direction.json"
        serialize.save(Direction.CAUSE, path)
        with pytest.raises(TypeError, match="SweepResult"):
            SweepResult.load(path)
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest test/serialize/test_serialize_results.py::TestSweepResultRoundTrip -v 2>&1 | tail -6`
Expected: FAIL — `TypeError: No serializer registered for SweepResult` (and the save/load tests fail with `AttributeError: 'SweepResult' object has no attribute 'save'`).

- [ ] **Step 3: Add `SweepResultSchema` to `pyphi/serialize/schema.py`**

In the `# --- Batch-run results` section, after `DataFrameSchema`:

```python
class SweepResultSchema(msgspec.Struct, frozen=True, tag="sweep_result"):
    df: DataFrameSchema
    results: tuple["Schema | float", ...]
    skipped: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...]
```

Add `SweepResultSchema` to the `Schema` union at the bottom of the module (after `| PhiPosteriorSchema`):

```python
    | SweepResultSchema
```

- [ ] **Step 4: Register the converter in `pyphi/serialize/convert.py`**

Add `from . import frames` to the imports (after `from . import arrays`).

Add before `_ensure_registered()`:

```python
def _register_sweep_result() -> None:
    from pyphi.sweep import SweepResult

    _ENCODERS[SweepResult] = lambda r: schema.SweepResultSchema(
        df=frames.dataframe_to_schema(r.df),
        results=tuple(
            obj if isinstance(obj, float) else to_schema(obj) for obj in r.results
        ),
        skipped=tuple(
            (formalism, tuple(subset), tuple(state))
            for formalism, subset, state in r.skipped
        ),
    )

    def _decode_sweep_result(s: schema.SweepResultSchema) -> Any:
        return SweepResult(
            df=frames.schema_to_dataframe(s.df),
            results=[
                obj if isinstance(obj, float) else from_schema(obj)
                for obj in s.results
            ],
            skipped=[
                (formalism, tuple(subset), tuple(state))
                for formalism, subset, state in s.skipped
            ],
        )

    _DECODERS[schema.SweepResultSchema] = _decode_sweep_result
```

Append `_register_sweep_result()` to the call list inside `_ensure_registered()` (after `_register_phi_posterior()`).

- [ ] **Step 5: Add the mixin to `SweepResult` in `pyphi/sweep.py`**

Add the import (after `from pyphi.direction import Direction`):

```python
from pyphi.serializable import Serializable
```

Change the class declaration:

```python
@dataclass(frozen=True)
class SweepResult(Serializable):
```

(The docstring is unchanged. Note: the generated dataclass `__eq__` is unusable on this class — the DataFrame field raises on ambiguous truth — which is why the tests compare field-by-field.)

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest test/serialize/test_serialize_results.py -v 2>&1 | tail -10`
Expected: all pass (codec tests + sweep tests).

- [ ] **Step 7: Run the neighboring suites**

Run: `uv run pytest test/test_sweep.py test/serialize/ -q 2>&1 | tail -3`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add pyphi/serialize/schema.py pyphi/serialize/convert.py pyphi/sweep.py test/serialize/test_serialize_results.py
git commit -m "Serialize SweepResult through pyphi.serialize"
```

(Append the standard trailers.)

---

### Task 3: OptimizationResult serialization

**Files:**
- Modify: `pyphi/serialize/schema.py` (same section; union)
- Modify: `pyphi/serialize/convert.py` (imports; new `_register_optimization_result`; `_ensure_registered` list)
- Modify: `pyphi/optimize.py` (mixin; delete bespoke `save`; drop unused imports)
- Modify: `test/test_optimize.py` (replace the bespoke-save test)
- Test: `test/serialize/test_serialize_results.py`

**Interfaces:**
- Consumes: `frames.dataframe_to_schema` / `frames.schema_to_dataframe` (Task 1); `arrays.array_to_bytes` / `arrays.bytes_to_array`; `_enc_optional` / `_dec_optional`; existing `SubstrateSchema` and `SIASchema` union.
- Produces: `schema.OptimizationResultSchema` (tag `"optimization_result"`); `OptimizationResult.save` / `OptimizationResult.load` via the mixin (the bespoke `save` is gone).

- [ ] **Step 1: Write the failing tests**

Append to the imports of `test/serialize/test_serialize_results.py`:

```python
import numpy as np

from pyphi.optimize import OptimizationResult
from pyphi.optimize import optimize
from pyphi.optimize import weight_axes
from pyphi.substrate_generator import ising
```

Append to the file:

```python
# The IIT 4.0 (2023) Fig. 1A substrate; (1, 0, 0) is reachable.
FIG1A_WEIGHTS = np.array(
    [
        [-0.2, 0.7, 0.2],
        [0.7, -0.2, 0.0],
        [0.0, -0.8, 0.2],
    ]
)


@pytest.fixture(scope="module")
def optimization_result():
    axis = weight_axes(
        [ising.probability] * 3, FIG1A_WEIGHTS, [(0, 1)], temperature=0.25
    )
    with config.override(**presets.iit4_2023, progress_bars=False):
        return optimize(
            axis,
            (1, 0, 0),
            [(0.2, 1.2)],
            seed=5,
            popsize=4,
            maxiter=3,
            parallel=False,
        )


def _assert_optimization_equal(a, b):
    assert isinstance(b, OptimizationResult)
    np.testing.assert_array_equal(a.best_params, b.best_params)
    assert a.best_objective == b.best_objective or (
        math.isnan(a.best_objective) and math.isnan(b.best_objective)
    )
    assert a.best_substrate == b.best_substrate
    assert a.best_sia == b.best_sia
    pd.testing.assert_frame_equal(a.trajectory, b.trajectory)
    assert a.bounds == b.bounds
    assert a.seed == b.seed
    assert a.direction == b.direction
    assert a.objective_name == b.objective_name
    assert a.settings == b.settings
    assert a.config_snapshot == b.config_snapshot
    assert a.n_evaluations == b.n_evaluations
    assert a.n_unreachable == b.n_unreachable


class TestOptimizationResultRoundTrip:
    @pytest.mark.parametrize("fmt", ["json", "msgpack"])
    def test_round_trip(self, optimization_result, fmt):
        assert optimization_result.best_sia is not None  # precondition
        data = serialize.dumps(optimization_result, format=fmt)
        _assert_optimization_equal(
            optimization_result, serialize.loads(data, format=fmt)
        )

    def test_save_load_path(self, optimization_result, tmp_path):
        path = tmp_path / "run.mpk.gz"
        optimization_result.save(path)
        _assert_optimization_equal(
            optimization_result, OptimizationResult.load(path)
        )

    def test_all_unreachable_nan_round_trips(self):
        # NaN best_objective (no reachable candidate) maps through None in the
        # schema and is restored as NaN; all-None and all-NaN trajectory
        # columns survive.
        n = 2
        trajectory = pd.DataFrame(
            {
                "eval": [0, 1],
                "generation": [0, 0],
                "p0": [0.1, 0.9],
                "objective": [math.nan] * n,
                "reachable": [False] * n,
                "partition": [None] * n,
                "cause_state": [None] * n,
                "effect_state": [None] * n,
                "partition_margin": [math.nan] * n,
                "cause_state_margin": [math.nan] * n,
                "effect_state_margin": [math.nan] * n,
            }
        )
        result = OptimizationResult(
            best_params=np.array([0.1]),
            best_objective=math.nan,
            best_substrate=examples.basic_substrate(),
            best_sia=None,
            trajectory=trajectory,
            bounds=[(0.0, 1.0)],
            seed=1,
            direction="maximize",
            objective_name="signed_normalized_phi",
            settings={
                "backend": "scipy.differential_evolution",
                "popsize": 4,
                "maxiter": 3,
                "tol": 0.01,
            },
            config_snapshot={"precision": 13, "formalism": None},
            n_evaluations=n,
            n_unreachable=n,
        )
        for fmt in ["json", "msgpack"]:
            back = serialize.loads(
                serialize.dumps(result, format=fmt), format=fmt
            )
            assert math.isnan(back.best_objective)
            assert back.best_sia is None
            _assert_optimization_equal(result, back)
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest test/serialize/test_serialize_results.py::TestOptimizationResultRoundTrip -v 2>&1 | tail -6`
Expected: FAIL — `TypeError: No serializer registered for OptimizationResult` (the save/load test fails on the bespoke save writing summary JSON that `load` cannot read).

- [ ] **Step 3: Add `OptimizationResultSchema` to `pyphi/serialize/schema.py`**

In the `# --- Batch-run results` section, after `SweepResultSchema`:

```python
class OptimizationResultSchema(
    msgspec.Struct, frozen=True, tag="optimization_result"
):
    """An :func:`~pyphi.optimize.optimize` outcome.

    ``best_objective`` is stored as None exactly when the run had no
    reachable candidate (NaN on the domain object; JSON cannot carry NaN).
    """

    best_params: bytes
    best_objective: float | None
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

Add `| OptimizationResultSchema` to the `Schema` union (after `| SweepResultSchema`).

- [ ] **Step 4: Register the converter in `pyphi/serialize/convert.py`**

Add `import math` to the standard-library imports at the top of the module (before `from collections.abc import Callable`, per isort).

Add after `_register_sweep_result`:

```python
def _register_optimization_result() -> None:
    from pyphi.optimize import OptimizationResult

    def _encode_optimization_result(r: Any) -> Any:
        best_objective = float(r.best_objective)
        return schema.OptimizationResultSchema(
            best_params=arrays.array_to_bytes(
                np.asarray(r.best_params, dtype=float)
            ),
            best_objective=(
                None if math.isnan(best_objective) else best_objective
            ),
            best_substrate=to_schema(r.best_substrate),
            best_sia=_enc_optional(r.best_sia),
            trajectory=frames.dataframe_to_schema(r.trajectory),
            bounds=tuple((float(lo), float(hi)) for lo, hi in r.bounds),
            seed=int(r.seed),
            direction=r.direction,
            objective_name=r.objective_name,
            settings=dict(r.settings),
            config_snapshot=dict(r.config_snapshot),
            n_evaluations=int(r.n_evaluations),
            n_unreachable=int(r.n_unreachable),
        )

    _ENCODERS[OptimizationResult] = _encode_optimization_result

    def _decode_optimization_result(s: schema.OptimizationResultSchema) -> Any:
        return OptimizationResult(
            best_params=arrays.bytes_to_array(s.best_params),
            best_objective=(
                math.nan if s.best_objective is None else s.best_objective
            ),
            best_substrate=from_schema(s.best_substrate),
            best_sia=_dec_optional(s.best_sia),
            trajectory=frames.schema_to_dataframe(s.trajectory),
            bounds=[(lo, hi) for lo, hi in s.bounds],
            seed=s.seed,
            direction=s.direction,
            objective_name=s.objective_name,
            settings=s.settings,
            config_snapshot=s.config_snapshot,
            n_evaluations=s.n_evaluations,
            n_unreachable=s.n_unreachable,
        )

    _DECODERS[schema.OptimizationResultSchema] = _decode_optimization_result
```

Append `_register_optimization_result()` to the call list inside `_ensure_registered()` (after `_register_sweep_result()`).

- [ ] **Step 5: Replace the bespoke save in `pyphi/optimize.py`**

1. Remove `from pathlib import Path` from the imports (its only user is the deleted method).
2. Add `from pyphi.serializable import Serializable` to the imports (after `from pyphi.landscape import _part_id`).
3. Change the class declaration:

```python
@dataclass(frozen=True)
class OptimizationResult(Serializable):
```

4. Delete the entire `save` method (the `def save(self, path: Any) -> None:` block including its docstring and body — everything from `def save` through the `Path(path).write_text(...)` line). Keep `to_pandas` as is.

- [ ] **Step 6: Update `test/test_optimize.py`**

Replace `test_result_save_and_to_pandas_roundtrip` (which parses the old summary JSON) with:

```python
def test_result_save_and_to_pandas_roundtrip(tmp_path):
    axis = weight_axes(
        [ising.probability] * 3, FIG1A_WEIGHTS, [(0, 1)], temperature=0.25
    )
    result = optimize(
        axis, STATE, [(0.2, 1.2)], seed=5, popsize=4, maxiter=3, parallel=False
    )
    assert result.to_pandas() is result.trajectory
    path = tmp_path / "run_seed5.json"
    result.save(path)
    back = OptimizationResult.load(path)
    np.testing.assert_array_equal(back.best_params, result.best_params)
    assert back.best_objective == pytest.approx(result.best_objective)
    assert back.best_substrate == result.best_substrate
    assert back.best_sia == result.best_sia
    pd.testing.assert_frame_equal(back.trajectory, result.trajectory)
```

Then fix the file's imports: remove `import json` (its only user was the old test body) and add `from pyphi.optimize import OptimizationResult` beside the other `pyphi.optimize` imports.

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest test/serialize/test_serialize_results.py test/test_optimize.py -q 2>&1 | tail -3`
Expected: all pass. (test_optimize.py takes a couple of minutes — it runs real optimizations.)

- [ ] **Step 8: Commit**

```bash
git add pyphi/serialize/schema.py pyphi/serialize/convert.py pyphi/optimize.py test/test_optimize.py test/serialize/test_serialize_results.py
git commit -m "Serialize OptimizationResult with full fidelity

The previous OptimizationResult.save wrote a summary JSON that dropped the
winning substrate and SIA and had no load path. save/load now delegate to
pyphi.serialize and round-trip the complete result."
```

(Append the standard trailers.)

---

### Task 4: Docs, changelog, ROADMAP

**Files:**
- Modify: `docs/howto/save-load.md` (the "What is serializable" section)
- Modify: `docs/howto/landscape.md:150` (bespoke-save sentence)
- Create: `changelog.d/result-serialization.feature.md`
- Modify: `ROADMAP.md` (M12 row ~line 626; wishlist section ~line 350)

**Interfaces:**
- Consumes: `pyphi.sweep`, `SweepResult.load` (Task 2) — the docs code cells execute at build time.
- Produces: nothing consumed by other tasks.

- [ ] **Step 1: Update `docs/howto/save-load.md`**

In the "What is serializable" section, after the code cell that loops over `[system.substrate, system, sia, ces]`, insert:

```markdown
Batch-run results round-trip too: a `pyphi.sweep` table with its raw
results, and an `optimize` outcome including the winning substrate and its
analysis. Their DataFrames are embedded in the document as
[parquet](https://parquet.apache.org/), so dtypes and NaN values survive
exactly.

```{code-cell} python
import pandas as pd

result = pyphi.sweep(system.substrate, states=[system.state], progress=False)
pyphi.save(result, out / "sweep.json")
loaded = pyphi.load(out / "sweep.json")
pd.testing.assert_frame_equal(loaded.df, result.df)
loaded.df
```
```

(Keep the surrounding prose about `Analysis` unchanged. The outer fence above is illustrative — in the MyST file the code cell is a normal ` ```{code-cell} python ` block.)

- [ ] **Step 2: Update `docs/howto/landscape.md`**

Line 150 currently reads:

```
`result.save("run_seed20260711.json")` persists the trajectory and metadata.
```

Replace with:

```
`result.save("run_seed20260711.json")` persists the complete result — the
winning substrate and its analysis, the trajectory, and the run metadata —
and `pyphi.optimize.OptimizationResult.load` reads it back.
```

- [ ] **Step 3: Create the changelog fragment**

Create `changelog.d/result-serialization.feature.md`:

```markdown
`SweepResult` and `OptimizationResult` now serialize through
`pyphi.serialize`: both gain `save`/`load` (JSON, msgpack, optional gzip),
with their DataFrames embedded as parquet (pyarrow is now a core
dependency). `OptimizationResult.save` previously wrote a summary that
dropped the winning substrate and SIA; it now writes the complete result,
and both types have a load path.
```

- [ ] **Step 4: Update `ROADMAP.md`**

The M12 entry (~line 626) currently reads:

```
- **`SweepResult`/`OptimizationResult` serialization (M12).** Neither has a load path, and
  `OptimizationResult.save` drops the winning substrate and SIA. Register both in
  `pyphi.serialize`, reusing the existing Substrate/SIA schemas.
```

Replace with:

```
- **`SweepResult`/`OptimizationResult` serialization (M12).** *Landed 2026-07-20:* both are
  registered in `pyphi.serialize` with full-fidelity `save`/`load` (DataFrames embedded as
  parquet; pyarrow is now a core dependency); the lossy bespoke `OptimizationResult.save`
  is gone. Spec: `docs/superpowers/specs/2026-07-19-result-serialization-design.md`.
```

In the "Wishlist / candidate new directions" section (~line 350), add a new bullet at the end of the list:

```
- **Parquet as the convention for DataFrame disk outputs.** pyarrow is a core dependency
  as of the M12 result serialization; adopt parquet for standalone DataFrame outputs —
  benchmark aggregate tables, experiment scripts, and the provenance writer (M14) —
  replacing ad-hoc CSV, so dtypes and NaN survive and outputs interoperate with the
  wider Arrow ecosystem.
```

- [ ] **Step 5: Build the docs**

Run: `just docs > /dev/null 2>&1; echo EXIT=$?` — if nonzero, rerun capturing output to a log and read the warnings. (If the build fails with stale-autosummary "failed to import" warnings in the main tree, that cleanup applies only there — worktree builds generate fresh stubs.)
Expected: exit 0.

- [ ] **Step 6: Commit**

```bash
git add docs/howto/save-load.md docs/howto/landscape.md changelog.d/result-serialization.feature.md ROADMAP.md
git commit -m "Document SweepResult/OptimizationResult serialization"
```

(Append the standard trailers.)

---

### Final verification (before finishing-a-development-branch)

- [ ] Pathless full suite in the worktree: `uv run pytest -q > /tmp/result-serialization-suite.log 2>&1; echo EXIT=$?` then **read the summary line of the log**. Expected: no failures; doctest sweep included.
- [ ] `uv run pyright pyphi/serialize/frames.py pyphi/serialize/convert.py pyphi/serialize/schema.py pyphi/sweep.py pyphi/optimize.py 2>&1 | tail -3` — no new errors.
- [ ] Note for merge time: after merging, the **main tree** full suite needs `uv sync`/`uv pip install pyarrow` in the main venv first (pyarrow was installed there during design exploration, but confirm), and the main-tree docs build may need `rm -rf docs/reference/_autosummary` (stale stubs).
