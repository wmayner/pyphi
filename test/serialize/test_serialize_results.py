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
            result = sweep(substrate, states=[(1, 0, 0)], compute="ces", parallel=False)
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
