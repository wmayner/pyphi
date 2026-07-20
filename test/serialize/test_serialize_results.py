"""Round-trip serialization of SweepResult and OptimizationResult."""

import math

import numpy as np
import pandas as pd
import pytest

from pyphi import examples
from pyphi import serialize
from pyphi.conf import config
from pyphi.conf import presets
from pyphi.direction import Direction
from pyphi.optimize import OptimizationResult
from pyphi.optimize import optimize
from pyphi.optimize import weight_axes
from pyphi.serialize import frames
from pyphi.substrate_generator import ising
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
        _assert_optimization_equal(optimization_result, OptimizationResult.load(path))

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
            back = serialize.loads(serialize.dumps(result, format=fmt), format=fmt)
            assert math.isnan(back.best_objective)
            assert back.best_sia is None
            _assert_optimization_equal(result, back)
