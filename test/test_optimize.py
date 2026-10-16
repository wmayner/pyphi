"""Tests for pyphi.optimize: black-box optimization over substrate weights."""

import numpy as np
import pandas as pd
import pytest

import pyphi
from pyphi import examples
from pyphi.optimization import OptimizationResult
from pyphi.optimization import _eval_one
from pyphi.optimization import _objective_value
from pyphi.optimization import optimize
from pyphi.optimization import weight_axes
from pyphi.substrate_generator import ising
from test.conftest import skip_if_no_emd_backend

# The IIT 4.0 (2023) Fig. 1A substrate; STATE is reachable with positive φ_s.
FIG1A_WEIGHTS = np.array(
    [
        [-0.2, 0.7, 0.2],
        [0.7, -0.2, 0.0],
        [0.0, -0.8, 0.2],
    ]
)
STATE = (1, 0, 0)


@pytest.fixture(autouse=True)
def _quiet():
    with pyphi.config.override(progress_bars=False):
        yield


def test_weight_axes_sets_indexed_entries_without_mutating():
    original = FIG1A_WEIGHTS.copy()
    axis = weight_axes(
        [ising.probability] * 3, FIG1A_WEIGHTS, [(0, 1), (1, 0)], temperature=0.25
    )
    substrate = axis(np.array([0.55, 0.35]))
    # Base matrix untouched.
    np.testing.assert_array_equal(FIG1A_WEIGHTS, original)
    # The built substrate is a real Substrate carrying the varied weights.
    assert substrate.node_labels is not None
    # A different vector yields a different substrate (weights actually applied).
    other = axis(np.array([0.10, 0.90]))
    one = np.asarray(substrate.joint_tpm())
    two = np.asarray(other.joint_tpm())
    assert one.shape == two.shape
    assert not np.array_equal(one, two)


def test_objective_value_named_and_callable():
    axis = weight_axes(
        [ising.probability] * 3, FIG1A_WEIGHTS, [(0, 1)], temperature=0.25
    )
    sia = pyphi.analyze(axis(np.array([0.7])), STATE, compute="sia")
    assert _objective_value(sia, "signed_normalized_phi") == pytest.approx(
        float(sia.signed_normalized_phi)
    )
    assert _objective_value(sia, lambda s: 2.0 * float(s.phi)) == pytest.approx(
        2.0 * float(sia.phi)
    )


def test_eval_one_reachable_row_carries_margins_and_sia():
    axis = weight_axes(
        [ising.probability] * 3, FIG1A_WEIGHTS, [(0, 1)], temperature=0.25
    )
    row = _eval_one(
        np.array([0.7]),
        builder=axis,
        state=STATE,
        subset=None,
        formalism=None,
        quantity="signed_normalized_phi",
    )
    assert row["reachable"] is True
    assert row["_sia"] is not None
    assert np.isfinite(row["objective"])
    assert row["cause_state"] == tuple(row["_sia"].system_state.cause.state)
    assert np.isfinite(row["partition_margin"])


def test_eval_one_unreachable_row_is_penalized_not_raised():
    substrate = examples.basic_substrate()  # deterministic; (0,1,1) never reached

    def build(_theta):
        return substrate

    row = _eval_one(
        np.array([0.0]),
        builder=build,
        state=(0, 1, 1),
        subset=None,
        formalism=None,
        quantity="signed_normalized_phi",
    )
    assert row["reachable"] is False
    assert row["_sia"] is None
    assert np.isnan(row["objective"])


def test_optimize_concentrates_above_random_mean():
    # The optimizer returns a point better than a typical blind draw. Beating a
    # same-budget random *max* is not a reliable claim at small n (random search
    # is competitive there — see the substrate-landscape FINDINGS); beating the
    # random *mean* is the robust, meaningful baseline.
    axis = weight_axes(
        [ising.probability] * 3, FIG1A_WEIGHTS, [(0, 1), (1, 0)], temperature=0.25
    )
    bounds = [(0.1, 1.3), (0.1, 1.3)]
    result = optimize(
        axis, STATE, bounds, seed=20260711, popsize=6, maxiter=8, parallel=False
    )
    rng = np.random.default_rng(20260711)
    n = result.n_evaluations
    baseline_mean = np.mean(
        [
            _eval_one(
                rng.uniform([0.1, 0.1], [1.3, 1.3]),
                builder=axis,
                state=STATE,
                subset=None,
                formalism=None,
                quantity="signed_normalized_phi",
            )["objective"]
            for _ in range(n)
        ]
    )
    assert result.best_objective >= baseline_mean
    assert result.goal == "maximize"
    assert result.seed == 20260711
    assert len(result.trajectory) == result.n_evaluations
    # The best row's objective matches best_objective.
    assert result.trajectory["objective"].max() == pytest.approx(result.best_objective)


def test_optimize_is_reproducible():
    axis = weight_axes(
        [ising.probability] * 3, FIG1A_WEIGHTS, [(0, 1)], temperature=0.25
    )
    bounds = [(0.2, 1.2)]
    kwargs = {"seed": 7, "popsize": 5, "maxiter": 5, "parallel": False}
    r1 = optimize(axis, STATE, bounds, **kwargs)
    r2 = optimize(axis, STATE, bounds, **kwargs)
    pd.testing.assert_frame_equal(r1.trajectory, r2.trajectory)
    np.testing.assert_array_equal(r1.best_params, r2.best_params)


def test_optimize_logs_unreachable_not_raised():
    substrate = examples.basic_substrate()
    result = optimize(
        lambda _t: substrate,
        (0, 1, 1),
        [(0.0, 1.0)],
        seed=1,
        popsize=4,
        maxiter=3,
        parallel=False,
    )
    assert result.n_unreachable == result.n_evaluations
    assert not result.trajectory["reachable"].any()


def test_optimize_rejects_unknown_quantity_name():
    axis = weight_axes(
        [ising.probability] * 3, FIG1A_WEIGHTS, [(0, 1)], temperature=0.25
    )
    with pytest.raises(ValueError, match="unknown quantity"):
        optimize(axis, STATE, [(0.2, 1.2)], seed=1, quantity="not_a_quantity")


def test_parallel_matches_sequential_best():
    axis = weight_axes(
        [ising.probability] * 3, FIG1A_WEIGHTS, [(0, 1), (1, 0)], temperature=0.25
    )
    bounds = [(0.1, 1.3), (0.1, 1.3)]
    kwargs = {"seed": 99, "popsize": 5, "maxiter": 5}
    seq = optimize(axis, STATE, bounds, parallel=False, **kwargs)
    par = optimize(axis, STATE, bounds, parallel=True, **kwargs)
    assert par.best_objective == pytest.approx(seq.best_objective)
    np.testing.assert_allclose(par.best_params, seq.best_params)


def test_optimize_callable_objective_and_minimize():
    axis = weight_axes(
        [ising.probability] * 3, FIG1A_WEIGHTS, [(0, 1)], temperature=0.25
    )
    result = optimize(
        axis,
        STATE,
        [(0.2, 1.2)],
        seed=3,
        quantity=lambda sia: float(sia.phi),
        goal="minimize",
        popsize=5,
        maxiter=5,
        parallel=False,
    )
    assert result.quantity_name == "<callable>"
    assert result.goal == "minimize"
    # Minimizing φ: the best is the smallest logged objective.
    assert result.best_objective == pytest.approx(result.trajectory["objective"].min())


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


@skip_if_no_emd_backend
def test_eval_one_iit3_returns_finite_phi_objective():
    axis = weight_axes(
        [ising.probability] * 3, FIG1A_WEIGHTS, [(0, 1)], temperature=0.25
    )
    row = _eval_one(
        np.array([0.7]),
        builder=axis,
        state=STATE,
        subset=None,
        formalism="IIT_3_0",
        quantity="phi",
    )
    assert row["reachable"] is True
    assert np.isfinite(row["objective"])
    assert isinstance(row["partition"], str)
    assert row["cause_state"] is None
    assert row["effect_state"] is None
    assert np.isnan(row["partition_margin"])
    assert np.isnan(row["cause_state_margin"])
    assert np.isnan(row["effect_state_margin"])


def test_objective_value_missing_attribute_raises_clear_error():
    class MinimalSIA:
        phi = 0.5

    assert _objective_value(MinimalSIA(), "phi") == 0.5
    with pytest.raises(ValueError, match="signed_normalized_phi"):
        _objective_value(MinimalSIA(), "signed_normalized_phi")


def test_an_objective_the_formalism_lacks_raises_its_own_error():
    # IIT 3.0 analyses have no signed normalized φₛ; the error must name that,
    # not scipy's complaint about its map-like callable.
    basic = examples.basic_substrate()
    with pytest.raises(ValueError, match="signed_normalized_phi"):
        optimize(
            lambda _: basic,
            (1, 0, 0),
            [(0, 1)],
            seed=1,
            formalism="iit3",
            maxiter=1,
            popsize=2,
            progress=False,
        )


def test_optimization_result_equality_does_not_raise():
    """The generated dataclass ``__eq__`` compared ndarray/DataFrame fields
    with ``==``, whose elementwise results have no truth value."""
    import numpy as np
    import pandas as pd

    from pyphi.optimization import OptimizationResult

    def make(objective):
        return OptimizationResult(
            best_params=np.array([0.1, 0.2]),
            best_objective=objective,
            best_substrate=None,
            best_sia=None,
            trajectory=pd.DataFrame({"eval": [0], "objective": [objective]}),
            bounds=[(0.0, 1.0)],
            seed=1,
            goal="maximize",
            quantity_name="phi",
            settings={},
            config_snapshot={},
            n_evaluations=1,
            n_unreachable=0,
        )

    assert make(0.5) == make(0.5)
    assert make(0.5) != make(0.6)
    assert make(0.5) != object()


def test_optimize_takes_quantity_not_objective():
    axis = weight_axes(
        [ising.probability] * 3, FIG1A_WEIGHTS, [(0, 1)], temperature=0.25
    )
    with pytest.raises(TypeError, match="objective"):
        optimize(axis, STATE, [(0.2, 1.2)], seed=1, objective="phi")  # pyright: ignore[reportCallIssue]


@pytest.mark.parametrize("subset", [None, (0, 1)])
def test_a_result_saved_by_a_release_candidate_loads_with_current_partitions(
    tmp_path, subset
):
    """rc1-rc3 wrote '[[0, 1], [2]]|[(2, 0), (2, 1)]' in the trajectory's
    partition column; loading gives the string the SIA frame writes."""
    from dataclasses import replace

    from pyphi.models.partitions import NullCut
    from pyphi.models.partitions import concise_partition
    from pyphi.partition import system_partitions
    from test.test_landscape import _rc_part_id

    axis = weight_axes(
        [ising.probability] * 3, FIG1A_WEIGHTS, [(0, 1)], temperature=0.25
    )
    with pyphi.config.override(**pyphi.iit4_2026):
        result = optimize(
            axis,
            STATE,
            [(0.2, 1.2)],
            seed=5,
            popsize=4,
            maxiter=3,
            subset=subset,
            parallel=False,
        )
        units = result.best_sia.node_indices
        labels = result.best_substrate.node_labels
        partitions = [
            *system_partitions(units, node_labels=labels),
            NullCut(units, labels),
        ]
    rc = {concise_partition(p): _rc_part_id(p) for p in partitions}
    trajectory = result.trajectory.copy()
    trajectory["partition"] = [rc.get(p, p) for p in trajectory["partition"]]
    assert (trajectory["partition"] != result.trajectory["partition"]).any()
    path = tmp_path / "result.json"
    replace(result, trajectory=trajectory).save(path)
    back = OptimizationResult.load(path)
    pd.testing.assert_frame_equal(back.trajectory, result.trajectory)
    # The conversion does not depend on the configuration active at load time.
    from pyphi.landscape import _set_partition_ids

    for active in (
        pyphi.conf.presets.by_name["IIT_3_0"],
        {**pyphi.iit4_2026, "system_partition_include_total": True},
    ):
        _set_partition_ids.cache_clear()
        with pyphi.config.override(**active):
            back = OptimizationResult.load(path)
        pd.testing.assert_frame_equal(back.trajectory, result.trajectory)


@pytest.mark.parametrize(
    "formalism",
    [
        "IIT_4_0_2026",
        "IIT_4_0_2023",
        pytest.param("IIT_3_0", marks=[pytest.mark.slow, skip_if_no_emd_backend]),
    ],
)
def test_every_trajectory_row_matches_a_direct_analysis(formalism):
    """Each evaluated point's objective equals φ_s of the substrate at that
    point computed directly, and the best row is the reported optimum."""
    axis = weight_axes(
        [ising.probability] * 3, FIG1A_WEIGHTS, [(0, 1), (1, 0)], temperature=0.25
    )
    with pyphi.config.override(**pyphi.conf.presets.by_name[formalism]):
        result = optimize(
            axis,
            STATE,
            [(0.1, 1.3), (0.1, 1.3)],
            seed=11,
            popsize=3,
            maxiter=2,
            quantity="phi",
            formalism=formalism,
            parallel=False,
        )
        df = result.to_pandas()
        direct = [
            pyphi.analyze(axis(np.array([row["p0"], row["p1"]])), STATE, compute="sia")
            for _, row in df.iterrows()
        ]
        best = pyphi.analyze(axis(np.asarray(result.best_params)), STATE, compute="sia")
    assert any(sia.phi > 0 for sia in direct)
    for (_, row), sia in zip(df.iterrows(), direct, strict=True):
        assert row["objective"] == pytest.approx(float(sia.phi), rel=1e-10, abs=1e-10)
    assert result.best_objective == pytest.approx(float(best.phi), rel=1e-10, abs=1e-10)
    assert result.best_objective == df["objective"].max()
