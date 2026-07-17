"""Tests for pyphi.optimize: black-box optimization over substrate weights."""

import json

import numpy as np
import pandas as pd
import pytest

import pyphi
from pyphi import examples
from pyphi.optimize import _eval_one
from pyphi.optimize import _objective_value
from pyphi.optimize import optimize
from pyphi.optimize import weight_axes
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
        objective="signed_normalized_phi",
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
        objective="signed_normalized_phi",
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
                objective="signed_normalized_phi",
            )["objective"]
            for _ in range(n)
        ]
    )
    assert result.best_objective >= baseline_mean
    assert result.direction == "maximize"
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


def test_optimize_rejects_unknown_objective_name():
    axis = weight_axes(
        [ising.probability] * 3, FIG1A_WEIGHTS, [(0, 1)], temperature=0.25
    )
    with pytest.raises(ValueError, match="unknown objective"):
        optimize(axis, STATE, [(0.2, 1.2)], seed=1, objective="not_a_quantity")


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
        objective=lambda sia: float(sia.phi),
        direction="minimize",
        popsize=5,
        maxiter=5,
        parallel=False,
    )
    assert result.objective_name == "<callable>"
    assert result.direction == "minimize"
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
    payload = json.loads(path.read_text())
    assert payload["seed"] == 5
    assert len(payload["trajectory"]) == result.n_evaluations
    assert payload["best_objective"] == pytest.approx(result.best_objective)


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
        objective="phi",
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
