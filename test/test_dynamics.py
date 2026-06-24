"""Value-based tests for pyphi.dynamics (the functions not already covered by
test_tpm.py's ising stationary-distribution test)."""

import numpy as np

from pyphi.dynamics import apply_clamp
from pyphi.dynamics import insert_clamp
from pyphi.dynamics import mean_dynamics
from pyphi.dynamics import number_of_units
from pyphi.dynamics import simulate


def test_apply_clamp():
    # apply_clamp overwrites in place by index (no length change).
    assert apply_clamp({1: 0}, (1, 1, 1)) == (1, 0, 1)
    assert apply_clamp({}, (1, 1)) == (1, 1)  # empty clamp is identity


def test_insert_clamp():
    # insert_clamp inserts the clamped values at their indices (length grows).
    assert insert_clamp({1: 9}, (1, 1)) == (1, 9, 1)
    assert insert_clamp({}, (1, 1)) == (1, 1)  # empty clamp is identity


def test_number_of_units():
    tpm = np.zeros((2, 2, 2))  # state-by-node, 2 binary units
    assert number_of_units(tpm) == 2


def test_simulate_deterministic_tpm():
    # A state-by-node TPM with every entry == 1 sends both units to ON each step,
    # independent of the RNG (P(on) = 1 > any threshold in [0, 1)).
    tpm = np.ones((2, 2, 2))
    rng = np.random.default_rng(0)
    path = simulate(tpm, initial_state=(0, 0), timesteps=3, rng=rng)
    assert path == [(0, 0), (1, 1), (1, 1)]


def test_simulate_rejects_wrong_length_initial_state():
    tpm = np.ones((2, 2, 2))
    rng = np.random.default_rng(0)
    try:
        simulate(tpm, initial_state=(0, 0, 0), timesteps=2, rng=rng)
    except ValueError as e:
        assert "initial_state" in str(e)
    else:
        raise AssertionError("expected ValueError for wrong-length initial_state")


def test_mean_dynamics_deterministic():
    # All-ones TPM -> every trajectory converges to ON; the per-step mean over all
    # initial states converges to 1 for both units after the first transition.
    tpm = np.ones((2, 2, 2))
    rng = np.random.default_rng(0)
    mean = mean_dynamics(tpm, repetitions=2, timesteps=3, rng=rng)
    # mean has shape (timesteps+1, N); steps 1.. are all ON.
    assert np.allclose(mean[1:], 1.0)


def _sbn_from_sbs(sbs):
    from pyphi import convert

    # state-by-state (rows=current, cols=next) -> multidim state-by-node
    return convert.state_by_state2state_by_node(sbs)


def test_settle_reaches_fixed_point():
    from pyphi.dynamics import settle

    # 2-unit system: deterministic map driving any state to (1, 1) (le-index 3)
    sbs = np.zeros((4, 4))
    sbs[:, 3] = 1.0
    tpm = _sbn_from_sbs(sbs)
    trajectory = settle(tpm, initial_state=(0, 0))
    assert trajectory[-1] == (1, 1)
    assert isinstance(trajectory, list)


def test_settle_already_fixed_returns_length_one():
    from pyphi.dynamics import settle

    sbs = np.zeros((4, 4))
    sbs[:, 3] = 1.0
    tpm = _sbn_from_sbs(sbs)
    assert settle(tpm, initial_state=(1, 1)) == [(1, 1)]


def test_settle_raises_on_limit_cycle():
    import pytest

    from pyphi.dynamics import settle
    from pyphi.exceptions import NonConvergenceError

    # 1-unit system that flips every step: (0,)->(1,)->(0,)->...
    sbs = np.array([[0.0, 1.0], [1.0, 0.0]])
    tpm = _sbn_from_sbs(sbs)
    with pytest.raises(NonConvergenceError, match="cycle"):
        settle(tpm, initial_state=(0,))


def test_settle_clamp_holds_units_fixed():
    from pyphi.dynamics import settle

    # both units flip toward all-ON, but clamp unit 0 OFF -> fixed point (0, 1)
    sbs = np.zeros((4, 4))
    sbs[:, 3] = 1.0
    tpm = _sbn_from_sbs(sbs)
    assert settle(tpm, initial_state=(0, 0), clamp={0: 0})[-1] == (0, 1)
