"""Value-based tests for pyphi.dynamics."""

import numpy as np

from pyphi import convert
from pyphi import utils
from pyphi.dynamics import apply_clamp
from pyphi.dynamics import insert_clamp
from pyphi.dynamics import mean_dynamics
from pyphi.dynamics import number_of_units
from pyphi.dynamics import simulate


def test_simulate_reproduces_ising_stationary_distribution():
    """``simulate`` reproduces the analytical Ising stationary distribution.

    The Ising TPM is conditionally independent, so the state-by-node path
    samples each unit independently from its marginal and still reproduces the
    joint stationary distribution. Validates the public simulator against a
    saved analytical ground truth.
    """
    rng = np.random.default_rng(42)
    sbs = np.load("test/data/ising_tpm.npy")
    analytical = np.load("test/data/ising_stationary_distribution.npy")
    sbn = convert.state_by_state2state_by_node(sbs)  # multidimensional state-by-node
    n = sbn.shape[-1]
    # 2e6 steps puts the Monte-Carlo error floor (~1/sqrt(N_eff), with a
    # mixing time of ~3 steps here) safely under the 1e-3 tolerance at this
    # fixed seed.
    timesteps = 2_000_000
    path = simulate(sbn, initial_state=(0,) * n, timesteps=timesteps, rng=rng)
    # Map each state-vector to its little-endian decimal index (the row order
    # of the saved TPM / stationary distribution) and histogram the occupancy.
    index_of = {state: i for i, state in enumerate(utils.all_states(n))}
    counts = np.bincount([index_of[state] for state in path], minlength=sbs.shape[0])
    empirical = counts / timesteps
    assert np.allclose(empirical, analytical, atol=1e-3, rtol=0)


def test_simulate_state_by_state_reproduces_ising_stationary_distribution():
    """The full-joint state-by-state path (a pandas DataFrame) reproduces the
    analytical Ising stationary distribution.

    Complements the state-by-node test: this exercises the inverse-CDF sampler
    that draws each next state from the full joint next-state distribution.
    """
    import pandas as pd

    rng = np.random.default_rng(0)
    sbs = np.load("test/data/ising_tpm.npy")
    analytical = np.load("test/data/ising_stationary_distribution.npy")
    states = list(utils.all_states(int(np.log2(sbs.shape[0]))))
    labels = pd.MultiIndex.from_tuples(states)
    df = pd.DataFrame(sbs, index=labels, columns=labels)
    timesteps = 2_000_000
    path = simulate(df, initial_state=states[0], timesteps=timesteps, rng=rng)
    index_of = {state: i for i, state in enumerate(states)}
    counts = np.bincount(
        [index_of[tuple(int(x) for x in state)] for state in path],
        minlength=sbs.shape[0],
    )
    empirical = counts / timesteps
    assert np.allclose(empirical, analytical, atol=1e-3, rtol=0)


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


def test_settle_in_exactly_max_steps_returns():
    from pyphi.dynamics import settle

    # Every state maps to (1, 1): a one-step settle from (0, 0).
    sbs = np.zeros((4, 4))
    sbs[:, 3] = 1
    tpm = convert.state_by_state2state_by_node(sbs)
    trajectory = settle(tpm, (0, 0), max_steps=1)
    assert trajectory == [(0, 0), (1, 1)]
    assert len(trajectory) - 1 == 1  # settling time == max_steps is allowed


def test_settle_raises_when_settling_time_exceeds_max_steps():
    import pytest

    from pyphi.dynamics import settle
    from pyphi.exceptions import NonConvergenceError

    # Chain (0,0) -> (1,0) -> (1,1) -> (1,1): settling time 2.
    # Little-endian state indices: (0,0)=0, (1,0)=1, (0,1)=2, (1,1)=3.
    sbs = np.zeros((4, 4))
    sbs[0, 1] = 1
    sbs[1, 3] = 1
    sbs[2, 3] = 1
    sbs[3, 3] = 1
    tpm = convert.state_by_state2state_by_node(sbs)
    with pytest.raises(NonConvergenceError, match="max_steps"):
        settle(tpm, (0, 0), max_steps=1)
    assert settle(tpm, (0, 0), max_steps=2) == [(0, 0), (1, 0), (1, 1)]


def test_simulate_timesteps_none_with_mapping_clamp_raises():
    import numpy as np
    import pytest

    from pyphi import dynamics

    tpm = np.ones((4, 2)) * 0.5
    with pytest.raises(ValueError, match="timesteps=None"):
        dynamics.simulate(
            tpm,
            timesteps=None,
            initial_state=(0, 0),
            rng=np.random.default_rng(0),
        )


def test_simulate_seed_reproducible():
    tpm = convert.to_multidimensional(
        np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    )
    a = simulate(tpm, timesteps=20, seed=42)
    b = simulate(tpm, timesteps=20, seed=42)
    c = simulate(tpm, timesteps=20, seed=43)
    assert a == b
    assert a != c


def test_mean_dynamics_seed_reproducible():
    tpm = convert.to_multidimensional(
        np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    )
    a = mean_dynamics(tpm, repetitions=3, timesteps=5, seed=7)
    b = mean_dynamics(tpm, repetitions=3, timesteps=5, seed=7)
    assert np.array_equal(a, b)


def _ternary_cycle_joint():
    """Explicit-alphabet joint TPM of a deterministic 2-unit ternary substrate
    where each unit steps ``s -> (s + 1) mod 3``. Shape (3, 3, 2, 3)."""
    joint = np.zeros((3, 3, 2, 3))
    for a in range(3):
        for b in range(3):
            joint[a, b, 0, (a + 1) % 3] = 1.0
            joint[a, b, 1, (b + 1) % 3] = 1.0
    return joint


def test_number_of_units_explicit_alphabet_layout():
    # (3, 3, 2, 3) is (*alphabets, n_units, max_alphabet): 2 units, not 3.
    assert number_of_units(_ternary_cycle_joint()) == 2


def test_simulate_explicit_alphabet_deterministic():
    # Deterministic +1 cycle: (0,0) -> (1,1) -> (2,2) -> (0,0) -> ...
    path = simulate(_ternary_cycle_joint(), initial_state=(0, 0), timesteps=5, seed=0)
    assert path == [(0, 0), (1, 1), (2, 2), (0, 0), (1, 1)]


def test_simulate_explicit_alphabet_random_initial_state_covers_alphabet():
    # Each unit's random initial state is drawn from its own alphabet, so
    # ternary units can start in state 2.
    joint = _ternary_cycle_joint()
    initials = {simulate(joint, timesteps=1, seed=seed)[0] for seed in range(60)}
    assert all(len(state) == 2 for state in initials)
    assert any(2 in state for state in initials)


def test_simulate_explicit_alphabet_heterogeneous_padding():
    # Alphabets (2, 3): the binary unit's distribution occupies slots [:2] of
    # the padded trailing axis and must never be sampled from the padding.
    joint = np.zeros((2, 3, 2, 3))
    for a in range(2):
        for b in range(3):
            joint[a, b, 0, 1 - a] = 1.0  # binary unit flips
            joint[a, b, 1, (b + 1) % 3] = 1.0  # ternary unit cycles
    path = simulate(joint, initial_state=(0, 2), timesteps=4, seed=1)
    assert path == [(0, 2), (1, 0), (0, 1), (1, 2)]
    assert all(state[0] < 2 for state in path)


def test_settle_explicit_alphabet():
    from pyphi.dynamics import settle

    # Every state maps deterministically to (2, 2): a fixed point.
    joint = np.zeros((3, 3, 2, 3))
    joint[..., 0, 2] = 1.0
    joint[..., 1, 2] = 1.0
    assert settle(joint, (0, 0)) == [(0, 0), (2, 2)]
    assert settle(joint, (2, 2)) == [(2, 2)]


def test_settle_rejects_wrong_length_initial_state():
    import pytest

    from pyphi.dynamics import settle

    with pytest.raises(ValueError, match="initial_state"):
        settle(_ternary_cycle_joint(), (0, 0, 0))


def test_mean_dynamics_explicit_alphabet():
    # Initial states enumerate each unit's own alphabet (9 states, not 2**3),
    # and the mean trajectory has one column per unit.
    mean = mean_dynamics(_ternary_cycle_joint(), repetitions=2, timesteps=3, seed=0)
    assert mean.shape == (3, 2)
    # Deterministic +1 cycle: the mean over all 9 initial states is the mean
    # unit state, 1.0, at every step.
    assert np.allclose(mean, 1.0)


def test_simulate_state_by_state_initial_state_covers_full_alphabet():
    """The random initial state must be drawn from the TPM's own state
    labels, so non-binary units can start in states >= 2."""
    import pandas as pd

    states = [(0,), (1,), (2,)]
    tpm = pd.DataFrame(
        np.full((3, 3), 1 / 3),
        index=pd.Index(states),
        columns=pd.Index(states),
    )
    initials = {simulate(tpm, timesteps=1, seed=seed)[0] for seed in range(60)}
    assert (2,) in initials
