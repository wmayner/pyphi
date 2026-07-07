"""Tests for substrate estimation from data and the epistemic-uncertainty layer."""

import numpy as np
import pytest

import pyphi
from pyphi import examples
from pyphi.estimate import CoverageReport
from pyphi.estimate import PhiPosterior
from pyphi.estimate import SubstratePosterior
from pyphi.estimate import estimate_substrate
from pyphi.estimate import phi_posterior


@pytest.fixture(autouse=True)
def _quiet():
    with pyphi.config.override(progress_bars=False):
        yield


def _ground_truth_pon(substrate):
    """State-by-node P(unit = ON | current state), little-endian row order."""
    import itertools

    ft = substrate.factored_tpm
    n = ft.n_nodes
    pon = np.zeros((2**n, n))
    for state in itertools.product((0, 1), repeat=n):
        row = sum(bit << i for i, bit in enumerate(state))
        for i in range(n):
            factor = ft.factor(i)
            idx = (
                *(state[j] if factor.shape[j] > 1 else 0 for j in range(n)),
                1,
            )
            pon[row, i] = factor[idx]
    return pon


def _exhaustive_transitions(substrate, repeats=1):
    """Deterministic (current, next) pairs covering every state ``repeats`` times."""
    import itertools

    pon = _ground_truth_pon(substrate)
    n = pon.shape[1]
    current = []
    for state in itertools.product((0, 1), repeat=n):
        row = sum(bit << i for i, bit in enumerate(state))
        current.extend(
            (state, tuple((pon[row] > 0.5).astype(int))) for _ in range(repeats)
        )
    cur, nxt = zip(*current, strict=True)
    return np.array(cur), np.array(nxt)


def test_regime_is_required():
    data = (np.zeros((2, 3), dtype=int), np.zeros((2, 3), dtype=int))
    with pytest.raises(TypeError):
        estimate_substrate(data)  # pyright: ignore[reportCallIssue]
    with pytest.raises(ValueError, match="regime"):
        estimate_substrate(data, regime="empirical")


def test_binary_only():
    data = (np.full((2, 3), 2), np.zeros((2, 3), dtype=int))
    with pytest.raises(ValueError, match="binary"):
        estimate_substrate(data, regime="perturbational")


def test_counts_model_only():
    data = (np.zeros((2, 3), dtype=int), np.zeros((2, 3), dtype=int))
    with pytest.raises(NotImplementedError):
        estimate_substrate(data, regime="perturbational", model="glm")


def test_posterior_mean_recovers_asymmetric_ground_truth():
    """With exact deterministic transitions and a vanishing prior, the
    posterior concentrates on the true asymmetric TPM (this is also the
    endianness/axis-order guard: OR/AND/XOR has no symmetry to hide a
    reversed axis)."""
    substrate = examples.basic_substrate()
    data = _exhaustive_transitions(substrate, repeats=50)
    posterior = estimate_substrate(data, regime="perturbational", prior=1e-6)
    mean = posterior.alpha_on / (posterior.alpha_on + posterior.alpha_off)
    np.testing.assert_allclose(mean, _ground_truth_pon(substrate), atol=1e-4)


def test_sample_returns_working_substrate_and_phi_converges():
    substrate = examples.basic_substrate()
    data = _exhaustive_transitions(substrate, repeats=200)
    posterior = estimate_substrate(data, regime="perturbational", prior=0.05)
    sampled = posterior.sample(seed=7)
    assert isinstance(sampled, type(substrate))
    sia = pyphi.analyze(sampled, (1, 0, 0), compute="sia")
    assert float(sia.phi) == pytest.approx(0.415037, abs=0.05)


def test_sample_seed_discipline():
    substrate = examples.basic_substrate()
    data = _exhaustive_transitions(substrate)
    posterior = estimate_substrate(data, regime="perturbational")
    with pytest.raises(ValueError, match="seed"):
        posterior.sample()
    a = posterior.sample(seed=3)
    b = posterior.sample(seed=3)
    np.testing.assert_array_equal(
        np.asarray(a.factored_tpm.factor(0)), np.asarray(b.factored_tpm.factor(0))
    )
    rng = np.random.default_rng(3)
    c = posterior.sample(rng=rng)
    np.testing.assert_array_equal(
        np.asarray(a.factored_tpm.factor(0)), np.asarray(c.factored_tpm.factor(0))
    )


def test_trajectory_form_equals_pair_form():
    traj = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1], [1, 0, 0]])
    from_traj = estimate_substrate(traj, regime="observational")
    from_pairs = estimate_substrate((traj[:-1], traj[1:]), regime="observational")
    np.testing.assert_array_equal(from_traj.alpha_on, from_pairs.alpha_on)
    np.testing.assert_array_equal(from_traj.coverage.counts, from_pairs.coverage.counts)


def test_jeffreys_is_default_prior():
    data = (np.zeros((1, 2), dtype=int), np.zeros((1, 2), dtype=int))
    posterior = estimate_substrate(data, regime="perturbational")
    assert isinstance(posterior, SubstratePosterior)
    assert posterior.prior == pytest.approx(0.5)
    # An unvisited row sits at the bare prior.
    assert posterior.alpha_on[3, 0] == pytest.approx(0.5)
    assert posterior.alpha_off[3, 0] == pytest.approx(0.5)


def test_coverage_report():
    traj = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [1, 0, 0]])
    posterior = estimate_substrate(traj, regime="observational")
    report = posterior.coverage
    assert isinstance(report, CoverageReport)
    assert report.n_states == 8
    assert report.counts[0] == 2  # (0,0,0) observed as a current state twice
    assert report.counts[1] == 1  # (1,0,0) once (the final row has no successor)
    assert not report.is_complete
    assert report.fraction_covered == pytest.approx(2 / 8)
    assert (0, 1, 0) in report.uncovered_states
    assert (0, 0, 0) not in report.uncovered_states
    df = report.to_pandas()
    assert set(df.columns) >= {"state", "count"}
    assert len(df) == 8


def test_full_coverage_report_is_complete():
    substrate = examples.basic_substrate()
    data = _exhaustive_transitions(substrate)
    posterior = estimate_substrate(data, regime="perturbational")
    assert posterior.coverage.is_complete
    assert posterior.coverage.uncovered_states == ()
    assert posterior.coverage.fraction_covered == pytest.approx(1.0)


def test_provenance_records_estimator():
    substrate = examples.basic_substrate()
    data = _exhaustive_transitions(substrate, repeats=2)
    posterior = estimate_substrate(data, regime="perturbational", prior=0.5)
    record = posterior.provenance.estimator
    assert record is not None
    assert record["regime"] == "perturbational"
    assert record["model"] == "counts"
    assert record["prior"] == pytest.approx(0.5)
    assert record["n_transitions"] == 16
    assert record["n_states_observed"] == 8
    assert record["n_states_total"] == 8
    assert record["uncovered_state_count"] == 0


def test_provenance_records_observational_assertion():
    traj = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
    posterior = estimate_substrate(traj, regime="observational")
    record = posterior.provenance.estimator
    assert record["regime"] == "observational"
    assert record["uncovered_state_count"] == 6


@pytest.fixture(scope="module")
def grid3_posterior():
    """A seeded posterior over grid3 from sparse perturbational data."""
    substrate = examples.grid3_substrate()
    pon = _ground_truth_pon(substrate)
    rng = np.random.default_rng(20260708)
    current, nxt = [], []
    for row in range(8):
        state = tuple((row >> i) & 1 for i in range(3))
        for _ in range(5):
            current.append(state)
            nxt.append(tuple(rng.random(3) < pon[row]))
    data = (np.array(current), np.array(nxt, dtype=int))
    return estimate_substrate(data, regime="perturbational")


@pytest.fixture(scope="module")
def grid3_phi_posterior(grid3_posterior):
    with pyphi.config.override(progress_bars=False):
        return phi_posterior(grid3_posterior, (0, 0, 0), n_samples=40, seed=99)


def test_phi_posterior_seed_required(grid3_posterior):
    with pytest.raises(TypeError):
        phi_posterior(grid3_posterior, (0, 0, 0), n_samples=2)  # pyright: ignore[reportCallIssue]


def test_phi_posterior_is_reproducible(grid3_posterior):
    a = phi_posterior(grid3_posterior, (0, 0, 0), n_samples=5, seed=42)
    b = phi_posterior(grid3_posterior, (0, 0, 0), n_samples=5, seed=42)
    np.testing.assert_array_equal(a.samples, b.samples)
    assert a.complex_samples == b.complex_samples
    assert a.seed == 42


def test_phi_posterior_is_a_mixture(grid3_phi_posterior):
    pp = grid3_phi_posterior
    assert isinstance(pp, PhiPosterior)
    assert pp.samples.shape == (40,)
    assert 0.0 < pp.p_positive < 1.0  # both mixture components present
    lo, hi = pp.quantiles([0.025, 0.975])
    assert lo == pytest.approx(0.0)
    assert hi > 0.0
    cond = pp.conditional_quantiles([0.5])
    assert cond is not None and cond[0] > 0.0


def test_phi_posterior_refuses_float_coercion(grid3_phi_posterior):
    with pytest.raises(TypeError, match="p_positive"):
        float(grid3_phi_posterior)


def test_complex_identity_is_categorical(grid3_phi_posterior):
    identity = grid3_phi_posterior.complex_identity
    assert sum(identity.values()) == pytest.approx(1.0)
    # grid3's exact {0} vs {2} tie at the true TPM is broken randomly by
    # the data, so more than one identity appears.
    assert len(identity) > 1
    assert grid3_phi_posterior.complex_samples[0] in identity


def test_phi_posterior_carries_regime_and_coverage(grid3_phi_posterior):
    assert grid3_phi_posterior.regime == "perturbational"
    assert grid3_phi_posterior.coverage.is_complete


def test_observational_result_reports_partial_coverage():
    traj = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [1, 0, 0]] * 3)
    posterior = estimate_substrate(traj, regime="observational")
    pp = phi_posterior(posterior, (1, 0, 0), n_samples=3, seed=1)
    assert pp.regime == "observational"
    assert not pp.coverage.is_complete
    with pytest.raises(TypeError, match=r"unconstrained|uncovered"):
        float(pp)
