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


def test_estimate_substrate_rejects_nan_prior():
    data = (np.zeros((2, 3), dtype=int), np.zeros((2, 3), dtype=int))
    with pytest.raises(ValueError, match="prior"):
        estimate_substrate(data, regime="perturbational", prior=float("nan"))


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


def test_phi_posterior_rejects_zero_samples(grid3_posterior):
    with pytest.raises(ValueError, match="n_samples"):
        phi_posterior(grid3_posterior, (0, 0, 0), n_samples=0, seed=1)


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


def test_infer_cm_saturates_on_estimated_tpm(grid3_posterior):
    """The exact-equality connectivity oracle reports every edge present on
    any continuously-estimated TPM: sampling noise exceeds 10^-precision on
    every axis. This documents the defect edge_probability replaces."""
    sample = grid3_posterior.sample(seed=0)
    inferred = sample.factored_tpm.infer_cm()
    assert inferred.all()  # including the truly absent edges 0->2 and 2->0


def test_edge_probability_discriminates_where_infer_cm_cannot():
    # grid3's weakest true edges (1->0, 1->2) vary the target's P(ON) by only
    # ~0.072, so the discrimination threshold must sit below that signal and
    # above the estimation noise floor. At 0.05 with enough data per state the
    # separation is exact and seed-robust (present edges fire 1.0, absent 0.0);
    # a threshold above 0.072 would only let the weak edges fire via noise.
    substrate = examples.grid3_substrate()
    pon = _ground_truth_pon(substrate)
    rng = np.random.default_rng(20260708)
    current, nxt = [], []
    for row in range(8):
        state = tuple((row >> i) & 1 for i in range(3))
        for _ in range(2000):
            current.append(state)
            nxt.append(tuple(rng.random(3) < pon[row]))
    posterior = estimate_substrate(
        (np.array(current), np.array(nxt, dtype=int)), regime="perturbational"
    )
    prob = posterior.edge_probability(n_samples=200, seed=5, threshold=0.05)
    true_cm = substrate.cm
    # Truly absent edges get low probability; present edges get high.
    assert prob[0, 2] < 0.1 and prob[2, 0] < 0.1
    assert (prob[true_cm.astype(bool)] > 0.9).all()


def test_observational_twin_nonidentifiability():
    """Two substrates identical on the observed orbit but different on the
    unvisited rows produce identical observational data yet materially
    different Φ. Deterministic: basic_substrate's free-running dynamics are
    deterministic, so no sampling is involved."""
    from pyphi import Substrate
    from pyphi import convert
    from pyphi import dynamics

    substrate = examples.basic_substrate()
    pon = _ground_truth_pon(substrate)
    rng = np.random.default_rng(1)  # unused by deterministic dynamics
    # Simulate from the state-by-node multidimensional form (the form
    # dynamics.simulate documents); both substrates go through the same path.
    traj = np.array(
        dynamics.simulate(
            convert.to_multidimensional(pon), (1, 0, 0), timesteps=50, rng=rng
        )
    )
    visited = {sum(bit << i for i, bit in enumerate(row)) for row in map(tuple, traj)}
    assert len(visited) == 3  # the orbit covers 3 of 8 states

    twin_pon = pon.copy()
    for row in set(range(8)) - visited:
        twin_pon[row] = 1.0 - twin_pon[row]  # arbitrary but deterministic
    twin = Substrate(tpm=convert.to_multidimensional(twin_pon))

    twin_traj = np.array(
        dynamics.simulate(
            convert.to_multidimensional(twin_pon), (1, 0, 0), timesteps=50, rng=rng
        )
    )
    np.testing.assert_array_equal(traj, twin_traj)  # identical data...

    phi_true = float(pyphi.analyze(substrate, (1, 0, 0), compute="sia").phi)
    phi_twin = float(pyphi.analyze(twin, (1, 0, 0), compute="sia").phi)
    assert phi_true == pytest.approx(0.415037, abs=1e-5)
    assert abs(phi_true - phi_twin) > 0.05  # ...materially different Φ


def test_epsilon_boundary_sensitivity():
    """Pushing a deterministic TPM off the 0/1 boundary by epsilon lowers Φ
    monotonically: indeterminism (real or estimated) shrinks selectivity."""
    from pyphi import Substrate
    from pyphi import convert

    pon = _ground_truth_pon(examples.basic_substrate())
    phis = []
    for eps in (0.0, 0.001, 0.02):
        smoothed = np.clip(pon, eps, 1.0 - eps)
        substrate = Substrate(tpm=convert.to_multidimensional(smoothed))
        phis.append(float(pyphi.analyze(substrate, (1, 0, 0), compute="sia").phi))
    assert phis[0] == pytest.approx(0.415037, abs=1e-5)
    assert phis[0] > phis[1] > phis[2]
    assert phis[1] == pytest.approx(0.413, abs=0.005)
    assert phis[2] == pytest.approx(0.374, abs=0.01)


def test_top_level_exports():
    assert pyphi.estimate_substrate is estimate_substrate
    assert pyphi.phi_posterior is phi_posterior


@pytest.mark.slow
def test_grid3_mixture_acceptance(grid3_posterior):
    """From five perturbational samples per state, the Φ posterior over
    grid3 at (0,0,0) is a genuine mixture: substantial mass on
    reducibility, a conditional density that brackets the true Φ, and a
    contested complex identity concentrated on the symmetric pair."""
    with pyphi.config.override(progress_bars=False):
        pp = phi_posterior(grid3_posterior, (0, 0, 0), n_samples=150, seed=2026)
    # The reference run (300 draws) gave P(phi > 0) = 0.20; a different
    # counting stream shifts this, so assert the band, not the point.
    # Observed here (seed=2026, 150 draws): p_positive = 0.24, conditional
    # 95% interval = [0.0012, 0.111], complex identity split
    # {0}: 0.39, {2}: 0.35, {1}: 0.25; wall time ~5 s.
    assert 0.05 < pp.p_positive < 0.5
    lo, hi = pp.conditional_quantiles([0.025, 0.975])
    assert lo < 0.024666 < hi  # brackets the true phi
    identity = pp.complex_identity
    assert identity[(0,)] + identity.get((2,), 0.0) > 0.5


# ---- selection-margin screening ----


def _exact_mean_posterior(substrate, tightness=200.0):
    """A posterior whose Beta means equal the substrate's true TPM exactly,
    with spread controlled by ``tightness`` (larger = tighter)."""
    from pyphi.provenance import Provenance

    pon = _ground_truth_pon(substrate)
    n = pon.shape[1]
    return SubstratePosterior(
        alpha_on=pon * tightness + 1e-9,
        alpha_off=(1.0 - pon) * tightness + 1e-9,
        regime="perturbational",
        prior=0.0,
        coverage=CoverageReport(
            counts=np.full(pon.shape[0], 1, dtype=np.int64), n_units=n
        ),
        node_labels=substrate.node_labels,
        provenance=Provenance.capture(estimator={"regime": "perturbational"}),
    )


def test_mean_substrate_matches_beta_means():
    posterior = _exact_mean_posterior(examples.grid3_substrate())
    mean = posterior.mean_substrate()
    expected = posterior.alpha_on / (posterior.alpha_on + posterior.alpha_off)
    np.testing.assert_allclose(_ground_truth_pon(mean), expected)
    assert list(mean.node_labels) == list(posterior.node_labels)


def test_screen_off_by_default(grid3_posterior):
    pp = phi_posterior(grid3_posterior, (0, 0, 0), n_samples=3, seed=1)
    assert pp.screen_margin is None
    assert pp.screened is False
    assert pp.reference_margins is None


def test_screen_refuses_on_tied_grid3():
    # grid3's true TPM has an exactly tied partition selection, so the
    # screen must refuse for any positive threshold — and record why.
    substrate = examples.grid3_substrate()
    posterior = _exact_mean_posterior(substrate)
    screened = phi_posterior(
        posterior, (0, 0, 0), n_samples=5, seed=7, screen_margin=1e-6
    )
    unscreened = phi_posterior(posterior, (0, 0, 0), n_samples=5, seed=7)
    assert screened.screened is False
    assert screened.reference_margins is not None
    assert screened.reference_margins["complex"] == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_array_equal(screened.samples, unscreened.samples)
    assert screened.complex_samples == unscreened.complex_samples


def test_screen_engages_and_matches_unscreened(monkeypatch):
    # basic_substrate's selections are clearly untied; under a tight
    # posterior the screen engages and reproduces the unscreened run.
    substrate = examples.basic_substrate()
    posterior = _exact_mean_posterior(substrate, tightness=500.0)
    calls = {"n": 0}
    from pyphi import estimate as estimate_mod

    true_maximal_complex = estimate_mod.maximal_complex

    def counting(*args, **kwargs):
        calls["n"] += 1
        return true_maximal_complex(*args, **kwargs)

    monkeypatch.setattr(estimate_mod, "maximal_complex", counting)

    screened = phi_posterior(
        posterior, (1, 0, 0), n_samples=8, seed=11, screen_margin=1e-3
    )
    screened_calls = calls["n"]
    calls["n"] = 0
    unscreened = phi_posterior(posterior, (1, 0, 0), n_samples=8, seed=11)
    unscreened_calls = calls["n"]

    assert screened.screened is True
    assert screened.screen_margin == 1e-3
    assert screened_calls == 1  # the reference only
    assert unscreened_calls == 8  # one per draw
    np.testing.assert_array_equal(screened.samples, unscreened.samples)
    assert screened.p_positive == pytest.approx(unscreened.p_positive)
    np.testing.assert_allclose(
        screened.quantiles([0.025, 0.5, 0.975]),
        unscreened.quantiles([0.025, 0.5, 0.975]),
    )
    # Identity fixed at the reference's answer; the tight posterior makes
    # the unscreened run agree.
    assert screened.complex_identity == unscreened.complex_identity


def test_screened_posterior_round_trips():
    from pyphi import serialize

    substrate = examples.basic_substrate()
    posterior = _exact_mean_posterior(substrate, tightness=500.0)
    pp = phi_posterior(posterior, (1, 0, 0), n_samples=3, seed=5, screen_margin=1e-3)
    restored = serialize.loads(serialize.dumps(pp))
    assert restored.screen_margin == pp.screen_margin
    assert restored.screened == pp.screened
    assert restored.reference_margins == pp.reference_margins
    np.testing.assert_array_equal(restored.samples, pp.samples)
