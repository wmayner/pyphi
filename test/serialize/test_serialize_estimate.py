"""Round-trip serialization of estimation-layer objects."""

import numpy as np
import pytest

import pyphi
from pyphi import serialize
from pyphi.estimate import estimate_substrate
from pyphi.estimate import phi_posterior

FORMATS = ["json", "msgpack"]


def round_trip(obj, fmt):
    return serialize.loads(serialize.dumps(obj, format=fmt), format=fmt)


@pytest.fixture(scope="module")
def posterior():
    traj = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [1, 0, 0]] * 5)
    return estimate_substrate(traj, regime="observational")


@pytest.mark.parametrize("fmt", FORMATS)
def test_substrate_posterior_round_trip(posterior, fmt):
    restored = round_trip(posterior, fmt)
    np.testing.assert_array_equal(restored.alpha_on, posterior.alpha_on)
    np.testing.assert_array_equal(restored.alpha_off, posterior.alpha_off)
    assert restored.regime == posterior.regime
    assert restored.prior == pytest.approx(posterior.prior)
    np.testing.assert_array_equal(restored.coverage.counts, posterior.coverage.counts)
    assert restored.provenance.estimator == posterior.provenance.estimator
    # A restored posterior is fully functional.
    a = restored.sample(seed=11)
    b = posterior.sample(seed=11)
    np.testing.assert_array_equal(
        np.asarray(a.factored_tpm.factor(0)), np.asarray(b.factored_tpm.factor(0))
    )


@pytest.mark.parametrize("fmt", FORMATS)
def test_phi_posterior_round_trip(posterior, fmt):
    with pyphi.config.override(progress_bars=False):
        pp = phi_posterior(posterior, (1, 0, 0), n_samples=4, seed=3)
    restored = round_trip(pp, fmt)
    np.testing.assert_array_equal(restored.samples, pp.samples)
    assert restored.complex_samples == pp.complex_samples
    assert restored.seed == pp.seed
    assert restored.regime == pp.regime
    assert restored.p_positive == pytest.approx(pp.p_positive)
    with pytest.raises(TypeError):
        float(restored)  # coercion semantics survive the round trip
