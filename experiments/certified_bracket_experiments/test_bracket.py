import math

import pyphi

from experiments.certified_bracket_experiments import bracket as B


def _ces(name):
    system = getattr(pyphi.examples, name)()
    return system.ces()


def test_identity_reconstructs_sum_phi_relations_on_grid3():
    pyphi.config.progress_bars = False
    ces = _ces("grid3_system")
    profile = B.profile_from_distinctions(list(ces.distinctions))
    reconstructed = B.sum_phi_relations_lower(profile)
    assert math.isclose(
        reconstructed, float(ces.sum_phi_relations), abs_tol=1e-9
    )


def test_measured_certificate_upper_bounds_true_sum_phi_relations_on_grid3():
    pyphi.config.progress_bars = False
    ces = _ces("grid3_system")
    profile = B.profile_from_distinctions(list(ces.distinctions))
    cert = profile.self_sum + B.measured_cross_certificate(profile)
    assert cert >= float(ces.sum_phi_relations) - 1e-9
    # FINDINGS reference: grid3 state-keyed bound ≈ 9.94, true Σφ_r ≈ 3.78.
    assert math.isclose(cert, 9.94, abs_tol=0.2)
