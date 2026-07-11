import math

import pyphi
from pyphi.formalism.iit4 import bounds as PB

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


def test_partial_upper_reduces_to_measured_when_none_uncomputed():
    pyphi.config.progress_bars = False
    ces = _ces("grid3_system")
    profile = B.profile_from_distinctions(list(ces.distinctions))
    measured = profile.self_sum + B.measured_cross_certificate(profile)
    partial = B.sum_phi_relations_partial_upper(profile, uncomputed_sizes=[], n=3)
    assert math.isclose(partial, measured, rel_tol=1e-12)


def test_partial_upper_never_exceeds_general_ceiling():
    n = 3
    ces = _ces("grid3_system")
    profile = B.profile_from_distinctions(list(ces.distinctions))
    general = float(PB.sum_phi_relations_upper_bound(n, "GENERAL").value)
    # Even with a large uncomputed set the bound stays under GENERAL.
    partial = B.sum_phi_relations_partial_upper(
        profile, uncomputed_sizes=[1, 2, 3, 2, 1], n=n
    )
    assert partial <= general + 1e-9


def test_partial_upper_brackets_true_sum_on_grid3_with_one_dropped():
    # Drop one real distinction into M_u; the partial upper must still bound
    # the true Σφ_r of the FULL structure.
    pyphi.config.progress_bars = False
    ces = _ces("grid3_system")
    distinctions = list(ces.distinctions)
    dropped = distinctions[-1]
    kept = distinctions[:-1]
    profile = B.profile_from_distinctions(kept)
    n = 3
    upper = B.sum_phi_relations_partial_upper(
        profile, uncomputed_sizes=[len(dropped.mechanism)], n=n
    )
    assert upper >= float(ces.sum_phi_relations) - 1e-9
