"""End-to-end: a natively-built endorsement substrate flows through the matching
layer.

The matching paper's perceptual substrate is built from state-dependent
``resonator`` units and ``selective``/``serial`` composites. This test rebuilds
that structure natively with :func:`pyphi.substrate_generator.create_substrate`
(no dependency on the original ``substrate_modeler``), confirms its dynamic TPM
matches the recorded original (the ``endorsement`` golden), and runs it through
:class:`pyphi.matching.PerceptualSystem` to confirm the native substrate is
accepted by the existing perception machinery.
"""

from pathlib import Path

import numpy as np

from pyphi import utils
from pyphi.matching import PerceptualSystem
from pyphi.matching import triggering_coefficient
from pyphi.substrate_generator import create_substrate
from test.substrate_generator.golden_configs import GOLDEN_CONFIGS

FIXTURE = Path(__file__).parents[1] / "data" / "substrate_generator" / "dynamic_tpms.npz"


def _endorsement_substrate():
    return create_substrate(GOLDEN_CONFIGS["endorsement"], labels=("A", "B", "C"))


def test_native_endorsement_substrate_matches_original():
    sub = _endorsement_substrate()
    native = np.asarray(sub.tpm.to_joint())[..., 1]
    reference = np.load(FIXTURE)["endorsement"]
    assert np.allclose(native, reference, atol=1e-12)


def test_native_substrate_flows_through_perceptual_system():
    sub = _endorsement_substrate()
    ps = PerceptualSystem(sub, system_indices=(1, 2), sensory_indices=(0,))

    ttpm = ps.triggered_tpm(tau=2, tau_clamp=1)
    # One binary sensory unit, two system units.
    assert ttpm.array.shape == (2, 2, 2)

    states = ps.triggered_states(tau=2, tau_clamp=1)
    assert set(states) == set(utils.all_states(1))
    for response in states.values():
        assert len(response) == 2

    # Triggering coefficient for a system mechanism is well-defined in [0, 1].
    stimulus = (1,)
    response = states[stimulus]
    coeff = triggering_coefficient(ttpm, (1,), (response[0],), stimulus)
    assert 0.0 <= coeff.value <= 1.0
