"""cause_tpm and effect_tpm dispatch on TPM Protocol."""

from __future__ import annotations

import numpy as np

from pyphi.core.tpm.factored import FactoredTPM
from pyphi.core.tpm.joint import JointTPM
from pyphi.core.tpm.marginalization import cause_tpm
from pyphi.core.tpm.marginalization import effect_tpm


def test_cause_tpm_factored_dispatch_matches_joint() -> None:
    rng = np.random.default_rng(2026)
    joint_arr = rng.uniform(size=(2, 2, 2, 3))
    joint = JointTPM(joint_arr)
    factored = FactoredTPM.from_joint(joint_arr, alphabet_sizes=(2, 2, 2))
    state = (0, 1, 0)
    node_indices = (0, 1, 2)

    via_joint = cause_tpm(joint, state, node_indices)
    via_factored = cause_tpm(factored, state, node_indices)
    np.testing.assert_allclose(via_factored.to_array(), via_joint.to_array(), atol=1e-10)


def test_effect_tpm_factored_dispatch_matches_joint() -> None:
    rng = np.random.default_rng(99)
    joint_arr = rng.uniform(size=(2, 2, 2, 3))
    joint = JointTPM(joint_arr)
    factored = FactoredTPM.from_joint(joint_arr, alphabet_sizes=(2, 2, 2))
    background = {0: 1}

    via_joint = effect_tpm(joint, background)
    via_factored = effect_tpm(factored, background)
    np.testing.assert_allclose(via_factored.to_array(), via_joint.to_array(), atol=1e-10)
