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
    np.testing.assert_allclose(
        np.asarray(via_factored), np.asarray(via_joint), atol=1e-10
    )


def test_effect_tpm_factored_dispatch_matches_joint() -> None:
    rng = np.random.default_rng(99)
    joint_arr = rng.uniform(size=(2, 2, 2, 3))
    joint = JointTPM(joint_arr)
    factored = FactoredTPM.from_joint(joint_arr, alphabet_sizes=(2, 2, 2))
    background = {0: 1}

    via_joint = effect_tpm(joint, background)
    via_factored = effect_tpm(factored, background)
    assert isinstance(via_factored, FactoredTPM)
    # Extract P(on) slices from the factored result and compare to the SBN
    # joint result which stores P(node_i=1 | s_t) in the last axis.
    joint_sbn = via_joint.to_array()  # shape: (1, 2, 2, 3)
    for i in range(factored.n_nodes):
        p_on_factored = via_factored.factor(i)[..., 1]  # shape: (1, 2, 2)
        np.testing.assert_allclose(
            p_on_factored,
            joint_sbn[..., i],
            atol=1e-10,
            err_msg=f"factor {i} P(on) mismatch",
        )


def test_effect_tpm_kary_factored_returns_factored_tpm() -> None:
    """effect_tpm on a k>2 FactoredTPM returns a FactoredTPM."""
    f0 = np.full((3, 3, 3), 1.0 / 3.0)
    f1 = np.full((3, 3, 3), 1.0 / 3.0)
    factored = FactoredTPM(factors=[f0, f1], alphabet_sizes=(3, 3))
    result = effect_tpm(factored, background={0: 0})
    assert isinstance(result, FactoredTPM)


def test_cause_tpm_returns_cause_posterior_for_jointtpm_input() -> None:
    """cause_tpm wraps the backward_tpm result in CausePosterior for JointTPM input."""
    from pyphi.core.tpm.cause_posterior import CausePosterior

    rng = np.random.default_rng(2026)
    joint_arr = rng.uniform(size=(2, 2, 2, 3))
    joint = JointTPM(joint_arr)
    result = cause_tpm(joint, state=(0, 1, 0), node_indices=(0, 1, 2))
    assert isinstance(result, CausePosterior)


def test_cause_tpm_returns_cause_posterior_for_factored_input() -> None:
    """cause_tpm wraps the backward_tpm result in CausePosterior for FactoredTPM input."""
    from pyphi.core.tpm.cause_posterior import CausePosterior

    rng = np.random.default_rng(2026)
    joint_arr = rng.uniform(size=(2, 2, 2, 3))
    factored = FactoredTPM.from_joint(joint_arr, alphabet_sizes=(2, 2, 2))
    result = cause_tpm(factored, state=(0, 1, 0), node_indices=(0, 1, 2))
    assert isinstance(result, CausePosterior)
