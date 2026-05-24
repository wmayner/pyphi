"""Causal marginalization — named operations against IIT 4.0 Eq. 3 / Eq. 4."""

# The FactoredTPM dispatch path for cause_tpm / effect_tpm currently
# materializes the joint conditional via SBN-form stacking and delegates
# to the legacy backward_tpm. This is a binary-only bridge; native
# per-factor Bayesian inversion is reserved for the multi-valued
# milestone.

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from pyphi.tpm import backward_tpm as _legacy_backward_tpm

from .base import TPM
from .cause_posterior import CausePosterior
from .factored import FactoredTPM
from .joint import JointTPM


def cause_tpm(
    tpm: TPM,
    state: tuple[int, ...],
    node_indices: tuple[int, ...],
) -> CausePosterior:
    """Backward TPM — IIT 4.0 Eq. 3."""
    if isinstance(tpm, FactoredTPM):
        return _cause_tpm_factored(tpm, state, node_indices)
    if isinstance(tpm, JointTPM):
        return CausePosterior(_legacy_backward_tpm(tpm._inner, state, node_indices))
    arr = tpm.to_array()
    legacy = JointTPM(arr)
    return CausePosterior(_legacy_backward_tpm(legacy._inner, state, node_indices))


def effect_tpm(
    tpm: TPM,
    background: Mapping[int, int],
) -> TPM:
    """Forward TPM conditioned on external state — IIT 4.0 Eq. 4."""
    if isinstance(tpm, FactoredTPM):
        return _effect_tpm_factored(tpm, background)
    return tpm.condition(background)


def _cause_tpm_factored(
    factored: FactoredTPM,
    state: tuple[int, ...],
    node_indices: tuple[int, ...],
) -> CausePosterior:
    """Compute the cause TPM from a factored TPM.

    Converts the factored form to the binary state-by-node representation,
    then applies the Bayesian inversion via the joint backward-TPM path.
    """
    if not all(a == 2 for a in factored.alphabet_sizes):
        raise NotImplementedError(
            f"FactoredTPM marginalization requires binary alphabets; "
            f"got alphabet_sizes={factored.alphabet_sizes}. "
            f"Multi-valued substrate analysis is the next milestone."
        )
    n = factored.n_nodes
    sbn = np.stack([factored.factor(i)[..., 1] for i in range(n)], axis=-1)
    joint = JointTPM(sbn)
    return CausePosterior(_legacy_backward_tpm(joint._inner, state, node_indices))


def _effect_tpm_factored(
    factored: FactoredTPM,
    background: Mapping[int, int],
) -> JointTPM:
    """Condition a factored TPM on background nodes.

    Converts the factored form to the binary state-by-node representation,
    then applies conditioning via the joint path.
    """
    if not all(a == 2 for a in factored.alphabet_sizes):
        raise NotImplementedError(
            f"FactoredTPM marginalization requires binary alphabets; "
            f"got alphabet_sizes={factored.alphabet_sizes}. "
            f"Multi-valued substrate analysis is the next milestone."
        )
    n = factored.n_nodes
    sbn = np.stack([factored.factor(i)[..., 1] for i in range(n)], axis=-1)
    joint = JointTPM(sbn)
    return joint.condition(background)
