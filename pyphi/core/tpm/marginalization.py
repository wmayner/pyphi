"""Causal marginalization — named operations against IIT 4.0 Eq. 3 / Eq. 4."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from pyphi import exceptions
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
        if all(a == 2 for a in tpm.alphabet_sizes):
            return _cause_tpm_factored_binary(tpm, state, node_indices)
        return _cause_tpm_factored_kary(tpm, state, node_indices)
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


def _cause_tpm_factored_binary(
    factored: FactoredTPM,
    state: tuple[int, ...],
    node_indices: tuple[int, ...],
) -> CausePosterior:
    """Binary cause TPM via the SBN-form joint backward-TPM path.

    Stacks the per-node ``P(node_i = 1 | s_t)`` slices into state-by-node form
    and applies the joint Bayesian inversion. Output shape matches the legacy
    contract ``(*alphabet_sizes, n_observed_nodes)``.
    """
    n = factored.n_nodes
    sbn = np.stack([factored.factor(i)[..., 1] for i in range(n)], axis=-1)
    joint = JointTPM(sbn)
    return CausePosterior(_legacy_backward_tpm(joint._inner, state, node_indices))


def _cause_tpm_factored_kary(
    factored: FactoredTPM,
    state: tuple[int, ...],
    node_indices: tuple[int, ...],
) -> CausePosterior:
    """Native k-ary cause posterior via per-factor likelihood product.

    Computes ``P(s_t | s_{t+1, M} = state_M)`` over the joint past-state
    space. The likelihood at each past joint state is the product of
    per-mechanism-node factor lookups; normalized over ``s_t``. Output
    shape is ``(*alphabet_sizes,)``.
    """
    alphabet_sizes = factored.alphabet_sizes
    likelihood = np.ones(alphabet_sizes, dtype=np.float64)
    for i in node_indices:
        likelihood = likelihood * factored.factor(i)[..., state[i]]
    total = likelihood.sum()
    if total <= 0.0:
        raise exceptions.StateUnreachableBackwardsError(state)
    posterior = likelihood / total
    return CausePosterior(posterior)


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
