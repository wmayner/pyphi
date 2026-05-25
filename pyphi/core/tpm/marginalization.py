"""Causal marginalization — named operations against IIT 4.0 Eq. 3 / Eq. 4."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from pyphi import exceptions

from .base import TPM
from .factored import FactoredTPM
from .joint import JointTPM


def cause_tpm(
    tpm: TPM,
    state: tuple[int, ...],
    node_indices: tuple[int, ...],
) -> FactoredTPM:
    """Cause TPM — IIT 4.0 Eq. 4.

    Returns a FactoredTPM with one factor per substrate unit (each factor
    of shape ``(*alphabet_sizes, k_i)``). Output factors represent
    ``P(s_i,t | s_{M,t+1} = state_M)`` per output unit, with background
    units marginalized under ``pr_bg / norm`` weighting.
    """
    if isinstance(tpm, FactoredTPM):
        if all(a == 2 for a in tpm.alphabet_sizes):
            return _cause_tpm_factored_binary(tpm, state, node_indices)
        return _cause_tpm_factored_kary(tpm, state, node_indices)
    if isinstance(tpm, JointTPM):
        factored = FactoredTPM.from_joint(tpm._inner)
        return cause_tpm(factored, state, node_indices)
    arr = tpm.to_array()
    factored = FactoredTPM.from_joint(arr)
    return cause_tpm(factored, state, node_indices)


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
) -> FactoredTPM:
    """Binary cause TPM — dispatches to the unified k-ary path.

    Binary and k-ary substrates share the same Bayesian inversion math;
    this entry point is retained as the explicit binary dispatch site.
    """
    return _cause_tpm_factored_kary(factored, state, node_indices)


def _cause_tpm_factored_kary(
    factored: FactoredTPM,
    state: tuple[int, ...],
    node_indices: tuple[int, ...],
) -> FactoredTPM:
    """Native k-ary cause TPM via per-output-unit Bayesian inversion with
    background weighting.

    Implements IIT 4.0 Eq. 4. For each system unit ``i`` and each possible
    output value ``s_i`` the returned factor stores

        factor_i(s_t)[s_i] = Σ_{w_t} P(s_i | s_t, w_t) · (pr_bg(s_t, w_t) / norm)

    where ``pr_bg`` is the joint likelihood of the observed mechanism state
    summed over the system past, ``norm`` is the joint likelihood summed
    over all past states, and the outer sum runs over background past
    states. Returned as a FactoredTPM with shape ``(*alphabet_sizes, k_i)``
    per output unit (background input dims are size 1 after the keepdims
    sum).
    """
    n = factored.n_nodes
    alphabet_sizes = factored.alphabet_sizes
    all_indices = tuple(range(n))
    system_indices = tuple(sorted(node_indices))
    background_indices = tuple(sorted(set(all_indices) - set(system_indices)))

    # Joint Bernoulli/categorical likelihood of the observed state given
    # past: pr_joint(s_t) = ∏_i factor_i(s_t)[state[i]]
    pr_joint = np.ones(alphabet_sizes, dtype=np.float64)
    for i in all_indices:
        pr_joint = pr_joint * factored.factor(i)[..., state[i]]

    # pr_bg(s_t) = Σ_{s_{M,t}} pr_joint(s_t); keepdims preserves shape
    # for broadcasting against per-factor forward values.
    if system_indices:
        pr_bg = pr_joint.sum(axis=system_indices, keepdims=True)
    else:
        pr_bg = pr_joint.copy()

    norm = pr_joint.sum()
    if norm <= 0.0:
        raise exceptions.StateUnreachableBackwardsError(state)

    weight = pr_bg / norm

    out_factors = []
    for i in all_indices:
        forward_i = factored.factor(i)  # shape (*alphabet_sizes, k_i)
        weighted = forward_i * weight[..., np.newaxis]
        if background_indices:
            weighted = weighted.sum(axis=background_indices, keepdims=True)
        out_factors.append(weighted)

    return FactoredTPM(factors=out_factors)


def _effect_tpm_factored(
    factored: FactoredTPM,
    background: Mapping[int, int],
) -> FactoredTPM:
    """Condition a factored TPM on background nodes."""
    return factored.condition(background)
