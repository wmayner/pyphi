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
    """Backward TPM — IIT 4.0 Eq. 4.

    Internally the per-output-unit factors are computed under a unified
    FactoredTPM-based path. The dispatcher currently wraps the result in
    CausePosterior to preserve the existing return contract; a later task
    switches the return type to FactoredTPM directly.
    """
    if isinstance(tpm, FactoredTPM):
        if all(a == 2 for a in tpm.alphabet_sizes):
            factored_out = _cause_tpm_factored_binary(tpm, state, node_indices)
        else:
            factored_out = _cause_tpm_factored_kary(tpm, state, node_indices)
        return _as_cause_posterior(factored_out)
    if isinstance(tpm, JointTPM):
        return CausePosterior(_legacy_backward_tpm(tpm._inner, state, node_indices))
    arr = tpm.to_array()
    legacy = JointTPM(arr)
    return CausePosterior(_legacy_backward_tpm(legacy._inner, state, node_indices))


def _as_cause_posterior(factored: FactoredTPM) -> CausePosterior:
    """Bridge a FactoredTPM cause result back to the legacy SBN-shaped
    CausePosterior. Stacks per-unit on-probability slices into the
    ``(*alphabet_sizes, n)`` array that downstream consumers currently
    expect.

    Binary-only: k-ary cause TPMs have no SBN-equivalent shape and must
    be consumed via ``factor(i)`` directly.
    """
    n = factored.n_nodes
    if not all(a == 2 for a in factored.alphabet_sizes):
        raise NotImplementedError(
            "Non-binary cause TPMs are not representable in SBN-form. "
            "Use the FactoredTPM-returning path."
        )
    on_slices = np.stack([factored.factor(i)[..., 1] for i in range(n)], axis=-1)
    return CausePosterior(on_slices)


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
    """Binary cause TPM via SBN-form Bayesian inversion.

    Stacks per-output-unit ``P(node_i = 1 | s_t)`` slices into the
    state-by-node form, applies the legacy backward TPM, and re-expands
    the trailing on-probability axis into explicit ``[P(off), P(on)]``
    factors per output unit.
    """
    n = factored.n_nodes
    sbn = np.stack([factored.factor(i)[..., 1] for i in range(n)], axis=-1)
    joint = JointTPM(sbn)
    raw = np.asarray(_legacy_backward_tpm(joint._inner, state, node_indices))
    out_factors = []
    for i in range(n):
        on = raw[..., i]
        off = 1.0 - on
        out_factors.append(np.stack([off, on], axis=-1))
    return FactoredTPM(factors=out_factors)


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
