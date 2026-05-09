"""Causal marginalization — named operations against IIT 4.0 Eq. 3 / Eq. 4.

Replaces the implicit ``_backward_tpm()`` side effect in
``System.__init__`` with documented free functions.
"""

from __future__ import annotations

from collections.abc import Mapping

from pyphi.tpm import backward_tpm as _legacy_backward_tpm

from .explicit import ExplicitTPM


def cause_tpm(
    tpm: ExplicitTPM,
    state: tuple[int, ...],
    node_indices: tuple[int, ...],
) -> ExplicitTPM:
    """Backward TPM — IIT 4.0 Eq. 3.

    Conditions the forward TPM on the state of the given node indices and
    inverts to obtain the cause-side conditional distribution.
    """
    return ExplicitTPM(_legacy_backward_tpm(tpm._inner, state, node_indices))


def effect_tpm(
    tpm: ExplicitTPM,
    background: Mapping[int, int],
) -> ExplicitTPM:
    """Forward TPM conditioned on external state — IIT 4.0 Eq. 4."""
    return tpm.condition(background)
