"""Triggering coefficients: how much a stimulus caused a mechanism's state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TriggeringCoefficient:
    """The extent to which a stimulus caused a mechanism's state (Eq. 7).

    Attributes
    ----------
    value : float
        The triggering coefficient t(x, m) ∈ [0, 1], the connectedness
        normalized by the mechanism state's self-information (Eq. 7). It is 1
        when the stimulus determines the mechanism state and 0 when the
        stimulus had no role in bringing it about.
    connectedness : float
        The connectedness c(x, m), the positive pointwise mutual information
        of the mechanism state given the stimulus (Eq. 5): log₂(p / q) when the
        stimulus raised the probability of the state, else 0.
    p : float
        Pr(M = m | ∂S = x), the conditional probability of the mechanism state
        given the stimulus.
    q : float
        Pr(M = m), the marginal probability of the mechanism state under a
        uniform prior over stimuli.
    """

    value: float
    connectedness: float
    p: float
    q: float


def triggering_coefficient(triggered_tpm, mechanism, state, stimulus):
    """Compute the triggering coefficient of a mechanism state given a stimulus.

    Parameters
    ----------
    triggered_tpm : TriggeredTPM
        The system's fixed-lag response distribution.
    mechanism : tuple of int
        The system units composing the mechanism.
    state : tuple of int
        The mechanism state whose triggering is measured.
    stimulus : tuple of int
        The sensory-interface state acting as the trigger.

    Returns
    -------
    TriggeringCoefficient
        The coefficient together with its intermediate quantities.
    """
    p = triggered_tpm.conditional_probability(mechanism, state, stimulus)
    q = triggered_tpm.marginal_probability(mechanism, state)
    # Connectedness is the positive PMI: zero unless the stimulus raised the
    # probability of the mechanism state (Eq 5).
    if p > 0 and q > 0 and p >= q:
        connectedness = float(np.log2(p / q))
    else:
        connectedness = 0.0
    # Normalize by the mechanism state's self-information (Eq 7).
    information = -float(np.log2(q)) if q > 0 else 0.0
    value = connectedness / information if information > 0 else 0.0
    return TriggeringCoefficient(value=value, connectedness=connectedness, p=p, q=q)
