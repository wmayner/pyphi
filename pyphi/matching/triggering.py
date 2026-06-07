"""Triggering coefficients: how much a stimulus caused a mechanism's state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TriggeringCoefficient:
    """The extent to which a stimulus caused a mechanism's state (Eq 7).

    ``value`` is t(x, m) in [0, 1]; ``connectedness`` is c(x, m) (positive PMI,
    Eq 5); ``p`` and ``q`` are Pr(M=m | dS=x) and Pr(M=m).
    """

    value: float
    connectedness: float
    p: float
    q: float


def triggering_coefficient(triggered_tpm, mechanism, state, stimulus):
    """Compute the triggering coefficient for a mechanism state given a stimulus."""
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
