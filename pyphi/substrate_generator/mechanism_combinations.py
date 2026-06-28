# substrate_generator/mechanism_combinations.py
"""Combination strategies for composite units, ported from ``substrate_modeler``.

A composite unit holds several sub-mechanisms and combines their activation
probabilities into one. In the present=past substrate TPM every sub-mechanism is
evaluated at the same from-state, so a combination is simply a function of the
list of sub-mechanism probabilities (the original library's per-input-state TPM
expansion is unnecessary here).

:func:`composite` builds a unit function from a list of sub-mechanism specs and a
combination strategy.
"""

import numpy as np

from .mechanisms import MECHANISMS


def selective(probs, **kwargs):
    """The sub-mechanism probability farthest from 0.5 (the most decisive)."""
    probs = np.asarray(probs, dtype=float)
    return float(probs[np.argmax(np.abs(probs - 0.5))])


def average(probs, **kwargs):
    """The mean of the sub-mechanism probabilities."""
    return float(np.mean(probs))


def maximal(probs, **kwargs):
    """The maximum sub-mechanism probability."""
    return float(np.max(probs))


def first_necessary(probs, *, steepness=5.0, offset=0.5, **kwargs):
    """The first (primary) sub-mechanism, boosted toward 1 when the others are
    inactive — but only if the primary is already above 0.5.

    ``steepness`` and ``offset`` parameterize the logistic boost (the original
    used the fixed values 5 and 0.5, which are the defaults here).
    """
    primary = probs[0]
    if primary > 0.5:
        non_primary = np.prod([1 - p for p in probs[1:]])
        max_boost = 1 - primary
        boost = max_boost / (1 + np.e ** (-steepness * (1 - non_primary - offset)))
        return float(primary + boost)
    return float(primary)


def integrator(probs, **kwargs):
    """The sum of the sub-mechanism probabilities, clamped to [0, 1]."""
    return float(min(1.0, max(0.0, np.sum(probs))))


def serial(probs, **kwargs):
    """Series combination: ``1 - prod(1 - p)`` over the sub-mechanisms."""
    remainder = 1.0
    for p in probs:
        remainder -= p * remainder
    return float(1 - remainder)


MECHANISM_COMBINATIONS = {
    "selective": selective,
    "average": average,
    "maximal": maximal,
    "first_necessary": first_necessary,
    "integrator": integrator,
    "serial": serial,
}


def composite(sub_specs, combination="selective", **combination_kwargs):
    """Build a composite unit function from sub-mechanism specs.

    Args:
        sub_specs: An iterable of dicts, each ``{"mechanism": name | callable,
            "inputs": tuple[int] | None, "params": dict}``. Each sub-mechanism is
            evaluated at the full from-state and reads its own ``inputs``.
        combination: A name in :data:`MECHANISM_COMBINATIONS` or a callable
            mapping a sequence of probabilities to a single probability.

    Keyword Args:
        **combination_kwargs: Passed to the combination strategy (e.g.
            ``steepness`` / ``offset`` for ``first_necessary``).

    Returns:
        A unit function ``f(element, weights, state, **kwargs) -> float``.
    """
    if isinstance(combination, str):
        combine = MECHANISM_COMBINATIONS[combination]
    else:
        combine = combination

    resolved = []
    for spec in sub_specs:
        mech = spec["mechanism"]
        func = MECHANISMS[mech] if isinstance(mech, str) else mech
        inputs = spec.get("inputs")
        inputs = tuple(inputs) if inputs is not None else None
        resolved.append((func, inputs, dict(spec.get("params", {}))))

    def composite_func(element, weights, state, **kwargs):
        probs = [
            func(element, weights, state, inputs=inputs, **params)
            for func, inputs, params in resolved
        ]
        return combine(probs, **combination_kwargs)

    return composite_func
