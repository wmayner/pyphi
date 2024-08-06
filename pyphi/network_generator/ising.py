# network_generator/ising.py
"""Utitlies for implementing the Ising model."""

from toolz import curry

from . import utils


def energy(element, weights, state):
    """Return the energy associated with the given spin."""
    return utils.total_weighted_input(element, weights, state)


@curry
def probability(
    element,
    weights,
    state,
    temperature=1.0,
    field=0.0,
    constant_log_odds=False,
    **kwargs,
):
    """Return the probability that the given spin flips."""
    if temperature == 0:
        raise NotImplementedError("temperature is 0: need to decide correct behavior")

    if constant_log_odds:
        total_input_weight = weights[:, element].sum()
        if total_input_weight != 0:
            # Scale temperature by total input weight
            # This has the effect of ensuring that the ratio of log-odds ON to OFF, given
            # all inputs to a node are ON, is constant regardless of total weight
            temperature = temperature * total_input_weight

    state = utils.binary2spin(state)
    E = energy(element, weights, state)
    return utils.sigmoid(E, temperature=temperature, field=field)
