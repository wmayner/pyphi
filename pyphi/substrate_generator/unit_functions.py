# substrate_generator/unit_functions.py
"""Library of functions for single units."""

import numpy as np

from . import utils


def logical_or_function(element, weights, state, **kwargs):
    return utils.total_weighted_input(element, weights, state) >= 1


def logical_and_function(element, weights, state, **kwargs):
    # Convention: i,j means i -> j
    num_inputs = (weights[:, element] > 0).sum()
    return utils.total_weighted_input(element, weights, state) >= num_inputs


def logical_parity_function(element, weights, state, **kwargs):
    return utils.total_weighted_input(element, weights, state) % 2 >= 1


def logical_nor_function(element, weights, state, **kwargs):
    return not (logical_or_function(element, weights, state))


def logical_nand_function(element, weights, state, **kwargs):
    return not (logical_and_function(element, weights, state))


def logical_nparity_function(element, weights, state, **kwargs):
    return not (logical_parity_function(element, weights, state))


def naka_rushton(element, weights, state, exponent=2.0, threshold=1.0, **kwargs):
    x = utils.total_weighted_input(element, weights, state) ** exponent
    return x / (x + threshold)


def boolean_function(element, weights, state, on_inputs=(), **kwargs):
    """An arbitrary boolean function of the element's inputs.

    The element is ON exactly when the tuple of its (weighted) input states is
    one of ``on_inputs``. All weights must be 0 or 1.

    Parameters
    ----------
    element : int
        Index of the element whose output is being computed.
    weights : numpy.ndarray
        The weight matrix (entries restricted to 0 or 1).
    state : numpy.ndarray
        The state of the substrate.
    on_inputs : tuple of tuple, optional
        The input patterns for which the element is ON. All patterns must have
        the same length, which must equal the number of nonzero input weights.

    Returns
    -------
    bool
        The output of the element.

    Raises
    ------
    NotImplementedError
        If any weight is neither 0 nor 1.
    ValueError
        If the ``on_inputs`` patterns differ in length, or their length does not
        match the number of nonzero input weights.
    """
    if np.any((weights != 1) & (weights != 0)):
        raise NotImplementedError("weights must be 0 or 1")
    if len(set(map(len, on_inputs))) != 1:
        raise ValueError("on_inputs must all be the same length")

    inputs = tuple(utils.weighted_inputs(element, weights, state))

    # Get the length of the first on_input, or use len(inputs) if on_inputs is empty
    first_on_input = next(iter(on_inputs), inputs)
    if len(inputs) != len(first_on_input):
        raise ValueError("nonzero input weights and on_input lengths must match")

    return inputs in on_inputs


def gauss(x, mu, sigma):
    return np.exp(-0.5 * (((x - mu) / sigma) ** 2))


def gaussian(
    element,
    weights,
    state,
    mu=0.0,
    sigma=0.5,
    **kwargs,
):
    state = utils.binary2spin(state)
    x = utils.total_weighted_input(element, weights, state)
    return gauss(x, mu, sigma)
