# -*- coding: utf-8 -*-
# network_generator/unit_functions.py
"""Library of functions for single units."""

import numpy as np
from toolz import curry

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


@curry
def naka_rushton(element, weights, state, exponent=2.0, threshold=1.0, **kwargs):
    x = utils.total_weighted_input(element, weights, state) ** exponent
    return x / (x + threshold)


@curry
def boolean_function(element, weights, state, on_inputs=(), **kwargs):
    """An arbitrary boolean function.

    Arguments:
        element (int): The index of the element whose output is being computed.
        weights (np.ndarray): The weight matrix.
        state (np.ndarray): The state of the network.

    Keyword Arguments:
        on_inputs (tuple): The input states that return True.

    Returns:
        bool: The output of the element.
    """
    if np.any((weights != 1) & (weights != 0)):
        raise NotImplementedError("weights must be 0 or 1")
    if len(set(map(len, on_inputs))) != 1:
        raise ValueError("on_inputs must all be the same length")

    inputs = tuple(utils.weighted_inputs(element, weights, state))

    if len(inputs) != len(next(iter(on_inputs), len(inputs))):
        raise ValueError("nonzero input weights and on_input lengths must match")

    return inputs in on_inputs


def gauss(x, mu, sigma):
    return np.exp(-0.5 * (((x - mu) / sigma) ** 2))


@curry
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
