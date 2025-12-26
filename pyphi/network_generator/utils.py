# -*- coding: utf-8 -*-
# network_generator/utils.py
"""Utilities for creating systems."""

import numpy as np


def weighted_inputs(element, weights, state):
    weights, state = inputs(element, weights, state)
    return weights * state


def inputs(element, weights, state, ordering="topological", layers=None):
    """Return the inputs being sent to the given element.

    Inputs are returned in the order specified by `ordering`.
    Topological ordering rotates the indices so that the element is first.
    """
    state = np.array(state)
    _input_weights = input_weights(element, weights)
    if layers is None:
        layers = [list(range(weights.shape[0]))]
    if ordering == "topological":
        _input_weights, state = to_topological_ordering(
            element, _input_weights, state, layers
        )
    idx = np.nonzero(_input_weights)
    return _input_weights[idx], state[idx]


def to_topological_ordering(element, weights, state, layers):
    topo_input_weights = []
    topo_state = []
    layer_sizes = set()
    for layer in layers:
        layer_sizes.add(len(layer))
        if len(layer_sizes) > 1:
            raise NotImplemented(
                "cannot use topological ordering with different layer sizes"
            )
        layer = sorted(layer)
        layer_input_weights = weights[layer]
        layer_state = state[layer]
        topo_input_weights.extend(np.roll(layer_input_weights, -element))
        topo_state.extend(np.roll(layer_state, -element))
    return np.array(topo_input_weights), np.array(topo_state)


def total_weighted_input(element, weights, state):
    """Return the amount of weighted input being sent to the given element."""
    return np.dot(state, weights[:, element])


def total_input_weight(element, weights):
    """Return the sum of connection weights being sent to the given element."""
    return np.sum(weights[:, element])


def input_weights(element, weights):
    """Return the connection weights being sent to the given element."""
    return weights[:, element]


def sigmoid(energy, temperature=1.0, field=0.0):
    """The logistic function."""
    return 1 / (1 + np.exp(-(energy - field) / temperature))


def inverse_sigmoid(p, sum_w, field):
    """The inverse of the logistic function."""
    return np.log(p / (1 - p)) / (sum_w - field)


def binary2spin(binary_state):
    """Return the Ising spin state corresponding to the given binary state.

    This just replaces 0 with -1.
    """
    state = np.array(binary_state)
    state[np.where(state == 0)] = -1
    return state
