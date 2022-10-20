# -*- coding: utf-8 -*-
# network_generator/utils.py

import numpy as np


def input_weight(element, weights, state):
    """Return the amount of input weight being sent to the given element."""
    return np.dot(state, weights[:, element])


def to_topological_ordering(element, input_weights, state, layers):
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
        layer_input_weights = input_weights[layer]
        layer_state = state[layer]
        topo_input_weights.extend(np.roll(layer_input_weights, -element))
        topo_state.extend(np.roll(layer_state, -element))
    return np.array(topo_input_weights), np.array(topo_state)


def inputs(element, weights, state, ordering="topological", layers=None):
    """Return the inputs being sent to the given element.

    Inputs are returned in the order specified by `ordering`.
    Topological ordering rotates the indices so that the element is first.
    """
    state = np.array(state)
    input_weights = weights[:, element]
    if layers is None:
        layers = [list(range(weights.shape[0]))]
    if ordering == "topological":
        input_weights, state = to_topological_ordering(
            element, input_weights, state, layers
        )
    idx = input_weights > 0
    return input_weights[idx] * np.array(state)[idx]


def binary2spin(binary_state):
    """Return the Ising spin state corresponding to the given binary state.

    This just replaces 0 with -1.
    """
    state = np.array(binary_state)
    state[np.where(state == 0)] = -1
    return state
