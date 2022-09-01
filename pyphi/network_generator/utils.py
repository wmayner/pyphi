# -*- coding: utf-8 -*-
# network_generator/utils.py

import numpy as np


def input_weight(element, weights, state):
    """Return the amount of input weight being sent to the given element."""
    return np.dot(state, weights[:, element])


def inputs(element, weights, state):
    """Return the inputs being sent to the given element."""
    input_weights = weights[:, element]
    idx = input_weights > 0
    return input_weights[idx] * np.array(state)[idx]
