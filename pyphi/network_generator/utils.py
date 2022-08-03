# utils.py

import numpy as np


def input_weight(element, weights, state):
    """Return the amount of input weight being sent to the given element."""
    return np.dot(state, weights[:, element])
