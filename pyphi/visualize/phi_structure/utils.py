# visualize/phi_structure/utils.py
"""Common utilities for plotting |big_phi|-structures."""

from math import isclose

import numpy as np


def rescale(values, target_range):
    values = np.array(list(values))
    _min = values.min()
    _max = values.max()
    if isclose(_min, _max, rel_tol=1e-9, abs_tol=1e-9):
        x = np.ones(len(values)) * np.mean(target_range)
        return x
    return target_range[0] + (
        ((values - _min) * (target_range[1] - target_range[0])) / (_max - _min)
    )
