# network_generator.py

import string

import numpy as np

from ..network import Network
from ..utils import all_states
from . import ising, utils


def logical_or_function(element, weights, state):
    return utils.input_weight(element, weights, state) >= 1


UNIT_FUNCTIONS = {
    "ising": ising.probability,
    "or": logical_or_function,
}


def build_tpm(unit_function, weights, **kwargs):
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("weights must be a square matrix")
    N = weights.shape[0]
    tpm = np.zeros([2] * N + [N])
    for state in all_states(N):
        for element in range(N):
            tpm[state + (element,)] = unit_function(element, weights, state, **kwargs)
    return tpm


def build_network(
    unit_function,
    weights,
    node_labels=None,
    **kwargs,
):
    """Returns a PyPhi network given a weight matrix and a unit function.

    Args:
        unit_function (Callable): The function of a unit; must have signature
            (index, weights, state) and return a probability.
        weights: (ArrayLike) The weight matrix describing the system's connectivity.

    Keyword Args:
        **kwargs: Additional keyword arguments are passed through to the unit function.

    Returns:
        Network: A PyPhi network.
    """
    if node_labels is None:
        node_labels = string.ascii_uppercase[: weights.shape[0]]
    tpm = build_tpm(unit_function, weights, **kwargs)
    cm = weights.astype(int)
    return Network(tpm, cm=cm, node_labels=node_labels)
