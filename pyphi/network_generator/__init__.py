# network_generator/__init__.py
"""High-level interface for creating systems by specifying architecture."""

import string
from typing import Callable, Iterable, Union

import numpy as np
from numpy.typing import ArrayLike

from ..labels import NodeLabels
from ..network import Network
from ..utils import all_states
from . import ising, unit_functions, weights

UNIT_FUNCTIONS = {
    "ising": ising.probability,
    "boolean": unit_functions.boolean_function,
    "gaussian": unit_functions.gaussian,
    "naka_rushton": unit_functions.naka_rushton,
    "or": unit_functions.logical_or_function,
    "and": unit_functions.logical_and_function,
    "parity": unit_functions.logical_parity_function,
    "nor": unit_functions.logical_nor_function,
    "nand": unit_functions.logical_nand_function,
    "nparity": unit_functions.logical_nparity_function,
}


def build_tpm(
    unit_functions: Union[str, Callable, Iterable[Callable]],
    weights: ArrayLike,
    **kwargs,
):
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("weights must be a square matrix")

    N = weights.shape[0]

    if isinstance(unit_functions, Iterable):
        unit_functions = list(unit_functions)
        if len(unit_functions) != weights.shape[0]:
            raise ValueError(
                "Number of unit functions must match number of nodes in weight "
                "matrix"
            )
    else:
        unit_functions = [unit_functions] * N

    tpm = np.zeros([2] * N + [N])
    for state in all_states(N):
        for element, func in enumerate(unit_functions):
            if isinstance(func, str):
                func = UNIT_FUNCTIONS[func]
            tpm[state + (element,)] = func(element, weights, state, **kwargs)
    return tpm


def build_network(
    unit_functions: Union[Callable, Iterable[Callable]],
    weights: ArrayLike,
    node_labels: NodeLabels = None,
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
    tpm = build_tpm(unit_functions, weights, **kwargs)
    cm = (weights != 0).astype(int)
    return Network(tpm, cm=cm, node_labels=node_labels)
