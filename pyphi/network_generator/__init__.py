# network_generator/__init__.py
"""High-level interface for creating systems by specifying architecture."""

import string
from collections.abc import Callable
from collections.abc import Iterable
from typing import Any
from typing import Union

import numpy as np
from numpy.typing import NDArray

from pyphi.labels import NodeLabels
from pyphi.network import Network
from pyphi.utils import all_states

from . import ising
from . import unit_functions
from . import weights

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
    unit_functions: str | Callable | Iterable[Callable],
    weights: NDArray[Any],
    **kwargs,
):
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("weights must be a square matrix")

    N = weights.shape[0]

    # Normalize unit_functions to a list
    if isinstance(unit_functions, str):
        # Single function name string - use for all nodes
        unit_functions_list: list[str | Callable] = [unit_functions] * N
    elif callable(unit_functions):
        # Single function - use for all nodes
        unit_functions_list = [unit_functions] * N
    else:
        # Iterable of functions
        unit_functions_list = list(unit_functions)
        if len(unit_functions_list) != weights.shape[0]:
            raise ValueError(
                "Number of unit functions must match number of nodes in weight matrix"
            )

    tpm = np.zeros([2] * N + [N])
    for state in all_states(N):
        for element, func in enumerate(unit_functions_list):
            if isinstance(func, str):
                func = UNIT_FUNCTIONS[func]
            tpm[(*state, element)] = func(element, weights, state, **kwargs)
    return tpm


def build_network(
    unit_functions: Callable | Iterable[Callable],
    weights: NDArray[Any],
    node_labels: NodeLabels | None = None,
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
        # Create default labels from uppercase letters
        N = weights.shape[0]
        node_labels = NodeLabels(string.ascii_uppercase[:N], range(N))
    tpm = build_tpm(unit_functions, weights, **kwargs)
    cm = (weights != 0).astype(int)
    return Network(tpm, cm=cm, node_labels=node_labels)
