# substrate_generator/__init__.py
"""High-level interface for creating systems by specifying architecture."""

import string
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from functools import partial
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pyphi.labels import NodeLabels
from pyphi.substrate import Substrate
from pyphi.utils import all_states

from . import ising
from . import unit_functions
from .mechanism_combinations import MECHANISM_COMBINATIONS
from .mechanism_combinations import composite
from .mechanisms import MECHANISMS
from .mechanisms import STATE_DEPENDENT
from .mechanisms import WEIGHTED

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

# Register the ported substrate_modeler mechanisms under any name not already
# bound by the weighted-threshold logical gates above (so ``"and"``/``"or"``
# keep their existing meaning in :func:`build_substrate`).
UNIT_FUNCTIONS.update(
    {name: func for name, func in MECHANISMS.items() if name not in UNIT_FUNCTIONS}
)

__all__ = [
    "MECHANISMS",
    "MECHANISM_COMBINATIONS",
    "UNIT_FUNCTIONS",
    "build_substrate",
    "build_tpm",
    "composite",
    "create_substrate",
]


def build_tpm(
    unit_functions: str | Callable | Iterable[str | Callable],
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
                unit_func = UNIT_FUNCTIONS[func]
            else:
                unit_func = func
            tpm[(*state, element)] = unit_func(element, weights, state, **kwargs)
    return tpm


def build_substrate(
    unit_functions: str | Callable | Iterable[str | Callable],
    weights: NDArray[Any],
    node_labels: NodeLabels | None = None,
    **kwargs,
):
    """Returns a PyPhi substrate given a weight matrix and a unit function.

    Args:
        unit_function (Callable): The function of a unit; must have signature
            (index, weights, state) and return a probability.
        weights: (ArrayLike) The weight matrix describing the system's connectivity.

    Keyword Args:
        **kwargs: Additional keyword arguments are passed through to the unit function.

    Returns:
        Substrate: A PyPhi substrate.
    """
    if node_labels is None:
        # Create default labels from uppercase letters
        N = weights.shape[0]
        node_labels = NodeLabels(string.ascii_uppercase[:N], range(N))
    tpm = build_tpm(unit_functions, weights, **kwargs)
    cm = (weights != 0).astype(int)
    return Substrate(tpm, cm=cm, node_labels=node_labels)


def _sub_specs(spec: Mapping[str, Any]):
    """Yield the simple sub-specs of a node spec (the spec itself if not composite)."""
    if "composite" in spec:
        yield from spec["composite"]
    else:
        yield spec


def _connectivity_indices(spec: Mapping[str, Any]):
    """Yield every input index of a node spec (for connectivity markers)."""
    for sub in _sub_specs(spec):
        for i in sub["inputs"]:
            yield int(i)


def _real_weight_pairs(spec: Mapping[str, Any]):
    """Yield ``(input_index, weight)`` for the *weighted* sub-mechanisms only.

    Input weights come from ``params['input_weights']`` (or ``params['weights']``),
    aligned to ``inputs`` in order; further inputs (e.g. modulators) are left as
    connectivity markers.
    """
    for sub in _sub_specs(spec):
        if sub["mechanism"] not in WEIGHTED:
            continue
        params = sub.get("params", {})
        input_weights = params.get("input_weights", params.get("weights"))
        if input_weights is None:
            continue
        for i, w in zip(sub["inputs"], input_weights, strict=False):
            yield int(i), float(w)


def _is_state_dependent(spec: Mapping[str, Any]) -> bool:
    if "composite" in spec:
        return any(sub["mechanism"] in STATE_DEPENDENT for sub in spec["composite"])
    return spec.get("mechanism") in STATE_DEPENDENT


def _node_function(spec: Mapping[str, Any]) -> Callable:
    """Resolve a node spec to a bound unit function ``f(element, weights, state)``."""
    if "composite" in spec:
        return composite(
            spec["composite"],
            spec.get("mechanism_combination", "selective"),
            **spec.get("combination_params", {}),
        )
    base = MECHANISMS[spec["mechanism"]]
    return partial(base, inputs=tuple(spec["inputs"]), **spec.get("params", {}))


def create_substrate(
    node_params: Mapping[int, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    labels: Sequence[str] | NodeLabels | None = None,
) -> Substrate:
    """Build a :class:`~pyphi.substrate.Substrate` from per-node specifications.

    Mirrors the per-node construction of Bjørn Juel's ``substrate_modeler``: each
    node names a ``mechanism`` (a key in :data:`MECHANISMS`), its ``inputs``, and
    a ``params`` dict; the resulting substrate TPM equals the original library's
    ``dynamic_tpm`` (present state = past state).

    Args:
        node_params: Either an integer-keyed mapping or an ordered iterable of
            node specs. Each spec is a dict with keys:

            - ``"mechanism"`` (str): mechanism name, **or**
            - ``"composite"`` (list of sub-specs) + optional
              ``"mechanism_combination"`` (str, default ``"selective"``) and
              ``"combination_params"`` (dict);
            - ``"inputs"`` (tuple[int]): the unit's input indices;
            - ``"params"`` (dict): mechanism parameters (e.g. ``input_weights``,
              ``determinism``, ``threshold``, ``weight_scale_mapping``).

    Keyword Args:
        labels: Optional node labels; defaults to ``A, B, C, ...``.

    Returns:
        Substrate: A PyPhi substrate.

    State-dependent mechanisms (:data:`STATE_DEPENDENT`) read the unit's own
    state, so a self-loop is inserted into the connectivity matrix when the spec
    does not already include one.
    """
    if isinstance(node_params, Mapping):
        if not all(isinstance(k, int) for k in node_params):
            raise TypeError(
                "node_params mapping must be keyed by integer node index; "
                "pass an ordered list otherwise"
            )
        specs = [node_params[k] for k in sorted(node_params)]
    else:
        specs = list(node_params)

    n = len(specs)
    weights = np.zeros((n, n))
    functions: list[Callable] = []
    for j, spec in enumerate(specs):
        # Pass 1: connectivity markers for every input edge.
        for i in _connectivity_indices(spec):
            weights[i, j] = 1.0
        # Pass 2: real edge weights override markers for weighted mechanisms.
        for i, w in _real_weight_pairs(spec):
            weights[i, j] = w
        # Self-loop so a state-dependent unit's own-state dependence is in the CM.
        if _is_state_dependent(spec) and weights[j, j] == 0:
            weights[j, j] = 1.0
        functions.append(_node_function(spec))

    if labels is None:
        labels = NodeLabels(string.ascii_uppercase[:n], range(n))
    elif not isinstance(labels, NodeLabels):
        labels = NodeLabels(tuple(labels), range(n))
    return build_substrate(functions, weights, node_labels=labels)
