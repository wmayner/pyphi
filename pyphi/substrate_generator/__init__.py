# substrate_generator/__init__.py
"""High-level interface for creating systems by specifying architecture."""

import difflib
import inspect
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


def _check_unit_function_kwargs(
    unit_functions: str | Callable | Iterable[str | Callable],
    kwargs: Mapping[str, Any],
) -> None:
    """Raise if a keyword names no parameter of any of the unit functions.

    Keywords given to :func:`build_tpm` are passed to every unit function, and
    the unit functions accept extra keywords so that one set of keywords can
    serve a mix of functions. A misspelt keyword would therefore change the
    substrate silently; this check rejects it instead. A callable outside
    :data:`UNIT_FUNCTIONS` that accepts ``**kwargs`` may read any keyword, so
    when one is present the check is skipped.

    Parameters
    ----------
    unit_functions : str or Callable or Iterable[str or Callable]
        The unit functions, as accepted by :func:`build_tpm`.
    kwargs : Mapping[str, Any]
        The keywords that will be passed to every unit function.

    Raises
    ------
    TypeError
        If a keyword is not a named parameter of at least one unit function.
    """
    if not kwargs:
        return
    if isinstance(unit_functions, str) or callable(unit_functions):
        unit_functions = [unit_functions]
    builtin = list(UNIT_FUNCTIONS.values())
    names: set[str] = set()
    for unit_function in unit_functions:
        func = (
            UNIT_FUNCTIONS.get(unit_function, unit_function)
            if isinstance(unit_function, str)
            else unit_function
        )
        try:
            params = list(inspect.signature(func).parameters.values())
        except (TypeError, ValueError):  # not introspectable: accept anything
            return
        if func not in builtin and any(p.kind is p.VAR_KEYWORD for p in params):
            return
        names.update(
            p.name for p in params[3:] if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        )
    for key in kwargs:
        if key not in names:
            close = difflib.get_close_matches(key, names, n=1)
            hint = f" Did you mean {close[0]!r}?" if close else ""
            raise TypeError(
                f"unexpected keyword argument {key!r}: no unit function takes "
                f"it (they take {sorted(names)}).{hint}"
            )


def build_tpm(
    unit_functions: str | Callable | Iterable[str | Callable],
    weights: NDArray[Any],
    **kwargs,
):
    """Build a binary state-by-node TPM from unit functions and a weight matrix.

    Each unit function is evaluated at every one of the ``2**N`` substrate
    states, returning the probability that its node is ON at the next step.

    Parameters
    ----------
    unit_functions : str or Callable or Iterable[str or Callable]
        A single unit function (a name in :data:`UNIT_FUNCTIONS` or a callable
        with signature ``f(element, weights, state, **kwargs) -> float``)
        applied to every node, or an iterable of ``N`` such functions, one per
        node.
    weights : numpy.ndarray
        Square ``N × N`` connection weight matrix; ``weights[i, j]`` couples
        input ``i`` to node ``j``.

    Other Parameters
    ----------------
    **kwargs
        Passed through to every unit function. Each keyword must be a named
        parameter of at least one of the unit functions.

    Returns
    -------
    numpy.ndarray
        A state-by-node TPM of shape ``(2,) * N + (N,)``; entry
        ``tpm[state + (j,)]`` is the probability that node ``j`` is ON given
        ``state``.

    Raises
    ------
    ValueError
        If ``weights`` is not square, or if an iterable of unit functions does
        not have one entry per node.
    TypeError
        If a keyword argument is not a parameter of any unit function.
    """
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

    unknown = sorted(
        {f for f in unit_functions_list if isinstance(f, str)} - UNIT_FUNCTIONS.keys()
    )
    if unknown:
        raise ValueError(
            f"unknown unit function(s) {unknown}; the names are "
            f"{sorted(UNIT_FUNCTIONS)}, or pass a callable"
        )
    _check_unit_function_kwargs(unit_functions_list, kwargs)
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
    """Build a PyPhi substrate from a weight matrix and unit functions.

    The connectivity matrix is derived from the nonzero entries of ``weights``.

    Parameters
    ----------
    unit_functions : str or Callable or Iterable[str or Callable]
        A single unit function applied to every node, or an iterable of one
        function per node. Each function has signature
        ``f(element, weights, state, **kwargs) -> float`` returning a
        probability; a string names an entry in :data:`UNIT_FUNCTIONS`.
    weights : numpy.ndarray
        Square weight matrix describing the substrate's connectivity.
    node_labels : NodeLabels, optional
        Labels for the nodes. Defaults to ``A, B, C, ...``.

    Other Parameters
    ----------------
    **kwargs
        Passed through to every unit function. Each keyword must be a named
        parameter of at least one of the unit functions.

    Returns
    -------
    Substrate
        A PyPhi substrate whose TPM is built by :func:`build_tpm` and whose
        connectivity matrix marks the nonzero entries of ``weights``.
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


_SIMPLE_SPEC_KEYS = ("mechanism", "inputs", "params")
_COMPOSITE_SPEC_KEYS = ("composite", "mechanism_combination", "combination_params")


def _reject_unknown(names: Iterable[str], allowed: Iterable[str], what: str) -> None:
    """Raise ``TypeError`` for a name outside ``allowed``, with a close match."""
    allowed = sorted(allowed)
    for name in names:
        if name not in allowed:
            close = difflib.get_close_matches(name, allowed, n=1)
            hint = f" Did you mean {close[0]!r}?" if close else ""
            raise TypeError(
                f"{what} has an unknown key {name!r}; the keys are {allowed}.{hint}"
            )


def _named_options(func: Any, skip: int) -> set[str] | None:
    """The keyword options ``func`` names after its first ``skip`` parameters,
    or ``None`` when a callable outside PyPhi accepts any keyword."""
    params = list(inspect.signature(func).parameters.values())
    builtin = func in MECHANISMS.values() or func in MECHANISM_COMBINATIONS.values()
    if not builtin and any(p.kind is p.VAR_KEYWORD for p in params):
        return None
    return {
        p.name
        for p in params[skip:]
        if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD) and p.name != "inputs"
    }


def _check_spec(spec: Mapping[str, Any], where: str) -> None:
    """Raise ``TypeError`` for a key or parameter that a node spec does not use.

    A misspelt key or parameter would otherwise be dropped and the parameter
    left at its default, changing the substrate without warning.
    """
    if "composite" in spec:
        _reject_unknown(spec, _COMPOSITE_SPEC_KEYS, where)
        combination = spec.get("mechanism_combination", "selective")
        combine = (
            MECHANISM_COMBINATIONS[combination]
            if isinstance(combination, str)
            else combination
        )
        options = _named_options(combine, 1)
        if options is not None:
            _reject_unknown(
                spec.get("combination_params", {}),
                options,
                f"{where}'s combination_params",
            )
        for k, sub in enumerate(spec["composite"]):
            _check_spec(sub, f"{where}'s composite[{k}]")
        return
    _reject_unknown(spec, _SIMPLE_SPEC_KEYS, where)
    mechanism = spec["mechanism"]
    func = MECHANISMS[mechanism] if isinstance(mechanism, str) else mechanism
    options = _named_options(func, 3)
    if options is None:
        return
    if mechanism in WEIGHTED:
        options.add("input_weights")
    _reject_unknown(spec.get("params", {}), options, f"{where}'s params")


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

    Each node names a ``mechanism`` (a key in :data:`MECHANISMS`), its
    ``inputs``, and a ``params`` dict. The resulting substrate TPM matches the
    ``dynamic_tpm`` of the ``substrate_modeler`` library, in which the present
    state equals the past state.

    Parameters
    ----------
    node_params : Mapping[int, Mapping] or Sequence[Mapping]
        Either an integer-keyed mapping (nodes taken in sorted key order) or an
        ordered iterable of node specs. Each spec is a dict with keys:

        - ``"mechanism"`` (str): mechanism name, **or**
        - ``"composite"`` (list of sub-specs) plus optional
          ``"mechanism_combination"`` (str, default ``"selective"``) and
          ``"combination_params"`` (dict);
        - ``"inputs"`` (tuple[int]): the unit's input indices;
        - ``"params"`` (dict): mechanism parameters (e.g. ``input_weights``,
          ``determinism``, ``threshold``, ``weight_scale_mapping``).
    labels : Sequence[str] or NodeLabels, optional
        Node labels. Defaults to ``A, B, C, ...``.

    Returns
    -------
    Substrate
        A PyPhi substrate.

    Raises
    ------
    TypeError
        If ``node_params`` is a mapping whose keys are not all integers, a spec
        has a key not listed above, or ``params`` or ``combination_params``
        names an option the mechanism or combination does not take.

    Notes
    -----
    The connectivity matrix is assembled in two passes: every input edge is
    first marked with weight 1, then the real edge weights of weighted
    mechanisms (:data:`~pyphi.substrate_generator.mechanisms.WEIGHTED`) overwrite
    those markers. State-dependent mechanisms
    (:data:`~pyphi.substrate_generator.mechanisms.STATE_DEPENDENT`) read the
    unit's own state, so a self-loop is inserted into the connectivity matrix
    when the spec does not already include one.
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
    for j, spec in enumerate(specs):
        _check_spec(spec, f"the spec of node {j}")

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


def random_substrate(
    n: int,
    *,
    seed: int,
    epsilon: float = 0.0,
    cm: NDArray[np.int_] | None = None,
) -> Substrate:
    """A seeded random binary substrate of ``n`` units.

    Each unit's ON probability, for every previous state, is drawn
    independently and uniformly from ``(epsilon, 1 − epsilon)`` using an
    isolated generator seeded with ``seed``, so the substrate is exactly
    reproducible from ``(n, seed, epsilon)``.

    Parameters
    ----------
    n : int
        Number of binary units.
    seed : int
        Seed for the isolated random generator.
    epsilon : float, optional
        Margin keeping probabilities away from 0 and 1 (deterministic
        entries); ``0.0`` (the default) permits near-deterministic rows.
    cm : numpy.ndarray, optional
        Connectivity matrix. If ``None``, connectivity is inferred from
        the sampled TPM (generically fully connected).

    Returns
    -------
    Substrate
        The sampled substrate.
    """
    rng = np.random.default_rng(seed)
    tpm = rng.uniform(epsilon, 1.0 - epsilon, size=(2**n, n))
    return Substrate(tpm, cm=cm)
