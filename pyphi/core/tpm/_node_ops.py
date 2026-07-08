"""Per-node-marginal array operations.

Marginalization and conditioning over a single unit's conditional
distribution, stored as a plain ndarray whose leading axes are the joint
input state at |t| and whose trailing axis is that unit's own state. These
are the array-level operations the repertoire algebra composes; they carry
no distribution type of their own.
"""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Mapping
from itertools import chain
from typing import Any

import numpy as np
from numpy.typing import NDArray


def marginalize_out(
    array: NDArray[np.float64], node_indices: Iterable[int]
) -> NDArray[np.float64]:
    """Marginalize the given input axes out of a per-unit conditional.

    Sums ``array`` over ``node_indices`` (keeping those axes as singletons)
    and divides by the product of their sizes, i.e. averages under a uniform
    distribution over the marginalized units.

    Parameters
    ----------
    array : numpy.ndarray
        A per-unit conditional whose leading axes index the input state.
    node_indices : Iterable[int]
        Input axes to marginalize out.

    Returns
    -------
    numpy.ndarray
        ``array`` with each marginalized axis collapsed to a singleton.
    """
    indices = list(node_indices)
    if not indices:
        return array
    return array.sum(tuple(indices), keepdims=True) / (
        np.array(array.shape)[indices].prod()
    )


def condition(
    array: NDArray[np.float64], fixed: Mapping[int, int]
) -> NDArray[np.float64]:
    """Condition a per-unit conditional on fixed input states.

    Fixes each input axis ``i`` present in ``fixed`` to state ``fixed[i]``,
    re-inserting a singleton axis so the number of dimensions is unchanged.
    Axes already of size 1 are left untouched.

    Parameters
    ----------
    array : numpy.ndarray
        A per-unit conditional whose leading axes index the input state.
    fixed : Mapping[int, int]
        Input axis index → state to condition on.

    Returns
    -------
    numpy.ndarray
        ``array`` with the fixed axes collapsed to singletons at their state.
    """
    selectors: list[Any] = [[slice(None)]] * (array.ndim - 1)
    for i, state_i in fixed.items():
        if array.shape[i] != 1:
            selectors[i] = [state_i, np.newaxis]
    return array[tuple(chain.from_iterable(selectors))]
