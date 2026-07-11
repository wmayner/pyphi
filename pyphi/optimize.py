"""Black-box optimization of IIT quantities over substrate parameters.

Where :mod:`pyphi.landscape` analyzes the φ landscape along one axis, this
module searches it: :func:`optimize` runs a seeded population method over a
bounded box of connection weights, looking for a substrate that maximizes an
IIT quantity — by default the signed normalized system irreducibility φₛ,
which stays continuous across minimum-information-partition switches and so
gives a gradient-free search no discontinuities to trip on
(``experiments/substrate_landscape_experiments/FINDINGS.md``).

:func:`weight_axes` builds the search space for the common case: a map from a
parameter vector to a :func:`~pyphi.substrate_generator.build_substrate`
substrate, varying a chosen set of weight-matrix entries.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


def weight_axes(
    unit_functions: Any,
    weights: NDArray[Any],
    indices: Sequence[tuple[int, int]],
    **kwargs: Any,
) -> Callable[[NDArray[Any]], Any]:
    """Return a parameter axis varying several weights of a generated substrate.

    The vector analogue of :func:`pyphi.landscape.weight_axis`.

    Parameters
    ----------
    unit_functions : str or Callable or Iterable
        Unit function(s), as accepted by
        :func:`pyphi.substrate_generator.build_substrate`.
    weights : ArrayLike
        The base weight matrix; ``weights[i, j]`` is the connection from unit
        ``i`` to unit ``j``. Copied on every call — never mutated.
    indices : Sequence[tuple[int, int]]
        The ``(i, j)`` entries to vary, in the order the parameter vector
        indexes them.

    Returns
    -------
    _WeightAxis
        A picklable callable mapping a length-``len(indices)`` vector θ to the
        substrate built from the weight matrix with ``weights[indices[k]] =
        θ[k]``. Additional keyword arguments forward to ``build_substrate`` on
        every call.

    Notes
    -----
    The returned axis is a picklable object, not a closure, so a population can
    be evaluated across worker processes (``optimize(..., parallel=True)``).

    Setting a weight to exactly 0 removes the connection from the derived
    connectivity matrix (``cm = weights != 0``), a discrete topology change,
    exactly as for :func:`pyphi.landscape.weight_axis`.
    """
    return _WeightAxis(
        unit_functions=unit_functions,
        base=np.array(weights, dtype=float),
        entries=[(int(i), int(j)) for i, j in indices],
        kwargs=dict(kwargs),
    )


@dataclass(frozen=True)
class _WeightAxis:
    """A picklable vector → substrate map varying a fixed set of weight entries.

    A module-level callable rather than a closure so the process backend can
    pickle it when a population is evaluated in parallel.
    """

    unit_functions: Any
    base: NDArray[Any]
    entries: list[tuple[int, int]]
    kwargs: dict[str, Any]

    def __call__(self, theta: NDArray[Any]) -> Any:
        from pyphi.substrate_generator import build_substrate

        varied = self.base.copy()
        for (i, j), value in zip(self.entries, np.asarray(theta), strict=True):
            varied[i, j] = value
        return build_substrate(self.unit_functions, varied, **self.kwargs)
