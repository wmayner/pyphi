"""The joint (dense) form of a substrate TPM.

The joint peer of :class:`~pyphi.core.tpm.factored.FactoredTPM` under the
:class:`~pyphi.core.tpm.base.TPM` Protocol. A read-only snapshot of the joint
conditional ``P(sₜ₊₁ | sₜ)`` materialized as one ndarray in the
explicit-alphabet layout ``(*alphabet_sizes, n_nodes, max_alphabet)``: per
output unit ``i``, the distribution over its next state occupies slots
``[:alphabet_sizes[i]]`` of the trailing axis, and trailing slots are zero when
alphabets are heterogeneous. Produced by
:meth:`~pyphi.core.tpm.factored.FactoredTPM.to_joint` and
:meth:`~pyphi.substrate.Substrate.joint_tpm`.

Notes
-----
The array is copied at construction; the view holds no reference to its source
and does not track later mutation of it. ``FactoredTPM`` is the canonical
stored representation — this is a derived view for serialization, inspection,
and display, not for computation.
"""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from pyphi.display import LOW
from pyphi.display import Description
from pyphi.display import Displayable
from pyphi.display import Row
from pyphi.display import Section
from pyphi.models.pandas import ToPandasMixin
from pyphi.utils import all_states
from pyphi.utils import np_hash

from . import _display
from ._node_ops import condition as _condition


class JointTPM(Displayable, ToPandasMixin):
    """Read-only joint (dense) form of a substrate TPM. See module docstring."""

    __slots__ = ("_alphabet_sizes", "_array", "_node_labels")

    def __init__(
        self,
        data: ArrayLike,
        node_labels: Sequence[str] | None = None,
        alphabet_sizes: Sequence[int] | None = None,
    ) -> None:
        self._array = np.array(data, dtype=np.float64)  # copy = eager snapshot
        self._array.setflags(write=False)  # read-only value type
        self._node_labels = tuple(node_labels) if node_labels is not None else None
        if alphabet_sizes is None:
            # Unconditioned layout: the leading input axes give the per-unit
            # alphabets. (On conditioned arrays the fixed axes are singletons,
            # so pass the true sizes explicitly.)
            alphabet_sizes = self._array.shape[: self.n_nodes]
        self._alphabet_sizes = tuple(int(s) for s in alphabet_sizes)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(s) for s in self._array.shape)

    @property
    def n_nodes(self) -> int:
        # Explicit-alphabet layout: the second-to-last axis is the unit axis.
        return int(self._array.shape[-2])

    @property
    def alphabet_sizes(self) -> tuple[int, ...]:
        """The per-unit output alphabet sizes.

        Unchanged by :meth:`condition`, which collapses input axes to
        singletons but leaves each unit's output distribution intact.
        """
        return self._alphabet_sizes

    @property
    def _input_axis_sizes(self) -> tuple[int, ...]:
        """Sizes of the input-state axes (conditioned axes are singletons)."""
        return tuple(int(s) for s in self._array.shape[: self.n_nodes])

    def to_array(self) -> NDArray[np.float64]:
        return self._array

    def __array__(self, dtype: Any = None, copy: Any = None) -> NDArray[np.float64]:
        arr = self._array
        return arr.astype(dtype) if dtype is not None else arr

    def __getitem__(self, key: Any) -> Any:
        return self._array[key]

    def condition(self, fixed: Mapping[int, int]) -> JointTPM:
        """Return the joint view with the given input units fixed to a state.

        The conditioned axes collapse to singletons; the number of dimensions
        is unchanged, and :attr:`alphabet_sizes` still reports the true
        per-unit output alphabets.
        """
        return JointTPM(
            _condition(self._array, dict(fixed)),
            self._node_labels,
            alphabet_sizes=self._alphabet_sizes,
        )

    def array_equal(self, other: object) -> bool:
        return np.array_equal(self._array, np.asarray(other))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, JointTPM):
            return NotImplemented
        return np.array_equal(self._array, other._array)

    def __hash__(self) -> int:
        return np_hash(self._array)

    # ---- display ----

    def _unit_labels(self) -> list[str]:
        return list(self._node_labels or (str(i) for i in range(self.n_nodes)))

    def grid_section(self) -> Section:
        """The transition grid as a labeled display :class:`Section`.

        Mirrors :meth:`~pyphi.core.tpm.factored.FactoredTPM.grid_section`,
        reading each unit's per-state distribution from the explicit-alphabet
        array. Binary substrates get one ``P(on)`` column per unit; non-binary
        ones get one column per ``(unit, next-state)`` pair.
        """
        n = self.n_nodes
        a = self.alphabet_sizes
        input_sizes = self._input_axis_sizes
        arr = self._array
        unit_labels = self._unit_labels()
        if all(size == 2 for size in a):
            grid = _display.state_by_node_grid(
                unit_labels=unit_labels,
                state_axis_sizes=input_sizes,
                prob_on_for_state=lambda state: [arr[state][i][1] for i in range(n)],
            )
            label = "P(next unit on | current state)"
        else:
            grid = _display.distribution_grid(
                unit_labels=unit_labels,
                alphabet_sizes=a,
                state_axis_sizes=input_sizes,
                dist_for_state=lambda state: [arr[state][i][: a[i]] for i in range(n)],
            )
            label = "P(next unit = state | current state)"
        return Section(label=label, body=(grid,))

    def _describe(self, verbosity: int) -> Description:
        n = self.n_nodes
        a = self._input_axis_sizes
        total = int(np.prod(a)) if a else 1
        compact = f"JointTPM({n} units, {total} states)"
        if verbosity == LOW:  # skip building the grid for the one-liner form
            return Description(title="JointTPM", compact=compact)
        return Description(
            title="JointTPM",
            subtitle=f"{n} units · {total} states",
            sections=(
                Section(rows=(Row("Units", n), Row("States", total))),
                self.grid_section(),
            ),
            compact=compact,
        )

    def _to_pandas(self):
        import pandas as pd

        n = self.n_nodes
        a = self.alphabet_sizes
        labels = self._unit_labels()
        arr = self._array
        states = list(all_states(self._input_axis_sizes))
        if all(size == 2 for size in a):
            data = [[float(arr[s][i][1]) for i in range(n)] for s in states]
            index = (
                pd.MultiIndex.from_tuples(states, names=[f"in_{i}" for i in range(n)])
                if n > 1
                else pd.Index([s[0] for s in states], name="in_0")
            )
            return pd.DataFrame(data, index=index, columns=pd.Index(labels))
        rows = [
            {
                "state": s,
                "unit": labels[i],
                "next_state": next_state,
                "probability": float(p),
            }
            for s in states
            for i in range(n)
            for next_state, p in enumerate(arr[s][i][: a[i]])
        ]
        return pd.DataFrame(rows)

    def to_xarray(self) -> Any:
        """Return the joint as a labeled :class:`xarray.DataArray`.

        Dims are ``("u0", ..., "u{N-1}", "unit", "out")``: the leading axes
        index each unit's current state, ``unit`` selects the output unit, and
        ``out`` its next state. Values are the explicit-alphabet joint. Requires
        the optional ``xarray`` dependency.
        """
        xr = _display.require_xarray()
        n = self.n_nodes
        in_dims = tuple(f"u{j}" for j in range(n))
        coords: dict[str, list[int]] = {
            in_dims[j]: list(range(self._input_axis_sizes[j])) for j in range(n)
        }
        coords["unit"] = list(range(n))
        coords["out"] = list(range(int(self._array.shape[-1])))
        return xr.DataArray(
            self._array,
            dims=(*in_dims, "unit", "out"),
            coords=coords,
        )
