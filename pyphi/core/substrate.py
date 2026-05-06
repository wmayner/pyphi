"""Substrate value type — a frozen set of Units with a connectivity matrix."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from pyphi.labels import NodeLabels

from .unit import Unit


def _array_eq(a: NDArray[np.int_], b: NDArray[np.int_]) -> bool:
    return a.shape == b.shape and bool(np.array_equal(a, b))


@dataclass(frozen=True, eq=False)
class Substrate:
    """An immutable substrate: tuple of :class:`Unit` plus connectivity matrix.

    Roughly today's :class:`pyphi.network.Network` minus the TPM (which
    moves to :class:`pyphi.core.causal_model.CausalModel`).
    """

    units: tuple[Unit, ...]
    connectivity_matrix: NDArray[np.int_]

    @cached_property
    def n_units(self) -> int:
        return len(self.units)

    @cached_property
    def node_labels(self) -> NodeLabels:
        labels = tuple(u.label for u in self.units)
        indices = tuple(u.index for u in self.units)
        return NodeLabels(labels, indices)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Substrate):
            return NotImplemented
        return self.units == other.units and _array_eq(
            self.connectivity_matrix, other.connectivity_matrix
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.units,
                self.connectivity_matrix.tobytes(),
                self.connectivity_matrix.shape,
            )
        )
