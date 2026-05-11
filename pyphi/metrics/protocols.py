"""Protocol types for metric callables.

Three Protocol classes capture the shape diversity in pyphi's metric
machinery. Each registered metric satisfies exactly one of these
Protocols; the registries are typed against the corresponding Protocol.

- ``DistributionMetric``: (p, q) -> float. Distribution-to-distribution
  distance. Symmetric or asymmetric (see ``asymmetric`` attribute).
- ``StateAwareMetric``: (p, state) -> float. Pointwise probability at a
  specified state.
- ``CompositeMetric``: (forward, partitioned, selectivity, *, state)
  -> DistanceResult. Multi-input metric returning rich metadata; used
  by GID / INTRINSIC_INFORMATION / INTRINSIC_SPECIFICATION at the
  system level.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Protocol
from typing import runtime_checkable

from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from pyphi.metrics.distribution import DistanceResult


@runtime_checkable
class DistributionMetric(Protocol):
    """Distribution-to-distribution distance."""

    name: str
    asymmetric: bool

    def __call__(self, p: ArrayLike, q: ArrayLike) -> float: ...


@runtime_checkable
class StateAwareMetric(Protocol):
    """Pointwise probability at a specified state."""

    name: str

    def __call__(self, p: ArrayLike, state: object) -> float: ...


@runtime_checkable
class CompositeMetric(Protocol):
    """Multi-input metric returning DistanceResult metadata."""

    name: str

    def __call__(
        self,
        forward: ArrayLike,
        partitioned: ArrayLike,
        selectivity: ArrayLike | None = None,
        *,
        state: object | None = None,
    ) -> DistanceResult: ...
