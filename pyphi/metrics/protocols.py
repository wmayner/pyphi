"""Protocol types for metric callables.

Four Protocol classes capture the shape diversity in pyphi's metric
machinery. Each registered metric satisfies exactly one of these
Protocols; the registries are typed against the corresponding Protocol.

- ``DistributionMetric``: (p, q) -> float | DistanceResult.
  Distribution-to-distribution distance. Symmetric or asymmetric (see
  ``asymmetric`` attribute). Most implementations return a plain float,
  but several (ID, AID, L1, EMD, etc.) return DistanceResult.
- ``StateAwareMetric``: (p, state) -> float | DistanceResult. Pointwise
  probability at a specified state. INTRINSIC_DIFFERENTIATION returns
  DistanceResult.
- ``CompositeMetric``: (forward, partitioned, selectivity, *, state)
  -> DistanceResult. Multi-input metric returning rich metadata; used
  by GID / INTRINSIC_INFORMATION / INTRINSIC_SPECIFICATION at the
  system level.
- ``StatefulDistributionMetric``: (p, q, state) -> float | DistanceResult.
  Two-distribution metric evaluated at a specified state. Both
  distributions are load-bearing. IIT_4.0_SMALL_PHI variants return
  DistanceResult.

Plain functions don't carry the class-level ``name``/``asymmetric``
attributes the runtime-checkable Protocols declare, so ``isinstance``
returns False for them. The ``satisfies_*`` helpers below capture the
structural intent by inspecting parameter names with
``inspect.signature``; the typed registries use these at registration
time and the protocol-pinning tests use the same helpers.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING
from typing import Protocol
from typing import runtime_checkable

from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from collections.abc import Callable

    from pyphi.metrics.distribution import DistanceResult


@runtime_checkable
class DistributionMetric(Protocol):
    """Distribution-to-distribution distance."""

    name: str
    asymmetric: bool

    def __call__(self, p: ArrayLike, q: ArrayLike) -> float | DistanceResult: ...


@runtime_checkable
class StateAwareMetric(Protocol):
    """Pointwise probability at a specified state."""

    name: str

    def __call__(self, p: ArrayLike, state: object) -> float | DistanceResult: ...


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


@runtime_checkable
class StatefulDistributionMetric(Protocol):
    """Two-distribution metric evaluated at a specified state.

    Both distributions are load-bearing; the state selects a single
    element from the resulting pointwise array.
    """

    name: str

    def __call__(
        self,
        p: ArrayLike,
        q: ArrayLike,
        state: object,
    ) -> float | DistanceResult: ...


# ---------------------------------------------------------------------------
# Structural classification helpers
# ---------------------------------------------------------------------------


def _required_params(func: Callable[..., object]) -> list[str]:
    """Return the names of required positional parameters (no default)."""
    sig = inspect.signature(func)
    return [
        p.name
        for p in sig.parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]


def _all_params(func: Callable[..., object]) -> list[str]:
    """Return all parameter names (required and optional)."""
    return list(inspect.signature(func).parameters.keys())


def satisfies_distribution_metric(func: Callable[..., object]) -> bool:
    """Return True if ``func`` has the (p, q) shape of a DistributionMetric."""
    return _required_params(func) == ["p", "q"]


def satisfies_state_aware_metric(func: Callable[..., object]) -> bool:
    """Return True if ``func`` has the (p, state) shape of a StateAwareMetric."""
    return _required_params(func) == ["p", "state"]


def satisfies_composite_metric(func: Callable[..., object]) -> bool:
    """Return True if ``func`` has the (forward, partitioned, selectivity, ...)
    shape of a CompositeMetric.

    Matches on parameter-name substrings to permit the canonical PyPhi
    spellings (``forward_repertoire``, ``partitioned_forward_repertoire``,
    ``selectivity_repertoire``).
    """
    params = _all_params(func)
    if len(params) < 3:
        return False
    return (
        "forward" in params[0]
        and "partitioned" in params[1]
        and "selectivity" in params[2]
    )


def satisfies_stateful_distribution_metric(func: Callable[..., object]) -> bool:
    """Return True if ``func`` has the (p, q, state) shape of a
    StatefulDistributionMetric."""
    return _required_params(func) == ["p", "q", "state"]
