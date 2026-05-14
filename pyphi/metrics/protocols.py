"""Protocol types for measure callables.

Four Protocol classes capture the shape diversity in pyphi's measure
machinery. Each registered measure satisfies exactly one of these
Protocols; the registries are typed against the corresponding Protocol.

- ``DistributionMeasure``: (p, q) -> float | DistanceResult.
  Distribution-to-distribution distance. Symmetric or asymmetric (see
  ``asymmetric`` attribute). Most implementations return a plain float,
  but several (ID, AID, L1, EMD, etc.) return DistanceResult.
- ``StateAwareMeasure``: (p, state) -> float | DistanceResult. Pointwise
  probability at a specified state. INTRINSIC_DIFFERENTIATION returns
  DistanceResult.
- ``CompositeMeasure``: (forward, partitioned, selectivity, *, state)
  -> DistanceResult. Multi-input measure returning rich metadata; used
  by GID / INTRINSIC_INFORMATION / INTRINSIC_SPECIFICATION at the
  system level.
- ``StatefulDistributionMeasure``: (p, q, state) -> float | DistanceResult.
  Two-distribution measure evaluated at a specified state. Both
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
class DistributionMeasure(Protocol):
    """Distribution-to-distribution distance."""

    name: str
    asymmetric: bool

    def __call__(self, p: ArrayLike, q: ArrayLike) -> float | DistanceResult: ...


@runtime_checkable
class StateAwareMeasure(Protocol):
    """Pointwise probability at a specified state."""

    name: str

    def __call__(self, p: ArrayLike, state: object) -> float | DistanceResult: ...


@runtime_checkable
class CompositeMeasure(Protocol):
    """Multi-input measure returning DistanceResult metadata.

    ``applies_ii_cap`` is True only for ``INTRINSIC_INFORMATION``; it
    gates the Eq. 23 cap ``φ_s = min(φ_c, φ_e, ii(s))``.

    ``partition_measure`` names the measure used to score partitions
    when this composite is the system measure. ``None`` means
    "use self"; ``INTRINSIC_INFORMATION`` sets it to GID so that
    partition integration is computed with GID and the cap is layered
    on top.
    """

    name: str
    applies_ii_cap: bool
    partition_measure: CompositeMeasure | None

    def __call__(
        self,
        forward: ArrayLike,
        partitioned: ArrayLike,
        selectivity: ArrayLike | None = None,
        *,
        state: object | None = None,
    ) -> DistanceResult: ...


@runtime_checkable
class StatefulDistributionMeasure(Protocol):
    """Two-distribution measure evaluated at a specified state.

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


def satisfies_distribution_measure(func: Callable[..., object]) -> bool:
    """Return True if ``func`` has the (p, q) shape of a DistributionMeasure."""
    return _required_params(func) == ["p", "q"]


def satisfies_state_aware_measure(func: Callable[..., object]) -> bool:
    """Return True if ``func`` has the (p, state) shape of a StateAwareMeasure."""
    return _required_params(func) == ["p", "state"]


def satisfies_composite_measure(func: Callable[..., object]) -> bool:
    """Return True if ``func`` has the (forward, partitioned, selectivity, ...)
    shape of a CompositeMeasure.

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


def satisfies_stateful_distribution_measure(func: Callable[..., object]) -> bool:
    """Return True if ``func`` has the (p, q, state) shape of a
    StatefulDistributionMeasure."""
    return _required_params(func) == ["p", "q", "state"]
