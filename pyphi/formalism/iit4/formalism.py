# pyright: strict
"""Concrete IIT 4.0 formalism classes.

Two variants are registered:

- ``IIT_4_0_2023``: Albantakis et al. 2023. Default metric
  ``GENERALIZED_INTRINSIC_DIFFERENCE``.
- ``IIT_4_0_2026``: Mayner, Marshall, Tononi 2026. Default metric
  ``INTRINSIC_INFORMATION`` with the ``ii(s) = min(i_diff, i_spec)`` cap.

Both delegate to the algorithms in :mod:`pyphi.formalism.iit4` (the
``__init__`` module that holds ``sia``, ``phi_structure``,
``system_intrinsic_information``).
"""

from __future__ import annotations

from typing import Any
from typing import Literal

from pyphi.conf import config
from pyphi.parallel import MapReduce

from . import (
    phi_structure as _phi_structure,  # pyright: ignore[reportUnknownVariableType]
)
from . import sia as _sia  # pyright: ignore[reportUnknownVariableType]


def _find_mip_iit4(
    subsystem: Any,
    direction: Any,
    mechanism: Any,
    purview: Any,
    repertoire: Any,
    partitions: Any,
    state: Any,
    parallel_kwargs: Any,
    **kwargs: Any,  # noqa: ARG001
) -> Any:
    """IIT 4.0 mechanism MIP: maximize over candidate specified states,
    minimize over partitions per state.

    Extracted from the legacy ``Subsystem.find_mip`` IIT 4.0 branch
    (formerly ``subsystem.py:983-1004``). The active formalism is the
    owner of this dispatch; ``Subsystem`` provides candidate-system
    context and the per-state helper.
    """
    from pyphi import resolve_ties

    if state is None:
        specified_states = subsystem.intrinsic_information(
            direction, mechanism, purview
        ).ties
    else:
        specified_states = [state]

    mips = MapReduce(
        subsystem._find_mip_single_state,
        specified_states,
        map_kwargs={
            "direction": direction,
            "mechanism": mechanism,
            "purview": purview,
            "repertoire": repertoire,
            "partitions": partitions,
            "parallel_kwargs": parallel_kwargs,
        },
        desc="Finding MIP for maximum intrinsic information states",
        **parallel_kwargs,
    ).run()

    ties = tuple(resolve_ties.states(mips))  # type: ignore[arg-type]
    for tie in ties:
        tie.set_state_ties(ties)
    return ties[0]


class IIT4_2023Formalism:
    """IIT 4.0 (Albantakis et al. 2023) — GID-based mechanism integration."""

    name: str = "IIT_4_0_2023"
    exact: Literal[True] = True
    default_metric: str = "GENERALIZED_INTRINSIC_DIFFERENCE"
    compatible_metrics: frozenset[str] = frozenset(
        {"GENERALIZED_INTRINSIC_DIFFERENCE", "INTRINSIC_INFORMATION"}
    )
    partition_scheme: str | None = "ALL"

    def evaluate_mechanism(
        self,
        subsystem: Any,
        direction: Any,
        mechanism: Any,
        purview: Any,
        **kwargs: Any,
    ) -> Any:
        """Public mechanism-level evaluation. Calls back through
        ``Subsystem.find_mip`` to preserve the short-circuit logic
        (empty purview, unreachable state) the public method owns."""
        return subsystem.find_mip(direction, mechanism, purview, **kwargs)

    def _find_mechanism_mip(
        self,
        subsystem: Any,
        direction: Any,
        mechanism: Any,
        purview: Any,
        **kwargs: Any,
    ) -> Any:
        """Internal mechanism-MIP search. Called by ``Subsystem.find_mip``
        after its short-circuit checks; contains the IIT 4.0 logic
        (maximize over specified states, minimize over partitions)."""
        return _find_mip_iit4(subsystem, direction, mechanism, purview, **kwargs)

    def evaluate_system(self, subsystem: Any, **kwargs: Any) -> Any:
        """Delegate to :func:`pyphi.formalism.iit4.sia`."""
        return _sia(subsystem, **kwargs)

    def build_phi_structure(self, subsystem: Any, **kwargs: Any) -> Any:
        """Delegate to :func:`pyphi.formalism.iit4.phi_structure`."""
        return _phi_structure(subsystem, **kwargs)


class IIT4_2026Formalism:
    """IIT 4.0 (Mayner, Marshall, Tononi 2026) — intrinsic-information cap.

    Uses the ``INTRINSIC_INFORMATION`` metric with the ``ii(s) = min(i_diff,
    i_spec)`` cap from Eq. 23. Implementation reuses the IIT 4.0 (2023)
    algorithms; only the metric configuration differs. The metric override
    is applied via :class:`pyphi.conf.PyphiConfig.override` so legacy
    sites that still read ``config.REPERTOIRE_DISTANCE`` see the right
    metric. The metric-API unification (P5) replaces this indirection.
    """

    name: str = "IIT_4_0_2026"
    exact: Literal[True] = True
    default_metric: str = "INTRINSIC_INFORMATION"
    compatible_metrics: frozenset[str] = frozenset(
        {"INTRINSIC_INFORMATION", "GENERALIZED_INTRINSIC_DIFFERENCE"}
    )
    partition_scheme: str | None = "ALL"

    def evaluate_mechanism(
        self,
        subsystem: Any,
        direction: Any,
        mechanism: Any,
        purview: Any,
        **kwargs: Any,
    ) -> Any:
        with config.override(REPERTOIRE_DISTANCE=self.default_metric):
            return subsystem.find_mip(direction, mechanism, purview, **kwargs)

    def _find_mechanism_mip(
        self,
        subsystem: Any,
        direction: Any,
        mechanism: Any,
        purview: Any,
        **kwargs: Any,
    ) -> Any:
        with config.override(REPERTOIRE_DISTANCE=self.default_metric):
            return _find_mip_iit4(subsystem, direction, mechanism, purview, **kwargs)

    def evaluate_system(self, subsystem: Any, **kwargs: Any) -> Any:
        with config.override(REPERTOIRE_DISTANCE=self.default_metric):
            return _sia(subsystem, **kwargs)

    def build_phi_structure(self, subsystem: Any, **kwargs: Any) -> Any:
        with config.override(REPERTOIRE_DISTANCE=self.default_metric):
            return _phi_structure(subsystem, **kwargs)
