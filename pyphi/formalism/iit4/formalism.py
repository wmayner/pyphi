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

from . import (
    phi_structure as _phi_structure,  # pyright: ignore[reportUnknownVariableType]
)
from . import sia as _sia  # pyright: ignore[reportUnknownVariableType]


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
        """Delegate to ``Subsystem.find_mip``.

        The cut-over commit will move the IIT 4.0 branch of ``find_mip``
        directly into this method; for now the legacy dispatch path
        handles routing.
        """
        return subsystem.find_mip(direction, mechanism, purview, **kwargs)

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
    is applied via :class:`pyphi.conf.PyphiConfig.override` when this
    formalism's ``evaluate_*`` methods are called from a context that
    hasn't already pinned the metric — the cut-over commit will replace
    this with proper formalism-driven dispatch.
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

    def evaluate_system(self, subsystem: Any, **kwargs: Any) -> Any:
        with config.override(REPERTOIRE_DISTANCE=self.default_metric):
            return _sia(subsystem, **kwargs)

    def build_phi_structure(self, subsystem: Any, **kwargs: Any) -> Any:
        with config.override(REPERTOIRE_DISTANCE=self.default_metric):
            return _phi_structure(subsystem, **kwargs)
