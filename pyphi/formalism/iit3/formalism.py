# pyright: strict
"""Concrete IIT 3.0 formalism class.

Delegates to the legacy :mod:`pyphi.compute.subsystem` SIA path
(distribution-distance-based, bipartition-only). The implementation files
have not moved out of ``compute/subsystem.py`` in this commit; the
formalism class is the new owner of the dispatch boundary, and a future
cleanup can relocate the implementation if desired.
"""

from __future__ import annotations

from typing import Any
from typing import Literal


class IIT3Formalism:
    """IIT 3.0 (Oizumi et al. 2014) — distribution-distance phi computation."""

    name: str = "IIT_3_0"
    exact: Literal[True] = True
    default_metric: str = "EMD"
    compatible_metrics: frozenset[str] = frozenset(
        {
            "EMD",
            "L1",
            "KLD",
            "ENTROPY_DIFFERENCE",
            "PSQ2",
            "MP2Q",
            "ABSOLUTE_INTRINSIC_DIFFERENCE",
            "INTRINSIC_DIFFERENCE",
        }
    )
    partition_scheme: str | None = "BI"

    def evaluate_mechanism(
        self,
        subsystem: Any,
        direction: Any,
        mechanism: Any,
        purview: Any,
        **kwargs: Any,
    ) -> Any:
        """Delegate to ``Subsystem.find_mip``."""
        return subsystem.find_mip(direction, mechanism, purview, **kwargs)

    def evaluate_system(self, subsystem: Any, **kwargs: Any) -> Any:
        """Delegate to :func:`pyphi.compute.subsystem.sia` (the IIT 3.0 path).

        Note: ``Subsystem.sia()`` is currently hardcoded to the IIT 4.0
        path regardless of ``IIT_VERSION``; the cut-over commit fixes this
        so the formalism owns the dispatch.
        """
        from pyphi import compute

        return compute.subsystem.sia(subsystem, **kwargs)

    def build_phi_structure(self, subsystem: Any, **kwargs: Any) -> Any:
        """IIT 3.0 has no Φ-structure; raises ``NotImplementedError``.

        Consumers should branch on ``formalism.name`` (or check the name
        against ``"IIT_3_0"``) before calling this method, or use
        ``evaluate_system`` to get the IIT 3.0 SIA which carries the
        cause-effect structure as ``sia.ces``.
        """
        del subsystem, kwargs
        raise NotImplementedError(
            "IIT 3.0 has no Φ-structure (distinctions + relations); "
            "use evaluate_system().ces for the cause-effect structure."
        )
