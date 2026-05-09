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

from pyphi.conf import config
from pyphi.formalism.base import check_metric_compatible


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

    @property
    def config(self):
        """The active :class:`FormalismConfig` view over the global config.

        Future work (P11): replace this live view with a per-instance frozen
        snapshot so workers receive the formalism with its config attached
        rather than reading globals.
        """
        from pyphi.conf import config as _global

        return _global.formalism

    def evaluate_mechanism(
        self,
        subsystem: Any,
        direction: Any,
        mechanism: Any,
        purview: Any,
        **kwargs: Any,
    ) -> Any:
        """Public mechanism-level evaluation. Calls back through
        ``queries.find_mip`` to preserve the short-circuit logic
        (empty purview, unreachable state) the public dispatcher owns."""
        from pyphi.formalism.queries import find_mip

        return find_mip(subsystem, direction, mechanism, purview, **kwargs)

    def _find_mechanism_mip(
        self,
        subsystem: Any,
        direction: Any,
        mechanism: Any,
        purview: Any,
        repertoire: Any = None,
        partitions: Any = None,
        state: Any = None,
        parallel_kwargs: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Internal mechanism-MIP search. Called by ``queries.find_mip``
        after its short-circuit checks. IIT 3.0 has no candidate
        specified-states phase — there's a single, unique MIP per
        (mechanism, purview), found by minimizing over partitions.
        """
        from pyphi.formalism.queries import (
            _find_mip_single_state,  # pyright: ignore[reportPrivateUsage]
        )

        check_metric_compatible(self, config.formalism.repertoire_distance)
        if state is not None:
            raise ValueError("passing `state` is not supported with IIT 3.0")
        return _find_mip_single_state(  # pyright: ignore[reportPrivateUsage]
            subsystem,
            None,
            direction,
            mechanism,
            purview,
            repertoire,
            partitions,
            parallel_kwargs,
            **kwargs,
        )

    def evaluate_mechanism_partition(
        self,
        subsystem: Any,
        direction: Any,
        mechanism: Any,
        purview: Any,
        partition: Any,
        repertoire: Any = None,
        partitioned_repertoire: Any = None,
        repertoire_distance: str | None = None,
        partitioned_repertoire_kwargs: Any = None,
        **kwargs: Any,
    ) -> Any:
        """IIT 3.0 mechanism-partition integration: distribution-distance
        between unpartitioned and partitioned repertoires."""
        from pyphi.conf import fallback
        from pyphi.metrics.distribution import (
            repertoire_distance as _repertoire_distance,  # pyright: ignore[reportUnknownVariableType]
        )
        from pyphi.models import RepertoireIrreducibilityAnalysis
        from pyphi.utils import state_of

        check_metric_compatible(self, config.formalism.repertoire_distance)
        repertoire_distance = fallback(
            repertoire_distance, config.formalism.repertoire_distance
        )
        if repertoire is None:
            repertoire = subsystem.repertoire(direction, mechanism, purview)
        if partitioned_repertoire is None:
            partitioned_repertoire_kwargs = partitioned_repertoire_kwargs or {}
            partitioned_repertoire = subsystem.partitioned_repertoire(
                direction, partition, **partitioned_repertoire_kwargs
            )
        phi = _repertoire_distance(
            repertoire,
            partitioned_repertoire,
            direction=direction,
            repertoire_distance=repertoire_distance,
            **kwargs,
        )
        return RepertoireIrreducibilityAnalysis(
            phi=phi,
            direction=direction,
            mechanism=mechanism,
            purview=purview,
            partition=partition,
            repertoire=repertoire,
            partitioned_repertoire=partitioned_repertoire,
            mechanism_state=state_of(mechanism, subsystem.state),
            purview_state=state_of(purview, subsystem.state),
            specified_state=kwargs.get("state"),
            node_labels=subsystem.node_labels,
            selectivity=None,
        )

    def evaluate_system(self, subsystem: Any, **kwargs: Any) -> Any:
        """Delegate to :func:`pyphi.compute.subsystem.sia` (the IIT 3.0 path)."""
        check_metric_compatible(self, config.formalism.repertoire_distance)
        from pyphi import compute

        return compute.subsystem.sia(subsystem, **kwargs)

    def build_phi_structure(self, subsystem: Any, **kwargs: Any) -> Any:
        """IIT 3.0 has no Φ-structure; raises ``NotImplementedError``.

        Use ``evaluate_system(subsystem).ces`` to obtain the cause-effect
        structure.
        """
        del subsystem, kwargs
        raise NotImplementedError(
            "IIT 3.0 has no Φ-structure (distinctions + relations); "
            "use evaluate_system().ces for the cause-effect structure."
        )
