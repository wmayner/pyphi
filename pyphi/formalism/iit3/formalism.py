# pyright: strict
"""Concrete IIT 3.0 formalism class.

Delegates to the IIT 3.0 SIA algorithms in :mod:`pyphi.formalism.iit3`
(distribution-distance-based, bipartition-only).
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import ClassVar
from typing import Literal

from pyphi.conf import config
from pyphi.conf.formalism import FormalismConfig
from pyphi.formalism.base import check_metric_compatible


def _default_formalism_config() -> FormalismConfig:
    from pyphi.conf import config as _global

    return _global.formalism


@dataclass(frozen=True)
class IIT3Formalism:
    """IIT 3.0 (Oizumi et al. 2014) — distribution-distance phi computation."""

    name: ClassVar[str] = "IIT_3_0"
    exact: ClassVar[Literal[True]] = True
    compatible_metrics: ClassVar[frozenset[str]] = frozenset(
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
    partition_scheme: ClassVar[str | None] = "BI"

    config: FormalismConfig = field(default_factory=_default_formalism_config)

    def evaluate_mechanism(
        self,
        system: Any,
        direction: Any,
        mechanism: Any,
        purview: Any,
        **kwargs: Any,
    ) -> Any:
        """Public mechanism-level evaluation. Calls back through
        ``queries.find_mip`` to preserve the short-circuit logic
        (empty purview, unreachable state) the public dispatcher owns."""
        from pyphi.formalism.queries import find_mip

        return find_mip(system, direction, mechanism, purview, **kwargs)

    def _find_mechanism_mip(
        self,
        system: Any,
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

        check_metric_compatible(self, config.formalism.iit.mechanism_phi_measure)
        if state is not None:
            raise ValueError("passing `state` is not supported with IIT 3.0")
        return _find_mip_single_state(  # pyright: ignore[reportPrivateUsage]
            system,
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
        system: Any,
        direction: Any,
        mechanism: Any,
        purview: Any,
        partition: Any,
        repertoire: Any = None,
        partitioned_repertoire: Any = None,
        repertoire_distance: Any = None,
        partitioned_repertoire_kwargs: Any = None,
        **kwargs: Any,
    ) -> Any:
        """IIT 3.0 mechanism-partition integration: distribution-distance
        between unpartitioned and partitioned repertoires.

        ``repertoire_distance`` is a Protocol-typed metric callable
        (resolved here from config if not provided); ``mechanism_metric``
        is threaded through to the partitioned-repertoire helper.
        """
        from pyphi.metrics.distribution import (
            repertoire_distance as _repertoire_distance,  # pyright: ignore[reportUnknownVariableType]
        )
        from pyphi.metrics.distribution import resolve_mechanism_metric
        from pyphi.models import RepertoireIrreducibilityAnalysis
        from pyphi.utils import state_of

        check_metric_compatible(self, config.formalism.iit.mechanism_phi_measure)
        if repertoire_distance is None:
            repertoire_distance = resolve_mechanism_metric(
                config.formalism.iit.mechanism_phi_measure
            )
        # Internal helpers below the formalism boundary require an
        # explicit ``mechanism_metric``; resolve it here.
        mechanism_metric = kwargs.pop("mechanism_metric", repertoire_distance)
        if repertoire is None:
            repertoire = system.repertoire(direction, mechanism, purview)
        if partitioned_repertoire is None:
            partitioned_repertoire_kwargs = partitioned_repertoire_kwargs or {}
            partitioned_repertoire = system.partitioned_repertoire(
                direction,
                partition,
                mechanism_metric=mechanism_metric,
                **partitioned_repertoire_kwargs,
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
            mechanism_state=state_of(mechanism, system.state),
            purview_state=state_of(purview, system.state),
            specified_state=kwargs.get("state"),
            node_labels=system.node_labels,
            selectivity=None,
        )

    def evaluate_system(self, system: Any, **kwargs: Any) -> Any:
        """Delegate to the IIT 3.0 ``sia`` in :mod:`pyphi.formalism.iit3`.

        IIT 3.0 has no specified-state phase, so metric kwargs are not
        threaded through this method. The system-level metric is read
        from ``config.formalism.iit.mechanism_phi_measure`` inside the
        underlying ``sia`` implementation; compatibility is checked
        against the active formalism's ``compatible_metrics`` here.
        Callers attempting to pass ``system_metric`` /
        ``specification_metric`` receive a :class:`TypeError` rather
        than a silent no-op.
        """
        check_metric_compatible(self, config.formalism.iit.mechanism_phi_measure)
        from pyphi.formalism.iit3 import sia as _sia

        return _sia(system, **kwargs)

    def build_phi_structure(self, system: Any, **kwargs: Any) -> Any:
        """IIT 3.0 has no Φ-structure; raises ``NotImplementedError``.

        Use ``evaluate_system(system).ces`` to obtain the cause-effect
        structure.
        """
        del system, kwargs
        raise NotImplementedError(
            "IIT 3.0 has no Φ-structure (distinctions + relations); "
            "use evaluate_system().ces for the cause-effect structure."
        )
