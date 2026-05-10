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

from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import ClassVar
from typing import Literal

from pyphi.conf import config
from pyphi.conf.formalism import FormalismConfig
from pyphi.formalism.base import check_metric_compatible
from pyphi.parallel import MapReduce

from . import (
    phi_structure as _phi_structure,  # pyright: ignore[reportUnknownVariableType]
)
from . import sia as _sia  # pyright: ignore[reportUnknownVariableType]


def _default_formalism_config() -> FormalismConfig:
    from pyphi.conf import config as _global

    return _global.formalism


def _evaluate_partition_iit4(
    system: Any,
    direction: Any,
    mechanism: Any,
    purview: Any,
    partition: Any,
    repertoire: Any = None,
    partitioned_repertoire: Any = None,
    **kwargs: Any,
) -> Any:
    """IIT 4.0 mechanism-partition integration.

    State-aware: takes a forward repertoire, a partitioned forward repertoire,
    and a scalar selectivity, then calls a GID-style metric. Replaces the
    legacy ``System.evaluate_partition`` IIT 4.0 branch.

    Compatibility is the caller's responsibility; ``IIT4_2023Formalism``
    and ``IIT4_2026Formalism`` validate before calling this helper.
    """
    from pyphi import metrics
    from pyphi.conf import config
    from pyphi.conf import fallback
    from pyphi.models import RepertoireIrreducibilityAnalysis
    from pyphi.utils import state_of

    repertoire_distance = fallback(
        kwargs.pop("repertoire_distance", None), config.formalism.iit.repertoire_measure
    )
    # Mechanism-level partition evaluation uses GID; INTRINSIC_INFORMATION
    # is a system-level composite (Eq. 23) that reduces to GID at this level.
    if repertoire_distance == "INTRINSIC_INFORMATION":
        repertoire_distance = "GENERALIZED_INTRINSIC_DIFFERENCE"

    if repertoire is None:
        repertoire = system.repertoire(direction, mechanism, purview)

    func = metrics.distribution.measures[repertoire_distance]
    assert not isinstance(repertoire, (int, float)), "GID requires full repertoire"

    purview_state = kwargs["state"].state
    selectivity = float(repertoire.squeeze()[purview_state])
    forward_pr = system.forward_probability(direction, mechanism, purview, purview_state)
    if partitioned_repertoire is None:
        partitioned_pr = system.partitioned_repertoire(
            direction, partition, state=purview_state
        )
    else:
        partitioned_pr = partitioned_repertoire

    phi: Any = func(  # pyright: ignore[reportUnknownVariableType]
        forward_repertoire=forward_pr,
        partitioned_forward_repertoire=partitioned_pr,
        selectivity_repertoire=selectivity,
    )

    return RepertoireIrreducibilityAnalysis(
        phi=phi,  # pyright: ignore[reportUnknownArgumentType]
        direction=direction,
        mechanism=mechanism,
        purview=purview,
        partition=partition,
        repertoire=forward_pr,
        partitioned_repertoire=partitioned_pr,
        mechanism_state=state_of(mechanism, system.state),
        purview_state=state_of(purview, system.state),
        specified_state=kwargs.get("state"),
        node_labels=system.node_labels,
        selectivity=selectivity,
    )


def _find_mip_iit4(
    system: Any,
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

    Extracted from the legacy ``System.find_mip`` IIT 4.0 branch
    (formerly ``system.py:983-1004``). The active formalism is the
    owner of this dispatch; ``System`` provides candidate-system
    context and the per-state helper.
    """
    from pyphi import resolve_ties

    if state is None:
        specified_states = system.intrinsic_information(
            direction, mechanism, purview
        ).ties
    else:
        specified_states = [state]

    from functools import partial

    from pyphi.formalism.queries import (
        _find_mip_single_state,  # pyright: ignore[reportPrivateUsage]
    )

    mips = MapReduce(
        partial(_find_mip_single_state, system),  # pyright: ignore[reportPrivateUsage]
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


@dataclass(frozen=True)
class IIT4_2023Formalism:
    """IIT 4.0 (Albantakis et al. 2023) — GID-based mechanism integration."""

    name: ClassVar[str] = "IIT_4_0_2023"
    exact: ClassVar[Literal[True]] = True
    default_metric: ClassVar[str] = "GENERALIZED_INTRINSIC_DIFFERENCE"
    compatible_metrics: ClassVar[frozenset[str]] = frozenset(
        {"GENERALIZED_INTRINSIC_DIFFERENCE", "INTRINSIC_INFORMATION"}
    )
    partition_scheme: ClassVar[str | None] = "ALL"

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
        **kwargs: Any,
    ) -> Any:
        """Internal mechanism-MIP search. Called by ``queries.find_mip``
        after its short-circuit checks; contains the IIT 4.0 logic
        (maximize over specified states, minimize over partitions)."""
        check_metric_compatible(self, config.formalism.iit.repertoire_measure)
        return _find_mip_iit4(system, direction, mechanism, purview, **kwargs)

    def evaluate_mechanism_partition(
        self,
        system: Any,
        direction: Any,
        mechanism: Any,
        purview: Any,
        partition: Any,
        **kwargs: Any,
    ) -> Any:
        """IIT 4.0 mechanism-partition integration: forward repertoires +
        scalar selectivity feed a GID-style metric."""
        check_metric_compatible(self, config.formalism.iit.repertoire_measure)
        return _evaluate_partition_iit4(
            system, direction, mechanism, purview, partition, **kwargs
        )

    def evaluate_system(self, system: Any, **kwargs: Any) -> Any:
        """Delegate to :func:`pyphi.formalism.iit4.sia`."""
        check_metric_compatible(self, config.formalism.iit.repertoire_measure)
        return _sia(system, **kwargs)

    def build_phi_structure(self, system: Any, **kwargs: Any) -> Any:
        """Delegate to :func:`pyphi.formalism.iit4.phi_structure`."""
        check_metric_compatible(self, config.formalism.iit.repertoire_measure)
        return _phi_structure(system, **kwargs)


@dataclass(frozen=True)
class IIT4_2026Formalism:
    """IIT 4.0 (Mayner, Marshall, Tononi 2026) — intrinsic-information cap.

    Uses the ``INTRINSIC_INFORMATION`` metric with the ``ii(s) = min(i_diff,
    i_spec)`` cap from Eq. 23. Implementation reuses the IIT 4.0 (2023)
    algorithms; only the metric configuration differs. The metric override
    is applied via :class:`pyphi.conf.PyphiConfig.override` so legacy
    sites that still read ``config.formalism.iit.repertoire_measure`` see the right
    metric.
    """

    name: ClassVar[str] = "IIT_4_0_2026"
    exact: ClassVar[Literal[True]] = True
    default_metric: ClassVar[str] = "INTRINSIC_INFORMATION"
    compatible_metrics: ClassVar[frozenset[str]] = frozenset(
        {"INTRINSIC_INFORMATION", "GENERALIZED_INTRINSIC_DIFFERENCE"}
    )
    partition_scheme: ClassVar[str | None] = "ALL"

    config: FormalismConfig = field(default_factory=_default_formalism_config)

    def evaluate_mechanism(
        self,
        system: Any,
        direction: Any,
        mechanism: Any,
        purview: Any,
        **kwargs: Any,
    ) -> Any:
        from pyphi.formalism.queries import find_mip

        with config.override(repertoire_measure=self.default_metric):
            return find_mip(system, direction, mechanism, purview, **kwargs)

    def _find_mechanism_mip(
        self,
        system: Any,
        direction: Any,
        mechanism: Any,
        purview: Any,
        **kwargs: Any,
    ) -> Any:
        check_metric_compatible(self, config.formalism.iit.repertoire_measure)
        with config.override(repertoire_measure=self.default_metric):
            return _find_mip_iit4(system, direction, mechanism, purview, **kwargs)

    def evaluate_mechanism_partition(
        self,
        system: Any,
        direction: Any,
        mechanism: Any,
        purview: Any,
        partition: Any,
        **kwargs: Any,
    ) -> Any:
        """Same shape as IIT 4.0 (2023) mechanism-partition integration; the
        2026 variant differs only at the system level (the ``ii(s)`` cap)."""
        check_metric_compatible(self, config.formalism.iit.repertoire_measure)
        with config.override(repertoire_measure=self.default_metric):
            return _evaluate_partition_iit4(
                system, direction, mechanism, purview, partition, **kwargs
            )

    def evaluate_system(self, system: Any, **kwargs: Any) -> Any:
        check_metric_compatible(self, config.formalism.iit.repertoire_measure)
        with config.override(repertoire_measure=self.default_metric):
            return _sia(system, **kwargs)

    def build_phi_structure(self, system: Any, **kwargs: Any) -> Any:
        check_metric_compatible(self, config.formalism.iit.repertoire_measure)
        with config.override(repertoire_measure=self.default_metric):
            return _phi_structure(system, **kwargs)
