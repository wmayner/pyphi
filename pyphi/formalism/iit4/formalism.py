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
from typing import cast

from pyphi.conf import config
from pyphi.conf.formalism import FormalismConfig
from pyphi.formalism.base import check_metric_compatible
from pyphi.metrics.protocols import CompositeMetric
from pyphi.metrics.protocols import DistributionMetric
from pyphi.metrics.protocols import StateAwareMetric
from pyphi.metrics.protocols import StatefulDistributionMetric
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
    *,
    mechanism_metric: Any,
    repertoire: Any = None,
    partitioned_repertoire: Any = None,
    **kwargs: Any,
) -> Any:
    """IIT 4.0 mechanism-partition integration.

    State-aware: takes a forward repertoire, a partitioned forward repertoire,
    and a scalar selectivity, then calls a GID-style metric.

    ``mechanism_metric`` is a Protocol-typed mechanism-level metric callable
    (StateAwareMetric, StatefulDistributionMetric, or CompositeMetric)
    resolved at the formalism-class boundary; compatibility is the
    caller's responsibility (``IIT4_2023Formalism`` and
    ``IIT4_2026Formalism`` validate before calling this helper).
    """
    from pyphi.models import RepertoireIrreducibilityAnalysis
    from pyphi.utils import state_of

    if repertoire is None:
        repertoire = system.repertoire(direction, mechanism, purview)

    assert not isinstance(repertoire, (int, float)), "GID requires full repertoire"

    purview_state = kwargs["state"].state
    selectivity = float(repertoire.squeeze()[purview_state])
    forward_pr = system.forward_probability(direction, mechanism, purview, purview_state)
    if partitioned_repertoire is None:
        partitioned_pr = system.partitioned_repertoire(
            direction,
            partition,
            mechanism_metric=mechanism_metric,
            state=purview_state,
        )
    else:
        partitioned_pr = partitioned_repertoire

    phi: Any = mechanism_metric(  # pyright: ignore[reportUnknownVariableType]
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
    *,
    mechanism_metric: Any,
    specification_metric: Any,
    repertoire: Any,
    partitions: Any,
    state: Any,
    parallel_kwargs: Any,
    **kwargs: Any,  # noqa: ARG001
) -> Any:
    """IIT 4.0 mechanism MIP: maximize over candidate specified states,
    minimize over partitions per state.

    ``mechanism_metric`` and ``specification_metric`` are Protocol-typed
    metric callables provided by the active formalism.
    ``specification_metric`` drives the intrinsic-information search
    over candidate purview states; ``mechanism_metric`` drives the
    per-partition phi.
    """
    from pyphi import resolve_ties

    if state is None:
        specified_states = system.intrinsic_information(
            direction,
            mechanism,
            purview,
            specification_metric=specification_metric,
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
            "mechanism_metric": mechanism_metric,
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
    """IIT 4.0 (Albantakis et al. 2023) — GID-based integration at all scopes."""

    name: ClassVar[str] = "IIT_4_0_2023"
    exact: ClassVar[Literal[True]] = True
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
        *,
        mechanism_metric: CompositeMetric
        | StateAwareMetric
        | StatefulDistributionMetric
        | None = None,
        specification_metric: CompositeMetric
        | StateAwareMetric
        | StatefulDistributionMetric
        | DistributionMetric
        | None = None,
        **kwargs: Any,
    ) -> Any:
        """Internal mechanism-MIP search. Called by ``queries.find_mip``
        after its short-circuit checks; contains the IIT 4.0 logic
        (maximize over specified states, minimize over partitions).

        Explicit ``mechanism_metric``/``specification_metric`` override the
        config-driven fallback; when omitted, they resolve from
        ``config.formalism.iit.mechanism_phi_measure`` /
        ``specification_measure`` respectively.
        """
        from pyphi.metrics.distribution import resolve_mechanism_metric

        if mechanism_metric is None:
            check_metric_compatible(self, config.formalism.iit.mechanism_phi_measure)
            # ``check_metric_compatible`` above guarantees the configured
            # name is in this formalism's ``compatible_metrics`` (GID /
            # INTRINSIC_INFORMATION), neither of which is a
            # ``DistributionMetric`` — narrow the broader resolver
            # return type to match the declared kwarg union.
            mechanism_metric = cast(
                "CompositeMetric | StateAwareMetric | StatefulDistributionMetric",
                resolve_mechanism_metric(config.formalism.iit.mechanism_phi_measure),
            )
        if specification_metric is None:
            specification_metric = resolve_mechanism_metric(
                config.formalism.iit.specification_measure
            )
        return _find_mip_iit4(
            system,
            direction,
            mechanism,
            purview,
            mechanism_metric=mechanism_metric,
            specification_metric=specification_metric,
            **kwargs,
        )

    def evaluate_mechanism_partition(
        self,
        system: Any,
        direction: Any,
        mechanism: Any,
        purview: Any,
        partition: Any,
        *,
        mechanism_metric: CompositeMetric
        | StateAwareMetric
        | StatefulDistributionMetric
        | None = None,
        **kwargs: Any,
    ) -> Any:
        """IIT 4.0 mechanism-partition integration: forward repertoires +
        scalar selectivity feed a GID-style metric.

        Explicit ``mechanism_metric`` overrides the config-driven fallback.
        """
        from pyphi.metrics.distribution import resolve_mechanism_metric

        if mechanism_metric is None:
            check_metric_compatible(self, config.formalism.iit.mechanism_phi_measure)
            # ``check_metric_compatible`` above guarantees the configured
            # name is in this formalism's ``compatible_metrics`` (GID /
            # INTRINSIC_INFORMATION), neither of which is a
            # ``DistributionMetric`` — narrow the broader resolver
            # return type to match the declared kwarg union.
            mechanism_metric = cast(
                "CompositeMetric | StateAwareMetric | StatefulDistributionMetric",
                resolve_mechanism_metric(config.formalism.iit.mechanism_phi_measure),
            )
        return _evaluate_partition_iit4(
            system,
            direction,
            mechanism,
            purview,
            partition,
            mechanism_metric=mechanism_metric,
            **kwargs,
        )

    def evaluate_system(
        self,
        system: Any,
        *,
        system_metric: CompositeMetric | None = None,
        specification_metric: CompositeMetric
        | StateAwareMetric
        | StatefulDistributionMetric
        | DistributionMetric
        | None = None,
        **kwargs: Any,
    ) -> Any:
        """Delegate to :func:`pyphi.formalism.iit4.sia`.

        Explicit ``system_metric``/``specification_metric`` override the
        config-driven fallback. When omitted, they resolve from
        ``config.formalism.iit.system_phi_measure`` /
        ``specification_measure`` — user config is authoritative.
        """
        from pyphi.metrics.distribution import resolve_mechanism_metric
        from pyphi.metrics.distribution import resolve_system_metric

        if system_metric is None:
            check_metric_compatible(self, config.formalism.iit.system_phi_measure)
            system_metric = resolve_system_metric(
                config.formalism.iit.system_phi_measure
            )
        else:
            check_metric_compatible(self, system_metric.name)
        if specification_metric is None:
            specification_metric = resolve_mechanism_metric(
                config.formalism.iit.specification_measure
            )
        return _sia(
            system,
            system_metric=system_metric,
            specification_metric=specification_metric,
            **kwargs,
        )

    def build_phi_structure(
        self,
        system: Any,
        *,
        system_metric: CompositeMetric | None = None,
        specification_metric: CompositeMetric
        | StateAwareMetric
        | StatefulDistributionMetric
        | DistributionMetric
        | None = None,
        **kwargs: Any,
    ) -> Any:
        """Delegate to :func:`pyphi.formalism.iit4.phi_structure`.

        Explicit ``system_metric``/``specification_metric`` override the
        config-driven fallback.
        """
        from pyphi.metrics.distribution import resolve_mechanism_metric
        from pyphi.metrics.distribution import resolve_system_metric

        if system_metric is None:
            check_metric_compatible(self, config.formalism.iit.system_phi_measure)
            system_metric = resolve_system_metric(
                config.formalism.iit.system_phi_measure
            )
        else:
            check_metric_compatible(self, system_metric.name)
        if specification_metric is None:
            specification_metric = resolve_mechanism_metric(
                config.formalism.iit.specification_measure
            )
        return _phi_structure(
            system,
            system_metric=system_metric,
            specification_metric=specification_metric,
            **kwargs,
        )


@dataclass(frozen=True)
class IIT4_2026Formalism:
    """IIT 4.0 (Mayner, Marshall, Tononi 2026) — intrinsic-information cap.

    Mechanism phi uses GID per Eqs. 19-20 (same as IIT 4.0 2023). System
    phi uses ``INTRINSIC_INFORMATION`` with the ``ii(s) = min(i_diff,
    i_spec)`` cap from Eq. 23 — that's the 2026-specific divergence.
    Scope-explicit overrides ensure each level uses the right metric.
    """

    name: ClassVar[str] = "IIT_4_0_2026"
    exact: ClassVar[Literal[True]] = True
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

        return find_mip(system, direction, mechanism, purview, **kwargs)

    def _find_mechanism_mip(
        self,
        system: Any,
        direction: Any,
        mechanism: Any,
        purview: Any,
        *,
        mechanism_metric: CompositeMetric
        | StateAwareMetric
        | StatefulDistributionMetric
        | None = None,
        specification_metric: CompositeMetric
        | StateAwareMetric
        | StatefulDistributionMetric
        | DistributionMetric
        | None = None,
        **kwargs: Any,
    ) -> Any:
        """Mechanism-level MIP search.

        Explicit ``mechanism_metric``/``specification_metric`` override the
        config-driven fallback.
        """
        from pyphi.metrics.distribution import resolve_mechanism_metric

        if mechanism_metric is None:
            check_metric_compatible(self, config.formalism.iit.mechanism_phi_measure)
            # ``check_metric_compatible`` above guarantees the configured
            # name is in this formalism's ``compatible_metrics`` (GID /
            # INTRINSIC_INFORMATION), neither of which is a
            # ``DistributionMetric`` — narrow the broader resolver
            # return type to match the declared kwarg union.
            mechanism_metric = cast(
                "CompositeMetric | StateAwareMetric | StatefulDistributionMetric",
                resolve_mechanism_metric(config.formalism.iit.mechanism_phi_measure),
            )
        if specification_metric is None:
            specification_metric = resolve_mechanism_metric(
                config.formalism.iit.specification_measure
            )
        return _find_mip_iit4(
            system,
            direction,
            mechanism,
            purview,
            mechanism_metric=mechanism_metric,
            specification_metric=specification_metric,
            **kwargs,
        )

    def evaluate_mechanism_partition(
        self,
        system: Any,
        direction: Any,
        mechanism: Any,
        purview: Any,
        partition: Any,
        *,
        mechanism_metric: CompositeMetric
        | StateAwareMetric
        | StatefulDistributionMetric
        | None = None,
        **kwargs: Any,
    ) -> Any:
        """Same shape as IIT 4.0 (2023) mechanism-partition integration; the
        2026 variant differs only at the system level (the ``ii(s)`` cap).

        Explicit ``mechanism_metric`` overrides the config-driven fallback.
        """
        from pyphi.metrics.distribution import resolve_mechanism_metric

        if mechanism_metric is None:
            check_metric_compatible(self, config.formalism.iit.mechanism_phi_measure)
            # ``check_metric_compatible`` above guarantees the configured
            # name is in this formalism's ``compatible_metrics`` (GID /
            # INTRINSIC_INFORMATION), neither of which is a
            # ``DistributionMetric`` — narrow the broader resolver
            # return type to match the declared kwarg union.
            mechanism_metric = cast(
                "CompositeMetric | StateAwareMetric | StatefulDistributionMetric",
                resolve_mechanism_metric(config.formalism.iit.mechanism_phi_measure),
            )
        return _evaluate_partition_iit4(
            system,
            direction,
            mechanism,
            purview,
            partition,
            mechanism_metric=mechanism_metric,
            **kwargs,
        )

    def evaluate_system(
        self,
        system: Any,
        *,
        system_metric: CompositeMetric | None = None,
        specification_metric: CompositeMetric
        | StateAwareMetric
        | StatefulDistributionMetric
        | DistributionMetric
        | None = None,
        **kwargs: Any,
    ) -> Any:
        """Delegate to :func:`pyphi.formalism.iit4.sia`.

        Explicit ``system_metric``/``specification_metric`` override the
        config-driven fallback. When omitted, ``system_metric`` resolves
        from ``config.formalism.iit.system_phi_measure`` (user config is
        authoritative). The ``ii(s)`` cap (Eq. 23) fires when the
        resolved metric's ``name`` is ``"INTRINSIC_INFORMATION"``.
        """
        from pyphi.metrics.distribution import resolve_mechanism_metric
        from pyphi.metrics.distribution import resolve_system_metric

        if system_metric is None:
            check_metric_compatible(self, config.formalism.iit.system_phi_measure)
            system_metric = resolve_system_metric(
                config.formalism.iit.system_phi_measure
            )
        else:
            check_metric_compatible(self, system_metric.name)
        if specification_metric is None:
            specification_metric = resolve_mechanism_metric(
                config.formalism.iit.specification_measure
            )
        return _sia(
            system,
            system_metric=system_metric,
            specification_metric=specification_metric,
            **kwargs,
        )

    def build_phi_structure(
        self,
        system: Any,
        *,
        system_metric: CompositeMetric | None = None,
        specification_metric: CompositeMetric
        | StateAwareMetric
        | StatefulDistributionMetric
        | DistributionMetric
        | None = None,
        **kwargs: Any,
    ) -> Any:
        """Delegate to :func:`pyphi.formalism.iit4.phi_structure`.

        Explicit ``system_metric``/``specification_metric`` override the
        config-driven fallback.
        """
        from pyphi.metrics.distribution import resolve_mechanism_metric
        from pyphi.metrics.distribution import resolve_system_metric

        if system_metric is None:
            check_metric_compatible(self, config.formalism.iit.system_phi_measure)
            system_metric = resolve_system_metric(
                config.formalism.iit.system_phi_measure
            )
        else:
            check_metric_compatible(self, system_metric.name)
        if specification_metric is None:
            specification_metric = resolve_mechanism_metric(
                config.formalism.iit.specification_measure
            )
        return _phi_structure(
            system,
            system_metric=system_metric,
            specification_metric=specification_metric,
            **kwargs,
        )
