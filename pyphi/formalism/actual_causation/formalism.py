"""The AC_2019 actual-causation formalism object."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import ClassVar

from pyphi.conf import config
from pyphi.conf.formalism import FormalismConfig
from pyphi.direction import Direction
from pyphi.formalism.base import check_measure_compatible

from . import compute


def _default_formalism_config() -> FormalismConfig:
    from pyphi.conf import config as _global

    return _global.formalism


def _resolve_ac_measures(
    formalism: Any,
    *,
    alpha_measure_name: str | None = None,
    partitioned_repertoire_scheme_name: str | None = None,
    background_scheme_name: str | None = None,
    alpha_aggregation_name: str | None = None,
) -> dict[str, Any]:
    """Resolve AC measure/scheme config into callables, checking compatibility.

    Mirrors IIT's ``_resolve_system_measures``: an explicitly-passed name
    overrides config; otherwise the live ``config.formalism.actual_causation``
    is read so runtime overrides take effect. The chosen ``alpha_measure``
    name is checked against ``formalism.compatible_measures`` before
    resolution; the three schemes resolve from their registries by name.
    """
    from pyphi.measures.distribution import resolve_actual_causation_measure

    ac = config.formalism.actual_causation
    alpha_name = (
        alpha_measure_name if alpha_measure_name is not None else ac.alpha_measure
    )
    check_measure_compatible(formalism, alpha_name)

    pr_name = (
        partitioned_repertoire_scheme_name
        if partitioned_repertoire_scheme_name is not None
        else ac.partitioned_repertoire_scheme
    )
    bg_name = (
        background_scheme_name
        if background_scheme_name is not None
        else ac.background_scheme
    )
    agg_name = (
        alpha_aggregation_name
        if alpha_aggregation_name is not None
        else ac.alpha_aggregation
    )

    return {
        "alpha_measure": resolve_actual_causation_measure(alpha_name),
        "partitioned_repertoire_scheme": compute.partitioned_repertoire_schemes[pr_name],
        "background_scheme": compute.background_strategies[bg_name],
        "alpha_aggregation": compute.alpha_aggregations[agg_name],
    }


@dataclass(frozen=True)
class AC2019Formalism:
    """Actual Causation formalism (Albantakis et al. 2019, "What Caused What?").

    The registered AC analog of the IIT formalism objects. Selected when
    ``config.formalism.actual_causation.version == "AC_2019"``. Each method
    resolves the configured ``alpha_measure`` (checked against
    :attr:`compatible_measures`) and AC schemes via :func:`_resolve_ac_measures`,
    then delegates to the algorithms in :mod:`.compute`.
    """

    name: ClassVar[str] = "AC_2019"
    compatible_measures: ClassVar[frozenset[str]] = frozenset({"PMI", "WPMI"})

    config: FormalismConfig = field(default_factory=_default_formalism_config)

    def evaluate_account(
        self, transition: Any, direction: Any = Direction.BIDIRECTIONAL, **kwargs: Any
    ) -> Any:
        """Compute the account (bidirectional) or directed account of a transition.

        A bidirectional call with no mechanism/purview restriction returns a
        full :class:`~pyphi.models.Account`; any other call returns a
        :class:`~pyphi.models.DirectedAccount`, threading ``mechanisms`` /
        ``purviews`` / ``allow_neg`` from ``kwargs``.
        """
        mechanisms = kwargs.get("mechanisms")
        purviews = kwargs.get("purviews")
        allow_neg = kwargs.get("allow_neg", False)
        resolved = _resolve_ac_measures(self)
        alpha_measure = resolved["alpha_measure"]
        scheme = resolved["partitioned_repertoire_scheme"]
        if (
            direction == Direction.BIDIRECTIONAL
            and mechanisms is None
            and purviews is None
        ):
            return compute._account(
                transition,
                direction,
                alpha_measure=alpha_measure,
                partitioned_repertoire_scheme=scheme,
            )
        return compute._directed_account(
            transition,
            direction,
            mechanisms,
            purviews,
            allow_neg,
            alpha_measure=alpha_measure,
            partitioned_repertoire_scheme=scheme,
        )

    def evaluate_system(
        self, transition: Any, direction: Any = Direction.BIDIRECTIONAL, **kwargs: Any
    ) -> Any:
        """Compute the system irreducibility analysis (big-alpha) of a transition."""
        resolved = _resolve_ac_measures(self)
        return compute._sia(
            transition,
            direction,
            alpha_measure=resolved["alpha_measure"],
            partitioned_repertoire_scheme=resolved["partitioned_repertoire_scheme"],
            **kwargs,
        )

    def evaluate_mechanism(
        self,
        transition: Any,
        direction: Any,
        mechanism: Any,
        purview: Any,
        **kwargs: Any,
    ) -> Any:
        """Compute the mechanism MIP over a purview (today's ``find_mip``)."""
        resolved = _resolve_ac_measures(self)
        return compute._find_mip(
            transition,
            direction,
            mechanism,
            purview,
            kwargs.get("allow_neg", False),
            alpha_measure=resolved["alpha_measure"],
            partitioned_repertoire_scheme=resolved["partitioned_repertoire_scheme"],
        )

    def evaluate_causal_link(
        self, transition: Any, direction: Any, mechanism: Any, **kwargs: Any
    ) -> Any:
        """Compute the maximally-irreducible causal link (``find_causal_link``)."""
        resolved = _resolve_ac_measures(self)
        return compute._find_causal_link(
            transition,
            direction,
            mechanism,
            kwargs.get("purviews"),
            kwargs.get("allow_neg", False),
            alpha_measure=resolved["alpha_measure"],
            partitioned_repertoire_scheme=resolved["partitioned_repertoire_scheme"],
        )
