# pyright: strict
"""Concrete IIT 3.0 formalism class.

Delegates to the IIT 3.0 SIA algorithms in :mod:`pyphi.formalism.iit3`
(distribution-distance-based, bipartition-only).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import ClassVar
from typing import Literal

from pyphi.conf import config
from pyphi.formalism.base import check_measure_compatible
from pyphi.formalism.base import check_sia_tie_strategy_compatible


@dataclass(frozen=True)
class IIT3Formalism:
    """IIT 3.0 (Oizumi et al. 2014) — distribution-distance phi computation."""

    name: ClassVar[str] = "IIT_3_0"
    exact: ClassVar[Literal[True]] = True
    uses_system_phi_measure: ClassVar[bool] = False
    compatible_measures: ClassVar[frozenset[str]] = frozenset(
        {
            "EMD",
            "L1",
            "KLD",
            "ENTROPY_DIFFERENCE",
            "PSQ2",
            "MP2Q",
            "AID",
            "ID",
        }
    )
    # IIT 3.0 derives system Φ from the CES distance, so ces_measure defines
    # its answer. EMD is the 2014 paper's measure; SUM_SMALL_PHI is the
    # small-phi-difference variant used by Gómez et al. (2020) for
    # multi-valued elements (where EMD is unavailable).
    compatible_ces_measures: ClassVar[frozenset[str]] = frozenset(
        {"EMD", "SUM_SMALL_PHI"}
    )
    # IIT 3.0 MIP searches are plain minima with no specified-state pins.
    has_state_pins: ClassVar[bool] = False
    partition_scheme: ClassVar[str | None] = "JOINT_BIPARTITION"
    compatible_system_partition_schemes: ClassVar[frozenset[str] | None] = frozenset(
        {"DIRECTED_BIPARTITION", "DIRECTED_BIPARTITION_CUT_ONE"}
    )
    compatible_mechanism_partition_schemes: ClassVar[frozenset[str] | None] = frozenset(
        {"JOINT_BIPARTITION", "WEDGE_TRIPARTITION"}
    )
    # IIT 3.0 SIA results carry only raw phi and the MIP -- no normalized phi
    # or purview -- so SIA-level tie strategies reading those attributes have
    # nothing to read.
    compatible_sia_tie_strategies: ClassVar[frozenset[str] | None] = frozenset(
        {"PHI", "NEGATIVE_PHI", "PARTITION_LEX", "NONE"}
    )

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
        after its short-circuit checks.

        Unlike IIT 4.0, IIT 3.0 has no candidate specified-state phase: a
        single search over the mechanism partitions minimizes φ for the fixed
        (mechanism, purview) pair. Passing ``state`` therefore raises
        :class:`ValueError`. Partitions that tie on minimum φ are resolved by
        :mod:`pyphi.resolve_ties`.
        """
        from pyphi.formalism.queries import (
            _find_mip_single_state,  # pyright: ignore[reportPrivateUsage]
        )

        check_measure_compatible(self, config.formalism.iit.mechanism_phi_measure)
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

        ``repertoire_distance`` is a Protocol-typed measure callable
        (resolved here from config if not provided); ``mechanism_measure``
        is threaded through to the partitioned-repertoire helper.
        """
        from pyphi.measures.distribution import (
            repertoire_distance as _repertoire_distance,  # pyright: ignore[reportUnknownVariableType]
        )
        from pyphi.measures.distribution import resolve_mechanism_measure
        from pyphi.models import RepertoireIrreducibilityAnalysis
        from pyphi.utils import state_of

        check_measure_compatible(self, config.formalism.iit.mechanism_phi_measure)
        if repertoire_distance is None:
            repertoire_distance = resolve_mechanism_measure(
                config.formalism.iit.mechanism_phi_measure,
                system.substrate.factored_tpm.alphabet_sizes,
            )
        # Internal helpers below the formalism boundary require an
        # explicit ``mechanism_measure``; resolve it here.
        mechanism_measure = kwargs.pop("mechanism_measure", repertoire_distance)
        if repertoire is None:
            repertoire = system.repertoire(direction, mechanism, purview)
        if partitioned_repertoire is None:
            partitioned_repertoire_kwargs = partitioned_repertoire_kwargs or {}
            partitioned_repertoire = system.partitioned_repertoire(
                direction,
                partition,
                mechanism_measure=mechanism_measure,
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

        IIT 3.0 has no specified-state phase, so measure kwargs are not
        threaded through this method. The system-level measure is read
        from ``config.formalism.iit.mechanism_phi_measure`` inside the
        underlying ``sia`` implementation; compatibility is checked
        against the active formalism's ``compatible_measures`` here.
        Callers attempting to pass ``system_measure`` /
        ``specification_measure`` receive a :class:`TypeError` rather
        than a silent no-op.
        """
        check_measure_compatible(self, config.formalism.iit.mechanism_phi_measure)
        check_sia_tie_strategy_compatible(self, config.formalism.iit.sia_tie_resolution)
        from pyphi.formalism.iit3 import sia as _sia

        return _sia(system, **kwargs)

    def build_ces(self, system: Any, **kwargs: Any) -> Any:
        """IIT 3.0 CES is exactly the set of distinctions (no relations).

        Returns the distinctions specified by the system; this is the IIT
        3.0 cause-effect structure.
        """
        from pyphi.formalism.iit3 import (
            _compute_distinctions as _ces,  # pyright: ignore[reportPrivateUsage]
        )

        return _ces(system, **kwargs)
