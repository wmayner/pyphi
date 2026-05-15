# actual.py
"""
Methods for computing actual causation of systems and mechanisms.

If you use this module, please cite the following papers:

    Albantakis L, Marshall W, Hoel E, Tononi G (2019).
    What Caused What? A quantitative Account of Actual Causation Using
    Dynamical Causal Substrates.
    *Entropy*, 21 (5), pp. 459.
    `<https://doi.org/10.3390/e21050459>`_

    Mayner WGP, Marshall W, Albantakis L, Findlay G, Marchman R, Tononi G. (2018).
    PyPhi: A toolbox for integrated information theory.
    *PLOS Computational Biology* 14(7): e1006343.
    `<https://doi.org/10.1371/journal.pcbi.1006343>`_
"""

import contextlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from functools import cached_property
from itertools import chain
from types import MappingProxyType
from typing import Any

import numpy as np

from pyphi.registry import Registry

from . import conf
from . import connectivity
from . import exceptions
from . import resolve_ties
from . import utils
from . import validate
from .conf import config
from .direction import Direction
from .measures.distribution import actual_causation_measures as measures
from .measures.protocols import DistributionMeasure
from .models import Account
from .models import AcRepertoireIrreducibilityAnalysis
from .models import AcSystemIrreducibilityAnalysis
from .models import CausalLink
from .models import DirectedAccount
from .models import DirectedJointPartition
from .models import Event
from .models import NullCut
from .models import _null_ac_ria
from .models import _null_ac_sia
from .models import fmt
from .models.partitions import DirectedBipartition
from .parallel import MapReduce
from .partition import mechanism_partitions
from .substrate import Substrate
from .system import System

log = logging.getLogger(__name__)


class PartitionedRepertoireSchemeRegistry(Registry):
    """Registry of partitioned-repertoire computation schemes for actual causation.

    Schemes consume ``(transition_system, direction, partition)`` and
    return the partitioned repertoire as a probability distribution
    consistent with the parent System's TPM shape.
    """

    desc = "partitioned-repertoire schemes"


class BackgroundStrategyRegistry(Registry):
    """Registry of background-conditioning strategies for actual causation.

    Strategies consume ``(substrate, before_state, external_indices)`` and
    return either ``None`` (signaling uniform causal marginalization) or
    a state-weight callable.
    """

    desc = "background-conditioning strategies"


class AlphaAggregationRegistry(Registry):
    """Registry of α-aggregation rules for actual causation.

    Aggregators consume ``(rho, rho_partitioned)`` and return α — the
    integrated information of an actual cause/effect link.
    """  # noqa: RUF002

    desc = "α-aggregation rules"  # noqa: RUF001


partitioned_repertoire_schemes = PartitionedRepertoireSchemeRegistry()
background_strategies = BackgroundStrategyRegistry()
alpha_aggregations = AlphaAggregationRegistry()


@partitioned_repertoire_schemes.register("PRODUCT")
def _partitioned_repertoire_product(
    transition_system: Any,
    direction: Direction,
    partition: Any,
) -> Any:
    import functools

    from pyphi.core import repertoire_algebra as ra

    repertoires = [
        ra.repertoire(transition_system, direction, part.mechanism, part.purview)
        for part in partition
    ]
    return functools.reduce(np.multiply, repertoires)


@background_strategies.register("UNIFORM")
def _background_uniform(
    substrate: Any,  # noqa: ARG001
    before_state: Any,  # noqa: ARG001
    external_indices: Any,  # noqa: ARG001
) -> Any:
    return None


@alpha_aggregations.register("SUBTRACTIVE")
def _alpha_subtractive(rho: float, rho_partitioned: float) -> float:
    return rho - rho_partitioned


def _resolve_ac_kwargs() -> dict[str, Any]:
    """Resolve actual-causation formalism config into explicit kwargs.

    Public AC entry points (``sia``, ``account``, ``directed_account``,
    ``Transition.find_mip`` and friends) read the active configuration
    once at their boundary and thread the resolved values through
    internal helpers. The returned dict carries:

    - ``alpha_measure``: a :class:`DistributionMeasure` resolved from
      :data:`actual_causation_measures` by name
      (``config.formalism.actual_causation.alpha_measure``).
    - ``partitioned_repertoire_scheme``: a callable from
      :data:`partitioned_repertoire_schemes` keyed by
      ``config.formalism.actual_causation.partitioned_repertoire_scheme``.
    - ``background_scheme``: a callable from
      :data:`background_strategies` keyed by
      ``config.formalism.actual_causation.background_scheme``.
    - ``alpha_aggregation``: a callable from :data:`alpha_aggregations`
      keyed by ``config.formalism.actual_causation.alpha_aggregation``.
    """
    from pyphi.measures.distribution import resolve_actual_causation_measure

    ac = config.formalism.actual_causation
    return {
        "alpha_measure": resolve_actual_causation_measure(ac.alpha_measure),
        "partitioned_repertoire_scheme": partitioned_repertoire_schemes[
            ac.partitioned_repertoire_scheme
        ],
        "background_scheme": background_strategies[ac.background_scheme],
        "alpha_aggregation": alpha_aggregations[ac.alpha_aggregation],
    }


@dataclass(frozen=True, eq=False)
class TransitionSystem:
    """A directional view of a state transition.

    Implements :class:`pyphi.protocols.SystemPublicInterface` via the
    standard System surface (cause_tpm, effect_tpm, cm, node_indices,
    state, repertoire methods, etc.).

    The TPMs are conditioned on ``before_state`` for every substrate
    index outside ``cause_indices`` (the asymmetric background-conditioning
    rule from the 2019 Albantakis et al. formalism). The mechanism-
    evaluation ``state`` is ``after_state`` for the CAUSE direction and
    ``before_state`` for the EFFECT direction. Two TransitionSystem
    instances live inside each :class:`Transition`, one per direction.
    """

    substrate: Substrate
    before_state: tuple[int, ...]
    after_state: tuple[int, ...]
    cause_indices: tuple[int, ...]
    effect_indices: tuple[int, ...]
    direction: Direction
    partition: DirectedBipartition = field(default=None)  # type: ignore[assignment]
    noise_background: bool = False

    def __post_init__(self) -> None:
        validate.state_length(self.before_state, self.substrate.size)
        validate.state_length(self.after_state, self.substrate.size)
        validate.node_states(self.before_state)
        validate.node_states(self.after_state)
        coerce = self.substrate.node_labels.coerce_to_indices
        object.__setattr__(self, "cause_indices", coerce(self.cause_indices))
        object.__setattr__(self, "effect_indices", coerce(self.effect_indices))
        if self.partition is None:
            object.__setattr__(
                self, "partition", NullCut(self.node_indices, self.substrate.node_labels)
            )
        # The paper (Albantakis et al. 2019, Section 2.4) imposes only the
        # Realization axiom on a transition: p_u(after | before) > 0 over
        # the full system TPM. Subsystem forward-reachability on the
        # causally marginalized TPM (Eq. 2-4) has no paper basis — the
        # marginalized TPM is a tool for computing repertoires, not a
        # dynamical TPM with reachability semantics.

    @cached_property
    def node_indices(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.cause_indices) | set(self.effect_indices)))

    @cached_property
    def state(self) -> tuple[int, ...]:
        return (
            self.after_state if self.direction == Direction.CAUSE else self.before_state
        )

    @cached_property
    def external_indices(self) -> tuple[int, ...]:
        if self.noise_background:
            return ()
        all_indices = set(self.substrate.node_indices)
        return tuple(sorted(all_indices - set(self.cause_indices)))

    @cached_property
    def node_labels(self) -> Any:
        return self.substrate.node_labels

    @cached_property
    def proper_state(self) -> Any:
        return utils.state_of(self.node_indices, self.state)

    @cached_property
    def _underlying_system(self) -> Any:
        with config.override(validate_system_states=False):
            return System(
                substrate=self.substrate,
                state=self.before_state,
                node_indices=self.node_indices,
                partition=self.partition,
            )

    @cached_property
    def cause_tpm(self) -> Any:
        return self._underlying_system.cause_tpm

    @cached_property
    def effect_tpm(self) -> Any:
        from pyphi.core.tpm.explicit import ExplicitTPM as _TypedTPM
        from pyphi.core.tpm.marginalization import effect_tpm as _marginalize_effect

        legacy_tpm = self.substrate.tpm
        if hasattr(legacy_tpm, "to_array"):
            typed = _TypedTPM(legacy_tpm.to_array())
        else:
            typed = _TypedTPM(legacy_tpm)
        external_state = utils.state_of(self.external_indices, self.before_state)
        background = dict(zip(self.external_indices, external_state, strict=False))
        result = _marginalize_effect(typed, background)
        return result._inner if hasattr(result, "_inner") else result

    @cached_property
    def cm(self) -> Any:
        return self._underlying_system.cm

    @cached_property
    def proper_cause_tpm(self) -> Any:
        return self._underlying_system.proper_cause_tpm

    @cached_property
    def proper_effect_tpm(self) -> Any:
        return np.asarray(self.effect_tpm.squeeze())[..., list(self.node_indices)]

    @cached_property
    def proper_cm(self) -> Any:
        return self._underlying_system.proper_cm

    @cached_property
    def connectivity_matrix(self) -> Any:
        return self.cm

    @cached_property
    def partition_indices(self) -> tuple[int, ...]:
        return self.node_indices

    @cached_property
    def partition_node_labels(self) -> Any:
        return self.node_labels

    @cached_property
    def is_partitioned(self) -> bool:
        return not isinstance(self.partition, NullCut)

    @cached_property
    def size(self) -> int:
        return len(self.node_indices)

    @cached_property
    def tpm_size(self) -> int:
        return self.substrate.size

    @cached_property
    def nodes(self) -> Any:
        from pyphi.node import generate_nodes

        return generate_nodes(
            self.cause_tpm,
            self.effect_tpm,
            self.cm,
            self.state,
            self.node_indices,
            self.node_labels,
        )

    @cached_property
    def partitioned_mechanisms(self) -> Any:
        return list(self.partition.all_cut_mechanisms())

    @cached_property
    def _index2node(self) -> dict[int, Any]:
        return {node.index: node for node in self.nodes}

    @cached_property
    def null_distinction(self) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.null_distinction(self)

    @cached_property
    def null_concept(self) -> Any:
        return self.null_distinction

    def apply_cut(self, partition: DirectedBipartition) -> "TransitionSystem":
        return replace(self, partition=partition)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TransitionSystem):
            return NotImplemented
        return (
            self.substrate == other.substrate
            and self.before_state == other.before_state
            and self.after_state == other.after_state
            and self.cause_indices == other.cause_indices
            and self.effect_indices == other.effect_indices
            and self.direction == other.direction
            and self.partition == other.partition
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.substrate,
                self.before_state,
                self.after_state,
                self.cause_indices,
                self.effect_indices,
                self.direction,
                self.partition,
            )
        )

    def __len__(self) -> int:
        return len(self.node_indices)

    def __str__(self) -> str:
        labels = self.node_labels.coerce_to_labels(self.node_indices)
        joined = ", ".join(str(label) for label in labels)
        return f"TransitionSystem({self.direction}, {joined})"

    def cause_repertoire(self, mechanism: Any, purview: Any, **kw: Any) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.cause_repertoire(self, mechanism, purview, **kw)

    def effect_repertoire(self, mechanism: Any, purview: Any, **kw: Any) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.effect_repertoire(self, mechanism, purview, **kw)

    def repertoire(
        self, direction: Direction, mechanism: Any, purview: Any, **kw: Any
    ) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.repertoire(self, direction, mechanism, purview, **kw)

    def unconstrained_cause_repertoire(self, purview: Any) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.unconstrained_cause_repertoire(self, purview)

    def unconstrained_effect_repertoire(self, purview: Any) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.unconstrained_effect_repertoire(self, purview)

    def unconstrained_repertoire(self, direction: Direction, purview: Any) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.unconstrained_repertoire(self, direction, purview)

    def partitioned_repertoire(
        self,
        direction: Direction,
        partition: Any,
        *,
        partitioned_repertoire_scheme: Any,
        **kw: Any,
    ) -> Any:
        return partitioned_repertoire_scheme(self, direction, partition, **kw)

    def expand_cause_repertoire(
        self, repertoire_array: Any, *, new_purview: Any | None = None
    ) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.expand_cause_repertoire(
            self, repertoire_array, new_purview=new_purview
        )

    def expand_effect_repertoire(
        self, repertoire_array: Any, *, new_purview: Any | None = None
    ) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.expand_effect_repertoire(
            self, repertoire_array, new_purview=new_purview
        )

    def expand_repertoire(
        self,
        direction: Direction,
        repertoire_array: Any,
        new_purview: Any | None = None,
    ) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.expand_repertoire(
            self, direction, repertoire_array, new_purview=new_purview
        )

    def forward_cause_repertoire(
        self, mechanism: Any, purview: Any, purview_state: Any | None = None
    ) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.forward_cause_repertoire(self, mechanism, purview, purview_state)

    def forward_effect_repertoire(self, mechanism: Any, purview: Any, **kw: Any) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.forward_effect_repertoire(self, mechanism, purview, **kw)

    def forward_repertoire(
        self,
        direction: Direction,
        mechanism: Any,
        purview: Any,
        purview_state: Any | None = None,
        **kw: Any,
    ) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.forward_repertoire(
            self, direction, mechanism, purview, purview_state, **kw
        )

    def unconstrained_forward_cause_repertoire(
        self, mechanism: Any, purview: Any
    ) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.unconstrained_forward_cause_repertoire(self, mechanism, purview)

    def unconstrained_forward_effect_repertoire(
        self, mechanism: Any, purview: Any
    ) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.unconstrained_forward_effect_repertoire(self, mechanism, purview)

    def unconstrained_forward_repertoire(
        self, direction: Direction, mechanism: Any, purview: Any
    ) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.unconstrained_forward_repertoire(self, direction, mechanism, purview)

    def forward_cause_probability(
        self,
        mechanism: Any,
        purview: Any,
        purview_state: Any,
        mechanism_state: Any | None = None,
    ) -> float:
        from pyphi.core import repertoire_algebra as ra

        return ra.forward_cause_probability(
            self, mechanism, purview, purview_state, mechanism_state
        )

    def forward_effect_probability(
        self, mechanism: Any, purview: Any, purview_state: Any
    ) -> float:
        from pyphi.core import repertoire_algebra as ra

        return ra.forward_effect_probability(self, mechanism, purview, purview_state)

    def forward_probability(
        self,
        direction: Direction,
        mechanism: Any,
        purview: Any,
        purview_state: Any,
        **kw: Any,
    ) -> float:
        from pyphi.core import repertoire_algebra as ra

        return ra.forward_probability(
            self, direction, mechanism, purview, purview_state, **kw
        )

    def cause_info(
        self,
        mechanism: Any,
        purview: Any,
        *,
        mechanism_measure: Any,
        **kw: Any,
    ) -> float:
        from pyphi.core import repertoire_algebra as ra

        kw.setdefault("repertoire_distance", mechanism_measure)
        return ra.cause_info(self, mechanism, purview, **kw)

    def effect_info(
        self,
        mechanism: Any,
        purview: Any,
        *,
        mechanism_measure: Any,
        **kw: Any,
    ) -> float:
        from pyphi.core import repertoire_algebra as ra

        kw.setdefault("repertoire_distance", mechanism_measure)
        return ra.effect_info(self, mechanism, purview, **kw)

    def cause_effect_info(
        self,
        mechanism: Any,
        purview: Any,
        *,
        mechanism_measure: Any,
        **kw: Any,
    ) -> float:
        from pyphi.core import repertoire_algebra as ra

        kw.setdefault("repertoire_distance", mechanism_measure)
        return ra.cause_effect_info(self, mechanism, purview, **kw)

    def intrinsic_information(
        self,
        direction: Direction,
        mechanism: Any,
        purview: Any,
        *,
        specification_measure: Any,
        **kw: Any,
    ) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.intrinsic_information(
            self,
            direction,
            mechanism,
            purview,
            specification_measure=specification_measure,
            **kw,
        )

    def potential_purviews(
        self,
        direction: Direction,
        mechanism: Any,
        purviews: Any | None = None,
        **kw: Any,
    ) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.potential_purviews(self, direction, mechanism, purviews, **kw)

    def indices2nodes(self, indices: Any) -> Any:
        from pyphi.core import repertoire_algebra as ra

        return ra.indices2nodes(self, indices)

    def cache_info(self) -> dict[str, Any]:
        from pyphi.core import repertoire_algebra as ra

        return ra.cache_info()

    def clear_caches(self) -> None:
        from pyphi.core import repertoire_algebra as ra

        ra.clear_caches(self)

    def sia(self, **kw: Any) -> Any:
        raise NotImplementedError(
            "TransitionSystem does not support IIT formalism dispatch. "
            "Use pyphi.actual.sia(transition) for actual-causation analysis."
        )

    def ces(self, **kw: Any) -> Any:
        raise NotImplementedError(
            "TransitionSystem does not support IIT ces. "
            "Use pyphi.actual.account(transition, direction) instead."
        )

    def distinctions(self, **kw: Any) -> Any:
        raise NotImplementedError(
            "TransitionSystem does not support IIT distinctions. "
            "Use pyphi.actual.account(transition, direction) instead."
        )

    def find_mip(
        self, direction: Direction, mechanism: Any, purview: Any, **kw: Any
    ) -> Any:
        raise NotImplementedError(
            "TransitionSystem does not expose IIT mechanism MIP search. "
            "Use Transition.find_mip(direction, mechanism, purview) instead."
        )

    def cause_mip(self, mechanism: Any, purview: Any, **kw: Any) -> Any:
        raise NotImplementedError("Use Transition.find_mip instead.")

    def effect_mip(self, mechanism: Any, purview: Any, **kw: Any) -> Any:
        raise NotImplementedError("Use Transition.find_mip instead.")

    def phi_cause_mip(self, mechanism: Any, purview: Any, **kw: Any) -> float:
        raise NotImplementedError("Use Transition.find_mip instead.")

    def phi_effect_mip(self, mechanism: Any, purview: Any, **kw: Any) -> float:
        raise NotImplementedError("Use Transition.find_mip instead.")

    def phi(self, mechanism: Any, purview: Any, **kw: Any) -> float:
        raise NotImplementedError("AC has no IIT-style phi. See pyphi.actual.")

    def find_mice(self, direction: Direction, mechanism: Any, **kw: Any) -> Any:
        raise NotImplementedError(
            "Use Transition.find_causal_link(direction, mechanism) instead."
        )

    def mic(self, mechanism: Any, **kw: Any) -> Any:
        raise NotImplementedError("Use Transition.find_actual_cause instead.")

    def mie(self, mechanism: Any, **kw: Any) -> Any:
        raise NotImplementedError("Use Transition.find_actual_effect instead.")

    def phi_max(self, mechanism: Any) -> float:
        raise NotImplementedError("AC has no IIT-style phi_max.")

    def distinction(self, mechanism: Any) -> Any:
        raise NotImplementedError("AC has no IIT distinctions.")

    def all_distinctions(self, **kw: Any) -> Any:
        raise NotImplementedError("AC has no IIT distinctions.")

    def evaluate_partition(
        self,
        direction: Direction,
        mechanism: Any,
        purview: Any,
        partition: Any,
        **kw: Any,
    ) -> Any:
        raise NotImplementedError("Use Transition.find_mip / Transition.repertoire.")

    @classmethod
    def from_substrate(
        cls,
        substrate: Substrate,
        before_state: Any,
        after_state: Any,
        cause_indices: Any,
        effect_indices: Any,
        direction: Direction,
        partition: DirectedBipartition | None = None,
        **kwargs: Any,
    ) -> "TransitionSystem":
        return cls(
            substrate=substrate,
            before_state=tuple(before_state),
            after_state=tuple(after_state),
            cause_indices=tuple(cause_indices),
            effect_indices=tuple(effect_indices),
            direction=direction,
            partition=partition,  # type: ignore[arg-type]
            **kwargs,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "substrate": self.substrate,
            "before_state": list(self.before_state),
            "after_state": list(self.after_state),
            "cause_indices": list(self.cause_indices),
            "effect_indices": list(self.effect_indices),
            "direction": self.direction,
            "partition": self.partition,
            "noise_background": self.noise_background,
        }


@dataclass(frozen=True, eq=False)
class Transition:
    """A state transition over a substrate, holding two TransitionSystem views.

    Implements the actual-causation framework of Albantakis, Marshall, Hoel,
    and Tononi (2019). The cause-side and effect-side analyses live in
    :class:`TransitionSystem` instances accessed via :attr:`cause_system` and
    :attr:`effect_system`, keyed by Direction in :attr:`system`.

    Args:
        substrate (Substrate): The substrate the system belongs to.
        before_state (tuple[int]): The state of the substrate at time |t-1|.
        after_state (tuple[int]): The state of the substrate at time |t|.
        cause_indices (tuple[int] or tuple[str]): Indices of nodes in the
            cause system.
        effect_indices (tuple[int] or tuple[str]): Indices of nodes in the
            effect system.

    Keyword Args:
        partition (DirectedBipartition): The partition applied to this transition.
            Defaults to a :class:`NullCut` over the union of cause and effect indices.
        noise_background (bool): If ``True``, background conditions are
            noised instead of frozen.
    """

    substrate: Substrate
    before_state: tuple[int, ...]
    after_state: tuple[int, ...]
    cause_indices: tuple[int, ...]
    effect_indices: tuple[int, ...]
    partition: DirectedBipartition = field(default=None)  # type: ignore[assignment]
    noise_background: bool = False

    def __post_init__(self) -> None:
        coerce = self.substrate.node_labels.coerce_to_indices
        object.__setattr__(self, "cause_indices", coerce(self.cause_indices))
        object.__setattr__(self, "effect_indices", coerce(self.effect_indices))
        if self.partition is None:
            object.__setattr__(
                self, "partition", NullCut(self.node_indices, self.substrate.node_labels)
            )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Transition):
            return NotImplemented
        return (
            self.substrate == other.substrate
            and self.before_state == other.before_state
            and self.after_state == other.after_state
            and self.cause_indices == other.cause_indices
            and self.effect_indices == other.effect_indices
            and self.partition == other.partition
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.substrate,
                self.before_state,
                self.after_state,
                self.cause_indices,
                self.effect_indices,
                self.partition,
            )
        )

    def __len__(self) -> int:
        return len(self.node_indices)

    def __bool__(self) -> bool:
        return len(self) > 0

    def __repr__(self) -> str:
        return fmt.fmt_transition(self)

    def __str__(self) -> str:
        return repr(self)

    @cached_property
    def node_indices(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.cause_indices) | set(self.effect_indices)))

    @property
    def node_labels(self) -> Any:
        return self.substrate.node_labels

    @cached_property
    def cause_system(self) -> "TransitionSystem":
        return TransitionSystem(
            substrate=self.substrate,
            before_state=self.before_state,
            after_state=self.after_state,
            cause_indices=self.cause_indices,
            effect_indices=self.effect_indices,
            direction=Direction.CAUSE,
            partition=self.partition,
            noise_background=self.noise_background,
        )

    @cached_property
    def effect_system(self) -> "TransitionSystem":
        return TransitionSystem(
            substrate=self.substrate,
            before_state=self.before_state,
            after_state=self.after_state,
            cause_indices=self.cause_indices,
            effect_indices=self.effect_indices,
            direction=Direction.EFFECT,
            partition=self.partition,
            noise_background=self.noise_background,
        )

    @cached_property
    def system(self) -> Mapping[Direction, "TransitionSystem"]:
        return MappingProxyType(
            {
                Direction.CAUSE: self.cause_system,
                Direction.EFFECT: self.effect_system,
            }
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "substrate": self.substrate,
            "before_state": list(self.before_state),
            "after_state": list(self.after_state),
            "cause_indices": list(self.cause_indices),
            "effect_indices": list(self.effect_indices),
            "partition": self.partition,
        }

    def apply_cut(self, partition: DirectedBipartition) -> "Transition":
        return replace(self, partition=partition)

    def cause_repertoire(self, mechanism, purview):
        """Return the cause repertoire."""
        return self.repertoire(Direction.CAUSE, mechanism, purview)

    def effect_repertoire(self, mechanism, purview):
        """Return the effect repertoire."""
        return self.repertoire(Direction.EFFECT, mechanism, purview)

    def unconstrained_cause_repertoire(self, purview):
        """Return the unconstrained cause repertoire of the occurence."""
        return self.cause_repertoire((), purview)

    def unconstrained_effect_repertoire(self, purview):
        """Return the unconstrained effect repertoire of the occurence."""
        return self.effect_repertoire((), purview)

    def repertoire(self, direction, mechanism, purview):
        """Return the cause or effect repertoire function based on a direction.

        Args:
            direction (str): The temporal direction, specifiying the cause or
                effect repertoire.
        """
        system = self.system[direction]
        node_labels = system.node_labels

        if not set(purview).issubset(self.purview_indices(direction)):
            raise ValueError(
                f"{fmt.fmt_mechanism(purview, node_labels)} is not a "
                f"{direction} purview in {self}"
            )

        if not set(mechanism).issubset(self.mechanism_indices(direction)):
            raise ValueError(
                f"{fmt.fmt_mechanism(mechanism, node_labels)} is no a "
                f"{direction} mechanism in {self}"
            )

        return system.repertoire(direction, mechanism, purview)

    def state_probability(
        self,
        direction,
        repertoire,
        purview,
    ):
        """Compute the probability of the purview in its current state given
        the repertoire.

        Collapses the dimensions of the repertoire that correspond to the
        purview nodes onto their state. All other dimension are already
        singular and thus receive 0 as the conditioning index.

        Args:
            direction: The temporal direction (CAUSE or EFFECT).
            repertoire: The repertoire array to index into.
            purview: The purview nodes.

        Returns:
            float: A single probabilty.
        """
        purview_state = self.purview_state(direction)
        system = self.system[direction]

        # Determine which nodes the repertoire dimensions correspond to.
        # If repertoire.ndim equals substrate size, dimensions are for all
        # substrate nodes. If repertoire.ndim equals system size, dimensions
        # are for system nodes.
        if repertoire.ndim == system.substrate.size:
            node_indices = system.substrate.node_indices
        else:
            node_indices = system.node_indices

        index = tuple(
            purview_state[node] if node in purview else 0 for node in node_indices
        )
        return repertoire[index]

    def probability(self, direction, mechanism, purview):
        """Probability that the purview is in its current state given the
        state of the mechanism.
        """
        repertoire = self.repertoire(direction, mechanism, purview)

        return self.state_probability(direction, repertoire, purview)

    def unconstrained_probability(self, direction, purview):
        """Unconstrained probability of the purview."""
        return self.probability(direction, (), purview)

    def purview_state(self, direction):
        """The state of the purview when we are computing coefficients in
        ``direction``.

        For example, if we are computing the cause coefficient of a mechanism
        in ``after_state``, the direction is``CAUSE`` and the ``purview_state``
        is ``before_state``.
        """
        return {Direction.CAUSE: self.before_state, Direction.EFFECT: self.after_state}[
            direction
        ]

    def mechanism_state(self, direction):
        """The state of the mechanism when computing coefficients in
        ``direction``.
        """
        return self.system[direction].state

    def mechanism_indices(self, direction):
        """The indices of nodes in the mechanism system."""
        return {
            Direction.CAUSE: self.effect_indices,
            Direction.EFFECT: self.cause_indices,
        }[direction]

    def purview_indices(self, direction):
        """The indices of nodes in the purview system."""
        return {
            Direction.CAUSE: self.cause_indices,
            Direction.EFFECT: self.effect_indices,
        }[direction]

    def _ratio(self, direction, mechanism, purview):
        # Use the pointwise mutual information
        return probability_distance(
            self.probability(direction, mechanism, purview),
            self.unconstrained_probability(direction, purview),
            measure="PMI",
        )

    def cause_ratio(self, mechanism, purview):
        """The cause ratio of the ``purview`` given ``mechanism``.

        Always evaluated with PMI (pointwise mutual information), per
        the 2019 Albantakis et al. formalism, independent of
        ``config.formalism.actual_causation.alpha_measure``.
        """
        return self._ratio(Direction.CAUSE, mechanism, purview)

    def effect_ratio(self, mechanism, purview):
        """The effect ratio of the ``purview`` given ``mechanism``.

        Always evaluated with PMI (pointwise mutual information), per
        the 2019 Albantakis et al. formalism, independent of
        ``config.formalism.actual_causation.alpha_measure``.
        """
        return self._ratio(Direction.EFFECT, mechanism, purview)

    def partitioned_repertoire(
        self,
        direction,
        partition,
        *,
        partitioned_repertoire_scheme=None,
    ):
        """Compute the repertoire over the partition in the given direction."""
        if partitioned_repertoire_scheme is None:
            partitioned_repertoire_scheme = partitioned_repertoire_schemes[
                config.formalism.actual_causation.partitioned_repertoire_scheme
            ]
        return self.system[direction].partitioned_repertoire(
            direction,
            partition,
            partitioned_repertoire_scheme=partitioned_repertoire_scheme,
        )

    def partitioned_probability(
        self,
        direction,
        partition,
        *,
        partitioned_repertoire_scheme=None,
    ):
        """Compute the probability of the mechanism over the purview in
        the partition.
        """
        repertoire = self.partitioned_repertoire(
            direction,
            partition,
            partitioned_repertoire_scheme=partitioned_repertoire_scheme,
        )
        return self.state_probability(direction, repertoire, partition.purview)

    # MIP methods
    # =========================================================================

    # TODO: alias to `irreducible_cause/effect ratio?
    def find_mip(
        self,
        direction,
        mechanism,
        purview,
        allow_neg=False,
        *,
        alpha_measure: DistributionMeasure | None = None,
        partitioned_repertoire_scheme=None,
    ):
        """Find the ratio minimum information partition for a mechanism
        over a purview.

        Args:
            direction (str): |CAUSE| or |EFFECT|
            mechanism (tuple[int]): A mechanism.
            purview (tuple[int]): A purview.

        Keyword Args:
            allow_neg (boolean): If true, ``alpha`` is allowed to be negative.
                Otherwise, negative values of ``alpha`` will be treated as if
                they were 0.
            alpha_measure (DistributionMeasure): Resolved alpha measure callable.
                When ``None``, ``config.formalism.actual_causation.alpha_measure``
                is resolved at the call boundary.
            partitioned_repertoire_scheme: Resolved partitioned-repertoire
                scheme callable. When ``None``,
                ``config.formalism.actual_causation.partitioned_repertoire_scheme``
                is resolved at the call boundary.

        Returns:
            AcRepertoireIrreducibilityAnalysis: The irreducibility analysis for
            the mechanism.
        """
        if alpha_measure is None or partitioned_repertoire_scheme is None:
            resolved = _resolve_ac_kwargs()
            if alpha_measure is None:
                alpha_measure = resolved["alpha_measure"]
            if partitioned_repertoire_scheme is None:
                partitioned_repertoire_scheme = resolved["partitioned_repertoire_scheme"]

        if not purview:
            return _null_ac_ria(
                self.mechanism_state(direction), direction, mechanism, purview
            )

        probability = self.probability(direction, mechanism, purview)
        candidates: list[AcRepertoireIrreducibilityAnalysis] = []
        for partition in mechanism_partitions(mechanism, purview, self.node_labels):
            partitioned_probability = self.partitioned_probability(
                direction,
                partition,
                partitioned_repertoire_scheme=partitioned_repertoire_scheme,
            )
            alpha = probability_distance(
                probability,
                partitioned_probability,
                alpha_measure=alpha_measure,
            )
            # Reducibility short-circuit: |alpha|=0 (or negative when
            # disallowed) means the mechanism is reducible against this
            # partition; no need to keep searching since min |alpha| can't
            # go lower.
            if utils.eq(alpha, 0) or (alpha < 0 and not allow_neg):
                return _null_ac_ria(
                    self.mechanism_state(direction),
                    direction,
                    mechanism,
                    purview,
                    partition,
                )
            candidates.append(
                AcRepertoireIrreducibilityAnalysis(
                    state=self.mechanism_state(direction),
                    direction=direction,
                    mechanism=mechanism,
                    purview=purview,
                    partition=partition,
                    probability=probability,
                    partitioned_probability=partitioned_probability,
                    node_labels=self.node_labels,
                    alpha=alpha,
                )
            )
        if not candidates:
            return None
        context = resolve_ties.ResolutionContext(max_escalation_level="Determinism")
        outcome = resolve_ties.resolve_ac_partition_tie(candidates, context=context)
        winner = outcome.resolved
        if winner is not None and len(outcome.tied_set) > 1:
            winner.set_partition_ties(outcome.tied_set)
        return winner

    # Phi_max methods
    # =========================================================================

    def potential_purviews(self, direction, mechanism, purviews=None):
        """Return all purviews that could belong to the |MIC|/|MIE|.

        Filters out trivially-reducible purviews.

        Args:
            direction (str): Either |CAUSE| or |EFFECT|.
            mechanism (tuple[int]): The mechanism of interest.

        Keyword Args:
            purviews (tuple[int]): Optional subset of purviews of interest.
        """
        system = self.system[direction]
        return [
            purview
            for purview in system.potential_purviews(direction, mechanism, purviews)  # pyright: ignore[reportCallIssue]
            if set(purview).issubset(self.purview_indices(direction))
        ]

    def find_causal_link(
        self,
        direction,
        mechanism,
        purviews=None,
        allow_neg=False,
        *,
        alpha_measure: DistributionMeasure | None = None,
        partitioned_repertoire_scheme=None,
    ):
        """Return the maximally irreducible cause or effect ratio for a
        mechanism.

        Args:
            direction (str): The temporal direction, specifying cause or
                effect.
            mechanism (tuple[int]): The mechanism to be tested for
                irreducibility.

        Keyword Args:
            purviews (tuple[int]): Optionally restrict the possible purviews
                to a subset of the system. This may be useful for _e.g._
                finding only concepts that are "about" a certain subset of
                nodes.
            alpha_measure (DistributionMeasure): Resolved alpha measure
                callable. When ``None``,
                ``config.formalism.actual_causation.alpha_measure`` is
                resolved at the call boundary.
            partitioned_repertoire_scheme: Resolved partitioned-repertoire
                scheme callable. When ``None``,
                ``config.formalism.actual_causation.partitioned_repertoire_scheme``
                is resolved at the call boundary.

        Returns:
            CausalLink: The maximally-irreducible actual cause or effect.
        """
        if alpha_measure is None or partitioned_repertoire_scheme is None:
            resolved = _resolve_ac_kwargs()
            if alpha_measure is None:
                alpha_measure = resolved["alpha_measure"]
            if partitioned_repertoire_scheme is None:
                partitioned_repertoire_scheme = resolved["partitioned_repertoire_scheme"]

        purviews = self.potential_purviews(direction, mechanism, purviews)

        # Find the maximal RIA over the remaining purviews.
        if not purviews:
            max_ria = _null_ac_ria(
                self.mechanism_state(direction), direction, mechanism, None
            )
            return CausalLink(max_ria)

        # Finds rias with maximum alpha
        all_ria = [
            self.find_mip(
                direction,
                mechanism,
                purview,
                allow_neg=allow_neg,
                alpha_measure=alpha_measure,
                partitioned_repertoire_scheme=partitioned_repertoire_scheme,
            )
            for purview in purviews
        ]
        # Filter out None values and bail if no candidates have alpha > 0.
        valid_ria = [ria for ria in all_ria if ria is not None and bool(ria)]
        if not valid_ria:
            return []
        context = resolve_ties.ResolutionContext(max_escalation_level="Determinism")
        outcome = resolve_ties.resolve_ac_causal_link_tie(valid_ria, context=context)
        winner = outcome.resolved
        assert winner is not None, "AC causal-link cascade returned no winner"
        extended_purview = tuple(r.purview for r in outcome.tied_set)
        purview_ties = tuple(outcome.tied_set) if len(outcome.tied_set) > 1 else None
        return CausalLink(winner, extended_purview, purview_ties=purview_ties)

    def find_actual_cause(self, mechanism, purviews=None, **kw):
        """Return the actual cause of a mechanism."""
        return self.find_causal_link(Direction.CAUSE, mechanism, purviews, **kw)

    def find_actual_effect(self, mechanism, purviews=None, **kw):
        """Return the actual effect of a mechanism."""
        return self.find_causal_link(Direction.EFFECT, mechanism, purviews, **kw)

    def find_mice(self, *args, **kwargs):
        """Backwards-compatible alias for :func:`find_causal_link`."""
        return self.find_causal_link(*args, **kwargs)


# =============================================================================
# Accounts
# =============================================================================


def directed_account(
    transition,
    direction,
    mechanisms=None,
    purviews=None,
    allow_neg=False,
    *,
    alpha_measure: DistributionMeasure | None = None,
    partitioned_repertoire_scheme=None,
):
    """Return the set of all |CausalLinks| of the specified direction.

    Keyword Args:
        alpha_measure (DistributionMeasure): Resolved alpha measure callable.
            When ``None``, ``config.formalism.actual_causation.alpha_measure``
            is resolved at the call boundary.
        partitioned_repertoire_scheme: Resolved partitioned-repertoire scheme
            callable. When ``None``, the active
            ``config.formalism.actual_causation.partitioned_repertoire_scheme``
            is resolved at the call boundary.
    """
    if alpha_measure is None or partitioned_repertoire_scheme is None:
        resolved = _resolve_ac_kwargs()
        if alpha_measure is None:
            alpha_measure = resolved["alpha_measure"]
        if partitioned_repertoire_scheme is None:
            partitioned_repertoire_scheme = resolved["partitioned_repertoire_scheme"]

    if mechanisms is None:
        mechanisms = utils.powerset(
            transition.mechanism_indices(direction), nonempty=True
        )
    links = [
        transition.find_causal_link(
            direction,
            mechanism,
            purviews=purviews,
            allow_neg=allow_neg,
            alpha_measure=alpha_measure,
            partitioned_repertoire_scheme=partitioned_repertoire_scheme,
        )
        for mechanism in mechanisms
    ]

    # Filter out causal links with zero alpha
    return DirectedAccount(filter(None, links))


def account(
    transition,
    direction=Direction.BIDIRECTIONAL,
    *,
    alpha_measure: DistributionMeasure | None = None,
    partitioned_repertoire_scheme=None,
):
    """Return the set of all causal links for a |Transition|.

    Args:
        transition (Transition): The transition of interest.

    Keyword Args:
        direction (Direction): By default the account contains actual causes
            and actual effects.
        alpha_measure (DistributionMeasure): Resolved alpha measure callable.
            When ``None``, ``config.formalism.actual_causation.alpha_measure``
            is resolved at the call boundary.
        partitioned_repertoire_scheme: Resolved partitioned-repertoire scheme
            callable. When ``None``, the active
            ``config.formalism.actual_causation.partitioned_repertoire_scheme``
            is resolved at the call boundary.
    """
    if alpha_measure is None or partitioned_repertoire_scheme is None:
        resolved = _resolve_ac_kwargs()
        if alpha_measure is None:
            alpha_measure = resolved["alpha_measure"]
        if partitioned_repertoire_scheme is None:
            partitioned_repertoire_scheme = resolved["partitioned_repertoire_scheme"]

    if direction != Direction.BIDIRECTIONAL:
        return directed_account(
            transition,
            direction,
            alpha_measure=alpha_measure,
            partitioned_repertoire_scheme=partitioned_repertoire_scheme,
        )

    return Account(
        directed_account(
            transition,
            Direction.CAUSE,
            alpha_measure=alpha_measure,
            partitioned_repertoire_scheme=partitioned_repertoire_scheme,
        )
        + directed_account(
            transition,
            Direction.EFFECT,
            alpha_measure=alpha_measure,
            partitioned_repertoire_scheme=partitioned_repertoire_scheme,
        )
    )


def probability_distance(
    p: float,
    q: float,
    measure: str | None = None,
    *,
    alpha_measure: DistributionMeasure | None = None,
) -> float:
    """Compute the distance between two probabilities in actual causation.

    Args:
        p (float): The first probability.
        q (float): The second probability.

    Keyword Args:
        measure (str): Optional measure name registered in
            :data:`pyphi.measures.distribution.actual_causation_measures`.
            Mutually exclusive with ``alpha_measure``.
        alpha_measure (DistributionMeasure): Optional resolved measure callable
            (e.g., from
            :func:`pyphi.measures.distribution.resolve_actual_causation_measure`).
            Internal callers thread the resolved object through to avoid
            repeated registry lookups; external callers may pass ``measure``.
            If both are ``None``, the active configuration's
            ``alpha_measure`` is resolved.

    Returns:
        float: The probability distance between ``p`` and ``q``.
    """
    if alpha_measure is not None and measure is not None:
        raise ValueError(
            "probability_distance accepts at most one of "
            "`measure` or `alpha_measure`; got both."
        )
    if alpha_measure is None:
        name = (
            config.formalism.actual_causation.alpha_measure
            if measure is None
            else measure
        )
        measure_func = measures[name]
    else:
        measure_func = alpha_measure
    dist = measure_func(p, q)
    return round(dist, config.numerics.precision)


# =============================================================================
# AcSystemIrreducibilityAnalysiss and System cuts
# =============================================================================


def account_distance(A1, A2):
    """Return the distance between two accounts. Here that is just the
    difference in sum(alpha)

    Args:
        A1 (Account): The first account.
        A2 (Account): The second account

    Returns:
        float: The distance between the two accounts.
    """
    return sum([action.alpha for action in A1]) - sum([action.alpha for action in A2])


def _evaluate_partition(
    partition,
    transition,
    unpartitioned_account,
    direction=Direction.BIDIRECTIONAL,
    *,
    alpha_measure: DistributionMeasure,
    partitioned_repertoire_scheme,
):
    """Find the |AcSystemIrreducibilityAnalysis| for a given partition."""
    partitioned_transition = transition.apply_cut(partition)
    partitioned_account = account(
        partitioned_transition,
        direction,
        alpha_measure=alpha_measure,
        partitioned_repertoire_scheme=partitioned_repertoire_scheme,
    )

    log.debug("Finished evaluating %s.", partition)
    alpha = account_distance(unpartitioned_account, partitioned_account)

    return AcSystemIrreducibilityAnalysis(
        alpha=round(alpha, config.numerics.precision),
        direction=direction,
        account=unpartitioned_account,
        partitioned_account=partitioned_account,
        partition=partition,
        before_state=transition.before_state,
        after_state=transition.after_state,
        size=len(transition),
        node_indices=transition.node_indices,
        cause_indices=transition.cause_indices,
        effect_indices=transition.effect_indices,
        node_labels=transition.substrate.node_labels,
    )


# TODO: implement CUT_ONE approximation?
def _get_partitions(transition, direction):
    """A list of possible partitions of a transition."""
    n = transition.substrate.size

    if direction is Direction.BIDIRECTIONAL:
        yielded = set()
        for partition in chain(
            _get_partitions(transition, Direction.CAUSE),
            _get_partitions(transition, Direction.EFFECT),
        ):
            cm = utils.np_hashable(partition.cut_matrix(n))
            if cm not in yielded:
                yielded.add(cm)
                yield partition

    else:
        mechanism = transition.mechanism_indices(direction)
        purview = transition.purview_indices(direction)
        for inner_partition in mechanism_partitions(
            mechanism, purview, transition.node_labels
        ):
            yield DirectedJointPartition(
                direction, inner_partition, transition.node_labels
            )


def sia(transition, direction=Direction.BIDIRECTIONAL, **kwargs):
    """Return the minimal information partition of a transition in a specific
    direction.

    Args:
        transition (Transition): The candidate system.

    Returns:
        AcSystemIrreducibilityAnalysis: A nested structure containing all the
        data from the intermediate calculations. The top level contains the
        basic irreducibility information for the given system.
    """
    validate.direction(direction, allow_bi=True)
    log.info("Calculating big-alpha for %s...", transition)

    if not transition:
        log.info("Transition %s is empty; returning null SIA immediately.", transition)
        return _null_ac_sia(transition, direction)

    if not connectivity.is_weak(transition.substrate.cm, transition.node_indices):
        log.info(
            "%s is not strongly/weakly connected; returning null SIA immediately.",
            transition,
        )
        return _null_ac_sia(transition, direction)

    resolved = _resolve_ac_kwargs()
    alpha_measure = resolved["alpha_measure"]
    partitioned_repertoire_scheme = resolved["partitioned_repertoire_scheme"]

    log.debug("Finding unpartitioned account...")
    unpartitioned_account = account(
        transition,
        direction,
        alpha_measure=alpha_measure,
        partitioned_repertoire_scheme=partitioned_repertoire_scheme,
    )
    log.debug("Found unpartitioned account.")

    if not unpartitioned_account:
        log.info("Empty unpartitioned account; returning null AC SIA immediately.")
        return _null_ac_sia(transition, direction)

    cuts = _get_partitions(transition, direction)

    parallel_kwargs = conf.parallel_kwargs(
        dict(config.infrastructure.parallel_partition_evaluation), **kwargs
    )
    result = MapReduce(
        _evaluate_partition,
        cuts,
        map_kwargs={
            "transition": transition,
            "direction": direction,
            "unpartitioned_account": unpartitioned_account,
            "alpha_measure": alpha_measure,
            "partitioned_repertoire_scheme": partitioned_repertoire_scheme,
        },
        reduce_func=min,
        reduce_kwargs={
            "default": _null_ac_sia(transition, direction, alpha=float("inf"))
        },
        shortcircuit_func=utils.is_falsy,
        **parallel_kwargs,
    ).run()
    log.info("Finished calculating big-ac-phi data for %s.", transition)
    log.debug("RESULT: \n%s", result)
    return result


# =============================================================================
# Complexes
# =============================================================================


# TODO: Fix this to test whether the transition is possible
def transitions(substrate, before_state, after_state):
    """Return a generator of all **possible** transitions of a substrate."""
    # TODO: Does not return systems that are in an impossible transitions.

    # Elements without inputs are reducibe effects,
    # elements without outputs are reducible causes.
    possible_causes = np.where(np.sum(substrate.cm, 1) > 0)[0]
    possible_effects = np.where(np.sum(substrate.cm, 0) > 0)[0]

    for cause_subset in utils.powerset(possible_causes, nonempty=True):
        for effect_subset in utils.powerset(possible_effects, nonempty=True):
            with contextlib.suppress(exceptions.StateUnreachableError):
                yield Transition(
                    substrate, before_state, after_state, cause_subset, effect_subset
                )


def nexus(substrate, before_state, after_state, direction=Direction.BIDIRECTIONAL):
    """Return a tuple of all irreducible nexus of the substrate."""
    validate.is_substrate(substrate)

    sias = (
        sia(transition, direction)
        for transition in transitions(substrate, before_state, after_state)
    )
    return tuple(sorted(filter(None, sias), reverse=True))


def causal_nexus(
    substrate, before_state, after_state, direction=Direction.BIDIRECTIONAL
):
    """Return the causal nexus of the substrate."""
    validate.is_substrate(substrate)

    log.info("Calculating causal nexus...")
    result = nexus(substrate, before_state, after_state, direction)
    if result:
        result = max(result)
    else:
        null_transition = Transition(substrate, before_state, after_state, (), ())
        result = _null_ac_sia(null_transition, direction)

    log.info("Finished calculating causal nexus.")
    log.debug("RESULT: \n%s", result)
    return result


# =============================================================================
# True Causes
# =============================================================================


# TODO: move this to __str__
def nice_true_ces(tc):
    """Format a true |Distinctions|."""
    cause_list = []
    next_list = []
    cause = "<--"
    effect = "-->"
    for event in tc:
        if event.direction == Direction.CAUSE:
            cause_list.append(
                [
                    f"{round(event.alpha, 4):.4f}",
                    event.mechanism,
                    cause,
                    event.purview,
                ]
            )
        elif event.direction == Direction.EFFECT:
            next_list.append(
                [
                    f"{round(event.alpha, 4):.4f}",
                    event.mechanism,
                    effect,
                    event.purview,
                ]
            )
        else:
            validate.direction(event.direction)

    true_list = [
        (cause_list[event], next_list[event]) for event in range(len(cause_list))
    ]
    return true_list


def _actual_causes(substrate, previous_state, current_state, nodes, mechanisms=None):
    log.info("Calculating true causes ...")
    transition = Transition(substrate, previous_state, current_state, nodes, nodes)

    return directed_account(transition, Direction.CAUSE, mechanisms=mechanisms)


def _actual_effects(substrate, current_state, next_state, nodes, mechanisms=None):
    log.info("Calculating true effects ...")
    transition = Transition(substrate, current_state, next_state, nodes, nodes)

    return directed_account(transition, Direction.EFFECT, mechanisms=mechanisms)


def events(substrate, previous_state, current_state, next_state, nodes, mechanisms=None):
    """Find all events (mechanisms with actual causes and actual effects)."""
    actual_causes = _actual_causes(
        substrate, previous_state, current_state, nodes, mechanisms
    )
    actual_effects = _actual_effects(
        substrate, current_state, next_state, nodes, mechanisms
    )
    actual_mechanisms = {c.mechanism for c in actual_causes} & {
        c.mechanism for c in actual_effects
    }

    if not actual_mechanisms:
        return ()

    def index(actual_causes_or_effects):
        """Filter out unidirectional occurences and return a dictionary keyed
        by the mechanism of the cause or effect.
        """
        return {
            o.mechanism: o
            for o in actual_causes_or_effects
            if o.mechanism in actual_mechanisms
        }

    actual_causes = index(actual_causes)
    actual_effects = index(actual_effects)

    return tuple(
        Event(actual_causes[m], actual_effects[m]) for m in sorted(actual_mechanisms)
    )


# TODO: do we need this? it's just a re-structuring of the `events` results
# TODO: rename to `actual_ces`?
def true_ces(system, previous_state, next_state):
    """Set of all sets of elements that have true causes and true effects.

    .. note::
        Since the true |Distinctions| is always about the full system,
        the background conditions don't matter and the system should be
        conditioned on the current state.
    """
    substrate = system.substrate
    nodes = system.node_indices
    state = system.state

    _events = events(substrate, previous_state, state, next_state, nodes)

    if not _events:
        log.info("Finished calculating, no echo events.")
        return None

    result = tuple(
        [event.actual_cause for event in _events]
        + [event.actual_effect for event in _events]
    )
    log.info("Finished calculating true events.")
    log.debug("RESULT: \n%s", result)

    return result


def true_events(
    substrate,
    previous_state,
    current_state,
    next_state,
    indices=None,
    major_complex=None,
):
    """Return all mechanisms that have true causes and true effects within the
    complex.

    Args:
        substrate (Substrate): The substrate to analyze.
        previous_state (tuple[int]): The state of the substrate at ``t - 1``.
        current_state (tuple[int]): The state of the substrate at ``t``.
        next_state (tuple[int]): The state of the substrate at ``t + 1``.

    Keyword Args:
        indices (tuple[int]): The indices of the major complex.
        major_complex (AcSystemIrreducibilityAnalysis): The major complex. If
            ``major_complex`` is given then ``indices`` is ignored.

    Returns:
        tuple[Event]: List of true events in the major complex.
    """
    # TODO: validate triplet of states

    if major_complex:
        nodes = major_complex.node_indices
    elif indices:
        nodes = indices
    else:
        major_complex = substrate.maximal_complex(current_state)
        nodes = major_complex.node_indices  # pyright: ignore[reportOptionalMemberAccess]

    return events(substrate, previous_state, current_state, next_state, nodes)


def extrinsic_events(
    substrate,
    previous_state,
    current_state,
    next_state,
    indices=None,
    major_complex=None,
):
    """Set of all mechanisms that are in the major complex but which have true
    causes and effects within the entire substrate.

    Args:
        substrate (Substrate): The substrate to analyze.
        previous_state (tuple[int]): The state of the substrate at ``t - 1``.
        current_state (tuple[int]): The state of the substrate at ``t``.
        next_state (tuple[int]): The state of the substrate at ``t + 1``.

    Keyword Args:
        indices (tuple[int]): The indices of the major complex.
        major_complex (AcSystemIrreducibilityAnalysis): The major complex. If
            ``major_complex`` is given then ``indices`` is ignored.

    Returns:
        tuple(actions): List of extrinsic events in the major complex.
    """
    if major_complex:
        mc_nodes = major_complex.node_indices
    elif indices:
        mc_nodes = indices
    else:
        major_complex = substrate.maximal_complex(current_state)
        mc_nodes = major_complex.node_indices  # pyright: ignore[reportOptionalMemberAccess]

    mechanisms = list(utils.powerset(mc_nodes, nonempty=True))
    all_nodes = substrate.node_indices

    return events(
        substrate,
        previous_state,
        current_state,
        next_state,
        all_nodes,
        mechanisms=mechanisms,
    )
