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
from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import Any

import numpy as np

from . import exceptions
from . import utils
from . import validate
from .conf import config
from .direction import Direction
from .models import Event
from .models import NullCut
from .models import _null_ac_sia
from .models import fmt
from .models.partitions import DirectedBipartition
from .serializable import Serializable
from .substrate import Substrate
from .system import System

if TYPE_CHECKING:
    from pyphi.formalism.base import ActualCausationFormalism

log = logging.getLogger(__name__)


# The System surface that TransitionSystem delegates to its underlying
# System. This is an explicit allow-list: ``__getattr__`` (which fires only
# on attribute-lookup misses) delegates a name iff it appears here, and
# raises AttributeError otherwise. Names handled locally (dataclass fields,
# cached properties, AC-specific methods, IIT-formalism stubs) are found by
# normal attribute resolution before ``__getattr__`` is ever consulted, so
# they do not appear here.
#
# Allow-list rather than block-list by design: a new method added to
# ``System`` does NOT silently leak through TransitionSystem. In particular
# the IIT-formalism methods (sia, ces, find_mip, ...) are deliberately
# absent — they are stubbed locally to raise NotImplementedError, and any
# future IIT method is unsupported-by-default rather than silently returning
# a meaningless result computed over the background-conditioned system.
_DELEGATED_TO_SYSTEM: frozenset[str] = frozenset(
    {
        # TPM / structural surface:
        "cause_marginal",
        "effect_marginal",
        "proper_cause_marginal",
        "proper_effect_marginal",
        "cm",
        "proper_cm",
        "connectivity_matrix",
        "node_labels",
        "nodes",
        "_index2node",
        "proper_state",
        "size",
        "tpm_size",
        "null_distinction",
        "null_concept",
        # Repertoire algebra:
        "cause_repertoire",
        "effect_repertoire",
        "repertoire",
        "unconstrained_cause_repertoire",
        "unconstrained_effect_repertoire",
        "unconstrained_repertoire",
        "expand_cause_repertoire",
        "expand_effect_repertoire",
        "expand_repertoire",
        "forward_cause_repertoire",
        "forward_effect_repertoire",
        "forward_repertoire",
        "forward_cause_probability",
        "forward_effect_probability",
        "forward_probability",
        "unconstrained_forward_cause_repertoire",
        "unconstrained_forward_effect_repertoire",
        "unconstrained_forward_repertoire",
        "cause_info",
        "effect_info",
        "cause_effect_info",
        "intrinsic_information",
        "potential_purviews",
        "indices2nodes",
        "cache_info",
        "clear_caches",
        "to_networkx",
    }
)


@dataclass(frozen=True, eq=False)
class TransitionSystem:
    """A directional view of a state transition.

    Implements :class:`pyphi.protocols.SystemPublicInterface` by holding
    an underlying :class:`pyphi.system.System` (via
    :attr:`_underlying_system`) and delegating the protocol surface
    through :meth:`__getattr__`. The underlying System is constructed
    with ``external_indices = substrate.indices - cause_indices`` (or
    ``()`` when :attr:`noise_background` is True), encoding the AC
    paper's Section 3.3 extended-background convention applied to
    substrate units outside the cause set.

    The mechanism-evaluation ``state`` is direction-aware:
    ``after_state`` for the CAUSE direction (Bayesian-inverting from the
    observed effect) and ``before_state`` for the EFFECT direction
    (forward conditioning on the observed cause). Two
    :class:`TransitionSystem` instances live inside each
    :class:`Transition`, one per direction.

    The shared System surface delegated to the underlying System is the
    explicit allow-list :data:`_DELEGATED_TO_SYSTEM`; everything else is
    handled locally or unsupported.
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
        alphabet_sizes = self.substrate.factored_tpm.alphabet_sizes
        validate.node_states(self.before_state, alphabet_sizes)
        validate.node_states(self.after_state, alphabet_sizes)
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
        external = (
            ()
            if self.noise_background
            else tuple(
                sorted(set(self.substrate.node_indices) - set(self.cause_indices))
            )
        )
        with config.override(validate_system_states=False):
            return System(
                substrate=self.substrate,
                state=self.state,
                node_indices=self.node_indices,
                partition=self.partition,
                external_indices=external,
            )

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
    def partitioned_mechanisms(self) -> Any:
        return list(self.partition.all_cut_mechanisms())

    def apply_cut(self, partition: DirectedBipartition) -> "TransitionSystem":
        return replace(self, partition=partition)

    def partitioned_repertoire(
        self,
        direction: Direction,
        partition: Any,
        *,
        partitioned_repertoire_scheme: Any,
        **kw: Any,
    ) -> Any:
        """Compute the partitioned repertoire using the AC-paper scheme.

        Unlike ``System.partitioned_repertoire`` (which uses a
        mechanism-measure for IIT), AC's partitioned repertoire is the
        product of per-part repertoires (Eq. 8 in the 2019 paper),
        dispatched via the ``partitioned_repertoire_scheme`` registry.
        """
        return partitioned_repertoire_scheme(self, direction, partition, **kw)

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

    def __getattr__(self, name: str) -> Any:
        """Delegate the shared System surface to the underlying System.

        Fires only on attribute-lookup misses. Delegates a name iff it is in
        the explicit :data:`_DELEGATED_TO_SYSTEM` allow-list; otherwise
        raises AttributeError. This keeps unsupported and future System
        methods (notably the IIT-formalism methods) from silently leaking
        through.
        """
        if name in _DELEGATED_TO_SYSTEM:
            return getattr(self._underlying_system, name)
        raise AttributeError(name)

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


@dataclass(frozen=True, eq=False)
class Transition(Serializable):
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
        from pyphi.formalism.actual_causation.compute import probability_distance

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
            from pyphi.formalism.actual_causation.compute import (
                partitioned_repertoire_schemes,
            )

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
    def find_mip(self, direction, mechanism, purview, allow_neg=False, **kwargs):
        """Find the ratio minimum information partition for a mechanism
        over a purview.

        Dispatches through the active actual-causation formalism
        (``config.formalism.actual_causation.version``).
        """
        return _active_ac_formalism().evaluate_mechanism(
            self, direction, mechanism, purview, allow_neg=allow_neg, **kwargs
        )

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
            for purview in system.potential_purviews(
                direction, mechanism, purviews=purviews
            )
            if set(purview).issubset(self.purview_indices(direction))
        ]

    def find_causal_link(
        self, direction, mechanism, purviews=None, allow_neg=False, **kwargs
    ):
        """Return the maximally irreducible cause or effect ratio for a
        mechanism.

        Dispatches through the active actual-causation formalism
        (``config.formalism.actual_causation.version``).
        """
        return _active_ac_formalism().evaluate_causal_link(
            self, direction, mechanism, purviews=purviews, allow_neg=allow_neg, **kwargs
        )

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


def _active_ac_formalism() -> "ActualCausationFormalism":
    """Return the actual-causation formalism selected by config.

    Looks up ``ACTUAL_CAUSATION_FORMALISM_REGISTRY`` by
    ``config.formalism.actual_causation.version``. Imported lazily inside
    the function body to avoid an import cycle (mirroring
    :mod:`pyphi.formalism.queries`).
    """
    from pyphi.conf import config
    from pyphi.formalism.base import ACTUAL_CAUSATION_FORMALISM_REGISTRY

    return ACTUAL_CAUSATION_FORMALISM_REGISTRY[config.formalism.actual_causation.version]


def directed_account(
    transition,
    direction,
    mechanisms=None,
    purviews=None,
    allow_neg=False,
    **kwargs,
):
    """Return the set of all |CausalLinks| of the specified direction.

    Dispatches through the active actual-causation formalism
    (``config.formalism.actual_causation.version``).
    """
    return _active_ac_formalism().evaluate_account(
        transition,
        direction,
        mechanisms=mechanisms,
        purviews=purviews,
        allow_neg=allow_neg,
        **kwargs,
    )


def account(transition, direction=Direction.BIDIRECTIONAL, **kwargs):
    """Return the set of all causal links for a |Transition|.

    Dispatches through the active actual-causation formalism
    (``config.formalism.actual_causation.version``).
    """
    return _active_ac_formalism().evaluate_account(transition, direction, **kwargs)


def sia(transition, direction=Direction.BIDIRECTIONAL, **kwargs):
    """Return the minimal information partition of a transition in a specific
    direction.

    Dispatches through the active actual-causation formalism
    (``config.formalism.actual_causation.version``).
    """
    return _active_ac_formalism().evaluate_system(transition, direction, **kwargs)


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
