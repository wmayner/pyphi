# models/actual_causation.py
"""Objects that represent structures used in actual causation."""

from __future__ import annotations

import contextvars
from collections import namedtuple
from collections.abc import Sequence
from typing import Any
from typing import ClassVar

from pyphi import utils
from pyphi.direction import Direction

from . import cmp
from . import fmt

_SERIALIZING_AS_TIE_PEER: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "ac_serializing_as_tie_peer", default=False
)

# TODO(slipperyhank): add second state
_acria_attributes = [
    "alpha",
    "state",
    "direction",
    "mechanism",
    "purview",
    "partition",
    "probability",
    "partitioned_probability",
]
_acria_attributes_for_eq = [
    "alpha",
    "state",
    "direction",
    "mechanism",
    "purview",
    "probability",
]


def greater_than_zero(alpha):
    """Return ``True`` if alpha is greater than zero, accounting for
    numerical errors.
    """
    return bool(alpha > 0 and not utils.eq(alpha, 0))


class AcRepertoireIrreducibilityAnalysis(cmp.Orderable):
    """A minimum information partition for ac_coef calculation.


    These can be compared with the built-in Python comparison operators (``<``,
    ``>``, etc.). First, |alpha| values are compared. Then, if these are equal
    up to |PRECISION|, the size of the mechanism is compared.

    Attributes:
        alpha (float):
            This is the difference between the mechanism's unpartitioned and
            partitioned actual probability.
        state (tuple[int]):
            state of system in specified direction (cause, effect)
        direction (str):
            The temporal direction specifiying whether this analysis should be
            calculated with cause or effect repertoires.
        mechanism (tuple[int]):
            The mechanism to analyze.
        purview (tuple[int]):
            The purview over which the unpartitioned actual probability differs
            the least from the actual probability of the partition.
        partition (tuple[Part, Part]):
            The partition that makes the least difference to the mechanism's
            repertoire.
        probability (float):
            The probability of the state in the previous/next timestep.
        partitioned_probability (float):
            The probability of the state in the partitioned repertoire.
    """

    def __init__(
        self,
        alpha,
        state,
        direction,
        mechanism,
        purview,
        partition,
        probability,
        partitioned_probability,
        node_labels=None,
    ):
        self.alpha = alpha
        self.state = state
        self.direction = direction
        self.mechanism = mechanism
        self.purview = purview
        self.partition = partition
        self.probability = probability
        self.partitioned_probability = partitioned_probability
        self.node_labels = node_labels
        self._partition_ties: tuple[AcRepertoireIrreducibilityAnalysis, ...] | None = (
            None
        )

    __slots__ = ()

    @property
    def partition_ties(
        self,
    ) -> tuple[AcRepertoireIrreducibilityAnalysis, ...] | None:
        """Tuple of AcRIAs tied with this one at the cascade's min |alpha|
        level over the MIP search, or ``None`` if no tie."""
        return self._partition_ties

    def set_partition_ties(
        self, ties: Sequence[AcRepertoireIrreducibilityAnalysis] | None
    ) -> None:
        """Attach a tied AcRIA set to this analysis. The tied set is
        shared by reference among peers; each tied member exposes the
        same tuple via ``.partition_ties``."""
        if ties is None or len(tuple(ties)) <= 1:
            self._partition_ties = None
            return
        tied = tuple(ties)
        for member in tied:
            member._partition_ties = tied

    unorderable_unless_eq: ClassVar[list[str]] = ["direction"]

    def order_by(self):
        # Here we enforce that ties are broken in favor of smaller purviews
        return [self.alpha, len(self.mechanism), -len(self.purview)]

    def __eq__(self, other):
        # TODO(slipperyhank): include 2nd state here?
        if type(other) is not type(self):
            return NotImplemented
        return cmp.general_eq(self, other, _acria_attributes_for_eq)

    def __bool__(self):
        """An |AcRepertoireIrreducibilityAnalysis| is ``True`` if it has
        |alpha > 0|.
        """
        return greater_than_zero(self.alpha)

    @property
    def phi(self):
        """Alias for |alpha| for PyPhi utility functions."""
        return self.alpha

    def __hash__(self):
        attrs = tuple(getattr(self, attr) for attr in _acria_attributes_for_eq)
        return hash(attrs)

    def to_json(self):
        """Return a JSON-serializable representation."""
        dct: dict[str, Any] = {attr: getattr(self, attr) for attr in _acria_attributes}
        if _SERIALIZING_AS_TIE_PEER.get():
            return dct
        if self._partition_ties is None:
            return dct
        partition_peers = tuple(t for t in self._partition_ties if t is not self)
        if not partition_peers:
            return dct
        from pyphi.jsonify import jsonify

        token = _SERIALIZING_AS_TIE_PEER.set(True)
        try:
            dct["_partition_tie_peers"] = [jsonify(p.to_json()) for p in partition_peers]
        finally:
            _SERIALIZING_AS_TIE_PEER.reset(token)
        return dct

    @classmethod
    def from_json(cls, data):
        """Reconstruct an AcRIA, restoring tied peer set when present."""
        partition_peers_raw: Any = data.pop("_partition_tie_peers", ())
        partition_peers = tuple(cls(**dict(p)) for p in partition_peers_raw)
        instance = cls(**data)
        if partition_peers:
            tied = (instance, *partition_peers)
            instance._partition_ties = tied
            for peer in partition_peers:
                peer._partition_ties = tied
        return instance

    def __repr__(self):
        return fmt.make_repr(self, _acria_attributes)

    def __str__(self):
        return "RepertoireIrreducibilityAnalysis\n" + fmt.indent(fmt.fmt_ac_sia(self))


def _null_ac_ria(state, direction, mechanism, purview, partition=None):
    """The irreducibility AC analysis for a reducible causal link."""
    return AcRepertoireIrreducibilityAnalysis(
        state=state,
        direction=direction,
        mechanism=mechanism,
        purview=purview,
        partition=partition,
        probability=None,
        partitioned_probability=None,
        alpha=0.0,
    )


class CausalLink(cmp.Orderable):
    """A maximally irreducible actual cause or effect.

    These can be compared with the built-in Python comparison operators (``<``,
    ``>``, etc.). First, |alpha| values are compared. Then, if these are equal
    up to |PRECISION|, the size of the mechanism is compared.
    """

    def __init__(
        self,
        ria,
        extended_purview=None,
        *,
        purview_ties: Sequence[AcRepertoireIrreducibilityAnalysis] | None = None,
    ):
        self._ria = ria
        self._extended_purview = (
            tuple(extended_purview) if extended_purview is not None else None
        )
        self._purview_ties: tuple[AcRepertoireIrreducibilityAnalysis, ...] | None = (
            tuple(purview_ties)
            if purview_ties is not None and len(tuple(purview_ties)) > 1
            else None
        )

    @property
    def alpha(self):
        """float: The difference between the mechanism's unpartitioned and
        partitioned actual probabilities.
        """
        return self._ria.alpha

    @property
    def phi(self):
        """Alias for |alpha| for PyPhi utility functions."""
        return self.alpha

    @property
    def direction(self):
        """Direction: Either |CAUSE| or |EFFECT|."""
        return self._ria.direction

    @property
    def mechanism(self):
        """list[int]: The mechanism for which the action is evaluated."""
        return self._ria.mechanism

    @property
    def purview(self):
        """list[int]: The purview over which this mechanism's |alpha| is
        maximal.
        """
        return self._ria.purview

    @property
    def extended_purview(self):
        """tuple[tuple[int]]: List of purviews over which this causal link is
        maximally irreducible.

        Note: It will contain multiple purviews iff causal link has
        undetermined actual causes/effects (e.g. two irreducible causes with same alpha
        over different purviews).
        """
        return self._extended_purview

    @property
    def purview_ties(
        self,
    ) -> tuple[AcRepertoireIrreducibilityAnalysis, ...] | None:
        """Tuple of tied :class:`AcRepertoireIrreducibilityAnalysis`
        instances under symmetric over-determination — minimal candidates
        sharing alpha_max with non-comparable purviews (Albantakis et al.
        2019, Definition 1 outcome 2). ``None`` when the actual cause
        is unique."""
        return self._purview_ties

    @property
    def ria(self):
        """AcRepertoireIrreducibilityAnalysis: The irreducibility analysis for
        this mechanism.
        """
        return self._ria

    @property
    def node_labels(self):
        return self._ria.node_labels

    def __repr__(self):
        return fmt.make_repr(self, ["ria", "extended_purview"])

    def __str__(self):
        return "CausalLink\n" + fmt.indent(fmt.fmt_causal_link(self))

    unorderable_unless_eq = AcRepertoireIrreducibilityAnalysis.unorderable_unless_eq

    def order_by(self):
        return self.ria.order_by()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CausalLink):
            return NotImplemented
        return self.ria == other.ria

    def __hash__(self):
        return hash(self._ria)

    def __bool__(self):
        """An |CausalLink| is ``True`` if |alpha > 0|."""
        return greater_than_zero(self.alpha)

    def to_json(self):
        """Return a JSON-serializable representation."""
        dct: dict[str, Any] = {
            "ria": self.ria,
            "extended_purview": self._extended_purview,
        }
        if self._purview_ties is not None:
            from pyphi.jsonify import jsonify

            token = _SERIALIZING_AS_TIE_PEER.set(True)
            try:
                dct["_purview_tie_peers"] = [
                    jsonify(p.to_json()) for p in self._purview_ties
                ]
            finally:
                _SERIALIZING_AS_TIE_PEER.reset(token)
        return dct

    @classmethod
    def from_json(cls, data):
        """Reconstruct a CausalLink, restoring the tied purview set."""
        peers_raw: Any = data.pop("_purview_tie_peers", ())
        peers = tuple(AcRepertoireIrreducibilityAnalysis(**dict(p)) for p in peers_raw)
        return cls(
            ria=data["ria"],
            extended_purview=data.get("extended_purview"),
            purview_ties=peers if peers else None,
        )


class Event(namedtuple("Event", ["actual_cause", "actual_effect"])):
    """A mechanism which has both an actual cause and an actual effect.

    Attributes:
        actual_cause (CausalLink): The actual cause of the mechanism.
        actual_effect (CausalLink): The actual effect of the mechanism.
    """

    @property
    def mechanism(self):
        """The mechanism of the event."""
        assert self.actual_cause.mechanism == self.actual_effect.mechanism
        return self.actual_cause.mechanism


class Account(cmp.Orderable, Sequence):
    """The set of |CausalLinks| with |alpha > 0|. This includes both actual
    causes and actual effects.
    """

    def __init__(self, causal_links):
        self.causal_links = tuple(causal_links)

    def __len__(self):
        return len(self.causal_links)

    def __iter__(self):
        return iter(self.causal_links)

    def __getitem__(self, i):
        return self.causal_links[i]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Account):
            return NotImplemented
        return self.causal_links == other.causal_links

    def __hash__(self):
        return hash(self.causal_links)

    def __add__(self, other: object) -> Account:
        if not isinstance(other, Account):
            return NotImplemented
        return self.__class__(self.causal_links + other.causal_links)

    @property
    def irreducible_causes(self):
        """The set of irreducible causes in this |Account|."""
        return tuple(link for link in self if link.direction is Direction.CAUSE)

    @property
    def irreducible_effects(self):
        """The set of irreducible effects in this |Account|."""
        return tuple(link for link in self if link.direction is Direction.EFFECT)

    def __repr__(self):
        return fmt.make_repr(self, ["causal_links"])

    def __str__(self):
        return fmt.fmt_account(self)

    def to_json(self):
        return {"causal_links": tuple(self)}

    @classmethod
    def from_json(cls, dct):
        return cls(dct["causal_links"])


class DirectedAccount(Account):
    """The set of |CausalLinks| with |alpha > 0| for one direction of a
    transition.
    """


_ac_sia_attributes = [
    "alpha",
    "direction",
    "account",
    "partitioned_account",
    "partition",
    "before_state",
    "after_state",
    "size",
    "node_indices",
    "cause_indices",
    "effect_indices",
    "node_labels",
]


# TODO(slipperyhank): Check if we do the same, i.e. take the bigger system, or
# take the smaller?
class AcSystemIrreducibilityAnalysis(cmp.Orderable):
    """An analysis of transition-level irreducibility (|big_alpha|).

    Contains the |big_alpha| value of the |Transition|, the causal account, and
    all the intermediate results obtained in the course of computing them.

    Attributes:
        alpha (float): The |big_alpha| value for the transition when taken
            against this analysis, *i.e.* the difference between the
            unpartitioned account and this analysis's partitioned account.
        account (Account): The account of the whole transition.
        partitioned_account (Account): The account of the partitioned
            transition.
        partition (DirectedJointPartition): The minimal partition.
        before_state (tuple[int, ...]): The state of the substrate at time |t-1|.
        after_state (tuple[int, ...]): The state of the substrate at time |t|.
        size (int): Number of nodes in the transition.
        node_indices (tuple[int, ...]): Indices of nodes in the transition.
        node_labels (NodeLabels): Labels corresponding to ``node_indices``.
    """

    def __init__(
        self,
        alpha=None,
        direction=None,
        account=None,
        partitioned_account=None,
        partition=None,
        before_state=None,
        after_state=None,
        size=None,
        node_indices=None,
        cause_indices=None,
        effect_indices=None,
        node_labels=None,
        config=None,
    ):
        self.alpha = alpha
        self.direction = direction
        self.account = account
        self.partitioned_account = partitioned_account
        self.partition = partition
        self.before_state = before_state
        self.after_state = after_state
        self.size = size
        self.node_indices = node_indices
        self.cause_indices = cause_indices
        self.effect_indices = effect_indices
        self.node_labels = node_labels
        # ConfigSnapshot of the layered config at construction time.
        # Lazy-snapshot if None: callers that don't pass one still get a
        # recorded config (matching SystemIrreducibilityAnalysis).
        if config is None:
            from pyphi.conf import config as _global

            config = _global.snapshot()
        self.config = config

    def _repr_columns(self):
        return fmt.fmt_ac_sia_columns(self)

    def _repr_html_(self) -> str:
        return fmt.html_columns(self._repr_columns(), title=self.__class__.__name__)

    def __repr__(self):
        body = "\n".join(fmt.align_columns(self._repr_columns()))
        body = fmt.header(self.__class__.__name__, body, under_char=fmt.HEADER_BAR_2)
        body = fmt.center(body)
        return fmt.box(body)

    def __str__(self):
        return fmt.fmt_ac_sia(self)

    unorderable_unless_eq: ClassVar[list[str]] = ["direction"]

    # TODO: shouldn't the minimal irreducible account be chosen?
    def order_by(self):
        return [self.alpha, self.size]

    def __eq__(self, other):
        if type(other) is not type(self):
            return NotImplemented
        return cmp.general_eq(self, other, _ac_sia_attributes)

    def __bool__(self):
        """An |AcSystemIrreducibilityAnalysis| is ``True`` if it has
        |big_alpha > 0|.
        """
        return greater_than_zero(self.alpha)

    def __hash__(self):
        return hash(
            (
                self.alpha,
                self.account,
                self.partitioned_account,
                self.partition,
                self.before_state,
                self.after_state,
                self.size,
                self.node_indices,
                self.cause_indices,
                self.effect_indices,
            )
        )

    def to_json(self):
        return {attr: getattr(self, attr) for attr in _ac_sia_attributes}


def _null_ac_sia(transition, direction, alpha=0.0):
    """Return an |AcSystemIrreducibilityAnalysis| with zero |big_alpha| and
    empty accounts.
    """
    return AcSystemIrreducibilityAnalysis(
        direction=direction,
        alpha=alpha,
        account=(),
        partitioned_account=(),
        partition=transition.partition,
        before_state=transition.before_state,
        after_state=transition.after_state,
        size=len(transition),
        node_indices=transition.node_indices,
        cause_indices=transition.cause_indices,
        effect_indices=transition.effect_indices,
        node_labels=transition.substrate.node_labels,
    )
