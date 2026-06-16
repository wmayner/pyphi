# models/actual_causation.py
"""Objects that represent structures used in actual causation."""

from __future__ import annotations

import contextvars
from collections import namedtuple
from collections.abc import Sequence
from typing import Any

from pyphi import utils
from pyphi.direction import Direction
from pyphi.display import Description
from pyphi.display import Displayable
from pyphi.display import Row
from pyphi.display import Section
from pyphi.display import tone_of
from pyphi.display.numbers import format_value
from pyphi.display.tables import capped_table
from pyphi.models.explanation import Explanation
from pyphi.models.explanation import Finding

from . import cmp
from . import fmt
from .diff import Change
from .diff import ResultDiff
from .diff import _diff_common
from .partitions import concise_partition

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


def greater_than_zero(alpha):
    """Return ``True`` if alpha is greater than zero, accounting for
    numerical errors.
    """
    return bool(alpha > 0 and not utils.eq(alpha, 0))


class AcRepertoireIrreducibilityAnalysis(Displayable, cmp.Orderable):
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
        reasons=None,
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
        self.reasons = reasons or []
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

    def is_orderable_with(self, other: object) -> bool:
        return isinstance(other, AcRepertoireIrreducibilityAnalysis) and (
            self.direction == other.direction
        )

    def order_by(self):
        # Here we enforce that ties are broken in favor of smaller purviews
        return [self.alpha, len(self.mechanism), -len(self.purview)]

    def __eq__(self, other: object) -> bool:  # noqa: PLR0911
        # TODO(slipperyhank): include 2nd state here?
        if not isinstance(other, AcRepertoireIrreducibilityAnalysis):
            return NotImplemented
        if self.state != other.state:
            return False
        if self.direction != other.direction:
            return False
        if self.mechanism != other.mechanism:
            return False
        if self.purview != other.purview:
            return False
        if not utils.eq(self.alpha, other.alpha):
            return False
        return utils.eq(self.probability, other.probability)

    def __bool__(self):
        """An |AcRepertoireIrreducibilityAnalysis| is ``True`` if it has
        |alpha > 0|.
        """
        return greater_than_zero(self.alpha)

    @property
    def phi(self):
        """Alias for |alpha| for PyPhi utility functions."""
        return self.alpha

    def explain(self) -> Explanation:
        """A typed account of why this actual cause/effect link's |alpha| came
        out as it did."""
        findings = [
            Finding(kind="null_result", label="Null result", value=reason)
            for reason in (self.reasons or [])
        ]
        if self.purview:
            findings.append(Finding(kind="purview", label="Purview", value=self.purview))
        if self.partition is not None:
            findings.append(
                Finding(
                    kind="winning_partition",
                    label="Partition",
                    value=concise_partition(self.partition),
                )
            )
        return Explanation(
            subject=f"α = {format_value(self.alpha)}",  # noqa: RUF001
            level="mechanism",
            findings=tuple(findings),
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.state,
                self.direction,
                self.mechanism,
                self.purview,
            )
        )

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

    def _describe(self, verbosity: int) -> Description:  # noqa: ARG002
        cls = type(self).__name__
        mechanism_str = fmt.fmt_mechanism(self.mechanism, self.node_labels)
        purview_str = fmt.fmt_mechanism(self.purview, self.node_labels)
        partition_str = (
            concise_partition(self.partition) if self.partition is not None else None
        )
        return Description(
            title=cls,
            sections=(
                Section(
                    rows=(
                        Row("α", self.alpha),  # noqa: RUF001
                        Row(
                            "Direction",
                            str(self.direction),
                            tone=tone_of(self.direction),
                        ),
                        Row("Mechanism", mechanism_str),
                        Row("Purview", purview_str),
                        Row("State", str(self.state)),
                        Row("Partition", partition_str),
                        Row("Probability", self.probability),
                        Row("Partitioned probability", self.partitioned_probability),
                    ),
                ),
            ),
            compact=(
                f"{cls}(α={format_value(self.alpha)}, "  # noqa: RUF001
                f"{self.direction}, {mechanism_str}→{purview_str})"
            ),
        )


def _null_ac_ria(state, direction, mechanism, purview, partition=None, reasons=None):
    """The irreducibility AC analysis for a reducible causal link.

    ``reasons`` records why (a list of
    :class:`~pyphi.models.explanation.NullResultReason`).
    """
    return AcRepertoireIrreducibilityAnalysis(
        state=state,
        direction=direction,
        mechanism=mechanism,
        purview=purview,
        partition=partition,
        probability=None,
        partitioned_probability=None,
        alpha=0.0,
        reasons=reasons,
    )


class CausalLink(Displayable, cmp.Orderable):
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

    def explain(self) -> Explanation:
        """A typed account of why this causal link's |alpha| came out as it
        did, delegated to the underlying AcRIA."""
        return self._ria.explain()

    @property
    def node_labels(self):
        return self._ria.node_labels

    def _describe(self, verbosity: int) -> Description:  # noqa: ARG002
        cls = type(self).__name__
        node_labels = self.node_labels
        mechanism_str = fmt.fmt_mechanism(self.mechanism, node_labels)
        if self._extended_purview is not None and len(self._extended_purview) > 1:
            purview_str = fmt.fmt_extended_purview(self._extended_purview, node_labels)
        else:
            purview_str = fmt.fmt_mechanism(self.purview, node_labels)
        return Description(
            title=cls,
            sections=(
                Section(
                    rows=(
                        Row("α", self.alpha),  # noqa: RUF001
                        Row(
                            "Direction",
                            str(self.direction),
                            tone=tone_of(self.direction),
                        ),
                        Row("Mechanism", mechanism_str),
                        Row("Purview", purview_str),
                    ),
                ),
            ),
            compact=(
                f"{cls}(α={format_value(self.alpha)}, "  # noqa: RUF001
                f"{self.direction}, {mechanism_str}→{purview_str})"
            ),
        )

    def is_orderable_with(self, other: object) -> bool:
        return isinstance(other, CausalLink) and (self.direction == other.direction)

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


class Account(Displayable, cmp.Orderable, Sequence):
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

    @property
    def _sum_alpha(self):
        """Total alpha across all causal links."""
        return sum(link.alpha for link in self.causal_links)

    def _describe(self, verbosity: int) -> Description:  # noqa: ARG002
        cls = type(self).__name__
        num_links = len(self.causal_links)
        headers = ("Direction", "Mechanism", "Purview", "α")  # noqa: RUF001
        table = capped_table(
            headers,
            self.causal_links,
            lambda link: (
                str(link.direction),
                fmt.fmt_mechanism(link.mechanism, link.node_labels),
                fmt.fmt_mechanism(link.purview, link.node_labels),
                link.alpha,
            ),
            total=num_links,
            cell_tones=lambda link: (tone_of(link.direction), None, None, None),
        )
        return Description(
            title=cls,
            sections=(
                Section(
                    rows=(
                        Row("Causal links", num_links),
                        Row("Σα", self._sum_alpha),
                    ),
                ),
                Section(label="Causal links", body=(table,)),
            ),
            compact=(f"{cls}({num_links} links, Σα={format_value(self._sum_alpha)})"),
        )

    def explain(self) -> Explanation:
        """A typed account listing each irreducible causal link with its
        |alpha|."""
        findings = [
            Finding(
                kind="link",
                label=f"{link.direction}: {link.mechanism} → {link.purview}",
                value=link.alpha,
                tone=tone_of(link.direction),
            )
            for link in self.causal_links
        ]
        return Explanation(
            subject=f"Account ({len(self.causal_links)} links)",
            level="system",
            findings=tuple(findings),
        )

    def diff(self, other) -> ResultDiff:
        """Structured delta from this account to ``other`` (``a.diff(b)``).

        Causal links are keyed by direction + mechanism + purview; a link
        present in both is *changed* when its |alpha| differs. An account
        carries no :class:`ConfigSnapshot`, so ``config_diff`` is empty.
        """
        from pyphi import utils
        from pyphi.data_structures import PyPhiFloat

        if not isinstance(other, Account):
            raise TypeError(
                f"cannot diff {type(self).__name__} against {type(other).__name__}"
            )

        def key(link):
            return (str(link.direction), link.mechanism, link.purview)

        a_by = {key(link): link for link in self.causal_links}
        b_by = {key(link): link for link in other.causal_links}
        changes: list[Change] = []
        changes.extend(
            Change("link_lost", k, a_value=a_by[k].alpha)
            for k in a_by.keys() - b_by.keys()
        )
        changes.extend(
            Change("link_gained", k, b_value=b_by[k].alpha)
            for k in b_by.keys() - a_by.keys()
        )
        changes.extend(
            Change("link_changed", k, a_by[k].alpha, b_by[k].alpha)
            for k in a_by.keys() & b_by.keys()
            if not utils.eq(a_by[k].alpha, b_by[k].alpha)
        )
        return ResultDiff(
            subject=f"ΔΣα ({len(self)} → {len(other)} links)",
            level="system",
            delta_phi=PyPhiFloat(float(other._sum_alpha) - float(self._sum_alpha)),
            mip_changed=False,
            changes=tuple(changes),
            config_diff={},
            substrate_note=None,
        )

    def to_json(self):
        return {"causal_links": tuple(self)}

    @classmethod
    def from_json(cls, dct):
        return cls(dct["causal_links"])


class DirectedAccount(Account):
    """The set of |CausalLinks| with |alpha > 0| for one direction of a
    transition.
    """


# TODO(slipperyhank): Check if we do the same, i.e. take the bigger system, or
# take the smaller?
class AcSystemIrreducibilityAnalysis(Displayable, cmp.Orderable):
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

    alpha: float  # Override parent to allow None during init

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
        reasons=None,
    ):
        self.alpha = alpha  # type: ignore[assignment]
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
        self.reasons = reasons or []
        # ConfigSnapshot of the layered config at construction time.
        # Lazy-snapshot if None: callers that don't pass one still get a
        # recorded config (matching SystemIrreducibilityAnalysis).
        if config is None:
            from pyphi.conf import config as _global

            config = _global.snapshot()
        self.config = config

    def _system_label(self) -> str | None:
        node_indices = self.node_indices
        node_labels = self.node_labels
        if node_labels is not None and node_indices is not None:
            return ",".join(
                str(label) for label in node_labels.coerce_to_labels(node_indices)
            )
        if node_indices is not None:
            return ",".join(str(i) for i in node_indices)
        return None

    def _describe(self, verbosity: int) -> Description:  # noqa: ARG002
        cls = type(self).__name__
        account = self.account
        num_links = len(account) if account is not None else None
        sum_alpha = sum(link.alpha for link in account) if account is not None else None
        partition_str = (
            concise_partition(self.partition) if self.partition is not None else None
        )
        before_str = (
            fmt.state(self.before_state) if self.before_state is not None else None
        )
        after_str = fmt.state(self.after_state) if self.after_state is not None else None
        return Description(
            title=cls,
            sections=(
                Section(
                    rows=(
                        Row("α", self.alpha),  # noqa: RUF001
                        Row(
                            "Direction",
                            str(self.direction) if self.direction is not None else None,
                            tone=tone_of(self.direction),
                        ),
                        Row("System", self._system_label()),
                        Row("Before state", before_str),
                        Row("After state", after_str),
                        Row("Partition", partition_str),
                        Row("Causal links", num_links),
                        Row("Σα", sum_alpha),
                    ),
                ),
            ),
            compact=f"{cls}(α={format_value(self.alpha)})",  # noqa: RUF001
        )

    def explain(self) -> Explanation:
        """A typed account of why this transition's |big_alpha| came out as it
        did. A runner-up / alpha-gap is not retained for actual causation."""
        findings = [
            Finding(kind="null_result", label="Null result", value=reason)
            for reason in (self.reasons or [])
        ]
        # A null short-circuit's partition is the trivial default, not a MIP;
        # only a computed result has a meaningful winning partition.
        if not self.reasons and self.partition is not None:
            findings.append(
                Finding(
                    kind="winning_partition",
                    label="Partition",
                    value=concise_partition(self.partition),
                )
            )
        return Explanation(
            subject=f"α = {format_value(self.alpha)}",  # noqa: RUF001
            level="system",
            findings=tuple(findings),
        )

    def diff(self, other) -> ResultDiff:
        """Structured delta from this analysis to ``other`` (``a.diff(b)``)."""
        if not isinstance(other, AcSystemIrreducibilityAnalysis):
            raise TypeError(
                f"cannot diff {type(self).__name__} against {type(other).__name__}"
            )
        common = _diff_common(self, other)
        return ResultDiff(
            subject=f"Δα = {format_value(common['delta_phi'])}",
            level="system",
            delta_phi=common["delta_phi"],
            mip_changed=common["mip_changed"],
            config_diff=common["config_diff"],
            substrate_note=common["substrate_note"],
        )

    def is_orderable_with(self, other: object) -> bool:
        return isinstance(other, AcSystemIrreducibilityAnalysis) and (
            self.direction == other.direction
        )

    # TODO: shouldn't the minimal irreducible account be chosen?
    def order_by(self):
        return [self.alpha, self.size]

    def __eq__(self, other: object) -> bool:  # noqa: PLR0911
        if not isinstance(other, AcSystemIrreducibilityAnalysis):
            return NotImplemented
        if self.direction != other.direction:
            return False
        if self.account != other.account:
            return False
        if self.partitioned_account != other.partitioned_account:
            return False
        if self.partition != other.partition:
            return False
        if self.before_state != other.before_state:
            return False
        if self.after_state != other.after_state:
            return False
        if self.size != other.size:
            return False
        if self.node_indices != other.node_indices:
            return False
        if self.cause_indices != other.cause_indices:
            return False
        if self.effect_indices != other.effect_indices:
            return False
        if self.node_labels != other.node_labels:
            return False
        return utils.eq(self.alpha, other.alpha)

    def __bool__(self):
        """An |AcSystemIrreducibilityAnalysis| is ``True`` if it has
        |big_alpha > 0|.
        """
        return greater_than_zero(self.alpha)

    def __hash__(self) -> int:
        return hash(
            (
                self.direction,
                self.account,
                self.partitioned_account,
                self.partition,
                self.before_state,
                self.after_state,
                self.size,
                self.node_indices,
                self.cause_indices,
                self.effect_indices,
                self.node_labels,
            )
        )

    def to_json(self):
        attrs = (
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
        )
        return {attr: getattr(self, attr) for attr in attrs}


def _null_ac_sia(transition, direction, alpha=0.0, reasons=None):
    """Return an |AcSystemIrreducibilityAnalysis| with zero |big_alpha| and
    empty accounts. ``reasons`` records why (a list of
    :class:`~pyphi.models.explanation.NullResultReason`).
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
        reasons=reasons,
    )
