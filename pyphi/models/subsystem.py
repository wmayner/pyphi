# models/subsystem.py
"""Subsystem-level objects."""

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from toolz import concat

from pyphi.direction import Direction

from .. import utils
from ..conf import fallback
from . import cmp, fmt
from .mechanism import Concept, StateSpecification
from .pandas import ToDictMixin, ToPandasMixin

_sia_attributes = ["phi", "ces", "partitioned_ces", "subsystem", "cut_subsystem"]


@dataclass(frozen=True)
class SystemStateSpecification(ToDictMixin, ToPandasMixin):
    cause: StateSpecification
    effect: StateSpecification

    def __getitem__(self, direction: Direction) -> StateSpecification:
        if direction == Direction.CAUSE:
            return self.cause
        elif direction == Direction.EFFECT:
            return self.effect
        raise KeyError("Invalid direction")

    def _repr_columns(self, prefix=""):
        cols = []
        # TODO(4.0) create NullStateSpecification and use that instead of None
        if self.cause is not None:
            cols.extend(self.cause._repr_columns(prefix))
        else:
            cols.append((f"{prefix}{Direction.CAUSE}", None))
        if self.effect is not None:
            cols.extend(self.effect._repr_columns(prefix))
        else:
            cols.append((f"{prefix}{Direction.EFFECT}", None))
        return cols

    def __repr__(self):
        body = "\n".join(fmt.align_columns(self._repr_columns()))
        body = fmt.header("Specified System State", body, under_char=fmt.HEADER_BAR_3)
        return fmt.box(fmt.center(body))

    def __hash__(self):
        return hash((self.cause, self.effect))

    def to_json(self):
        return self.__dict__


def _concept_sort_key(concept):
    return (len(concept.mechanism), concept.mechanism)


def defaultdict_set():
    return defaultdict(set)


def _purview_inclusion(distinction_attr, distinctions, min_order, max_order):
    purview_inclusion_by_order = defaultdict(defaultdict_set)
    for distinction in distinctions:
        for subset in map(
            frozenset,
            utils.powerset(
                getattr(distinction, distinction_attr),
                nonempty=True,
                min_size=min_order,
                max_size=max_order,
            ),
        ):
            purview_inclusion_by_order[len(subset)][subset].add(distinction)
    return purview_inclusion_by_order


def _find_multiplicities(func, distinctions):
    """Return a mapping from purviews to multiplicities of the values of ``func``."""
    multiplicities = defaultdict_set()
    for d in distinctions:
        for direction in Direction.both():
            multiplicities[d.purview(direction)].add(func(d.mice(direction)))
    return multiplicities


def _get_mechanism(mice):
    return mice.mechanism


def _get_state(mice):
    return mice.specified_state.state


class CauseEffectStructure(cmp.Orderable, Sequence, ToPandasMixin):
    """A collection of concepts."""

    def __init__(self, concepts=(), subsystem=None, resolved_congruence=False):
        # Normalize the order of concepts
        # TODO(4.0) convert to set?
        self.concepts = tuple(sorted(concepts, key=_concept_sort_key))
        # self.subsystem = subsystem
        self._specifiers = None
        self._purview_inclusion_by_order = defaultdict(defaultdict_set)
        # Flag to indicate whether distinctions have been filtered according to
        # congruence with a SIA specified state
        # TODO(4.0) use a subclass instead, as with MICE?
        self._resolved_congruence = resolved_congruence
        self._sum_phi = None

    def __len__(self):
        return len(self.concepts)

    def __iter__(self):
        return iter(self.concepts)

    def __getitem__(self, value):
        if isinstance(value, slice):
            return type(self)(self.concepts[value])
        return self.concepts[value]

    def __repr__(self):
        # TODO(4.0) remove dependence on subsystem & time
        return fmt.make_repr(self, ["concepts", "subsystem"])

    def __str__(self):
        return fmt.fmt_ces(self)

    @cmp.sametype
    def __eq__(self, other):
        return self.concepts == other.concepts

    def __hash__(self):
        return hash(self.concepts)

    def order_by(self):
        return [self.concepts]

    def to_json(self):
        return {"concepts": self.concepts}

    @property
    def flat(self):
        """An iterator over causes and effects."""
        return concat([concept.cause, concept.effect] for concept in self)

    def flatten(self):
        """Return this as a FlatCauseEffectStructure."""
        return FlatCauseEffectStructure(self)

    def unflatten(self):
        """Return self."""
        # No-op; already unflattened
        return self

    def sum_phi(self):
        if self._sum_phi is None:
            self._sum_phi = sum(self.phis)
        return self._sum_phi

    @property
    def phis(self):
        """The |small_phi| values of each concept."""
        for concept in self:
            yield concept.phi

    @property
    def mechanisms(self):
        """The mechanism of each concept."""
        for concept in self:
            yield concept.mechanism

    def _purviews(self, direction):
        for concept in self:
            yield concept.purview(direction)

    def purviews(self, direction):
        """Return the purview of each concept in the given direction."""
        if isinstance(direction, Iterable):
            for _direction in direction:
                yield from self._purviews(_direction)
        else:
            yield from self._purviews(direction)

    @property
    def labeled_mechanisms(self):
        """The labeled mechanism of each concept."""
        # TODO(4.0) remove dependence on subsystem
        label = self.subsystem.node_labels.indices2labels
        return tuple(list(label(mechanism)) for mechanism in self.mechanisms)

    def purview_inclusion_of_intersection(self, min_order, max_order):
        return _purview_inclusion(
            "purview_intersection",
            distinctions=self,
            min_order=min_order,
            max_order=max_order,
        )

    def _purview_inclusion_of_union(self, min_order, max_order):
        return _purview_inclusion(
            "purview_union", distinctions=self, min_order=min_order, max_order=max_order
        )

    def purview_inclusion(self, max_order=None):
        """Return a mapping:

        {order: {frozenset[Unit]: {distinctions whose cause/effect purview
                                   union includes those Units}}}
        """
        if max_order is None or max_order not in self._purview_inclusion_by_order:
            self._purview_inclusion_by_order.update(
                # NOTE: We use the union of the cause/effect purviews
                self._purview_inclusion_of_union(
                    min_order=max(self._purview_inclusion_by_order, default=0) + 1,
                    max_order=max_order,
                )
            )
        max_order = fallback(max_order, float("inf"))
        for order, mapping in self._purview_inclusion_by_order.items():
            if order <= max_order:
                yield from mapping.items()

    def resolve_congruence(self, system_state: SystemStateSpecification):
        """Filter out incongruent distinctions."""
        # TODO(4.0) parallelize
        return type(self)(
            filter(
                lambda d: d is not None,
                (distinction.resolve_congruence(system_state) for distinction in self),
            ),
            resolved_congruence=True,
        )

    def mechanism_multiplicities(self):
        return _find_multiplicities(_get_mechanism, self)

    def state_multiplicities(self):
        return _find_multiplicities(_get_state, self)

    @property
    def resolved_congruence(self):
        return self._resolved_congruence


def flatten_distinctions(distinctions):
    return concat(
        (
            [distinction.cause, distinction.effect]
            if isinstance(distinction, Concept)
            else [distinction]
        )
        for distinction in distinctions
    )


class FlatCauseEffectStructure(CauseEffectStructure):
    """A collection of maximally-irreducible components in either causal
    direction."""

    def __init__(self, concepts=(), subsystem=None):
        if isinstance(concepts, CauseEffectStructure):
            subsystem = concepts.subsystem
        if not isinstance(concepts, FlatCauseEffectStructure):
            _concepts = flatten_distinctions(concepts)
        else:
            _concepts = iter(concepts)
        super().__init__(concepts=_concepts, subsystem=subsystem)

    def __str__(self):
        return fmt.fmt_ces(self, title="Flat cause-effect structure")

    @property
    def purviews(self):
        """The purview of each component."""
        for component in self:
            yield component.purview

    @property
    def specified_purviews(self):
        """The set of unique purviews specified by this CES."""
        return set(self.purviews)

    def specifiers(self, purview):
        """The components that specify the given purview."""
        purview = tuple(purview)
        try:
            return self._specifiers[purview]
        except TypeError:
            self._specifiers = defaultdict(list)
            for component in self:
                self._specifiers[component.purview].append(component)
            return self._specifiers[purview]

    def maximum_specifier(self, purview):
        """Return the components that maximally specify the given purview."""
        return max(self.specifiers(purview))

    def maximum_specifiers(self):
        """Return a mapping from each purview to its maximum specifier."""
        return {purview: self.maximum_specifier(purview) for purview in self.purviews}

    @property
    def flat(self):
        # No-op; already flat
        return self

    def flatten(self):
        # No-op; already flat
        return self

    def unflatten(self):
        mechanism_to_mice = defaultdict(dict)
        for mice in self:
            mechanism_to_mice[mice.mechanism][mice.direction] = mice
        return CauseEffectStructure(
            [
                Concept(
                    mechanism=mechanism,
                    cause=mice[Direction.CAUSE],
                    effect=mice[Direction.EFFECT],
                )
                for mechanism, mice in mechanism_to_mice.items()
            ],
        )

    def _purview_inclusion_of_union(self, min_order, max_order):
        return _purview_inclusion(
            "purview_units", distinctions=self, min_order=min_order, max_order=max_order
        )


class SystemIrreducibilityAnalysis(cmp.OrderableByPhi):
    """An analysis of system irreducibility (|big_phi|).

    Contains the |big_phi| value of the |Subsystem|, the cause-effect
    structure, and all the intermediate results obtained in the course of
    computing them.

    These can be compared with the built-in Python comparison operators (``<``,
    ``>``, etc.). First, |big_phi| values are compared. Then, if these are
    equal up to |PRECISION|, the one with the larger subsystem is greater.

    Attributes:
        phi (float): The |big_phi| value for the subsystem when taken against
            this analysis, *i.e.* the difference between the cause-effect
            structure and the partitioned cause-effect structure for this
            analysis.
        ces (CauseEffectStructure): The cause-effect structure of
            the whole subsystem.
        partitioned_ces (CauseEffectStructure): The cause-effect structure when
            the subsystem is cut.
        subsystem (Subsystem): The subsystem this analysis was calculated for.
        cut_subsystem (Subsystem): The subsystem with the minimal cut applied.
        time (float): The number of seconds it took to calculate.
    """

    def __init__(
        self,
        phi=None,
        ces=None,
        partitioned_ces=None,
        subsystem=None,
        cut_subsystem=None,
    ):
        self.phi = phi
        self.ces = ces
        self.partitioned_ces = partitioned_ces
        self.subsystem = subsystem
        self.cut_subsystem = cut_subsystem

    def __repr__(self):
        return fmt.make_repr(self, _sia_attributes)

    def __str__(self, ces=True):
        return fmt.fmt_sia(self, ces=ces)

    def print(self, ces=True):
        """Print this |SystemIrreducibilityAnalysis|, optionally without
        cause-effect structures.
        """
        print(self.__str__(ces=ces))

    @property
    def cut(self):
        """The unidirectional cut that makes the least difference to the
        subsystem.
        """
        return self.cut_subsystem.cut

    @property
    def network(self):
        """The network the subsystem belongs to."""
        return self.subsystem.network

    unorderable_unless_eq = ["network"]

    def __eq__(self, other):
        return cmp.general_eq(self, other, _sia_attributes)

    def __bool__(self):
        """A |SystemIrreducibilityAnalysis| is ``True`` if it has
        |big_phi > 0|.
        """
        return not utils.eq(self.phi, 0)

    def __hash__(self):
        return hash(
            (
                self.phi,
                self.ces,
                self.partitioned_ces,
                self.subsystem,
                self.cut_subsystem,
            )
        )

    def to_json(self):
        """Return a JSON-serializable representation."""
        return {
            attr: getattr(self, attr) for attr in _sia_attributes + ["small_phi_time"]
        }

    @classmethod
    def from_json(cls, dct):
        del dct["small_phi_time"]
        return cls(**dct)


def _null_ces(subsystem):
    """Return an empty CES."""
    return CauseEffectStructure((), subsystem=subsystem)


def _null_sia(subsystem, phi=0.0):
    """Return a |SystemIrreducibilityAnalysis| with zero |big_phi| and empty
    cause-effect structures.

    This is the analysis result for a reducible subsystem.
    """
    return SystemIrreducibilityAnalysis(
        subsystem=subsystem,
        cut_subsystem=subsystem,
        phi=phi,
        ces=_null_ces(subsystem),
        partitioned_ces=_null_ces(subsystem),
    )
