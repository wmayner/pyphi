# models/distinctions.py
"""``Distinctions`` — a collection of distinctions (concepts in IIT 3.0).

In IIT 4.0 paper terminology, the cause-effect structure of any candidate
system is *distinctions + relations* — that compound object lives in
:mod:`pyphi.models.ces` as :class:`CauseEffectStructure`. This module
holds just the bag-of-distinctions side.

The collection comes in two concrete subtypes that encode whether
ties on the per-distinction specified states have been disambiguated:

- :class:`UnresolvedDistinctions` — the default form returned by raw
  computation. Per-distinction specified states may still be tied, and
  no SIA-level ``system_state`` has been used to pick among them.
- :class:`ResolvedDistinctions` — the form after
  :meth:`UnresolvedDistinctions.resolve_congruence` has filtered each
  distinction's tied states down to the ones congruent with a SIA
  ``system_state``. Functions like :func:`pyphi.relations.relations` and
  :class:`~pyphi.models.ces.CauseEffectStructure` accept only this
  subtype, so passing unresolved distinctions is a static type error.

The base :class:`Distinctions` class is abstract — instantiation must
choose a subtype. IIT 3.0 has no per-distinction ties, so its computation
emits :class:`ResolvedDistinctions` directly (vacuously resolved); IIT
4.0 emits :class:`UnresolvedDistinctions` and resolves via the SIA's
``system_state`` later in the pipeline.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from collections.abc import Sequence
from itertools import chain
from typing import Any

from pyphi import utils
from pyphi.conf import fallback
from pyphi.direction import Direction
from pyphi.display import Description
from pyphi.display import Displayable
from pyphi.display import Row
from pyphi.display import Section
from pyphi.display.numbers import format_value
from pyphi.display.tables import capped_table

from . import cmp
from .pandas import ToPandasMixin
from .pandas import records_to_frame
from .state_specification import SystemStateSpecification

_DISTINCTION_COLUMNS = [
    "phi",
    "mechanism",
    "mechanism_state",
    "cause_purview",
    "effect_purview",
]


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


DISTINCTION_HEADERS = ("Mechanism", "φ_d", "Cause purview", "Effect purview")
DISTINCTION_HEADER_TONES = (None, None, "cause", "effect")


def distinction_table_row(d: Any) -> tuple[Any, ...]:
    """Display-table row for a distinction: mechanism, φ_d, cause/effect purviews."""
    return (
        getattr(d, "mechanism_label", None) or str(getattr(d, "mechanism", "")),
        getattr(d, "phi", None),
        str(getattr(d, "cause_purview", "")),
        str(getattr(d, "effect_purview", "")),
    )


class Distinctions(Displayable, cmp.Orderable, Sequence, ToPandasMixin):
    """Base class for a collection of distinctions.

    Holds the read-only operations shared by :class:`UnresolvedDistinctions`
    and :class:`ResolvedDistinctions`. Instantiable directly for the
    rare cases where the resolution status is genuinely unknown (e.g.,
    deserializing a pre-P11.9 JSON fixture); new code should construct
    one of the marker subtypes so passing the result to a function that
    requires a specific resolution status is checked at the type level.
    """

    def __init__(self, concepts: Iterable = ()):
        # Normalize the order of concepts
        self.concepts = tuple(sorted(concepts, key=_concept_sort_key))
        self._specifiers = None
        self._purview_inclusion_by_order = defaultdict(defaultdict_set)
        self._sum_phi = None

    def __len__(self):
        return len(self.concepts)

    def __iter__(self):
        return iter(self.concepts)

    def __getitem__(self, value):
        if isinstance(value, slice):
            return type(self)(self.concepts[value])
        return self.concepts[value]

    def _describe(self, verbosity: int) -> Description:  # noqa: ARG002
        cls = type(self).__name__
        num_d = len(self)
        sum_phi_d = self.sum_phi()
        table = capped_table(
            DISTINCTION_HEADERS,
            self,
            distinction_table_row,
            total=num_d,
            header_tones=DISTINCTION_HEADER_TONES,
        )
        return Description(
            title=cls,
            sections=(
                Section(rows=(Row("Distinctions", num_d), Row("Σφ_d", sum_phi_d))),
                Section(label="Distinctions", body=(table,)),
            ),
            compact=f"{cls}({num_d} distinctions, Σφ_d={format_value(sum_phi_d)})",
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Distinctions):
            return NotImplemented
        return self.concepts == other.concepts

    def __hash__(self):
        return hash(self.concepts)

    def order_by(self):
        return [self.concepts]

    def _to_pandas(self):
        rows = [concept._pandas_record() for concept in self.concepts]
        return records_to_frame(rows, index="mechanism", columns=_DISTINCTION_COLUMNS)

    @property
    def flat(self):
        """An iterator over causes and effects (one ``MICE`` per direction
        per concept), for callers that want to operate at the MICE level
        rather than the concept level.
        """
        return chain.from_iterable([concept.cause, concept.effect] for concept in self)

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
        # Get node_labels from the first concept if available
        if (
            self.concepts
            and hasattr(self.concepts[0], "node_labels")
            and self.concepts[0].node_labels is not None
        ):
            label = self.concepts[0].node_labels.indices2labels
            return tuple(list(label(mechanism)) for mechanism in self.mechanisms)
        # Fallback to numeric indices as strings
        return tuple(list(map(str, mechanism)) for mechanism in self.mechanisms)

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

    def mechanism_multiplicities(self):
        return _find_multiplicities(_get_mechanism, self)

    def state_multiplicities(self):
        return _find_multiplicities(_get_state, self)

    def resolve_congruence(
        self, system_state: SystemStateSpecification
    ) -> ResolvedDistinctions:
        """Filter each distinction's tied states down to the ones
        congruent with ``system_state``, dropping distinctions that have
        no congruent reading. Returns a :class:`ResolvedDistinctions`
        regardless of the input subtype — calling on an already-resolved
        bag is well-defined and just refilters.
        """
        return ResolvedDistinctions(
            filter(
                lambda d: d is not None,
                (distinction.resolve_congruence(system_state) for distinction in self),
            )
        )


class UnresolvedDistinctions(Distinctions):
    """Distinctions whose per-distinction tied states have not been disambiguated.

    Returned by raw computation paths that don't carry a SIA
    ``system_state``. Cannot be passed to functions that require a
    canonical specified state per distinction (relations,
    CauseEffectStructure construction); call :meth:`resolve_congruence`
    first.
    """


class ResolvedDistinctions(Distinctions):
    """Distinctions whose tied states have been disambiguated.

    Either constructed directly when no resolution is needed (IIT 3.0,
    where there are no tied states), or returned by
    :meth:`Distinctions.resolve_congruence` after the SIA determines a
    system-level ``system_state``. Required for
    :func:`pyphi.relations.relations` and the ``distinctions`` field of
    :class:`~pyphi.models.ces.CauseEffectStructure`.
    """


def _null_ces(system=None) -> ResolvedDistinctions:  # noqa: ARG001 - retained for backward-compatible signature
    """Return an empty CES.

    The empty case is vacuously resolved — there are no tied states to
    disambiguate — so the return type is :class:`ResolvedDistinctions`,
    suitable for any downstream function that requires resolved input.
    """
    return ResolvedDistinctions(())
