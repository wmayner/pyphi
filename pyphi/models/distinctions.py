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
from pyphi.serializable import Serializable

from . import cmp
from .pandas import ToPandasMixin
from .pandas import records_to_frame
from .state_specification import SystemStateSpecification

_DISTINCTION_COLUMNS = [
    "phi",
    "mechanism",
    "mechanism_state",
    "cause_purview",
    "cause_state",
    "effect_purview",
    "effect_state",
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
        getattr(d, "cause_purview_label", None) or str(getattr(d, "cause_purview", "")),
        getattr(d, "effect_purview_label", None)
        or str(getattr(d, "effect_purview", "")),
    )


class Distinctions(Displayable, cmp.Orderable, Sequence, ToPandasMixin, Serializable):
    """Base class for a collection of distinctions.

    Holds the read-only operations shared by :class:`UnresolvedDistinctions`
    and :class:`ResolvedDistinctions`. Instantiable directly for the rare
    cases where the resolution status is genuinely unknown (e.g. deserializing
    a stored fixture); construct one of the marker subtypes instead so that
    passing the result to a function requiring a specific resolution status is
    checked at the type level.
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

    def filter(self, predicate) -> Distinctions:
        """Return the distinctions satisfying ``predicate``.

        Preserves the runtime subtype, so filtering a
        :class:`ResolvedDistinctions` yields a :class:`ResolvedDistinctions`.
        """
        return type(self)(d for d in self if predicate(d))

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
        """The φ values of each concept."""
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
        """Resolve each distinction's tied readings against ``system_state``,
        per the Albantakis et al. 2023 S1 Text.

        Congruence with the system's specified cause-effect state is a
        requirement: non-congruent readings are dropped, and a distinction
        with no congruent reading in either direction is excluded from the
        structure. Ties among congruent readings are resolved jointly
        across distinctions by the Composition appeal — selecting the
        combination of readings that maximizes the structure integrated
        information Φ (Σφ_d is invariant across readings, so the
        comparison is over the analytical Σφ_r) — with residual Φ-ties
        closed by the Determinism convention (lexicographic purviews and
        states). Beyond :data:`_JOINT_READING_BOUND` combinations, a
        greedy per-distinction pass approximates the joint maximum and a
        :class:`~pyphi.warnings.PyPhiWarning` is emitted.

        Returns a :class:`ResolvedDistinctions` regardless of the input
        subtype — calling on an already-resolved bag is well-defined and
        just refilters.
        """
        return ResolvedDistinctions(
            _resolve_congruence_jointly(list(self), system_state)
        )


# Exhaustive joint resolution is exact up to this many reading
# combinations; beyond it, a greedy per-distinction pass approximates the
# joint maximum. Ties are symmetry artifacts of small toy models (S1: they
# are "unlikely to occur in realistic systems"), so the bound is far above
# anything seen in practice.
_JOINT_READING_BOUND = 4096


def _reading_key(reading: Any) -> tuple:
    """Deterministic lexicographic key for a reading (Determinism convention)."""
    spec = getattr(reading, "specified_state", None)
    state = getattr(spec, "state", None) if spec is not None else None
    return (tuple(reading.purview), repr(state))


def _congruent_reading_options(
    distinction: Any, system_state: SystemStateSpecification
) -> tuple[list, list] | None:
    """The distinction's congruent readings per direction, in canonical order.

    Returns ``None`` when either direction has no congruent reading (the
    distinction is excluded from the structure). Propagates the
    purview-selection margin from the original MICE winner to tied peers
    that never carried it.
    """
    from pyphi.resolve_ties import congruent_distinction_readings

    options = {}
    for direction in Direction.both():
        mice = distinction.mice(direction)
        if mice is None:
            return None
        readings = congruent_distinction_readings(
            mice.state_ties, mice.purview_ties, system_state[direction]
        )
        if not readings:
            return None
        for reading in readings:
            # The purview-selection margin describes the purview choice and
            # is shared across the tie set at the winning purview's φ;
            # congruence may select a tied peer that never carried the
            # winner's margin, so propagate it. The partition and state
            # margins belong to the peer's own RIA and are already correct.
            if reading is not mice and reading.purview_margin is None:
                reading.purview_margin = mice.purview_margin
        options[direction] = sorted(readings, key=_reading_key)
    return options[Direction.CAUSE], options[Direction.EFFECT]


def _resolve_congruence_jointly(
    distinctions: Sequence, system_state: SystemStateSpecification
) -> list:
    """Select each distinction's reading per the S1 Composition appeal.

    Ties in φ_d (and thus in the cause-effect state of a distinction) "may
    be resolved at the level of the cause-effect structure, by selecting
    the [reading] that maximizes the system's structure integrated
    information Φ" (Albantakis et al. 2023 S1 Text). A reading's relation
    support depends on the other distinctions' readings, so the
    maximization is joint: over the product of the multi-reading
    distinctions' congruent (cause, effect) pairs, scored by the
    analytical Σφ_r of the resulting structure (Σφ_d is invariant across
    readings). Residual Φ-ties resolve by the Determinism convention.
    """
    import itertools
    import math
    import warnings

    entries = []
    for distinction in distinctions:
        options = _congruent_reading_options(distinction, system_state)
        if options is None:
            continue  # no congruent reading: excluded from the structure
        causes, effects = options
        entries.append((distinction, [(c, e) for c in causes for e in effects]))
    if not entries:
        return []

    def build(choice_list: list) -> list:
        return [
            type(distinction)(
                mechanism=distinction.mechanism, cause=cause, effect=effect
            )
            for (distinction, _), (cause, effect) in zip(
                entries, choice_list, strict=True
            )
        ]

    choices = [pairs[0] for _, pairs in entries]
    variable = [i for i, (_, pairs) in enumerate(entries) if len(pairs) > 1]
    if not variable:
        return build(choices)

    from pyphi import numerics
    from pyphi.relations import AnalyticalRelations

    def score(choice_list: list) -> float:
        resolved = ResolvedDistinctions(build(choice_list))
        return float(AnalyticalRelations(resolved).sum_phi())

    n_combos = math.prod(len(entries[i][1]) for i in variable)
    if n_combos > _JOINT_READING_BOUND:
        from pyphi.warnings import PyPhiWarning

        warnings.warn(
            f"{n_combos} tied reading combinations exceed the exact joint "
            f"resolution bound ({_JOINT_READING_BOUND}); resolving greedily "
            f"per distinction, which may not attain the Φ-maximal structure",
            PyPhiWarning,
            stacklevel=4,
        )
        for i in variable:
            best_pair, best_score = None, None
            for pair in entries[i][1]:
                candidate = list(choices)
                candidate[i] = pair
                value = score(candidate)
                # numerics: tolerant — selection among readings.
                if best_score is None or (
                    value > best_score and not numerics.eq(value, best_score)
                ):
                    best_pair, best_score = pair, value
            choices[i] = best_pair
        return build(choices)

    def combo_key(combo: tuple) -> tuple:
        return tuple(
            (_reading_key(cause), _reading_key(effect)) for cause, effect in combo
        )

    best_score = None
    tied: list[tuple] = []
    for combo in itertools.product(*(entries[i][1] for i in variable)):
        candidate = list(choices)
        for i, pair in zip(variable, combo, strict=True):
            candidate[i] = pair
        value = score(candidate)
        # numerics: tolerant — Φ-tier membership is a selection.
        if best_score is None or (
            value > best_score and not numerics.eq(value, best_score)
        ):
            best_score, tied = value, [combo]
        elif numerics.eq(value, best_score):
            tied.append(combo)
    winner = min(tied, key=combo_key)
    for i, pair in zip(variable, winner, strict=True):
        choices[i] = pair
    return build(choices)


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
