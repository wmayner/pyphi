# relations.py
"""Implements the formalism for computing relations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from functools import cached_property
from functools import total_ordering
from itertools import product
from typing import TYPE_CHECKING
from typing import Any

from tqdm.auto import tqdm

from . import combinatorics
from . import conf
from . import utils
from .conf import config
from .conf import fallback
from .data_structures import PyPhiFloat
from .direction import Direction
from .display import Description
from .display import Displayable
from .display import Row
from .display import Section
from .display import Table
from .display.numbers import format_value
from .display.tables import capped_table
from .models import cmp
from .models.distinctions import ResolvedDistinctions
from .parallel import MapReduce
from .registry import Registry

if TYPE_CHECKING:
    from graphillion import setset  # noqa: F401

    from .formalism.iit4 import Distinction  # type: ignore[attr-defined]


class RelationFace(Displayable, frozenset):
    """A set of (potentially) related causes/effects."""

    phi: float  # Set in __new__

    def __new__(cls, *args, phi=None):
        self = super().__new__(cls, *args)
        if phi is None:
            raise ValueError("phi keyword argument is required")

        # Preserve DistanceResult type if possible, otherwise convert to PyPhiFloat
        from pyphi.data_structures.pyphi_float import PyPhiFloat
        from pyphi.measures.distribution import DistanceResult

        if isinstance(phi, DistanceResult):
            self.phi = phi  # type: ignore[misc]  # frozenset is immutable but we set this in __new__
        else:
            self.phi = PyPhiFloat(phi)  # type: ignore[misc]  # frozenset is immutable but we set this in __new__
        return self

    @total_ordering  # type: ignore[arg-type]  # total_ordering expects a class not instance
    def __lt__(self, other):
        return self.phi < other.phi  # type: ignore[attr-defined]  # phi is set in __new__

    @cached_property
    def overlap(self):
        """The set of elements that are in the purview of every relatum."""
        return set.intersection(*map(set, self.relata_purviews))

    @cached_property
    def congruent_overlap(self):
        """Return the congruent overlap(s) among the relata.

        These are the common purview elements among the relata whose specified
        states are consistent; that is, the largest subset of the union of the
        purviews such that each relatum specifies the same state for each
        element.
        """
        return set.intersection(*self.relata_units)

    # Alias
    @property
    def purview(self):
        """The purview of the relation face. Alias for ``congruent_overlap``."""
        return self.congruent_overlap

    @property
    def relata_units(self):
        """The Units in the purview of each cause/effect in this face."""
        return (set(relatum.purview_units) for relatum in self)

    @property
    def relata_purviews(self):
        """The purview of each cause/effect in this face."""
        return (relatum.purview for relatum in self)

    @property
    def distinctions(self):
        """The distinctions whose causes/effects are in this face."""
        return (relatum.parent for relatum in self)

    @property
    def num_distinctions(self):
        """The number of distinctions whose causes/effects are in this face."""
        return len(set(self.distinctions))

    def __bool__(self):
        return bool(self.congruent_overlap)

    def _describe(self, verbosity: int) -> Description:
        cls = type(self).__name__
        return Description(
            title=cls,
            sections=(
                Section(
                    rows=(
                        Row("Purview", str(sorted(self.purview))),
                        Row("Relata", len(self)),
                    ),
                ),
            ),
            compact=f"{cls}(purview={sorted(self.purview)}, relata={len(self)})",
        )

    # frozenset.__repr__ takes priority in the MRO; delegate to Displayable.
    def __repr__(self) -> str:
        return Displayable.__repr__(self)

    __str__ = __repr__

    def _repr_html_(self) -> str:
        return Displayable._repr_html_(self)

    def to_json(self):
        return {"relata": list(self)}

    @classmethod
    def from_json(cls, data):
        return cls(data["relata"])


class Relation(Displayable, frozenset, cmp.OrderableByPhi):
    """A set of relation faces forming the relation among a set of distinctions."""

    @property
    def is_self_relation(self):
        return len(self) == 1

    def _faces(self):
        """Yield faces of the relation."""
        # Exclude single-relatum faces for self-relations as a special case
        if self.is_self_relation:
            direction_set = [Direction.BIDIRECTIONAL]
        else:
            direction_set = Direction.all()

        distinctions = list(self)
        for directions in product(direction_set, repeat=len(self)):
            mice = []
            for direction, distinction in zip(directions, distinctions, strict=False):
                if direction is Direction.BIDIRECTIONAL:
                    mice.extend([distinction.cause, distinction.effect])
                else:
                    mice.append(distinction.mice(direction))
            face = RelationFace(mice, phi=self.phi)
            if face:
                yield face

    @cached_property
    def faces(self):
        return frozenset(self._faces())

    @property
    def num_faces(self):
        return len(self.faces)

    @cached_property
    def purview(self):
        # Special case for self-relations
        if self.is_self_relation:
            distinction = next(iter(self))
            return distinction.cause.purview_units & distinction.effect.purview_units

        return set.intersection(*(distinction.purview_union for distinction in self))

    @cached_property
    def phi(self) -> PyPhiFloat:  # type: ignore[override]  # Overrides OrderableByPhi.phi with cached_property
        return PyPhiFloat(
            len(self.purview) * min(self.distinction_phi_per_unique_purview_unit())
        )

    def distinction_phi_per_unique_purview_unit(self):
        return (relatum.phi / len(relatum.purview_union) for relatum in self)

    def __bool__(self):
        return utils.is_positive(self.phi)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Relation):
            return NotImplemented
        return frozenset.__eq__(self, other)

    def __hash__(self) -> int:
        return frozenset.__hash__(self)

    @cached_property
    def mechanisms(self):
        return {distinction.mechanism for distinction in self}

    def _describe(self, verbosity: int) -> Description:
        cls = type(self).__name__
        return Description(
            title=cls,
            sections=(
                Section(
                    rows=(
                        Row("φ_r", self.phi),
                        Row("Purview", str(sorted(self.purview))),
                        Row("Degree", len(self)),
                        Row("Faces", self.num_faces),
                    ),
                ),
            ),
            compact=f"{cls}(φ_r={format_value(self.phi)}, degree={len(self)})",
        )

    def to_json(self):
        """Return a JSON-serializable representation."""
        return {"distinctions": list(self)}

    @classmethod
    def from_json(cls, data):
        return cls(data["distinctions"])


def all_relations(distinctions, min_degree=2, max_degree=None, **kwargs):
    """Yield causal relations among a set of distinctions."""
    # Self relations
    yield from _self_relations(distinctions)
    # Non-self relations
    combinations = _combinations_with_nonempty_congruent_overlap(
        distinctions, min_degree=min_degree, max_degree=max_degree
    )

    def worker(combination):
        return Relation(distinctions[i] for i in combination)

    pkwargs = conf.parallel_kwargs(
        config.infrastructure.parallel_relation_evaluation, **kwargs
    )
    result = MapReduce(
        worker,
        combinations,
        desc="Evaluating relations",
        **pkwargs,  # type: ignore[arg-type]  # parallel_kwargs contains MapReduce params
    ).run()
    if result is not None:
        yield from result


def _self_relations(distinctions):
    return filter(None, (Relation([distinction]) for distinction in distinctions))


def _combinations_with_nonempty_congruent_overlap(
    components, min_degree=2, max_degree=None
):
    """Return combinations of distinctions with nonempty congruent overlap.

    Arguments:
        components (Distinctions): The distinctions to find overlaps
            among.
    """
    from graphillion import setset

    # TODO(4.0) remove mapping when/if distinctions allow O(1) random access
    mapping = {component: i for i, component in enumerate(components)}
    # Use integers to avoid expensive distinction hashing
    sets = [
        list(map(mapping.get, subset))
        for _, subset in components.purview_inclusion(max_order=1)
    ]
    setset.set_universe(range(len(components)))
    return combinatorics.union_powerset_family(
        sets, min_size=min_degree, max_size=max_degree
    )


def relations_table(relations: Relations) -> Table | None:
    """Capped display table of relations (relata, ``φ_r``, degree).

    Returns ``None`` for relation sets that are not row-enumerable (e.g.
    :class:`AnalyticalRelations`). The cap
    (``config.infrastructure.repr_max_table_rows``) bounds how many rows are
    materialized, so a huge relation set is not fully realized to display.
    """
    try:
        iter(relations)  # type: ignore[arg-type]
    except TypeError:
        return None
    return capped_table(
        ("Relata (mechanisms)", "φ_r", "Degree"),
        relations,  # type: ignore[arg-type]  # iterability guarded above
        lambda r: (str(sorted(r.mechanisms)), r.phi, len(r)),
        total=relations.num_relations(),
    )


class Relations(Displayable):
    """A set of relations among distinctions."""

    def __init__(self, *args, **kwargs):
        self._num_relations_cached = None
        self._sum_phi_cached = None
        self._apportioned_sum_phi_cached = None

    def sum_phi(self):
        if self._sum_phi_cached is None:
            self._sum_phi_cached = self._sum_phi()  # type: ignore[attr-defined]  # Defined in subclass
        return self._sum_phi_cached

    def apportioned_sum_phi(self):
        if self._apportioned_sum_phi_cached is None:
            self._apportioned_sum_phi_cached = self._apportioned_sum_phi()  # type: ignore[attr-defined]  # Defined in subclass
        return self._apportioned_sum_phi_cached

    def num_relations(self):
        if self._num_relations_cached is None:
            self._num_relations_cached = self._num_relations()  # type: ignore[attr-defined]  # Defined in subclass
        return self._num_relations_cached

    def _describe(self, verbosity: int) -> Description:
        cls = type(self).__name__
        num_r = self.num_relations()
        sum_phi_r = self.sum_phi()
        table = relations_table(self)
        relations_section = (
            (Section(label="Relations", body=(table,)),) if table is not None else ()
        )
        return Description(
            title=cls,
            sections=(
                Section(
                    rows=(
                        Row("Relations", num_r),
                        Row("Σφ_r", sum_phi_r),
                    ),
                ),
                *relations_section,
            ),
            compact=f"{cls}({num_r} relations, Σφ_r={format_value(sum_phi_r)})",
        )

    def to_json(self):
        """Return a JSON-serializable representation."""
        return {"relations": list(self)}  # type: ignore[arg-type]  # Self needs __iter__ in subclass

    @classmethod
    def from_json(cls, data):
        return cls(data["relations"])


class NullRelations(Relations):
    """An empty set of relations specified by a substrate whose formalism
    does not define relations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def __iter__(self):
        return iter(())

    def _sum_phi(self):
        return 0

    def _apportioned_sum_phi(self):
        return 0

    def _num_relations(self):
        return 0

    def __len__(self):
        return 0

    def to_json(self):
        return {"relations": []}

    @classmethod
    def from_json(cls, data):  # noqa: ARG003
        return cls()


class ConcreteRelations(frozenset, Relations):
    def _sum_phi(self):
        return sum(relation.phi for relation in self)

    def _apportioned_sum_phi(self):
        return sum(relation.phi / len(relation) for relation in self)

    def _num_relations(self):
        return len(self)

    # frozenset.__repr__ and __str__ take priority over Displayable in the MRO;
    # delegate explicitly so the unified display card is used instead.
    def __repr__(self) -> str:
        return Displayable.__repr__(self)

    __str__ = __repr__

    def _repr_html_(self) -> str:
        return Displayable._repr_html_(self)

    @cached_property
    def faces_by_degree(self):
        """Return a dictionary mapping degree to relation faces of that degree."""
        faces = defaultdict(list)
        for relation in tqdm(
            self,
            desc="Grouping relation faces by degree",
            leave=False,
        ):
            for face in relation.faces:
                faces[len(face)].append(face)
        return dict(faces)


class AnalyticalRelations(Relations):
    def __init__(self, distinctions):
        self.distinctions = distinctions
        super().__init__()

    @cached_property
    def self_relations(self):
        return tuple(_self_relations(self.distinctions))

    def _sum_phi(self):
        sum_phi = 0
        # Sum of phi excluding self-relations
        for _, overlapping_distinctions in self.distinctions.purview_inclusion(
            max_order=1
        ):
            sum_phi += combinatorics.sum_of_minimum_among_subsets(
                [
                    distinction.phi / len(distinction.purview_union)
                    for distinction in overlapping_distinctions
                ]
            )
        # Count self-relations
        sum_phi += sum(relation.phi for relation in self.self_relations)
        return sum_phi

    def _apportioned_sum_phi(self):
        apportioned = 0
        # Apportioned sum (Σ φ_r / |r|) excluding self-relations
        for _, overlapping_distinctions in self.distinctions.purview_inclusion(
            max_order=1
        ):
            apportioned += combinatorics.sum_of_minimum_over_size_among_subsets(
                [
                    distinction.phi / len(distinction.purview_union)
                    for distinction in overlapping_distinctions
                ]
            )
        # Self-relations have |r| = 1, so they enter at full phi
        apportioned += sum(relation.phi for relation in self.self_relations)
        return apportioned

    def _num_relations(self):
        count = 0
        # Compute number of relations excluding self-relations
        for purview, overlapping_distinctions in self.distinctions.purview_inclusion(
            max_order=None
        ):
            inclusion_exclusion_term = (-1) ** (len(purview) - 1)
            overlap_size_term = (
                2 ** len(overlapping_distinctions) - len(overlapping_distinctions) - 1
            )
            count += inclusion_exclusion_term * overlap_size_term
        # Count self-relations
        count += len(self.self_relations)
        return count

    def __len__(self):
        return self.num_relations()


class AnalyticalFoldRelations(AnalyticalRelations):
    """Closed-form sums over the relations incident to a set of seed
    distinctions within a parent structure.

    Every analytical quantity is a sum over relations, and a relation either
    touches the seed set ``F`` or it does not, so the incident total is
    ``total(D) - total(D\\F)`` over two plain :class:`AnalyticalRelations`.
    Self-relations of ``D\\F`` cancel in the difference; self-relations of the
    seeds survive. Enumeration (iteration, faces) is not supported -- use
    concrete relations for that.
    """

    def __init__(self, parent_distinctions, seeds):
        super().__init__(parent_distinctions)
        self._full = AnalyticalRelations(parent_distinctions)
        seed_mechanisms = {tuple(d.mechanism) for d in seeds}
        from pyphi.models.distinctions import ResolvedDistinctions

        complement = ResolvedDistinctions(
            d for d in parent_distinctions if tuple(d.mechanism) not in seed_mechanisms
        )
        self._complement = AnalyticalRelations(complement)

    def _sum_phi(self):
        return self._full.sum_phi() - self._complement.sum_phi()

    def _num_relations(self):
        return self._full.num_relations() - self._complement.num_relations()

    def _apportioned_sum_phi(self):
        return self._full.apportioned_sum_phi() - self._complement.apportioned_sum_phi()


def relations(
    distinctions: ResolvedDistinctions,
    relation_computation: str | None = None,
    **kwargs: Any,
) -> Relations:
    """Return causal relations among a set of distinctions.

    Requires :class:`~pyphi.models.distinctions.ResolvedDistinctions`:
    relations between distinctions whose tied specified states haven't
    been disambiguated by a SIA system_state can include phantom faces
    that wouldn't exist after resolution. Pass the result of
    :meth:`~pyphi.models.distinctions.Distinctions.resolve_congruence`
    or use :func:`pyphi.formalism.iit4.ces` to obtain a
    consistent structure.
    """
    return relation_computations[
        fallback(relation_computation, config.formalism.iit.relation_computation)  # type: ignore[index]  # config.Option descriptor
    ](distinctions, **kwargs)


class RelationComputationsRegistry(Registry):
    """Storage for functions for computing relations.

    Users can define custom schemes:

    Examples:
        >>> @relation_computations.register('NONE')  # doctest: +SKIP
        ... def no_relations(system, ces):
        ...    return Relations([])

    And use them by setting ``config.formalism.iit.relation_computation = 'NONE'``.
    """

    desc = "methods for computing relations"


relation_computations = RelationComputationsRegistry()


@relation_computations.register("CONCRETE")
def concrete_relations(
    distinctions: Iterable[Distinction], **kwargs: Any
) -> ConcreteRelations:
    return ConcreteRelations(all_relations(distinctions, **kwargs))


@relation_computations.register("ANALYTICAL")
def analytical_relations(
    distinctions: Iterable[Distinction], **kwargs: Any
) -> AnalyticalRelations:
    return AnalyticalRelations(distinctions)


# Functional alias
def relation(distinctions: Iterable[Distinction]) -> Relation:
    return Relation(distinctions)
