"""Differentiation across the structures triggered by a set of stimuli."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from itertools import combinations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .perception import Perception


def _component_perceptions(perception):
    """Yield (component, perception) for each component of one structure."""
    for distinction in perception.ces.distinctions:
        yield distinction, perception.distinction_perception(distinction)
    for relation in perception.ces.relations:  # pyright: ignore[reportGeneralTypeIssues]  # Relations base lacks __iter__; concrete subclasses provide it
        yield relation, perception.relation_perception(relation)


@dataclass(frozen=True)
class Differentiation:
    """The component union across triggered structures (Eq 15).

    A pure view over ``Perception`` objects: the distinctions and relations
    of the structures are pooled and deduplicated by value equality, and each
    unique component carries the maximum perception it attains in any of the
    structures. Duplicate structures collapse in the union, so sequence order
    and repeats do not affect the result.
    """

    perceptions: tuple[Perception, ...]

    @cached_property
    def projection(self) -> dict:
        """{component: maximum perception across structures containing it}."""
        projection = {}
        for perception in self.perceptions:
            for component, value in _component_perceptions(perception):
                existing = projection.get(component)
                if existing is None or value > existing:
                    projection[component] = value
        return projection

    @cached_property
    def differentiation(self) -> float:
        """Differentiation D (Eq 16): summed phi of the unique components."""
        return float(sum(float(component.phi) for component in self.projection))

    @cached_property
    def perceptual_differentiation(self) -> float:
        """Perceptual differentiation D_p (Eq 19): summed maximum perception."""
        return float(sum(self.projection.values()))

    @cached_property
    def analytical_differentiation(self) -> float:
        """Differentiation D (Eq 16), in closed form without enumerating
        concrete relations.

        Equal to :attr:`differentiation` wherever that is computable, but reads
        only each structure's ``distinctions`` (never ``relations``), so it is
        the path to use when the structures carry ``AnalyticalRelations`` (which
        are not iterable). D splits into the distinction union ``Σφ_d`` plus the
        relation union, the latter computed by inclusion-exclusion over the
        unique structures::

            Σ_r φ_r = Σ_{∅≠T} (-1)^(|T|+1) AnalyticalRelations(∩_{k∈T} D_k).sum_phi()

        Cost is ``2**K - 1`` analytical relation-sum calls for ``K`` unique
        structures; the method targets the small-``K`` (small sensory interface)
        regime where concrete relation enumeration is the bottleneck.
        """
        if not self.perceptions:
            return 0.0

        from pyphi.models.distinctions import ResolvedDistinctions
        from pyphi.relations import AnalyticalRelations

        # Distinction union term: Σφ_d over the identity-deduplicated union.
        union_distinctions: dict = {}
        for perception in self.perceptions:
            for distinction in perception.ces.distinctions:
                union_distinctions.setdefault(distinction, distinction)
        distinction_sum = sum(float(d.phi) for d in union_distinctions)

        # Relation union term: inclusion-exclusion over the unique structures
        # (deduplicated by distinction set — duplicates do not change the union).
        structures = list({frozenset(p.ces.distinctions) for p in self.perceptions})
        relation_sum = 0.0
        for size in range(1, len(structures) + 1):
            sign = 1.0 if size % 2 == 1 else -1.0
            for subset in combinations(structures, size):
                common = frozenset.intersection(*subset)
                if common:
                    relations = AnalyticalRelations(ResolvedDistinctions(common))
                    relation_sum += sign * float(relations.sum_phi())
        return float(distinction_sum + relation_sum)
