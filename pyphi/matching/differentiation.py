"""Differentiation across the structures triggered by a set of stimuli."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
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
