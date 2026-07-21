"""Declarative feasibility surfaces for scoped cause-effect analyses.

A scope states which mechanisms and purviews a computation considers.
Exclusions are explicit and certified — scope changes *what* is computed;
it never silently approximates. Constraint fields are named data (no
callables), so scopes serialize, ship to batch jobs, and land in
provenance. Partition sweeps cannot be scoped: a partial sweep would turn
φ into an upper bound.
"""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Iterator
from dataclasses import dataclass
from dataclasses import field

from pyphi.direction import Direction

__all__ = ["AxisScope", "CESScope", "resolve_scope"]


@dataclass(frozen=True)
class AxisScope:
    """A constraint on one axis of unit sets (mechanisms or purviews).

    Constraint fields combine by intersection. ``explicit`` is exclusive:
    an explicit list *is* the axis, so combining it with any other field
    raises :class:`ValueError`. The default (all fields ``None``) admits
    every candidate.
    """

    explicit: tuple[tuple[int, ...], ...] | None = None
    min_order: int | None = None
    max_order: int | None = None
    containing: tuple[int, ...] | None = None
    within: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        others = (self.min_order, self.max_order, self.containing, self.within)
        if self.explicit is not None and any(o is not None for o in others):
            raise ValueError(
                "explicit is exclusive: an explicit list is the axis and "
                "cannot combine with other constraint fields"
            )

    @property
    def unconstrained(self) -> bool:
        return (
            self.explicit is None
            and self.min_order is None
            and self.max_order is None
            and self.containing is None
            and self.within is None
        )

    def admits(self, units: tuple[int, ...]) -> bool:
        if self.explicit is not None:
            return tuple(sorted(units)) in {tuple(sorted(e)) for e in self.explicit}
        if self.min_order is not None and len(units) < self.min_order:
            return False
        if self.max_order is not None and len(units) > self.max_order:
            return False
        if self.containing is not None and not set(self.containing) <= set(units):
            return False
        return self.within is None or set(units) <= set(self.within)

    def select(self, candidates: Iterable[tuple[int, ...]]) -> Iterator[tuple[int, ...]]:
        """Yield the candidates this scope admits, preserving their order."""
        if self.explicit is not None:
            allowed = {tuple(sorted(e)) for e in self.explicit}
            for candidate in candidates:
                if tuple(sorted(candidate)) in allowed:
                    yield candidate
            return
        for candidate in candidates:
            if self.admits(candidate):
                yield candidate


@dataclass(frozen=True)
class CESScope:
    """The feasibility surface of a cause-effect structure computation."""

    mechanisms: AxisScope = field(default_factory=AxisScope)
    cause_purviews: AxisScope = field(default_factory=AxisScope)
    effect_purviews: AxisScope = field(default_factory=AxisScope)

    def purviews(self, direction: Direction) -> AxisScope:
        if direction == Direction.CAUSE:
            return self.cause_purviews
        return self.effect_purviews


def _units_to_indices(units: tuple, node_labels) -> tuple[int, ...]:
    return tuple(sorted(node_labels.coerce_to_indices(units)))


def _resolve_axis(scope: AxisScope, node_labels) -> AxisScope:
    return AxisScope(
        explicit=None
        if scope.explicit is None
        else tuple(_units_to_indices(e, node_labels) for e in scope.explicit),
        min_order=scope.min_order,
        max_order=scope.max_order,
        containing=None
        if scope.containing is None
        else _units_to_indices(scope.containing, node_labels),
        within=None
        if scope.within is None
        else _units_to_indices(scope.within, node_labels),
    )


def resolve_scope(scope: CESScope, node_labels) -> CESScope:
    """Return the scope with every unit reference normalized to indices."""
    return CESScope(
        mechanisms=_resolve_axis(scope.mechanisms, node_labels),
        cause_purviews=_resolve_axis(scope.cause_purviews, node_labels),
        effect_purviews=_resolve_axis(scope.effect_purviews, node_labels),
    )
