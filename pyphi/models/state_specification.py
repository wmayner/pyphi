# models/state_specification.py
"""Specified states and the per-unit ``UnitState`` value type used in
mechanism-level analysis.

``UnitState`` is the per-step (node, state) value used by
:class:`StateSpecification` and related formatting code. It is distinct
from :class:`pyphi.core.unit.Unit`, which is the substrate-level identity
of a node (without a state value).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import total_ordering
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from pyphi.conf import config
from pyphi.data_structures import PyPhiFloat
from pyphi.direction import Direction
from pyphi.metrics.distribution import DistanceResult
from pyphi.registry import Registry
from pyphi.warnings import warn_about_tie_serialization

from . import cmp
from . import fmt
from .pandas import ToDictMixin
from .pandas import ToPandasMixin


@total_ordering
@dataclass(frozen=True)
class UnitState:
    """A node together with its current state value.

    Distinct from :class:`pyphi.core.unit.Unit`, which is the
    substrate-level identity of a node (``index``, ``label``); a
    ``UnitState`` adds the per-step ``state`` value, used by
    :class:`StateSpecification` and related formatting code.
    """

    index: int
    state: int
    label: str | None = None

    def __hash__(self) -> int:
        return hash((self.index, self.state))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UnitState):
            return NotImplemented
        return (self.index, self.state) == (other.index, other.state)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, UnitState):
            return NotImplemented
        return (self.index, self.state) < (other.index, other.state)

    def __repr__(self) -> str:
        label = str(self.index) if self.label is None else self.label
        return label.lower() if self.state == 0 else label.upper()


@dataclass
class StateSpecification(ToDictMixin, ToPandasMixin):
    direction: Direction
    purview: tuple[int, ...]
    state: tuple[int, ...]
    intrinsic_information: PyPhiFloat | DistanceResult
    repertoire: ArrayLike
    unconstrained_repertoire: ArrayLike
    _ties: tuple[StateSpecification, ...] = ()

    def __post_init__(self):
        if not isinstance(self.intrinsic_information, DistanceResult):
            self.intrinsic_information = PyPhiFloat(self.intrinsic_information)

    def set_ties(self, ties: Iterable[StateSpecification]) -> None:
        object.__setattr__(self, "_ties", tuple(ties))

    @property
    def ties(self) -> tuple[StateSpecification, ...]:
        return self._ties

    def __getitem__(self, i: int) -> int:
        return self.state[i]

    def __eq__(self, other: object) -> bool:
        return cmp.general_eq(
            self,
            other,
            [
                "direction",
                "purview",
                "state",
                "intrinsic_information",
                "repertoire",
                "unconstrained_repertoire",
            ],
        )

    def __hash__(self) -> int:
        return hash(
            (self.direction, self.purview, self.state, self.intrinsic_information)
        )

    def _repr_columns(self, prefix: str = "") -> list[tuple[str, Any]]:
        # TODO(fmt) include purview
        return [
            (f"{prefix}{self.direction}", fmt.state(self.state)),
            (
                f"{prefix}II_{str(self.direction)[:1].lower()}",
                self.intrinsic_information,
            ),
        ]

    def __repr__(self) -> str:
        body = "\n".join(fmt.align_columns(self._repr_columns()))
        body = fmt.header(
            f"Specified {self.direction}", body, under_char=fmt.HEADER_BAR_3
        )
        return fmt.box(fmt.center(body))

    def is_congruent(self, other: StateSpecification) -> bool:
        ours = dict(zip(self.purview, self.state, strict=False))
        theirs = dict(zip(other.purview, other.state, strict=False))
        mutual = set(ours.keys()) & set(theirs.keys())
        return self.direction == other.direction and all(
            ours[purview_node] == theirs[purview_node] for purview_node in mutual
        )

    def to_json(self) -> dict[str, Any]:
        warn_about_tie_serialization(self.__class__.__name__, serialize=True)
        dct = self.to_dict()
        return dct

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> StateSpecification:
        warn_about_tie_serialization(cls.__name__, deserialize=True)
        for key in ["repertoire", "unconstrained_repertoire"]:
            data[key] = np.array(data[key])
        instance = cls(**data)
        object.__setattr__(instance, "_ties", (instance,))
        return instance


class DistinctionPhiNormalizationRegistry(Registry):
    """Storage for distinction |small_phi| normalizations."""

    desc = "functions for normalizing distinction |small_phi| values"


distinction_phi_normalizations = DistinctionPhiNormalizationRegistry()


@distinction_phi_normalizations.register("NONE")
def _(partition: object) -> int:  # noqa: ARG001
    return 1


@distinction_phi_normalizations.register("NUM_CONNECTIONS_CUT")
def _(partition: object) -> int | float | None:
    try:
        return 1 / partition.num_connections_cut()  # type: ignore[attr-defined]
    except ZeroDivisionError:
        return 1
    except AttributeError:
        return None


def normalization_factor(partition: object) -> int | float | None:
    key = config.formalism.iit.distinction_phi_normalization
    func = distinction_phi_normalizations[key]  # type: ignore[index]
    return func(partition)


@dataclass(frozen=True)
class SystemStateSpecification(ToDictMixin, ToPandasMixin):
    """A pair of cause/effect ``StateSpecification`` instances.

    Used at the system level (IIT 4.0 ``SIA`` and ``CauseEffectStructure``) to
    bundle the maximally-specifying cause and effect states.
    """

    cause: StateSpecification
    effect: StateSpecification

    def __getitem__(self, direction: Direction) -> StateSpecification:
        if direction == Direction.CAUSE:
            return self.cause
        if direction == Direction.EFFECT:
            return self.effect
        raise KeyError("Invalid direction")

    def _repr_columns(self, prefix: str = "") -> list[tuple[str, Any]]:
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

    def __repr__(self) -> str:
        body = "\n".join(fmt.align_columns(self._repr_columns()))
        body = fmt.header("Specified System State", body, under_char=fmt.HEADER_BAR_3)
        return fmt.box(fmt.center(body))

    def __hash__(self) -> int:
        return hash((self.cause, self.effect))

    def to_json(self) -> dict[str, Any]:
        return self.__dict__
