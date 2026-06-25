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

import pandas as pd
from numpy.typing import ArrayLike

from pyphi import utils
from pyphi.conf import config
from pyphi.data_structures import PyPhiFloat
from pyphi.direction import Direction
from pyphi.display import Description
from pyphi.display import Displayable
from pyphi.display import Row
from pyphi.display import Section
from pyphi.display import tone_of
from pyphi.measures.distribution import DistanceResult
from pyphi.registry import Registry

from . import cmp
from . import fmt
from .pandas import ToDictMixin
from .pandas import ToPandasMixin


@total_ordering
@dataclass(frozen=True, repr=False)
class UnitState(Displayable):
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

    def _describe(self, verbosity: int) -> Description:  # noqa: ARG002
        label = str(self.index) if self.label is None else self.label
        compact = label.lower() if self.state == 0 else label.upper()
        return Description(title="UnitState", compact=compact)


@dataclass(repr=False)
class StateSpecification(Displayable, ToDictMixin, ToPandasMixin):
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

    def __eq__(self, other: object) -> bool:  # noqa: PLR0911
        if not isinstance(other, StateSpecification):
            return NotImplemented
        if self.direction != other.direction:
            return False
        if self.purview != other.purview:
            return False
        if self.state != other.state:
            return False
        if not utils.eq(self.intrinsic_information, other.intrinsic_information):
            return False
        if not cmp.numpy_aware_eq(self.repertoire, other.repertoire):
            return False
        return cmp.numpy_aware_eq(
            self.unconstrained_repertoire, other.unconstrained_repertoire
        )

    def __hash__(self) -> int:
        return hash((self.direction, self.purview, self.state))

    def _repr_columns(self, prefix: str = "") -> list[tuple[str, Any]]:
        # TODO(fmt) include purview
        return [
            (f"{prefix}{self.direction}", fmt.state(self.state)),
            (
                f"{prefix}II_{str(self.direction)[:1].lower()}",
                self.intrinsic_information,
            ),
        ]

    def _describe(self, verbosity: int) -> Description:  # noqa: ARG002
        direction_label = str(self.direction)
        tone = tone_of(self.direction)
        return Description(
            title=f"Specified {direction_label}",
            tone=tone,
            sections=(
                Section(
                    rows=(
                        Row("Direction", direction_label, tone=tone),
                        Row("Purview", self.purview),
                        Row("Specified state", self.state),
                        Row("Intrinsic information", self.intrinsic_information),
                    ),
                ),
            ),
        )

    def is_congruent(self, other: StateSpecification) -> bool:
        ours = dict(zip(self.purview, self.state, strict=False))
        theirs = dict(zip(other.purview, other.state, strict=False))
        mutual = set(ours.keys()) & set(theirs.keys())
        return self.direction == other.direction and all(
            ours[purview_node] == theirs[purview_node] for purview_node in mutual
        )

    def _to_pandas(self):
        from .pandas import _DISTRIBUTION_COLUMNS
        from .pandas import distribution_rows
        from .pandas import records_to_frame

        rows = []
        for kind, rep in (
            ("repertoire", self.repertoire),
            ("unconstrained", self.unconstrained_repertoire),
        ):
            rows.extend(distribution_rows(self.direction, kind, self.purview, rep))
        return records_to_frame(rows, columns=_DISTRIBUTION_COLUMNS)


class DistinctionPhiNormalizationRegistry(Registry):
    """Storage for distinction |small_phi| normalizations."""

    desc = "functions for normalizing distinction |small_phi| values"


distinction_phi_normalizations = DistinctionPhiNormalizationRegistry()


@distinction_phi_normalizations.register("NONE")
def _(partition: object) -> int:  # noqa: ARG001
    return 1


@distinction_phi_normalizations.register("NUM_CONNECTIONS_CUT")
def _(partition: object) -> int | float | None:
    # A null/unconstrained analysis carries no partition; there is nothing to
    # normalize against, so normalization is undefined (None).
    if partition is None:
        return None
    num = partition.num_connections_cut()  # type: ignore[attr-defined]
    # A partition that severs no connections has no normalization scale.
    return 1 / num if num else 1


def normalization_factor(partition: object) -> int | float | None:
    key = config.formalism.iit.distinction_phi_normalization
    func = distinction_phi_normalizations[key]  # type: ignore[index]
    return func(partition)


@dataclass(frozen=True, repr=False)
class SystemStateSpecification(Displayable, ToDictMixin, ToPandasMixin):
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

    def _describe(self, verbosity: int) -> Description:  # noqa: ARG002
        sections = []
        for direction, spec in (("Cause", self.cause), ("Effect", self.effect)):
            tone = direction.lower()
            if spec is not None:
                sections.append(
                    Section(
                        label=direction,
                        tone=tone,
                        rows=(
                            Row("Purview", spec.purview),
                            Row("Specified state", spec.state),
                            Row("Intrinsic information", spec.intrinsic_information),
                        ),
                    )
                )
            else:
                sections.append(
                    Section(label=direction, tone=tone, rows=(Row("State", None),))
                )
        return Description(
            title="Specified System State",
            sections=tuple(sections),
        )

    def __hash__(self) -> int:
        return hash((self.cause, self.effect))

    def _to_pandas(self):
        from .pandas import _DISTRIBUTION_COLUMNS
        from .pandas import records_to_frame

        frames = [
            spec.to_pandas() for spec in (self.cause, self.effect) if spec is not None
        ]
        if not frames:
            return records_to_frame([], columns=_DISTRIBUTION_COLUMNS)
        return pd.concat(frames, ignore_index=True)
