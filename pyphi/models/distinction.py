# models/distinction.py
"""Distinction: the maximally irreducible cause and effect specified by a
mechanism (Albantakis et al. 2023). The IIT 3.0 paper terminology calls
the same object a *concept*; the alias :data:`Concept` below preserves
that vocabulary for callers using the IIT 3.0 idiom."""

from __future__ import annotations

from functools import cached_property
from typing import Any

import numpy as np

from pyphi import utils
from pyphi import validate
from pyphi.direction import Direction
from pyphi.display import Description
from pyphi.display import Displayable
from pyphi.display import Row
from pyphi.display import Section
from pyphi.display.numbers import format_value

from . import cmp
from .diff import ResultDiff
from .diff import _diff_common
from .explanation import Explanation
from .explanation import Finding
from .pandas import ToDictFromExplicitAttrsMixin
from .pandas import ToPandasMixin

_distinction_attributes = [
    "phi",
    "mechanism",
    "mechanism_state",
    "mechanism_label",
    "cause",
    "effect",
]


# TODO: make mechanism a property
# TODO: make phi a property
class Distinction(
    Displayable, cmp.OrderableByPhi, ToDictFromExplicitAttrsMixin, ToPandasMixin
):
    """The maximally irreducible cause and effect specified by a mechanism.

    These can be compared with the built-in Python comparison operators (``<``,
    ``>``, etc.). Comparison is based on |small_phi| value, then mechanism size.

    Attributes:
        mechanism (tuple[int]): The mechanism that the distinction consists of.
        cause (MaximallyIrreducibleCause): The |MIC| representing the
            maximally-irreducible cause of this distinction.
        effect (MaximallyIrreducibleEffect): The |MIE| representing the
            maximally-irreducible effect of this distinction.
        time (float): The number of seconds it took to calculate.
    """

    def __init__(
        self,
        mechanism=None,
        cause=None,
        effect=None,
    ):
        self.mechanism = mechanism
        self.cause = cause
        self.effect = effect
        # Attach references to this object on the cause and effect
        # TODO: document this
        assert self.cause is not None
        assert self.effect is not None
        self.cause.parent = self
        self.effect.parent = self

    def _describe(self, verbosity: int) -> Description:  # noqa: ARG002
        cls = type(self).__name__
        mechanism_label = getattr(self, "mechanism_label", None) or str(
            getattr(self, "mechanism", "")
        )
        cause_purview = getattr(self, "cause_purview", None)
        effect_purview = getattr(self, "effect_purview", None)
        cause_phi = getattr(self.cause, "phi", None) if self.cause is not None else None
        effect_phi = (
            getattr(self.effect, "phi", None) if self.effect is not None else None
        )

        # Extract just the state tuple (not the full StateSpecification card)
        _cause_spec = (
            getattr(self.cause, "specified_state", None)
            if self.cause is not None
            else None
        )
        _effect_spec = (
            getattr(self.effect, "specified_state", None)
            if self.effect is not None
            else None
        )
        cause_state = getattr(_cause_spec, "state", _cause_spec)
        effect_state = getattr(_effect_spec, "state", _effect_spec)

        return Description(
            title=cls,
            sections=(
                Section(
                    rows=(
                        Row("Mechanism", mechanism_label),
                        Row("φ_d", self.phi),
                    )
                ),
                Section(
                    label="Cause",
                    tone="cause",
                    rows=(
                        Row("Purview", str(cause_purview)),
                        Row("φ", cause_phi),
                        Row("Specified state", str(cause_state)),
                    ),
                ),
                Section(
                    label="Effect",
                    tone="effect",
                    rows=(
                        Row("Purview", str(effect_purview)),
                        Row("φ", effect_phi),
                        Row("Specified state", str(effect_state)),
                    ),
                ),
            ),
            compact=f"{cls}({mechanism_label}, φ_d={format_value(self.phi)})",
        )

    # TODO use cached_property
    @property
    def phi(self) -> float:  # type: ignore[override]
        """float: The size of the distinction.

        This is the minimum of the |small_phi| values of the distinction's |MIC|
        and |MIE|.
        """
        assert self.cause is not None
        assert self.effect is not None
        return min(self.cause.phi, self.effect.phi)

    def explain(self) -> Explanation:
        """A typed account of why this distinction's |small_phi| came out as it
        did: which direction (cause or effect) binds, plus that direction's own
        findings."""
        assert self.cause is not None
        assert self.effect is not None
        binding = (
            self.cause
            if float(self.cause.phi) <= float(self.effect.phi)
            else self.effect
        )
        is_cause = binding is self.cause
        findings = [
            Finding(
                kind="binding_direction",
                label="Binding direction",
                value="CAUSE" if is_cause else "EFFECT",
                detail=(("φ_cause", self.cause.phi), ("φ_effect", self.effect.phi)),
                tone="cause" if is_cause else "effect",
            ),
            *binding.explain().findings,
        ]
        return Explanation(
            subject=f"φ = {format_value(self.phi)}",
            level="mechanism",
            findings=tuple(findings),
        )

    def diff(self, other) -> ResultDiff:
        """Structured delta from this distinction to ``other`` (``a.diff(b)``).

        A distinction carries no :class:`ConfigSnapshot`, so ``config_diff`` is
        always empty.
        """
        if not isinstance(other, Distinction):
            raise TypeError(
                f"cannot diff {type(self).__name__} against {type(other).__name__}"
            )
        common = _diff_common(self, other)
        return ResultDiff(
            subject=f"Δφ = {format_value(common['delta_phi'])}",
            level="mechanism",
            delta_phi=common["delta_phi"],
            mip_changed=common["mip_changed"],
            changes=(),
            config_diff=common["config_diff"],
            substrate_note=common["substrate_note"],
        )

    # TODO: rename?
    def mice(self, direction):
        if direction is Direction.CAUSE:
            return self.cause
        if direction is Direction.EFFECT:
            return self.effect
        validate.direction(direction)
        return None

    @property
    def cause_purview(self):
        """tuple[int]: The cause purview."""
        return getattr(self.cause, "purview", None)

    @property
    def effect_purview(self):
        """tuple[int]: The effect purview."""
        return getattr(self.effect, "purview", None)

    @cached_property
    def both_purview_unit_sets(self):
        return [
            set(self.mice(direction).purview_units)  # type: ignore[union-attr]
            for direction in Direction.both()
        ]

    @cached_property
    def purview_union(self):
        return set.union(*self.both_purview_unit_sets)

    @cached_property
    def purview_intersection(self):
        return set.intersection(*self.both_purview_unit_sets)

    @property
    def cause_repertoire(self):
        """np.ndarray: The cause repertoire."""
        return getattr(self.cause, "repertoire", None)

    @property
    def effect_repertoire(self):
        """np.ndarray: The effect repertoire."""
        return getattr(self.effect, "repertoire", None)

    @property
    def mechanism_state(self):
        """tuple(int): The state of this mechanism."""
        assert self.cause is not None
        assert self.effect is not None
        if self.cause.mechanism_state != self.effect.mechanism_state:
            raise ValueError("Inconsistent cause and effect mechanism states!")
        return self.cause.mechanism_state

    @cached_property
    def mechanism_label(self):
        """tuple[str]: The labels of the mechanism nodes."""
        return self.node_labels.label_string(self.mechanism, self.mechanism_state)  # type: ignore[arg-type]

    def purview(self, direction):
        """Return the purview in the given direction."""
        assert self.cause is not None
        assert self.effect is not None
        if direction == Direction.CAUSE:
            return self.cause.purview
        if direction == Direction.EFFECT:
            return self.effect.purview
        raise ValueError("invalid direction")

    @property
    def node_labels(self):
        assert self.cause is not None
        assert self.effect is not None
        if self.cause.node_labels != self.effect.node_labels:
            raise ValueError("Inconsistent cause and effect node labels!")
        return self.cause.node_labels

    def __eq__(self, other: object) -> bool:  # noqa: PLR0911
        if not isinstance(other, Distinction):
            return NotImplemented
        if self.mechanism != other.mechanism:
            return False
        if self.mechanism_state != other.mechanism_state:
            return False
        if self.cause_purview != other.cause_purview:
            return False
        if self.effect_purview != other.effect_purview:
            return False
        if not utils.eq(self.phi, other.phi):
            return False
        if not cmp.numpy_aware_eq(self.cause_repertoire, other.cause_repertoire):
            return False
        return cmp.numpy_aware_eq(self.effect_repertoire, other.effect_repertoire)

    def __hash__(self) -> int:
        # Hash uses only strict-equality attrs from __eq__; phi and repertoires
        # are tolerance-compared in __eq__ so they cannot appear here without
        # violating the a == b -> hash(a) == hash(b) contract.
        return hash(
            (
                self.mechanism,
                self.mechanism_state,
                self.cause_purview,
                self.effect_purview,
            )
        )

    def __bool__(self):
        """A distinction is ``True`` if |small_phi > 0|."""
        return utils.is_positive(self.phi)

    def is_congruent(self, system_state):
        return all(
            self.mice(direction).is_congruent(system_state[direction])  # type: ignore[union-attr]
            for direction in Direction.both()
        )

    def resolve_congruence(self, system_state):
        """Select the cause and effect MICEs congruent with the SIA's
        system-level specified cause-effect state.

        For each direction, applies the distinction-state cascade per
        Albantakis et al. 2023 S1 Text: state ties within a purview
        resolve to the congruent MICE; cross-purview ties resolve to
        the largest congruent purview (the heuristic for "supports the
        most relations with other distinctions"). Returns ``None`` when
        no congruent MICE exists in either direction.
        """
        from pyphi.resolve_ties import ResolutionContext
        from pyphi.resolve_ties import resolve_distinction_tie

        context = ResolutionContext(max_escalation_level="Composition")
        chosen: dict[Direction, Any] = {}
        for direction in Direction.both():
            mice = self.mice(direction)
            if mice is None:
                return None
            chosen[direction] = resolve_distinction_tie(
                state_ties=mice.state_ties,
                purview_ties=mice.purview_ties,
                system_state_spec=system_state[direction],
                context=context,
            )
        if chosen[Direction.CAUSE] is None or chosen[Direction.EFFECT] is None:
            return None
        return type(self)(
            mechanism=self.mechanism,
            cause=chosen[Direction.CAUSE],
            effect=chosen[Direction.EFFECT],
        )

    def eq_repertoires(self, other):
        """Return whether this distinction has the same repertoires as another.

        .. warning::
            This only checks if the cause and effect repertoires are equal as
            arrays; mechanisms, purviews, or even the nodes that the mechanism
            and purview indices refer to, might be different.
        """
        return np.array_equal(
            self.cause_repertoire,  # pyright: ignore[reportArgumentType]
            other.cause_repertoire,  # type: ignore[arg-type]
        ) and np.array_equal(
            self.effect_repertoire,  # pyright: ignore[reportArgumentType]
            other.effect_repertoire,  # type: ignore[arg-type]
        )

    def emd_eq(self, other):
        """Return whether this distinction is equal to another in the context of
        an EMD calculation.
        """
        return (
            self.phi == other.phi
            and self.mechanism == other.mechanism
            and self.eq_repertoires(other)
        )

    _dict_attrs = _distinction_attributes

    def _pandas_record(self):
        labels = self.node_labels

        def labelled(nodes):
            if nodes is None:
                return None
            if labels is None:
                return tuple(nodes)
            return tuple(labels.coerce_to_labels(nodes))

        return {
            "phi": float(self.phi),
            "mechanism": labelled(self.mechanism),
            "mechanism_state": (
                None if self.mechanism_state is None else tuple(self.mechanism_state)
            ),
            "cause_purview": labelled(self.cause_purview),
            "effect_purview": labelled(self.effect_purview),
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Restore parent references to MICEs
        assert self.cause is not None
        assert self.effect is not None
        self.cause.parent = self
        self.effect.parent = self


# IIT 3.0 paper terminology calls a distinction a "concept". The alias
# preserves that vocabulary for callers using the IIT 3.0 idiom; the
# runtime class is identical.
Concept = Distinction
