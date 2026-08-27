# formalism/iit4/__init__.py
"""IIT 4.0 system-level analysis: SIA, distinctions, relations, Φ-structure.

Implements the algorithms from Albantakis et al. 2023 (and the 2026 extension
when configured via the ``IIT_4_0_2026`` formalism). Concrete formalism
classes wrapping these algorithms live in :mod:`pyphi.formalism.iit4.formalism`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import replace
from typing import Any

from pyphi import conf
from pyphi import connectivity
from pyphi import measures
from pyphi import numerics
from pyphi import resolve_ties
from pyphi import utils
from pyphi import validate
from pyphi.conf import config
from pyphi.conf import fallback
from pyphi.conf.snapshot import ConfigSnapshot
from pyphi.core import repertoire_algebra as repertoire
from pyphi.direction import Direction
from pyphi.display import FULL
from pyphi.display import PROVENANCE
from pyphi.display import Description
from pyphi.display import Displayable
from pyphi.display import Row
from pyphi.display import Section
from pyphi.display.numbers import format_value
from pyphi.formalism import iit3
from pyphi.formalism.queries import _never_shortcircuit
from pyphi.labels import NodeLabels
from pyphi.measures.distribution import DistanceResult
from pyphi.measures.protocols import CompositeMeasure
from pyphi.measures.protocols import DistributionMeasure
from pyphi.measures.protocols import StateAwareMeasure
from pyphi.measures.protocols import StatefulDistributionMeasure
from pyphi.measures.protocols import satisfies_composite_measure
from pyphi.models import cmp
from pyphi.models.ces import CauseEffectStructure
from pyphi.models.diff import ResultDiff
from pyphi.models.diff import _diff_common
from pyphi.models.distinctions import Distinctions
from pyphi.models.distinctions import ResolvedDistinctions
from pyphi.models.explanation import Explanation
from pyphi.models.explanation import Finding
from pyphi.models.explanation import NullResultReason
from pyphi.models.explanation import binding_direction_finding
from pyphi.models.explanation import runner_up_from_candidates
from pyphi.models.explanation import sia_runner_up_key
from pyphi.models.pandas import ToPandasMixin
from pyphi.models.partitions import DirectedBipartition
from pyphi.models.partitions import EdgeCut
from pyphi.models.partitions import NullCut
from pyphi.models.partitions import _cut_grid
from pyphi.models.partitions import concise_partition
from pyphi.models.ria import RepertoireIrreducibilityAnalysis
from pyphi.models.state_specification import StateSpecification
from pyphi.models.state_specification import SystemStateSpecification
from pyphi.parallel import map_reduce
from pyphi.partition import system_partition_types
from pyphi.partition import system_partitions
from pyphi.provenance import HasProvenance
from pyphi.provenance import Provenance
from pyphi.relations import ConcreteRelations
from pyphi.relations import Relations
from pyphi.relations import relations as compute_relations
from pyphi.serializable import Serializable
from pyphi.system import System

##############################################################################
# Information
##############################################################################


# TODO: refactor
def system_intrinsic_information(
    system: System,
    *,
    specification_measure: (
        DistributionMeasure
        | StateAwareMeasure
        | StatefulDistributionMeasure
        | CompositeMeasure
    ),
    directions: Iterable[Direction] | None = None,
) -> SystemStateSpecification:
    """Return the cause/effect states specified by the system.

    ``specification_measure`` is a Protocol-typed measure callable used to
    compute intrinsic information; passed explicitly by the active
    formalism rather than read from config.

    NOTE: State ties are arbitrarily broken.
    """
    directions = fallback(directions, Direction.both())
    # Ensure directions is not None before converting to tuple
    if directions is None:
        directions = Direction.both()
    directions = tuple(directions)
    # TODO move to Direction
    # TODO have validation methods return the validated value
    validate.directions(directions)
    # TODO(ties) deal with ties here
    ii = {
        direction: system.intrinsic_information(
            direction,
            mechanism=system.node_indices,
            purview=system.node_indices,
            specification_measure=specification_measure,
        )
        for direction in directions
    }
    # Get the state specifications
    # SystemStateSpecification's constructor should handle None values
    # if that's the expected behavior when a direction is not present
    return SystemStateSpecification(
        cause=ii.get(Direction.CAUSE),  # pyright: ignore[reportArgumentType] - Constructor handles Optional
        effect=ii.get(Direction.EFFECT),  # pyright: ignore[reportArgumentType] - Constructor handles Optional
    )


##############################################################################
# Integration
##############################################################################


def _optional_eq(
    a: float | DistanceResult | None,
    b: float | DistanceResult | None,
) -> bool:
    """Tolerance-equal comparison that handles ``None`` operands."""
    if a is None or b is None:
        return a is b
    return numerics.eq(a, b)


def _optional_float(x: Any) -> float | None:
    """``float(x)``, passing ``None`` through."""
    return float(x) if x is not None else None


def _intrinsic_differentiation_eq(a: dict | None, b: dict | None) -> bool:
    """Tolerance-aware equality for ``intrinsic_differentiation`` dicts.

    Returns True when both are ``None``, or when both have the same
    direction keys and each per-direction value is tolerance-equal.
    """
    if a is None or b is None:
        return a is b
    if set(a.keys()) != set(b.keys()):
        return False
    return all(numerics.eq(a[k], b[k]) for k in a)


@dataclass(repr=False)
class SystemIrreducibilityAnalysis(
    HasProvenance, Displayable, cmp.OrderableByPhi, ToPandasMixin, Serializable
):
    """System-level integrated information.

    ``phi`` is the non-negative integrated-information value defined by
    Eqs. 19-20 (the ``|·|+`` operator applied to the raw integration).
    ``signed_phi`` is the raw value before clamping; when negative, it
    indicates preventative causation that the clamp hides. Construction
    accepts the signed value as ``phi`` and ``__post_init__`` writes the
    clamped value to ``phi`` while preserving the raw value on
    ``signed_phi``. ``normalized_phi`` and ``signed_normalized_phi``
    follow the same pattern.

    ``partition_margin`` is the gap in (clamped) normalized φ between the
    MIP and the best competing partition, computed at selection (before
    the IIT 4.0 2026 ii(s) cap); it is zero when a competitor ties and
    ``None`` when there was no competitor or the partition sweep stopped
    early on a reducible partition (in which case no exact margin exists).
    Set ``shortcircuit_sia=False`` to evaluate every partition and obtain
    an exact margin even when φ_s = 0.
    """

    phi: float | DistanceResult
    partition: DirectedBipartition | DirectedBipartition | NullCut
    normalized_phi: float = 0
    cause: RepertoireIrreducibilityAnalysis | None = None
    effect: RepertoireIrreducibilityAnalysis | None = None
    system_state: SystemStateSpecification | None = None
    current_state: tuple[int, ...] | None = None
    node_indices: tuple[int, ...] | None = None
    node_labels: NodeLabels | None = None
    intrinsic_differentiation: dict | None = None
    reasons: list | None = None
    runner_up: Any = None
    partition_margin: float | None = None
    signed_phi: float | DistanceResult | None = None
    signed_normalized_phi: float | DistanceResult | None = None
    config: ConfigSnapshot | None = None
    provenance: Provenance | None = None

    def __post_init__(self):
        if self.config is None:
            from pyphi.conf import config as _global

            self.config = _global.snapshot()
        if self.provenance is None:
            self.provenance = Provenance.capture()
        # Snapshot the raw signed values *before* clamping.
        if self.signed_phi is None:
            self.signed_phi = self.phi
        if self.signed_normalized_phi is None:
            self.signed_normalized_phi = self.normalized_phi
        # Eqs 19-20: clamp negative integration to zero via ``|·|+``.
        clamped_phi = utils.positive_part(self.signed_phi)
        clamped_normalized = utils.positive_part(self.signed_normalized_phi)
        if not isinstance(self.phi, DistanceResult):
            self.phi = float(clamped_phi)
        else:
            # Clamp the numeric component while preserving the
            # DistanceResult's metadata.
            self.phi = type(self.phi)(clamped_phi, **self.phi._public_aux_data())
        if not isinstance(self.normalized_phi, DistanceResult):
            self.normalized_phi = float(clamped_normalized)
        else:
            self.normalized_phi = type(self.normalized_phi)(
                clamped_normalized, **self.normalized_phi._public_aux_data()
            )
        if not isinstance(self.signed_phi, DistanceResult):
            self.signed_phi = float(self.signed_phi)
        if not isinstance(self.signed_normalized_phi, DistanceResult):
            self.signed_normalized_phi = float(self.signed_normalized_phi)
        # ``intrinsic_differentiation`` stays None when not computed (e.g. on
        # null SIAs), so ``intrinsic_information`` reports None rather than a
        # fabricated zero.

    def order_by(self):
        return self.phi

    @property
    def ties(self):
        try:
            return self._ties
        except AttributeError:
            self._ties = [self]
            return self._ties

    def set_ties(self, ties):
        self._ties = ties

    @property
    def state_margins(self) -> dict[Direction, float | None]:
        """Per-direction intrinsic-information gap between the specified
        system state and the best competing state
        (:attr:`StateSpecification.state_margin`)."""
        margins: dict[Direction, float | None] = {}
        for direction in Direction.both():
            spec = (
                self.system_state[direction] if self.system_state is not None else None
            )
            margins[direction] = spec.state_margin if spec is not None else None
        return margins

    @property
    def tied_selections(self) -> tuple[str, ...]:
        """The selections whose margin is within ``config.numerics.precision``
        of zero — a subset of ``("partition", "cause_state", "effect_state")``.

        Empty when every selection has a clear winner. See also the
        per-direction :attr:`StateSpecification.state_margin` on
        ``system_state`` and :attr:`partition_margin`.
        """
        named = {
            "partition": self.partition_margin,
            "cause_state": self.state_margins[Direction.CAUSE],
            "effect_state": self.state_margins[Direction.EFFECT],
        }
        return tuple(
            name
            for name, margin in named.items()
            if margin is not None and numerics.eq(float(margin), 0.0)
        )

    @property
    def effectively_tied(self) -> bool:
        """Whether any selection margin is within
        ``config.numerics.precision`` of zero — i.e.
        :attr:`tied_selections` is non-empty."""
        return bool(self.tied_selections)

    @property
    def intrinsic_information(self) -> float | None:
        """The system intrinsic information ii(s) (Mayner et al. 2026, Eq. 23).

        The minimum over directions of min(i_spec, i_diff), where i_spec
        is the intrinsic information of the specified state and i_diff its
        intrinsic differentiation, each rectified by ``|·|⁺``.
        Partition-independent given the specified states. Under the
        IIT 4.0 (2026) formalism, φₛ = min{φ_c, φ_e, ii(s)}, so
        φₛ ≤ ii(s). ``None`` when no system state is available or the
        intrinsic differentiation was not computed (e.g. on null SIAs).
        """
        if self.system_state is None or not self.intrinsic_differentiation:
            return None
        terms = []
        for direction, differentiation in self.intrinsic_differentiation.items():
            spec = self.system_state[direction]
            if spec is not None:
                terms.append(float(utils.positive_part(spec.intrinsic_information)))
            terms.append(float(utils.positive_part(differentiation)))
        return min(terms)

    @property
    def integrated_fraction(self) -> float | None:
        """The integrated fraction φₛ / ii(s) of the system intrinsic information.

        Lies in [0, 1] under the IIT 4.0 (2026) formalism, where
        φₛ = min{φ_c, φ_e, ii(s)}; under formalisms without the
        intrinsic-information requirement it may exceed 1. ``None`` when
        ii(s) is unavailable or zero (no finite ratio is defined).
        """
        ii = self.intrinsic_information
        if ii is None or not numerics.is_positive(ii):
            return None
        return float(self.phi) / ii

    def resolve_system_state(self) -> None:
        """Update system_state to reflect the specified states resolved by the MIP.

        When the system has tied specified states, the MIP resolves the tie by
        selecting the state most vulnerable to the winning partition. This
        back-propagates that resolution into system_state so that downstream
        consumers (e.g., congruence filtering in ces) see the correct
        specified states.
        """
        if self.system_state is None:
            return
        new_cause = self.system_state.cause
        new_effect = self.system_state.effect
        if self.cause is not None and self.cause.specified_state is not None:
            new_cause = self.cause.specified_state
        if self.effect is not None and self.effect.specified_state is not None:
            new_effect = self.effect.specified_state
        if (
            new_cause is not self.system_state.cause
            or new_effect is not self.system_state.effect
        ):
            self.system_state = replace(
                self.system_state, cause=new_cause, effect=new_effect
            )

    def __eq__(self, other: object) -> bool:  # noqa: PLR0911
        if not isinstance(other, SystemIrreducibilityAnalysis):
            return NotImplemented
        if self.partition != other.partition:
            return False
        if self.cause != other.cause:
            return False
        if self.effect != other.effect:
            return False
        if self.system_state != other.system_state:
            return False
        if self.current_state != other.current_state:
            return False
        if self.node_indices != other.node_indices:
            return False
        if not numerics.eq(self.phi, other.phi):
            return False
        if not numerics.eq(self.normalized_phi, other.normalized_phi):
            return False
        if not _optional_eq(self.signed_phi, other.signed_phi):
            return False
        if not _optional_eq(self.signed_normalized_phi, other.signed_normalized_phi):
            return False
        return _intrinsic_differentiation_eq(
            self.intrinsic_differentiation, other.intrinsic_differentiation
        )

    def __bool__(self):
        """Whether φ_s > 0."""
        return numerics.is_positive(self.phi)

    def __hash__(self) -> int:
        return hash(
            (
                self.partition,
                self.system_state,
                self.current_state,
                self.node_indices,
            )
        )

    def _system_label(self) -> str | None:
        node_indices = self.node_indices
        node_labels = self.node_labels
        if node_labels is not None and node_indices is not None:
            return ",".join(
                str(label) for label in node_labels.coerce_to_labels(node_indices)
            )
        if node_indices is not None:
            return ",".join(str(i) for i in node_indices)
        return None

    def _pandas_record(self):
        return {
            "phi": float(self.phi),
            "normalized_phi": float(self.normalized_phi),
            "intrinsic_information": _optional_float(self.intrinsic_information),
            "integrated_fraction": _optional_float(self.integrated_fraction),
            "system": self._system_label(),
            "current_state": self.current_state,
            "partition": concise_partition(self.partition)
            if self.partition is not None
            else None,
            "partition_margin": _optional_float(self.partition_margin),
            "cause_state_margin": _optional_float(self.state_margins[Direction.CAUSE]),
            "effect_state_margin": _optional_float(self.state_margins[Direction.EFFECT]),
            "effectively_tied": self.effectively_tied,
        }

    def _describe(self, verbosity: int) -> Description:
        cls = type(self).__name__
        idiff = self.intrinsic_differentiation
        state = self.system_state
        sections = [
            Section(
                rows=(
                    Row("φ_s", self.phi),
                    Row("Normalized φ_s", self.normalized_phi),
                    Row("System", self._system_label()),
                    Row("Current state", self.current_state),
                ),
            )
        ]
        if state is not None and state.cause is not None:
            cause_rows = [
                Row("Specified state", state.cause.state),
                Row("Intrinsic information", state.cause.intrinsic_information),
                Row(
                    "Intrinsic differentiation",
                    idiff[Direction.CAUSE] if idiff else None,
                ),
            ]
            if verbosity >= FULL and state.cause.state_margin is not None:
                cause_rows.append(Row("State margin", state.cause.state_margin))
            sections.append(Section(label="Cause", tone="cause", rows=tuple(cause_rows)))
        if state is not None and state.effect is not None:
            effect_rows = [
                Row("Specified state", state.effect.state),
                Row(
                    "Intrinsic information",
                    state.effect.intrinsic_information,
                ),
                Row(
                    "Intrinsic differentiation",
                    idiff[Direction.EFFECT] if idiff else None,
                ),
            ]
            if verbosity >= FULL and state.effect.state_margin is not None:
                effect_rows.append(Row("State margin", state.effect.state_margin))
            sections.append(
                Section(label="Effect", tone="effect", rows=tuple(effect_rows))
            )
        mip_rows = []
        mip_body: tuple[Any, ...] = ()
        if self.partition is not None:
            mip_rows.append(Row("Partition", concise_partition(self.partition)))
            if self.partition.num_connections_cut():
                mip_body = (_cut_grid(self.partition),)
        mip_rows.append(Row("Tied MIPs", len(self.ties) - 1))
        if verbosity >= FULL:
            if self.partition_margin is not None:
                mip_rows.append(Row("Selection margin", self.partition_margin))
            if self.partition_margin is not None or any(
                margin is not None for margin in self.state_margins.values()
            ):
                mip_rows.append(Row("Effectively tied", self.effectively_tied))
        sections.append(Section(label="MIP", rows=tuple(mip_rows), body=mip_body))
        if self.reasons:
            reasons = ", ".join(getattr(r, "name", str(r)) for r in self.reasons)
            sections.append(Section(label="Reasons", rows=(Row("", reasons),)))
        if verbosity >= PROVENANCE and self.provenance is not None:
            from pyphi.display.provenance import provenance_section

            sections.append(provenance_section(self.provenance))
        return Description(
            title=cls,
            sections=tuple(sections),
            compact=f"{cls}(φ_s={format_value(self.phi)})",
        )

    def _findings(self) -> tuple[Finding, ...]:
        findings: list[Finding] = [
            Finding(kind="null_result", label="Null result", value=reason)
            for reason in self.reasons or []
        ]
        if self.partition is not None and bool(self):
            findings.append(
                Finding(
                    kind="winning_partition",
                    label="MIP",
                    value=concise_partition(self.partition),
                    detail=(("connections cut", self.partition.num_connections_cut()),),
                )
            )
        if self.runner_up is not None:
            findings.append(
                Finding(
                    kind="runner_up",
                    label="Runner-up partition",
                    value=concise_partition(self.runner_up.partition),
                )
            )
            if self.runner_up.normalized_phi is not None:
                # The runner-up was ranked by normalized φ, so the gap is
                # reported in the same quantity.
                gap_label = "normalized φ-gap to runner-up"
                gap = float(self.runner_up.normalized_phi) - float(self.normalized_phi)
            else:
                gap_label = "φ-gap to runner-up"
                gap = float(self.runner_up.phi) - float(self.phi)
            findings.append(Finding(kind="gap", label=gap_label, value=gap))
        if self.partition_margin is not None:
            findings.append(
                Finding(
                    kind="partition_margin",
                    label="MIP selection margin (normalized φ)",
                    value=self.partition_margin,
                )
            )
        state_margins = self.state_margins
        for direction in Direction.both():
            margin = state_margins[direction]
            if margin is not None:
                tone = "cause" if direction == Direction.CAUSE else "effect"
                findings.append(
                    Finding(
                        kind="state_margin",
                        label=f"Specified-{tone}-state margin (ii)",
                        value=margin,
                        tone=tone,
                    )
                )
        if self.partition_margin is not None or any(
            margin is not None for margin in state_margins.values()
        ):
            findings.append(
                Finding(
                    kind="effectively_tied",
                    label="Selection effectively tied",
                    value=self.effectively_tied,
                    detail=(("tied_selections", self.tied_selections),),
                )
            )
        if self.cause is not None and self.effect is not None:
            findings.append(binding_direction_finding(self.cause.phi, self.effect.phi))
        return tuple(findings)

    def explain(self) -> Explanation:
        """A typed account of why this φ_s value came out as it did."""
        return Explanation(
            subject=f"φ_s = {format_value(self.phi)}",
            level="system",
            findings=self._findings(),
        )

    def _binding_direction_changed(self, other) -> bool | None:
        """Whether the binding direction (the lower-φ side) flipped between the
        two analyses; ``None`` when either lacks both a cause and an effect."""
        if (
            self.cause is None
            or self.effect is None
            or other.cause is None
            or other.effect is None
        ):
            return None

        def _direction(cause_phi: float, effect_phi: float) -> str:
            if numerics.eq(float(cause_phi), float(effect_phi)):
                return "tied"
            return "cause" if float(cause_phi) < float(effect_phi) else "effect"

        a_dir = _direction(self.cause.phi, self.effect.phi)
        b_dir = _direction(other.cause.phi, other.effect.phi)
        return a_dir != b_dir

    def diff(self, other) -> ResultDiff:
        """Structured delta from this SIA to ``other`` (``a.diff(b)``)."""
        if not isinstance(other, SystemIrreducibilityAnalysis):
            raise TypeError(
                f"cannot diff {type(self).__name__} against {type(other).__name__}"
            )
        common = _diff_common(self, other)
        return ResultDiff(
            subject=f"Δφ_s = {format_value(common['delta_phi'])}",
            level="system",
            delta_phi=common["delta_phi"],
            mip_changed=common["mip_changed"],
            binding_direction_changed=self._binding_direction_changed(other),
            changes=(),
            config_diff=common["config_diff"],
            substrate_note=common["substrate_note"],
        )


class NullSystemIrreducibilityAnalysis(SystemIrreducibilityAnalysis):
    def __init__(self, node_indices=None, node_labels=None, **kwargs):
        from pyphi.models import NullCut

        # NullCut requires indices, use empty tuple if not provided
        indices = node_indices if node_indices is not None else ()
        super().__init__(
            phi=0,
            partition=NullCut(indices, node_labels),
            cause=None,
            effect=None,
            node_indices=node_indices,
            node_labels=node_labels,
            **kwargs,
        )


def normalization_factor(
    partition: DirectedBipartition | EdgeCut | NullCut,
) -> float:
    if hasattr(partition, "normalization_factor"):
        return partition.normalization_factor()  # pyright: ignore[reportAttributeAccessIssue]
    # For EdgeCut, we need to check hasattr before accessing attributes
    if hasattr(partition, "from_nodes") and hasattr(partition, "to_nodes"):
        return 1 / (len(partition.from_nodes) * len(partition.to_nodes))  # pyright: ignore[reportAttributeAccessIssue]
    # Default fallback
    return 1.0


def _integration_value_for_state(
    direction: Direction,
    system: System,
    cut_system: System,
    partition: DirectedBipartition,
    specified: StateSpecification,
    repertoire_distance: (
        DistributionMeasure
        | StateAwareMeasure
        | StatefulDistributionMeasure
        | CompositeMeasure
    ),
) -> RepertoireIrreducibilityAnalysis:
    """Compute the integration value for a single specified state."""
    mechanism = purview = system.node_indices
    if satisfies_composite_measure(repertoire_distance):
        partitioned_repertoire = cut_system.forward_repertoire(
            direction,
            mechanism,
            purview,
            specified.state,
        ).squeeze()[specified.state]
    else:
        partitioned_repertoire = cut_system.repertoire(
            direction, system.node_indices, system.node_indices
        )
    from pyphi.formalism.queries import evaluate_partition

    return evaluate_partition(
        system,
        direction,
        system.node_indices,
        system.node_indices,
        partition,  # pyright: ignore[reportArgumentType] - DirectedBipartition passed to JointBipartition param in IIT 4.0
        partitioned_repertoire=partitioned_repertoire,
        mechanism_measure=repertoire_distance,
        state=specified,
    )


def integration_value(
    direction: Direction,
    system: System,
    partition: DirectedBipartition,
    system_state: SystemStateSpecification,
    *,
    system_measure: CompositeMeasure,
    cut_system: System | None = None,
) -> RepertoireIrreducibilityAnalysis:
    """Compute the integration value for a partition along a direction.

    Evaluates against the spec stored at ``system_state[direction]``;
    tied specified states are handled at the orchestration layer (see
    :func:`sia`) by enumerating them and calling this function per
    candidate. ``system_measure`` is a Protocol-typed composite measure
    passed explicitly by the caller (no config fallback).

    The induced ``cut_system`` depends only on the partition, not the
    direction, so a caller evaluating both directions may build it once
    and pass it in to avoid rebuilding it per direction.
    """
    if cut_system is None:
        cut_system = system.apply_cut(partition)
    specified = system_state[direction]
    return _integration_value_for_state(
        direction,
        system,
        cut_system,
        partition,
        specified,
        system_measure,  # pyright: ignore[reportArgumentType]
    )


def intrinsic_differentiation_value(
    direction: Direction,
    system: System,
    specified_state: tuple[int, ...],
) -> float:
    """Intrinsic differentiation i_diff(s, s′) for the Eq. 23 cap.

    The forward repertoire is evaluated at the *specified* state s′
    (Mayner et al. 2026, Eqs. 4 and 6, with s′ per Eq. 12) — not at the
    current state. On the cause side the forward array holds the
    likelihoods p_c(s | s̄), which Eq. 11 normalizes into the posterior
    p←_c(s̄ | s) before the surprisal is taken; on the effect side the
    forward repertoire is already the conditional distribution p_e(s̄ | s).
    """
    mechanism = purview = system.node_indices

    unpartitioned_repertoire = repertoire.forward_repertoire(
        system,
        direction,
        mechanism,  # pyright: ignore[reportArgumentType]
        purview,  # pyright: ignore[reportArgumentType]
    )
    if direction == Direction.CAUSE:
        # Eq. 11: Bayes-normalize the cause likelihoods into p←_c(s̄ | s).
        # Divide into a fresh array — the repertoire is a shared cache entry.
        unpartitioned_repertoire = (
            unpartitioned_repertoire / unpartitioned_repertoire.sum()
        )

    return measures.distribution.intrinsic_differentiation(
        unpartitioned_repertoire,
        state=tuple(specified_state),
    )


def evaluate_partition(
    partition: DirectedBipartition,
    system: System,
    system_state: SystemStateSpecification,
    *,
    system_measure: CompositeMeasure,
    directions: Iterable[Direction] | None = None,
    intrinsic_differentiation: dict | None = None,
) -> SystemIrreducibilityAnalysis:
    """Evaluate a system-level partition and return the resulting SIA.

    ``system_measure`` is a Protocol-typed composite measure used at the
    system level; passed explicitly by the caller (no config fallback).
    Partition integration uses ``system_measure.partition_measure`` if
    set (otherwise ``system_measure`` itself), and the ``ii(s)`` cap
    (Eq. 23) is applied when ``system_measure.applies_ii_cap`` is True.

    ``intrinsic_differentiation`` depends only on ``(direction, system)``,
    not the partition; a caller evaluating many partitions of the same
    system may compute it once and pass it in to avoid rebuilding it per
    partition.
    """
    directions = fallback(directions, Direction.both())
    if directions is None:
        directions = Direction.both()
    directions = tuple(directions)
    validate.directions(directions)

    # Eqs. 19-20: partition integration uses the composite measure's
    # ``partition_measure`` (GID for II; self for GID). The ii(s) cap
    # (Eq. 23) is applied separately below.
    partition_distance: CompositeMeasure = (
        system_measure.partition_measure or system_measure
    )

    # The induced cut depends only on the partition, not the direction, so
    # build the cut System once and reuse it across directions.
    cut_system = system.apply_cut(partition)
    integration = {
        direction: integration_value(
            direction,
            system,
            partition,
            system_state,
            system_measure=partition_distance,
            cut_system=cut_system,
        )
        for direction in directions
    }

    if intrinsic_differentiation is None:
        intrinsic_differentiation = {
            direction: intrinsic_differentiation_value(
                direction, system, system_state[direction].state
            )
            for direction in directions
        }

    # Take the min over directions on the *signed* phi so the resulting
    # SIA's ``signed_phi`` metadata captures the raw preventative-cause
    # value when present. The canonical (clamped) ``phi`` is derived in
    # ``SystemIrreducibilityAnalysis.__post_init__`` via the |·|+ operator.
    # ``min`` and ``positive_part`` commute, so the clamped result is the
    # same as taking the min of clamped values.
    # numerics: exact — φ_s is defined as the minimum over directions.
    phi = min(integration[direction].signed_phi for direction in directions)

    # The Eq. 23 ii(s) cap is deliberately NOT applied here. Per the 2026
    # paper (Eqs. 21-23: the formalism "is the same as the IIT 4.0 definition
    # of φ_s ... until Equation (23)"), the minimum information partition is
    # selected on the *uncapped* normalized φ exactly as in IIT 4.0, and the
    # cap φ_s = min{φ_c, φ_e, ii(s)} is applied in ``sia`` to each SIA as
    # soon as its MIP is chosen (see ``_apply_ii_cap``).
    # ``intrinsic_differentiation`` and the system-state
    # ``intrinsic_information`` are carried on the SIA so the cap can be
    # applied there. Applying the cap per-partition would let it shift
    # which partition is the MIP — which can make the reported φ_s *exceed*
    # the 2023 value, contradicting the formalism.

    norm = normalization_factor(partition)
    normalized_phi = phi * norm

    result = SystemIrreducibilityAnalysis(
        phi=phi,
        normalized_phi=normalized_phi,
        cause=integration.get(Direction.CAUSE),
        effect=integration.get(Direction.EFFECT),
        partition=partition,
        system_state=system_state,
        current_state=system.proper_state,
        node_indices=system.node_indices,
        node_labels=system.node_labels,
        intrinsic_differentiation=intrinsic_differentiation,
    )
    return result


def _has_no_cause_or_effect(system_state):
    reasons = []
    for direction, reason in zip(
        Direction.both(),
        [NullResultReason.NO_CAUSE, NullResultReason.NO_EFFECT],
        strict=False,
    ):
        # Short-circuit for a direction with no cause/effect. Intrinsic
        # information that is zero up to ``config.numerics.precision`` —
        # including a floating-point residue of a mathematically-zero
        # ii(s) — specifies nothing, and forces φ_s = min over directions
        # to zero.
        spec = system_state[direction]
        if spec is not None and not numerics.is_positive(spec.intrinsic_information):
            reasons.append(reason)
    return reasons


def _cap_one(sia: SystemIrreducibilityAnalysis) -> None:
    """Apply the Eq. 23 cap to a single SIA's φ values, in place.

    The cap is taken on the raw ``signed_phi`` (so preventative-cause metadata
    is preserved) and the |·|+-clamped, normalized values are re-derived.
    Idempotent: re-applying with the same cap terms is a no-op.
    """
    ii = sia.intrinsic_information
    if ii is None:
        return
    # __post_init__ guarantees signed_phi is set (defaulting to phi) for any
    # constructed SIA, so it is non-None by the time the cap is applied.
    assert sia.signed_phi is not None
    # numerics: exact — Eq. 23 defines φ_s as a minimum, not a tie decision.
    capped_signed = min(float(sia.signed_phi), ii)
    norm = normalization_factor(sia.partition)
    capped_norm = capped_signed * norm if norm is not None else capped_signed
    sia.signed_phi = float(capped_signed)
    sia.phi = float(utils.positive_part(capped_signed))
    sia.signed_normalized_phi = float(capped_norm)
    sia.normalized_phi = float(utils.positive_part(capped_norm))
    # ii(s) is partition-independent given the specified states, so the same
    # state-level terms cap the runner-up partition's φ; the reported φ-gap
    # then compares capped to capped.
    # numerics: exact — Eq. 23 cap; φ_s is a minimum, not a tie decision.
    if sia.runner_up is not None and float(sia.runner_up.phi) > ii:
        runner_up_norm = normalization_factor(sia.runner_up.partition)
        capped_runner_up_normalized = (
            float(ii) * runner_up_norm
            if sia.runner_up.normalized_phi is not None and runner_up_norm is not None
            else sia.runner_up.normalized_phi
        )
        sia.runner_up = replace(
            sia.runner_up,
            phi=float(ii),
            normalized_phi=capped_runner_up_normalized,
        )


def _apply_ii_cap(
    sia: SystemIrreducibilityAnalysis,
) -> SystemIrreducibilityAnalysis:
    """Apply the IIT 4.0 (2026) intrinsic-information cap (Eq. 23).

    The MIP is selected on the uncapped normalized integrated information,
    exactly as in IIT 4.0 (Eqs. 21-22); this applies the cap
    ``φ_s = min{φ_c, φ_e, ii(s)}`` once per SIA after its MIP is chosen,
    where ``ii(s) = min_d min(i_spec_d, i_diff_d)`` is partition-independent
    given the (cause, effect) specified-state pair. The cap is applied to
    ``sia`` **and every member of its tie set**, each with its own cap
    terms, so tied SIAs stay mutually comparable — φ_s is the capped
    quantity by definition. Mutates and returns ``sia``.
    """
    members = {id(sia): sia}
    for tied in sia.ties or ():
        members.setdefault(id(tied), tied)
    for member in members.values():
        _cap_one(member)
    return sia


def sia(
    system: System,
    *,
    system_measure: CompositeMeasure,
    specification_measure: (
        DistributionMeasure
        | StateAwareMeasure
        | StatefulDistributionMeasure
        | CompositeMeasure
    ),
    directions: Iterable[Direction] | None = None,
    partition_scheme: str | None = None,
    partitions: Iterable | None = None,
    system_state: SystemStateSpecification | None = None,
    **kwargs,
) -> SystemIrreducibilityAnalysis:
    """Find the minimum information partition of a system.

    ``system_measure`` and ``specification_measure`` are Protocol-typed
    measure callables passed explicitly by the active formalism (no
    config fallback). ``system_measure`` drives system-level partition
    integration (and the ``ii(s)`` cap, if ``INTRINSIC_INFORMATION``);
    ``specification_measure`` drives the intrinsic-information
    computation of the system state.
    """
    partition_scheme = fallback(
        partition_scheme, config.formalism.iit.system_partition_scheme
    )
    assert partition_scheme is not None, "system_partition_scheme config must be set"

    # TODO: trivial reducibility

    # Check for degenerate cases
    # =========================================================================
    # Phi is necessarily zero if the system is:
    #   - not strongly connected;
    #   - empty;
    #   - an elementary micro mechanism (i.e. no nontrivial bipartitions).
    # So in those cases we immediately return a null SIA.
    def _null_sia(**kwargs):
        return NullSystemIrreducibilityAnalysis(
            system_state=system_state,
            node_indices=system.node_indices,
            node_labels=system.node_labels,
            **kwargs,
        )

    if not system:
        # system is empty
        return _null_sia(reasons=[NullResultReason.NO_SYSTEM])

    if not connectivity.is_strong(system.cm, system.node_indices):
        # system is not strongly connected
        return _null_sia(reasons=[NullResultReason.NO_STRONG_CONNECTIVITY])

    # Handle elementary micro mechanism cases.
    # Single macro element systems have nontrivial bipartitions because their
    #   bipartitions are over their micro elements.
    if len(system.partition_indices) == 1:
        # If the node lacks a self-loop, phi is trivially zero.
        if not system.cm[system.node_indices][system.node_indices]:
            return _null_sia(reasons=[NullResultReason.MONAD_WITH_NO_SELFLOOP])
        # Even if the node has a self-loop, we may still define phi to be zero.
        if not config.formalism.iit.single_micro_nodes_with_selfloops_have_phi:
            return _null_sia(
                reasons=[NullResultReason.MONAD_WITH_SELFLOOP_DEFINED_TO_BE_ZERO_PHI]
            )
    # =========================================================================

    if partitions is None:
        filter_func = None
        # Edge-cut schemes can yield cuts that leave the system strongly
        # connected; Eq. 14 requires the MIP search to consider only
        # disconnecting partitions.
        if getattr(
            system_partition_types[partition_scheme],
            "may_yield_non_disconnecting_cuts",
            False,
        ):

            def is_disconnecting_partition(partition):
                # Special case for length 1 systems so complete partition is included
                return (
                    not connectivity.is_strong(system.apply_cut(partition).proper_cm)
                ) or len(system) == 1

            filter_func = is_disconnecting_partition

        partitions = system_partitions(
            system.node_indices,
            node_labels=system.node_labels,
            partition_scheme=partition_scheme,
            filter_func=filter_func,
        )

    if system_state is None:
        system_state = system_intrinsic_information(
            system,
            specification_measure=specification_measure,
            directions=directions,
        )

    if config.formalism.iit.shortcircuit_sia:
        shortcircuit_reasons = _has_no_cause_or_effect(system_state)
        if shortcircuit_reasons:
            return _null_sia(reasons=shortcircuit_reasons)

    default_sia = _null_sia(reasons=[NullResultReason.NO_VALID_PARTITIONS])
    parallel_kwargs = conf.parallel_kwargs(
        dict(config.infrastructure.parallel_partition_evaluation), **kwargs
    )

    # Per Albantakis et al. 2023 S1: when cause/effect specified states
    # tie at maximum intrinsic information, the canonical winner is the
    # joint ``(cause_spec, effect_spec)`` pair whose system phi
    # ``φ_s = min(φ_c, φ_e)`` is greatest. The cascade enumerates the
    # Cartesian product of tied specs and selects via
    # :func:`resolve_ties.resolve_state_tie`. ``partitions`` is
    # materialized so each per-pair MIP search can iterate it
    # independently.
    if not isinstance(partitions, (list, tuple)):
        partitions = list(partitions)

    # A direction with no specified state (not requested, or null) contributes
    # a single ``None`` candidate so the cascade's Cartesian product ranges
    # over the other direction's ties alone.
    cause_specs = _spec_candidates(system_state.cause) or (None,)
    effect_specs = _spec_candidates(system_state.effect) or (None,)

    # Eq. 23 (2026): each SIA is capped by ii(s) as soon as its MIP is
    # chosen — the MIP itself is selected on *uncapped* normalized φ (Eqs.
    # 21-22), but everything downstream (the per-state cascade, the tie
    # set, the reported φ_s) sees the capped value, since φ_s is the capped
    # quantity by definition. Each tied specified-state pair carries its
    # own cap terms (i_diff depends on the specified state).
    apply_cap = getattr(system_measure, "applies_ii_cap", False)

    if len(cause_specs) <= 1 and len(effect_specs) <= 1:
        mip_sia = _find_mip_for_fixed_state(
            system=system,
            system_state=system_state,
            partitions=partitions,
            system_measure=system_measure,
            directions=directions,
            parallel_kwargs=parallel_kwargs,
            default_sia=default_sia,
        )
        if apply_cap:
            mip_sia = _apply_ii_cap(mip_sia)
    else:
        per_pair_sias: dict[tuple, SystemIrreducibilityAnalysis] = {}
        for c in cause_specs:
            for e in effect_specs:
                forced_state = _build_untied_system_state(c, e)
                key = (
                    c.state if c is not None else None,
                    e.state if e is not None else None,
                )
                pair_sia = _find_mip_for_fixed_state(
                    system=system,
                    system_state=forced_state,
                    partitions=partitions,
                    system_measure=system_measure,
                    directions=directions,
                    parallel_kwargs=parallel_kwargs,
                    default_sia=default_sia,
                )
                if apply_cap:
                    pair_sia = _apply_ii_cap(pair_sia)
                per_pair_sias[key] = pair_sia

        mip_sia = _resolve_pair_sias(system, per_pair_sias, system_state)

    if config.infrastructure.clear_system_caches_after_computing_sia:
        system.clear_caches()

    # n < 2: the n(n-1) system bound is trivially 0 and does not cover the
    # single-node self-loop phi convention
    # (``single_micro_nodes_with_selfloops_have_phi``), so it does not apply.
    n_units = len(system.node_indices)
    if config.infrastructure.validate_phi_bounds and n_units >= 2:
        from pyphi.formalism.iit4 import bounds

        bounds.check_phi_bound(
            mip_sia.phi,
            lambda: bounds.system_phi_upper_bound(n_units),
            system=system,
            label=f"SIA phi_s (nodes={tuple(system.node_indices)})",
        )

    return mip_sia


_sia = sia


def _resolve_pair_sias(
    system: System,
    per_pair_sias: dict,
    system_state: SystemStateSpecification,
) -> SystemIrreducibilityAnalysis:
    """Select the winning (cause, effect) specified-state pair per S1.

    ``per_pair_sias`` maps ``(cause_state, effect_state)`` keys to each
    pair's MIP-searched SIA. Applies the per-state max-min cascade at the
    Integration level, escalates φ_s ties to Composition, applies the
    canonical tie-break, and attaches the tie metadata. Shared by the
    in-process SIA search and the sharded-campaign merge, so both resolve
    state pairs identically.
    """
    # Apply the per-state max-min cascade at the Integration level.
    # The Composition step of the cascade requires a ``big_phi`` value
    # on each per-pair SIA (CES Φ), which the SIA does not carry; the
    # Integration budget keeps that key out of the resolver's read
    # path.
    ctx = resolve_ties.ResolutionContext(max_escalation_level="Integration")
    outcome = resolve_ties.resolve_state_tie(per_pair_sias, context=ctx)
    chosen_key = outcome.resolved
    tied_keys: list = list(outcome.tied_set or ())
    information_violation = False
    if chosen_key is None:
        assert tied_keys, "cascade outcome has neither winner nor ties"
        # S1: readings tied at φ_s escalate to Composition (Φ) — but only
        # among readings that exist. When every tied reading has φ_s = 0
        # the system is not a complex under any of them; nothing remains
        # for Φ to adjudicate, and the choice below is presentational.
        # A single-direction analysis has no cause-effect structure, so Φ
        # is undefined and the tie falls through to the canonical
        # representative.
        if (
            numerics.is_positive(float(per_pair_sias[tied_keys[0]].phi))
            and system_state.cause is not None
            and system_state.effect is not None
        ):
            winners, violation = _escalate_state_tie_to_composition(
                system, per_pair_sias, tied_keys
            )
            if violation:
                # Φ-tied readings with genuinely distinct structures: the
                # system does not comply with the information postulate
                # and does not qualify as a complex (S1).
                information_violation = True
            elif len(winners) == 1:
                chosen_key = winners[0]
            else:
                # Extrinsic Φ tie (intrinsically identical structures):
                # any representative is legitimate; fall through to the
                # canonical tie-break over the Φ-maximal subset.
                tied_keys = winners
    if information_violation:
        return NullSystemIrreducibilityAnalysis(
            system_state=system_state,
            node_indices=system.node_indices,
            node_labels=system.node_labels,
            reasons=[NullResultReason.NONUNIQUE_SYSTEM_STATE],
        )
    if chosen_key is None:
        # The canonical tie-break makes per-direction φ reporting
        # invariant under substrate relabeling. The specified states
        # are indexed over ``system.node_indices``, so
        # canonicalization only aligns with the substrate's node
        # space when the system spans it; for proper subsystems
        # (e.g. candidates in ``complexes()``) we keep the
        # deterministic enumeration-order representative.
        spans_substrate = len(system.node_indices) == system.substrate.tpm.n_nodes
        if spans_substrate:
            chosen_key = min(
                tied_keys,
                key=lambda key: _canonical_tie_break_key(system.substrate, key),
            )
        else:
            chosen_key = tied_keys[0]
    mip_sia = per_pair_sias[chosen_key]

    _restore_tie_metadata(
        mip_sia,
        original_cause=system_state.cause,
        original_effect=system_state.effect,
    )
    # Only readings genuinely tied with the winner at φ_s are ties; the
    # other evaluated specified-state readings lost the cascade outright.
    # numerics: tolerant — same equality the cascade's argmax uses.
    mip_sia.set_ties(
        tuple(
            pair_sia
            for pair_sia in per_pair_sias.values()
            if numerics.eq(float(pair_sia.phi), float(mip_sia.phi))
        )
    )
    return mip_sia


def sia_stride_search(
    system: System,
    partitions: Iterable,
    directions: Iterable[Direction] | None = None,
) -> tuple[str, Any]:
    """One campaign stride's SIA search: every (cause, effect) specified-state
    pair's **uncapped** MIP over ``partitions``.

    Returns ``("short_circuit", sia)`` when the search never consults the
    partitions (monad conventions, missing cause or effect under the
    short-circuit option), else ``("pairs", [(key, pair_sia), ...])`` with
    one uncapped local minimum per pair, keyed by ``(cause_state,
    effect_state)``. The pair enumeration is partition-independent, so
    every stride reports the same pair set. MIP selection compares
    *uncapped* normalized φ (Eqs. 21-22), so strides must not apply the
    intrinsic-information cap; :func:`merge_pair_minima` applies it once
    the cross-stride MIP per pair is chosen, exactly as the in-process
    search caps once its own MIP is chosen.
    """
    from pyphi.formalism.base import FORMALISM_REGISTRY
    from pyphi.formalism.iit4.formalism import _resolve_system_measures

    formalism = FORMALISM_REGISTRY[config.formalism.iit.version]
    system_measure, specification_measure = _resolve_system_measures(
        formalism, None, None
    )

    system_state = None

    def _null(**kwargs: Any) -> NullSystemIrreducibilityAnalysis:
        return NullSystemIrreducibilityAnalysis(
            system_state=system_state,
            node_indices=system.node_indices,
            node_labels=system.node_labels,
            **kwargs,
        )

    # Degenerate-case guards, mirroring :func:`sia`.
    if not system:
        return ("short_circuit", _null(reasons=[NullResultReason.NO_SYSTEM]))
    if not connectivity.is_strong(system.cm, system.node_indices):
        return (
            "short_circuit",
            _null(reasons=[NullResultReason.NO_STRONG_CONNECTIVITY]),
        )
    if len(system.partition_indices) == 1:
        if not system.cm[system.node_indices][system.node_indices]:
            return (
                "short_circuit",
                _null(reasons=[NullResultReason.MONAD_WITH_NO_SELFLOOP]),
            )
        if not config.formalism.iit.single_micro_nodes_with_selfloops_have_phi:
            return (
                "short_circuit",
                _null(
                    reasons=[NullResultReason.MONAD_WITH_SELFLOOP_DEFINED_TO_BE_ZERO_PHI]
                ),
            )
    system_state = system_intrinsic_information(
        system,
        specification_measure=specification_measure,
        directions=directions,
    )
    if config.formalism.iit.shortcircuit_sia:
        shortcircuit_reasons = _has_no_cause_or_effect(system_state)
        if shortcircuit_reasons:
            return ("short_circuit", _null(reasons=shortcircuit_reasons))
    default_sia = _null(reasons=[NullResultReason.NO_VALID_PARTITIONS])
    parallel_kwargs = conf.parallel_kwargs(
        dict(config.infrastructure.parallel_partition_evaluation)
    )
    partitions = list(partitions)
    pairs = []
    for c in _spec_candidates(system_state.cause) or (None,):
        for e in _spec_candidates(system_state.effect) or (None,):
            key = (
                c.state if c is not None else None,
                e.state if e is not None else None,
            )
            pair_sia = _find_mip_for_fixed_state(
                system=system,
                system_state=_build_untied_system_state(c, e),
                partitions=partitions,
                system_measure=system_measure,
                directions=directions,
                parallel_kwargs=parallel_kwargs,
                default_sia=default_sia,
            )
            pairs.append((key, pair_sia))
    return ("pairs", pairs)


def merge_pair_minima(
    system: System, merged_pairs: dict
) -> SystemIrreducibilityAnalysis:
    """Finish a sharded SIA from the per-pair cross-stride minima.

    The merge-side counterpart of the in-process search tail: applies the
    intrinsic-information cap (when the active system measure carries one)
    to each pair's globally minimal SIA, then runs the pair-selection
    cascade via :func:`_resolve_pair_sias`. A single-pair cell returns its
    capped minimum directly, matching the in-process single-pair fast
    path.
    """
    from pyphi.formalism.base import FORMALISM_REGISTRY
    from pyphi.formalism.iit4.formalism import _resolve_system_measures

    formalism = FORMALISM_REGISTRY[config.formalism.iit.version]
    system_measure, specification_measure = _resolve_system_measures(
        formalism, None, None
    )
    if getattr(system_measure, "applies_ii_cap", False):
        merged_pairs = {
            key: _apply_ii_cap(pair_sia) for key, pair_sia in merged_pairs.items()
        }
    if len(merged_pairs) == 1:
        return next(iter(merged_pairs.values()))
    system_state = system_intrinsic_information(
        system, specification_measure=specification_measure
    )
    return _resolve_pair_sias(system, merged_pairs, system_state)


def mechanism_state_pins(
    system: System,
    direction: Direction,
    mechanism: tuple[int, ...],
    purview: tuple[int, ...],
) -> tuple:
    """The specified-state pins a mechanism MIP search maximizes over.

    Partition-independent, so campaign shards enumerate the identical pin
    set the unsharded :func:`pyphi.formalism.queries.find_mip` uses.
    """
    from pyphi.measures.distribution import resolve_mechanism_measure

    specification_measure = resolve_mechanism_measure(
        config.formalism.iit.specification_measure
    )
    return system.intrinsic_information(
        direction,
        mechanism,
        purview,
        specification_measure=specification_measure,
    ).ties


def _escalate_state_tie_to_composition(
    system: System,
    per_pair_sias: dict,
    tied_keys: list,
) -> tuple[list, bool]:
    """Composition escalation for specified-state readings tied at φ_s > 0.

    Per Albantakis et al. (2023) S1, states tied at φ_s "need to be resolved
    in order to determine the system\'s cause-effect structure": the reading
    that maximizes the structure integrated information Φ is selected. The
    mechanism sweep is computed once and shared; each tied reading pays only
    its congruence resolution plus the closed-form relation sum.

    A Φ tie among readings whose structures are "actually identical from the
    intrinsic perspective" (isomorphic up to unit relabeling) is extrinsic
    and not a violation of the information postulate; a Φ tie among genuinely
    distinct structures is a violation, and the system does not qualify as a
    complex.

    Returns ``(winners, violation)``: ``winners`` is the Φ-maximal subset of
    ``tied_keys`` (a single key when Φ resolves the tie; the extrinsically
    tied subset otherwise), and ``violation`` is True for the strict
    non-isomorphic Φ tie.
    """
    from pyphi import automorphism
    from pyphi.relations import AnalyticalRelations

    unresolved_distinctions = iit3._compute_distinctions(system)
    structures: dict = {}
    big_phis: dict = {}
    for key in tied_keys:
        pair_sia = per_pair_sias[key]
        resolved = unresolved_distinctions.resolve_congruence(pair_sia.system_state)
        structure = CauseEffectStructure(
            sia=pair_sia,
            distinctions=resolved,
            relations=AnalyticalRelations(resolved),
        )
        structures[key] = structure
        big_phis[key] = float(structure.big_phi)
    best = max(big_phis.values())
    # numerics: tolerant — Φ-tier membership is a selection.
    winners = [key for key in tied_keys if numerics.eq(big_phis[key], best)]
    if len(winners) == 1:
        return winners, False
    first = structures[winners[0]]
    extrinsic = all(
        automorphism.are_structures_isomorphic(first, structures[key])
        for key in winners[1:]
    )
    return winners, not extrinsic


def _spec_candidates(state_spec: StateSpecification | None) -> tuple:
    """Return the tied set of state specs, or ``(state_spec,)`` if untied.

    Empty tuple when ``state_spec`` is ``None`` (degenerate / null state).
    """
    if state_spec is None:
        return ()
    return state_spec.ties if state_spec.ties else (state_spec,)


def _build_untied_system_state(
    cause: StateSpecification | None,
    effect: StateSpecification | None,
) -> SystemStateSpecification:
    """Build a ``SystemStateSpecification`` whose ``cause`` and ``effect``
    fields are the given specs with empty ``ties``. Used by the
    system-state cascade to force a single (cause, effect) pair through
    one MIP search.
    """
    cause_untied = replace(cause, _ties=()) if cause is not None else None
    effect_untied = replace(effect, _ties=()) if effect is not None else None
    return SystemStateSpecification(
        cause=cause_untied,  # pyright: ignore[reportArgumentType]
        effect=effect_untied,  # pyright: ignore[reportArgumentType]
    )


def _canonical_tie_break_key(substrate, key):
    """Permutation-invariant sort key for a tied ``(cause_state, effect_state)``
    pair, so per-direction φ reporting is deterministic up to relabeling.

    A ``None`` direction sorts first via the empty tuple. Used only for the
    residual cascade tie among reducible (``φ_s = 0``) states, where the system
    MIP is non-unique and enumeration order is otherwise label-dependent.
    """
    from pyphi.automorphism import canonical_state

    cause_state, effect_state = key
    return (
        canonical_state(substrate, cause_state) if cause_state is not None else (),
        canonical_state(substrate, effect_state) if effect_state is not None else (),
    )


def _restore_tie_metadata(
    sia_result: SystemIrreducibilityAnalysis,
    *,
    original_cause: StateSpecification | None,
    original_effect: StateSpecification | None,
) -> None:
    """Set ``sia_result.system_state.cause.ties`` and ``.effect.ties`` to
    the tied sets from ``original_cause`` / ``original_effect`` while
    keeping the chosen state values. Surfaces the full tied set to
    downstream consumers that inspect ``sia.system_state.cause.ties``.
    """
    if sia_result.system_state is None:
        return
    chosen_cause = sia_result.system_state.cause
    chosen_effect = sia_result.system_state.effect
    new_cause = chosen_cause
    new_effect = chosen_effect
    if chosen_cause is not None and original_cause is not None and original_cause.ties:
        new_cause = replace(chosen_cause, _ties=original_cause.ties)
    if (
        chosen_effect is not None
        and original_effect is not None
        and original_effect.ties
    ):
        new_effect = replace(chosen_effect, _ties=original_effect.ties)
    sia_result.system_state = replace(
        sia_result.system_state, cause=new_cause, effect=new_effect
    )


def _find_mip_for_fixed_state(
    *,
    system: System,
    system_state: SystemStateSpecification,
    partitions: Iterable,
    system_measure: CompositeMeasure,
    directions: Iterable[Direction] | None,
    parallel_kwargs: dict,
    default_sia: SystemIrreducibilityAnalysis,
) -> SystemIrreducibilityAnalysis:
    """Find the MIP for a given (fixed) system state.

    Runs the partition map-reduce with the supplied ``system_state``,
    selects the MIP via :func:`resolve_ties.sias`, and back-propagates
    the chosen state onto each tied MIP's ``system_state``.
    """
    resolved_directions = fallback(directions, Direction.both())
    if resolved_directions is None:
        resolved_directions = Direction.both()
    resolved_directions = tuple(resolved_directions)

    if not isinstance(partitions, (list, tuple)):
        partitions = list(partitions)

    # ``intrinsic_differentiation`` depends only on (direction, system, the
    # direction's specified state), not the partition, so compute it once here
    # and pass it to every partition rather than rebuilding it in each
    # ``evaluate_partition`` call.
    precomputed_intrinsic_differentiation = {
        direction: intrinsic_differentiation_value(
            direction, system, system_state[direction].state
        )
        for direction in resolved_directions
    }

    sias = map_reduce(
        evaluate_partition,
        partitions,
        map_kwargs={
            "system": system,
            "system_state": system_state,
            "system_measure": system_measure,
            "directions": resolved_directions,
            "intrinsic_differentiation": precomputed_intrinsic_differentiation,
        },
        shortcircuit_func=(
            utils.is_falsy
            if config.formalism.iit.shortcircuit_sia
            else _never_shortcircuit
        ),
        desc="Evaluating partitions",
        **parallel_kwargs,
    )

    candidates = list(sias) if sias is not None else []
    if not candidates:
        candidates = [default_sia]
    ties = tuple(resolve_ties.sias(candidates))
    mip_sia = ties[0]
    # Rank the runner-up by the same quantity that selected the MIP
    # (normalized φ under the default strategy), so the reported runner-up
    # is the actual nearest competitor.
    runner_up_key, key_is_normalized = sia_runner_up_key(
        config.formalism.iit.sia_tie_resolution
    )
    mip_sia.runner_up = runner_up_from_candidates(
        candidates,
        runner_up_key(mip_sia),
        key=runner_up_key,
        normalized=key_is_normalized,
    )
    others = [candidate for candidate in candidates if candidate is not mip_sia]
    # The margin is only meaningful when every partition was evaluated: a
    # short-circuited sweep yields a truncated prefix whose gap says nothing
    # about the full partition set.
    if others and len(candidates) == len(partitions):
        # Reports the margin to the nearest competitor; the MIP was already
        # selected above via resolve_ties.sias.
        # numerics: exact — reported margin, not a selection.
        gap = min(float(c.normalized_phi) for c in others) - float(
            mip_sia.normalized_phi
        )
        mip_sia.partition_margin = max(0.0, gap)
    for tied_mip in ties:
        tied_mip.resolve_system_state()
        tied_mip.set_ties(ties)
    return mip_sia


##############################################################################
# Composition
##############################################################################


class NullCauseEffectStructure(CauseEffectStructure):
    def __init__(self, **kwargs):
        kwargs.setdefault("sia", NullSystemIrreducibilityAnalysis())
        kwargs.setdefault("distinctions", ResolvedDistinctions([]))
        kwargs.setdefault("relations", ConcreteRelations([]))
        super().__init__(**kwargs)


def congruent_distinctions(
    system: System,
    *,
    specification_measure: (
        DistributionMeasure
        | StateAwareMeasure
        | StatefulDistributionMeasure
        | CompositeMeasure
    ),
    **kwargs: Any,
) -> Distinctions:
    """Return the system's distinctions without the system-partition search.

    The specified state used for congruence filtering is taken from the
    system's intrinsic information (Eq. 53) rather than from a system
    irreducibility analysis, so no system partition is evaluated.

    Returns
    -------
    ResolvedDistinctions
        When neither direction's specified state ties. These are exactly
        the distinctions of the system's :func:`ces`: a :func:`sia` over an
        untied specified state evaluates every partition against that same
        state and leaves it unchanged, so the cause-effect structure
        filters congruence against it too.
    UnresolvedDistinctions
        When either direction's specified state ties. The tie is broken by
        the φₛ cascade over the tied (cause, effect) pairs (Albantakis et
        al. 2023, S1 Text), which needs the partition search, so which
        distinctions are congruent is not determined here. Their number and
        Σφ_d are upper bounds on the cause-effect structure's.
    """
    distinctions = iit3._compute_distinctions(system, **kwargs)  # type: ignore[arg-type]
    system_state = system_intrinsic_information(
        system, specification_measure=specification_measure
    )
    tied = (
        len(_spec_candidates(system_state.cause)) > 1
        or len(_spec_candidates(system_state.effect)) > 1
    )
    if tied:
        return distinctions
    return distinctions.resolve_congruence(system_state)


def ces(
    system: System,
    *,
    system_measure: CompositeMeasure,
    specification_measure: (
        DistributionMeasure
        | StateAwareMeasure
        | StatefulDistributionMeasure
        | CompositeMeasure
    ),
    sia: SystemIrreducibilityAnalysis | None = None,
    distinctions: Distinctions | None = None,
    relations: Relations | None = None,
    sia_kwargs: dict | None = None,
    ces_kwargs: dict | None = None,
    relations_kwargs: dict | None = None,
) -> CauseEffectStructure:
    """Analyze the irreducible cause-effect structure of a system (Eq. 57).

    ``system_measure`` and ``specification_measure`` are Protocol-typed
    measure callables passed explicitly by the active formalism (no
    config fallback).
    """
    import time

    from pyphi.provenance import stamp_wall_time

    start = time.perf_counter()
    sia_kwargs = sia_kwargs or {}
    ces_kwargs = ces_kwargs or {}
    relations_kwargs = relations_kwargs or {}

    # Analyze irreducibility if not provided
    if sia is None:
        sia = _sia(
            system,
            system_measure=system_measure,
            specification_measure=specification_measure,
            **sia_kwargs,
        )

    # Compute distinctions if not provided
    if distinctions is None:
        # CES building dispatches find_mice through the active formalism;
        # the iit3 helper is reused because the outer mechanism x purview
        # iteration is the same in both formalisms.
        distinctions = iit3._compute_distinctions(system, **ces_kwargs)  # type: ignore[arg-type]
    # Resolve tied specified states against a SIA system_state. When the
    # SIA is null (degenerate substrate, e.g., not strongly connected)
    # we still want a usable bag of distinctions for diagnostics, so
    # fall back to the system's intrinsic-information state.
    if sia.system_state is not None:
        resolution_state = sia.system_state
    else:
        resolution_state = system_intrinsic_information(
            system, specification_measure=specification_measure
        )
    resolved_distinctions = distinctions.resolve_congruence(resolution_state)

    # Compute relations if not provided
    if relations is None:
        relations = compute_relations(resolved_distinctions, **relations_kwargs)

    result = CauseEffectStructure(
        sia=sia,
        distinctions=resolved_distinctions,
        relations=relations,
    )

    if config.infrastructure.validate_phi_bounds:
        from pyphi.formalism.iit4 import bounds

        n = len(system.node_indices)
        bounds.check_phi_bound(
            result.sum_phi_distinctions,
            lambda: bounds.sum_phi_distinctions_upper_bound(n),
            system=system,
            label="sum phi_distinctions",
        )
        bounds.check_phi_bound(
            result.sum_phi_relations,
            lambda: bounds.sum_phi_relations_upper_bound(n, bound="GENERAL"),
            system=system,
            label="sum phi_relations",
        )
        bounds.check_phi_bound(
            result.big_phi,
            lambda: bounds.big_phi_upper_bound(n, bound="GENERAL"),
            system=system,
            label="big_phi",
        )

    return stamp_wall_time(result, time.perf_counter() - start)
