"""Formalism layer of the PyPhi config.

Holds knobs that define the mathematical formalism — split into two
nested sub-namespaces:

- :class:`IITConfig` for IIT-formalism dispatch and IIT-specific knobs
  (which IIT version, which repertoire measure, which partition scheme,
  tie-resolution policy, etc.).
- :class:`ActualCausationConfig` for the actual-causation framework
  (which information measure, which partitioned-repertoire scheme,
  which background strategy, which alpha aggregation).

Bundled into the :class:`~pyphi.formalism.base.PhiFormalism` instance via
composition; the active formalism is rebuilt from the registry factory
whenever the IIT sub-config changes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from typing import Any

from pyphi.conf._helpers import yaml_repr

_VALID_DISTINCTION_PHI_NORMALIZATION = frozenset({"NONE", "NUM_CONNECTIONS_CUT"})
_VALID_RELATION_COMPUTATION = frozenset({"CONCRETE", "ANALYTICAL"})
_VALID_BACKGROUND_CONDITIONING = frozenset(
    {"CAUSAL_MARGINALIZATION", "CONDITION_CURRENT_STATE"}
)

_VALID_PARTITIONED_REPERTOIRE_SCHEMES = frozenset({"PRODUCT"})
_VALID_BACKGROUND_STRATEGIES = frozenset({"UNIFORM"})
_VALID_ALPHA_AGGREGATIONS = frozenset({"SUBTRACTIVE"})


@dataclass(frozen=True)
class IITConfig:
    """IIT-formalism configuration sub-namespace.

    Which IIT version computes results and the measures, partition schemes,
    background conditioning, short-circuiting, and tie-resolution policies
    it uses. Each attribute documents its own values; the presets
    ``pyphi.iit3``, ``pyphi.iit4_2023``, and ``pyphi.iit4_2026`` set them
    together so that a version never runs with options it does not define.
    """

    version: str = "IIT_4_0_2026"
    """Which IIT formalism computes results: ``"IIT_4_0_2026"`` (default;
    IIT 4.0 with the intrinsic-information requirement of Mayner et al.
    2026), ``"IIT_4_0_2023"`` (Albantakis et al. 2023), or ``"IIT_3_0"``
    (Oizumi et al. 2014). Select a version through a preset
    (``pyphi.iit3``, ``pyphi.iit4_2023``, ``pyphi.iit4_2026``) or the
    ``formalism=`` argument of :func:`pyphi.analyze`, which also set the
    measures and schemes the version requires; changing this option alone
    can leave the others at values the version rejects."""
    mechanism_phi_measure: str = "GENERALIZED_INTRINSIC_DIFFERENCE"
    """The distance between a mechanism's repertoire and its partitioned
    repertoire, from which a distinction's φ is computed. IIT 4.0 accepts
    ``"GENERALIZED_INTRINSIC_DIFFERENCE"`` (default) and
    ``"INTRINSIC_INFORMATION"``. IIT 3.0 accepts a distribution distance:
    ``"EMD"`` (the 2014 paper's earth mover's distance, set by the ``iit3``
    preset), ``"KLD"``, ``"L1"``, ``"ENTROPY_DIFFERENCE"``, ``"ID"``,
    ``"AID"``, ``"PSQ2"``, or ``"MP2Q"``. A version rejects a measure it
    does not define."""
    system_phi_measure: str = "INTRINSIC_INFORMATION"
    """How system integrated information φₛ is computed under IIT 4.0.
    ``"INTRINSIC_INFORMATION"`` (default) applies the intrinsic-information
    requirement, φₛ = min(φ_c, φ_e, ii(s)) (Mayner et al. 2026, Eq. 23);
    ``"GENERALIZED_INTRINSIC_DIFFERENCE"`` gives the 2023 definition,
    φₛ = min(φ_c, φ_e), without it. IIT 3.0 does not read this option."""
    specification_measure: str = "GENERALIZED_INTRINSIC_DIFFERENCE"
    """The measure that selects the specified state, the purview or system
    state with maximal intrinsic information, under IIT 4.0:
    ``"GENERALIZED_INTRINSIC_DIFFERENCE"`` (default),
    ``"INTRINSIC_INFORMATION"``, or ``"INTRINSIC_SPECIFICATION"``. IIT 3.0
    does not read this option."""
    ces_measure: str = "SUM_SMALL_PHI"
    """The distance between the unpartitioned and partitioned cause-effect
    structures, which is IIT 3.0's system-level Φ: ``"SUM_SMALL_PHI"``
    (default; the summed φ of the concepts the partition destroys or
    changes) or ``"EMD"`` (the 2014 paper's earth mover's distance over
    concept space, set by the ``iit3`` preset). IIT 4.0 accepts only
    ``"SUM_SMALL_PHI"``."""
    mechanism_partition_scheme: str = "JOINT_PARTITION_ALL"
    """How a mechanism and its purview are partitioned when a distinction's
    irreducibility is evaluated: ``"JOINT_PARTITION_ALL"`` (default; every
    partition of the mechanism and purview into any number of parts, the
    IIT 4.0 scheme), ``"JOINT_BIPARTITION"`` (bipartitions only, the IIT
    3.0 scheme, set by the ``iit3`` preset), or ``"WEDGE_TRIPARTITION"``
    (bipartitions of the mechanism with a third part cut from the
    purview)."""
    system_partition_scheme: str = "DIRECTED_SET_PARTITION"
    """How a system is partitioned when φₛ is evaluated:
    ``"DIRECTED_SET_PARTITION"`` (default; every set partition of the units
    with a direction assigned to each part, the IIT 4.0 scheme),
    ``"DIRECTED_BIPARTITION"`` (the IIT 3.0 scheme, set by the ``iit3``
    preset), ``"DIRECTED_BIPARTITION_CUT_ONE"``,
    ``"DIRECTED_BIPARTITION_SEQUENTIAL"``, ``"EDGE_CUT_ALL"``, or
    ``"EDGE_CUT_BIDIRECTIONAL"``. IIT 3.0 accepts only the two directed
    bipartition schemes. Under IIT 4.0 a non-default scheme computes a
    well-defined φₛ for that scheme, which is not the papers' value."""
    system_partition_include_total: bool = False
    """Whether the system partition search includes the total partition,
    which severs every connection (default ``False``). A single-unit system
    always includes it, since it has no other partition."""
    distinction_phi_normalization: str = "NUM_CONNECTIONS_CUT"
    """How a distinction's φ is normalized before candidate partitions are
    compared: ``"NUM_CONNECTIONS_CUT"`` (default; divided by the number of
    connections the partition severs, the IIT 4.0 rule) or ``"NONE"`` (the
    raw value, set by the ``iit3`` preset)."""
    background_conditioning: str = "CAUSAL_MARGINALIZATION"
    """How the units outside a candidate system (its background) enter its
    cause repertoires when the system is a proper subset of the substrate:
    ``"CAUSAL_MARGINALIZATION"`` (default; the background's past is causally
    marginalized conditional on the current state, the extended background
    of IIT 4.0, Albantakis et al. 2023, Eq. 4) or
    ``"CONDITION_CURRENT_STATE"`` (the background is fixed at its observed
    current state, the PyPhi 1.x convention, set by the ``iit3`` preset and
    the only value IIT 3.0 accepts). The effect side conditions the
    background at its current state under both, and a system that is the
    whole substrate has no background, so neither is affected."""
    relation_computation: str = "ANALYTICAL"
    """How the relations of a Φ-structure are computed: ``"ANALYTICAL"``
    (default; relation counts and φ sums in closed form, without
    enumerating relations, so individual relations cannot be listed) or
    ``"CONCRETE"`` (every relation enumerated, which grows exponentially
    with the number of distinctions). See :mod:`pyphi.relations`."""
    assume_partitions_cannot_create_new_concepts: bool = False
    """IIT 3.0 only. When ``True``, evaluating a system partition considers
    only the mechanisms that were concepts in the unpartitioned system,
    which is faster but misses concepts a partition creates. Default
    ``False``."""
    shortcircuit_sia: bool = True
    """When ``True`` (default), IIT 4.0 analyses stop early on detected
    reducibility: a system whose specified state has no cause or no effect
    returns a null result without a partition search, and the system- and
    mechanism-level partition sweeps stop at the first partition with zero
    integrated information. Computed φ values are unchanged; early stops
    leave the selection margins undefined, since the remaining partitions
    were never evaluated. When ``False``, every partition is evaluated and
    the margins are exact. Does not affect IIT 3.0's own early exits."""
    shortcircuit_distinctions: bool = True
    """When ``True`` (default), evaluating a distinction stops early on
    detected reducibility: if the effect direction has no candidate
    purviews, neither search runs, and if the cause's maximally irreducible
    purview has φ = 0, the effect search is skipped. The distinction's φ,
    the minimum over the two directions, is unchanged, so cause-effect
    structures are identical either way; only the contents of zero-φ
    distinctions differ (the skipped direction is a null result without
    margins or ties). Applies to every formalism."""
    single_micro_nodes_with_selfloops_have_phi: bool = True
    """Whether a single-unit system whose unit has a self-connection can
    have positive φₛ (default ``True``). When ``False``, such systems have
    φₛ = 0 by definition, the PyPhi 1.x convention set by the ``iit3``
    preset. A single unit without a self-connection has φₛ = 0 regardless."""
    state_tie_resolution: str = "PHI"
    """How a tie among candidate specified states is broken: a strategy name
    or a list applied in order, keeping the candidates that are extremal
    under each. ``"PHI"`` (default) keeps the states with maximal φ, and
    the tie survives in the result if several remain. Other strategies:
    ``"NORMALIZED_PHI"``, ``"PURVIEW_SIZE"``, their ``"NEGATIVE_*"``
    forms, ``"PARTITION_LEX"``, and ``"NONE"`` (keep all). See
    :doc:`/howto/tie-breaking`."""
    mip_tie_resolution: Sequence[str] = field(
        default_factory=lambda: ("NORMALIZED_PHI", "NEGATIVE_PHI")
    )
    """How a tie among a mechanism's candidate partitions at the minimum is
    broken, a list of strategies applied in order. The default
    ``("NORMALIZED_PHI", "NEGATIVE_PHI")`` keeps the partitions with the
    smallest normalized φ, then those with the largest raw φ; the ``iit3``
    preset uses ``("PHI", "PARTITION_LEX")``. See
    :doc:`/howto/tie-breaking`."""
    purview_tie_resolution: str | Sequence[str] = "PHI"
    """How a tie among purviews with maximal φ is broken, a strategy name or
    a list applied in order. ``"PHI"`` (default) keeps every purview at the
    maximum, so the tie is reported on the result; the ``iit3`` preset uses
    ``("PHI", "PURVIEW_SIZE")``, which then keeps the largest purview, the
    PyPhi 1.x convention. See :doc:`/howto/tie-breaking`."""
    sia_tie_resolution: Sequence[str] = field(
        default_factory=lambda: ("NORMALIZED_PHI", "NEGATIVE_PHI", "PARTITION_LEX")
    )
    """How a tie among system partitions at the minimum is broken, a list of
    strategies applied in order. The default ``("NORMALIZED_PHI",
    "NEGATIVE_PHI", "PARTITION_LEX")`` keeps the partitions with the
    smallest normalized φ, then the largest raw φ, then the first in
    lexicographic order, so one minimum partition is always
    selected; a ``partition_margin`` of zero on the result records that
    it was tied. The ``iit3`` preset uses ``("PHI", "PARTITION_LEX")``.
    See :doc:`/howto/tie-breaking`."""

    __repr__ = yaml_repr

    def __post_init__(self) -> None:
        # A frozen config must not share mutable containers with callers,
        # presets, or snapshots; sequence-valued fields are stored as tuples.
        for name in (
            "mip_tie_resolution",
            "purview_tie_resolution",
            "sia_tie_resolution",
        ):
            value = getattr(self, name)
            if isinstance(value, list):
                object.__setattr__(self, name, tuple(value))
        for name in (
            "assume_partitions_cannot_create_new_concepts",
            "system_partition_include_total",
            "shortcircuit_sia",
            "shortcircuit_distinctions",
            "single_micro_nodes_with_selfloops_have_phi",
        ):
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be bool; got {type(value).__name__}")
        if (
            self.distinction_phi_normalization
            not in _VALID_DISTINCTION_PHI_NORMALIZATION
        ):
            raise ValueError(
                f"distinction_phi_normalization={self.distinction_phi_normalization!r} "
                f"not in {sorted(_VALID_DISTINCTION_PHI_NORMALIZATION)}"
            )
        if self.relation_computation not in _VALID_RELATION_COMPUTATION:
            raise ValueError(
                f"relation_computation={self.relation_computation!r} "
                f"not in {sorted(_VALID_RELATION_COMPUTATION)}"
            )
        if self.background_conditioning not in _VALID_BACKGROUND_CONDITIONING:
            raise ValueError(
                f"background_conditioning={self.background_conditioning!r} "
                f"not in {sorted(_VALID_BACKGROUND_CONDITIONING)}"
            )


@dataclass(frozen=True)
class ActualCausationConfig:
    """Actual-causation configuration sub-namespace.

    Decomposes the 2019 Albantakis et al. AC framework into its
    parameterized choices. Defaults match the published formalism;
    alternative registered values let users investigate variants.
    """

    version: str = "AC_2019"
    """The actual-causation formalism: ``"AC_2019"`` (Albantakis et al. 2019)
    is the only registered version."""
    alpha_measure: str = "PMI"
    """The pointwise information measure behind a causal link's strength α:
    ``"PMI"`` (default; pointwise mutual information, the 2019 paper) or
    ``"WPMI"`` (pointwise mutual information weighted by the probability of
    the occurrence)."""
    mechanism_partition_scheme: str = "JOINT_PARTITION_ALL"
    """The family of partitions searched for an occurrence's minimum
    information partition: ``"JOINT_PARTITION_ALL"`` (default; the 2019
    paper's family, Eq. 7 and Fig. 3B, which excludes the single-part cuts
    the paper forbids for first-order occurrences), ``"JOINT_BIPARTITION"``
    (a variant that admits those cuts and so gives lower α on first-order
    occurrences), or ``"WEDGE_TRIPARTITION"``. Read independently of the
    IIT option of the same name."""
    partitioned_repertoire_scheme: str = "PRODUCT"
    """How the partitioned repertoire of an occurrence is formed:
    ``"PRODUCT"`` (the product of the parts' repertoires) is the only
    registered value."""
    background_scheme: str = "UNIFORM"
    """How units outside the transition are treated: ``"UNIFORM"``
    (marginalized under a uniform distribution) is the only registered
    value."""
    alpha_aggregation: str = "SUBTRACTIVE"
    """How α is obtained from the unpartitioned and partitioned information:
    ``"SUBTRACTIVE"`` (their difference) is the only registered value."""

    __repr__ = yaml_repr

    def __post_init__(self) -> None:
        if (
            self.partitioned_repertoire_scheme
            not in _VALID_PARTITIONED_REPERTOIRE_SCHEMES
        ):
            raise ValueError(
                f"partitioned_repertoire_scheme={self.partitioned_repertoire_scheme!r} "
                f"not in {sorted(_VALID_PARTITIONED_REPERTOIRE_SCHEMES)}"
            )
        if self.background_scheme not in _VALID_BACKGROUND_STRATEGIES:
            raise ValueError(
                f"background_scheme={self.background_scheme!r} "
                f"not in {sorted(_VALID_BACKGROUND_STRATEGIES)}"
            )
        if self.alpha_aggregation not in _VALID_ALPHA_AGGREGATIONS:
            raise ValueError(
                f"alpha_aggregation={self.alpha_aggregation!r} "
                f"not in {sorted(_VALID_ALPHA_AGGREGATIONS)}"
            )


@dataclass(frozen=True)
class FormalismConfig:
    """Formalism-scoped configuration.

    Thin holder of :class:`IITConfig` and :class:`ActualCausationConfig`.
    Both travel with each :class:`~pyphi.formalism.base.PhiFormalism`
    instance and are snapshotted onto every result object.
    """

    iit: IITConfig = field(default_factory=IITConfig)
    actual_causation: ActualCausationConfig = field(
        default_factory=ActualCausationConfig
    )

    __repr__ = yaml_repr

    def __post_init__(self) -> None:
        if not isinstance(self.iit, IITConfig):
            raise ValueError(f"iit must be IITConfig; got {type(self.iit).__name__}")
        if not isinstance(self.actual_causation, ActualCausationConfig):
            raise ValueError(
                f"actual_causation must be ActualCausationConfig; "
                f"got {type(self.actual_causation).__name__}"
            )

    def as_kwargs(self) -> dict[str, Any]:
        """Return a flat dict of leaf-field name to value for ``override(**...)``.

        Field names that collide between the IIT and AC sub-namespaces
        (e.g. ``version``, ``mechanism_partition_scheme``) are excluded —
        flat overrides on those names are ambiguous. To round-trip a
        colliding-name change, set the sub-namespace wholesale via
        ``replace(formalism, iit=...)`` or ``config.iit = ...``.
        """
        from pyphi.conf._field_routing import colliding_formalism_fields

        excluded = colliding_formalism_fields()
        out: dict[str, Any] = {}
        for sub_name in ("iit", "actual_causation"):
            sub_layer = getattr(self, sub_name)
            for f in fields(sub_layer):
                if f.name in excluded:
                    continue
                out[f.name] = getattr(sub_layer, f.name)
        return out
