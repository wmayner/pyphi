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

    Background conditioning (``background_conditioning``)
        How substrate units outside the candidate system (the background)
        enter cause repertoires when the system is a proper subset of its
        substrate:

        - ``"CAUSAL_MARGINALIZATION"`` (default): the background past is
          causally marginalized conditional on the current state — the
          "extended background" of IIT 4.0 (Albantakis et al. 2023, Eq. 4).
          Definitional for IIT 4.0.
        - ``"CONDITION_CURRENT_STATE"``: the background is fixed at its
          observed current state — the convention of PyPhi 1.x and the
          post-2014 IIT 3.0 literature. Selected by ``presets.iit3`` so
          that IIT 3.0 analyses of proper-subset systems reproduce
          published results.

        The IIT 3.0 paper itself (Oizumi et al. 2014, Box 1) fixes the
        background at its actual *past* state on the cause side. That
        convention requires the past state as an input, which no PyPhi
        version has ever taken, and it is not implemented.

        The effect side conditions the background at its current state
        under every convention, and full-substrate systems have no
        background, so the setting affects neither. Actual-causation
        analyses are unaffected: the AC background rule is set by
        ``ActualCausationConfig.background_scheme``.

    Reducibility short-circuiting (``shortcircuit_sia``)
        When ``True`` (default), IIT 4.0 analyses stop early on detected
        reducibility: a system whose specified state has no cause or no
        effect returns a null SIA without a partition search, and the
        system- and mechanism-level partition sweeps stop at the first
        partition with zero integrated information. Early stops leave
        selection margins undefined (``partition_margin`` is ``None``)
        because the remaining partitions were never evaluated. When
        ``False``, every partition is evaluated: reducible cases cost the
        full sweep, computed φ values are unchanged, and selection
        margins are exact everywhere. Does not gate IIT 3.0's early-exit
        logic.

    Distinction short-circuiting (``shortcircuit_distinctions``)
        When ``True`` (default), evaluating a distinction stops early on
        detected reducibility: if the effect direction has no candidate
        purviews, neither MICE search runs, and if the cause MICE comes
        out with φ = 0, the effect search is skipped. A skipped
        direction is a null MICE carrying the
        ``OTHER_DIRECTION_REDUCIBLE`` reason; its φ reads 0 as a
        placeholder (that direction's own maximal φ is unknown), and its
        selection margins and ties are absent. The distinction's φ —
        the minimum across directions — is unaffected, so cause-effect
        structures are identical either way; only the contents of
        zero-φ distinctions differ. When ``False``, both directions are
        always evaluated in full, with exact margins and complete ties.
        Applies to every formalism, including IIT 3.0 concepts.
    """

    version: str = "IIT_4_0_2026"
    mechanism_phi_measure: str = "GENERALIZED_INTRINSIC_DIFFERENCE"
    system_phi_measure: str = "INTRINSIC_INFORMATION"
    specification_measure: str = "GENERALIZED_INTRINSIC_DIFFERENCE"
    ces_measure: str = "SUM_SMALL_PHI"
    mechanism_partition_scheme: str = "JOINT_PARTITION_ALL"
    system_partition_scheme: str = "DIRECTED_SET_PARTITION"
    system_partition_include_total: bool = False
    distinction_phi_normalization: str = "NUM_CONNECTIONS_CUT"
    background_conditioning: str = "CAUSAL_MARGINALIZATION"
    relation_computation: str = "ANALYTICAL"
    assume_partitions_cannot_create_new_concepts: bool = False
    shortcircuit_sia: bool = True
    shortcircuit_distinctions: bool = True
    single_micro_nodes_with_selfloops_have_phi: bool = True
    state_tie_resolution: str = "PHI"
    mip_tie_resolution: Sequence[str] = field(
        default_factory=lambda: ("NORMALIZED_PHI", "NEGATIVE_PHI")
    )
    purview_tie_resolution: str | Sequence[str] = "PHI"
    sia_tie_resolution: Sequence[str] = field(
        default_factory=lambda: ("NORMALIZED_PHI", "NEGATIVE_PHI", "PARTITION_LEX")
    )

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
    alpha_measure: str = "PMI"
    # The partition family for actual-causation MIP search. JOINT_PARTITION_ALL
    # is the Albantakis et al. (2019) family (Eq. 7 + Fig. 3B: all partitions of
    # the occurrence, excluding the m=1 non-full-cut cases forbidden for
    # first-order occurrences). Other registered schemes are deliberate
    # variants — notably JOINT_BIPARTITION admits those m=1 partitions and so
    # yields alpha below the published values on first-order occurrences.
    mechanism_partition_scheme: str = "JOINT_PARTITION_ALL"
    partitioned_repertoire_scheme: str = "PRODUCT"
    background_scheme: str = "UNIFORM"
    alpha_aggregation: str = "SUBTRACTIVE"

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
