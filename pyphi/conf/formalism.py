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

from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from typing import Any

from pyphi.conf._helpers import yaml_repr

_VALID_DISTINCTION_PHI_NORMALIZATION = frozenset({"NONE", "NUM_CONNECTIONS_CUT"})
_VALID_RELATION_COMPUTATION = frozenset({"CONCRETE", "ANALYTICAL"})

_VALID_PARTITIONED_REPERTOIRE_SCHEMES = frozenset({"PRODUCT", "FORWARD_PROBABILITY"})
_VALID_BACKGROUND_STRATEGIES = frozenset({"UNIFORM", "STATIONARY", "OBSERVED"})
_VALID_ALPHA_AGGREGATIONS = frozenset({"SUBTRACTIVE", "RATIO"})


@dataclass(frozen=True)
class IITConfig:
    """IIT-formalism configuration sub-namespace."""

    version: str = "IIT_4_0_2023"
    mechanism_phi_measure: str = "GENERALIZED_INTRINSIC_DIFFERENCE"
    system_phi_measure: str = "GENERALIZED_INTRINSIC_DIFFERENCE"
    specification_measure: str = "GENERALIZED_INTRINSIC_DIFFERENCE"
    differentiation_measure: str = "INTRINSIC_DIFFERENTIATION"
    ces_measure: str = "SUM_SMALL_PHI"
    mechanism_partition_scheme: str = "ALL"
    system_partition_scheme: str = "SET_UNI/BI"
    system_partition_include_complete: bool = False
    distinction_phi_normalization: str = "NUM_CONNECTIONS_CUT"
    relation_computation: str = "CONCRETE"
    assume_partitions_cannot_create_new_concepts: bool = False
    shortcircuit_sia: bool = True
    single_micro_nodes_with_selfloops_have_phi: bool = True
    state_tie_resolution: str = "PHI"
    mip_tie_resolution: list[str] = field(
        default_factory=lambda: ["NORMALIZED_PHI", "NEGATIVE_PHI"]
    )
    purview_tie_resolution: str | list[str] = "PHI"
    sia_tie_resolution: list[str] = field(
        default_factory=lambda: ["NORMALIZED_PHI", "NEGATIVE_PHI", "PARTITION_LEX"]
    )

    __repr__ = yaml_repr

    def __post_init__(self) -> None:
        for name in (
            "assume_partitions_cannot_create_new_concepts",
            "system_partition_include_complete",
            "shortcircuit_sia",
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


@dataclass(frozen=True)
class ActualCausationConfig:
    """Actual-causation configuration sub-namespace.

    Decomposes the 2019 Albantakis et al. AC framework into its
    parameterized choices. Defaults match the published formalism;
    alternative registered values let users investigate variants.
    """

    alpha_measure: str = "PMI"
    mechanism_partition_scheme: str = "ALL"
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
        (currently only ``mechanism_partition_scheme``) are excluded — flat
        overrides on those names are ambiguous. To round-trip a
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
