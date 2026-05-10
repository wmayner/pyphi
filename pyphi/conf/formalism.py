"""Formalism layer of the PyPhi config.

Holds knobs that define the IIT mathematical formalism: which metric, which
partition scheme, which tie-resolution policy. Bundled into the
:class:`~pyphi.formalism.base.PhiFormalism` instance via composition; the
active formalism is rebuilt from the registry factory whenever this config
layer changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

_VALID_DISTINCTION_PHI_NORMALIZATION = frozenset({"NONE", "NUM_CONNECTIONS_CUT"})
_VALID_RELATION_COMPUTATION = frozenset({"CONCRETE", "ANALYTICAL"})


@dataclass(frozen=True)
class FormalismConfig:
    """Formalism-scoped configuration.

    These knobs collectively define what mathematical object PyPhi computes.
    They travel with each :class:`~pyphi.formalism.base.PhiFormalism`
    instance and are snapshotted onto every result object so reproducibility
    doesn't depend on the live global config.
    """

    formalism: str = "IIT_4_0_2023"
    assume_cuts_cannot_create_new_concepts: bool = False
    repertoire_distance: str = "GENERALIZED_INTRINSIC_DIFFERENCE"
    repertoire_distance_specification: str = "GENERALIZED_INTRINSIC_DIFFERENCE"
    repertoire_distance_differentiation: str = "GENERALIZED_INTRINSIC_DIFFERENCE"
    ces_distance: str = "SUM_SMALL_PHI"
    actual_causation_measure: str = "PMI"
    partition_type: str = "ALL"
    system_partition_type: str = "SET_UNI/BI"
    system_partition_include_complete: bool = False
    distinction_phi_normalization: str = "NUM_CONNECTIONS_CUT"
    relation_computation: str = "CONCRETE"
    state_tie_resolution: str = "PHI"
    mip_tie_resolution: list[str] = field(
        default_factory=lambda: ["NORMALIZED_PHI", "NEGATIVE_PHI"]
    )
    purview_tie_resolution: str = "PHI"
    shortcircuit_sia: bool = True
    single_micro_nodes_with_selfloops_have_phi: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.assume_cuts_cannot_create_new_concepts, bool):
            raise ValueError(
                "assume_cuts_cannot_create_new_concepts must be bool; "
                f"got {type(self.assume_cuts_cannot_create_new_concepts).__name__}"
            )
        if not isinstance(self.system_partition_include_complete, bool):
            raise ValueError(
                "system_partition_include_complete must be bool; "
                f"got {type(self.system_partition_include_complete).__name__}"
            )
        if not isinstance(self.shortcircuit_sia, bool):
            raise ValueError(
                "shortcircuit_sia must be bool; got "
                f"{type(self.shortcircuit_sia).__name__}"
            )
        if not isinstance(self.single_micro_nodes_with_selfloops_have_phi, bool):
            raise ValueError(
                "single_micro_nodes_with_selfloops_have_phi must be bool; "
                f"got {type(self.single_micro_nodes_with_selfloops_have_phi).__name__}"
            )
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
