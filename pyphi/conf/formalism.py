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
    system_cuts: str = "3.0_STYLE"
    distinction_phi_normalization: str = "NUM_CONNECTIONS_CUT"
    relation_computation: str = "CONCRETE"
    state_tie_resolution: str = "PHI"
    mip_tie_resolution: list[str] = field(
        default_factory=lambda: ["NORMALIZED_PHI", "NEGATIVE_PHI"]
    )
    purview_tie_resolution: str = "PHI"
    shortcircuit_sia: bool = True
    single_micro_nodes_with_selfloops_have_phi: bool = True
