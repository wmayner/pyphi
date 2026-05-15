"""Canonical IIT formalism presets.

Each preset is a :class:`dict` consumable by :meth:`config.override`:

- :data:`iit3` — IIT 3.0 (Oizumi et al. 2014).
- :data:`iit4_2023` — IIT 4.0 (Albantakis et al. 2023).
- :data:`iit4_2026` — IIT 4.0 with the intrinsic-information cap
  (Mayner, Marshall, Tononi 2026).

Usage::

    from pyphi import config, iit3

    with config.override(**iit3):
        ...  # computations use the IIT 3.0 formalism

A preset specifies its formalism sub-namespaces wholesale via
:class:`~pyphi.conf.formalism.IITConfig` and
:class:`~pyphi.conf.formalism.ActualCausationConfig` instances, so
applying a preset resets every field in those sub-namespaces to its
canonical value for that paper. Fields outside the formalism layer
(e.g. ``precision``) appear as additional top-level keys when the
paper's behavior depends on them.
"""

from __future__ import annotations

from typing import Any

from pyphi.conf.formalism import ActualCausationConfig
from pyphi.conf.formalism import IITConfig

iit3: dict[str, Any] = {
    "iit": IITConfig(
        version="IIT_3_0",
        mechanism_phi_measure="EMD",
        ces_measure="EMD",
        mechanism_partition_scheme="JOINT_BIPARTITION",
        system_partition_scheme="DIRECTED_BIPARTITION",
        single_micro_nodes_with_selfloops_have_phi=False,
        purview_tie_resolution=["PHI", "PURVIEW_SIZE"],
        assume_partitions_cannot_create_new_concepts=False,
    ),
    "actual_causation": ActualCausationConfig(
        alpha_measure="PMI",
    ),
    # pyemd's EMD comparison tolerance fails at finer precisions.
    "precision": 6,
}

iit4_2023: dict[str, Any] = {
    "iit": IITConfig(
        version="IIT_4_0_2023",
    ),
}

iit4_2026: dict[str, Any] = {
    "iit": IITConfig(
        version="IIT_4_0_2026",
        # System phi caps differentiation with specification per Eq. 23.
        system_phi_measure="INTRINSIC_INFORMATION",
    ),
}

__all__ = [
    "iit3",
    "iit4_2023",
    "iit4_2026",
]
