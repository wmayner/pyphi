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

# IIT 3.0 (Oizumi, Albantakis, Tononi 2014).
#
# Mirrors ``pyphi_config_3.0.yml``. Fields not listed here (the
# IIT-4.0-only ``specification_measure``, ``differentiation_measure``,
# ``distinction_phi_normalization``, ``relation_computation``, the
# specified-state ``state_tie_resolution``, ``system_phi_measure``, etc.)
# are unused on the IIT 3.0 code path and left at their library defaults.
iit3: dict[str, Any] = {
    "iit": IITConfig(
        version="IIT_3_0",
        # Distribution-distance measures used at the mechanism and CES
        # levels. The 2015 EMD-fix in ``_emd`` enforces inter-constellation-
        # only mass flow; see ``pyphi/measures/ces.py:194-213``.
        mechanism_phi_measure="EMD",
        ces_measure="EMD",
        # IIT 3.0 partition schemes.
        mechanism_partition_scheme="JOINT_BIPARTITION",
        system_partition_scheme="DIRECTED_BIPARTITION",
        # Paper-faithful: a single node with a self-loop does not generate
        # phi by itself.
        single_micro_nodes_with_selfloops_have_phi=False,
        # Two-step purview tie resolution: prefer larger phi, break
        # remaining ties by larger purview. Matches PyPhi 1.x's
        # ``pyphi_config_3.0.yml`` default.
        purview_tie_resolution=["PHI", "PURVIEW_SIZE"],
        mip_tie_resolution=["PHI", "PARTITION_LEX"],
        sia_tie_resolution=["PHI", "PARTITION_LEX"],
        # Paper-faithful: a cut can introduce a new concept; PyPhi does
        # not optimize this away.
        assume_partitions_cannot_create_new_concepts=False,
    ),
    "actual_causation": ActualCausationConfig(
        # PMI is the paper-canonical alpha measure for IIT 3.0 actual
        # causation (Albantakis et al. 2019).
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
