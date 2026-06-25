"""Pin the built-in contents of every registry.

If a registrant module stops being imported during package initialization,
its registry loses entries and the matching assertion here fails loudly,
instead of a measure or scheme silently vanishing.
"""

import pytest

import pyphi

EXPECTED_REGISTRY_CONTENTS = {
    "partition.partition_types": {
        "JOINT_BIPARTITION",
        "JOINT_PARTITION_ALL",
        "WEDGE_TRIPARTITION",
    },
    "partition.system_partition_types": {
        "DIRECTED_BIPARTITION",
        "DIRECTED_BIPARTITION_CUT_ONE",
        "DIRECTED_BIPARTITION_SEQUENTIAL",
        "DIRECTED_SET_PARTITION",
        "EDGE_CUT_ALL",
        "EDGE_CUT_BIDIRECTIONAL",
        "TEMPORAL_DIRECTED_BIPARTITION",
        "TEMPORAL_DIRECTED_BIPARTITION_CUT_ONE",
    },
    "relations.relation_computations": {"ANALYTICAL", "CONCRETE"},
    "resolve_ties.phi_object_tie_resolution_strategies": {
        "NEGATIVE_NORMALIZED_PHI",
        "NEGATIVE_PHI",
        "NEGATIVE_PURVIEW_SIZE",
        "NONE",
        "NORMALIZED_PHI",
        "PARTITION_LEX",
        "PHI",
        "PURVIEW_SIZE",
    },
    "models.state_specification.distinction_phi_normalizations": {
        "NONE",
        "NUM_CONNECTIONS_CUT",
    },
    "measures.ces.measures": {"EMD", "SUM_SMALL_PHI"},
    "measures.distribution.distribution_measures": {
        "AID",
        "BLD",
        "EMD",
        "ENTROPY_DIFFERENCE",
        "ID",
        "KLD",
        "KLM",
        "L1",
        "MP2Q",
        "PSQ2",
    },
    "measures.distribution.stateful_distribution_measures": {
        "APMI",
        "IIT_4.0_SMALL_PHI",
        "IIT_4.0_SMALL_PHI_NO_ABSOLUTE_VALUE",
    },
    "measures.distribution.state_aware_measures": {"INTRINSIC_DIFFERENTIATION"},
    "measures.distribution.composite_measures": {
        "GENERALIZED_INTRINSIC_DIFFERENCE",
        "INTRINSIC_INFORMATION",
        "INTRINSIC_SPECIFICATION",
    },
    "measures.distribution.actual_causation_measures": {"PMI", "WPMI"},
    "formalism.actual_causation.compute.partitioned_repertoire_schemes": {"PRODUCT"},
    "formalism.actual_causation.compute.background_strategies": {"UNIFORM"},
    "formalism.actual_causation.compute.alpha_aggregations": {"SUBTRACTIVE"},
    "formalism.FORMALISM_REGISTRY": {"IIT_3_0", "IIT_4_0_2023", "IIT_4_0_2026"},
    "formalism.ACTUAL_CAUSATION_FORMALISM_REGISTRY": {"AC_2019"},
}


def _resolve(dotted: str):
    obj = pyphi
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


@pytest.mark.parametrize("dotted_path, expected", EXPECTED_REGISTRY_CONTENTS.items())
def test_registry_contents(dotted_path, expected):
    registry = _resolve(dotted_path)
    assert set(registry.keys()) == expected
