"""The set of golden fixtures.

Each fixture here is a (network, state, subsystem, config) tuple captured for
regression testing. To add a new fixture, define a ``GoldenFixture`` and append
to ``ALL_FIXTURES``. To regenerate the stored data after adding or modifying a
fixture::

    uv run pytest test/test_golden_regression.py --regenerate-golden -k <name>

Initial set covers the matrix dimensions:

- Same network under multiple formalism configs (verifies dispatch)
- Reference networks from the literature (basic, xor, rule110)
- A reducible network (low phi)
- IIT 3.0 vs 4.0 paths

Excluded from the initial set: tied-state cases that are flaky on develop
(see ``test_sia_big_subsys_all_complete_*``); these need P5/P6 work first.
"""

from __future__ import annotations

from pyphi import examples

from .fixture import GoldenFixture

# IIT 4.0 default config (current pyphi defaults)
IIT_4_CONFIG = {
    "IIT_VERSION": "4.0",
    "REPERTOIRE_DISTANCE": "GENERALIZED_INTRINSIC_DIFFERENCE",
    "SYSTEM_PARTITION_TYPE": "SET_UNI/BI",
    "PROGRESS_BARS": False,
    "PARALLEL": False,
}

# IIT 3.0 regression config
IIT_3_CONFIG = {
    "IIT_VERSION": "3.0",
    "REPERTOIRE_DISTANCE": "EMD",
    "PARTITION_TYPE": "BI",
    "SYSTEM_PARTITION_TYPE": "DIRECTED_BI",
    "ACTUAL_CAUSATION_MEASURE": "PMI",
    "PURVIEW_TIE_RESOLUTION": ["PHI", "PURVIEW_SIZE"],
    "PROGRESS_BARS": False,
    "PARALLEL": False,
}

# Skip layers that don't apply to a given formalism
SKIP_FOR_IIT_3 = frozenset({"phi_structure"})

ALL_FIXTURES: list[GoldenFixture] = [
    # ============== Basic 3-node network ==============
    # The standard 'basic' network under three configs to verify dispatch.
    GoldenFixture(
        name="basic_iit4_gid",
        description="Basic 3-node network (Marshall et al. 2023 Fig. 1 reference) "
        "in state (1,0,0), full subsystem, IIT 4.0 + GENERALIZED_INTRINSIC_DIFFERENCE.",
        config_overrides=IIT_4_CONFIG,
        network_factory=examples.basic_network,
        state=(1, 0, 0),
    ),
    GoldenFixture(
        name="basic_iit3_emd",
        description="Basic 3-node network in state (1,0,0), full subsystem, "
        "IIT 3.0 + EMD distance + BI partitions.",
        config_overrides=IIT_3_CONFIG,
        network_factory=examples.basic_network,
        state=(1, 0, 0),
        skip_layers=SKIP_FOR_IIT_3,
    ),
    GoldenFixture(
        name="basic_iit4_intrinsic_information",
        description="Basic 3-node network, IIT 4.0 + INTRINSIC_INFORMATION metric "
        "(distinct from GID; verifies metric dispatch).",
        config_overrides={
            **IIT_4_CONFIG,
            "REPERTOIRE_DISTANCE": "INTRINSIC_INFORMATION",
        },
        network_factory=examples.basic_network,
        state=(1, 0, 0),
    ),
    # ============== XOR ==============
    GoldenFixture(
        name="xor_iit4_gid",
        description="XOR 3-node network in state (0,0,0), IIT 4.0 + GID.",
        config_overrides=IIT_4_CONFIG,
        network_factory=examples.xor_network,
        state=(0, 0, 0),
    ),
    GoldenFixture(
        name="xor_iit3_emd",
        description="XOR 3-node network in state (0,0,0), IIT 3.0 + EMD.",
        config_overrides=IIT_3_CONFIG,
        network_factory=examples.xor_network,
        state=(0, 0, 0),
        skip_layers=SKIP_FOR_IIT_3,
    ),
    # ============== Reducible (low phi) ==============
    GoldenFixture(
        name="reducible_iit4_gid",
        description="Reducible network — fault lines should yield low phi. "
        "Regression target for the reducibility short-circuit logic.",
        config_overrides=IIT_4_CONFIG,
        network_factory=examples.disjunction_conjunction_network,
        state=(0, 0, 0, 0),
    ),
    # ============== rule110 (CA) ==============
    GoldenFixture(
        name="rule110_iit4_gid",
        description="Rule 110 cellular automaton, 3-node window, IIT 4.0 + GID.",
        config_overrides=IIT_4_CONFIG,
        network_factory=examples.rule110_network,
        state=(1, 0, 1),
    ),
    # ============== Simple network ==============
    GoldenFixture(
        name="grid3_iit4_gid",
        description="Grid3 3-node network in state (1,0,0), IIT 4.0 + GID. "
        "Symmetric architecture; useful for catching tie-resolution regressions.",
        config_overrides=IIT_4_CONFIG,
        network_factory=examples.grid3_network,
        state=(1, 0, 0),
    ),
]
