"""The set of golden fixtures.

Each fixture here is a (network, state, subsystem, config) tuple captured for
regression testing. To add a new fixture, define a ``GoldenFixture`` and append
to ``ALL_FIXTURES``. To regenerate the stored data after adding or modifying a
fixture::

    uv run pytest test/test_golden_regression.py --regenerate-golden -k <name>

Naming convention: ``<network>_<formalism>``.

Three formalisms are tested:

- ``iit3_emd``: IIT 3.0 (Oizumi, Albantakis, Tononi 2014). Distribution-distance
  based; uses EMD over full repertoires and BI mechanism partitions.
- ``iit4_2023``: IIT 4.0 paper formalism (Albantakis et al. 2023). State-centric
  ``ii(s) = informativeness * selectivity`` via ``GENERALIZED_INTRINSIC_DIFFERENCE``.
- ``iit4_2026``: 2026 refinement (Mayner, Marshall, Tononi 2026,
  ``papers/2026__mayner-et-al__intrinsic-cause-effect-power.pdf``). Adds the
  intrinsic-differentiation requirement: ``ii(s) = min(i_diff, i_spec)``,
  capping system phi by the requirement that a system must provide itself
  with alternative cause-effect states. Dispatched via
  ``REPERTOIRE_DISTANCE=INTRINSIC_INFORMATION`` which the recent commit
  ``c61d95d9`` re-purposed as a system-level mode flag (still uses GID for
  partition integration, but applies the ii(s) cap on phi).

All five reference networks (basic, xor, rule110, grid3, disjunction_conjunction)
get a fixture under each of iit4_2023 and iit4_2026 — even where SIA phi
collapses to 0 under both, the fixture still pins the per-mechanism MIPs and
PhiStructure shape, which can discriminate at lower layers.

Excluded from the initial set: tied-state cases that are flaky on develop
(see ``test_sia_big_subsys_all_complete_*``); these need P5/P6 work first.
"""

from __future__ import annotations

from pyphi import examples

from .fixture import GoldenFixture

# IIT 4.0 (2023) — Albantakis et al. 2023, GID metric, no ii(s) cap.
IIT_4_2023_CONFIG = {
    "IIT_VERSION": "4.0",
    "REPERTOIRE_DISTANCE": "GENERALIZED_INTRINSIC_DIFFERENCE",
    "SYSTEM_PARTITION_TYPE": "SET_UNI/BI",
    "PROGRESS_BARS": False,
    "PARALLEL": False,
}

# IIT 4.0 (2026) — Mayner, Marshall, Tononi 2026. ii(s) = min(i_diff, i_spec)
# cap on system phi. INTRINSIC_INFORMATION is a system-level mode flag in this
# regime (per commit c61d95d9): GID is still used for partition integration,
# but phi is capped by min_d(min(i_diff_d, i_spec_d)).
IIT_4_2026_CONFIG = {
    **IIT_4_2023_CONFIG,
    "REPERTOIRE_DISTANCE": "INTRINSIC_INFORMATION",
}

# IIT 3.0 — Oizumi/Albantakis/Tononi 2014. Distribution-distance based.
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

# (network, factory, state, description-prefix) tuples used to generate
# fixtures across all three formalisms.
_NETWORKS: list[tuple[str, object, tuple[int, ...], str]] = [
    (
        "basic",
        examples.basic_network,
        (1, 0, 0),
        "Basic 3-node network (Marshall et al. 2023 Fig. 1 reference).",
    ),
    (
        "xor",
        examples.xor_network,
        (0, 0, 0),
        "XOR 3-node network. Symmetric, deterministic.",
    ),
    (
        "rule110",
        examples.rule110_network,
        (1, 0, 1),
        "Rule 110 cellular automaton, 3-node window.",
    ),
    (
        "grid3",
        examples.grid3_network,
        (1, 0, 0),
        "Grid3 3-node network. Symmetric architecture; useful for catching "
        "tie-resolution regressions.",
    ),
    (
        "reducible",
        examples.disjunction_conjunction_network,
        (0, 0, 0, 0),
        "Disjunction-conjunction 4-node network. Reducible — phi collapses to 0; "
        "regression target for the reducibility short-circuit.",
    ),
]


def _make_fixtures() -> list[GoldenFixture]:
    """Build the cross-product of (network, formalism)."""
    fixtures: list[GoldenFixture] = []

    for net_name, factory, state, desc in _NETWORKS:
        # IIT 4.0 (2023) — paper formalism
        fixtures.append(
            GoldenFixture(
                name=f"{net_name}_iit4_2023",
                description=f"{desc} IIT 4.0 (Albantakis et al. 2023) + GID.",
                config_overrides=IIT_4_2023_CONFIG,
                network_factory=factory,  # type: ignore[arg-type]
                state=state,
            )
        )
        # IIT 4.0 (2026) — intrinsic differentiation cap
        fixtures.append(
            GoldenFixture(
                name=f"{net_name}_iit4_2026",
                description=(
                    f"{desc} IIT 4.0 (Mayner, Marshall, Tononi 2026) — adds "
                    "ii(s) = min(i_diff, i_spec) cap via INTRINSIC_INFORMATION mode."
                ),
                config_overrides=IIT_4_2026_CONFIG,
                network_factory=factory,  # type: ignore[arg-type]
                state=state,
            )
        )

    # IIT 3.0 only for the smaller binary networks (4-node disjunction_conjunction
    # under DIRECTED_BI is fine; we just don't add EMD coverage for 4-node).
    for net_name in ("basic", "xor"):
        net_idx = next(i for i, (n, *_) in enumerate(_NETWORKS) if n == net_name)
        _, factory, state, desc = _NETWORKS[net_idx]
        fixtures.append(
            GoldenFixture(
                name=f"{net_name}_iit3_emd",
                description=(
                    f"{desc} IIT 3.0 (Oizumi et al. 2014) + EMD + BI partitions."
                ),
                config_overrides=IIT_3_CONFIG,
                network_factory=factory,  # type: ignore[arg-type]
                state=state,
                skip_layers=SKIP_FOR_IIT_3,
            )
        )

    return fixtures


ALL_FIXTURES: list[GoldenFixture] = _make_fixtures()
