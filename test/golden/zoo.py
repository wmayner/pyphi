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

import itertools

import numpy as np

from pyphi import Network
from pyphi import examples

from .fixture import GoldenFixture


def _logistic_3node_k8() -> Network:
    """3-node fully-connected logistic network with weights 0.3 and k=8.

    Constructed deterministically (not from any external example). The
    state-by-node TPM is built by applying a sigmoid with steepness k=8
    to weighted inputs in {-1, +1}. At k=8 with the given weights, the
    network is "barely stochastic" — most transitions are near-deterministic
    but with enough remaining uncertainty that i_diff ends up in the regime
    where the 2026 ii(s) cap activates *partially*.
    """
    k = 8.0
    weights = np.full((3, 3), 0.3)
    cm = np.ones((3, 3), dtype=int)
    states = list(itertools.product([-1, 1], repeat=3))
    tpm = np.zeros((8, 3))
    for i, s in enumerate(states):
        for j in range(3):
            inp = sum(weights[ki, j] * s[ki] for ki in range(3))
            tpm[i, j] = 1.0 / (1.0 + np.exp(-k * inp))
    return Network(tpm, cm)


# IIT 4.0 (2023) — Albantakis et al. 2023, GID metric, no ii(s) cap.
IIT_4_2023_CONFIG = {
    "FORMALISM": "IIT_4_0_2023",
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
    "FORMALISM": "IIT_4_0_2026",
    "REPERTOIRE_DISTANCE": "INTRINSIC_INFORMATION",
}

# IIT 3.0 — Oizumi/Albantakis/Tononi 2014. Distribution-distance based.
IIT_3_CONFIG = {
    "FORMALISM": "IIT_3_0",
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
        # grid3 (1,0,0) currently produces SIA phi = -0.0729 because PyPhi
        # drops the |·|+ operator from Eqs. 19-20 of the IIT 4.0 paper
        # (Albantakis et al. 2023) for visibility into "preventative"-style
        # phi values. With negative phi allowed, the MIP selector
        # `sia_minimization_key = (normalized_phi, -phi)` at
        # new_big_phi/__init__.py:498 picks the *most* negative partition
        # rather than the one closest to zero — semantically misaligned with
        # MIP = "the partition that makes the least difference" (IIT 3.0 Box 1
        # Glossary; IIT 4.0 Eq. 23 + surrounding text). The paper-faithful
        # reading is: grid3 (1,0,0) is *reducible* (multiple partitions yield
        # |·|+ phi = 0), so SIA.phi should be 0 with the signed value (-0.0729)
        # captured as metadata. The fixture is intentionally retained as a
        # change-detection oracle pinning the current (pre-redesign) numbers
        # so any future fix lights up here and is regenerated deliberately.
        "Grid3 3-node network. Symmetric architecture; useful for catching "
        "tie-resolution regressions. Pins the current behavior where |·|+ "
        "is not enforced (see comment above).",
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
    """Build the cross-product of (network, formalism), plus targeted extras."""
    fixtures: list[GoldenFixture] = []

    # ============== Targeted extra fixtures ==============

    # Subsystem as strict subset of the network. Different code path:
    # subsystem.cause_repertoire / effect_repertoire return network-shaped
    # arrays for non-empty mechanisms but subsystem-shaped arrays for empty
    # mechanisms (max_entropy_distribution). The actual.py audit found a real
    # bug here (test_state_probability_strict_subsystem); P7 will rewrite the
    # whole repertoire algebra and needs golden coverage of this case.
    fixtures.append(
        GoldenFixture(
            name="basic_subset_iit4_2023",
            description="basic_network with subsystem = nodes (1, 2) only "
            "(strict subset of the 3-node network). Exercises the "
            "subsystem-shaped repertoire path that the actual.py audit found "
            "buggy. Critical for P7's subsystem rewrite.",
            config_overrides=IIT_4_2023_CONFIG,
            network_factory=examples.basic_network,
            state=(1, 0, 0),
            node_indices=(1, 2),
        )
    )

    # IIT 3.0 with TRI mechanism partitions. Different combinatorial path
    # from BI; exercised by test_prevention but no golden coverage today.
    fixtures.append(
        GoldenFixture(
            name="basic_iit3_emd_tri",
            description="basic_network IIT 3.0 + EMD with PARTITION_TYPE=TRI "
            "(tripartitions). Different combinatorial path than BI; supports "
            "the partition-algebra consolidation in P6.",
            config_overrides={**IIT_3_CONFIG, "PARTITION_TYPE": "TRI"},
            network_factory=examples.basic_network,
            state=(1, 0, 0),
            skip_layers=SKIP_FOR_IIT_3,
        )
    )

    # Symmetric IIT 3.0 strict-subset coverage. The actual.py shape bug
    # surfaced during the audit was an IIT 3.0 path issue; the underlying
    # shape inconsistency in pyphi/subsystem.py:380 is shared across both
    # formalisms but the IIT 3.0 dispatch through compute.subsystem.sia
    # exercises a different control-flow path than the IIT 4.0 case.
    fixtures.append(
        GoldenFixture(
            name="basic_subset_iit3_emd",
            description="basic_network with subsystem = nodes (1, 2), "
            "IIT 3.0 + EMD. Symmetric subset-of-network coverage; "
            "exercises the IIT 3.0 SIA path on a strict-subset subsystem.",
            config_overrides=IIT_3_CONFIG,
            network_factory=examples.basic_network,
            state=(1, 0, 0),
            node_indices=(1, 2),
            skip_layers=SKIP_FOR_IIT_3,
        )
    )

    # Partial 2026 cap activation. All other 2026 fixtures either match
    # the 2023 value (cap is no-op when phi is small/negative) or collapse
    # to 0 (cap dominates because some distinction has i_diff = 0 in
    # deterministic networks). This logistic 3-node network at k=8 is
    # "barely stochastic" enough that all distinctions have positive
    # i_diff while phi remains larger than the smallest of them — so
    # the cap binds at a non-trivial intermediate value, which is the
    # most informative regime for testing the cap's min() logic.
    # Empirically (2026-05): phi_2023 = 0.0366, phi_2026 = 0.0032
    # (cap reduces by ~91%).
    fixtures.append(
        GoldenFixture(
            name="logistic3_k8_iit4_2023",
            description="3-node logistic network (sigmoid activation, k=8, "
            "weights 0.3) in state (0,0,0). IIT 4.0 (2023). Companion to "
            "logistic3_k8_iit4_2026 — together they exercise the partial "
            "ii(s) cap activation regime.",
            config_overrides=IIT_4_2023_CONFIG,
            network_factory=_logistic_3node_k8,
            state=(0, 0, 0),
        )
    )
    fixtures.append(
        GoldenFixture(
            name="logistic3_k8_iit4_2026",
            description="3-node logistic network (sigmoid activation, k=8, "
            "weights 0.3) in state (0,0,0). IIT 4.0 (2026). The 2026 "
            "ii(s) cap activates *partially* here: phi_2023 ~ 0.037, "
            "phi_2026 ~ 0.003 — neither no-op nor full collapse. This is "
            "the regime that exercises the cap's min() logic at a "
            "non-edge point.",
            config_overrides=IIT_4_2026_CONFIG,
            network_factory=_logistic_3node_k8,
            state=(0, 0, 0),
        )
    )

    # ============== Cross product of network x formalism ==============

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
