"""The set of golden fixtures.

Each fixture here is a (substrate, state, system, config) tuple captured for
regression testing. To add a new fixture, define a ``GoldenFixture`` and append
to ``ALL_FIXTURES``. To regenerate the stored data after adding or modifying a
fixture::

    uv run pytest test/test_golden_regression.py --regenerate-golden -k <name>

Naming convention: ``<substrate>_<formalism>``.

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

All five reference substrates (basic, xor, rule110, grid3, disjunction_conjunction)
get a fixture under each of iit4_2023 and iit4_2026 — even where SIA phi
collapses to 0 under both, the fixture still pins the per-mechanism MIPs and
PhiStructure shape, which can discriminate at lower layers.

Excluded from the initial set: tied-state cases that are flaky on develop
(see ``test_sia_big_subsys_all_complete_*``); these need P5/P6 work first.
"""

from __future__ import annotations

import itertools

import numpy as np

from pyphi import Substrate
from pyphi import examples

from .fixture import GoldenFixture


def _logistic_3node_k8() -> Substrate:
    """3-node fully-connected logistic substrate with weights 0.3 and k=8.

    Constructed deterministically (not from any external example). The
    state-by-node TPM is built by applying a sigmoid with steepness k=8
    to weighted inputs in {-1, +1}. At k=8 with the given weights, the
    substrate is "barely stochastic" — most transitions are near-deterministic
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
    return Substrate(tpm, cm)


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

# (substrate, factory, state, description-prefix) tuples used to generate
# fixtures across all three formalisms.
_NETWORKS: list[tuple[str, object, tuple[int, ...], str]] = [
    (
        "basic",
        examples.basic_substrate,
        (1, 0, 0),
        "Basic 3-node substrate (Marshall et al. 2023 Fig. 1 reference).",
    ),
    (
        "xor",
        examples.xor_substrate,
        (0, 0, 0),
        "XOR 3-node substrate. Symmetric, deterministic.",
    ),
    (
        "rule110",
        examples.rule110_substrate,
        (1, 0, 1),
        "Rule 110 cellular automaton, 3-node window.",
    ),
    (
        "grid3",
        examples.grid3_substrate,
        (1, 0, 0),
        # grid3 (1,0,0) is reducible under the paper-faithful reading:
        # multiple partitions yield |·|+ phi = 0. Before the |·|+ clamp was
        # applied, PyPhi reported SIA phi = -0.0729 ("preventative cause"
        # phi). After the clamp, phi = 0 with the signed value retained as
        # metadata (`signed_phi = -0.0729`). The fixture pins both values
        # so future changes to either side of the clamp surface here.
        "Grid3 3-node substrate. Symmetric architecture; useful for catching "
        "tie-resolution regressions. Tests the |·|+ clamp: phi = 0 (paper-"
        "faithful) with signed_phi = -0.0729 retained as preventative-cause "
        "metadata.",
    ),
    (
        "reducible",
        examples.disjunction_conjunction_substrate,
        (0, 0, 0, 0),
        "Disjunction-conjunction 4-node substrate. Reducible — phi collapses to 0; "
        "regression target for the reducibility short-circuit.",
    ),
]


def _make_fixtures() -> list[GoldenFixture]:
    """Build the cross-product of (substrate, formalism), plus targeted extras."""
    fixtures: list[GoldenFixture] = []

    # ============== Targeted extra fixtures ==============

    # System as strict subset of the substrate. Different code path:
    # system.cause_repertoire / effect_repertoire return substrate-shaped
    # arrays for non-empty mechanisms but system-shaped arrays for empty
    # mechanisms (max_entropy_distribution). The actual.py audit found a real
    # bug here (test_state_probability_strict_system); P7 will rewrite the
    # whole repertoire algebra and needs golden coverage of this case.
    fixtures.append(
        GoldenFixture(
            name="basic_subset_iit4_2023",
            description="basic_substrate with system = nodes (1, 2) only "
            "(strict subset of the 3-node substrate). Exercises the "
            "system-shaped repertoire path that the actual.py audit found "
            "buggy. Critical for P7's system rewrite.",
            config_overrides=IIT_4_2023_CONFIG,
            substrate_factory=examples.basic_substrate,
            state=(1, 0, 0),
            node_indices=(1, 2),
        )
    )

    # IIT 3.0 with TRI mechanism partitions. Different combinatorial path
    # from BI; exercised by test_prevention but no golden coverage today.
    fixtures.append(
        GoldenFixture(
            name="basic_iit3_emd_tri",
            description="basic_substrate IIT 3.0 + EMD with PARTITION_TYPE=TRI "
            "(tripartitions). Different combinatorial path than BI; supports "
            "the partition-algebra consolidation in P6.",
            config_overrides={**IIT_3_CONFIG, "PARTITION_TYPE": "TRI"},
            substrate_factory=examples.basic_substrate,
            state=(1, 0, 0),
            skip_layers=SKIP_FOR_IIT_3,
        )
    )

    # Symmetric IIT 3.0 strict-subset coverage. The actual.py shape bug
    # surfaced during the audit was an IIT 3.0 path issue; the underlying
    # shape inconsistency in pyphi/system.py:380 is shared across both
    # formalisms but the IIT 3.0 dispatch through compute.system.sia
    # exercises a different control-flow path than the IIT 4.0 case.
    fixtures.append(
        GoldenFixture(
            name="basic_subset_iit3_emd",
            description="basic_substrate with system = nodes (1, 2), "
            "IIT 3.0 + EMD. Symmetric subset-of-substrate coverage; "
            "exercises the IIT 3.0 SIA path on a strict-subset system.",
            config_overrides=IIT_3_CONFIG,
            substrate_factory=examples.basic_substrate,
            state=(1, 0, 0),
            node_indices=(1, 2),
            skip_layers=SKIP_FOR_IIT_3,
        )
    )

    # Partial 2026 cap activation. All other 2026 fixtures either match
    # the 2023 value (cap is no-op when phi is small/negative) or collapse
    # to 0 (cap dominates because some distinction has i_diff = 0 in
    # deterministic substrates). This logistic 3-node substrate at k=8 is
    # "barely stochastic" enough that all distinctions have positive
    # i_diff while phi remains larger than the smallest of them — so
    # the cap binds at a non-trivial intermediate value, which is the
    # most informative regime for testing the cap's min() logic.
    # Empirically (2026-05): phi_2023 = 0.0366, phi_2026 = 0.0032
    # (cap reduces by ~91%).
    fixtures.append(
        GoldenFixture(
            name="logistic3_k8_iit4_2023",
            description="3-node logistic substrate (sigmoid activation, k=8, "
            "weights 0.3) in state (0,0,0). IIT 4.0 (2023). Companion to "
            "logistic3_k8_iit4_2026 — together they exercise the partial "
            "ii(s) cap activation regime.",
            config_overrides=IIT_4_2023_CONFIG,
            substrate_factory=_logistic_3node_k8,
            state=(0, 0, 0),
            slow=True,
        )
    )
    fixtures.append(
        GoldenFixture(
            name="logistic3_k8_iit4_2026",
            description="3-node logistic substrate (sigmoid activation, k=8, "
            "weights 0.3) in state (0,0,0). IIT 4.0 (2026). The 2026 "
            "ii(s) cap activates *partially* here: phi_2023 ~ 0.037, "
            "phi_2026 ~ 0.003 — neither no-op nor full collapse. This is "
            "the regime that exercises the cap's min() logic at a "
            "non-edge point.",
            config_overrides=IIT_4_2026_CONFIG,
            substrate_factory=_logistic_3node_k8,
            state=(0, 0, 0),
            slow=True,
        )
    )

    # ============== Cross product of substrate x formalism ==============

    for net_name, factory, state, desc in _NETWORKS:
        # IIT 4.0 (2023) — paper formalism
        fixtures.append(
            GoldenFixture(
                name=f"{net_name}_iit4_2023",
                description=f"{desc} IIT 4.0 (Albantakis et al. 2023) + GID.",
                config_overrides=IIT_4_2023_CONFIG,
                substrate_factory=factory,  # type: ignore[arg-type]
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
                substrate_factory=factory,  # type: ignore[arg-type]
                state=state,
                slow=True,
            )
        )

    # IIT 3.0 only for the smaller binary substrates (4-node disjunction_conjunction
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
                substrate_factory=factory,  # type: ignore[arg-type]
                state=state,
                skip_layers=SKIP_FOR_IIT_3,
            )
        )

    return fixtures


ALL_FIXTURES: list[GoldenFixture] = _make_fixtures()
