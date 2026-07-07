"""PyPhi 1.x oracle: which cause-side background convention did 1.2.0 use?

Runs genuine PyPhi 1.2.0 on the discriminating fixture from
``verify_background.py`` (3-node substrate, system S={A,B}, background
W={C} at c_now=0) and compares the mechanism-{A}/purview-{B} cause
repertoire against the two predicted vectors:

  eq4    — causally marginalize W conditional on the current state
           (IIT 4.0 Eq. 4; what PyPhi 2.0's IIT 3.0 mode does)
  legacy — condition W at its current state (condition_tpm on externals)

Control: reproduces the anchored ``basic`` IIT 3.0 SIA phi (2.3125)
under the same config as ``scripts/gen_iit3_emd_oracle.py`` before the
oracle is trusted.

Run in an isolated PyPhi 1.2.0 env (see gen_iit3_emd_oracle.py docstring)::

    VENV=<scratch>/pyphi-1x-oracle/.venv
    uv venv --python 3.9 "$VENV"
    VIRTUAL_ENV="$VENV" uv pip install "pyphi==1.2.0"
    "$VENV/bin/python" pyphi_1x_background_oracle_runner.py > out.json

Run from a directory with no ``pyphi_config.yml`` so 1.x uses defaults
plus the flags set below.
"""
# This runner targets the PyPhi 1.x API (pyphi.Network / Subsystem /
# pyphi.compute), which does not exist in the 2.0 package, so pyright
# cannot resolve it here (same suppression as scripts/gen_iit3_emd_oracle.py).
# pyright: reportCallIssue=false, reportAttributeAccessIssue=false
import json
import sys

import numpy as np

import pyphi

# Same flags as scripts/gen_iit3_emd_oracle.py.
pyphi.config.PARALLEL_CONCEPT_EVALUATION = False
pyphi.config.PARALLEL_CUT_EVALUATION = False
pyphi.config.PARALLEL_COMPLEX_EVALUATION = False
pyphi.config.PROGRESS_BARS = False
pyphi.config.MEASURE = "EMD"
pyphi.config.USE_SMALL_PHI_DIFFERENCE_FOR_CES_DISTANCE = False
pyphi.config.PARTITION_TYPE = "BI"
pyphi.config.PRECISION = 6
pyphi.config.CUT_ONE_APPROXIMATION = False
pyphi.config.PICK_SMALLEST_PURVIEW = False


def control_basic():
    net = pyphi.examples.basic_network()
    sub = pyphi.Subsystem(net, (1, 0, 0), range(net.size))
    phi = float(pyphi.compute.sia(sub).phi)
    assert abs(phi - 2.3125) < 1e-4, f"control failed: basic sia.phi = {phi}"
    return phi


def pA1(b, c):
    return 0.9 if (b or c) else 0.1


def main():
    control = control_basic()

    # Discriminating fixture: p(A'=1|b,c) with parents B,C; B'=copy(A);
    # C'=0.5 no parents. State-by-node TPM, little-endian rows (A fastest).
    rows = []
    for c in (0, 1):
        for b in (0, 1):
            for a in (0, 1):
                rows.append((pA1(b, c), float(a), 0.5))
    net = pyphi.Network(
        np.array(rows),
        cm=np.array([[0, 1, 0], [1, 0, 0], [1, 0, 0]]),
        node_labels=("A", "B", "C"),
    )
    sub = pyphi.Subsystem(net, (1, 0, 0), (0, 1))  # S={A,B}, W={C}, c_now=0

    # (a) Repertoire-level discriminator: mechanism {A}, purview {B}.
    cause = np.asarray(sub.cause_repertoire((0,), (1,))).squeeze()

    w = np.array([0.5 * (pA1(0, c) + pA1(1, c)) for c in (0, 1)])
    w /= w.sum()
    eq4 = np.array([sum(pA1(b, c) * w[c] for c in (0, 1)) for b in (0, 1)])
    eq4 /= eq4.sum()
    legacy = np.array([pA1(b, 0) for b in (0, 1)])
    legacy /= legacy.sum()

    matches_eq4 = bool(np.allclose(cause, eq4))
    matches_legacy = bool(np.allclose(cause, legacy))

    # (b) Effect side: both conventions agree; B'=copy(A) with a_now=1 -> (0,1).
    effect = np.asarray(sub.effect_repertoire((0,), (1,))).squeeze()
    effect_expected = np.array([0.0, 1.0])
    effect_matches = bool(np.allclose(effect, effect_expected))

    # (c) End-to-end corroboration: 1.x IIT 3.0 SIA big phi under oracle config.
    sia = pyphi.compute.sia(sub)

    if matches_eq4 == matches_legacy:
        verdict = "INCONCLUSIVE: cause repertoire matches neither/both predictions"
    elif matches_legacy:
        verdict = (
            "PyPhi 1.2.0 conditions background units at their CURRENT state "
            "on the cause side (legacy convention)"
        )
    else:
        verdict = (
            "PyPhi 1.2.0 causally marginalizes background units conditional "
            "on the current state (Eq. 4 convention)"
        )

    out = {
        "oracle": f"pyphi=={pyphi.__version__}",
        "numpy": np.__version__,
        "python": sys.version,
        "config": {
            "MEASURE": "EMD",
            "PARTITION_TYPE": "BI",
            "PRECISION": 6,
            "PICK_SMALLEST_PURVIEW": False,
            "USE_SMALL_PHI_DIFFERENCE_FOR_CES_DISTANCE": False,
            "CUT_ONE_APPROXIMATION": False,
        },
        "control_basic_sia_phi": control,
        "control_passed": True,
        "fixture": {
            "state": [1, 0, 0],
            "system_nodes": [0, 1],
            "background_nodes": [2],
            "mechanism": [0],
            "purview": [1],
        },
        "cause_repertoire": {
            "observed": [float(x) for x in cause],
            "predicted_eq4_marginalized": [float(x) for x in eq4],
            "predicted_legacy_conditioned": [float(x) for x in legacy],
            "matches_eq4": matches_eq4,
            "matches_legacy": matches_legacy,
        },
        "effect_repertoire": {
            "observed": [float(x) for x in effect],
            "predicted_shared": [float(x) for x in effect_expected],
            "matches": effect_matches,
        },
        "sia": {
            "phi": float(sia.phi),
            "cut": str(sia.cut),
            "ces_size": len(sia.ces),
            "reference_2_0_marginalized_semantics": 0.4160651261,
            "reference_2_0_legacy_semantics_approx": 0.72,
            "note": (
                "2.0 references computed under presets.iit3, whose config may "
                "differ from the oracle flags; the repertoire-level result is "
                "the decisive discriminator."
            ),
        },
        "verdict": verdict,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
