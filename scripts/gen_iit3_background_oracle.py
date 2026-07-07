"""Generate the PyPhi 1.x reference for cause-side background conditioning.

Reproducer for
``test/data/iit3-canonical/background_conditioning_oracle.json``, consumed
by ``test/integration/test_background_conditioning_oracle.py``.

Records, from a genuine PyPhi 1.2.0 install:

1. The cause repertoire of mechanism {A} over purview {B} for the
   proper-subset system S={A,B} of the 3-unit noisy-OR substrate
   (background W={C}, state (1,0,0)) — the value discriminates
   current-state conditioning of W (1.x) from IIT 4.0 Eq. 4 causal
   marginalization, and both predictions are stored alongside the
   observation.
2. The end-to-end IIT 3.0 SIA phi for that system.
3. SIA phi for every proper subset of the ``basic`` example network in
   state (1,0,0) — independent anchors for the complex-search values.

Control: reproduces the anchored ``basic`` full-substrate value
(2.3125 = 37/16) before the oracle is trusted.

Environment setup (isolated; does not touch the project venv)::

    VENV=/tmp/pyphi-1x-oracle/.venv
    uv venv --python 3.9 "$VENV"
    VIRTUAL_ENV="$VENV" uv pip install "pyphi==1.2.0"
    "$VENV/bin/python" scripts/gen_iit3_background_oracle.py \
        > test/data/iit3-canonical/background_conditioning_oracle.json

Run from a directory with no ``pyphi_config.yml`` so 1.x uses defaults plus
the flags set below (same flags as ``gen_iit3_emd_oracle.py``).
"""
# This reproducer targets the PyPhi 1.x API (pyphi.Network / Subsystem /
# compute), which does not exist in the 2.0 package, so pyright cannot
# resolve it here.
# pyright: reportCallIssue=false, reportAttributeAccessIssue=false

import json
import sys

import numpy as np

import pyphi

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


def p_a_on(b, c):
    return 0.9 if (b or c) else 0.1


def main():
    # Control: the anchored full-substrate basic value.
    basic = pyphi.examples.basic_network()
    control = float(
        pyphi.compute.sia(pyphi.Subsystem(basic, (1, 0, 0), range(basic.size))).phi
    )
    assert abs(control - 2.3125) < 1e-4, f"control failed: {control}"

    # Discriminating substrate: A = noisy OR of (B, C); B' = copy(A);
    # C' = 0.5 with no parents. State-by-node TPM, little-endian rows.
    rows = [
        (p_a_on(b, c), float(a), 0.5) for c in (0, 1) for b in (0, 1) for a in (0, 1)
    ]
    net = pyphi.Network(
        np.array(rows),
        cm=np.array([[0, 1, 0], [1, 0, 0], [1, 0, 0]]),
        node_labels=("A", "B", "C"),
    )
    sub = pyphi.Subsystem(net, (1, 0, 0), (0, 1))

    cause = np.asarray(sub.cause_repertoire((0,), (1,))).squeeze()
    # Predictions under each convention (uniform background prior for Eq. 4).
    w = np.array([0.5 * (p_a_on(0, c) + p_a_on(1, c)) for c in (0, 1)])
    w /= w.sum()
    eq4 = np.array([sum(p_a_on(b, c) * w[c] for c in (0, 1)) for b in (0, 1)])
    eq4 /= eq4.sum()
    legacy = np.array([p_a_on(b, 0) for b in (0, 1)])
    legacy /= legacy.sum()

    effect = np.asarray(sub.effect_repertoire((0,), (1,))).squeeze()
    sia = pyphi.compute.sia(sub)

    basic_subsets = {}
    for nodes in [(0, 1, 2), (1, 2), (0, 2), (0, 1)]:
        s = pyphi.Subsystem(basic, (1, 0, 0), nodes)
        subset_sia = pyphi.compute.sia(s)
        basic_subsets[str(nodes)] = {
            "phi": float(subset_sia.phi),
            "cut": str(subset_sia.cut),
            "n_concepts": len(subset_sia.ces),
        }

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
        },
        "effect_repertoire": {"observed": [float(x) for x in effect]},
        "sia": {
            "phi": float(sia.phi),
            "cut": str(sia.cut),
            "ces_size": len(sia.ces),
        },
        "basic_network_subsets": basic_subsets,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
