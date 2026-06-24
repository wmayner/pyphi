"""Generate the independent PyPhi 1.x IIT 3.0 EMD reference for rule110 / grid3.

This script is the *reproducer* for the canonical references
``test/data/iit3-canonical/rule110_sia_phi_canonical.json`` and
``test/data/iit3-canonical/grid3_sia_phi_canonical.json``, consumed by
``test/test_golden_regression.py::test_iit3_emd_sia_phi_matches_pyphi_1x_reference``.

Why this exists: the ``rule110_iit3_emd`` and ``grid3_iit3_emd`` golden fixtures
are not published worked examples (unlike ``basic``), so the only independent
anchor for their IIT 3.0 SIA phi is a genuine PyPhi 1.x computation. Their sia.phi
also depends on the load-bearing ``PURVIEW_SIZE`` purview tie-break, so the
reference records both tie-break resolutions to document the sensitivity.

It MUST be run against a genuine PyPhi 1.x install (it uses the pre-2.0 API:
``pyphi.Network`` / ``pyphi.Subsystem`` / ``pyphi.compute.sia`` and the flat
UPPERCASE config). It lives under ``scripts/`` (outside the pytest ``testpaths``)
so the current 2.0 suite never imports it.

Environment setup (isolated, does not touch the project venv)::

    uv venv --python 3.9 /tmp/pyphi-1x-oracle/.venv
    VIRTUAL_ENV=/tmp/pyphi-1x-oracle/.venv uv pip install "pyphi==1.2.0"
    VIRTUAL_ENV=/tmp/pyphi-1x-oracle/.venv /tmp/pyphi-1x-oracle/.venv/bin/python \
        scripts/gen_iit3_emd_oracle.py

Control: the script first recomputes the anchored ``basic`` value (2.3125 = 37/16)
and asserts it before trusting the oracle on rule110/grid3 — if the 1.x harness
cannot reproduce a known-good value it is not correctly invoked.

Config matches the 2.0 golden: EMD mechanism + EMD CES-distance measure,
JOINT_BIPARTITION mechanism partitions (``PARTITION_TYPE='BI'``),
DIRECTED_BIPARTITION system cuts, precision 6. ``PICK_SMALLEST_PURVIEW=False``
(PyPhi 1.x's default) is the larger-purview resolution that corresponds to 2.0's
``purview_tie_resolution=['PHI', 'PURVIEW_SIZE']`` applied with ``operation=max``.
"""
# This reproducer targets the PyPhi 1.x API (pyphi.Network / Subsystem / compute),
# which does not exist in the 2.0 package, so pyright cannot resolve it here.
# pyright: reportAttributeAccessIssue=false

import json

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

# Arrays copied verbatim from pyphi/examples.py on the 2.0 branch.
RULE110_TPM = np.array(
    [
        [0, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
        [0, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [0, 0, 0],
    ]
)
GRID3_TPM = np.array(
    [
        [
            [[0.04742587, 0.02931223, 0.04742587], [0.04742587, 0.07585818, 0.88079708]],
            [[0.11920292, 0.81757448, 0.11920292], [0.11920292, 0.92414182, 0.95257413]],
        ],
        [
            [[0.88079708, 0.07585818, 0.04742587], [0.88079708, 0.18242552, 0.88079708]],
            [[0.95257413, 0.92414182, 0.11920292], [0.95257413, 0.97068777, 0.95257413]],
        ],
    ]
)
GRID3_CM = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]])

NETS = {
    "rule110": {"tpm": RULE110_TPM, "cm": np.ones((3, 3), int), "state": (1, 0, 1)},
    "grid3": {"tpm": GRID3_TPM, "cm": GRID3_CM, "state": (1, 0, 0)},
}


def _sia(spec, *, pick_smallest_purview):
    pyphi.config.PICK_SMALLEST_PURVIEW = pick_smallest_purview
    net = pyphi.Network(spec["tpm"], cm=spec["cm"])
    sub = pyphi.Subsystem(net, spec["state"], range(net.size))
    sia = pyphi.compute.sia(sub)
    concepts = [
        {
            "mechanism": [int(i) for i in c.mechanism],
            "phi": float(c.phi),
            "cause_purview": [int(i) for i in c.cause.purview],
            "effect_purview": [int(i) for i in c.effect.purview],
        }
        for c in sia.ces
    ]
    return {
        "sia_phi": float(sia.phi),
        "cut": str(sia.cut),
        "ces_size": len(sia.ces),
        "sum_small_phi": float(sum(c.phi for c in sia.ces)),
        "concepts": concepts,
    }


def _control_basic():
    net = pyphi.examples.basic_network()
    sub = pyphi.Subsystem(net, (1, 0, 0), range(net.size))
    phi = float(pyphi.compute.sia(sub).phi)
    assert abs(phi - 2.3125) < 1e-4, (
        f"control failed: basic sia.phi = {phi}, want 2.3125"
    )
    return phi


def main():
    control = _control_basic()
    out = {
        "oracle": f"pyphi=={pyphi.__version__}",
        "control_basic_sia_phi": control,
        "results": {
            name: {
                "state": list(spec["state"]),
                "pick_largest_purview": _sia(spec, pick_smallest_purview=False),
                "pick_smallest_purview": _sia(spec, pick_smallest_purview=True),
            }
            for name, spec in NETS.items()
        },
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
