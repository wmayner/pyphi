"""Golden tests against the original matching research code.

The reference values in ``test/data/matching/reference_goldens.json`` were
computed by the original implementation of the matching formalism (the
external matching research repo at commit 5fbca77, running on its pinned
pre-2.0 pyphi at 78ff3934); see the fixture's metadata for the generation
recipe. Expected agreement:

- Triggered TPM rows, marginal distributions, and the conditional and
  marginal probabilities p and q are pure probability manipulations and must
  agree exactly.
- Triggering coefficients: this implementation uses the positive part of the
  PMI (Eq 5-7), while the original returns the raw log-ratio, so the exact
  relation is ``new == max(0, ref)``. Reference values are null where the
  original produced non-finite results at probability-zero/one edges.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import pyphi
from pyphi.matching import build_triggered_tpm
from pyphi.matching.triggering import triggering_coefficient

FIXTURE = Path(__file__).parent.parent / "data" / "matching" / "reference_goldens.json"

with FIXTURE.open() as f:
    _GOLDENS = json.load(f)

CASES = {case["name"]: case for case in _GOLDENS["cases"]}
BLOCKS = _GOLDENS["results"]
TOL = 1e-12


def _block_id(block):
    return f"{block['case']}-tau{block['tau']}-clamp{block['tau_clamp']}"


@pytest.fixture(scope="module")
def triggered_tpms():
    tpms = {}
    for block in BLOCKS:
        case = CASES[block["case"]]
        substrate = pyphi.Substrate(np.array(case["tpm"]))
        tpms[_block_id(block)] = build_triggered_tpm(
            substrate,
            sensory_indices=tuple(case["sensory_indices"]),
            system_indices=tuple(case["system_indices"]),
            tau=block["tau"],
            tau_clamp=block["tau_clamp"],
        )
    return tpms


@pytest.mark.parametrize("block", BLOCKS, ids=_block_id)
def test_triggered_tpm_rows_match_reference(block, triggered_tpms):
    t = triggered_tpms[_block_id(block)]
    for i, x in enumerate(block["stimuli"]):
        for j, s in enumerate(block["system_states"]):
            assert float(t.row(tuple(x))[tuple(s)]) == pytest.approx(
                block["rows"][i][j], abs=TOL
            )


@pytest.mark.parametrize("block", BLOCKS, ids=_block_id)
def test_marginal_distribution_matches_reference(block, triggered_tpms):
    t = triggered_tpms[_block_id(block)]
    system = tuple(CASES[block["case"]]["system_indices"])
    for j, s in enumerate(block["system_states"]):
        assert t.marginal_probability(system, tuple(s)) == pytest.approx(
            block["marginal"][j], abs=TOL
        )


@pytest.mark.parametrize("block", BLOCKS, ids=_block_id)
def test_triggering_coefficients_match_reference(block, triggered_tpms):
    t = triggered_tpms[_block_id(block)]
    for rec in block["triggering"]:
        tc = triggering_coefficient(
            t, tuple(rec["mechanism"]), tuple(rec["state"]), tuple(rec["stimulus"])
        )
        assert tc.p == pytest.approx(rec["p"], abs=TOL)
        assert tc.q == pytest.approx(rec["q"], abs=TOL)
        if rec["connectedness"] is not None:
            assert tc.connectedness == pytest.approx(
                max(0.0, rec["connectedness"]), abs=TOL
            )
        if rec["value"] is not None:
            assert tc.value == pytest.approx(max(0.0, rec["value"]), abs=TOL)


def test_reference_coverage_is_not_vacuous():
    records = [rec for block in BLOCKS for rec in block["triggering"]]
    finite = [rec for rec in records if rec["value"] is not None]
    negative = [rec for rec in finite if rec["connectedness"] < 0]
    multi_sensory = [
        block for block in BLOCKS if len(CASES[block["case"]]["sensory_indices"]) > 1
    ]
    assert len(records) >= 150
    assert len(finite) >= 200 or len(finite) >= len(records) // 2
    assert len(negative) >= 50  # the positive-part relation is exercised
    assert multi_sensory  # multi-unit sensory axes are exercised
