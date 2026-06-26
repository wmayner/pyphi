"""Golden tests against the author's original bounds code.

The reference values in ``test/data/bounds/reference_goldens.json`` were
computed by the experiment code released with Zaeemzadeh & Tononi (2024)
(github.com/zaeemzadeh/IIT-bounds) running on its pinned pre-2.0 pyphi
(branch feature/iit-4.0); see the fixture metadata for the generation
recipe. phi*_e values went through the original pipeline's find_mip on
the high-selectivity construction; this implementation computes them in
closed form from the S3-appendix binomial formulas, so agreement is a
cross-implementation check of both the formulas and the measure
semantics. The relation sums follow the published code's profile
evaluation (see the module docstring for the paper-text difference).
"""

import json
from pathlib import Path

import pytest

from pyphi.formalism.iit4 import bounds

FIXTURE = Path(__file__).parent.parent / "data" / "bounds" / "reference_goldens.json"

with FIXTURE.open() as f:
    _GOLDENS = json.load(f)

RESULTS = _GOLDENS["results"]
REL = 1e-9


@pytest.mark.parametrize("entry", RESULTS, ids=lambda e: f"n={e['n']}")
def test_phi_e_star_matches_reference(entry):
    n = entry["n"]
    for k_str, reference in entry["phi_e_star"].items():
        actual = bounds._phi_e_star(n, int(k_str))
        assert actual == pytest.approx(reference, rel=REL, abs=1e-12), (
            f"n={n}, k={k_str}"
        )


@pytest.mark.parametrize("entry", RESULTS, ids=lambda e: f"n={e['n']}")
def test_sum_phi_distinctions_iii_matches_reference(entry):
    actual = float(bounds.sum_phi_distinctions_upper_bound(entry["n"], bound="III"))
    assert actual == pytest.approx(entry["sum_phi_distinctions_iii"], rel=REL)


@pytest.mark.parametrize("entry", RESULTS, ids=lambda e: f"n={e['n']}")
@pytest.mark.parametrize(
    "bound_id,key",
    [
        ("I", "sum_phi_relations_i"),
        ("II", "sum_phi_relations_ii"),
        ("III", "sum_phi_relations_iii"),
    ],
)
def test_sum_phi_relations_matches_reference(entry, bound_id, key):
    actual = float(bounds.sum_phi_relations_upper_bound(entry["n"], bound=bound_id))
    assert actual == pytest.approx(entry[key], rel=REL)


def test_fixture_is_nonvacuous():
    assert len(RESULTS) >= 5
    assert any(entry["n"] >= 6 for entry in RESULTS)
