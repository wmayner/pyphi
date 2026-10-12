"""The competitor set of Eq. 17 holds only systems whose units have an update
grain, in micro updates, of at most the candidate's own (Marshall et al. 2026,
f(U^J, W^J, τ_J))."""

import numpy as np
import pytest

import pyphi
from pyphi import examples
from pyphi.macro import MacroUnit
from pyphi.macro import SearchBounds
from pyphi.macro import coarse_grain
from pyphi.macro import complexes
from pyphi.macro import intrinsic_units
from pyphi.macro.search import competing_systems
from pyphi.substrate import Substrate

BOUNDS = SearchBounds(max_update_grain=2, max_constituents=3)


def test_competitors_have_update_grain_at_most_the_units():
    gamma = MacroUnit((0, 1, 2), 1, coarse_grain(3, {2, 3}))
    with pyphi.config.override(**pyphi.iit4_2026):
        competitors = competing_systems(
            examples.grid3_substrate(),
            gamma,
            ((0, 0, 0), (0, 0, 0)),
            bounds=BOUNDS,
        )
    assert competitors
    assert all(unit.micro_grain <= 1 for system in competitors for unit in system.units)


def test_each_decomposition_is_judged_once_per_update_grain():
    with pyphi.config.override(**pyphi.iit4_2026):
        result = intrinsic_units(
            examples.grid3_substrate(),
            ((0, 0, 0), (0, 0, 0)),
            bounds=SearchBounds(max_update_grain=2, max_constituents=2),
        )
    pair = {
        v.micro_grain: v.verdict for v in result.verdicts if v.constituents == (0, 1)
    }
    assert set(pair) == {1, 2}
    # At τ_J = 1 only the micro singletons {0} and {1} compete; at τ_J = 2 the
    # grain-2 singletons and their pairings join them.
    assert pair[1].num_competitors == 2
    assert pair[2].num_competitors == 7


def test_unit_valid_at_its_own_grain_is_not_rejected_by_slower_competitors():
    """The pair {0, 1} at update grain 1 has φₛ 0.058293 and beats both micro
    units (0.029146 each), but loses at grain 2 to the grain-2 unit over {0}
    (0.118426). When every grain shared the grain-2 verdict, no unit over {0, 1}
    existed and the complexes were {0} and {1} at grain 2 and {2} at grain 1
    (φₛ 0.118426, 0.029311, 0.011319). Under Eq. 17 the grain-1 unit over
    {0, 1} is valid, and as a one-unit system it is the maximal complex."""
    tpm = (
        np.array(
            [
                [98, 2, 18],
                [64, 82, 23],
                [64, 95, 16],
                [20, 25, 32],
                [59, 98, 82],
                [52, 62, 92],
                [95, 78, 79],
                [12, 31, 22],
            ]
        )
        / 100
    )
    with pyphi.config.override(**pyphi.iit4_2026):
        result = complexes(
            Substrate(tpm),
            ((0, 0, 0), (0, 0, 0)),
            bounds=SearchBounds(max_update_grain=2),
        )
    assert [
        (c.node_indices, tuple(u.micro_grain for u in c.units)) for c in result.complexes
    ] == [((0, 1), (1,)), ((2,), (1,))]
    assert [float(c.phi) for c in result.complexes] == pytest.approx(
        [0.30400618689010006, 0.011318656964009828], abs=1e-12
    )
