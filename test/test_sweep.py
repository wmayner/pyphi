"""pyphi.sweep: cartesian batch driver over states / subsets / formalisms."""

from __future__ import annotations

import math

from pyphi import System
from pyphi import examples
from pyphi import sweep
from pyphi.conf import config
from pyphi.conf import presets
from pyphi.sweep import SweepResult


def test_sweep_over_states_shape_and_parity():
    substrate = examples.basic_substrate()  # 3 binary nodes; 6 of 8 reachable
    with config.override(**presets.iit4_2023):
        result = sweep(substrate, states="all")
    assert isinstance(result, SweepResult)
    assert len(result.df) == 6  # one row per reachable state
    assert len(result.skipped) == 2  # 2 unreachable, recorded
    assert result.df.index.name == "state"  # only states vary -> single index
    assert "formalism" in result.df.columns  # constant axis -> column
    assert len(result.results) == 6
    # parity: each row's phi == a direct recompute
    with config.override(**presets.iit4_2023):
        for state, row in result.df.iterrows():
            direct = System(substrate, state).sia().phi
            assert math.isclose(row["phi"], float(direct))


def test_sweep_explicit_unreachable_state_fails_loud():
    import pytest

    from pyphi import exceptions

    substrate = examples.basic_substrate()
    with config.override(**presets.iit4_2023):  # noqa: SIM117
        with pytest.raises(exceptions.StateUnreachableForwardsError):
            sweep(substrate, states=[(0, 1, 0)])


def test_sweep_over_formalisms_multiindex():
    substrate = examples.basic_substrate()
    result = sweep(substrate, states=[(1, 0, 0)], formalisms=["IIT_4_0_2023", "IIT_3_0"])
    assert list(result.df.index.names) == ["formalism"]  # only formalism varies
    assert set(result.df.index) == {"IIT_4_0_2023", "IIT_3_0"}


def test_sweep_compute_ces_columns():
    substrate = examples.basic_substrate()
    with config.override(**presets.iit4_2023):
        result = sweep(substrate, states=[(1, 0, 0)], compute="ces")
        direct = System(substrate, (1, 0, 0)).ces()
    row = result.df.iloc[0]
    assert row["n_distinctions"] == len(direct.distinctions)
    assert math.isclose(row["sum_phi_r"], float(direct.relations.sum_phi()))
    assert math.isclose(row["phi"], float(direct.sia.phi))


def test_sweep_over_subsets_enumerates_powerset():
    substrate = examples.basic_substrate()  # 3 nodes -> 7 non-empty subsets
    with config.override(**presets.iit4_2023):
        result = sweep(substrate, states=[(1, 0, 0)], subsets="all")
    # subsets="all" auto-enumerates, so cells unreachable for a sub-subsystem
    # are skipped and recorded rather than raising.
    assert len(result.df) + len(result.skipped) == 7
    if len(result.df):
        assert result.df.index.name == "subset"


def test_sweep_custom_callable():
    substrate = examples.basic_substrate()
    with config.override(**presets.iit4_2023):
        result = sweep(substrate, states=[(1, 0, 0)], compute=lambda s: s.sia())
    assert "phi" in result.df.columns
    assert len(result.results) == 1


def test_sweep_seed_stamped_on_results():
    substrate = examples.basic_substrate()
    with config.override(**presets.iit4_2023):
        result = sweep(substrate, states=[(1, 0, 0)], seed=1234)
    assert result.results[0].provenance.seed == 1234


def test_parallel_equals_sequential():
    import pandas as pd

    substrate = examples.basic_substrate()
    with config.override(**presets.iit4_2023):
        seq = sweep(substrate, states="all", parallel=False)
        par = sweep(substrate, states="all", parallel=True)
    pd.testing.assert_frame_equal(
        seq.df.sort_index(), par.df.sort_index(), check_like=True
    )
    assert len(par.results) == len(seq.results) == 6
    assert len(par.skipped) == len(seq.skipped) == 2
