"""pyphi.sweep: cartesian batch driver over states / subsets / formalisms."""

from __future__ import annotations

import math

from pyphi import System
from pyphi import examples
from pyphi import sweep
from pyphi.conf import config
from pyphi.conf import presets
from pyphi.direction import Direction
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
    # Genuine-difference guard: the axis must change the computed value, not
    # just the index labels (the two formalisms disagree on this substrate).
    phi_by_formalism = result.df["phi"]
    assert phi_by_formalism["IIT_4_0_2023"] != phi_by_formalism["IIT_3_0"]


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


def test_sweep_unknown_compute_string_raises_valueerror():
    """A compute-string typo must raise, not crash opaquely in the cell."""
    import pytest

    substrate = examples.basic_substrate()
    with config.override(**presets.iit4_2023):  # noqa: SIM117
        with pytest.raises(ValueError, match="SIA"):
            sweep(substrate, states=[(1, 0, 0)], compute="SIA")


def test_sweep_progress_false_silences_sequential_path():
    """progress=False must reach the cells on the sequential path too.

    The documented force semantics: the inner computations see
    ``progress_bars=False`` regardless of the config default.
    """
    substrate = examples.basic_substrate()
    seen: list[bool] = []

    def probe(system):
        seen.append(config.infrastructure.progress_bars)
        return system.sia()

    with config.override(**presets.iit4_2023, progress_bars=True):
        sweep(
            substrate,
            states=[(1, 0, 0)],
            compute=probe,
            parallel=False,
            progress=False,
        )
    assert seen == [False]


def test_sia_rows_carry_selection_margins():
    substrate = examples.basic_substrate()
    state = (1, 0, 0)
    result = sweep(substrate, states=[state])
    row = result.df.iloc[0]
    sia = System(substrate, state).sia()
    assert math.isclose(row["partition_margin"], float(sia.partition_margin))
    assert math.isclose(
        row["cause_state_margin"], float(sia.state_margins[Direction.CAUSE])
    )
    assert math.isclose(
        row["effect_state_margin"], float(sia.state_margins[Direction.EFFECT])
    )
    assert bool(row["effectively_tied"]) == sia.effectively_tied


def test_iit3_sia_rows_have_no_margins():
    substrate = examples.basic_substrate()
    result = sweep(substrate, states=[(1, 0, 0)], formalisms=["IIT_3_0"])
    row = result.df.iloc[0]
    assert row["partition_margin"] is None or math.isnan(row["partition_margin"])
    assert row["effectively_tied"] is None


class TestSubstratesAxis:
    def test_single_substrate_constant_column(self):
        substrate = examples.basic_substrate()
        result = sweep(
            substrate,
            states=[(1, 0, 0)],
            formalisms=["IIT_4_0_2026"],
            parallel=False,
            progress=False,
        )
        assert list(result.df["substrate"]) == [0]

    def test_dict_labels_become_index_level(self):
        subs = {"basic": examples.basic_substrate(), "xor": examples.xor_substrate()}
        result = sweep(
            subs,
            states=[(1, 0, 1)],
            formalisms=["IIT_4_0_2026"],
            parallel=False,
            progress=False,
        )
        assert result.df.index.name == "substrate"
        assert set(result.df.index) == {"basic", "xor"}

    def test_sequence_labels_are_positions(self):
        subs = [examples.basic_substrate(), examples.xor_substrate()]
        result = sweep(
            subs,
            states=[(1, 0, 1)],
            formalisms=["IIT_4_0_2026"],
            parallel=False,
            progress=False,
        )
        assert set(result.df.index) == {0, 1}

    def test_all_states_enumerated_per_substrate(self):
        # Substrates of different sizes coexist under states="all".
        subs = {
            "small": examples.basic_substrate(),
            "fig4": examples.fig4_substrate(),
        }
        result = sweep(
            subs,
            states="all",
            formalisms=["IIT_4_0_2026"],
            parallel=False,
            progress=False,
        )
        computed_plus_skipped = len(result.df) + len(result.skipped)
        n_small = len(examples.basic_substrate())
        n_fig4 = len(examples.fig4_substrate())
        assert computed_plus_skipped == 2**n_small + 2**n_fig4

    def test_skipped_entries_are_4_tuples(self):
        result = sweep(
            examples.basic_substrate(),
            states="all",
            formalisms=["IIT_4_0_2026"],
            parallel=False,
            progress=False,
        )
        assert len(result.skipped) > 0
        for entry in result.skipped:
            label, formalism, _subset, _state = entry
            assert label == 0
            assert formalism == "IIT_4_0_2026"


def test_sweep_default_formalism_honors_ambient_customizations():
    """sweep(formalisms=None) must compute under the live config, exactly as
    pyphi.analyze does — not silently reset it to the version preset.

    The fixture has power: under IIT_3_0 the customized ces_measure changes
    Phi from 2.3125 (preset) to 1.0833, so a preset substitution is visible.
    """
    from dataclasses import replace

    substrate = examples.basic_substrate()
    state = (1, 0, 0)
    preset = presets.by_name["IIT_3_0"]
    custom_iit = replace(preset["iit"], ces_measure="SUM_SMALL_PHI")
    with config.override(
        **{k: v for k, v in preset.items() if k != "iit"},
        iit=custom_iit,
        parallel=False,
        progress_bars=False,
    ):
        direct = float(System(substrate, state).sia().phi)
        preset_value = None
        with config.override(iit=preset["iit"]):
            preset_value = float(System(substrate, state).sia().phi)
        assert direct != preset_value  # the customization has an effect
        result = sweep(substrate, states=[state], parallel=False)
        assert float(result.df["phi"].iloc[0]) == direct
        # The table still reports the active version name.
        assert result.df["formalism"].iloc[0] == "IIT_3_0"


def test_sweep_result_equality_does_not_raise():
    """The generated dataclass ``__eq__`` compared DataFrames with ``==``,
    whose elementwise result has no truth value; equality must be usable."""
    import pandas as pd

    from pyphi.sweep import SweepResult

    a = SweepResult(df=pd.DataFrame({"x": [1, 2]}), results=[1], skipped=[])
    b = SweepResult(df=pd.DataFrame({"x": [1, 2]}), results=[1], skipped=[])
    c = SweepResult(df=pd.DataFrame({"x": [1, 3]}), results=[1], skipped=[])
    assert a == b
    assert a != c
    assert a != object()
