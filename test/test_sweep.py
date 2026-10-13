"""pyphi.sweep: cartesian batch driver over states / subsets / formalisms."""

from __future__ import annotations

import math

import pytest

from pyphi import System
from pyphi import examples
from pyphi import sweep
from pyphi.conf import config
from pyphi.conf import presets
from pyphi.direction import Direction
from pyphi.sweeps import SweepResult


def test_sweep_over_states_shape_and_parity():
    substrate = examples.basic_substrate()  # 3 binary nodes; 6 of 8 reachable
    with config.override(**presets.iit4_2023):
        result = sweep(substrate, states="all")
    assert isinstance(result, SweepResult)
    assert len(result.df) == 6  # one row per reachable state
    assert len(result.skipped) == 2  # 2 unreachable, recorded
    # Long format: every axis is a column, whether or not it varies.
    assert list(result.df.columns[:4]) == ["substrate", "formalism", "subset", "state"]
    assert len(result.results) == 6
    # parity: each row's phi == a direct recompute
    with config.override(**presets.iit4_2023):
        for _, row in result.df.iterrows():
            direct = System(substrate, row["state"]).sia().phi
            assert math.isclose(row["phi"], float(direct))


def test_sweep_explicit_unreachable_state_fails_loud():
    import pytest

    from pyphi import exceptions

    substrate = examples.basic_substrate()
    with config.override(**presets.iit4_2023):  # noqa: SIM117
        with pytest.raises(exceptions.StateUnreachableForwardsError):
            sweep(substrate, states=[(0, 1, 0)])


def test_sweep_over_formalisms():
    substrate = examples.basic_substrate()
    result = sweep(substrate, states=[(1, 0, 0)], formalisms=["IIT_4_0_2023", "IIT_3_0"])
    assert list(result.df["formalism"]) == ["IIT_4_0_2023", "IIT_3_0"]
    # Genuine-difference guard: the axis must change the computed value, not
    # just the formalism labels (the two formalisms disagree on this substrate).
    phi_by_formalism = result.df.set_index("formalism")["phi"]
    assert phi_by_formalism["IIT_4_0_2023"] != phi_by_formalism["IIT_3_0"]


def test_sweep_compute_ces_columns():
    substrate = examples.basic_substrate()
    with config.override(**presets.iit4_2023):
        result = sweep(substrate, states=[(1, 0, 0)], compute="ces")
        direct = System(substrate, (1, 0, 0)).ces()
    row = result.df.iloc[0]
    assert row["n_distinctions"] == len(direct.distinctions)
    assert math.isclose(row["sum_phi_r"], float(direct.relations.sum_phi()))
    assert math.isclose(row["sum_phi_d"], float(direct.distinctions.sum_phi()))
    assert math.isclose(row["big_phi"], float(direct.big_phi))
    assert math.isclose(row["phi"], float(direct.sia.phi))


def test_sweep_over_subsets_enumerates_powerset():
    substrate = examples.basic_substrate()  # 3 nodes -> 7 non-empty subsets
    with config.override(**presets.iit4_2023):
        result = sweep(substrate, states=[(1, 0, 0)], subsets="all")
    # subsets="all" auto-enumerates, so cells unreachable for a sub-subsystem
    # are skipped and recorded rather than raising.
    assert len(result.df) + len(result.skipped) == 7
    assert "subset" in result.df.columns


def test_sweep_custom_callable():
    substrate = examples.basic_substrate()
    with config.override(**presets.iit4_2023):
        result = sweep(substrate, states=[(1, 0, 0)], compute=lambda s: s.sia())
    assert "phi" in result.df.columns
    assert len(result.results) == 1


def test_parallel_equals_sequential():
    import pandas as pd

    substrate = examples.basic_substrate()
    with config.override(**presets.iit4_2023):
        seq = sweep(substrate, states="all", parallel=False)
        par = sweep(substrate, states="all", parallel=True)
    pd.testing.assert_frame_equal(seq.df, par.df)
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

    def test_dict_labels_become_substrate_values(self):
        subs = {"basic": examples.basic_substrate(), "xor": examples.xor_substrate()}
        result = sweep(
            subs,
            states=[(1, 0, 1)],
            formalisms=["IIT_4_0_2026"],
            parallel=False,
            progress=False,
        )
        assert list(result.df["substrate"]) == ["basic", "xor"]

    def test_sequence_labels_are_positions(self):
        subs = [examples.basic_substrate(), examples.xor_substrate()]
        result = sweep(
            subs,
            states=[(1, 0, 1)],
            formalisms=["IIT_4_0_2026"],
            parallel=False,
            progress=False,
        )
        assert list(result.df["substrate"]) == [0, 1]

    def test_all_states_enumerated_per_substrate(self):
        # Substrates of different sizes coexist under states="all".
        subs = {
            "small": examples.basic_substrate(),
            "fig4": examples.iit3_2014_fig4_substrate(),
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
        n_fig4 = len(examples.iit3_2014_fig4_substrate())
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

    from pyphi.sweeps import SweepResult

    a = SweepResult(df=pd.DataFrame({"x": [1, 2]}), results=[1], skipped=[])
    b = SweepResult(df=pd.DataFrame({"x": [1, 2]}), results=[1], skipped=[])
    c = SweepResult(df=pd.DataFrame({"x": [1, 3]}), results=[1], skipped=[])
    assert a == b
    assert a != c
    assert a != object()


def test_flat_label_subset_is_one_subset_like_indices():
    """A flat sequence of labels is one subset, as analyze(subset=...) reads it."""
    substrate = examples.basic_substrate()
    with config.override(**presets.iit4_2023):
        by_label = sweep(substrate, states=[(1, 0, 0)], subsets=("A", "B"))
        by_index = sweep(substrate, states=[(1, 0, 0)], subsets=(0, 1))
    assert len(by_label.df) + len(by_label.skipped) == 1
    assert list(by_label.df["phi"]) == list(by_index.df["phi"])


def test_explicit_axis_reads_flat_sequences_as_one_entry():
    from pyphi.campaign import _axis_as_json
    from pyphi.sweeps import _explicit_axis

    assert _explicit_axis(("A", "B"), "subsets") == [("A", "B")]
    assert _explicit_axis((0, 1), "subsets") == [(0, 1)]
    assert _explicit_axis([("A", "B"), ("C",)], "subsets") == [("A", "B"), ("C",)]
    assert _axis_as_json(("A", "B"), "subsets") == [["A", "B"]]


# ---- axis normalization ----


def _fig1a():
    return examples.iit4_2023_fig1a_substrate()


def test_iterator_axes_survive_several_formalisms_and_substrates():
    import itertools

    substrate = _fig1a()
    result = sweep(
        substrate,
        states=itertools.product([0, 1], repeat=len(substrate)),
        formalisms=["IIT_4_0_2023", "IIT_4_0_2026"],
        progress=False,
    )
    assert len(result.df) == 2 * 2 ** len(substrate)
    result = sweep(
        [substrate, substrate],
        states=[(1, 0, 0)],
        subsets=itertools.combinations(range(3), 2),
        progress=False,
    )
    assert len(result.df) == 2 * 3


def test_subset_column_holds_resolved_indices(tmp_path):
    result = sweep(
        _fig1a(),
        states=(1, 0, 0),
        subsets=[("A", "B"), ("B", "A"), (1, 0), (0, 1, 2)],
        progress=False,
    )
    assert list(result.df.subset) == [(0, 1), (0, 1), (0, 1), (0, 1, 2)]
    assert result.df.groupby("subset").size().to_dict() == {(0, 1): 3, (0, 1, 2): 1}
    # A table built from mixed spellings saves.
    result.save(tmp_path / "mixed.json")


def test_duplicate_formalisms_run_once():
    with config.override(**presets.iit3):
        result = sweep(
            examples.basic_substrate(),
            states=(1, 0, 0),
            formalisms=["iit3", "IIT_3_0", "IIT_3_0"],
            progress=False,
        )
    assert len(result.df) == 1


def test_subsets_none_is_the_whole_system():
    result = sweep(_fig1a(), states=[(1, 0, 0)], subsets=None, progress=False)
    assert list(result.df.subset) == [(0, 1, 2)]


@pytest.mark.parametrize(
    ("subsets", "match"),
    [
        ([(0, 1), "full"], "valid only as the whole subsets value"),
        (("A", 1), "not a mix"),
        ((True, False), "boolean"),
        ([(0, 5)], "out of range"),
    ],
)
def test_invalid_subsets_raise_before_computing(subsets, match):
    with pytest.raises(ValueError, match=match):
        sweep(_fig1a(), states=(1, 0, 0), subsets=subsets)


def test_boolean_state_array_is_one_state():
    import numpy as np

    result = sweep(_fig1a(), states=np.array([True, False, False]), progress=False)
    assert list(result.df.state) == [(1, 0, 0)]


def test_unknown_states_mode_message_describes_states():
    with pytest.raises(ValueError, match="one state tuple"):
        sweep(_fig1a(), states="bogus")


def test_callable_scalar_fills_a_value_column():
    result = sweep(_fig1a(), states=[(1, 0, 0)], compute=_forty_two, progress=False)
    assert "phi" not in result.df.columns
    assert result.df.value.tolist() == [42]


def test_callable_object_adds_no_invented_columns():
    result = sweep(_fig1a(), states=[(1, 0, 0)], compute=_an_object, progress=False)
    assert list(result.df.columns) == ["substrate", "formalism", "subset", "state"]


def _forty_two(system):
    return 42


def _an_object(system):
    return object()
