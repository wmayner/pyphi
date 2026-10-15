"""The analysis entry points share one argument vocabulary, and sweep tables
are long-format."""

import inspect

import numpy as np
import pandas as pd
import pytest

import pyphi
from pyphi import examples
from pyphi import macro
from pyphi.actual import Transition
from pyphi.conf import config
from test.conftest import skip_if_no_emd_backend

ENTRY_POINTS = [
    pyphi.analyze,
    pyphi.sweep,
    pyphi.optimize,
    pyphi.landscape_section,
    pyphi.perturb,
    pyphi.phi_posterior,
    pyphi.estimate_analysis,
    pyphi.Substrate.complexes,
    macro.complexes,
]


def test_shared_arguments_are_keyword_only_and_named_alike():
    for f in ENTRY_POINTS:
        params = inspect.signature(f).parameters
        for name in (
            "subset",
            "subsets",
            "formalism",
            "formalisms",
            "compute",
            "parallel",
            "progress",
            "seed",
            "bounds",
            "prune",
        ):
            if name in params and f is not pyphi.optimize:
                assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, (f, name)
        assert "parallel_kwargs" not in params, f
        assert "direction" not in params, f
    # optimize's search box is positional: it fixes the search dimension.
    assert "goal" in inspect.signature(pyphi.optimize).parameters
    assert "seed" not in inspect.signature(pyphi.sweep).parameters


def test_campaigns_take_no_provenance_only_seed():
    from pyphi import campaign

    for f in (campaign.prepare, campaign.prepare_ces):
        assert "seed" not in inspect.signature(f).parameters, f


@pytest.mark.parametrize(
    "f",
    [
        macro.complexes,
        macro.intrinsic_units,
        macro.valid_systems,
        macro.is_intrinsic_unit,
        macro.competing_systems,
    ],
)
def test_macro_search_options_are_keyword_only(f):
    params = inspect.signature(f).parameters
    assert "parallel_kwargs" not in params
    for name in ("bounds", "parallel", "prune", "progress"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, name


def test_transition_options_are_keyword_only():
    params = inspect.signature(Transition).parameters
    for name in ("partition", "noise_background"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, name


def test_compute_vocabulary_is_shared():
    s = examples.basic_substrate()
    with config.override(**pyphi.iit4_2026, progress_bars=False):
        for compute in (None, "sia", "ces", "distinctions"):
            pyphi.analyze(s, (1, 0, 0), compute=compute)
            pyphi.sweep(s, states=[(1, 0, 0)], compute=compute, progress=False)
            pyphi.estimate_analysis(s, compute=compute)


def test_estimate_analysis_takes_formalism():
    s = examples.basic_substrate()
    with config.override(**pyphi.iit3):
        expected = pyphi.estimate_analysis(s)
    estimate = pyphi.estimate_analysis(s, formalism="IIT_3_0")
    assert estimate == expected
    assert estimate != pyphi.estimate_analysis(s)


def test_estimate_analysis_says_a_callable_cannot_be_priced():
    with pytest.raises(ValueError, match="cannot price a callable"):
        pyphi.estimate_analysis(
            examples.basic_substrate(), compute=lambda system: system.sia()
        )


def test_sweep_unknown_compute_names_the_vocabulary():
    with pytest.raises(ValueError, match="distinctions"):
        pyphi.sweep(examples.basic_substrate(), states=[(1, 0, 0)], compute="Ces")


def test_sweep_df_is_long_format_whatever_the_axis_sizes():
    s = examples.basic_substrate()
    one = pyphi.sweep(s, states=[(1, 0, 0)], progress=False).df
    two = pyphi.sweep(s, states=[(1, 0, 0), (0, 0, 1)], progress=False).df
    for df in (one, two):
        assert {"substrate", "formalism", "subset", "state"} <= set(df.columns)
        assert isinstance(df.index, pd.RangeIndex)
    assert list(two["state"]) == [(1, 0, 0), (0, 0, 1)]


def test_sweep_full_rows_report_big_phi():
    with config.override(**pyphi.iit4_2026):
        df = pyphi.sweep(
            examples.iit4_2023_fig1a_substrate(),
            states=[(0, 1, 1)],
            compute=None,
            progress=False,
        ).df
    assert {"phi", "big_phi", "sum_phi_d", "sum_phi_r"} <= set(df.columns)
    row = df.iloc[0]
    assert row.big_phi == pytest.approx(row.sum_phi_d + row.sum_phi_r)


def test_sweep_ces_rows_match_analysis_to_pandas():
    s = examples.iit4_2023_fig1a_substrate()
    with config.override(**pyphi.iit4_2026, progress_bars=False):
        swept = pyphi.sweep(s, states=[(0, 1, 1)], compute="ces").df
        full = pyphi.sweep(s, states=[(0, 1, 1)], compute=None).df
        expected = pyphi.analyze(s, (0, 1, 1)).to_pandas()
    # The analysis also gives the system's state, which sweep rows leave out.
    expected = expected.drop(columns="current_state")
    for df in (swept, full):
        pd.testing.assert_frame_equal(df[list(expected.columns)], expected)


def test_analysis_rows_line_up_with_sweep_rows_for_a_subset():
    """state is the substrate's state in both tables; current_state, the
    subset's, is only in the analysis."""
    s = examples.iit4_2023_fig1a_substrate()
    with config.override(**pyphi.iit4_2026, progress_bars=False):
        swept = pyphi.sweep(s, states=[(0, 1, 1)], subsets=[(1, 2)], compute="ces").df
        row = pyphi.analyze(s, (0, 1, 1), subset=(1, 2)).to_pandas()
    assert list(row.columns[:4]) == ["formalism", "subset", "state", "current_state"]
    assert row["state"].iloc[0] == (0, 1, 1)
    assert row["current_state"].iloc[0] == (1, 1)
    combined = pd.concat([swept, row], ignore_index=True)
    assert list(combined["state"]) == [(0, 1, 1), (0, 1, 1)]
    shared = [c for c in row.columns if c in swept.columns]
    pd.testing.assert_frame_equal(
        combined.loc[[1], shared].reset_index(drop=True),
        swept[shared].reset_index(drop=True),
    )


def test_callable_structure_sweep_round_trips_unchanged():
    # A callable returning structures leaves no Analysis columns to backfill.
    with config.override(**pyphi.iit4_2026):
        r = pyphi.sweep(
            examples.basic_substrate(),
            states=[(1, 0, 0)],
            compute=lambda system: system.ces(),
            progress=False,
        )
    back = pyphi.serialize.loads(pyphi.serialize.dumps(r))
    pd.testing.assert_frame_equal(back.df, r.df)


def test_sweep_distinctions_rows():
    with config.override(**pyphi.iit4_2026):
        r = pyphi.sweep(
            examples.iit4_2023_fig1a_substrate(),
            states=[(0, 1, 1)],
            compute="distinctions",
            progress=False,
        )
    assert {"n_distinctions", "sum_phi_d"} <= set(r.df.columns)
    assert r.df.n_distinctions[0] == len(r.results[0])


def test_sweep_accepts_numpy_states_and_validates_axes():
    s = examples.basic_substrate()
    two = pyphi.sweep(s, states=np.array([[1, 0, 0], [0, 0, 1]]), progress=False)
    assert list(two.df["state"]) == [(1, 0, 0), (0, 0, 1)]
    one = pyphi.sweep(s, states=tuple(np.array([1, 0, 0])), progress=False)
    assert list(one.df["state"]) == [(1, 0, 0)]
    assert all(type(x) is int for x in one.df["state"][0])
    sub = pyphi.sweep(s, states=[(1, 0, 0)], subsets=(0, 1, 2), progress=False)
    assert list(sub.df["subset"]) == [(0, 1, 2)]
    pyphi.sweep(
        s, states=[(1, 0, 0)], subsets=np.array([[0, 1, 2], [0, 1]], dtype=object)
    )
    with pytest.raises(ValueError, match="states"):
        pyphi.sweep(s, states=[], progress=False)
    with pytest.raises(ValueError, match="subsets"):
        pyphi.sweep(s, states=[(1, 0, 0)], subsets=[], progress=False)
    with pytest.raises(ValueError, match="all"):
        pyphi.sweep(s, states="All", progress=False)
    with pytest.raises(ValueError, match="full"):
        pyphi.sweep(s, states=[(1, 0, 0)], subsets="whole", progress=False)


def test_sweep_repr_is_compact():
    r = pyphi.sweep(examples.iit4_2023_fig1a_substrate(), states="all", progress=False)
    assert repr(r).count("\n") < 40
    assert "SweepResult" in repr(r)
    assert hasattr(r, "_repr_html_")


def test_landscape_reprs_are_compact():
    builder = pyphi.weight_axis(
        "sigmoid",
        np.array([[-0.2, 0.7, 0.2], [0.7, -0.2, 0.0], [0.0, -0.8, 0.2]]),
        (0, 1),
    )
    section = pyphi.landscape_section(
        builder, (1, 0, 0), np.linspace(0.1, 1.0, 6), progress=False
    )
    assert repr(section).count("\n") < 30
    assert "LandscapeSection" in repr(section)
    perturbation = pyphi.perturb(builder, (1, 0, 0), 0.5)
    assert repr(perturbation).count("\n") < 30
    assert "Perturbation" in repr(perturbation)
    record = perturbation.to_pandas()
    assert len(record) == 1
    record = record.iloc[0]
    assert record["derivative"] == perturbation.derivative
    assert record["partition_margin"] == perturbation.margins["partition"]


def test_analyze_grains_false_means_no_search():
    a = pyphi.analyze(examples.basic_substrate(), (1, 0, 0), grains=False)
    assert isinstance(a, pyphi.Analysis)


def test_analyze_parallel_and_progress_do_not_change_the_result():
    s = examples.iit4_2023_fig1a_substrate()
    with config.override(**pyphi.iit4_2026):
        expected = pyphi.analyze(s, (0, 1, 1), compute="sia", progress=False)
        for parallel in (True, False):
            sia = pyphi.analyze(
                s, (0, 1, 1), compute="sia", parallel=parallel, progress=False
            )
            assert sia.phi == expected.phi


@pytest.mark.parametrize("value", [1, "yes", "partitions"])
@pytest.mark.parametrize(
    "run",
    [
        lambda s, p: pyphi.analyze(s, (1, 0, 0), compute="sia", parallel=p),
        lambda s, p: pyphi.analyze(s, (1, 0, 0), grains=True, parallel=p),
        lambda s, p: pyphi.sweep(s, states=[(1, 0, 0)], parallel=p),
        lambda s, p: s.complexes((1, 0, 0), parallel=p),
    ],
    ids=["analyze", "grains", "sweep", "complexes"],
)
def test_parallel_must_be_a_bool(run, value):
    with pytest.raises(TypeError, match="parallel must be True, False or None"):
        run(examples.basic_substrate(), value)


@pytest.mark.parametrize("value", [1, "no"])
@pytest.mark.parametrize(
    "run",
    [
        lambda s, p: pyphi.analyze(s, (1, 0, 0), compute="sia", progress=p),
        lambda s, p: pyphi.analyze(s, (1, 0, 0), grains=True, progress=p),
        lambda s, p: pyphi.sweep(s, states=[(1, 0, 0)], progress=p),
        lambda s, p: s.complexes((1, 0, 0), progress=p),
        lambda s, p: pyphi.System(s, (1, 0, 0)).sia(progress=p),
        lambda s, p: pyphi.optimize(
            lambda _: s, (1, 0, 0), [(0, 1)], seed=0, maxiter=1, popsize=2, progress=p
        ),
    ],
    ids=["analyze", "grains", "sweep", "complexes", "System.sia", "optimize"],
)
def test_progress_must_be_a_bool(run, value):
    with pytest.raises(TypeError, match="progress must be True, False or None"):
        run(examples.basic_substrate(), value)


def test_parallel_accepts_a_numpy_bool():
    s = examples.basic_substrate()
    with config.override(**pyphi.iit4_2026):
        assert (
            pyphi.analyze(s, (1, 0, 0), compute="sia", parallel=np.True_).phi
            == pyphi.analyze(s, (1, 0, 0), compute="sia").phi
        )


def test_analyze_parallel_reaches_the_grain_search(monkeypatch):
    from pyphi.macro import search

    seen = []
    evaluate = search._evaluate_systems

    def recording(systems, memo, parallel_kwargs=None, system_states=None):
        seen.append((parallel_kwargs or {}).get("parallel"))
        return evaluate(systems, memo, parallel_kwargs, system_states)

    monkeypatch.setattr(search, "_evaluate_systems", recording)
    s = examples.basic_substrate()
    results = {}
    with config.override(**pyphi.iit4_2026, progress_bars=False):
        for parallel in (False, True):
            seen.clear()
            results[parallel] = pyphi.analyze(
                s, (1, 0, 0), grains=True, parallel=parallel
            )
            assert seen
            assert set(seen) == {parallel}
    assert [c.phi for c in results[True].complexes] == [
        c.phi for c in results[False].complexes
    ]


def test_optimize_goal_replaces_direction():
    builder = pyphi.weight_axes(
        "sigmoid",
        np.array([[-0.2, 0.7, 0.2], [0.7, -0.2, 0.0], [0.0, -0.8, 0.2]]),
        [(0, 1), (1, 0)],
    )
    with config.override(**pyphi.iit4_2026):
        r = pyphi.optimize(
            builder,
            (1, 0, 0),
            [(0.1, 1.0), (0.1, 1.0)],
            seed=0,
            goal="minimize",
            maxiter=1,
            popsize=2,
            parallel=False,
            progress=False,
        )
    assert r.goal == "minimize"
    with pytest.raises(ValueError, match="goal"):
        pyphi.optimize(builder, (1, 0, 0), [(0.1, 1.0), (0.1, 1.0)], seed=0, goal="max")


@pytest.mark.parametrize(
    ("state", "kwargs", "match"),
    [
        ((1, 0), {}, "state"),
        ((1, 0, 2), {}, "state"),
        ((1, 0, 0), {"subset": (5,)}, "5"),
        ((1, 0, 0), {"bounds": [(0.1, 1.0)]}, "bounds"),
    ],
)
def test_optimize_validates_inputs_before_searching(state, kwargs, match):
    builder = pyphi.weight_axes(
        "sigmoid",
        np.array([[-0.2, 0.7, 0.2], [0.7, -0.2, 0.0], [0.0, -0.8, 0.2]]),
        [(0, 1), (1, 0)],
    )
    bounds = kwargs.pop("bounds", [(0.1, 1.0), (0.1, 1.0)])
    with pytest.raises(ValueError, match=match):
        pyphi.optimize(
            builder,
            state,
            bounds,
            seed=0,
            maxiter=1,
            popsize=2,
            parallel=False,
            progress=False,
            **kwargs,
        )


@pytest.mark.parametrize("state", [None, "AB"])
def test_grain_search_names_a_state_that_is_not_a_sequence(state):
    with pytest.raises(TypeError, match="a state is a sequence"):
        pyphi.analyze(examples.basic_substrate(), state, grains=True)


@pytest.mark.parametrize(
    "formalism",
    ["IIT_4_0_2026", pytest.param("IIT_3_0", marks=skip_if_no_emd_backend)],
)
def test_mechanism_queries_do_not_depend_on_unit_order(formalism):
    """A mechanism or purview given out of order gives the same φ as sorted."""
    from pyphi.direction import Direction

    system = examples.basic_system()
    with config.override(**pyphi.conf.presets.by_name[formalism]):
        for method in ("cause_mip", "effect_mip", "phi_cause_mip", "phi_effect_mip"):
            query = getattr(system, method)
            sorted_value = query((0, 2), (0, 1, 2))
            shuffled_value = query((2, 0), (2, 1, 0))
            assert getattr(shuffled_value, "phi", shuffled_value) == getattr(
                sorted_value, "phi", sorted_value
            )
        assert system.cause_mip((0, 2), (2, 1, 0)).phi > 0
        assert system.phi((0, 2), (2, 1, 0)) == system.phi((0, 2), (0, 1, 2))
        assert system.find_mip(Direction.EFFECT, (0,), (1, 0)) == system.find_mip(
            Direction.EFFECT, (0,), (0, 1)
        )
        with pytest.raises(TypeError, match="got None"):
            system.phi((0,), None)
        with pytest.raises(TypeError, match="got None"):
            system.cause_mip(None, (0,))


def test_transition_queries_do_not_depend_on_unit_order():
    from pyphi.direction import Direction

    transition = Transition(
        examples.actual_causation_substrate(), (0, 1), (1, 0), (0, 1), (0, 1)
    )
    link = transition.find_actual_cause((0,), purviews=[(1, 0)])
    assert link.alpha == transition.find_actual_cause((0,), purviews=[(0, 1)]).alpha
    assert link.alpha > 0
    assert (
        transition.find_mip(Direction.CAUSE, (1, 0), (1, 0)).alpha
        == transition.find_mip(Direction.CAUSE, (0, 1), (0, 1)).alpha
    )
    with pytest.raises(TypeError, match="got None"):
        transition.find_mip(Direction.CAUSE, (0,), None)
    with pytest.raises(TypeError, match="got None"):
        transition.find_actual_cause(None)
