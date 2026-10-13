"""The candidate system is named ``subset`` everywhere, and the analysis
methods on ``Substrate`` and ``System`` reject keyword arguments they do not
use instead of ignoring them."""

import pytest

import pyphi
from pyphi import actual
from pyphi import examples
from pyphi.formalism import iit4


def test_substrate_sia_takes_subset_like_analyze():
    with pyphi.config.override(**pyphi.iit4_2026):
        s = examples.iit4_2023_fig1a_substrate()
        expected = pyphi.analyze(s, (0, 1, 1), subset=(0, 1)).phi
        sia = s.sia((0, 1, 1), subset=(0, 1))
    assert sia.node_indices == (0, 1)
    assert sia.phi == pytest.approx(expected)


@pytest.mark.parametrize(
    "call",
    [
        lambda s: s.sia((0, 1, 1), partition_schme="X"),
        lambda s: s.sia((0, 1, 1), indices=(0, 1)),
        lambda s: s.ces((0, 1, 1), bogus=1),
        lambda s: s.complexes((0, 1, 1), bogus=1),
        lambda s: s.maximal_complex((0, 1, 1), bogus=1),
        lambda s: s.irreducible_sias((0, 1, 1), bogus=1),
        lambda s: s.all_sias((0, 1, 1), bogus=1),
        lambda s: pyphi.System(s, (0, 1, 1)).sia(bogus=1),
        lambda s: pyphi.System(s, (0, 1, 1)).ces(bogus=1),
        lambda s: pyphi.System(s, (0, 1, 1)).distinctions(bogus=1),
        lambda s: pyphi.System(s, (0, 1, 1)).mic((0,), bogus=1),
        lambda s: pyphi.System(s, (0, 1, 1)).mie((0,), bogus=1),
        lambda s: pyphi.System(s, (0, 1, 1)).phi((0,), (1,), bogus=1),
        lambda s: pyphi.System(s, (0, 1, 1), node_indices=(0, 1)),
    ],
)
def test_unknown_keywords_raise(call):
    with pytest.raises(TypeError, match=r"bogus|partition_schme|indices"):
        call(examples.iit4_2023_fig1a_substrate())


def test_formalism_sia_rejects_unknown_keywords():
    with pyphi.config.override(**pyphi.iit4_2026):
        system = pyphi.System(examples.iit4_2023_fig1a_substrate(), (0, 1, 1))
        with pytest.raises(TypeError, match="bogus"):
            pyphi.formalism.sia(system, bogus=1)
        with pytest.raises(TypeError, match="bogus"):
            iit4.sia(
                system,
                system_measure=None,
                specification_measure=None,
                bogus=1,
            )


def test_iit3_sia_rejects_options_it_does_not_use():
    s = examples.basic_noisy_selfloop_substrate()
    with pytest.raises(TypeError, match="partition_scheme"):
        pyphi.System(s, (1, 0, 0)).sia(
            formalism="IIT_3_0", partition_scheme="DIRECTED_BIPARTITION"
        )


def test_complexes_honors_formalism():
    s = examples.basic_noisy_selfloop_substrate()
    with pyphi.config.override(**pyphi.iit3):
        expected = [(c.node_indices, round(c.phi, 6)) for c in s.complexes((1, 0, 0))]
    got = [
        (c.node_indices, round(c.phi, 6))
        for c in s.complexes((1, 0, 0), formalism="IIT_3_0")
    ]
    assert got == expected
    assert got == [((1, 2), 0.347632)]
    best = s.maximal_complex((1, 0, 0), formalism="IIT_3_0")
    assert (best.node_indices, round(best.phi, 6)) == expected[0]


def test_system_and_substrate_sia_honor_formalism():
    s = examples.basic_noisy_selfloop_substrate()
    expected = pyphi.analyze(s, (1, 0, 0), formalism="IIT_3_0").phi
    assert expected == pytest.approx(0.240118, abs=1e-6)
    assert pyphi.System(s, (1, 0, 0)).sia(formalism="IIT_3_0").phi == pytest.approx(
        expected
    )
    assert s.sia((1, 0, 0), formalism="IIT_3_0").phi == pytest.approx(expected)


def test_system_constructor_takes_subset_keyword_only():
    s = examples.basic_substrate()
    assert pyphi.System(s, (1, 0, 0), subset=(0, 1)).node_indices == (0, 1)
    with pytest.raises(TypeError):
        pyphi.System(s, (1, 0, 0), (0, 1))
    assert not hasattr(pyphi.System, "from_substrate")


def test_system_keeps_its_subset_through_a_cut():
    system = pyphi.System(examples.basic_substrate(), (1, 0, 0), subset=(0, 1))
    cut = pyphi.models.partitions.DirectedBipartition(
        pyphi.Direction.EFFECT, (0,), (1,), system.node_labels
    )
    assert system.apply_partition(cut).node_indices == (0, 1)


def test_replace_keeps_the_subset():
    import copy
    import dataclasses

    system = pyphi.System(examples.basic_substrate(), (1, 0, 0), subset=(0, 1))
    assert system.subset == system.node_indices == (0, 1)
    for replace in (copy.replace, dataclasses.replace):
        changed = replace(system, state=(1, 1, 0))
        assert changed.state == (1, 1, 0)
        assert changed.node_indices == (0, 1)


def test_actual_events_take_subset():
    swap = pyphi.Substrate([[0, 0], [0, 1], [1, 0], [1, 1]], cm=[[0, 1], [1, 0]])
    assert actual.events(swap, (1, 0), (0, 1), (1, 0), subset=(0, 1)) == (
        actual.events(swap, (1, 0), (0, 1), (1, 0))
    )
    with pytest.raises(TypeError):
        actual.events(swap, (1, 0), (0, 1), (1, 0), (0, 1))


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda s: s.sia((1, 0, 0), subsett=(0, 1)),
            r"Substrate\.sia\(\).*'subsett'.*Did you mean 'subset'",
        ),
        (
            lambda s: s.ces((1, 0, 0), indices=(0, 1)),
            r"Substrate\.ces\(\).*'indices'.*subset=",
        ),
        (
            lambda s: s.sia((1, 0, 0), partition_schme="X"),
            r"Did you mean 'partition_scheme'",
        ),
        (lambda s: pyphi.System(s, (1, 0, 0), (0, 1)), r"subset="),
        (lambda s: pyphi.System(s, (1, 0, 0), nodes=(0, 1)), r"'nodes'.*subset="),
    ],
)
def test_misspelt_and_old_keywords_name_the_called_method(call, message):
    with pytest.raises(TypeError, match=message):
        call(examples.iit4_2023_fig1a_substrate())


@pytest.mark.parametrize(
    "call",
    [
        lambda s: s.find_mice(pyphi.Direction.CAUSE, (0,), bogus=1),
        lambda s: s.find_mip(pyphi.Direction.CAUSE, (0,), (0,), bogus=1),
        lambda s: s.cause_mip((0,), (0,), bogus=1),
        lambda s: s.effect_mip((0,), (0,), bogus=1),
        lambda s: s.phi_cause_mip((0,), (0,), bogus=1),
        lambda s: s.phi_effect_mip((0,), (0,), bogus=1),
        lambda s: s.all_distinctions(bogus=1),
    ],
)
def test_system_mechanism_queries_reject_unknown_keywords(call):
    system = pyphi.System(examples.iit4_2023_fig1a_substrate(), (1, 0, 0))
    with pytest.raises(TypeError, match="bogus"):
        call(system)


def test_system_mechanism_queries_keep_their_options():
    with pyphi.config.override(**pyphi.iit4_2026):
        system = pyphi.System(examples.iit4_2023_fig1a_substrate(), (1, 0, 0))
        mice = system.find_mice(pyphi.Direction.CAUSE, (0,), purviews=[(0,)])
        mip = system.find_mip(pyphi.Direction.CAUSE, (0,), (0,), parallel=False)
    assert mice.phi == pytest.approx(mip.phi)


@pytest.mark.parametrize(
    "call",
    [
        lambda tr: actual.sia(tr, bogus=1),
        lambda tr: actual.account(tr, bogus=1),
        lambda tr: actual.account(tr, allow_negative=True),
        lambda tr: tr.find_actual_cause((0,), bogus=1),
        lambda tr: tr.find_actual_effect((0,), bogus=1),
        lambda tr: tr.find_causal_link(pyphi.Direction.CAUSE, (0,), bogus=1),
    ],
)
def test_actual_causation_entry_points_reject_unknown_keywords(call):
    transition = actual.Transition(
        examples.actual_causation_substrate(), (0, 1), (1, 0), (0, 1), (0, 1)
    )
    with pytest.raises(TypeError, match=r"bogus|allow_negative"):
        call(transition)


def test_system_sia_takes_measure_names():
    from pyphi.measures.distribution import resolve_system_measure

    with pyphi.config.override(**pyphi.iit4_2026):
        system = pyphi.System(examples.iit4_2023_fig1a_substrate(), (1, 0, 0))
        name = pyphi.config.formalism.iit.system_phi_measure
        by_name = system.sia(system_measure=name).phi
        by_object = system.sia(system_measure=resolve_system_measure(name)).phi
        assert by_name == pytest.approx(by_object) == pytest.approx(system.sia().phi)
        with pytest.raises(TypeError, match="system_measure must be"):
            system.sia(system_measure=lambda *_, **__: 0.0)
        with pytest.raises(ValueError, match="NOPE"):
            system.sia(specification_measure="NOPE")


def test_unit_function_keywords_are_checked():
    import numpy as np

    from pyphi.substrate_generator import build_substrate
    from pyphi.substrate_generator import ising

    weights = np.array([[-0.2, 0.7, 0.2], [0.7, -0.2, 0.0], [0.0, -0.8, 0.2]])
    units = [ising.probability] * 3
    build_substrate(units, weights, temperature=0.25)
    with pytest.raises(TypeError, match="Did you mean 'temperature'"):
        build_substrate(units, weights, temprature=0.25)
    with pytest.raises(TypeError, match="temprature"):
        pyphi.weight_axis(units, weights, (0, 1), temprature=0.25)
    with pytest.raises(TypeError, match="temprature"):
        pyphi.weight_axes(units, weights, [(0, 1)], temprature=0.25)
    # A keyword shared across a mix of unit functions is accepted when at
    # least one of them names it.
    build_substrate(["ising", "gaussian", "ising"], weights, temperature=1, mu=0.1)


def test_user_unit_function_with_var_keywords_takes_any_keyword():
    import numpy as np

    from pyphi.substrate_generator import build_substrate

    def sigmoid(element, weights, state, **kwargs):
        x = np.dot(weights[:, element], state)
        return 1 / (1 + np.exp(-kwargs["beta"] * x))

    weights = np.array([[0.0, 1.0], [1.0, 0.0]])
    substrate = build_substrate(sigmoid, weights, beta=2.0)
    assert substrate.tpm.shape[0] == 2
    # A user callable without **kwargs is still checked.

    def fixed(element, weights, state, beta=1.0):
        return 0.5

    with pytest.raises(TypeError, match="Did you mean 'beta'"):
        build_substrate(fixed, weights, bata=2.0)


def test_true_events_take_maximal_complex():
    swap = pyphi.Substrate([[0, 0], [0, 1], [1, 0], [1, 1]], cm=[[0, 1], [1, 0]])
    states = ((1, 0), (0, 1), (1, 0))
    with pytest.raises(ValueError, match="maximal_complex"):
        actual.true_events(
            swap, *states, subset=(0, 1), maximal_complex=swap.maximal_complex((0, 1))
        )
    with pytest.raises(TypeError, match="major_complex"):
        actual.true_events(swap, *states, major_complex=None)


def test_system_signature_shows_its_parameters():
    import inspect

    parameters = inspect.signature(pyphi.System).parameters
    assert list(parameters)[:2] == ["substrate", "state"]
    assert parameters["subset"].kind is inspect.Parameter.KEYWORD_ONLY
    assert "partition" in parameters


def test_actual_sia_rejects_internal_parallel_hooks():
    transition = examples.prevention_transition()
    with pytest.raises(TypeError, match="reduce_func"):
        actual.sia(transition, reduce_func=lambda xs: list(xs)[:1])
    assert actual.sia(transition, parallel=False, progress=False).alpha == (
        actual.sia(transition).alpha
    )


@pytest.mark.parametrize("hook", ["reduce_func", "shortcircuit_func", "map_kwargs"])
def test_system_mip_queries_reject_internal_parallel_hooks(hook):
    system = pyphi.System(examples.iit4_2023_fig1a_substrate(), (1, 0, 0))
    with pytest.raises(TypeError, match=hook):
        system.cause_mip((0,), (0,), **{hook: None})


def test_actual_causation_options_are_keyword_only():
    import inspect

    for function, names in [
        (actual.Transition.find_causal_link, ["allow_neg"]),
        (actual.Transition.find_mip, ["allow_neg"]),
        (actual.directed_account, ["mechanisms", "purviews", "allow_neg"]),
    ]:
        parameters = inspect.signature(function).parameters
        for name in names:
            assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY, name


def test_actual_star_exports_its_event_functions():
    for name in [
        "events",
        "true_events",
        "extrinsic_events",
        "true_ces",
        "nice_true_ces",
    ]:
        assert name in actual.__all__


@pytest.mark.parametrize(
    "name", ["all_sias", "irreducible_sias", "complexes", "maximal_complex"]
)
def test_module_complex_functions_match_the_substrate_methods(name):
    import inspect

    from pyphi import substrate as substrate_module

    function = getattr(substrate_module, name)
    method = getattr(pyphi.Substrate, name)
    function_parameters = inspect.signature(function).parameters
    assert function_parameters["candidates"].kind is inspect.Parameter.KEYWORD_ONLY
    assert (
        list(function_parameters)[2:] == list(inspect.signature(method).parameters)[2:]
    )


def test_module_complexes_takes_formalism():
    from pyphi import substrate as substrate_module

    s = examples.basic_substrate()
    with pyphi.config.override(**pyphi.iit4_2026):
        expected = s.complexes((1, 0, 0), formalism="IIT_3_0")
        found = substrate_module.complexes(s, (1, 0, 0), formalism="IIT_3_0")
    assert [c.phi for c in found] == [c.phi for c in expected]
    assert [c.node_indices for c in found] == [c.node_indices for c in expected]
