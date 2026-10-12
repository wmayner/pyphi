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
