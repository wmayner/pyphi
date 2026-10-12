"""The grain search reports only irreducible complexes, and its result prints
as a compact card."""

import pyphi
from pyphi import examples


def test_grain_search_returns_no_complex_when_every_candidate_is_reducible():
    with pyphi.config.override(**pyphi.iit4_2026):
        for substrate, state in [
            (examples.basic_substrate(), (1, 0, 0)),
            (examples.xor_substrate(), (0, 0, 0)),
        ]:
            assert substrate.complexes(state) == ()
            result = pyphi.analyze(substrate, state, grains=True)
            assert tuple(result.complexes) == ()
            assert result.ties == ()
            assert not result.maximal_complex


def test_grain_search_complexes_all_have_positive_phi():
    with pyphi.config.override(**pyphi.iit4_2026):
        result = pyphi.analyze(
            examples.iit4_2023_fig1a_substrate(), (0, 1, 1), grains=True
        )
    assert result.complexes
    assert all(c.phi > 0 for c in result.complexes)


def test_complexes_result_prints_compactly():
    with pyphi.config.override(**pyphi.iit4_2026):
        result = pyphi.analyze(
            examples.iit4_2023_fig1a_substrate(), (0, 1, 1), grains=True
        )
    assert len(repr(result)) < 3000
    assert hasattr(result, "_repr_html_")


def test_complexes_result_exports_one_row_per_complex():
    with pyphi.config.override(**pyphi.iit4_2026):
        result = pyphi.analyze(
            examples.iit4_2023_fig1a_substrate(), (0, 1, 1), grains=True
        )
    df = result.to_pandas()
    assert list(df["node_indices"]) == [c.node_indices for c in result.complexes]
    assert list(df["phi"]) == [c.phi for c in result.complexes]
    assert list(df["grains"]) == [(1,), (1,)]
