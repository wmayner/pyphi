from collections import Counter

import pytest


@pytest.fixture(scope="module")
def xor_projection():
    from pyphi import examples
    from pyphi.visualize.projection import project_ces

    return project_ces(examples.xor_system().ces())


def test_spectrum_aggregates_count_and_sum_phi(xor_projection):
    import plotly.graph_objects as go

    from pyphi.visualize.render.spectrum import render_relation_spectrum
    from pyphi.visualize.theme import DEFAULT_THEME

    fig = render_relation_spectrum(xor_projection, DEFAULT_THEME)
    assert isinstance(fig, go.Figure)
    (bar,) = fig.data
    assert isinstance(bar, go.Bar)
    # Degrees on the x-axis, Σφ as bar height, count in customdata.
    degrees = list(bar.x)
    expected_count = Counter(e.degree for e in xor_projection.edges)
    assert degrees == sorted(expected_count)
    for degree, height, count in zip(bar.x, bar.y, bar.customdata, strict=True):
        relevant = [e.phi for e in xor_projection.edges if e.degree == degree]
        assert height == pytest.approx(sum(relevant))
        assert count[0] == len(relevant)


def test_plot_ces_spectrum_view_runs():
    import plotly.graph_objects as go

    from pyphi import examples
    from pyphi.visualize import plot_ces

    fig = plot_ces(examples.xor_system().ces(), view="spectrum")
    assert isinstance(fig, go.Figure)
    assert any(isinstance(t, go.Bar) for t in fig.data)


def test_spectrum_is_cap_independent_on_analytical():
    import pyphi
    from pyphi import examples
    from pyphi.visualize import plot_ces

    with pyphi.config.override(relation_computation="ANALYTICAL"):
        ces = examples.xor_system().ces()
        spectrum = ces.relations.degree_spectrum()
        fig = plot_ces(ces, view="spectrum", max_relations=1)
    # Bars reflect the full closed-form spectrum, not the single rendered edge.
    (bar,) = fig.data
    assert list(bar.x) == sorted(spectrum)
    assert list(bar.y) == pytest.approx([spectrum[d][1] for d in sorted(spectrum)])
