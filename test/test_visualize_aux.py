"""Tests for the auxiliary visualize modules (matplotlib-based)."""

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pytest


@pytest.fixture(scope="module")
def xor_system():
    from pyphi import examples

    return examples.xor_system()


def test_plot_dynamics_returns_figure_without_showing():
    from pyphi.visualize.dynamics import plot_dynamics

    data = np.array([[0, 1, 0], [1, 0, 1]])  # (timesteps=2, units=3)
    fig, ax = plot_dynamics(data, node_labels=["A", "B", "C"], title="t")
    assert isinstance(ax, matplotlib.axes.Axes)
    # Time runs horizontally: image is (units, timesteps).
    image = ax.get_images()[0].get_array()
    assert image is not None
    assert image.shape == (3, 2)
    assert np.array_equal(image, data.T)
    assert ax.get_title() == "t"
    assert [t.get_text() for t in ax.get_yticklabels()] == ["A", "B", "C"]
    plt.close(fig)


def test_system_graph_exact(xor_system):
    from pyphi.visualize.connectivity import _system_graph

    g, colors = _system_graph(xor_system)
    assert sorted(g.nodes) == ["A", "B", "C"]
    # xor connectivity: all off-diagonal edges.
    assert len(g.edges) == 6
    assert ("A", "A") not in g.edges
    # All units are in the system, state (0, 0, 0).
    assert colors == ["lightblue"] * 3


def test_plot_system_draws_and_returns_graph(xor_system):
    from pyphi.visualize.connectivity import plot_system

    fig, ax = plt.subplots()
    g = plot_system(xor_system, ax=ax)
    assert sorted(g.nodes) == ["A", "B", "C"]
    plt.close(fig)


def test_plot_tpm_exported_and_labeled():
    from pyphi.visualize import plot_tpm

    tpm = np.eye(4)
    fig, ax = plot_tpm(tpm)
    image = ax.get_images()[0].get_array()
    assert image is not None
    assert np.array_equal(image, tpm)
    # 2-bit state labels on both axes (little-endian: first unit varies
    # fastest, per pyphi.utils.all_states).
    assert [t.get_text() for t in ax.get_xticklabels()] == ["00", "10", "01", "11"]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["00", "10", "01", "11"]
    plt.close(fig)


def test_distribution_frame_exact():
    from pyphi.visualize.distribution import _distribution_frame

    d = np.array([0.5, 0.25, 0.125, 0.125])
    frame, default_label = _distribution_frame([d])
    assert list(frame["state"]) == ["00", "10", "01", "11"]
    assert list(frame["probability"]) == [0.5, 0.25, 0.125, 0.125]
    assert set(frame["hue"]) == {"0"}
    assert default_label == "AB"


def test_distribution_frame_multiple_with_labels():
    from pyphi.visualize.distribution import _distribution_frame

    d = np.array([0.5, 0.5])
    frame, _ = _distribution_frame([d, d], labels=["x", "y"])
    assert len(frame) == 4
    assert list(frame["hue"]) == ["x", "x", "y", "y"]


def test_distribution_frame_validates():
    from pyphi.visualize.distribution import _distribution_frame

    with pytest.raises(ValueError, match="sum to 1"):
        _distribution_frame([np.array([0.5, 0.6])])
    # Disabled validation lets unnormalized data through.
    frame, _ = _distribution_frame([np.array([0.5, 0.6])], validate=False)
    assert len(frame) == 2


def test_distribution_frame_large_uses_integer_states():
    from pyphi.visualize.distribution import _distribution_frame

    d = np.full(128, 1 / 128)
    frame, default_label = _distribution_frame([d])
    assert default_label is None
    assert list(frame["state"])[:3] == [0, 1, 2]


def test_plot_distribution_bars():
    from pyphi.visualize import plot_distribution

    fig, ax = plot_distribution(np.array([0.5, 0.25, 0.125, 0.125]))
    assert sum(len(c) for c in ax.containers) == 4
    plt.close(fig)


@pytest.fixture(scope="module")
def xor_sia(xor_system):
    return xor_system.sia()


def test_repertoire_comparison_values(xor_system, xor_sia):
    from pyphi.direction import Direction
    from pyphi.visualize.distribution import _repertoire_comparison

    reps = _repertoire_comparison(xor_system, xor_sia)
    assert set(reps) == {Direction.CAUSE, Direction.EFFECT}
    for by_label in reps.values():
        assert set(by_label) == {"unpartitioned", "partitioned"}
        for r in by_label.values():
            assert r.shape == (2, 2, 2)
    # Partitioning changes the repertoires.
    cause = reps[Direction.CAUSE]
    assert not np.allclose(cause["unpartitioned"], cause["partitioned"])
    # Forward repertoires are unnormalized.
    assert cause["unpartitioned"].sum() == pytest.approx(2.0)


def test_plot_repertoires_smoke(xor_system, xor_sia):
    from pyphi.visualize.distribution import plot_repertoires

    fig, axes, reps = plot_repertoires(xor_system, xor_sia)
    assert len(axes) == 2
    assert len(reps) == 2
    plt.close(fig)
