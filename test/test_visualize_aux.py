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
    # 2-bit state labels on both axes.
    assert [t.get_text() for t in ax.get_xticklabels()] == ["00", "01", "10", "11"]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["00", "01", "10", "11"]
    plt.close(fig)
