"""Tests for the auxiliary visualize modules (matplotlib-based)."""

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np


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
