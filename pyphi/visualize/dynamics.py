# visualize/dynamics.py
"""Visualize state trajectories."""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike


def plot_dynamics(
    data: ArrayLike, node_labels=None, title="", fig=None, ax=None, figsize=(25, 5)
):
    """Plot an array of states over time.

    Arguments:
        data (ArrayLike): An array of states with shape (timesteps, units).

    Returns:
        tuple: The matplotlib figure and axes.
    """
    # Plot time horizontally.
    data = np.transpose(data)
    if ax is None:
        figure = plt.figure(figsize=figsize) if fig is None else fig
        axes = figure.gca()
    else:
        axes = ax
        figure = fig if fig is not None else axes.figure
    im = axes.imshow(data, aspect="auto", interpolation="none", vmin=0, vmax=1)
    axes.grid(False)
    axes.set_title(title)
    axes.set_ylabel("Substrate state")
    axes.set_xlabel("Time")
    if node_labels is not None:
        axes.set_yticks(range(len(node_labels)), node_labels)
    plt.colorbar(im, ax=axes)
    return figure, axes
