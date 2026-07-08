# visualize/dynamics.py
"""Visualize state trajectories."""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike


def plot_dynamics(
    data: ArrayLike, node_labels=None, title="", fig=None, ax=None, figsize=(25, 5)
):
    """Plot an array of states over time.

    States are shown as an image with time on the horizontal axis and substrate
    units on the vertical axis; cell brightness encodes each unit's state.

    Parameters
    ----------
    data : ArrayLike
        An array of states with shape ``(timesteps, units)``.
    node_labels : optional
        Labels for the substrate units, used as y-axis tick labels.
    title : str, optional
        Axes title.
    fig : matplotlib.figure.Figure, optional
        Existing figure to draw into.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw into.
    figsize : tuple of float, optional
        Figure size, used only when a new figure is created.

    Returns
    -------
    tuple
        The matplotlib figure and axes.
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
