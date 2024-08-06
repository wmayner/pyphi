# visualize/dynamics.py
"""Visualize state trajectories."""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike


def plot_dynamics(data: ArrayLike, node_labels=None, title=""):
    """Plot an array of states over time.

    Arguments:
        data (ArrayLike): An array of states with shape (timesteps, units).
    """
    # Plot time horizontally
    data = np.transpose(data)
    fig = plt.figure(figsize=(25, 5))
    ax = plt.imshow(data, aspect="auto", interpolation="none", vmin=0, vmax=1)
    plt.grid(False)
    plt.title(title)
    plt.ylabel("Substrate state")
    plt.xlabel("Time")
    if node_labels is not None:
        plt.yticks(range(len(node_labels)), node_labels)
    plt.colorbar()
    plt.show()
    return fig, ax
