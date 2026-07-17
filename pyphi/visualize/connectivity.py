# visualize/connectivity.py
"""Visualize system connectivity information."""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .distribution import all_states_str

NODE_COLORS = {
    # (in system, state)
    (False, 0): "lightgrey",
    (False, 1): "darkgrey",
    (True, 0): "lightblue",
    (True, 1): "darkblue",
}


def _node_color(in_system, state, num_states):
    """Color for one unit: hue family by membership, intensity by state.

    Binary units keep the exact ``NODE_COLORS`` entries; units with a larger
    alphabet interpolate the same family from its light end (state 0) to its
    dark end (state ``num_states - 1``).
    """
    if num_states <= 2:
        return NODE_COLORS[(in_system, state)]
    light, dark = ("lightblue", "darkblue") if in_system else ("lightgrey", "darkgrey")
    fraction = state / (num_states - 1)
    return tuple(
        (1 - fraction) * np.array(mcolors.to_rgb(light))
        + fraction * np.array(mcolors.to_rgb(dark))
    )


def plot_graph(g, **kwargs):
    kwargs = {
        **{
            "with_labels": True,
            "arrowsize": 20,
            "node_size": 600,
            "font_color": "white",
        },
        **kwargs,
    }
    nx.draw_networkx(
        g,
        **kwargs,
    )


def _system_graph(system):
    """Directed graph of the system's connectivity and per-unit colors."""
    g = nx.from_numpy_array(system.cm, create_using=nx.DiGraph)
    nx.relabel_nodes(
        g,
        dict(zip(range(system.substrate.size), system.node_labels, strict=False)),
        copy=False,
    )
    sizes = system.substrate.tpm.alphabet_sizes
    colors = [
        _node_color(i in system.node_indices, system.state[i], sizes[i])
        for i in range(system.substrate.size)
    ]
    return g, colors


def plot_system(system, **kwargs):
    g, colors = _system_graph(system)
    kwargs.setdefault("node_color", colors)
    plot_graph(g, **kwargs)
    return g


def _tick_labels(n, square, states):
    """Axis labels for a TPM axis of length ``n``.

    Explicit ``states`` win when their count matches; a square matrix with a
    power-of-two side is labeled with little-endian bit strings (a binary
    state-by-state TPM); anything else gets integer state indices.
    """
    if states is not None and len(states) == n:
        return list(states)
    if square and n >= 2 and (n & (n - 1)) == 0:
        return list(all_states_str(int(np.log2(n))))
    return [str(i) for i in range(n)]


def plot_tpm(
    tpm,
    figsize=(10, 12),
    clim=None,
    cmap="viridis",
    label_fontsize=8,
    show_label_threshold=64,
    xticks_top=True,
    states=None,
):
    """Plot a TPM as a heatmap with state tick labels.

    Parameters
    ----------
    tpm : np.ndarray
        A 2-D transition probability matrix, typically state-by-state.
    states : Sequence[str], optional
        Explicit state labels. An axis is labeled with them when its length
        equals ``len(states)``. If None, a square matrix with a power-of-two
        side is labeled with little-endian bit strings, and integer state
        indices are used otherwise.
    """
    fig = plt.figure(figsize=figsize)
    ax = plt.axes()
    im = ax.imshow(tpm, cmap=cmap)
    plt.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cax = fig.add_axes(  # pyright: ignore[reportCallIssue]
        [  # pyright: ignore[reportArgumentType]
            ax.get_position().x1 + 0.05,
            ax.get_position().y0,
            0.05,
            ax.get_position().height,
        ]
    )
    plt.colorbar(im, cax=cax)
    if clim is not None:
        im.set_clim(*clim)
    square = tpm.shape[0] == tpm.shape[1]
    if tpm.shape[1] <= show_label_threshold:
        ax.set_xticks(
            list(range(tpm.shape[1])),
            labels=_tick_labels(tpm.shape[1], square, states),
            rotation=90,
            fontsize=label_fontsize,
        )
        ax.xaxis.set_ticks_position("top" if xticks_top else "bottom")
        ax.xaxis.set_label_position("top" if xticks_top else "bottom")
    if tpm.shape[0] <= show_label_threshold:
        ax.set_yticks(
            list(range(tpm.shape[0])),
            labels=_tick_labels(tpm.shape[0], square, states),
            fontsize=label_fontsize,
        )
    return fig, ax
