# visualize/connectivity.py
"""Visualize system connectivity information."""

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


def plot_system(system, **kwargs):
    g = nx.from_numpy_array(system.cm, create_using=nx.DiGraph)
    nx.relabel_nodes(
        g,
        dict(zip(range(system.substrate.size), system.node_labels, strict=False)),
        copy=False,
    )
    if "node_color" not in kwargs:
        kwargs["node_color"] = [
            NODE_COLORS[(i in system.node_indices, system.state[i])]
            for i in range(system.substrate.size)
        ]
    plot_graph(g, **kwargs)
    return g


def plot_tpm(
    tpm,
    figsize=(10, 12),
    clim=None,
    cmap="viridis",
    label_fontsize=8,
    show_label_threshold=64,
    xticks_top=True,
):
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
    if tpm.shape[1] <= show_label_threshold:
        ax.set_xticks(
            list(range(tpm.shape[1])),
            labels=all_states_str(int(np.log2(tpm.shape[1]))),
            rotation=90,
            fontsize=label_fontsize,
        )
        ax.xaxis.set_ticks_position("top" if xticks_top else "bottom")
        ax.xaxis.set_label_position("top" if xticks_top else "bottom")
    if tpm.shape[0] <= show_label_threshold:
        ax.set_yticks(
            list(range(tpm.shape[0])),
            labels=all_states_str(int(np.log2(tpm.shape[0]))),
            fontsize=label_fontsize,
        )
    return fig, ax
