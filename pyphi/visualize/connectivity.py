# visualize/connectivity.py
"""Visualize system connectivity information."""

import networkx as nx

NODE_COLORS = {
    # (in subsystem, state)
    (False, 0): "lightgrey",
    (False, 1): "darkgrey",
    (True, 0): "lightblue",
    (True, 1): "darkblue",
}


def plot_graph(g, **kwargs):
    kwargs = {
        **dict(
            with_labels=True,
            arrowsize=20,
            node_size=600,
            font_color="white",
        ),
        **kwargs,
    }
    nx.draw_networkx(
        g,
        **kwargs,
    )


def plot_subsystem(subsystem, **kwargs):
    g = nx.from_numpy_matrix(subsystem.cm, create_using=nx.DiGraph)
    nx.relabel_nodes(
        g, dict(zip(range(subsystem.network.size), subsystem.node_labels)), copy=False
    )
    if "node_color" not in kwargs:
        kwargs["node_color"] = [
            NODE_COLORS[(i in subsystem.node_indices, subsystem.state[i])]
            for i in range(subsystem.network.size)
        ]
    plot_graph(g, **kwargs)
    return g
