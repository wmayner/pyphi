# graph.py
"""Bridge between PyPhi value types and networkx directed graphs.

Topology only. By default the exported graph uses the TPM-inferred causal
connectivity (B19 ``FactoredTPM.infer_cm``); pass ``connectivity="declared"``
to use the substrate's declared connectivity matrix verbatim.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pyphi.deferred.deferred_import import networkx as nx

if TYPE_CHECKING:
    import networkx
    import pandas

    from pyphi.substrate import Substrate
    from pyphi.system import System


def _edge_matrix(substrate: Substrate, connectivity: str = "inferred") -> np.ndarray:
    """Return the chosen connectivity matrix as an integer ndarray.

    ``"inferred"`` (default) returns the true causal edges
    (:meth:`FactoredTPM.infer_cm`); ``"declared"`` returns ``substrate.cm``.
    """
    if connectivity == "inferred":
        return np.asarray(substrate.factored_tpm.infer_cm(), dtype=int)
    if connectivity == "declared":
        return np.asarray(substrate.cm, dtype=int)
    raise ValueError(
        f"connectivity must be 'inferred' or 'declared'; got {connectivity!r}"
    )


def _index_digraph(
    substrate: Substrate, connectivity: str = "inferred"
) -> networkx.DiGraph:
    """Integer-node DiGraph (nodes 0..n-1) over the chosen connectivity."""
    matrix = _edge_matrix(substrate, connectivity)
    g = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
    # cm is binary, so from_numpy_array's weight=1 attributes carry no
    # information; drop them so the graph is genuinely unweighted.
    for _, _, data in g.edges(data=True):
        data.pop("weight", None)
    return g


def substrate_to_networkx(
    substrate: Substrate, connectivity: str = "inferred"
) -> networkx.DiGraph:
    """Return a node-labeled directed graph over the substrate's connectivity."""
    g = _index_digraph(substrate, connectivity)
    nx.relabel_nodes(
        g,
        dict(zip(range(substrate.size), substrate.node_labels, strict=False)),
        copy=False,
    )
    return g


def substrate_from_networkx(
    graph: networkx.DiGraph,
    tpm,
    *,
    node_labels=None,
) -> Substrate:
    """Build a :class:`Substrate` from a DiGraph topology plus a TPM.

    The graph supplies the connectivity (its adjacency, self-loops kept) and the
    node order (``node_labels`` if given, else ``list(graph.nodes())``). ``tpm``
    is required and must accept the forms ``Substrate(tpm=...)`` accepts, with a
    unit order matching the node order. Construction runs the default-on B19
    connectivity validator, so a topology that omits a real TPM edge is rejected.
    """
    from pyphi.substrate import Substrate

    nodes = list(node_labels) if node_labels is not None else list(graph.nodes())
    n = len(nodes)
    # Let the canonical constructor determine the TPM's true unit count (robust
    # across the accepted tpm shapes — state-by-node, multidimensional, and the
    # explicit-alphabet joint form, whose last axis is the alphabet, not nodes).
    n_units = Substrate(tpm).size
    if n != n_units:
        raise ValueError(
            f"graph has {n} node(s) but the TPM implies {n_units} unit(s); "
            "node count and TPM unit count must match"
        )
    cm = nx.to_numpy_array(graph, nodelist=nodes, dtype=int)
    return Substrate(tpm, cm=cm, node_labels=[str(node) for node in nodes])


def system_to_networkx(
    system: System, connectivity: str = "inferred"
) -> networkx.DiGraph:
    """Return the substrate graph with per-node state and membership attributes.

    Each node gains ``state`` (its current value) and ``in_system`` (whether the
    node is in ``system.node_indices`` vs the background).
    """
    g = substrate_to_networkx(system.substrate, connectivity)
    in_system = set(system.node_indices)
    for index, label in enumerate(system.substrate.node_labels):
        g.nodes[label]["state"] = int(system.state[index])
        g.nodes[label]["in_system"] = index in in_system
    return g


def to_graphml(substrate: Substrate, path: str, connectivity: str = "inferred") -> None:
    """Write the substrate graph to a GraphML file."""
    nx.write_graphml(substrate_to_networkx(substrate, connectivity), path)


def to_adjacency(
    substrate: Substrate, connectivity: str = "inferred"
) -> pandas.DataFrame:
    """Return the chosen connectivity matrix as a node-labeled DataFrame."""
    import pandas as pd

    labels = pd.Index(list(substrate.node_labels))
    return pd.DataFrame(
        _edge_matrix(substrate, connectivity), index=labels, columns=labels
    )


def is_strongly_connected(substrate: Substrate, connectivity: str = "inferred") -> bool:
    """Whether the substrate graph is strongly connected (a complex must be)."""
    return nx.is_strongly_connected(_index_digraph(substrate, connectivity))


def strongly_connected_components(
    substrate: Substrate, connectivity: str = "inferred"
) -> list[tuple[int, ...]]:
    """Strongly connected components as tuples of node indices."""
    return [
        tuple(sorted(component))
        for component in nx.strongly_connected_components(
            _index_digraph(substrate, connectivity)
        )
    ]


def is_dag(substrate: Substrate, connectivity: str = "inferred") -> bool:
    """Whether the substrate graph is acyclic."""
    return nx.is_directed_acyclic_graph(_index_digraph(substrate, connectivity))


def simple_cycles(
    substrate: Substrate, connectivity: str = "inferred"
) -> list[list[int]]:
    """All simple cycles as lists of node indices."""
    return [
        list(cycle)
        for cycle in nx.simple_cycles(_index_digraph(substrate, connectivity))
    ]


def in_degree(substrate: Substrate, connectivity: str = "inferred") -> dict[int, int]:
    """In-degree per node index."""
    return dict(_index_digraph(substrate, connectivity).in_degree())


def out_degree(substrate: Substrate, connectivity: str = "inferred") -> dict[int, int]:
    """Out-degree per node index."""
    return dict(_index_digraph(substrate, connectivity).out_degree())
