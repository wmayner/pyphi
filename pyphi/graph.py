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

    from pyphi.substrate import Substrate


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
