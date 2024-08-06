# graphs.py
"""Graph-theoretic algorithms."""

from .exceptions import MissingOptionalDependenciesError

try:
    import igraph as ig
except ModuleNotFoundError as exc:
    raise MissingOptionalDependenciesError(
        MissingOptionalDependenciesError.MSG.format(dependencies="graphs")
    ) from exc


def maximal_independent_sets(nx_graph):
    """Yield the maximal independent sets of a NetworkX graph.

    Uses igraph's implementation of `maximal_independent_vertex_sets`.

    Time complexity is exponential in the worst case.
    """
    # Convert to igraph for faster maximal independent set implementation
    G = ig.Graph.from_networkx(nx_graph)
    for vertices in G.maximal_independent_vertex_sets():
        # Convert node IDs to mechanisms
        yield G.vs[vertices]["_nx_name"]


def largest_independent_sets(nx_graph):
    """Yield the largest independent set(s) of a NetworkX graph.

    Uses igraph's implementation of `largest_independent_vertex_sets`.

    Time complexity is exponential in the worst case.
    """
    # Convert to igraph for faster maximal independent set implementation
    G = ig.Graph.from_networkx(nx_graph)
    for vertices in G.largest_independent_vertex_sets():
        # Convert node IDs to mechanisms
        yield G.vs[vertices]["_nx_name"]
