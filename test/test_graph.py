import numpy as np
import pytest

import pyphi
from pyphi.substrate import Substrate


def _phantom_edge_substrate():
    """2-node binary substrate whose declared cm over-specifies one edge.

    node0' = node1 (copy)  -> real edge 1->0
    node1' = 0 (constant)  -> no real edge into node1
    Declared cm adds a phantom edge 0->1 (over-specification, legal under B19).
    """
    # state-by-node, LOLI order: (0,0),(1,0),(0,1),(1,1)
    tpm = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    declared_cm = np.array([[0, 1], [1, 0]])
    sub = Substrate(tpm, cm=declared_cm, node_labels=["A", "B"])
    # Precondition: the inferred (real) connectivity is edge 1->0 only.
    assert np.array_equal(sub.factored_tpm.infer_cm(), np.array([[0, 0], [1, 0]]))
    return sub


def test_to_networkx_inferred_edges_match_infer_cm():
    sub = pyphi.examples.grid3_substrate()
    g = sub.to_networkx()  # inferred default
    inferred = sub.factored_tpm.infer_cm()
    labels = list(sub.node_labels)
    expected = {
        (labels[a], labels[b])
        for a in range(sub.size)
        for b in range(sub.size)
        if inferred[a, b]
    }
    assert set(g.edges()) == expected
    assert set(g.nodes()) == set(labels)


def test_to_networkx_inferred_vs_declared_drops_phantom_edge():
    sub = _phantom_edge_substrate()
    inferred = sub.to_networkx()
    declared = sub.to_networkx(connectivity="declared")
    assert set(inferred.edges()) == {("B", "A")}
    assert set(declared.edges()) == {("A", "B"), ("B", "A")}


def test_to_networkx_preserves_self_loops_and_kary_labels():
    sub = pyphi.examples.gomez_p53_mdm2_substrate()  # k-ary (ternary P)
    g = sub.to_networkx()
    inferred = sub.factored_tpm.infer_cm()
    labels = list(sub.node_labels)
    assert g.number_of_nodes() == sub.size
    assert set(g.nodes()) == set(labels)
    # Self-loops preserved where the inferred matrix has a nonzero diagonal.
    for i in range(sub.size):
        assert g.has_edge(labels[i], labels[i]) == bool(inferred[i, i])


def test_to_networkx_rejects_unknown_connectivity():
    sub = pyphi.examples.basic_substrate()
    with pytest.raises(ValueError, match="connectivity"):
        sub.to_networkx(connectivity="bogus")


def test_from_networkx_round_trip():
    sub = pyphi.examples.grid3_substrate()
    g = sub.to_networkx()  # inferred; grid3's declared cm equals its inferred cm
    rebuilt = Substrate.from_networkx(
        g, sub.joint_tpm(), node_labels=list(sub.node_labels)
    )
    assert np.array_equal(np.asarray(rebuilt.cm), sub.factored_tpm.infer_cm())
    assert list(rebuilt.node_labels) == list(sub.node_labels)


def test_from_networkx_size_mismatch_raises():
    sub = pyphi.examples.grid3_substrate()  # 3 nodes
    g = sub.to_networkx()
    two_node_tpm = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="node"):
        Substrate.from_networkx(g, two_node_tpm)


def test_from_networkx_under_specified_topology_rejected():
    # A graph missing a real TPM edge must be rejected by B19's validator.
    import networkx as nx

    # node0'=node1 (real edge 1->0); supply a graph with NO edges.
    tpm = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    empty = nx.DiGraph()
    empty.add_nodes_from([0, 1])
    with pytest.raises(Exception, match="under-specified"):
        Substrate.from_networkx(empty, tpm)


def test_system_to_networkx_node_attributes():
    system = pyphi.examples.basic_system()
    g = system.to_networkx()
    labels = list(system.substrate.node_labels)
    for i, label in enumerate(labels):
        assert g.nodes[label]["state"] == system.state[i]
        assert g.nodes[label]["in_system"] == (i in system.node_indices)
    # Every node carries both attributes.
    assert all("state" in d and "in_system" in d for _, d in g.nodes(data=True))
