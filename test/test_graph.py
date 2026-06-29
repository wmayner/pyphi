import numpy as np
import pytest

import pyphi
from pyphi import graph
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


def test_to_graphml_round_trip(tmp_path):
    import networkx as nx

    sub = pyphi.examples.grid3_substrate()
    path = tmp_path / "grid3.graphml"
    sub.to_graphml(str(path))
    reread = nx.read_graphml(str(path))
    original = sub.to_networkx()
    assert set(reread.nodes()) == set(original.nodes())
    assert set(reread.edges()) == set(original.edges())


def test_to_adjacency_labeled_dataframe():
    sub = pyphi.examples.grid3_substrate()
    df = sub.to_adjacency()
    labels = list(sub.node_labels)
    assert list(df.index) == labels
    assert list(df.columns) == labels
    assert np.array_equal(df.to_numpy(), sub.factored_tpm.infer_cm())


def test_topology_helpers_on_cyclic_substrate():
    import networkx as nx

    sub = pyphi.examples.grid3_substrate()
    g = sub.to_networkx()
    assert graph.is_strongly_connected(sub) == nx.is_strongly_connected(
        nx.convert_node_labels_to_integers(g)
    )
    sccs = graph.strongly_connected_components(sub)
    assert all(isinstance(c, tuple) for c in sccs)
    assert sum(len(c) for c in sccs) == sub.size
    assert isinstance(graph.is_dag(sub), bool)
    assert set(graph.in_degree(sub)) == set(range(sub.size))
    assert set(graph.out_degree(sub)) == set(range(sub.size))


def test_topology_helpers_dag_detection():
    # node0'=node1 (edge 1->0), node1'=0 (constant): a single edge, acyclic.
    tpm = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    sub = Substrate(tpm, cm=np.array([[0, 0], [1, 0]]), node_labels=["A", "B"])
    assert graph.is_dag(sub) is True
    assert graph.simple_cycles(sub) == []
    assert graph.in_degree(sub) == {0: 1, 1: 0}
    assert graph.out_degree(sub) == {0: 0, 1: 1}


# --- DBN export ---------------------------------------------------------------


def _asymmetric_dbn_substrate():
    """3-node binary substrate with distinct parent sets.

    A' = A          (self-loop; parent {A})
    B' = A and C    (parents {A, C})
    C' = 1           (constant; no parents)

    state-by-node, LOLI order: index = A + 2*B + 4*C.
    """
    tpm = np.array(
        [
            [0.0, 0.0, 1.0],  # A0 B0 C0
            [1.0, 0.0, 1.0],  # A1 B0 C0
            [0.0, 0.0, 1.0],  # A0 B1 C0
            [1.0, 0.0, 1.0],  # A1 B1 C0
            [0.0, 0.0, 1.0],  # A0 B0 C1
            [1.0, 1.0, 1.0],  # A1 B0 C1
            [0.0, 0.0, 1.0],  # A0 B1 C1
            [1.0, 1.0, 1.0],  # A1 B1 C1
        ]
    )
    sub = Substrate(tpm, node_labels=["A", "B", "C"])
    # Precondition: inferred parents are A<-{A}, B<-{A,C}, C<-{}.
    assert np.array_equal(
        sub.factored_tpm.infer_cm(),
        np.array([[1, 1, 0], [0, 0, 0], [0, 1, 0]]),
    )
    return sub


def test_dbn_dict_edges_match_inferred_cm():
    sub = _asymmetric_dbn_substrate()
    dbn = graph.substrate_to_dbn_dict(sub)
    inferred = sub.factored_tpm.infer_cm()
    labels = list(sub.node_labels)
    expected = {
        (labels[a], labels[b])
        for a in range(sub.size)
        for b in range(sub.size)
        if inferred[a, b]
    }
    assert set(dbn["edges"]) == expected
    assert dbn["variables"] == {"A": 2, "B": 2, "C": 2}


def test_dbn_dict_cpd_shapes_and_parents():
    sub = _asymmetric_dbn_substrate()
    cpds = graph.substrate_to_dbn_dict(sub)["cpds"]
    # A' = A: one parent (itself), table P(A'|A) shape (2, 2).
    assert cpds["A"]["parents"] == ["A"]
    assert cpds["A"]["table"].shape == (2, 2)
    # B' = A and C: parents in ascending index order [A, C], shape (2, 2, 2).
    assert cpds["B"]["parents"] == ["A", "C"]
    assert cpds["B"]["table"].shape == (2, 2, 2)
    # C' = 1: parentless marginal, shape (2,).
    assert cpds["C"]["parents"] == []
    assert cpds["C"]["table"].shape == (2,)


def test_dbn_dict_cpd_values_and_normalization():
    sub = _asymmetric_dbn_substrate()
    cpds = graph.substrate_to_dbn_dict(sub)["cpds"]
    # P(A'|A): A=0 -> [1,0], A=1 -> [0,1].
    np.testing.assert_allclose(cpds["A"]["table"], [[1.0, 0.0], [0.0, 1.0]])
    # P(B'|A,C): on iff A=1 and C=1.
    b = cpds["B"]["table"]  # axes (A, C, B')
    np.testing.assert_allclose(b[1, 1], [0.0, 1.0])
    np.testing.assert_allclose(b[0, 1], [1.0, 0.0])
    np.testing.assert_allclose(b[1, 0], [1.0, 0.0])
    # P(C') = always 1.
    np.testing.assert_allclose(cpds["C"]["table"], [0.0, 1.0])
    # Every CPD's last axis (the child distribution) sums to 1.
    for cpd in cpds.values():
        np.testing.assert_allclose(cpd["table"].sum(axis=-1), 1.0)


def test_dbn_dict_cpd_matches_reduced_factor():
    sub = _asymmetric_dbn_substrate()
    ftpm = sub.factored_tpm
    cm = ftpm.infer_cm()
    cpds = graph.substrate_to_dbn_dict(sub)["cpds"]
    labels = list(sub.node_labels)
    for i, label in enumerate(labels):
        parents = [a for a in range(sub.size) if cm[a, i]]
        index = tuple(slice(None) if a in parents else 0 for a in range(sub.size))
        np.testing.assert_allclose(cpds[label]["table"], ftpm.factor(i)[index])


def test_dbn_dict_kary_cpd_shapes():
    sub = pyphi.examples.gomez_p53_mdm2_substrate()  # k-ary (ternary node)
    ftpm = sub.factored_tpm
    cm = ftpm.infer_cm()
    alphabets = ftpm.alphabet_sizes
    cpds = graph.substrate_to_dbn_dict(sub)["cpds"]
    labels = list(sub.node_labels)
    for i, label in enumerate(labels):
        parents = [a for a in range(sub.size) if cm[a, i]]
        expected_shape = (*(alphabets[a] for a in parents), alphabets[i])
        assert cpds[label]["table"].shape == expected_shape
        np.testing.assert_allclose(cpds[label]["table"].sum(axis=-1), 1.0)
