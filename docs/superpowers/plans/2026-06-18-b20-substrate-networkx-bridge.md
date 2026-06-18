# B20 — Substrate ↔ networkx graph bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a thin, labeled bridge between PyPhi's `Substrate`/`System` and `networkx`, defaulting to the TPM-inferred causal connectivity.

**Architecture:** A new focused module `pyphi/graph.py` holds all graph logic (DiGraph construction, `from_networkx`, export, topology helpers). `Substrate`/`System` get thin delegating methods. `networkx` is loaded lazily via a `DeferredNetworkX` mirroring the existing `DeferredPlotly`, so `import pyphi` never loads it. A `connectivity="inferred"|"declared"` keyword threads through every graph-producing entry point; `"inferred"` uses B19's `factored_tpm.infer_cm()`.

**Tech Stack:** Python 3.12+, networkx (optional, `visualize` extra), pandas (core), numpy, pytest.

**Spec:** `docs/superpowers/specs/2026-06-18-b20-substrate-networkx-bridge-design.md`

## Global Constraints

- Python 3.12+ only; no backward-compatibility shims.
- `networkx` stays optional (declared in the `visualize` extra, `networkx>=2.6.2`); never eagerly imported at `import pyphi`.
- Topology only — no edge weights, no Bayesian-network/CPD/DBN semantics, no igraph.
- `connectivity` defaults to `"inferred"` everywhere (B19 `factored_tpm.infer_cm()`); `"declared"` uses `substrate.cm`.
- Edge convention: matrix entry `[a, b] == 1` means a directed edge `a → b`; `nx.from_numpy_array(M, create_using=nx.DiGraph)` realizes this. Self-loops (diagonal) are preserved.
- Use `uv run` for all Python commands. Final verification runs `uv run pytest` **with no path argument** (B20 is public surface; the `pyphi/` doctest sweep must run).
- Do not bypass pre-commit hooks. Stage only the files each task names (the tree has unrelated untracked work; never `git add -A`).

---

### Task 1: Deferred networkx import + `Substrate.to_networkx`

**Files:**
- Modify: `pyphi/deferred/deferred_import.py` (add `DeferredNetworkX` + `networkx` instance)
- Create: `pyphi/graph.py`
- Modify: `pyphi/substrate.py` (add `to_networkx` method)
- Test: `test/test_graph.py` (new), `test/test_lazy_imports.py`

**Interfaces:**
- Produces: `pyphi.graph.substrate_to_networkx(substrate, connectivity="inferred") -> networkx.DiGraph`
- Produces: `pyphi.graph._edge_matrix(substrate, connectivity="inferred") -> numpy.ndarray` and `pyphi.graph._index_digraph(substrate, connectivity="inferred") -> networkx.DiGraph` (integer-node graph; used by later tasks' helpers)
- Produces: `Substrate.to_networkx(connectivity="inferred") -> networkx.DiGraph`

- [ ] **Step 1: Write the failing tests**

Create `test/test_graph.py`:

```python
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
```

Add to `test/test_lazy_imports.py` (a new test mirroring the xarray one):

```python
def test_networkx_not_loaded_at_pyphi_import():
    """``import pyphi`` must not eagerly load networkx (optional, visualize extra)."""
    assert not _check_module_after_import("networkx")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest test/test_graph.py test/test_lazy_imports.py::test_networkx_not_loaded_at_pyphi_import -q`
Expected: FAIL (`pyphi.graph` does not exist / `Substrate.to_networkx` undefined).

- [ ] **Step 3: Add `DeferredNetworkX`**

In `pyphi/deferred/deferred_import.py`, after the `DeferredPlotly` block, add:

```python
class DeferredNetworkX:
    _networkx = None

    @classmethod
    def networkx(cls):
        if cls._networkx is None:
            try:
                import networkx

                cls._networkx = networkx
            except ModuleNotFoundError as exc:
                raise MissingOptionalDependenciesError(
                    MissingOptionalDependenciesError.MSG.format(dependencies="visualize")
                ) from exc
        return cls._networkx

    def __getattr__(self, attr):
        return getattr(self.networkx(), attr)


networkx = DeferredNetworkX()
```

- [ ] **Step 4: Create `pyphi/graph.py` with the core construction**

```python
# graph.py
"""Bridge between PyPhi value types and networkx directed graphs.

Topology only. By default the exported graph uses the TPM-inferred causal
connectivity (B19 ``FactoredTPM.infer_cm``); pass ``connectivity="declared"``
to use the substrate's declared connectivity matrix verbatim.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Literal

import numpy as np

from pyphi.deferred.deferred_import import networkx as nx

if TYPE_CHECKING:
    import networkx
    import pandas

    from pyphi.substrate import Substrate
    from pyphi.system import System

Connectivity = Literal["inferred", "declared"]


def _edge_matrix(substrate: Substrate, connectivity: Connectivity = "inferred") -> np.ndarray:
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


def _index_digraph(substrate: Substrate, connectivity: Connectivity = "inferred") -> networkx.DiGraph:
    """Integer-node DiGraph (nodes 0..n-1) over the chosen connectivity."""
    matrix = _edge_matrix(substrate, connectivity)
    g = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
    # cm is binary, so from_numpy_array's weight=1 attributes carry no
    # information; drop them so the graph is genuinely unweighted.
    for _, _, data in g.edges(data=True):
        data.pop("weight", None)
    return g


def substrate_to_networkx(
    substrate: Substrate, connectivity: Connectivity = "inferred"
) -> networkx.DiGraph:
    """Return a node-labeled directed graph over the substrate's connectivity."""
    g = _index_digraph(substrate, connectivity)
    nx.relabel_nodes(
        g,
        dict(zip(range(substrate.size), substrate.node_labels, strict=False)),
        copy=False,
    )
    return g
```

- [ ] **Step 5: Add `Substrate.to_networkx`**

In `pyphi/substrate.py`, add a method on `Substrate` (place it near `to_json`):

```python
def to_networkx(self, connectivity: str = "inferred") -> Any:
    """Return a node-labeled :class:`networkx.DiGraph` of the substrate.

    By default edges are the TPM-inferred causal connectivity; pass
    ``connectivity="declared"`` to use the declared ``cm`` verbatim. Requires
    the ``visualize`` extra (networkx).
    """
    from pyphi import graph

    return graph.substrate_to_networkx(self, connectivity)
```

(`Any` is already imported in `substrate.py`.)

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest test/test_graph.py test/test_lazy_imports.py -q`
Expected: PASS (all Task 1 tests + the existing lazy-import tests).

- [ ] **Step 7: Commit**

```bash
git add pyphi/deferred/deferred_import.py pyphi/graph.py pyphi/substrate.py test/test_graph.py test/test_lazy_imports.py
git commit -m "Add Substrate.to_networkx with deferred networkx import

New pyphi/graph.py bridges Substrate to a labeled networkx DiGraph, defaulting
to the TPM-inferred causal connectivity (B19 infer_cm) so over-specified cm
phantom edges are dropped; connectivity='declared' uses cm verbatim. networkx
loads lazily via DeferredNetworkX (never at import pyphi)."
```

---

### Task 2: `Substrate.from_networkx`

**Files:**
- Modify: `pyphi/graph.py` (add `substrate_from_networkx`)
- Modify: `pyphi/substrate.py` (add `from_networkx` classmethod)
- Test: `test/test_graph.py`

**Interfaces:**
- Consumes: `Substrate.to_networkx` (Task 1).
- Produces: `pyphi.graph.substrate_from_networkx(graph, tpm, *, node_labels=None) -> Substrate`
- Produces: `Substrate.from_networkx(graph, tpm, *, node_labels=None) -> Substrate` (classmethod)

- [ ] **Step 1: Write the failing tests**

Add to `test/test_graph.py`:

```python
def test_from_networkx_round_trip():
    sub = pyphi.examples.grid3_substrate()
    g = sub.to_networkx()  # inferred; grid3's declared cm equals its inferred cm
    rebuilt = Substrate.from_networkx(g, sub.joint_tpm(), node_labels=list(sub.node_labels))
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest test/test_graph.py -q -k from_networkx`
Expected: FAIL (`from_networkx` undefined).

- [ ] **Step 3: Implement `substrate_from_networkx`**

Add to `pyphi/graph.py`:

```python
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
    tpm_arr = np.asarray(tpm, dtype=float)
    # Unit count of a state-by-node tpm is its last axis; for a 2-D (states x
    # nodes) array that is shape[-1].
    n_units = tpm_arr.shape[-1]
    if n != n_units:
        raise ValueError(
            f"graph has {n} node(s) but the TPM implies {n_units} unit(s); "
            "node count and TPM unit count must match"
        )
    cm = nx.to_numpy_array(graph, nodelist=nodes, dtype=int)
    return Substrate(tpm, cm=cm, node_labels=[str(node) for node in nodes])
```

- [ ] **Step 4: Add `Substrate.from_networkx`**

In `pyphi/substrate.py`:

```python
@classmethod
def from_networkx(cls, graph: Any, tpm: Any, *, node_labels: Any = None) -> "Substrate":
    """Build a :class:`Substrate` from a networkx DiGraph topology and a TPM.

    The graph supplies connectivity and node order; ``tpm`` supplies the
    dynamics (required). Runs the B19 connectivity validator, so a graph that
    omits a real TPM-implied edge is rejected.
    """
    from pyphi import graph as graph_module

    return graph_module.substrate_from_networkx(graph, tpm, node_labels=node_labels)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest test/test_graph.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyphi/graph.py pyphi/substrate.py test/test_graph.py
git commit -m "Add Substrate.from_networkx (topology + required TPM)

The graph supplies connectivity and node order; the TPM supplies dynamics.
Size mismatch raises ValueError; an under-specified topology is rejected by
B19's connectivity validator at construction."
```

---

### Task 3: `System.to_networkx` with state + membership attributes

**Files:**
- Modify: `pyphi/graph.py` (add `system_to_networkx`)
- Modify: `pyphi/system.py` (add `to_networkx` method)
- Test: `test/test_graph.py`

**Interfaces:**
- Consumes: `substrate_to_networkx` (Task 1).
- Produces: `pyphi.graph.system_to_networkx(system, connectivity="inferred") -> networkx.DiGraph` with node attrs `state` (int) and `in_system` (bool).
- Produces: `System.to_networkx(connectivity="inferred") -> networkx.DiGraph`.

- [ ] **Step 1: Write the failing test**

Add to `test/test_graph.py`:

```python
def test_system_to_networkx_node_attributes():
    system = pyphi.examples.basic_system()
    g = system.to_networkx()
    labels = list(system.substrate.node_labels)
    for i, label in enumerate(labels):
        assert g.nodes[label]["state"] == system.state[i]
        assert g.nodes[label]["in_system"] == (i in system.node_indices)
    # Every node carries both attributes.
    assert all("state" in d and "in_system" in d for _, d in g.nodes(data=True))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest test/test_graph.py::test_system_to_networkx_node_attributes -q`
Expected: FAIL (`System.to_networkx` undefined).

- [ ] **Step 3: Implement `system_to_networkx`**

Add to `pyphi/graph.py`:

```python
def system_to_networkx(system: System, connectivity: Connectivity = "inferred") -> networkx.DiGraph:
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
```

- [ ] **Step 4: Add `System.to_networkx`**

In `pyphi/system.py`:

```python
def to_networkx(self, connectivity: str = "inferred") -> Any:
    """Return the substrate's :class:`networkx.DiGraph` with node attributes.

    Each node carries ``state`` and ``in_system``. Edges default to the
    TPM-inferred causal connectivity (``connectivity="declared"`` for the
    declared ``cm``).
    """
    from pyphi import graph

    return graph.system_to_networkx(self, connectivity)
```

(Confirm `Any` is imported in `system.py`; it is used in existing signatures there.)

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest test/test_graph.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyphi/graph.py pyphi/system.py test/test_graph.py
git commit -m "Add System.to_networkx with state and membership node attributes

Delegates to the substrate graph and annotates each node with its current
state and whether it is in the system vs the background."
```

---

### Task 4: GraphML + adjacency export

**Files:**
- Modify: `pyphi/graph.py` (add `to_graphml`, `to_adjacency`)
- Modify: `pyphi/substrate.py` (add `to_graphml`, `to_adjacency` methods)
- Test: `test/test_graph.py`

**Interfaces:**
- Consumes: `substrate_to_networkx`, `_edge_matrix` (Task 1).
- Produces: `pyphi.graph.to_graphml(substrate, path, connectivity="inferred") -> None`
- Produces: `pyphi.graph.to_adjacency(substrate, connectivity="inferred") -> pandas.DataFrame`
- Produces: `Substrate.to_graphml(path, connectivity="inferred")`, `Substrate.to_adjacency(connectivity="inferred")`.

- [ ] **Step 1: Write the failing tests**

Add to `test/test_graph.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest test/test_graph.py -q -k "graphml or adjacency"`
Expected: FAIL (`to_graphml` / `to_adjacency` undefined).

- [ ] **Step 3: Implement the export functions**

Add to `pyphi/graph.py`:

```python
def to_graphml(substrate: Substrate, path: str, connectivity: Connectivity = "inferred") -> None:
    """Write the substrate graph to a GraphML file."""
    nx.write_graphml(substrate_to_networkx(substrate, connectivity), path)


def to_adjacency(substrate: Substrate, connectivity: Connectivity = "inferred") -> pandas.DataFrame:
    """Return the chosen connectivity matrix as a node-labeled DataFrame."""
    import pandas as pd

    labels = list(substrate.node_labels)
    return pd.DataFrame(_edge_matrix(substrate, connectivity), index=labels, columns=labels)
```

- [ ] **Step 4: Add the `Substrate` methods**

In `pyphi/substrate.py`:

```python
def to_graphml(self, path: str, connectivity: str = "inferred") -> None:
    """Write the substrate graph to a GraphML file (see :meth:`to_networkx`)."""
    from pyphi import graph

    graph.to_graphml(self, path, connectivity)

def to_adjacency(self, connectivity: str = "inferred") -> Any:
    """Return the connectivity matrix as a node-labeled ``pandas.DataFrame``."""
    from pyphi import graph

    return graph.to_adjacency(self, connectivity)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest test/test_graph.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyphi/graph.py pyphi/substrate.py test/test_graph.py
git commit -m "Add GraphML and labeled-adjacency export for substrates

to_graphml writes the substrate graph; to_adjacency returns the chosen
connectivity matrix as a node-labeled DataFrame (P14d labeled-export convention)."
```

---

### Task 5: Topology helpers

**Files:**
- Modify: `pyphi/graph.py` (add the six helper functions)
- Test: `test/test_graph.py`

**Interfaces:**
- Consumes: `_index_digraph` (Task 1).
- Produces (all `(substrate, connectivity="inferred")`):
  - `is_strongly_connected(...) -> bool`
  - `strongly_connected_components(...) -> list[tuple[int, ...]]`
  - `is_dag(...) -> bool`
  - `simple_cycles(...) -> list[list[int]]`
  - `in_degree(...) -> dict[int, int]`
  - `out_degree(...) -> dict[int, int]`

- [ ] **Step 1: Write the failing tests**

Add to `test/test_graph.py`:

```python
def test_topology_helpers_on_cyclic_substrate():
    sub = pyphi.examples.grid3_substrate()
    g = sub.to_networkx()
    import networkx as nx

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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest test/test_graph.py -q -k "topology"`
Expected: FAIL (helpers undefined).

- [ ] **Step 3: Implement the helpers**

Add to `pyphi/graph.py`:

```python
def is_strongly_connected(substrate: Substrate, connectivity: Connectivity = "inferred") -> bool:
    """Whether the substrate graph is strongly connected (a complex must be)."""
    return nx.is_strongly_connected(_index_digraph(substrate, connectivity))


def strongly_connected_components(
    substrate: Substrate, connectivity: Connectivity = "inferred"
) -> list[tuple[int, ...]]:
    """Strongly connected components as tuples of node indices."""
    return [
        tuple(sorted(component))
        for component in nx.strongly_connected_components(
            _index_digraph(substrate, connectivity)
        )
    ]


def is_dag(substrate: Substrate, connectivity: Connectivity = "inferred") -> bool:
    """Whether the substrate graph is acyclic."""
    return nx.is_directed_acyclic_graph(_index_digraph(substrate, connectivity))


def simple_cycles(
    substrate: Substrate, connectivity: Connectivity = "inferred"
) -> list[list[int]]:
    """All simple cycles as lists of node indices."""
    return [list(cycle) for cycle in nx.simple_cycles(_index_digraph(substrate, connectivity))]


def in_degree(substrate: Substrate, connectivity: Connectivity = "inferred") -> dict[int, int]:
    """In-degree per node index."""
    return dict(_index_digraph(substrate, connectivity).in_degree())


def out_degree(substrate: Substrate, connectivity: Connectivity = "inferred") -> dict[int, int]:
    """Out-degree per node index."""
    return dict(_index_digraph(substrate, connectivity).out_degree())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest test/test_graph.py -q`
Expected: PASS (all Task 1–5 tests).

- [ ] **Step 5: Commit**

```bash
git add pyphi/graph.py test/test_graph.py
git commit -m "Add IIT-relevant topology helpers over the substrate graph

is_strongly_connected, strongly_connected_components, is_dag, simple_cycles,
in_degree, out_degree as thin networkx wrappers returning node-index results
on the inferred-connectivity graph."
```

---

### Task 6: Roadmap, changelog, and full verification

**Files:**
- Create: `changelog.d/b20-networkx-bridge.feature.md`
- Modify: `ROADMAP.md` (B20 dashboard row ~line 37; Wave 2 archive bullet ~line 99; "Landed" prose line ~line 16)

- [ ] **Step 1: Write the changelog fragment**

Create `changelog.d/b20-networkx-bridge.feature.md`:

```markdown
Added a networkx bridge for substrates: `Substrate.to_networkx()` /
`from_networkx()`, `System.to_networkx()` (with per-node state and membership
attributes), `to_graphml()` / `to_adjacency()` export, and topology helpers
(`pyphi.graph.is_strongly_connected`, `strongly_connected_components`,
`is_dag`, `simple_cycles`, `in_degree`, `out_degree`). The exported graph
defaults to the TPM-inferred causal connectivity, so connectivity-matrix edges
that the TPM does not actually realize are dropped; pass
`connectivity="declared"` for the declared matrix. Requires the `visualize`
extra (networkx).
```

- [ ] **Step 2: Update the ROADMAP dashboard row for B20**

Change the B20 row status from `⬜ open` to `✅ landed` and update the one-line:

```markdown
| B20 substrate graph bridge | ✅ landed | 2 | `Substrate.to_networkx()`/`from_networkx()`, `System.to_networkx()` (state + membership node attrs), GraphML/adjacency export, and IIT-relevant topology helpers (`pyphi/graph.py`). Defaults to the **TPM-inferred** causal connectivity (B19 `infer_cm`) so over-specified-`cm` phantom edges are dropped; `from_networkx` inherits B19's under-specified-topology rejection. networkx stays optional (deferred import, `visualize` extra). Excludes BN/CPD semantics (→ N11). |
```

- [ ] **Step 3: Update the Wave 2 archive bullet and the Landed prose line**

In the Wave 2 archive, update the `**B20 — substrate ↔ networkx graph bridge.**` bullet to past tense describing what landed (the surface above, the inferred-default connectivity, the B19 composition in both directions). In the `### ✅ Landed` prose line near the top, append `· B20`.

- [ ] **Step 4: Run the full verification gate**

Run: `uv run pytest`
Expected: PASS with **no path argument** (collects `pyphi/` + `test/` doctests and the new `test/test_graph.py`). Run the slow lane in the background per the project's parallel-test guidance if needed; the gate is the full no-path run.

- [ ] **Step 5: Commit**

```bash
git add changelog.d/b20-networkx-bridge.feature.md ROADMAP.md
git commit -m "Mark B20 landed: substrate networkx bridge; changelog + roadmap"
```

---

## Self-Review

**Spec coverage:**
- Module layout / deferred import (spec 4.1) → Task 1.
- Connectivity selection inferred/declared (spec 4.2) → Task 1 (`_edge_matrix`), threaded through all tasks.
- `to_networkx` Substrate + System (spec 4.3) → Task 1 + Task 3.
- `from_networkx` (spec 4.4) → Task 2.
- GraphML + adjacency export (spec 4.5) → Task 4.
- Topology helpers (spec 4.6) → Task 5.
- Error handling (spec 4.7): unknown connectivity → Task 1 test; size mismatch → Task 2; under-spec rejection via B19 → Task 2; missing networkx → deferred import (Task 1).
- Testing (spec 5): round-trip, inferred-vs-declared, k-ary, System attrs, helpers, export round-trip, lazy import → Tasks 1–5.
- Roadmap + changelog (spec 7) → Task 6.

**Type consistency:** `connectivity` is `str`/`Literal["inferred","declared"]` defaulting to `"inferred"` in every signature. Helpers return node *indices* (built on `_index_digraph`, no relabel); `to_networkx`/`system_to_networkx` return *label*-noded graphs (relabeled). `_edge_matrix` returns `np.ndarray`; `to_adjacency` wraps it in a labeled DataFrame.

**Placeholder scan:** none — every code step shows complete code.

**Note on the joint-TPM accessor (Task 2):** `Substrate.joint_tpm()` (substrate.py:243) returns the explicit-alphabet joint array shape the constructor's `tpm=` argument forwards to `FactoredTPM.from_joint`, so `Substrate.from_networkx(g, sub.joint_tpm())` round-trips cleanly. (A bare 2-D state-by-node array — as in the `_phantom_edge_substrate` / size-mismatch tests — is also accepted by the constructor and by `from_networkx`'s `shape[-1]` unit-count check.)
