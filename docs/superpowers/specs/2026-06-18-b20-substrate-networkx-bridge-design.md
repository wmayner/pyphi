# B20 — Substrate ↔ networkx graph bridge: Design

**Status:** approved
**Date:** 2026-06-18
**Wave:** 2 (pre-freeze, surface-affecting)
**Pairs with:** P11.95c (automorphism canonicalization), P14d (labeled export), B19 (CM/TPM consistency — `infer_cm`)

---

## 1. Motivation

PyPhi value types describe a directed graph (the connectivity matrix) but expose no way to hand
that graph to the wider Python graph ecosystem. Users who want degree/SCC/cycle analysis, GraphML
export, or interop with networkx-based tools must reconstruct the graph by hand. The only existing
networkx use is one private figure builder (`visualize/connectivity.py::_system_graph`,
`nx.from_numpy_array(system.cm, create_using=nx.DiGraph)`), not a reusable bridge.

B20 adds a thin, labeled bridge: `Substrate.to_networkx()` / `from_networkx()`, a `System.to_networkx()`
that annotates nodes with state and membership, GraphML + adjacency export, and a minimal set of
IIT-relevant topology helpers. It is **topology only** — it explicitly excludes Bayesian-network /
CPD / dynamic-BN semantics, which are the separate N11 item.

**The connectivity-inference subtlety (the central design point).** A substrate's declared `cm` can
be *over-specified*: B19 (`FactoredTPM.infer_cm`) infers the true causal edge set (an edge `a→b`
exists iff factor `b` is non-constant along input axis `a`, precision-aware) and B19 deliberately
**permits** the declared `cm` to be a superset of those real edges — it only rejects
under-specification. So exporting `cm` raw would put causally-inert phantom edges into the graph,
which would mislead exactly the topology questions the bridge is for (strong-connectivity, cycles,
SCCs → "is this substrate a complex?"). The bridge therefore defaults to the **TPM-inferred**
connectivity, with the declared `cm` available on request.

## 2. Goals

- Export a labeled `networkx.DiGraph` from a `Substrate` (and a `System`, with node state +
  membership attributes), defaulting to the causally-real (TPM-inferred) edge set.
- Reconstruct a `Substrate` from a DiGraph topology plus a supplied TPM.
- GraphML and labeled-adjacency export.
- A small set of IIT-relevant topology helpers (thin networkx wrappers).
- Keep `networkx` optional (the existing `visualize` extra), never eagerly imported at
  `import pyphi`.

## 3. Non-goals

- Edge weights (the `Substrate` retains no weights — they live only in a `substrate_generator`).
- Bayesian-network / CPD / 2-timeslice-DBN semantics (→ N11).
- igraph.
- Any new graph algorithm PyPhi would reimplement — hand users the `DiGraph` for anything beyond
  the minimal helper set.

## 4. Design

### 4.1 Module layout

A new focused module `pyphi/graph.py` holds all graph logic; the value types get thin delegating
methods so `substrate.py` / `system.py` do not grow:

- `pyphi/graph.py` — DiGraph construction, the `from_networkx` constructor helper, GraphML +
  adjacency export, topology helpers.
- `Substrate.to_networkx(...)`, `Substrate.from_networkx(...)` (classmethod),
  `Substrate.to_graphml(path, ...)`, `Substrate.to_adjacency(...)` — thin delegations to
  `pyphi/graph.py`.
- `System.to_networkx(...)` — builds the substrate graph, then sets per-node attributes.
- `pyphi/deferred/deferred_import.py` — add `DeferredNetworkX` mirroring the existing
  `DeferredPlotly`, so `import pyphi` never loads networkx and a missing dependency raises
  `MissingOptionalDependenciesError` with a clear message. `networkx` stays declared in the
  `visualize` extra (already present there).

### 4.2 Connectivity selection (inferred vs declared)

A `connectivity` keyword threads through every graph-producing entry point:

```
connectivity: Literal["inferred", "declared"] = "inferred"
```

- `"inferred"` (default) — edges from `substrate.factored_tpm.infer_cm()` (B19): the true causal
  connectivity. Drops phantom edges that the declared `cm` over-specifies.
- `"declared"` — edges from `substrate.cm` as specified.

Because B19 guarantees `cm ⊇ inferred`, the inferred graph is always a subset of the declared one
(never the reverse surprise). The inference cost is `O(N² · factor_size)` — negligible at the
`n ≲ 10` substrates Φ reaches; computed fresh per call (no caching — YAGNI).

### 4.3 `to_networkx`

`Substrate.to_networkx(connectivity="inferred") -> nx.DiGraph`:
build `nx.from_numpy_array(edge_matrix, create_using=nx.DiGraph)` where `edge_matrix` is the
inferred or declared CM, then relabel integer nodes to `self.node_labels` (the
`visualize/connectivity.py` pattern). Edges are exactly the chosen matrix's nonzeros; self-loops
(diagonal entries) are preserved; the graph is unweighted. A substrate with no edges yields a
DiGraph with all nodes and no edges.

`System.to_networkx(connectivity="inferred") -> nx.DiGraph`: build the substrate graph, then set
node attributes:
- `state` — the unit's current value (`system.state[i]`).
- `in_system` — `bool`, whether the node is in `system.node_indices` (vs background).

### 4.4 `from_networkx`

`Substrate.from_networkx(graph, tpm, *, node_labels=None) -> Substrate`:
the graph supplies topology (CM = its adjacency, self-loops kept) and node labels (from
`graph.nodes`, unless `node_labels` overrides); **`tpm` is required** (a substrate without dynamics
is meaningless). Construction goes through the normal `Substrate` constructor, which runs B19's
default-on `validate_connectivity` — so a supplied topology that *omits* a real TPM edge is rejected
there (over-specification stays legal). A node-count vs TPM-unit-count mismatch raises `ValueError`
before construction with a message naming both counts.

Round-trip: `substrate → to_networkx("inferred") → from_networkx(g, substrate.tpm)` reproduces the
causal topology. When the original `cm == inferred` (the normal case) this is exact on `cm` and
labels; when the original `cm` was over-specified, the round-tripped `cm` is the canonical inferred
subset.

### 4.5 Export

- `Substrate.to_graphml(path, connectivity="inferred") -> None` → `nx.write_graphml(self.to_networkx(...), path)`.
- `Substrate.to_adjacency(connectivity="inferred") -> pandas.DataFrame` → a node-labeled adjacency
  DataFrame (rows/cols = node labels, values = the chosen CM), consistent with the P14d labeled-export
  convention (`pandas` is a core dependency).

### 4.6 Topology helpers

Module functions in `pyphi/graph.py`, each `(substrate, connectivity="inferred")`, thin wrappers
over networkx mapping to IIT-relevant questions:

- `is_strongly_connected(substrate, ...) -> bool` (a complex must be strongly connected)
- `strongly_connected_components(substrate, ...) -> list[tuple[int, ...]]`
- `is_dag(substrate, ...) -> bool`
- `simple_cycles(substrate, ...) -> list[list[int]]`
- `in_degree(substrate, ...) -> dict[Any, int]`
- `out_degree(substrate, ...) -> dict[Any, int]`

Anything beyond this, users get from the returned `DiGraph`.

### 4.7 Error handling

- Missing `networkx` → `MissingOptionalDependenciesError` via `DeferredNetworkX` (raised on first
  use, never at import).
- `from_networkx` node/TPM size mismatch → `ValueError`.
- `from_networkx` topology that under-specifies the TPM's real edges → `ConfigurationError` from
  B19's `validate_connectivity` (inherited, not re-implemented).

## 5. Testing

`test/test_graph.py`:

- **Round-trip:** `substrate → to_networkx → from_networkx(g, tpm)` yields equal `cm` and labels on
  a substrate whose `cm == inferred`.
- **Inferred vs declared (the key test):** construct a substrate with `cm` over-specified by one
  causally-inert edge (built with `validate_connectivity=False`); assert `to_networkx()` (inferred)
  omits that edge while `to_networkx("declared")` keeps it.
- **Edge fidelity:** the inferred graph's edge set equals `factored_tpm.infer_cm()` nonzeros,
  self-loops preserved; node count and labels preserved on a **binary and a k-ary** substrate (CM is
  binary regardless of alphabet).
- **`System.to_networkx`:** node attributes `state` and `in_system` correct for a subset system
  (some nodes background).
- **Topology helpers:** pinned values on example networks (e.g. grid3 strong-connectivity), a
  constructed DAG (`is_dag` True, `simple_cycles` empty), and a known cyclic case.
- **Export:** `to_graphml` write → `nx.read_graphml` read round-trip via `tmp_path`;
  `to_adjacency` returns a labeled DataFrame matching the chosen CM.
- **Optional-dependency:** `DeferredNetworkX` raises `MissingOptionalDependenciesError` when networkx
  is absent (assert on the deferred object's behavior; the env has networkx so this is a targeted
  unit check, not a full uninstall).
- **Lazy import:** extend `test/test_lazy_imports.py` to assert `networkx` is not loaded by
  `import pyphi`.

Verification runs `uv run pytest` **with no path argument** (B20 is public surface; the `pyphi/`
doctest sweep must run).

## 6. Risks and mitigations

- **Default differs from `substrate.cm`.** `to_networkx()` defaulting to inferred can surprise a
  user comparing against `substrate.cm`. Mitigated by the explicit `connectivity` parameter and
  clear docstrings; the default is the correct one for causal graph analysis.
- **`infer_cm` on a substrate built with `validate_connectivity=False`.** If `cm` under-specifies
  (e.g. the residue input-unit idiom), `inferred` can exceed `declared`; both are available and
  documented. Not a correctness problem — the inferred set is the causal ground truth.
- **networkx optional.** Handled by the established deferred-import pattern; no new mandatory
  dependency.

## 7. Acceptance criteria

- `Substrate.to_networkx`/`from_networkx`/`to_graphml`/`to_adjacency`, `System.to_networkx`, and the
  six topology helpers exist with the `connectivity` parameter defaulting to `"inferred"`.
- `import pyphi` does not load networkx (lazy-import test green).
- `test/test_graph.py` green, including the inferred-vs-declared phantom-edge test and the k-ary
  case.
- `uv run pytest` (no path argument) green, including doctests.
- ROADMAP dashboard shows B20 ✅; changelog fragment present.
