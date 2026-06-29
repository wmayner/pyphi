# N11 — Lightweight dynamic Bayesian network (DBN) export — Design

## Goal

Render a `Substrate` as a **2-timeslice dynamic Bayesian network** (2-TBN):
each substrate node `X` becomes two variables, `(X, 0)` at time `t` and
`(X, 1)` at time `t+1`; the per-node TPM factors become conditional
probability distributions (CPDs). This enables downstream pgmpy /
d-separation / Markov-blanket workflows **without adding a pgmpy
dependency**.

The static single-slice view of a substrate is unfaithful — substrates are
cyclic by construction (feedback, self-loops). The 2-TBN unrolling across one
timestep is the correct acyclic target: all edges cross the time boundary
`t → t+1`, so the unrolled graph is a DAG even when the substrate is not.

## Scope

In scope:

- Two export functions in `pyphi/graph.py`, exposed as `Substrate` methods:
  - `Substrate.to_dbn()` → `networkx.DiGraph` (d-separation-ready, pgmpy-adaptable).
  - `Substrate.to_dbn_dict()` → plain `dict` (JSON-friendly, no networkx import).

Out of scope (unchanged from B20 and the standing dependency rules):

- No `pgmpy` dependency. The export is a plain structure callers adapt.
- No `System`-level entry point. The DBN is dynamics + topology; it does not
  depend on a state or a node subset. (This also keeps it off the
  `PUBLIC_SYSTEM_ATTRS` / `SystemPublicInterface` / `_DELEGATED_TO_SYSTEM`
  surface contract that any new public `System` method must be threaded
  through.)
- No `connectivity="declared"` knob (see "Parents are inferred, always").
- No weights, no igraph, no GraphML/serialization of the CPD tables beyond the
  dict form.

## Architecture

### Variable identity and edges

- Each substrate node with label `X` yields two DBN variables, the tuples
  `(X, 0)` and `(X, 1)`. The integer second element is the timeslice.
- The only edges are inter-slice: `(parent, 0) → (child, 1)`, one per inferred
  causal edge into `child`.
- Self-loops are kept: if `X` is its own inferred parent, `(X, 0) → (X, 1)` is
  present and `X` appears in its own parent list.
- All `2N` variables appear in the graph, including any `(X, 0)` that is never
  a parent and any `(X, 1)` with no parents (a marginal CPD).

### Parents are inferred, always

The parents of `child` are exactly its inferred in-neighbors — the nodes its
factor actually depends on (`FactoredTPM.infer_edge` / `infer_cm`, B19/B20).
This is a correctness requirement, not a default: a CPD is constant along a
declared-but-causally-inert input axis, so listing such a node as a parent
would corrupt any d-separation or Markov-blanket query computed on the result.
Therefore N11 exposes **no** `connectivity=` parameter (unlike the topology
exports in B20, where "declared" is a meaningful alternative).

### CPD extraction

`FactoredTPM.factor(i)` is `P(Xᵢ' | s_t)` with shape
`alphabet_sizes + (aᵢ,)`: one input axis per substrate node (the full state at
`t`) followed by the child's own distribution axis. The CPD is that array
reduced to its parent axes — index `[0]` along every non-parent input axis,
which is exact because the factor is constant along those axes by definition of
"not a parent".

The resulting table has conditioning axes in **ascending parent-index order**,
followed by the child-distribution axis last:

```
table[p₁, …, p_k, :] == P(Xᵢ' | parents = (p₁, …, p_k))
```

The accompanying `parents` list holds the parent labels in that same axis
order. Each `table[p₁, …, p_k, :]` sums to 1. A parentless node yields a
shape-`(aᵢ,)` marginal with `parents == []`.

### Shared core and two outputs

A private helper `_dbn_factors(substrate)` yields, per node `i` in index order,
the triple `(child_label, parent_labels, cpd_table)`. Both public functions
consume it:

- `substrate_to_dbn(substrate) -> networkx.DiGraph`
  - Adds all `2N` variables. Each `(X, 1)` node carries attributes `cpd`
    (the ndarray), `parents` (ordered label list), and `time` (1); each
    `(X, 0)` node carries `time` (0).
  - Adds the inter-slice edges.
  - Requires networkx (deferred import via `pyphi.deferred`, `visualize`
    extra), consistent with the rest of `pyphi/graph.py`.

- `substrate_to_dbn_dict(substrate) -> dict`
  - Returns:
    ```python
    {
        "variables": {label: alphabet_size, ...},      # all nodes, index order
        "edges": [(parent_label, child_label), ...],   # inter-slice, t → t+1
        "cpds": {label: {"parents": [...], "table": ndarray}, ...},
    }
    ```
  - Pure numpy / dict — **no networkx import**, so it works without the
    `visualize` extra.

### Substrate methods

```python
def to_dbn(self) -> Any: ...        # -> networkx.DiGraph
def to_dbn_dict(self) -> dict: ...  # plain dict
```

Both delegate to `pyphi.graph` via a local import, matching the existing
`to_networkx` / `to_adjacency` wiring in `pyphi/substrate.py`.

## Error handling

No new failure modes. CPD extraction is total over any valid `FactoredTPM`
(binary or k-ary, any alphabet sizes). `to_dbn` raises the existing deferred
`ImportError` if networkx is absent; `to_dbn_dict` never needs it.

## Testing (`test/test_graph.py`)

- **Asymmetric binary substrate** (parents differ per node, so an axis or
  parent-ordering slip surfaces — the lesson from the TriggeredTPM axis bug,
  where symmetric fixtures hid an axis/endianness error):
  - inter-slice edges equal the inferred-cm edges;
  - each CPD table shape equals `parent_alphabets + (child_alphabet,)`;
  - each table equals the factor reduced along its non-parent axes;
  - every conditioned row sums to 1.
- **Acyclicity**: `networkx.is_directed_acyclic_graph(G)` is `True` for a
  substrate that is itself cyclic (e.g. has self-loops) — the defining property
  of the 2-TBN unroll.
- **Self-loop case**: `(X, 0) → (X, 1)` edge present and `X ∈ parents(X)`.
- **k-ary case**: a non-binary alphabet substrate exports CPD tables of the
  correct heterogeneous shape.
- **dict ↔ DiGraph agreement**: `to_dbn_dict` and `to_dbn` report the same
  parents and identical CPD tables for the same substrate.
- **Parentless node**: a node whose factor is constant in every input yields a
  shape-`(aᵢ,)` marginal CPD with `parents == []` and no incoming edges.

## Documentation

- Docstrings on both `graph.py` functions and both `Substrate` methods,
  describing the 2-TBN unroll, the inferred-parents rule, and the CPD axis
  order — describing the final behavior, with no migration narrative.
- Changelog fragment under `changelog.d/`.
- ROADMAP N11 row flipped to landed with a one-line summary.
