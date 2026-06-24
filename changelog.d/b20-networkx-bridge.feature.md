Added a networkx bridge for substrates: `Substrate.to_networkx()` /
`from_networkx()`, `System.to_networkx()` (with per-node state and membership
attributes), `to_graphml()` / `to_adjacency()` export, and topology helpers
(`pyphi.graph.is_strongly_connected`, `strongly_connected_components`,
`is_dag`, `simple_cycles`, `in_degree`, `out_degree`). The exported graph
defaults to the TPM-inferred causal connectivity, so connectivity-matrix edges
that the TPM does not actually realize are dropped; pass
`connectivity="declared"` for the declared matrix. Requires the `visualize`
extra (networkx).
