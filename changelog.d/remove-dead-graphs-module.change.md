Removed the unused `pyphi.graphs` module and the `graphs` optional-dependency
extra (`igraph`). Its `maximal_independent_sets` / `largest_independent_sets`
helpers were superseded by the graphillion `setset` logic in
`pyphi.combinatorics` / `pyphi.relations`, had no remaining callers, and were
broken by the 2.0 rename sweep. `networkx` remains available via the
`visualize` extra.
