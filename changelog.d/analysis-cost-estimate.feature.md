Added `pyphi.estimate_analysis()`: an analytic pre-flight that counts the
workload of a single-system analysis — system partitions under the active
scheme, candidate mechanisms, connectivity-pruned purview evaluations, and
mechanism-partition sweeps — without computing any φ. The MCP server's
`analyze` guard now gates on these estimated counts rather than node
counts, and a new `estimate_cost` MCP tool exposes the estimate.
