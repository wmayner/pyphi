The MCP server's always-loaded instructions now carry the gotchas reference
alongside the primer, so the mistakes that produce wrong results — reporting φₛ
as Φ, reading a state as big-endian, treating Φ = 0 as "no structure" — are in
front of the assistant before its first tool call instead of waiting behind
`get_iit_reference("gotchas")`. The primer's abbreviated version of the same
material is removed. The other topics remain on demand.
