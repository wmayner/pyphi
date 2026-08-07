`pyphi.analyze(..., compute="distinctions")` computes a system's distinctions
without the system-partition search, and the MCP server's `analyze` tool takes
the same option. Under IIT 4.0 unfolding a cause-effect structure computes a
system irreducibility analysis first, and over a sparse substrate that search
is most of the running time — on the nine-unit `propagation_delay` example it
is 195 of the 196 seconds, so the distinctions alone take 13.

The distinctions come back filtered for congruence with the system's specified
state, exactly as a cause-effect structure filters them, whenever that state is
untied — the specified state a system irreducibility analysis would start from
is available without evaluating a single partition. When the state ties, the
tie is broken by the φₛ cascade over the tied cause/effect pairs, which does
need the partition search; the unfiltered distinctions are returned instead, as
an `UnresolvedDistinctions` rather than a `ResolvedDistinctions`. The two are
worth telling apart: congruence filtering can remove any number of
distinctions, including all of them, so an unfiltered count and Σφ_d are upper
bounds on the structure's rather than estimates of it. The MCP tool reports
which of the two it has under a `congruence` key, and renames the unfiltered
counts to `num_distinctions_upper_bound` and `sum_phi_distinctions_upper_bound`
so they cannot be read as the structure's.

`System.distinctions()` gains a `congruent` argument for the same thing, and
`pyphi.cost.estimate_analysis` accepts `compute="distinctions"`.
