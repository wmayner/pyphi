`pyphi.cost.estimate_analysis` now counts the specified-state search as its own
work axis, `specified_state_evaluations`. The search maximizes intrinsic
information over the whole system as both mechanism and purview, so it performs
two forward repertoire evaluations per system state — a cost that grows with
the size of the system rather than of any mechanism, and one that none of the
partition or purview axes bound. The MCP server's `analyze` guard checks the
new axis, so an analysis whose cost is dominated by the search is refused with
that reason named instead of being admitted on modest partition counts.
The `performance` reference topic now calls the axis out, since it is the one
that dominates a large *sparse* system: thinning connectivity shrinks every
other axis through purview pruning but leaves this one untouched.
