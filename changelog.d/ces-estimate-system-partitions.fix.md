`pyphi.cost.estimate_analysis(substrate, compute="ces")` now counts the
system-partition axis under IIT 4.0, where unfolding a cause-effect structure
computes a system irreducibility analysis before it unfolds anything (Eq. 57).
It previously reported `system_partitions=None` for that analysis under every
formalism, which was right only for IIT 3.0, whose cause-effect structure is
the bare distinctions.

The MCP server's `analyze` guard inherited the undercount, and refused an
analysis on whichever single axis matched the requested `compute`. It now
weighs every axis the analysis walks against that axis's own limit. A sparse
substrate could be trivial on the distinction axis and enormous on the
system-partition axis, so a `compute="ces"` request that ran for hours was
waved through on a count of a few dozen mechanism-partition sweeps. Such a
request is now refused without `confirm_large`, and the refusal points at
`compute="distinctions"`, which skips that axis.
