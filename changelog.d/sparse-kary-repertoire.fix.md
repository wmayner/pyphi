Fixed cause and effect repertoire computation for multi-valued (k > 2) units.
Node TPM construction derived the dimensions to marginalize from a binary-only
heuristic, so for k-ary units the node's own previous-state dimension and any
severed-input dimension were never collapsed. This had two consequences:
networks that were both sparsely connected and heterogeneous-alphabet raised a
shape error when building repertoires, and on k-ary networks system partitions
were silent no-ops on k-ary dimensions — a partition never severed the
dependency, so integrated information was under-reported. Both are now correct,
verified to machine precision against an independent reference of the
repertoire definitions and against the three-candidate voting example of
Albantakis et al. 2019 (Figure 11). Binary networks are unaffected.
