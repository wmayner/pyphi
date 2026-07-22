Multi-cell `prepare_ces` campaigns accept per-cell congruence-resolution
states: a mapping keyed by the full `(label, formalism, subset, state)`
cell tuples, a state-keyed mapping when the other axes are singletons,
or a callable `cell -> specification`. One campaign spanning many states
plans its shard ladder once, suppresses SIA shards for every cell, and
collects each structure congruence-resolved against its own state —
previously this required one single-cell campaign per state, re-planning
the identical ladder every time. `collect`'s `resolution_state` override
accepts the same forms. Values are validated at preparation time (a
wrong type now fails immediately with a pointer at
`system_intrinsic_information`, not deep inside congruence resolution at
collect time). Resolution states are stored per cell as
`resolution_state-<cell>.json.gz`.
