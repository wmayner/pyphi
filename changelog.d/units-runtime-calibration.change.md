Campaign work units now weight their two axes by measured cost. A purview
evaluation is charged `pyphi.cost.PURVIEW_EVALUATION_UNITS` (12) rather than
1, which is what it costs relative to one mechanism partition, so a unit
means the same amount of work whichever rung of the shard-planning ladder
produced the shard carrying it. `pyphi.cost.SECONDS_PER_UNIT`,
`units_for_runtime()`, and `runtime_seconds()` convert between units and CPU
seconds, so `units_per_job` can be set from a per-shard runtime target.
