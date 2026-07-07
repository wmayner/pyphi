Retuned per-level `sequential_threshold` defaults to match measured
per-item costs, since the threshold is now the sole dispatch gate:
`parallel_partition_evaluation` 1024 → 64 (system partitions cost ~ms-10s
each, so small partition counts — including all IIT 3.0 cut sets below
n≈11 — now parallelize); `parallel_mechanism_partition_evaluation` and
`parallel_relation_evaluation` 1024 → 8192 (mechanism partitions ~50 µs
and lazy relation construction ~µs showed no parallel benefit below that
size — relation dispatch cost is dominated by pickling results back).
