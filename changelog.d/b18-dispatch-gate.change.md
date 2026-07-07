The parallel dispatch gate now parallelizes any workload at or above the
level's `sequential_threshold`, spreading it across all cores via the
chunker's worker floor even when it fits within a single `chunksize`.
Previously workloads at or below one chunksize always ran sequentially, so
e.g. a 4,000-relation-candidate workload got no parallelism while 4,097
candidates fanned out. An explicitly configured `chunksize` now governs
chunk granularity only; a cost-sampled chunksize still gates dispatch,
since it estimates the number of items per ~1 s of work. Measured in
`benchmarks/b18_dispatch_gate.py`: 3-4x faster purview evaluation at 64-230
purviews, 2.5-4x faster system-partition evaluation at 64-2048 partitions,
1.2-1.6x faster complex evaluation at 16-31 candidates, with no regression
on reducible (short-circuiting) systems.
