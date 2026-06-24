Rebuilt the developer benchmark suite for the 2.0 architecture. ASV
benchmarks now run off the golden-regression fixtures (every fixture across
the three formalisms, layered as repertoires / mechanism MIPs / phi-structure
/ SIA, plus parallel, cache, EMD, and Actual Causation benchmarks). A nightly
workflow accumulates results and alerts on wall-time regressions, and a
deterministic cProfile call-count gate (`test/test_perf_counters.py`) fails CI
on any change to hot-path call counts. The `justfile` gained `bench`,
`bench-dashboard`, `bench-compare`, `perf-gate`, and `perf-pins` recipes for
running these locally.
