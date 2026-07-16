The MCP server now teaches and controls parallelization. A new
`parallelization` reference topic explains the conditions under which
work actually parallelizes (the global `parallel` gate, each level's own
flag, and the per-level `sequential_threshold`), the seven parallelizable
levels, and which levels pay
off for which workloads. The `analyze` tool accepts `parallel` (`true` for the
recommended levels, a list of level names for explicit control, `false` to
force a sequential run) and `workers` arguments scoped to the call, and the
new `configure_parallel` tool reads or persistently sets the server's
parallelization configuration.
