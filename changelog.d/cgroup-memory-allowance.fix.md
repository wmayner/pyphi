The cache memory limit is now measured against the memory the process is
actually allowed. `maximum_cache_memory_percentage` took its denominator from
total physical memory, which is no bound at all on a process confined to a
smaller allocation — a scheduler-managed job, a container, a cgroup. It now
reads the process's cgroup allowance (v2 `memory.max`, falling back to v1
`memory.limit_in_bytes`, and to the hierarchy root inside a container's cgroup
namespace), and uses physical memory only when no allowance is reported.

Campaign shard execution derives its ceiling the same way. It previously used
the memory request recorded at *planning* time, so a job granted more memory
than planning predicted kept the smaller ceiling and its caches never got the
extra room. That silently defeated a paired experiment: a shard rerun at a
four-times-larger request stopped growing at exactly the same occupancy as the
original, because both were enforcing the planned figure. The planned request
now stands in only where the allocation cannot be read.
