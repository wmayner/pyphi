A campaign shard's cache ceiling now falls back to the memory its submit
file requested, read from the `PYPHI_SHARD_MEMORY` environment variable the
generated submit file exports. It stands between the cgroup limit and the
request recorded at planning time, so raising a job's memory raises the
ceiling with it even on a pool that grants memory without confining the job
to it, where no cgroup limit is readable.
