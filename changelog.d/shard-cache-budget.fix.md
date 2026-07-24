Bounded the in-memory caches of a campaign shard by the shard's own memory
request. A shard evaluates every mechanism it carries against one long-lived
`System`, whose cached repertoires are released only when that `System` is
collected, so a shard packing many mechanisms accumulated cache entries with no
effective ceiling: `maximum_cache_memory_percentage` measures against the
machine's total RAM, which does not bound a job confined to a smaller
allocation. The new `maximum_cache_memory_bytes` option gives an absolute
ceiling, set automatically during shard execution, and `shard_memory_bytes` now
includes the cache allowance it grants, so the request and the enforced ceiling
come from the same figure.
