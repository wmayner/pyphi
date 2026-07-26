`maximum_cache_memory_bytes` and `maximum_cache_memory_percentage` are renamed
to `memory_ceiling_bytes` and `memory_ceiling_percentage`. Both are compared
against the process's *total* resident memory, of which the in-memory caches are
generally a small part — sampled through a 21-unit cause-effect-structure shard,
they held 70–130 MB against 2.6 GB resident. The old names read as a bound on
cache size, which invites sizing them from expected cache occupancy and being
wrong by more than an order of magnitude.

Behaviour is unchanged: the caches remain what responds to the ceiling, since
they are the only component that can give memory back on request. A configuration
file still using the old names fails at load with a `ConfigurationError` naming
the unknown field.
