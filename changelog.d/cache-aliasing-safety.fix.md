Fixed cache and aliasing safety holes: arrays returned by the kernel
repertoire cache and `max_entropy_distribution()` are now read-only, so
caller mutation raises instead of silently corrupting every later
computation on equivalent systems; `FactoredTPM` copies and freezes its
factors at construction, so mutating the arrays passed in can no longer
change a hashed value type or stale the substrate fingerprint; disk
result-cache hits are decoded with the requesting system's node labels
instead of the computing system's; and `cache_repertoires = false` now
actually disables kernel repertoire caching.
