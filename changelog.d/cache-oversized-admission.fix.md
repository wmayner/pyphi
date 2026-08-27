A cache entry larger than the store's whole byte budget is now refused up front. Previously the eviction loop drained the entire working set before refusing it.
