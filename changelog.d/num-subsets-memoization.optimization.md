`pyphi.combinatorics.num_subsets_larger_than_one_element` is no longer memoized,
and so no longer carries `cache_info()` / `cache_clear()`. It evaluates
`2**n - n - 1` in about 109 ns, against roughly 250 ns for the cache lookup that
was wrapping it, so caching it cost more than it saved.
