In-memory caches now evict under memory pressure instead of freezing. Once
resident memory reaches `memory_ceiling_bytes` (or
`memory_ceiling_percentage`), a cache used to refuse every new entry for
the rest of the process, so which results stayed cached was decided by whichever
happened to be computed first. It now holds its occupancy at the level it had
reached and admits new entries by discarding the least recently used ones. On a
scoped cause-effect structure sweep with the ceiling binding over the last 58%
of the work, that cut the cost of the ceiling from 1.46× the unbounded run to
1.04×, and raised the hit rate from 72.6% to 95.5% against an unbounded 95.6% —
while holding fewer entries and less resident memory than freezing did.

Occupancy is measured in bytes rather than entries, since the argument spaces
differ in kind: the combinatorial index tables are keyed on a sequence length
alone, giving one entry per system size but values growing as 2ᴺ or 3ᴺ, while
`max_entropy_distribution` is keyed on a purview, giving one entry per subset.
Neither is bounded by a count. An entry too large to fit the whole budget is
skipped rather than allowed to displace everything else, and a ceiling reached
during a transient spike is re-checked and lifted if memory frees up again.

Eviction does not lower resident memory — freeing Python objects returns their
memory to the process allocator, rarely to the operating system. What it changes
is which entries a fixed allocation is spent on.

`pyphi.cache.info()` now reports `nbytes` and `evictions` alongside hits, misses,
and entry count.
