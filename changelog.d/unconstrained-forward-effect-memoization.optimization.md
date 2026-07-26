`unconstrained_forward_effect_repertoire` is no longer memoized, completing the
pair with its cause-direction counterpart. `intrinsic_information` requests each
`(mechanism, purview)` once, so an entry is essentially never read back: a
1,242-shard campaign recorded 100.1 million stores, 54.1 million evictions, and
**zero** reads. Lifting the purview-order cap does not change this — the call
count grows tenfold across a sweep and repeats stay at zero — so the cache
stored and discarded, and did so more expensively at higher orders.

One reuse does exist, on systems small enough that their own units form a
candidate purview: a system-level request can coincide with the distinction
whose mechanism and purview are both the whole system. On `rule110` that is a
single repeat in fifty calls, worth eight extra `effect_repertoire` calls out of
4,689. The per-state loop that carries the real cost keeps its own caching one
level down, in `effect_repertoire`.
