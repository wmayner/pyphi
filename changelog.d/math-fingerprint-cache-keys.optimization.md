The repertoire kernel cache and the potential-purview cache are now keyed on a
label-free content fingerprint of the system's mathematics rather than on object
identity. Mathematically-equivalent systems — re-constructed copies, or the same
substrate under different node/state labels — now reuse each other's cached
results instead of recomputing, and `potential_purviews` is shared across every
substrate with the same connectivity (so a parameter sweep over a fixed topology
reuses it). Cached entries are still released promptly when the systems that
produced them are garbage-collected. No computed value changes.

The internal per-`Substrate` `purview_cache` constructor argument and attribute
were removed (the cache is now a single module-level object).
