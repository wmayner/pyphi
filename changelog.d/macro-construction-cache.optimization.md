The macro TPM construction now caches its mapping-independent Steps 1-2
intermediates (the discounted transition matrix and the sequence-class
distributions, Eqs. 26-31 of Marshall et al. 2024) per substrate, keyed on
the unit's footprint, micro grain, and apportionment structure — so
grain-search candidates that differ only in their mapping reuse the
expensive construction prefix. Results are identical with the cache on or
off. Gated by the new infrastructure option `cache_macro_construction`
(default on); entries are evicted when the substrate is garbage-collected.
