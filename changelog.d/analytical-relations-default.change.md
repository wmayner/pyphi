`relation_computation` now defaults to `"ANALYTICAL"`: `ces.relations` is a
closed-form summary that answers aggregate queries (`sum_phi()`,
`num_relations()`, `degree_spectrum()`, `strongest(k)`, …) without
enumerating the exponentially large relation set, and agrees numerically
with the concrete backend. Iterating or indexing the summary raises
`TypeError`; use `.strongest(k)` for the top-k relations by φ_r,
`.materialize()` to enumerate explicitly, or set `relation_computation:
CONCRETE` under the `iit` sub-key of the `formalism` section in
`pyphi_config.yml` to restore enumerated relation sets. Plotting renders
the strongest 1000 relations by default when the set is not enumerable.
