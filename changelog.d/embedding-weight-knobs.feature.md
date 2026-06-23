The global-embedding layout of the hypergraph view (`pyphi.visualize.plot_ces`)
now exposes the relative weight of each composition block as
`HypergraphGeometry` fields: `embed_purview_weight`, `embed_mechanism_weight`,
and `embed_direction_weight`. They scale the purview, mechanism, and direction
contributions for both the PCA feature vectors and the MDS distance blend, so a
layout can be tuned to emphasize, say, mechanism similarity. Purview dominates
by default (`1.0` vs `0.5`/`0.5`), preserving the previous behavior.
