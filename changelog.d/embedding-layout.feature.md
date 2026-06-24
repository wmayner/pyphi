The hypergraph view (`pyphi.visualize.plot_ces`) gains a
`layout="embedding"` mode that positions each MICE by a deterministic embedding
of its composition (purview, mechanism, and direction) instead of the size
shells, so spatial proximity reflects compositional similarity. The method is
chosen by `HypergraphGeometry.embedding_method`: `"mds"` (default,
classical multidimensional scaling of a purview-overlap distance, which clusters
relation-forming MICE) or `"pca"` (a principal-component embedding). Both are
deterministic and need no new dependency.
