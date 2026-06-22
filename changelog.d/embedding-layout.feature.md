The simplicial-complex view (`pyphi.visualize.plot_ces`) gains a
`layout="embedding"` mode that positions each MICE by a deterministic embedding
of its composition (purview, mechanism, and direction) instead of the size
shells, so spatial proximity reflects compositional similarity. The method is
chosen by `SimplicialComplexGeometry.embedding_method`: `"pca"` (default, a
principal-component embedding) or `"mds"` (classical multidimensional scaling of
a purview-overlap distance). Both are deterministic and need no new dependency.
