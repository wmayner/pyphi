The hypergraph view's global-embedding layout (`pyphi.visualize.plot_ces`
with `layout="embedding"`) can now be flattened to a plane by setting
`HypergraphGeometry(embed_planar=True)`. The embedding is computed in two
components with `z = 0`, mirroring the planar scatter view. The default
remains the volumetric three-dimensional embedding.
