Added a pure projection layer for visualization
(`pyphi.visualize.projection`) and a new inclusion-lattice (Hasse) view of
cause-effect structures: `pyphi.visualize.plot_ces(ces, view="lattice")`
draws distinctions ranked by an inclusion partial order (`order="mechanism"`
by subset relation on mechanisms, or `order="purview_union"` on
cause/effect-purview unions), sized by total relation phi and colored by
distinction phi. Vertical placement is configurable via `rank="chain"`
(compact) or `rank="size"` (gaps at sizes with no distinctions), horizontal
placement via `layout="barycentric"` (crossing-reducing, the default) or
`layout="sorted"` (label order), and the marker encodings via `size_by` and
`color_by`. A `Theme` dataclass replaces ad-hoc theme overrides. The 3-D
hypergraph view is rebuilt on the same projection
(`view="hypergraph"`, with `show` element selection and a
`HypergraphGeometry` layout dataclass), `highlight_phi_fold` is
reimplemented over it, and the legacy `pyphi.visualize.ces` module is
removed. Two further views complete the set: `view="scatter"` (distinctions
on a deterministic PCA composition embedding, sized by total relation phi,
colored by relational role, with connectedness symbols) and `view="matrix"`
(a distinctions-by-distinctions heatmap of shared relation phi, with
self-relation strength on the diagonal).
