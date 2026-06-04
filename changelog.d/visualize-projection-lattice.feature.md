Added a pure projection layer for visualization
(`pyphi.visualize.projection`) and a new inclusion-lattice (Hasse) view of
phi-structures: `pyphi.visualize.plot_phi_structure(ces, view="lattice")`
draws distinctions ranked by an inclusion partial order (`order="mechanism"`
by subset relation on mechanisms, or `order="purview_union"` on
cause/effect-purview unions), sized by total relation phi and colored by
distinction phi. Vertical placement is configurable via `rank="chain"`
(compact) or `rank="size"` (gaps at sizes with no distinctions), horizontal
placement via `layout="barycentric"` (crossing-reducing, the default) or
`layout="sorted"` (label order), and the marker encodings via `size_by` and
`color_by`. A `Theme` dataclass replaces ad-hoc theme overrides. The
legacy 3-D plot remains available as `pyphi.visualize.ces.plot_phi_structure`
until its rebuild lands.
