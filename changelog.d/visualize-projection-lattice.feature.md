Added a pure projection layer for visualization
(`pyphi.visualize.projection`) and a new inclusion-lattice (Hasse) view of
phi-structures: `pyphi.visualize.plot_phi_structure(ces, view="lattice")`
draws distinctions ranked by an inclusion partial order (`order="mechanism"`
by subset relation on mechanisms, or `order="purview_union"` on
cause/effect-purview unions), sized by total relation phi and colored by
distinction phi. Horizontal placement is configurable via
`layout="barycentric"` (crossing-reducing, the default) or `layout="sorted"`
(label order), and a `Theme` dataclass replaces ad-hoc theme overrides. The
legacy 3-D plot remains available as `pyphi.visualize.ces.plot_phi_structure`
until its rebuild lands.
