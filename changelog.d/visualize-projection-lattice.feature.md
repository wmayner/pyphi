Added a pure projection layer for visualization
(`pyphi.visualize.projection`) and a new inclusion-lattice (Hasse) view of
phi-structures: `pyphi.visualize.plot_phi_structure(ces, view="lattice")`
draws distinctions ranked by purview inclusion, sized by total relation phi
and colored by distinction phi, with a `Theme` dataclass replacing ad-hoc
theme overrides. The legacy 3-D plot remains available as
`pyphi.visualize.ces.plot_phi_structure` until its rebuild lands.
