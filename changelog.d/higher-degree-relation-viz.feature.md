The hypergraph view (`pyphi.visualize.plot_ces`) now draws every
relation face as a star expansion by default — a hub at each face's centroid,
sized by φ, with spokes to its endpoints — so no relation degree is given
special visual weight. Previously degree-≥4 faces were silently dropped and
degree-3 faces were emphasized as filled triangles. `star_min_degree=` chooses
the lowest degree drawn as a star: the default `2` stars everything, `3`
restores degree-2 lines, and `4` additionally restores degree-3 triangles. A
`higher_faces` show element and a `degrees=` selection knob control which face
degrees render. All relation faces share one `relation φ` colorbar (alongside
the distinction-φ colorbar); colour opacity tracks φ on a fixed hue, so low-φ
relations fade toward transparency instead of obscuring the high-φ ones. Spokes
are a flat neutral grey, straight by default. The relation hue, fade range, hub
size, and spoke style are theme fields (`relation_rgb`, `relation_alpha_range`,
`hub_size_range`, `spoke_color`, `spoke_width`, `spoke_curvature`). Hovering a
relation face names its degree, φ, the units it is congruent over, and each
relatum (its distinction's mechanism, direction, and state-cased purview). A new
`view="spectrum"` panel summarizes relation count and Σφ per degree.
