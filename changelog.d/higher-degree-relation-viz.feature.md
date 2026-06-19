Relation faces of degree ≥4 are now visible in the simplicial-complex view
(`pyphi.visualize.plot_ces`), drawn as a star expansion — a hub at each face's
centroid, sized and colored by φ, with spokes to its endpoints. Previously these
faces were silently dropped. A `higher_faces` show element (on by default) and a
`degrees=` selection knob control which face degrees render, and a new
`view="spectrum"` panel summarizes relation count and Σφ per degree.
