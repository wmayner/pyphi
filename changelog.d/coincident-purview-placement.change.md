In the hypergraph view (`pyphi.visualize.plot_ces`), purview endpoints that
share a purview and direction are now separated by leaning each toward its
distinction's mechanism, so a dot sits on the mechanism-purview link that identifies
which distinction it belongs to. This replaces the previous arbitrary regular-polygon
spread, whose offset direction encoded nothing. The placement is deterministic and
collision-free (a tie-break handles mechanisms that are collinear from the shared
purview). The prior behavior is available via
`HypergraphGeometry(endpoint_placement="polygon")`.
