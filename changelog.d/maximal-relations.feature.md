Added `maximal_relations()`, `maximal_faces()`, and
`maximal_relations_by_distinction()` to the relations query surface: the
inclusion-maximal relations and relation faces (the facets of the relation
complex), computed in closed form from the atom incidence on every backend
— including analytical and fold backends — without enumerating relations.
Relation pandas records now include the overlap under a `purview` column
(for `RelationFace` records this renames the previously mislabeled
`relata` key).
