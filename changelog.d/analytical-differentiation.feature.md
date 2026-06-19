Added `pyphi.matching.Differentiation.analytical_differentiation`: the
cross-structure differentiation D computed in closed form, by inclusion-exclusion
over the unique triggered structures, without enumerating concrete relations.
It equals the concrete `differentiation` wherever that is computable, and is the
only way to compute D when the structures carry `AnalyticalRelations` (which are
not iterable). The perceptual differentiation stays concrete.
