Added `PhiFold`, the cause-effect-structure slice of one or more distinctions
with their incident relations. `CauseEffectStructure.fold(distinctions)` and
`.distinction_folds()` construct folds; `PhiFold.big_phi_contribution` gives the
fold's additive share of the structure's Φ (distinctions at full φ, relations
apportioned as φ_r/|r|), which tiles: summing it over a structure's
single-distinction folds recovers `big_phi`. Fold sums are computed in closed
form for analytical-relations structures via `AnalyticalFoldRelations`, with no
relation enumeration. `highlight_phi_fold(fold)` now accepts a lone fold and
renders it against its parent.
