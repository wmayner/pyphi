Added a query surface to relation sets, answering structural questions
without enumerating relations. On `AnalyticalRelations` every query is
closed-form over the distinction set (via the analytical-relations
supplement of Albantakis et al. 2023): φ_r moments and mean/std
(`sum_phi_moment`, `phi_mean_std`), per-degree counts and sums
(`num_relations_of_degree`, `sum_phi_of_degree`, `degree_spectrum`), the
maximum φ_r (`max_phi`), the exact φ_r histogram grouped at configured
precision (`phi_histogram`), the atom-pair binding matrix
(`binding_matrix`), and the total face count (`num_faces`). `strongest()`
yields relations lazily in exact descending-φ_r order (top-K or thresholded,
output-sensitively); `sample(n, seed=...)` draws unbiased coverage-weighted
relation samples with standard errors (`RelationSample`); `materialize()`
is the explicit, bounded escape hatch to `ConcreteRelations`.
`ConcreteRelations` answers the same queries by iteration, and Φ-folds
answer them restricted to their incident relations.
`CauseEffectStructure.distinction_importance()` ranks distinctions by their
additive contribution to Φ (the contributions tile `big_phi` exactly).
