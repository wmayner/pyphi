`plot_ces` now renders cause-effect structures backed by analytically-computed
relations: pass `max_relations=N` to draw the N strongest relations by φ_r
(node sizes and the spectrum view stay exact over the full structure).
`CauseEffectStructure.diff` now reports relation statistic deltas (Σφ_r, count,
and the per-degree spectrum) on both relation backends, alongside per-relation
gained/lost rows where relations are enumerable.
