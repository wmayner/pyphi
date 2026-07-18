Removed dead code: the unused `config` field on formalism objects (methods
resolve the live global config; the field was captured at import time and
misleading), `DistanceResult._preserve_aux_data`, the unused
`_acria_attributes` list and inert `__slots__`, the never-constructed
`POSTULATE_FAILURE` outcome status, `sum_of_min_times_avg_among_subsets`, and
the unused utility helpers `combs`, `comb_indices`, `specified_substate`,
`extremum_with_short_circuit`, `expsublog`, `expaddlog`, `all_same`,
`all_are_equal`, `all_extrema`/`all_minima`/`all_maxima`, and
`distribution.independent`. The `_part_id`/`_optional_float` helpers and
unreachable-exception constants duplicated between `landscape` and `optimize`
are consolidated in `landscape`.
