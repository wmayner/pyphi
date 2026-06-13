The intrinsic-units search drivers (`pyphi.macro.complexes`,
`intrinsic_units`, `valid_systems`, `is_intrinsic_unit`,
`competing_systems`) now parallelize their independent `phi_s`
evaluations across processes, controlled by the new
`config.infrastructure.parallel_macro_system_evaluation` option (off by
default) or a per-call `parallel_kwargs` argument. Results are identical
to sequential runs.
