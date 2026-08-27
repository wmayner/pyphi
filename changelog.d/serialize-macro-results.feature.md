`MacroSystem` and `ComplexesResult` (the return type of `macro.complexes()` and
`analyze(grains=...)`) can now be saved and loaded with `pyphi.serialize`.
Previously `save()` raised `TypeError: No serializer registered`. The stored
`MacroSystem` carries the macro construction — units, micro substrate and
history, and the construction's cause TPM — so a reloaded system reproduces the
original's repertoires without recomputation.
