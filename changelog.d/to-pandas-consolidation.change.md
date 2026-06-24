Unified `to_pandas()` across result objects into one labeled-export convention.
Scalar-record results (`RepertoireIrreducibilityAnalysis`, the MICE types,
`Distinction`) return a `Series`; `Distinctions` returns a `DataFrame` indexed
by labeled mechanism; `StateSpecification` and `SystemStateSpecification` return
a tidy long-format `DataFrame` with columns `direction`, `kind`, `purview`,
`state`, `probability`. Units render as labels throughout. This replaces the
previous `json_normalize`-based heuristic, whose dotted-path columns and
`Series`-vs-`DataFrame` guessing are gone.
