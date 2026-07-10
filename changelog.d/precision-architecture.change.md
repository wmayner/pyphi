φ, Φ, and α values are now plain floats with exact comparison semantics.
Tolerant comparison (up to `config.numerics.precision`) lives at decision
sites: the scalar predicates in the new `pyphi.numerics` module
(`eq`, `is_zero`, `is_positive`, `positive_mask`, `round_to_precision`)
and the tie-resolution cascades in `pyphi.resolve_ties`, which now
cluster float keys tolerantly so candidates tied up to precision are
always co-selected regardless of iteration order. The `PyPhiFloat` type
is removed; `DistanceResult` remains a float carrying metadata.
`pyphi.utils.eq` / `is_positive` / `is_nonpositive` moved to
`pyphi.numerics`.
