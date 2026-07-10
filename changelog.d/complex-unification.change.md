`pyphi.macro.ComplexesResult.complexes` now contains
`pyphi.models.complex.Complex` objects (with `units`, `node_indices` as the
micro footprint, `is_maximal`, and `excluded` records) instead of bare
`MacroSystem`s; `ComplexesResult.ties` holds Φ-tied cliques (tuples of
candidate systems) instead of pairs, and `ComplexesResult.maximal_complex`
returns the winner or a falsy null `Complex`. `Complex` and
`ExcludedCandidate` gained an optional `units` field, and both serialize
with it. The macro search drivers raise under IIT 3.0. During tie
escalation, Φ is computed once per system content fingerprint, so
symmetric tied cliques skip cause-effect-structure computation entirely.
The shared condensation machinery now resides in `pyphi.condensation`.
