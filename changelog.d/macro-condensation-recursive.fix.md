Fixed `pyphi.macro.complexes` computing only the first condensation layer:
the literal Eq. 19 predicate let already-excluded candidates veto other
candidates, so complexes on chain topologies (e.g. a system beaten only by
rivals that themselves lost to a stronger complex) were missing from the
result. Macro condensation now applies the recursive exclusion cascade
(Marshall et al. 2023, Algorithm A1) with S1 Composition escalation for
φₛ-tied cliques, matching `pyphi.substrate.complexes`. Composition
escalation now compares Φ at `config.numerics.precision` (previously raw
floats), so relabelings of one system tie instead of resolving on
floating-point summation noise.
