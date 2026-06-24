Substrate-level analysis (`all_sias`, `irreducible_sias`, `complexes`,
`maximal_complex`) is now exposed as methods on `Substrate` and as
formalism-agnostic free functions in `pyphi.substrate`. The per-formalism
duplicates in `pyphi.formalism.iit3` (`all_complexes`, `complexes`,
`major_complex`, `condensed`) and `pyphi.formalism.iit4` (`all_complexes`,
`irreducible_complexes`, `maximal_complex`) have been removed.

Naming has been corrected to match paper terminology: a "complex" is a
non-overlapping local maximum of |big_phi| under the exclusion postulate,
not merely a candidate system with |big_phi| > 0. The former
`iit3.complexes` returned the latter — that semantic is preserved as
`substrate.irreducible_sias`. The former `iit3.condensed` returned the
actual set of complexes — that semantic is now `substrate.complexes`.
