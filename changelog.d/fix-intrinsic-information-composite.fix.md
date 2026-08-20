Fixed two defects in the `INTRINSIC_INFORMATION` composite measure (reachable
via `mechanism_phi_measure` / `specification_measure`; the default pipeline
is unaffected). The cause-side intrinsic differentiation was computed from
the unnormalized forward likelihoods instead of the Bayes posterior of
Eq. 11, overstating ii by −log₂ of the normalizer wherever the
differentiation term binds. And the differentiation operand was squeezed
while the specification operand kept the repertoire's canonical rank, so the
elementwise minimum broadcast across singleton axes — producing wrong ii
values, wrong-length specified states, and an IndexError on the
config-routed distinction path.
