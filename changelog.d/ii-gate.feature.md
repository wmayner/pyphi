The macro grain search now skips partition sweeps whose outcome is
certified by the intrinsic-information cap. Under the default formalism
(IIT 4.0 (2026), `system_phi_measure="INTRINSIC_INFORMATION"`), each
candidate's intrinsic information bounds its φₛ by construction, so
`pyphi.macro.search.complexes()` and the intrinsic-unit criteria evaluate
a candidate's partitions only when its ceiling could still change the
outcome — results are identical, and skipped candidates appear in
`ComplexesResult.records` with their ceiling (`phi=None`, `gated=True`).
Control the behavior with the new `prune` keyword (`"certified"`, `"off"`,
or `None` for automatic) on `complexes()`, `intrinsic_units()`,
`is_intrinsic_unit()`, `competing_systems()`, and `valid_systems()`.
`prune="certified"` raises `ConfigurationError` under measures without
the cap, where the bounding inequality does not hold.
