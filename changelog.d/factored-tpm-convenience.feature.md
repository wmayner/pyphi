`FactoredTPM` gains dense-TPM convenience methods: `is_deterministic()`,
`permute_nodes(permutation)` (reorder units, returns a `FactoredTPM`), and
`subtpm(fixed_nodes, state)` (condition on and drop units, returns a
`FactoredTPM` over the free units).
