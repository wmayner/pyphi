Ported Bjørn Juel's `substrate_modeler` mechanism library into
`pyphi.substrate_generator`: 16 unit mechanisms (`sigmoid`, `resonator`,
`sor`, `gabor`, `mismatch_corrector`, `mismatch_pattern_detector`,
`modulated_sigmoid`, `stabilized_sigmoid`, `biased_sigmoid`, `copy`, `and`,
`or`, `xor`, `democracy`, `majority`, `weighted_mean`) in `mechanisms.py`, the
six composite-combination strategies (`selective`, `average`, `maximal`,
`first_necessary`, `integrator`, `serial`) plus a `composite()` factory in
`mechanism_combinations.py`, and a `create_substrate(node_params)` factory for
per-node construction. The state-dependent "endorsement" mechanism
(`resonator`) reproduces the matching paper's coupling, and a single stateless
`pyphi.Substrate` with a self-loop reproduces the original's `dynamic_tpm`
byte-for-byte (verified by golden fixtures generated from the original library).
This lets the matching perceptual substrate be built natively, with no
dependency on `substrate_modeler`.
