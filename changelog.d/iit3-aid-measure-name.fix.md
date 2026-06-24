Fixed a name mismatch that made the intrinsic-difference measures unselectable
under IIT 3.0: `IIT_3_0`'s compatible-measure set listed them as
`ABSOLUTE_INTRINSIC_DIFFERENCE` / `INTRINSIC_DIFFERENCE`, but the measure
registry uses the keys `AID` / `ID`, so neither spelling satisfied both the
compatibility check and the registry lookup. `config.formalism.iit.
mechanism_phi_measure` may now be set to `"AID"` or `"ID"` under IIT 3.0 (needed
for multi-valued substrates, where the default `EMD` is binary-only).
